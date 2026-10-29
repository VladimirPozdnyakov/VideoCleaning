//! Реализация [`CodecBackend`] через системные ffmpeg/ffprobe.
//!
//! Декодирование: `ffmpeg -i file -f rawvideo -pix_fmt rgba -` и чтение
//! `w*h*4` байт из stdout по одному кадру. Кадры не буферизуем целиком —
//! память ограничена независимо от длины видео.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use serde_json::Value;
use vc_core::Frame;

use crate::{CodecBackend, CodecError, FrameReader, VideoMetadata};

/// Пути к внешним инструментам (по умолчанию — из PATH).
/// Можно принудительно задать через переменные окружения
/// `FFMPEG_PATH` / `FFPROBE_PATH`.
#[derive(Clone, Debug)]
pub struct FfmpegBackend {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Default for FfmpegBackend {
    fn default() -> Self {
        let ff = std::env::var("FFMPEG_PATH").ok().map(PathBuf::from);
        let probe = std::env::var("FFPROBE_PATH").ok().map(PathBuf::from);
        Self {
            ffmpeg: ff.unwrap_or_else(|| PathBuf::from("ffmpeg")),
            ffprobe: probe.unwrap_or_else(|| PathBuf::from("ffprobe")),
        }
    }
}

pub(crate) fn tool_missing(err: std::io::Error) -> CodecError {
    // Самый частый случай — исполняемый файл не найден в PATH.
    if err.kind() == std::io::ErrorKind::NotFound {
        CodecError::ToolMissing
    } else {
        err.into()
    }
}

impl CodecBackend for FfmpegBackend {
    fn ffmpeg_path(&self) -> std::path::PathBuf {
        self.ffmpeg.clone()
    }

    fn probe(&self, path: &Path) -> Result<VideoMetadata, CodecError> {
        if !path.exists() {
            return Err(CodecError::NotFound(path.to_path_buf()));
        }
        let out = Command::new(&self.ffprobe)
            .args([
                "-v", "error",
                "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate",
                "-show_entries", "format=duration",
                "-of", "json",
            ])
            .arg(path)
            .stderr(Stdio::piped())
            .output()
            .map_err(tool_missing)?;

        if !out.status.success() {
            let msg = String::from_utf8_lossy(&out.stderr);
            return Err(CodecError::Probe(msg.trim().to_string()));
        }
        let json: Value = serde_json::from_slice(&out.stdout)
            .map_err(|e| CodecError::Probe(format!("ffprobe вернул не-JSON: {e}")))?;
        parse_probe_metadata(path.to_path_buf(), &json)
            .ok_or_else(|| CodecError::Probe("видеопоток не найден в файле".into()))
    }

    fn open(
        &self,
        path: &Path,
        seek_secs: Option<f64>,
    ) -> Result<Box<dyn FrameReader + Send>, CodecError> {
        if !path.exists() {
            return Err(CodecError::NotFound(path.to_path_buf()));
        }
        let meta = self.probe(path)?;

        let mut args: Vec<String> = vec!["-v".into(), "error".into()];
        if let Some(secs) = seek_secs {
            // -ss до -i: быстрый seek (до ключевого кадра), точность приемлема для превью.
            args.push("-ss".into());
            args.push(format!("{secs:.6}"));
        }
        args.push("-i".into());
        args.push(path.display().to_string());
        args.extend([
            // Только первый видеопоток; аудио отбрасываем (читается отдельно).
            "-map".into(),
            "0:v:0".into(),
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            "rgba".into(),
            "-".into(),
        ]);

        let mut child = Command::new(&self.ffmpeg)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(tool_missing)?;
        if std::env::var("VC_TRACE").is_ok() {
            eprintln!("trace: open seek={seek_secs:?}");
        }

        let stdout = child.stdout.take().ok_or_else(|| {
            CodecError::Decode("stdout ffmpeg не доступен".into())
        })?;

        Ok(Box::new(FfmpegFrameReader {
            child,
            stdout: Some(BufReader::new(stdout)),
            width: meta.width,
            height: meta.height,
        }))
    }
}

/// Читатель кадров из rawvideo-пайпа ffmpeg.
/// Drop убивает процесс, даже если чтение оборвалось на середине.
pub struct FfmpegFrameReader {
    child: Child,
    stdout: Option<BufReader<ChildStdout>>,
    width: u32,
    height: u32,
}

impl Drop for FfmpegFrameReader {
    fn drop(&mut self) {
        // stdin закрыт (Stdio::null), поэтому ffmpeg обычно завершится сам;
        // сигналим, чтобы не оставить зомби.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl FrameReader for FfmpegFrameReader {
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn read_frame(&mut self) -> Result<Option<Frame>, CodecError> {
        if self.width == 0 || self.height == 0 {
            return Err(CodecError::Decode("неверный размер кадра (0)".into()));
        }
        let mut buf = vec![0u8; Frame::byte_size_for(self.width, self.height)];
        let reader = self
            .stdout
            .as_mut()
            .ok_or_else(|| CodecError::Decode("stdout ffmpeg уже закрыт".into()))?;
        // Читаем ровно один кадр.
        match reader.read_exact(&mut buf) {
            Ok(()) => Ok(Some(Frame::new(self.width, self.height, buf)?)),
            // Короткое чтение до конца пайпа = EOF.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(CodecError::Io(e)),
        }
    }
}

/// Разбирает JSON ffprobe в метаданные. `None` — если видеопотока нет.
fn parse_probe_metadata(path: PathBuf, json: &Value) -> Option<VideoMetadata> {
    let streams = json.get("streams")?.as_array()?;

    let video = streams.iter().find(|s| s.get("codec_type").and_then(Value::as_str) == Some("video"))?;
    let width = video.get("width").and_then(Value::as_u64)? as u32;
    let height = video.get("height").and_then(Value::as_u64)? as u32;

    let fps = parse_r_frame_rate(video)?;

    let duration_secs = json
        .pointer("/format/duration")
        .and_then(Value::as_str)
        .and_then(|s| s.parse::<f64>().ok());

    let frame_count_by_seconds =
        duration_secs.map(|d| (d * fps).round() as u64);
    let frame_count = video
        .get("nb_frames")
        .and_then(Value::as_u64)
        .or(frame_count_by_seconds);

    Some(VideoMetadata {
        path,
        width,
        height,
        fps,
        duration_secs,
        frame_count,
        has_audio: streams
            .iter()
            .any(|s| s.get("codec_type").and_then(Value::as_str) == Some("audio")),
        codec: video
            .get("codec_name")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

/// `r_frame_rate` вида "30000/1001" → 29.97.
fn parse_r_frame_rate(video: &Value) -> Option<f64> {
    let s = video.get("r_frame_rate").and_then(Value::as_str)?;
    let (num, den) = s.split_once('/')?;
    let num: f64 = num.parse().ok()?;
    let den: f64 = den.parse().ok()?;
    if den <= 0.0 {
        return None;
    }
    Some(num / den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_typical_probe_output() {
        let json = json!({
            "streams": [
                { "codec_type": "audio", "codec_name": "aac" },
                { "codec_type": "video", "codec_name": "h264", "width": 1920,
                  "height": 1080, "r_frame_rate": "30000/1001", "nb_frames": 300 }
            ],
            "format": { "duration": "10.010000" }
        });
        let m = parse_probe_metadata(PathBuf::from("t.mp4"), &json).unwrap();
        assert_eq!(m.width, 1920);
        assert_eq!(m.height, 1080);
        assert!((m.fps - 29.97002997).abs() < 1e-6);
        assert!(m.has_audio);
        assert_eq!(m.frame_count, Some(300));
        assert_eq!(m.codec.as_deref(), Some("h264"));
    }

    #[test]
    fn no_video_stream_is_none() {
        let json = json!({ "streams": [ { "codec_type": "audio" } ] });
        assert!(parse_probe_metadata(PathBuf::from("a.m4a"), &json).is_none());
    }
}
