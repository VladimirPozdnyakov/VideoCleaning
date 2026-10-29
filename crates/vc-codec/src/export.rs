//! Экспорт готового видео: кадры → инпейнтинг по маскам → ffmpeg-кодирование.
//!
//! Декодер открывается ПОСЛЕДОВАТЕЛЬНО (один проход по всему видео), кадры
//! прогоняются через [`vc_inpaint::pipeline::inpaint_frame_dynamic`] (тот же
//! пайплайн, что и в превью — результат совпадает), сырые RGBA кадры
//! подаются на stdin ffmpeg-энкодера:
//!
//! ```text
//! ffmpeg -f rawvideo -pix_fmt rgba -s W×H -r FPS -i -
//!        -i SOURCE -map 0:v -map 1:a? -c:v libx264 -crf CRF -pix_fmt yuv420p
//!        -c:a copy OUT
//! ```
//!
//! Соседние кадры-доноры подаются через скользящее окно (декодируем вперёд
//! с опережением). Если кадр-донор вне окна — он недоступен, остаток
//! закрывает Telea (такое же деградативное поведение, как у превью с
//! LRU-кэшем вместо гигантских клонов).

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vc_core::{Frame, MaskRegion};

use crate::{CodecBackend, CodecError, VideoMetadata};
use vc_inpaint::Strategy;
use vc_inpaint::background::MAX_NEIGHBOR_SPAN;

/// Настройки кодирования.
#[derive(Clone, Debug)]
pub struct ExportConfig {
    /// Качество x264 (меньше = лучше; 18 — почти без потерь).
    pub crf: u32,
    /// x264 preset (medium — компромисс скорости/размера).
    pub preset: String,
    /// Лимит байт скользящего окна доноров (512 МБ по умолчанию).
    pub window_cap_bytes: usize,
    /// Расширение маски в px (закрывает свечение текста за краем).
    pub mask_dilate: u32,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            crf: 18,
            preset: "medium".into(),
            window_cap_bytes: 512 * 1024 * 1024,
            mask_dilate: 3,
        }
    }
}

/// Итог экспорта.
#[derive(Debug, Clone)]
pub struct ExportStats {
    /// Кадров всего прочитано.
    pub frames_in: u64,
    /// Кадров с активными масками (прогнаны через инпейнтинг).
    pub frames_modified: u64,
    /// Прервано пользователем (файл остаётся с уже записанными кадрами).
    pub cancelled: bool,
}

/// Один забег экспорта.
///
/// Прогресс — замыкание `on_progress(готово, всего)`; отмена —
/// `Arc<AtomicBool>`: на ближайшем кадре после выставления флага.
pub struct Exporter<'a, B: CodecBackend, P: FnMut(u64, u64)> {
    backend: &'a B,
    meta: &'a VideoMetadata,
    pub regions: Vec<MaskRegion>,
    pub strategy: Strategy,
    pub radius: u32,
    pub config: ExportConfig,
    cancel: Arc<AtomicBool>,
    on_progress: P,
}

impl<'a, B: CodecBackend, P: FnMut(u64, u64)> Exporter<'a, B, P> {
    pub fn new(
        backend: &'a B,
        meta: &'a VideoMetadata,
        cancel: Arc<AtomicBool>,
        on_progress: P,
    ) -> Self {
        Self {
            backend,
            meta,
            regions: Vec::new(),
            strategy: Strategy::Background,
            radius: 5,
            config: ExportConfig::default(),
            cancel,
            on_progress,
        }
    }

    fn cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Путь к ffmpeg: берётся у бэкенда (учитывает `FFMPEG_PATH`).
    fn ffmpeg_path(&self) -> PathBuf {
        self.backend.ffmpeg_path()
    }

    fn spawn_encoder(&self, src: &Path, out: &Path) -> Result<Encoder, CodecError> {
        let args: Vec<String> = [
            "-y".into(),
            "-v".into(), "error".into(),
            "-f".into(), "rawvideo".into(),
            "-pix_fmt".into(), "rgba".into(),
            "-s".into(), format!("{}x{}", self.meta.width, self.meta.height),
            "-r".into(), format!("{:.6}", self.meta.fps),
            "-i".into(), "-".into(),
            "-i".into(), src.display().to_string(),
            // Аудио из оригинала, если есть (`?` — не падать без дорожки).
            "-map".into(), "0:v".into(),
            "-map".into(), "1:a?".into(),
            "-c:v".into(), "libx264".into(),
            "-preset".into(), self.config.preset.clone(),
            "-crf".into(), self.config.crf.to_string(),
            "-pix_fmt".into(), "yuv420p".into(),
            "-c:a".into(), "copy".into(),
            out.display().to_string(),
        ]
        .into_iter()
        .collect();

        let mut child = Command::new(self.ffmpeg_path())
            .args(args)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(crate::ffmpeg::tool_missing)?;
        let stdin = child.stdin.take().expect("stdin piped");
        Ok(Encoder { child, stdin: Some(stdin) })
    }

    /// Полный проход по видео. `out_path` — назначение (обычно .mp4).
    pub fn run(&mut self, src: &Path, out_path: &Path) -> Result<ExportStats, CodecError> {
        let total = total_estimate(self.meta);
        let mut stats = ExportStats { frames_in: 0, frames_modified: 0, cancelled: false };
        let mut encoder = self.spawn_encoder(src, out_path)?;
        let mut errored: Option<CodecError> = None;

        // Скользящее окно доноров: span кадров в обе стороны, по части
        // лимита байт. Текущий кадр всегда внутри окна.
        let frame_bytes =
            (self.meta.width as usize).saturating_mul(self.meta.height as usize) * 4;
        let span = calculate_span(frame_bytes, self.config.window_cap_bytes);
        let mut window: HashMap<i64, Frame> = HashMap::new();
        let mut next_read: u64 = 0;
        let mut eof = false;
        let regions = std::mem::take(&mut self.regions);
        let strategy = self.strategy;
        let radius = self.radius;

        let mut reader = self.backend.open(src, None)?;
        loop {
            // Читаем вперёд, чтобы доноры [idx−span; idx+span] были готовы.
            while !eof && next_read <= stats.frames_in + span as u64 {
                match reader.read_frame() {
                    Ok(Some(f)) => {
                        window.insert(next_read as i64, f);
                        next_read += 1;
                    }
                    Ok(None) => eof = true,
                    Err(e) => {
                        errored = Some(e);
                        break;
                    }
                }
            }

            if self.cancel_requested() {
                stats.cancelled = true;
                break;
            }

            let idx = stats.frames_in;
            let Some(current) = window.remove(&(idx as i64)) else {
                break; // EOF (кадра idx больше нет)
            };
            stats.frames_in += 1;

            let processed = vc_inpaint::pipeline::inpaint_frame_dynamic_opts(
                &current,
                &regions,
                idx,
                radius,
                strategy,
                self.config.mask_dilate,
                |off| {
                    let j = idx as i64 + off;
                    if j < 0 {
                        return None;
                    }
                    window.get(&j).cloned()
                },
            );
            // Окно чистим от уехавших назад (idx удалён выше).
            let lo = idx as i64 - span;
            window.retain(|k, _| *k >= lo);

            let to_write = match &processed {
                Some(res) => {
                    stats.frames_modified += 1;
                    &res.frame
                }
                None => &current,
            };
            if let Err(e) = encoder.write(to_write) {
                errored = Some(e);
                break;
            }
            (self.on_progress)(stats.frames_in, total);
        }
        self.regions = regions;

        // Закрываем stdin: энкодер дописывает и финализирует контейнер.
        if let Err(e) = encoder.finish() {
            errored.get_or_insert(e);
        }
        if let Some(e) = errored {
            // Не оставляем битый файл при настоящей ошибке.
            if !stats.cancelled {
                let _ = std::fs::remove_file(out_path);
                return Err(e);
            }
        }
        Ok(stats)
    }
}

/// Локальный управляющий энкодером процесс.
struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
}

impl Encoder {
    fn write(&mut self, frame: &Frame) -> Result<(), CodecError> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| CodecError::Decode("stdin энкодера уже закрыт".into()))?;
        if let Err(e) = stdin.write_all(&frame.data) {
            // Энкодер умер раньше времени — вытащим его stderr в сообщение.
            let mut buf = String::new();
            if let Some(mut err) = self.child.stderr.take() {
                let _ = std::io::Read::read_to_string(&mut err, &mut buf);
            }
            let detail = buf.trim().to_string();
            return Err(if detail.is_empty() {
                CodecError::from(e)
            } else {
                CodecError::Decode(format!("энкодер умер: {detail}"))
            });
        }
        Ok(())
    }

    /// Завершает кодирование: закрывает stdin и ждёт процесса.
    fn finish(&mut self) -> Result<(), CodecError> {
        self.stdin = None; // drop закрывает пайп
        let status = self.child.wait()?;
        if !status.success() {
            let mut buf = String::new();
            if let Some(mut err) = self.child.stderr.take() {
                let _ = std::io::Read::read_to_string(&mut err, &mut buf);
            }
            return Err(CodecError::Decode(format!(
                "ffmpeg-энкодер завершился с ошибкой: {}",
                buf.trim()
            )));
        }
        Ok(())
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        self.stdin = None;
        // Если finish() не вызван (паника/ошибка) — убираем зомби.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Пол-окна на доноров назад, полвперед — Clip в [4; MAX_NEIGHBOR_SPAN].
fn calculate_span(frame_bytes: usize, cap_bytes: usize) -> i64 {
    if frame_bytes == 0 {
        return 4;
    }
    let frames_fit = cap_bytes / frame_bytes;
    (frames_fit / 2).clamp(4, MAX_NEIGHBOR_SPAN as usize) as i64
}

fn total_estimate(meta: &VideoMetadata) -> u64 {
    match meta.frame_count {
        Some(n) => n,
        None => (meta.duration_secs.unwrap_or(0.0) * meta.fps) as u64,
    }
}
