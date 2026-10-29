//! Интеграционный тест экспорта: lavfi-видео → маска → .mp4 с аудио.
//!
//! Требует ffmpeg/ffprobe в PATH (CI/окружение разработчика уже это имеет).

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use vc_codec::export::{ExportConfig, Exporter};
use vc_codec::{CodecBackend, FfmpegBackend};
use vc_core::{MaskRegion, Shape, UNTIL_END};
use vc_inpaint::Strategy;

fn space() -> PathBuf {
    // Выбираем доступную для внешних процессов папку.
    let base = std::env::var("VC_TMP").unwrap_or_else(|_| "/tmp/opencode".into());
    let p = PathBuf::from(base);
    let _ = std::fs::create_dir_all(&p);
    p
}

fn make_source(dir: &PathBuf, name: &str) -> Result<PathBuf, String> {
    let src = dir.join(name);
    let status = Command::new("ffmpeg")
        .args([
            "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest",
        ])
        .arg(&src)
        .status()
        .map_err(|e| format!("ffmpeg недоступен: {e}"))?;
    assert!(status.success(), "ffmpeg не сделал тестовое видео");
    Ok(src)
}

#[test]
fn export_produces_valid_video_with_audio() {
    let backend = FfmpegBackend::default();
    let dir = space();
    let src = make_source(&dir, "vc_export_src.mp4").expect("источник");
    let out = dir.join("vc_export_out.mp4");

    let meta = backend.probe(&src).expect("probe");
    assert_eq!(meta.width, 160);
    assert_eq!(meta.height, 120);

    // Маска-прямоугольник в углу на весь диапазон (Telea простое место).
    let region = MaskRegion {
        id: 1,
        shape: Shape::Rect { corners: [[10.0, 10.0], [40.0, 30.0]] },
        start: 0,
        end: UNTIL_END,
    };

    let mut ex = Exporter::new(&backend, &meta, Arc::new(AtomicBool::new(false)), |_, _| {});
    ex.regions = vec![region];
    ex.strategy = Strategy::Telea;
    let stats = ex.run(&src, &out).expect("экспорт");
    assert!(!stats.cancelled);
    assert!(stats.frames_in >= 8, "кадров в {} — мало", stats.frames_in);
    assert_eq!(stats.frames_modified, stats.frames_in, "маска активна весь клип");
    assert!(out.exists());

    // Проверяем результат: кадров >= 8 и есть аудио-дорожка.
    let probe = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-count_frames",
            "-show_entries", "format=nb_frames",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json",
        ])
        .arg(&out)
        .output()
        .expect("ffprobe");
    let text = String::from_utf8_lossy(&probe.stdout);
    assert!(text.contains("audio"), "аудио потеряно: {text}");
    assert!(text.contains("h264"), "видео не h264: {text}");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn export_cancelled_by_flag() {
    let backend = FfmpegBackend::default();
    let dir = space();
    let src = make_source(&dir, "vc_cancel_src.mp4").expect("источник");
    let out = dir.join("vc_cancel_out.mp4");
    let meta = backend.probe(&src).expect("probe");

    let already_canceled = Arc::new(AtomicBool::new(true));
    let region = MaskRegion {
        id: 1,
        shape: Shape::Rect { corners: [[0.0, 0.0], [30.0, 20.0]] },
        start: 0,
        end: UNTIL_END,
    };
    let mut ex = Exporter::new(&backend, &meta, Arc::clone(&already_canceled), |_, _| {});
    ex.regions = vec![region];
    ex.strategy = Strategy::Telea;
    let stats = ex.run(&src, &out).expect("экспорт с отменой");
    assert!(stats.cancelled);
    assert_eq!(stats.frames_in, 0);
    let _ = std::fs::remove_file(&out);
}
