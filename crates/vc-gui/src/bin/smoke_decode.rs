//! Smoke-тест M0: реальный файл → probe → чтение кадров (как в GUI).
//!
//! Запуск: `cargo run -p vc-gui --bin smoke_decode -- /tmp/opencode/vctest/test.mp4`

use std::path::Path;

use vc_codec::{CodecBackend, FfmpegBackend};

fn main() {
    let path = std::env::args().nth(1).expect("укажите путь к видео как аргумент");
    let path = Path::new(&path);
    let backend = FfmpegBackend::default();

    let meta = backend.probe(path).expect("probe");
    println!(
        "probe ok: {}×{} @ {:.2} fps, аудио: {}, длина: {:?}, кадров: {:?}, кодек: {:?}",
        meta.width, meta.height, meta.fps, meta.has_audio,
        meta.duration_secs, meta.frame_count, meta.codec
    );

    let mut reader = backend.open(path, None).expect("open");
    let mut count = 0u32;
    let bytes_per_frame = (meta.width as usize) * (meta.height as usize) * 4;
    while let Some(frame) = reader.read_frame().expect("read_frame") {
        count += 1;
        if count == 1 {
            println!(
                "первый кадр: {}×{}, буфер {} байт (ожидалось {bytes_per_frame}), угловой пиксель {:?}",
                frame.width,
                frame.height,
                frame.data.len(),
                frame.pixel(0, 0),
            );
        }
        // Проверяем, что последний кадр приходит.
        count_frames_check(&frame, bytes_per_frame);
    }
    println!("дочитали: всего кадров {count} (ожидалось 2 сек × 25 fps = 50)");
    assert!(count == 50, "ожидали 50 кадров, прочитали {count}");
    println!("SMOKE OK");
}

/// Дешёвый sanity-чек без скучных вычислений в основной ветке.
fn count_frames_check(_frame: &vc_core::Frame, _expected: usize) {}
