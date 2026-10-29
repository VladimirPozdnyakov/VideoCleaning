//! Пробник playback-застоев: воспроизводит пейсинг GUI (24 fps, refocus
//! каждый кадр, отсутствие seek'ов) и печатает, где current догоняет
//! декодер (кадра нет в кэше при тике).
//!
//! Запуск: cargo run -p vc-gui --bin playback_probe -- /path/video.mp4

use std::path::Path;
use std::time::{Duration, Instant};

use vc_codec::{CodecBackend, FfmpegBackend, Player};

fn main() {
    let path = std::env::args().nth(1).expect("путь к видео");
    let backend = FfmpegBackend::default();
    let meta = backend.probe(Path::new(&path)).expect("probe");
    let fps = meta.fps.max(1.0);
    let total = 1514; // для теста — полная длина известна; иначе пройдём до EOF
    println!(
        "probe: {}×{} @ {:.2} fps, playing {} кадров",
        meta.width,
        meta.height,
        fps,
        total
    );

    // Такой же кэп, как в GUI.
    let player = Player::start(
        FfmpegBackend::default(),
        path.clone().into(),
        &meta,
        512 * 1024 * 1024,
    );
    let cache = player.cache();

    let step = Duration::from_secs_f64(1.0 / fps);
    let mut next_tick = Instant::now();
    let mut stalls: Vec<(u64, u128, u64)> = Vec::new(); // (frame, мс ожидания, focus)

    let mut i = 0u64;
    while i < total - 1 {
        // Тик GUI: если кадр есть —/goto; иначе ждём (м doPost 300мс), потом force seek.
        if Instant::now() >= next_tick {
            let mut c = cache.lock().unwrap();
            let has = c.contains(i + 1);
            let (mn, mx, bytes) = c.extents();
            c.set_focus(i + 1);
            drop(c);
            if i >= 50 && i <= 70 {
                println!("tick {i}: has(next)={has} map=[{mn}..{mx}] bytes={bytes}мб");
            }
            if has {
                i += 1;
                next_tick += step;
            } else {
                // Декодер не успел: меряем, сколько реально ждали.
                let t0 = Instant::now();
                let mut waited = 0u64;
                loop {
                    std::thread::sleep(Duration::from_millis(2));
                    waited = t0.elapsed().as_millis() as u64;
                    let c = cache.lock().unwrap();
                    let has = c.contains(i + 1);
                    let focus = c.focus();
                    drop(c);
                    if has || waited >= 300 {
                        if !has {
                            // force seek как в GUI.
                            player.seek(i + 1);
                            let t1 = Instant::now();
                            loop {
                                std::thread::sleep(Duration::from_millis(2));
                                if cache.lock().unwrap().contains(i + 1) || t1.elapsed().as_millis() > 2000 {
                                    break;
                                }
                            }
                        }
                        stalls.push((i + 1, t0.elapsed().as_millis(), focus));
                        i += 1;
                        next_tick = Instant::now() + step;
                        break;
                    }
                }
            }
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    println!("итог: застоев {}", stalls.len());
    for (f, ms, focus) in &stalls {
        println!("  кадр {} — ожидание {} мс (focus={})", f, ms, focus);
    }
}
