//! Бенчмарк Telea на 1080p: маска «субтитр» ~1200×60.
//! Запуск: `cargo run -p vc-inpaint --release --bin telea_bench`

use std::time::Instant;

use vc_core::{Frame, Mask};
use vc_inpaint::telea::telea_inpaint;

fn build_frame(w: u32, h: u32) -> Frame {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            // «Фон»: плавный градиент + шумовая модуляция.
            let r = 40 + 3 * ((x / 97) % 61) as u8;
            let g = 40 + 3 * ((y / 97) % 61) as u8;
            let b = 128 + ((x * y) / 4096) % 64;
            data.extend_from_slice(&[r, g, b as u8, 255]);
        }
    }
    Frame::new(w, h, data).unwrap()
}

fn build_subtitle_mask(w: u32, h: u32) -> Mask {
    let mut m = Mask::empty(w, h);
    let x0 = (w / 2 - 600).max(0);
    let x1 = (w / 2 + 600).min(w);
    let y0 = h - 90;
    let y1 = h - 30;
    for y in y0..y1 {
        for x in x0..x1 {
            // «Текст»: полосатая маска, как реальные буквы (70% дыры).
            if (x / 12 + y / 4) % 10 != 0 {
                m.set(x, y, 255);
            }
        }
    }
    m
}

fn main() {
    let (w, h) = (1920u32, 1080u32);
    let frame = build_frame(w, h);
    let mask = build_subtitle_mask(w, h);
    println!(
        "1080p, маска-субтитр: {} дырявых пикселей",
        mask.count()
    );

    let t0 = Instant::now();
    let out = telea_inpaint(&frame, &mask, 5);
    let dt = t0.elapsed();
    println!("telea_inpaint(radius=5): {dt:?}");
    let s = out.pixel(960, h - 60).unwrap();
    println!("центр дыры после инпейнтинга: R={} G={} B={}", s[0], s[1], s[2]);
}
