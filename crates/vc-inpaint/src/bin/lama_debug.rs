//! Отладка фаз «умного удаления» на реальном кадре: ретушь, маска для
//! LaMa, результат LaMa. Дампы — 24-битный BMP в /tmp/opencode/dbg.
//!
//! Запуск: ffmpeg → *.rgb (rgb24, центральный кадр + соседи), затем
//! ./lama_debug <dir> <w> <h> <center_idx>

use vc_core::{Frame, MaskRegion, Shape, UNTIL_END};
use vc_inpaint::background;
use vc_inpaint::Strategy;

/// 24-битный BMP (BGR, строки снизу вверх, падение до 4 байт).
fn write_bmp(path: &str, w: usize, h: usize, rgba: &[u8]) {
    let row = w * 3;
    let pad = (4 - row % 4) % 4;
    let data_size = (row + pad) * h;
    let mut out = Vec::with_capacity(54 + data_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(54u32 + data_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    // всё остальное — нули (54 байт заголовок)
    out.resize(54, 0);
    for y in (0..h).rev() {
        for x in 0..w {
            let i = (y * w + x) * 4;
            out.push(rgba[i + 2]);
            out.push(rgba[i + 1]);
            out.push(rgba[i]);
        }
        for _ in 0..pad {
            out.push(0);
        }
    }
    std::fs::write(path, &out).unwrap();
}

fn overlay_mask(frame: &Frame, mask: &vc_core::Mask, r: u8, g: u8, b: u8) -> Frame {
    let mut data = frame.data.clone();
    for (i, m) in mask.bits.iter().enumerate() {
        if *m != 0 && i * 4 + 2 < data.len() {
            data[i * 4] = data[i * 4] / 2 + r / 2;
            data[i * 4 + 1] = data[i * 4 + 1] / 2 + g / 2;
            data[i * 4 + 2] = data[i * 4 + 2] / 2 + b / 2;
        }
    }
    Frame { width: frame.width, height: frame.height, data }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("использование: lama_debug <dir> <w> <h> <center_idx>");
        std::process::exit(1);
    }
    let dir = &args[1];
    let w: u32 = args[2].parse().unwrap();
    let h: u32 = args[3].parse().unwrap();
    let center: u64 = args[4].parse().unwrap();
    let load = |i: u64| -> Option<Frame> {
        let p = format!("{dir}/f_{:03}.rgb", i);
        let bytes = std::fs::read(&p).ok()?;
        if bytes.len() != (w * h * 3) as usize {
            eprintln!("баррак: {} {}B ≠ {}", p, bytes.len(), w * h * 3);
            return None;
        }
        let mut data = vec![0u8; (w * h * 4) as usize];
        for (j, px) in data.chunks_exact_mut(4).enumerate() {
            px[0] = bytes[j * 3];
            px[1] = bytes[j * 3 + 1];
            px[2] = bytes[j * 3 + 2];
            px[3] = 255;
        }
        Some(Frame { width: w, height: h, data })
    };
    let frame = load(center).expect("центральный кадр");
    write_bmp(&format!("{dir}/A_original.bmp"), w as usize, h as usize, &frame.data);

    // Прямоугольная маска поверх титров (как пользователь рисует).
    let regions = [MaskRegion {
        id: 1,
        shape: Shape::Rect { corners: [[900.0, 280.0], [1030.0, 780.0]] },
        start: 0,
        end: UNTIL_END,
    }];

    let lookup = |off: i64| -> Option<Frame> {
        let idx = (center as i64 + off).max(0) as u64;
        load(idx)
    };

    // Маска пользователя + дилатация 3 (как в GUI/pipeline).
    let user_mask = vc_core::rasterize(&regions, center, w, h).dilated(3);

    // ---- Фаза 1: ретушь (как в Neural) ----
    let retouch_cfg = background::RetouchConfig { static_to_hole: true, ..Default::default() };
    let rr = background::retouch_frame(
        &frame,
        &user_mask,
        &regions,
        center,
        &retouch_cfg,
        &mut { lookup },
    )
    .expect("ретушь");
    write_bmp(&format!("{dir}/B_retouched.bmp"), w as usize, h as usize, &rr.frame.data);
    write_bmp(
        &format!("{dir}/C_nn_hole.bmp"),
        w as usize,
        h as usize,
        &overlay_mask(&frame, &rr.nn_hole, 255, 0, 0).data,
    );
    println!("ретушь: used={:?}, nn_hole={}px", rr.used, rr.nn_hole.count());

    // ---- Фаза 2: маска, уходящая в LaMa ----
    // V2 (по умолчанию): весь регион пользователя.
    // V1 (VC_DEBUG_VARIANT=v1): только пиксели без доказательств ретуши
    // (штрихи текста) — узкие дыры дают чёткую текстуру вместо полосы.
    let hole = if std::env::var("VC_DEBUG_VARIANT").map_or(false, |v| v == "v1") {
        let mut h = user_mask.clone();
        for y in 0..user_mask.height {
            for x in 0..user_mask.width {
                if rr.replaced.value(x, y) != 0 && h.value(x, y) != 0 {
                    h.set(x, y, 0);
                }
            }
        }
        h
    } else {
        user_mask.clone()
    };
    println!("LaMa-маска (nn_hole ∩ ink): {}px", hole.count());
    write_bmp(
        &format!("{dir}/D_lama_hole.bmp"),
        w as usize,
        h as usize,
        &overlay_mask(&rr.frame, &hole, 255, 0, 0).data,
    );

    // ---- Фаза 3: LaMa ----
    if hole.count() > 0 {
        if let Some(out) = vc_inpaint::lama::repaint(&rr.frame, &hole) {
            write_bmp(&format!("{dir}/E_lama.bmp"), w as usize, h as usize, &out.data);
            println!("LaMa: ок");
        } else {
            println!("LaMa: None");
        }
    }

    // ---- Полный пайплайн Neural (для сравнения) ----
    if let Some(res) = vc_inpaint::pipeline::inpaint_frame_dynamic_opts(
        &frame,
        &regions,
        center,
        5,
        Strategy::Neural,
        3,
        |off| {
            let idx = (center as i64 + off).max(0) as u64;
            load(idx)
        },
    ) {
        write_bmp(&format!("{dir}/F_neural.bmp"), w as usize, h as usize, &res.frame.data);
        println!("Neural: filled_by={:?}", res.filled_by);
    }
}
