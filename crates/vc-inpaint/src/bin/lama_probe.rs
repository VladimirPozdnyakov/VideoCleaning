//! Probe LaMa через tract-onnx: загрузка графа, один инференс, тайминг.
//! Запуск: cargo run -q -p vc-inpaint --release --bin lama_probe

use std::time::Instant;

use vc_core::Mask;
use vc_inpaint::lama;

fn main() {
    if !lama::available() {
        eprintln!("модель не найдена: {}", lama::model_path().display());
        std::process::exit(1);
    }
    let frame_side = 512usize;
    // Синтетическое «фото»: градиент + полосы, дыра — квадрат в центре.
    let mut image = vec![0f32; 3 * frame_side * frame_side];
    for y in 0..frame_side {
        for x in 0..frame_side {
            let q = y * frame_side + x;
            image[q] = (x % 64) as f32 / 64.0 * 0.6 + (y % 61) as f32 / 61.0 * 0.3;
            image[frame_side * frame_side + q] = 0.4 + 0.4 * ((x / 17) % 2) as f32;
            image[2 * frame_side * frame_side + q] = ((x + y) % 53) as f32 / 53.0;
        }
    }
    // Сторона окна: адаптивная (аниме-модель), либо 512² (статическая LaMa).
    let side: usize = std::env::var("VC_PROBE_SIDE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| lama::window_side(112, 112));
    // Норма входа: 1 = [0..1] (как в инференсе), 255 = проверка гипотезы
    // «модель ожидает [0..255]». Ответ виден по сохранению фона за дырой.
    let scale: f32 = std::env::var("VC_PROBE_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);
    let mut hole = Mask::empty(side as u32, side as u32);
    let h0 = (side - 112) / 2;
    for y in h0..h0 + 112 {
        for x in h0..h0 + 112 {
            hole.set(x as u32, y as u32, 255);
        }
    }
    let win = &image[0..3 * side * side];
    // Подача в масштабе гипотезы: inference_window ждёт [0..1], поэтому
    // нормируем обратно для сравнения (выход делим),
    let fed: Vec<f32> = if (scale - 1.0).abs() < 1e-6 {
        win.to_vec()
    } else {
        win.iter().map(|v| v * scale).collect()
    };
    let Some(tb) = lama::warmup(side) else {
        eprintln!("план не собрался (см. VC_TRACE=1)");
        std::process::exit(2);
    };
    let t0 = Instant::now();
    let Some(raw) = lama::inference_window(&fed, &hole, side) else {
        eprintln!("инференс не удался (см. VC_TRACE=1)");
        std::process::exit(2);
    };
    let dt = t0.elapsed();
    // Выход можно читать в норме fed (модель сохраняет недырные пиксели
    // в норме входа) — сравнение фона без деления.
    let out = &raw[..];
    let mid = side / 2 * side + side / 2;
    let in_center = win[mid];
    let corner = 10 * side + 10;
    let corner_err = out[corner] - fed[corner];
    println!(
        "окно {side}² scale={scale}: план {tb:?}; инференс: {dt:?}; выход центра={:.3} (вход {in_center:.3}), угол_err={corner_err:.4}, выход угол={:.3}",
        out[mid],
        out[corner]
    );
    // Повторный прогон: цена без компиляции плана.
    let t1 = Instant::now();
    let _ = lama::inference_window(&fed, &hole, side).expect("второй прогон");
    println!("повторный инференс: {:?}", t1.elapsed());
}
