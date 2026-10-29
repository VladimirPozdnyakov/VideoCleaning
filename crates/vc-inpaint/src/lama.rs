//! LaMa (AnimeMangaInpainting, FourierUnitJIT) — предобученная модель
//! перерисовки через ONNX Runtime (динамическая библиотека из `lib/`,
//! путь переопределяется env `ORT_DYLIB_PATH`).
//!
//! Решение по референсу VSR + выбор пользователя (вариант B): качество
//! LaMa на аниме недостижимо в CPU-инференсе на pure-Rust трекере (tract),
//! поэтому рантайм — официальный ONNX Runtime (Apache-2.0), Rust-обёртка ort.
//!
//! Файл модели: `models/lama-manga-dynamic.onnx` (перекрывается env
//! `VC_LAMA_PATH`). Вход: image f32 [1,3,H,W] в [0..1] RGB + mask f32
//! [1,1,H,W] (1 = дыра); выход — та же геометрия.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use vc_core::{Frame, Mask};

/// Максимальная сторона окна инференса (кратность 8).
pub const MAX_SIDE: usize = 640;
/// Младшая сторона окна (кратность 8).
pub const MIN_SIDE: usize = 64;
/// Контекст вокруг дыры с каждой стороны (px).
const MARGIN: i64 = 96;

/// Куда смотреть за файлом модели.
pub fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("VC_LAMA_PATH") {
        return PathBuf::from(p);
    }
    let mut candidates = vec![PathBuf::from("models/lama-manga-dynamic.onnx")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("models/lama-manga-dynamic.onnx"));
        }
    }
    candidates
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("models/lama-manga-dynamic.onnx"))
}

/// Путь к динамической библиотеке onnxruntime: env ORT_DYLIB_PATH → `lib/`
/// рядом с рабочей папкой → рядом с exe.
fn ort_dylib() -> PathBuf {
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        return PathBuf::from(p);
    }
    let mut candidates = vec![
        PathBuf::from("lib/libonnxruntime.so"),
        PathBuf::from("lib/libonnxruntime.so.1.20.1"),
    ];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("lib/libonnxruntime.so"));
            candidates.push(dir.join("lib/libonnxruntime.so.1.20.1"));
        }
    }
    candidates
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("lib/libonnxruntime.so"))
}

/// Модель доступна? (файл модели И библиотека рантайма)
pub fn available() -> bool {
    model_path().exists() && ort_dylib().exists()
}

/// Сторона окна (кратность 8) под bbox дыры: дыра + контекст MARGIN
/// с каждой стороны, не больше MAX_SIDE, не меньше MIN_SIDE.
pub fn window_side(bw: u32, bh: u32) -> usize {
    let need = bw.max(bh) as i64 + 2 * MARGIN;
    let s = (need + 7) / 8 * 8;
    s.clamp(MIN_SIDE as i64, MAX_SIDE as i64) as usize
}

static SESSION: OnceLock<Option<Mutex<ort::session::Session>>> = OnceLock::new();

/// Сессия ORT (ленивое создание, затем многоразовое). `run` в rc.15
/// требует `&mut Session`, поэтому под Mutex (инференс всё равно один поток).
fn get_session() -> Option<&'static Mutex<ort::session::Session>> {
    SESSION
        .get_or_init(|| {
            let trace = std::env::var("VC_TRACE").is_ok();
            // Рантайм грузится явно: ort::init_from до любого другого API ort.
            // CUDA EP регистрируется по умолчанию (VC_GPU=0 отключает);
            // если CUDA-среда не собралась — ORT/наш фолбэк отпадают к CPU.
            let want_gpu = std::env::var("VC_GPU").map_or(true, |v| v != "0");
            let gpu_note = |trace: bool, msg: &str| {
                if trace {
                    eprintln!("trace: {msg}");
                }
            };
            let env = match ort::init_from(ort_dylib()) {
                Ok(builder) => {
                    let builder = if want_gpu {
                        builder.with_execution_providers([ort::ep::CUDA::default().build()])
                    } else {
                        builder
                    };
                    match builder.build() {
                        Ok(env) => env,
                        Err(e) => {
                            if trace {
                                eprintln!("trace: ORT env: {e}");
                            }
                            return None;
                        }
                    }
                }
                Err(e) => {
                    if trace {
                        eprintln!("trace: ORT dylib ({}): {e}", ort_dylib().display());
                    }
                    return None;
                }
            };
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1);
            let make_session = |env: &ort::environment::Environment,
                                want_gpu: bool|
             -> Result<ort::session::Session, String> {
                let bld = ort::session::Session::builder(env)
                    .map_err(|e| format!("опции сессии: {e}"))?;
                let mut bld = bld
                    .with_intra_threads(threads)
                    .map_err(|e| format!("потоки ORT: {e}"))?;
                let note = if want_gpu { " (с CUDA EP)" } else { "" };
                bld.commit_from_file(model_path())
                    .map_err(|e| format!("модель{note}: {e}"))
            };
            // Попытка 0: CUDA EP (уровень env). Если модель не поднялась — CPU.
            let session = if want_gpu {
                Some(make_session(&env, true))
            } else {
                Some(make_session(&env, false))
            };
            let session = match session {
                Some(Ok(s)) => {
                    gpu_note(trace, "LaMa/ORT создана (CUDA EP запросан)");
                    Mutex::new(s)
                }
                Some(Err(e)) => {
                    gpu_note(trace, &format!("LaMa/ORT c CUDA не поднялась ({e}); повтор на CPU"));
                    match make_session(&env, false) {
                        Ok(s) => {
                            gpu_note(trace, "LaMa/ORT создана на CPU");
                            Mutex::new(s)
                        }
                        Err(e2) => {
                            gpu_note(trace, &format!("LaMa/ORT на CPU: {e2}"));
                            return None;
                        }
                    }
                }
                None => return None,
            };
            Some(session)
        })
        .as_ref()
}

/// LaMa-инференс на окне `side`² (кратен 8): `image` — RGB по-канально в
/// [0..1] (3×side²), `hole` — маска в координатах окна.
/// Возврат: заполнение в [0..1] (норма выхода модели выравнивается
/// автоматически по сохранению фона — разные экспорты дают [0..1]/[0..255]).
pub fn inference_window(image: &[f32], hole: &Mask, side: usize) -> Option<Vec<f32>> {
    let session = get_session()?;
    let mask_f: Vec<f32> = (0..side * side)
        .map(|i| if hole.bits[i] != 0 { 1.0 } else { 0.0 })
        .collect();
    let mut session = session.lock().ok()?;
    let raw = run_session(&mut session, image, &mask_f, side)?;
    let n = side * side;
    let mut out = raw;
    if out.len() != 3 * n {
        return None;
    }
    // Норма выхода: недырные пиксели модель воспроизводит как есть —
    // измеряем коэффициент заново каждое окно (окно маленькое, ~сотни точек).
    let mut sampled = 0usize;
    let mut ratio_sum = 0f32;
    for q in (0..n).step_by((n / 256).max(1)) {
        if hole.bits[q] != 0 || image[q] <= 0.02 {
            continue;
        }
        // Любая из трёх плоскостей дефомировала бы оценку согласованно —
        // k-канал q одинаков, берем R.
        let r = out[q] / image[q];
        ratio_sum += r;
        sampled += 1;
    }
    let scale = if sampled > 0 {
        let k = ratio_sum / sampled as f32;
        if k > 8.0 { 255.0 } else { 1.0 }
    } else {
        1.0
    };
    if scale != 1.0 {
        for v in out.iter_mut() {
            *v /= scale;
        }
    }
    Some(out)
}

/// Инференс ORT: два входа-тензора, извлечение f32-выхода `inpainted`.
fn run_session(
    session: &mut ort::session::Session,
    image: &[f32],
    mask: &[f32],
    side: usize,
) -> Option<Vec<f32>> {
    use ort::value::Tensor;
    let img_tensor = Tensor::from_array(([1usize, 3, side, side], image.to_vec())).ok()?;
    let mask_tensor = Tensor::from_array(([1usize, 1, side, side], mask.to_vec())).ok()?;
    let outputs = session
        .run(vec![("image", img_tensor), ("mask", mask_tensor)])
        .ok()?;
    for (_name, value) in outputs.iter() {
        if let Ok((_shape, data)) = value.try_extract_tensor::<f32>() {
            return Some(data.to_vec());
        }
    }
    None
}

/// Предкомпиляция: прогрев сессии (первый run часто дорогой из-за аллокаций).
/// Возврат: время создания сессии, если модель доступна.
pub fn warmup(_side: usize) -> Option<std::time::Duration> {
    let t = std::time::Instant::now();
    get_session()?;
    // Ничему не мешает пустой мини-прогон на 64² (не требуется).
    Some(t.elapsed())
}

/// Старты окон по одной оси: покрываем [lo..=hi] окнами `side`
/// с перекрытием 50%, клампя к [0, frame_side−side].
fn window_starts(lo: usize, hi: usize, side: usize, frame_side: usize) -> Vec<usize> {
    let side_i = side as i64;
    let mut out = Vec::new();
    let mut x = lo as i64;
    loop {
        out.push(x.clamp(0, frame_side as i64 - side_i) as usize);
        if (x + side_i) as i64 > hi as i64 {
            break;
        }
        x += (side / 2) as i64;
    }
    out
}

/// Перерисовка дыры кадра предобученной LaMa (адаптивные окна, feather).
/// Возврат None — модель недоступна/не создалась.
pub fn repaint(frame: &Frame, hole: &Mask) -> Option<Frame> {
    if !available() {
        return None;
    }
    let (w, h) = (frame.width as usize, frame.height as usize);
    let Some(((bx0, by0), (bx1, by1))) = hole.bbox() else {
        return None;
    };
    // Окно не может быть больше кадра (и кратно 8 для FFC).
    let frame_min8 = (w.min(h) / 8 * 8).max(8);
    let side = window_side(bx1 - bx0 + 1, by1 - by0 + 1).min(frame_min8);

    let frame_rgb = triple(frame);
    let mut acc: Vec<f32> = vec![0f32; w * h * 3];
    let mut accw = vec![0f32; w * h];
    let trace = std::env::var("VC_TRACE").is_ok();

    for &x0 in &window_starts(bx0 as usize, bx1 as usize, side, w) {
        for &y0 in &window_starts(by0 as usize, by1 as usize, side, h) {
            // Окно кадра [0..1] и локальная маска дыры.
            let mut win = vec![0f32; side * side * 3];
            let mut local = Mask::empty(side as u32, side as u32);
            let mut any = false;
            for ry in 0..side {
                let gy = (y0 + ry).min(h - 1);
                for rx in 0..side {
                    let gx = (x0 + rx).min(w - 1);
                    let gi = gy * w + gx;
                    let q = ry * side + rx;
                    for c in 0..3 {
                        win[c * side * side + q] = frame_rgb[c][gi] as f32 / 255.0;
                    }
                    if hole.value(gx as u32, gy as u32) != 0 {
                        local.set(rx as u32, ry as u32, 255);
                        any = true;
                    }
                }
            }
            if !any {
                continue;
            }
            if trace {
                eprintln!("trace: lama окно ({x0},{y0}) side={side}");
            }
            let Some(pred) = inference_window(&win, &local, side) else {
                return None;
            };
            // Feather-вес по расстоянию до края окна.
            let ramp = (side / 8).max(8) as f32;
            for ry in 0..side {
                for rx in 0..side {
                    if local.value(rx as u32, ry as u32) == 0 {
                        continue;
                    }
                    let gi = (y0 + ry) * w + (x0 + rx);
                    let dmin = (rx as f32)
                        .min((side - 1 - rx) as f32)
                        .min(ry as f32)
                        .min((side - 1 - ry) as f32);
                    let wt = (dmin / ramp).clamp(0.2, 1.0);
                    accw[gi] += wt;
                    let q = ry * side + rx;
                    for c in 0..3 {
                        acc[c * w * h + gi] += wt * pred[c * side * side + q];
                    }
                }
            }
        }
    }

    let mut data = frame.data.clone();
    for y in 0..h {
        for x in 0..w {
            let gi = y * w + x;
            if hole.value(x as u32, y as u32) == 0 || accw[gi] <= 0.0 {
                continue;
            }
            for c in 0..3 {
                let v = (acc[c * w * h + gi] / accw[gi]).clamp(0.0, 1.0) * 255.0;
                data[gi * 4 + c] = v.round() as u8;
            }
        }
    }
    Some(Frame { width: frame.width, height: frame.height, data })
}

/// RGB по-канально (u8).
fn triple(frame: &Frame) -> [Vec<u8>; 3] {
    let n = (frame.width * frame.height) as usize;
    let mut r = vec![0u8; n];
    let mut g = vec![0u8; n];
    let mut b = vec![0u8; n];
    for (i, chunk) in frame.data.chunks_exact(4).enumerate() {
        r[i] = chunk[0];
        g[i] = chunk[1];
        b[i] = chunk[2];
    }
    [r, g, b]
}
