//! Нейросетевой инпейнтинг: малый U-Net-подобный CNN, обучаемый **на месте**.
//!
//! Два режима:
//! 1. **Self-supervised** (одиночный кадр): сеть учится восстанавливать
//!    случайно вырезанные кусочки известной области кропа; цель — исходные
//!    пиксели вне тренировочной дыры. Работает без соседей (начало видео).
//! 2. **Repaint с донорской супервизией** (M5c): для кадра с текстом соседние
//!    кадры, где маска ещё/уже пуста, дают честные цели — сеть обучается
//!    перерисовывать текстуру фона под текстом на основании ~кадров; предсказание
//!    делается скользящими окнами с feather-смешением, поэтому дыры любого
//!    размера покрываются одним обучением.
//!
//! Обучение происходит на самих кадрах (burn + Flex, чистый Rust), чекпоинтов
//! и ONNX нет. Рецептивное поле ~40 px (3 уровня энкодера).

use burn::nn::{
    PaddingConfig2d,
    activation::{Relu, Sigmoid},
    conv::{Conv2d, Conv2dConfig},
};
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use vc_core::{Frame, Mask, MaskRegion};

/// Настройки нейронного режима.
#[derive(Debug, Clone)]
pub struct NeuralConfig {
    /// Максимальная сторона кропа (кратна 8).
    pub max_side: u32,
    /// Шагов обучения на месте (на кадр repaint-режима).
    pub steps: usize,
    /// Скорость обучения Adam.
    pub lr: f64,
    /// Максимальная сторона тренировочной дыры (случайные патчи).
    pub trial_max: u32,
    /// Разброс доноров ±кадров для супервизии.
    pub donor_span: i64,
    /// Использовать предобученную LaMa (ONNX) для перерисовки спорных штрихов.
    pub use_lama: bool,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            max_side: 128,
            steps: 600,
            lr: 0.02,
            trial_max: 40,
            donor_span: 8,
            use_lama: true,
        }
    }
}

const HIDDEN1: usize = 16;
const HIDDEN2: usize = 32;
const HIDDEN3: usize = 32;

/// Модель: 5 входных каналов (RGB + карта «неизвестного» + пустой канал)
/// → энкодер /2,/4,/8 → ботлнек → декодер со skip-конкатенациями → sigmoid.
#[derive(Module, Debug)]
pub struct MiniInpaint {
    e1: Conv2d,
    e2: Conv2d,
    e3: Conv2d,
    b1: Conv2d,
    b2: Conv2d,
    d1: Conv2d,
    d2: Conv2d,
    d3: Conv2d,
    d4: Conv2d,
    out: Conv2d,
    relu: Relu,
    sig: Sigmoid,
}

impl MiniInpaint {
    fn new(device: &Device) -> Self {
        let c = |i: usize, o: usize, k: usize, stride: usize| {
            Conv2dConfig::new([i, o], [k, k])
                .with_padding(PaddingConfig2d::Same)
                .with_stride([stride, stride])
                .init(device)
        };
        Self {
            e1: c(5, HIDDEN1, 5, 2),
            e2: c(HIDDEN1, HIDDEN2, 3, 2),
            e3: c(HIDDEN2, HIDDEN3, 3, 2),
            b1: c(HIDDEN3, HIDDEN3, 3, 1),
            b2: c(HIDDEN3, HIDDEN3, 3, 1),
            d1: c(HIDDEN3, HIDDEN2, 3, 1),
            d2: c(HIDDEN2 + HIDDEN3, HIDDEN2, 3, 1),
            d3: c(HIDDEN2, 8, 3, 1),
            d4: c(8 + HIDDEN1, 8, 3, 1),
            out: c(8, 3, 3, 1),
            relu: Relu::new(),
            sig: Sigmoid::new(),
        }
    }

    /// Вход [B,5,H,W], выход [B,3,H,W] в [0;1]. H,W кратны 8.
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let r = |t: Tensor<4>| self.relu.forward(t);
        // /2 → /4 → /8
        let e1 = r(self.e1.forward(x.clone()));
        let e2 = r(self.e2.forward(e1.clone()));
        let e3 = r(self.e3.forward(e2.clone()));
        let bott = self.b2.forward(r(self.b1.forward(e3)));
        // /8 → /4
        let up1 = bott.repeat_dim(2, 2).repeat_dim(3, 2);
        let d = r(self.d1.forward(up1));
        let skip = Tensor::cat(vec![d, e2], 1);
        let d2 = r(self.d2.forward(skip));
        // /4 → /2
        let up2 = d2.repeat_dim(2, 2).repeat_dim(3, 2);
        let d3 = r(self.d3.forward(up2));
        let skip2 = Tensor::cat(vec![d3, e1], 1);
        let d4 = r(self.d4.forward(skip2));
        // /2 → полный размер
        let up3 = d4.repeat_dim(2, 2).repeat_dim(3, 2);
        self.sig.forward(self.out.forward(up3))
    }
}

/// Простой xorshift64 для воспроизводимых случайных патчей.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % (n.max(1) as u64)) as u32
    }
}

/// Окно кропа: содержит дыру [a0..=a1] с запасом контекста (не хуже 2× размера
/// дыры и не меньше 16 px), сторона кратна 8 (три уровня /2) и ≤ frame_side/max_side.
fn crop_window(a0: u32, a1: u32, frame_side: u32, max_side: u32) -> Option<(u32, u32)> {
    let raw = a1 + 1 - a0;
    if raw == 0 {
        return None;
    }
    let want = (raw * 2).max(raw + 16);
    let mut side = want.div_ceil(8) * 8;
    if side > max_side {
        // Ужимаем контекст: минимум до размера дыры.
        side = raw.div_ceil(8) * 8;
        if side > max_side || side > frame_side {
            return None;
        }
    }
    if side > frame_side {
        return None;
    }
    // Центрируем окно вокруг дыры, клампя к границам.
    let center = (a0 + a1) / 2;
    let mut x0 = center.saturating_sub(side / 2);
    if x0 + side > frame_side {
        x0 = frame_side - side;
    }
    Some((x0, x0 + side))
}

/// Серое значение пикселя (целочисленный luma).
#[inline]
fn luma(data: &[u8], idx: usize) -> u8 {
    let c = &data[idx * 4..idx * 4 + 3];
    ((c[0] as u32 * 299 + c[1] as u32 * 587 + c[2] as u32 * 114) / 1000) as u8
}

/// Оценка сдвига донора относительно кадра (грубый блок-матчинг, coarse-to-fine):
/// ищем (sdx, sdy), при котором донор(x+sdx, y+sdy) совпадает с кадром(x, y)
/// на кольце вокруг маски (пиксели известной области рядом с дырой).
/// Аниме/титры — почти чистый трансляционный сдвиг, этого достаточно.
fn estimate_shift(frame: &Frame, donor: &Frame, mask: &Mask, max_shift: i32) -> (i32, i32) {
    let ring = mask.dilated(6);
    // Сэмплы кольца: известные пиксели в пределах 6 px от маски.
    let mut samples: Vec<(i64, i64, u8)> = Vec::new();
    for y in 0..frame.height as usize {
        for x in 0..frame.width as usize {
            let i = y * frame.width as usize + x;
            if mask.bits[i] != 0 || ring.bits[i] == 0 {
                continue;
            }
            samples.push((x as i64, y as i64, luma(&frame.data, i)));
        }
    }
    if samples.len() < 32 {
        return (0, 0); // кольца нет — без параллакса не угадать
    }
    // Проредить до ~1200 точек (статистика SAD устойчива и на этом числе).
    let step = samples.len().div_ceil(1200);
    if step > 1 {
        samples = samples.iter().step_by(step).copied().collect();
    }
    crate::background::best_shift(donor, &samples, max_shift as i64)
}

/// Кроп окна донора напрямую из кадра (без материализации полного кадра):
/// возврат (rgb 3ch, clean 0/1).
fn donor_crop(
    donor: &Frame,
    donor_idx: u64,
    regions: &[MaskRegion],
    x0: u32,
    y0: u32,
    side: u32,
) -> (Vec<f32>, Vec<f32>) {
    let side_us = side as usize;
    let n = side_us * side_us;
    let mut rgb = vec![0f32; n * 3];
    let mut clean = vec![0f32; n];
    let dmask = vc_core::rasterize(regions, donor_idx, donor.width, donor.height);
    for ry in 0..side_us {
        for rx in 0..side_us {
            let gx = x0 as usize + rx;
            let gy = y0 as usize + ry;
            if gx >= donor.width as usize || gy >= donor.height as usize {
                continue;
            }
            let q = ry * side_us + rx;
            let src = gy * donor.width as usize + gx;
            let chunk = &donor.data[src * 4..src * 4 + 3];
            for (c, v) in chunk.iter().enumerate() {
                rgb[c * n + q] = *v as f32 / 255.0;
            }
            if dmask.bits[src] == 0 {
                clean[q] = 1.0;
            }
        }
    }
    (rgb, clean)
}

/// Сторона кропа: ≤ max_side и ≤ кадра, кратна 8 (три уровня /2).
fn effective_side(cfg: &NeuralConfig, w: u32, h: u32) -> u32 {
    let s = cfg.max_side.min(w).min(h);
    (s / 8 * 8).max(8)
}

/// Окна инференса, покрывающие боксы дыры со strides/перекрытием.
fn window_starts(cover_lo: u32, cover_hi: u32, side: u32, frame_side: u32, stride: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut x = cover_lo as i64;
    loop {
        out.push((x.clamp(0, (frame_side - side) as i64)) as u32);
        if (x as u32) + side >= cover_hi + 1 {
            break;
        }
        x += stride as i64;
        if x > cover_hi as i64 {
            break;
        }
    }
    out
}

/// M5c: настоящая перерисовка текста — сеть обучается по соседним кадрам
/// (донорская супервизия: где у донора маска пуста, его пиксель — честная
/// цель), затем предсказывает содержимое дыры скользящими окнами с
/// feather-смешением. Работает с дырами любого размера.
///
/// `None`, если доноры недоступны (нет соседей в кэше/у краёв видео).
pub fn neural_repaint(
    frame: &Frame,
    hole: &Mask,
    regions: &[MaskRegion],
    frame_index: u64,
    cfg: &NeuralConfig,
    lookup: &mut dyn FnMut(i64) -> Option<Frame>,
) -> Option<Frame> {
    let (w, h) = (frame.width, frame.height);
    let Some(((bx0, by0), (bx1, by1))) = hole.bbox() else {
        return None;
    };
    let side = effective_side(cfg, w, h);
    let side_us = side as usize;
    let n = side_us * side_us;

    // Доступные доноры: ±1..±span (технически смотрим через lookup).
    let mut offsets: Vec<i64> = Vec::new();
    for off in 1..=cfg.donor_span {
        offsets.push(-off);
        offsets.push(off);
    }
    offsets.retain(|&off| {
        let idx = frame_index as i64 + off;
        idx >= 0 && lookup(idx).is_some()
    });
    if offsets.is_empty() {
        return None; // доноров нет — repaint недоступен
    }
    // Взвешенный пул: сосед ±1 важнее дальних (фон ближе друг к другу).
    let mut pool: Vec<i64> = Vec::new();
    for &off in &offsets {
        let weight = match off.abs() {
            1 => 4,
            2 => 3,
            3 => 2,
            _ => 1,
        };
        pool.extend(std::iter::repeat_n(off, weight));
    }
    let trace = std::env::var("VC_TRACE").is_ok();

    let device = burn::tensor::Device::flex().autodiff();
    let mut net = MiniInpaint::new(&device);
    let mut opt = AdamConfig::new().init();
    let mut rng = Rng(frame_index.rotate_left(13) | 1);

    // Сдвиги доноров относительно кадра: оценка один раз на кадр на донора.
    let mut shifts: std::collections::HashMap<i64, (i32, i32)> = std::collections::HashMap::new();

    // RGB текущего кадра по-канально (для кропов окон).
    let frame_rgb_cache: Vec<f32> = donor_rgb_full(frame);

    // ---- Обучение: донорская супервизия + scatter-dropout известной области.
    let mut any_sup = false;
    for i in 0..cfg.steps {
        let off = pool[rng.below(pool.len() as u32) as usize];
        let donor_frame = lookup(off).expect("offsets проверены");
        let donor_idx = (frame_index as i64 + off) as u64;

        // Выровнять донор по текущему кадру: локальный сдвиг по кольцу.
        let (sdx, sdy) = *shifts
            .entry(off)
            .or_insert_with(|| estimate_shift(frame, &donor_frame, hole, 24));
        if trace && i == 0 {
            eprintln!("trace: донор {off:+} сдвиг ({sdx},{sdy})");
        }

        // Случайное окно вокруг дыры.
        let bw = bx1 - bx0 + 1;
        let bh = by1 - by0 + 1;
        let x0 = ((bx0 as i64) + rng.below(bw.max(1)) as i64 - (side as i64) / 2)
            .clamp(0, (w - side) as i64) as u32;
        let y0 = ((by0 as i64) + rng.below(bh.max(1)) as i64 - (side as i64) / 2)
            .clamp(0, (h - side) as i64) as u32;

        let (frm_rgb, hole_map) = {
            // rgb текущего кадра + настоящая маска в окне.
            let n_full = (w * h) as usize;
            let mut rgb = vec![0f32; n * 3];
            let mut mask = vec![0f32; n];
            for ry in 0..side_us {
                for rx in 0..side_us {
                    let gx = x0 as usize + rx;
                    let gy = y0 as usize + ry;
                    if gx >= w as usize || gy >= h as usize {
                        continue;
                    }
                    let q = ry * side_us + rx;
                    let src = gy * w as usize + gx;
                    for c in 0..3 {
                        rgb[c * n + q] = frame_rgb_cache[c * n_full + src];
                    }
                    if hole.value(x0 + rx as u32, y0 + ry as u32) != 0 {
                        mask[q] = 1.0;
                    }
                }
            }
            (rgb, mask)
        };
        // Донор смещается на вычисленный сдвиг: target(x,y) = donor(x+sdx, y+sdy).
        let dx0 = (x0 as i64 + sdx as i64).max(0) as u32;
        let dy0 = (y0 as i64 + sdy as i64).max(0) as u32;
        let (don_rgb, don_clean) = donor_crop(&donor_frame, donor_idx, regions, dx0, dy0, side);
        if don_clean.iter().any(|&v| v > 0.0) {
            any_sup = true;
        }

        // тренировочная маска: настоящая дыра + scatter-dropout известной области.
        let mut known = vec![0f32; n]; // 1 = пиксель подаётся обнулённым
        let mut w_sup = vec![0f32; n]; // 1 = честная донорская цель
        let mut w_drop = vec![0f32; n]; // 0.3 = самосогласованность известной области
        for j in 0..n {
            if hole_map[j] == 1.0 {
                known[j] = 1.0;
                w_sup[j] = don_clean[j]; // донор чист? учим на нём
            } else if rng.below(100) < 25 {
                known[j] = 1.0;
                w_drop[j] = 0.3;
            }
        }
        // entrada: RGB с обнулёнными unknown; каналы: rgb, known-mask, 0.
        let mut inp_rgb = frm_rgb.clone();
        for j in 0..n {
            if known[j] == 1.0 {
                for c in 0..3 {
                    inp_rgb[c * n + j] = 0.0;
                }
            }
        }
        let mut input_v = inp_rgb;
        input_v.extend_from_slice(&known);
        input_v.extend_from_slice(&vec![0f32; n]);

        // Целевой вектор: донор там, где есть супервизия; иначе кадр.
        let mut gt = vec![0f32; n * 3];
        for c in 0..3 {
            for j in 0..n {
                gt[c * n + j] = if w_sup[j] > 0.0 {
                    don_rgb[c * n + j]
                } else {
                    frm_rgb[c * n + j]
                };
            }
        }
        let mut w3 = vec![0f32; n * 3];
        for c in 0..3 {
            for j in 0..n {
                w3[c * n + j] = w_sup[j] + w_drop[j];
            }
        }

        let x = Tensor::<4>::from_data(
            TensorData::new(input_v, Shape::new([1, 5, side_us, side_us])),
            &device,
        );
        let gt_t = Tensor::<4>::from_data(
            TensorData::new(gt, Shape::new([1, 3, side_us, side_us])),
            &device,
        );
        let w_t = Tensor::<4>::from_data(
            TensorData::new(w3, Shape::new([1, 3, side_us, side_us])),
            &device,
        );

        let preds = net.forward(x);
        let diff = preds.sub(gt_t).mul(w_t);
        let loss = diff.clone().mul(diff).mean();
        if trace && i % 100 == 0 {
            eprintln!("trace: repaint шаг {i} loss={:?}", loss.try_to_vec_as::<f32>());
        }
        let grads = GradientsParams::from_grads(loss.backward(), &net);
        let t = i as f64 / cfg.steps.max(1) as f64;
        let lr = 1e-4 + cfg.lr * (0.5 + 0.5 * (std::f64::consts::PI * t).cos());
        net = opt.step(lr, net, grads);
    }
    if !any_sup {
        // Доноры есть, но у всех в окнах свои маски — супервизии не было.
        return None;
    }

    // ---- Инференс: скользящие окна с feather-смешением.
    let stride = (side / 2).max(16) as u32; // 50% перекрытие
    let xs = window_starts(bx0, bx1, side, w, stride);
    let ys = window_starts(by0, by1, side, h, stride);
    let mut acc = vec![0f32; (w * h) as usize * 3];
    let mut accw = vec![0f32; (w * h) as usize];

    for &x0 in &xs {
        'ys: for &y0 in &ys {
            // окно может не содержать дыры (клампится у краёв) — skip.
            let (frm_rgb, hole_map) = {
                let n_full = (w * h) as usize;
                let mut rgb = vec![0f32; n * 3];
                let mut mask = vec![0f32; n];
                for ry in 0..side_us {
                    for rx in 0..side_us {
                        let gx = x0 as usize + rx;
                        let gy = y0 as usize + ry;
                        if gx >= w as usize || gy >= h as usize {
                            continue;
                        }
                        let q = ry * side_us + rx;
                        let src = gy * w as usize + gx;
                        for c in 0..3 {
                            rgb[c * n + q] = frame_rgb_cache[c * n_full + src];
                        }
                        if hole.value(x0 + rx as u32, y0 + ry as u32) != 0 {
                            mask[q] = 1.0;
                        }
                    }
                }
                (rgb, mask)
            };
            if hole_map.iter().all(|&v| v == 0.0) {
                continue 'ys;
            }
            let mut input_v = frm_rgb.clone();
            input_v.extend_from_slice(&hole_map);
            input_v.extend_from_slice(&vec![0f32; n]);
            let x = Tensor::<4>::from_data(
                TensorData::new(input_v, Shape::new([1, 5, side_us, side_us])),
                &device,
            );
            let pred = net.forward(x).try_to_vec_as::<f32>().expect("pred f32");

            for ry in 0..side_us {
                for rx in 0..side_us {
                    let q = ry * side_us + rx;
                    if hole_map[q] != 1.0 {
                        continue;
                    }
                    // feather: ближе к краю окна — меньше вес.
                    let dmin = rx.min(side_us - 1 - rx).min(ry).min(side_us - 1 - ry);
                    let wt = (dmin as f32 / stride as f32).clamp(0.25, 1.0);
                    let gx = x0 as usize + rx;
                    let gy = y0 as usize + ry;
                    let gi = gy * w as usize + gx;
                    accw[gi] += wt;
                    for c in 0..3 {
                        acc[c * (w * h) as usize + gi] += wt * pred[c * n + q];
                    }
                }
            }
        }
    }

    let mut data = frame.data.clone();
    for gy in 0..h as usize {
        for gx in 0..w as usize {
            let gi = gy * w as usize + gx;
            if hole.value(gx as u32, gy as u32) == 0 || accw[gi] <= 0.0 {
                continue;
            }
            let idx = gi * 4;
            for c in 0..3 {
                let v = (acc[c * (w * h) as usize + gi] / accw[gi])
                    .clamp(0.0, 1.0)
                    * 255.0;
                data[idx + c] = v.round() as u8;
            }
        }
    }
    Some(Frame { width: w, height: h, data })
}


/// Вмещается ли дыра в кроп обучающего окна (без затрат на обучение).
///
/// Используется пайплайном: если нет — repaint невозможен на одиночном кропе
/// (но тилинг reptaint-режима это компенсирует).
pub fn hole_fits(hole: &Mask, frame: &Frame, cfg: &NeuralConfig) -> bool {
    let Some(((mx0, my0), (mx1, my1))) = hole.bbox() else {
        return false;
    };
    crop_window(mx0, mx1, frame.width, cfg.max_side).is_some()
        && crop_window(my0, my1, frame.height, cfg.max_side).is_some()
}

/// Полный кадр по-канально ([R; G; B]).
fn donor_rgb_full(frame: &Frame) -> Vec<f32> {
    let n = (frame.width * frame.height) as usize;
    let mut rgb = vec![0f32; n * 3];
    for (i, chunk) in frame.data.chunks_exact(4).enumerate() {
        rgb[i] = chunk[0] as f32 / 255.0;
        rgb[n + i] = chunk[1] as f32 / 255.0;
        rgb[2 * n + i] = chunk[2] as f32 / 255.0;
    }
    rgb
}

/// Кроп: RGB по-канально ([R; G; B] по side×side) и маска дыры (side×side) —
/// для self-supervised одиночного кропа.
fn crop_data(frame: &Frame, hole: &Mask, x0: u32, y0: u32, side: u32) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (frame.width, frame.height);
    let n = (side * side) as usize;
    let mut rgb = vec![0f32; n * 3];
    let mut hole_map = vec![0f32; n];
    for ry in 0..side {
        for rx in 0..side {
            let gx = x0 + rx;
            let gy = y0 + ry;
            let p = (ry * side + rx) as usize;
            if hole.value(gx, gy) != 0 && gx < w && gy < h {
                hole_map[p] = 1.0;
            } else if gx < w && gy < h {
                let px = frame.pixel(gx, gy).unwrap_or([0, 0, 0, 0]);
                rgb[p] = px[0] as f32 / 255.0;
                rgb[n + p] = px[1] as f32 / 255.0;
                rgb[2 * n + p] = px[2] as f32 / 255.0;
            }
        }
    }
    (rgb, hole_map)
}

/// Полный нейронный проход: обучение на месте + предсказание дыры.
pub fn neural_inpaint(frame: &Frame, hole: &Mask, cfg: &NeuralConfig, seed: u64) -> Option<Frame> {
    let (w, h) = (frame.width, frame.height);
    let Some(((mx0, my0), (mx1, my1))) = hole.bbox() else {
        return None;
    };
    let Some((cx0, cx1)) = crop_window(mx0, mx1, w, cfg.max_side) else {
        return None;
    };
    let Some((cy0, _cy1)) = crop_window(my0, my1, h, cfg.max_side) else {
        return None;
    };
    let side = cx1 - cx0;
    let side_us = side as usize;
    // Плавный бюджет: на маленьких кропах шагов больше (каждый дешевле),
    // суммарная вычислительная нагрузка остаётся примерно постоянной.
    let area = (side_us * side_us) as f32;
    let steps = ((cfg.steps as f32) * (128.0 * 128.0) / area)
        .ceil()
        .clamp(200.0, 3000.0) as usize;

    let device = burn::tensor::Device::flex().autodiff();
    let mut net = MiniInpaint::new(&device);
    let mut opt = AdamConfig::new().init();
    let mut rng = Rng(seed.wrapping_mul(2685821657726332327).rotate_left(17) | 1);

    let (rgb, hole_map) = crop_data(frame, hole, cx0, cy0, side);

    let zero = vec![0f32; side_us * side_us];
    let input5 = |rgb_t: &[f32], mask2: &[f32]| {
        let mut v = rgb_t.to_vec();
        v.extend_from_slice(mask2);
        v.extend_from_slice(&zero);
        Tensor::<4>::from_data(
            TensorData::new(v, Shape::new([1, 5, side_us, side_us])),
            &device,
        )
    };

    // Обучение: случайные «продырявленные» пиксели (p≈25%) на известной области,
    // плюс тренировочная маска канала = настоящая дыра для единообразия.
    for i in 0..steps {
        // Случайный dropout известных пикселей (≈25%) + настоящая дыра.
        let mut known = vec![0f32; side_us * side_us]; // 1 = обнулить и «неизвестно»
        let mut wgt = vec![0f32; side_us * side_us];
        for y in 0..side_us {
            for x in 0..side_us {
                let j = y * side_us + x;
                // Пиксель неизвестен: настоящая дыра, либо выброшенный.
                if hole_map[j] == 1.0 || rng.below(100) < 25 {
                    known[j] = 1.0;
                    // Супервизия — ТОЛЬКО пиксели известной области (hole-маска
                    // остаётся «загадкой», её контент заранее неизвестен).
                    wgt[j] = if hole_map[j] == 0.0 { 1.0 } else { 0.0 };
                }
            }
        }
        let mut rgb_t = rgb.clone();
        for i in 0..side_us * side_us {
            if known[i] == 1.0 {
                for c in 0..3 {
                    rgb_t[i * 3 + c] = 0.0;
                }
            }
        }

        let x = input5(&rgb_t, &known);
        let gt = Tensor::<4>::from_data(
            TensorData::new(rgb.to_vec(), Shape::new([1, 3, side_us, side_us])),
            &device,
        );
        let mut wgt3 = Vec::with_capacity(side_us * side_us * 3);
        for c in 0..3 {
            wgt3.extend_from_slice(&wgt);
            let _ = c;
        }
        let wgt_t = Tensor::<4>::from_data(
            TensorData::new(wgt3, Shape::new([1, 3, side_us, side_us])),
            &device,
        );

        let preds = net.forward(x);
        let diff = preds.sub(gt).mul(wgt_t);
        let loss = diff.clone().mul(diff).mean();
        if std::env::var("VC_TRACE").is_ok() {
            eprintln!("trace: nn step {i} loss={:?}", loss.try_to_vec_as::<f32>());
        }
        let grads = GradientsParams::from_grads(loss.backward(), &net);
        // Cosine-распад lr: подавляет осцилляции не-супервизированных пикселей.
        let t = i as f64 / steps.max(1) as f64;
        let lr = 1e-4 + cfg.lr * (0.5 + 0.5 * (std::f64::consts::PI * t).cos());
        net = opt.step(lr, net, grads);
    }

    // Предсказание настоящей дыры: канал маски = настоящая дыра, патч-канал пуст.
    let pred_t = net.forward(input5(&rgb, &hole_map));
    if std::env::var("VC_TRACE").is_ok() {
        let gt = Tensor::<4>::from_data(
            TensorData::new(rgb.to_vec(), Shape::new([1, 3, side_us, side_us])),
            &device,
        );
        let mut wgt3 = Vec::with_capacity(side_us * side_us * 3);
        for _ in 0..3 {
            wgt3.extend_from_slice(&hole_map);
        }
        let wgt_t = Tensor::<4>::from_data(
            TensorData::new(wgt3, Shape::new([1, 3, side_us, side_us])),
            &device,
        );
        let d = pred_t.clone().sub(gt).mul(wgt_t);
        eprintln!(
            "trace: final-eval loss={:?}",
            d.clone().mul(d).mean().try_to_vec_as::<f32>()
        );
    }
    let pred = pred_t.try_to_vec_as::<f32>().expect("f32 предсказание");

    let mut data = frame.data.clone();
    let n = side_us * side_us;
    for ry in 0..side {
        for rx in 0..side {
            let gx = cx0 + rx;
            let gy = cy0 + ry;
            if hole.value(gx, gy) == 0 {
                continue;
            }
            let idx = ((gy as usize * w as usize) + gx as usize) * 4;
            let p = (ry as usize * side_us) + rx as usize;
            for c in 0..3 {
                let v = (pred[c * n + p].clamp(0.0, 1.0) * 255.0).round() as u8;
                data[idx + c] = v;
            }
        }
    }
    Some(Frame { width: w, height: h, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_rect(w: u32, h: u32, corners: [[f32; 2]; 2]) -> Mask {
        vc_core::rasterize(
            &[vc_core::MaskRegion {
                id: 1,
                shape: vc_core::Shape::Rect { corners },
                start: 0,
                end: vc_core::UNTIL_END,
            }],
            0,
            w,
            h,
        )
    }

    #[test]
    fn flat_converges() {
        let frame = Frame::filled(32, 32, [180, 90, 40, 255]);
        let hole = mask_rect(32, 32, [[10.0, 10.0], [14.5, 14.5]]);
        let cfg = NeuralConfig { max_side: 32, steps: 120, ..Default::default() };
        let out = neural_inpaint(&frame, &hole, &cfg, 7).unwrap();
        // Пиксели дыры получаются примерно исходного цвета.
        for (x, y) in [(12, 12), (13, 12), (12, 13), (13, 13)] {
            let p = out.pixel(x, y).unwrap();
            for (c, want) in [180u8, 90, 40].iter().enumerate() {
                assert!(
                    (p[c] as i32 - *want as i32).abs() <= 12,
                    "({x},{y}) {p:?}"
                );
            }
        }
        // Пиксели вне дыры не тронуты, альфа сохранена.
        assert_eq!(out.pixel(0, 0), Some([180, 90, 40, 255]));
        assert_eq!(out.pixel(20, 20), Some([180, 90, 40, 255]));
    }

    #[test]
    fn oversized_hole_rejected() {
        let frame = Frame::filled(64, 64, [10, 10, 10, 255]);
        let hole = mask_rect(64, 64, [[5.0, 5.0], [50.5, 50.5]]);
        let cfg = NeuralConfig { max_side: 32, ..Default::default() };
        assert!(neural_inpaint(&frame, &hole, &cfg, 7).is_none());
    }

    /// Предикат «влезания» согласован с решением neural_inpaint:
    /// маленькая дыра влезает, большая — нет.
    #[test]
    fn hole_fits_matches_rejection() {
        let frame = Frame::filled(64, 64, [10, 10, 10, 255]);
        let cfg = NeuralConfig { max_side: 32, ..Default::default() };
        let small = mask_rect(64, 64, [[5.0, 5.0], [20.0, 20.0]]);
        let big = mask_rect(64, 64, [[5.0, 5.0], [50.5, 50.5]]);
        assert!(hole_fits(&small, &frame, &cfg));
        assert!(!hole_fits(&big, &frame, &cfg));
    }

    fn frame_px(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Frame {
        let mut data = vec![0u8; (w * h) as usize * 4];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                data[i..i + 4].copy_from_slice(&f(x, y));
            }
        }
        Frame::new(w, h, data).unwrap()
    }

    /// M5c: перерисовка с донорской супервизией — донор показывает чистый
    /// шахматный фон под текстом; сеть должна выучить фон, а не блюр.
    #[test]
    fn repaint_supervised_by_donors() {
        let bg = |x: u32, y: u32| if (x / 8 + y / 8) % 2 == 0 { 220u8 } else { 120u8 };
        // Кадр 0: в дыре — «текст» (тон), вне — фон.
        let frame0 = frame_px(64, 64, |x, y| {
            let v = if x >= 16 && x < 48 && y >= 16 && y < 48 { 30 } else { bg(x, y) };
            [v, v, v, 255]
        });
        // Донор −1: чистый фон везде.
        let donor = frame_px(64, 64, |x, y| [bg(x, y), bg(x, y), bg(x, y), 255]);
        let regions = vec![vc_core::MaskRegion {
            id: 1,
            shape: vc_core::Shape::Rect { corners: [[16.0, 16.0], [47.0, 47.0]] },
            start: 0,
            end: vc_core::UNTIL_END,
        }];
        let hole = vc_core::rasterize(&regions, 0, 64, 64);
        let cfg = NeuralConfig { max_side: 64, steps: 500, ..Default::default() };
        let mut lookup = |off: i64| -> Option<Frame> {
            if off == -1 || off == 1 { Some(donor.clone()) } else { None }
        };
        let out = super::neural_repaint(&frame0, &hole, &regions, 0, &cfg, &mut lookup)
            .expect("repaint с донорами");
        // В центре бывшей дыры должны получить фон-шахматку, а не «текст»/серый.
        let gx = 28u32;
        let gy = 28u32;
        let px = out.pixel(gx, gy).unwrap();
        let expect = bg(gx, gy) as i32;
        let got = px[0] as i32;
        assert!(
            (got - expect).abs() <= 50,
            "центр дыры: ожидали ≈{expect}, получили {got}"
        );
        // Углы кадра не тронуты.
        assert_eq!(out.pixel(0, 0).unwrap(), frame0.pixel(0, 0).unwrap());
    }

    /// Большая дыра (> кропа) покрывается тилингом: маска заполнена.
    #[test]
    fn repaint_tiled_big_hole() {
        let bg = |x: u32, y: u32| if (x / 8 + y / 8) % 2 == 0 { 240u8 } else { 60u8 };
        let frame0 = frame_px(96, 96, |x, y| {
            let v = if x >= 8 && x < 88 && y >= 8 && y < 88 { 20 } else { bg(x, y) };
            [v, v, v, 255]
        });
        let donor = frame_px(96, 96, |x, y| [bg(x, y), bg(x, y), bg(x, y), 255]);
        let regions = vec![vc_core::MaskRegion {
            id: 1,
            shape: vc_core::Shape::Rect { corners: [[8.0, 8.0], [87.0, 87.0]] },
            start: 0,
            end: vc_core::UNTIL_END,
        }];
        let hole = vc_core::rasterize(&regions, 0, 96, 96);
        let cfg = NeuralConfig { max_side: 32, steps: 240, ..Default::default() };
        assert!(!hole_fits(&hole, &frame0, &cfg), "дыра должна не влезать в кроп");
        let mut lookup = |off: i64| -> Option<Frame> {
            if off != 0 { Some(donor.clone()) } else { None }
        };
        let out = super::neural_repaint(&frame0, &hole, &regions, 0, &cfg, &mut lookup)
            .expect("repaint с тилингом");
        // центральной точке положено стать фоном (240/60), не «текстом» (20).
        let px = out.pixel(48, 48).unwrap();
        assert!(
            px[0] >= 35 && px[0] <= 250 && (px[0] as i32 - 20).abs() > 15,
            "центр: {px:?}"
        );
    }
}

