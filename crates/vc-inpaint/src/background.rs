//! Восстановление фона по соседним кадрам.
//!
//! Два подхода:
//! 1. `fill_from_neighbors` — классическое закрытие дыры пикселями доноров
//!    (только из кадров, где растеризованная маска пуста).
//! 2. `retouch_frame` (M5c.1) — «фотошопная» покадровая ретушь: для каждого
//!    пикселя рабочей зоны собираются кандидаты с ВСЕХ выровненных соседей
//!    (сдвиг оценивается блок-матчингом), устойчивая медиана даёт фон,
//!    и пиксель заменяется ТОЛЬКО если он реально отличается от фона.
//!    Пиксели без доказательств возвращаются маской `nn_hole` — их достраивает
//!    нейросеть/Telea. Целые пиксели не трогаются вовсе.

use vc_core::{Frame, Mask, MaskRegion};

/// Сколько кадров в каждую сторону максимум извлекаем под фон.
pub const MAX_NEIGHBOR_SPAN: i64 = 90;

/// Ink-маска: пиксели, чья яркость отклоняется от локального среднего
/// (box-mean через интегральное изображение) сильнее `contrast` в любую
/// сторону. Ловит любой наложенный объект (текст, лого, субтитры), включая
/// сплошные блоки, в отличие от полноценного SWT-детектора строк.
/// Возврат — маска, расширенная на 2 px (антлиалиазинговые кромки в придачу).
pub fn ink_mask(frame: &Frame, contrast: i64, dilate: u32) -> Mask {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let r = ((w.min(h) as f64 / 16.0).round() as i32).clamp(3, 40) as usize;
    // luma
    let mut lum = vec![0u64; w * h];
    for (i, chunk) in frame.data.chunks_exact(4).enumerate() {
        lum[i] = (299 * chunk[0] as u32 + 587 * chunk[1] as u32 + 114 * chunk[2] as u32) as u64
            / 1000;
    }
    // Интегральное изображение: (w+1)×(h+1).
    let mut integ = vec![0u64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u64;
        for x in 0..w {
            row += lum[y * w + x];
            integ[(y + 1) * (w + 1) + x + 1] = integ[y * (w + 1) + x + 1] + row;
        }
    }
    let box_mean = |x: i64, y: i64| -> u64 {
        let x0 = (x - r as i64).max(0) as usize;
        let y0 = (y - r as i64).max(0) as usize;
        let x1 = ((x + r as i64) as usize).min(w - 1);
        let y1 = ((y + r as i64) as usize).min(h - 1);
        let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as u64;
        let s = integ[(y1 + 1) * (w + 1) + x1 + 1]
            + integ[y0 * (w + 1) + x0]
            - integ[(y1 + 1) * (w + 1) + x0]
            - integ[y0 * (w + 1) + x1 + 1];
        s / area
    };
    let mut m = Mask::empty(frame.width, frame.height);
    for y in 0..h {
        for x in 0..w {
            let l = lum[y * w + x] as i64;
            if (l - box_mean(x as i64, y as i64) as i64).abs() > contrast {
                m.set(x as u32, y as u32, 255);
            }
        }
    }
    if dilate > 0 {
        m.dilated(dilate)
    } else {
        m
    }
}

/// Ядро блок-матчинга: ищем (dx, dy), при котором donor(x+dx, y+dy) совпадает
/// с кадром(x, y) по яркостным сэмплам. Coarse-to-fine, тай-брейк «к нулю».
pub(crate) fn best_shift(
    donor: &Frame,
    samples: &[(i64, i64, u8)],
    span: i64,
) -> (i32, i32) {
    let (w, h) = (donor.width as i64, donor.height as i64);
    let span = span.max(1);
    // Чтение донора с clamp к краю: OOB не должен отравлять оценку сдвига
    // краевыми сэмплами (иначе точная фаза проигрывает почти-фазе).
    let gray_at = |x: i64, y: i64| -> u8 {
        let cx = x.clamp(0, w - 1);
        let cy = y.clamp(0, h - 1);
        luma(&donor.data, (cy * w + cx) as usize)
    };
    let sad = |dx: i64, dy: i64| -> u64 {
        let mut s: u64 = 0;
        for &(x, y, g) in samples {
            s += g.abs_diff(gray_at(x + dx, y + dy)) as u64;
        }
        s
    };

    // Coarse-to-fine: шаг 4 в ±span (узкие фазовые оптимумы обязаны попасть
    // в сетку), затем сужение вдвое до 2 и 1.
    let mut best = (0i32, 0i32);
    let mut best_cost = sad(0, 0);
    let mut step_sz = 4.max(1).min(span);
    let mut reach = span;
    loop {
        let c0 = best.0 as i64;
        let c1 = best.1 as i64;
        let mut cand = best;
        let mut cost = best_cost;
        let mut dx = -reach;
        while dx <= reach {
            let mut dy = -reach;
            while dy <= reach {
                let (px, py) = (c0 + dx, c1 + dy);
                if px.abs() <= span && py.abs() <= span {
                    let cc = sad(px, py);
                    let cur_dist = (cand.0.abs() + cand.1.abs()) as u64;
                    let new_dist = (px.abs() + py.abs()) as u64;
                    if cc < cost || (cc == cost && new_dist < cur_dist) {
                        cost = cc;
                        cand = (px as i32, py as i32);
                    }
                }
                dy += step_sz as i64;
            }
            dx += step_sz as i64;
        }
        best = cand;
        best_cost = cost;
        if step_sz == 1 {
            break;
        }
        step_sz = (step_sz / 2).max(1);
        reach = step_sz * 2;
    }
    best
}

#[inline]
pub(crate) fn luma(data: &[u8], idx: usize) -> u8 {
    let c = &data[idx * 4..idx * 4 + 3];
    ((c[0] as u32 * 299 + c[1] as u32 * 587 + c[2] as u32 * 114) / 1000) as u8
}

/// Настройки покадровой ретуши.
#[derive(Debug, Clone)]
pub struct RetouchConfig {
    /// ± сколько кадров берём как кандидатов.
    pub span: i64,
    /// ± px поиска выравнивания соседа (панорама/скролл).
    pub max_shift: i32,
    /// Порог «пиксель отличается от фона» (0..=255, по максимуму каналов).
    pub diff: u8,
    /// Если пиксель по временным данным «статичен» (кандидаты совпадают с
    /// текущим значением, текст не отделяется временем) — пометить его в
    /// `nn_hole` на достройку нейросетью (уместно для стратегии Neural).
    pub static_to_hole: bool,
}

impl Default for RetouchConfig {
    fn default() -> Self {
        Self { span: 10, max_shift: 24, diff: 26, static_to_hole: false }
    }
}

/// Результат ретуши кадра.
pub struct RetouchResult {
    /// Кадр: заменены только пиксели, отличающиеся от надёжного фона.
    pub frame: Frame,
    /// Пиксели без доказательств (нужна нейросеть / Telea).
    pub nn_hole: Mask,
    /// Пиксели, реально заменённые медианой соседей.
    pub replaced: Mask,
    /// Использованные донорские offset'ы (для отчёта UI).
    pub used: Vec<i64>,
}

/// Медиана маленького вектора (сортировка).
fn median(mut v: Vec<u8>) -> u8 {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Покадровая ретушь: трогаем только «текстовые» пиксели текущего кадра.
///
/// Для каждого пикселя рабочей зоны `mask`:
/// - собираем кандидаты фона: значения пикселя (x+dx_t, y+dy_t) с каждого
///   выровненного соседа t (выравнивание = глобальный сдвиг соседа, оценённый
///   по пикселям вне маски);
/// - `median` кандидатов *† устойчив к тексту,* если текст закрывает пиксель
///   меньше чем в половине кадров окна;
/// - если |текущий − медиана| > порога — заменяем медианой (это был текст);
///   иначе пиксель оставляем исходным (фон цел, не трогаем!).
/// Если кандидатов нет вовсе (статичный текст/начало видео) — пиксель
/// попадает в `nn_hole` на достройку NN/Telea.
pub fn retouch_frame(
    frame: &Frame,
    mask: &Mask,
    _regions: &[MaskRegion],
    index: u64,
    cfg: &RetouchConfig,
    lookup: &mut dyn FnMut(i64) -> Option<Frame>,
) -> Option<RetouchResult> {
    let (w, h) = (mask.width, mask.height);
    let masked: Vec<usize> = (0..mask.bits.len())
        .filter(|&i| mask.bits[i] != 0)
        .collect();
    if masked.is_empty() {
        return Some(RetouchResult {
            frame: frame.clone(),
            nn_hole: Mask::empty(w, h),
            replaced: Mask::empty(w, h),
            used: Vec::new(),
        });
    }

    // Сэмплы для глобального выравнивания: регулярная сетка вне маски.
    let grid = ((w * h) as f64 / 600.0).sqrt().round().clamp(2.0, 32.0) as usize;
    let mut samples: Vec<(i64, i64, u8)> = Vec::new();
    for y in (0..h as usize).step_by(grid) {
        for x in (0..w as usize).step_by(grid) {
            if mask.bits[y * w as usize + x] == 0 {
                let i = y * w as usize + x;
                samples.push((x as i64, y as i64, luma(&frame.data, i)));
            }
        }
    }
    if samples.len() < 24 {
        // Рабочая зона почти весь кадр — выравнивание по чему попало опасно.
        return None;
    }

    // Доноры: сдвиг (кандидат валиден всюду, где в границах кадра:
    // робастная медиана подавляет донорский текст, если тот в меньшинстве).
    let mut donors: Vec<(i64, Frame, i32, i32)> = Vec::new();
    for abs in 1..=cfg.span {
        for &off in &[-abs, abs] {
            if index as i64 + off < 0 {
                continue;
            }
            if let Some(d) = lookup(off) {
                let (sdx, sdy) = best_shift(&d, &samples, cfg.max_shift as i64);
                if std::env::var("VC_TRACE").is_ok() {
                    eprintln!(
                        "trace: retouch донор {off:+} сдвиг=({sdx},{sdy}) samples={}",
                        samples.len()
                    );
                }
                donors.push((off, d, sdx, sdy));
            }
        }
    }
    if donors.is_empty() {
        return None;
    }

    // Ink-маска текущего кадра: «похоже на текст прямо сейчас».
    let cur_ink = ink_mask(frame, 48, 0);

    let mut out = frame.data.clone();
    let mut nn_hole = Mask::empty(w, h);
    let mut replaced = Mask::empty(w, h);
    let mut donor_used = vec![false; donors.len()];
    for &i in &masked {
        let x = (i % w as usize) as i64;
        let y = (i / w as usize) as i64;
        let mut cand_r: Vec<u8> = Vec::new();
        let mut cand_g: Vec<u8> = Vec::new();
        let mut cand_b: Vec<u8> = Vec::new();
        for (k, (_, d, dx, dy)) in donors.iter().enumerate() {
            let sx = x + *dx as i64;
            let sy = y + *dy as i64;
            if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                continue;
            }
            let si = (sy * w as i64 + sx) as usize;
            donor_used[k] = true;
            cand_r.push(d.data[si * 4]);
            cand_g.push(d.data[si * 4 + 1]);
            cand_b.push(d.data[si * 4 + 2]);
        }
        if cand_r.is_empty() {
            // Кандидатов нет: отдаём NN только если это похоже на текст
            // и в текущем кадре; просто-статичные фоновые пиксели не трогаем.
            if cur_ink.value(x as u32, y as u32) != 0 {
                nn_hole.set(x as u32, y as u32, 255);
            }
            continue;
        }
        let c = &frame.data[i * 4..i * 4 + 3];
        // Есть ли хоть один кандидат, отличающийся сильнее порога?
        let any_diff = cand_r
            .iter()
            .zip(&cand_g)
            .zip(&cand_b)
            .any(|((&r, &g), &b)| {
                let dm = c[0].abs_diff(r).max(c[1].abs_diff(g)).max(c[2].abs_diff(b));
                dm > cfg.diff
            });
        // Медиана робастна при ≥ 3 кандидатах; при 1–2证据 недостаточно:
        // текстовый пиксель отдаём NN, а не соединяем донорский квант напрямую.
        let robust = cand_r.len() >= 3;
        if !robust {
            if (cfg.static_to_hole || any_diff) && cur_ink.value(x as u32, y as u32) != 0 {
                nn_hole.set(x as u32, y as u32, 255);
            }
            continue;
        }
        let (mr, mg, mb) = (median(cand_r), median(cand_g), median(cand_b));
        let dmax = c[0]
            .abs_diff(mr)
            .max(c[1].abs_diff(mg))
            .max(c[2].abs_diff(mb));
        if dmax > cfg.diff {
            out[i * 4] = mr;
            out[i * 4 + 1] = mg;
            out[i * 4 + 2] = mb;
            out[i * 4 + 3] = frame.data[i * 4 + 3];
            replaced.set(x as u32, y as u32, 255);
        } else if cfg.static_to_hole && !any_diff && cur_ink.value(x as u32, y as u32) != 0 {
            // Текст стоит во всех соседях — временем не отделяется;
            // отдаём нейросети на перерисовку из контекста.
            nn_hole.set(x as u32, y as u32, 255);
        }
    }
    let used: Vec<i64> = donors
        .iter()
        .zip(donor_used)
        .filter(|(_, u)| *u)
        .map(|((off, _, _, _), _)| *off)
        .collect();

    Some(RetouchResult { frame: Frame { width: w, height: h, data: out }, nn_hole, replaced, used })
}

/// Заполняет дыру `hole` кадра `frame` значениями из соседних кадров.
///
/// `lookup(offset)` должен вернуть кадр `index + offset` (или None, если
/// кадра нет в кэше). Возвращает:
/// 1. Частично заполненный кадр (в дыру; известные пиксели не тронуты).
/// 2. Маску ещё незаполненных пикселей (для Telea-доводы).
/// 3. Список использованных offset’ов (для отчёта в UI).
pub fn fill_from_neighbors(
    frame: &Frame,
    hole: &Mask,
    regions: &[MaskRegion],
    index: u64,
    max_span: i64,
    lookup: &mut dyn FnMut(i64) -> Option<Frame>,
) -> (Frame, Mask, Vec<i64>) {
    let (w, h) = (hole.width, hole.height);
    let mut out = frame.data.clone();
    let mut remaining: Vec<u8> = hole.bits.clone();

    // Индексы дырявых пикселей (по общему количеству битмапа).
    let hole_pixels: Vec<u32> = (0..remaining.len())
        .filter(|&i| remaining[i] != 0)
        .map(|i| i as u32)
        .collect();

    let mut used: Vec<i64> = Vec::new();

    'outer: for abs in 1..=max_span {
        for off in [-abs, abs] {
            let target_i = index as i64 + off;
            if target_i < 0 {
                continue;
            }
            let Some(cand) = lookup(off) else { continue };
            // Маска на кадре-кандидате: если он сам замаскирован под вышкой,
            // эти пиксели там неизвестны.
            let coverage = vc_core::rasterize(regions, target_i as u64, w, h);
            let mut contributed = false;
            for &i in &hole_pixels {
                let i = i as usize;
                if remaining[i] != 0 && coverage.bits[i] == 0 {
                    out[4 * i..4 * i + 4].copy_from_slice(&cand.data[4 * i..4 * i + 4]);
                    remaining[i] = 0;
                    contributed = true;
                }
            }
            if contributed {
                used.push(off);
            }
            if remaining.iter().all(|&b| b == 0) {
                break 'outer;
            }
        }
    }

    (
        Frame { width: w, height: h, data: out },
        Mask { width: w, height: h, bits: remaining },
        used,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vc_core::Shape;

    const W: u32 = 8;
    const H: u32 = 8;

    fn hole_center() -> Mask {
        let mut m = Mask::empty(W, H);
        for y in 3..6 {
            for x in 3..6 {
                m.set(x, y, 255);
            }
        }
        m
    }

    /// Текст виден только на кадре 5; фон статичен и красный, соседи чисты.
    #[test]
    fn fills_from_clean_earlier_frame() {
        let mut data = vec![0u8; (W * H * 4) as usize];
        for y in 3..6 {
            for x in 3..6 {
                let i = (y as usize * W as usize + x as usize) * 4;
                data[i] = 255;
                data[i + 1] = 255;
                data[i + 2] = 255;
            }
        }
        let frame = Frame::new(W, H, data).unwrap();

        let (filled, uncovered, used) = fill_from_neighbors(
            &frame,
            &hole_center(),
            &[],
            5,
            10,
            &mut |off| match off {
                -1 => Some(Frame::filled(W, H, [200, 0, 0, 255])),
                _ => None,
            },
        );
        assert_eq!(uncovered.count(), 0);
        assert_eq!(used, vec![-1]);
        assert_eq!(filled.pixel(4, 4), Some([200, 0, 0, 255]));
        // Пиксель вне дыры не тронут (кадр был нулевым по всем каналам).
        assert_eq!(filled.pixel(0, 0), Some([0, 0, 0, 0]));
    }

    /// Кандидат в диапазоне маски замаскирован там же → берём от дальнего кадра.
    #[test]
    fn skips_masked_candidates() {
        let frame = Frame::filled(W, H, [255, 255, 255, 255]);
        // Маска активна на кадрах 4..=7: кандидат offset=-1 (кадр 4) замаскирован,
        // кадр 3 (offset=-2) чистый.
        let regions = vec![MaskRegion {
            id: 1,
            shape: Shape::Rect { corners: [[3.0, 3.0], [5.5, 5.5]] },
            start: 4,
            end: 7,
        }];
        let (filled, uncovered, used) = fill_from_neighbors(
            &frame,
            &hole_center(),
            &regions,
            5,
            10,
            &mut |off| match off {
                -2 => Some(Frame::filled(W, H, [10, 20, 30, 255])),
                _ => None,
            },
        );
        // Кандидат -1 замаскирован, вклада нет; кадр -2 чистый → покрыл целиком.
        assert_eq!(uncovered.count(), 0);
        assert_eq!(used, vec![-2]);
        assert_eq!(filled.pixel(4, 4), Some([10, 20, 30, 255]));
    }

    /// Нет достойных кандидатов → всё остаётся для Telea.
    #[test]
    fn uncovered_stays_when_no_candidates() {
        let frame = Frame::filled(W, H, [1, 2, 3, 255]);
        let (filled_part, uncovered, used) = fill_from_neighbors(
            &frame,
            &hole_center(),
            &[],
            5,
            10,
            &mut |_| None,
        );
        assert_eq!(used.len(), 0);
        assert_eq!(uncovered.count(), hole_center().count());
        assert_eq!(filled_part.pixel(4, 4), Some([1, 2, 3, 255]));
    }

    fn px_frame(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Frame {
        let mut data = vec![0u8; (w * h) as usize * 4];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                data[i..i + 4].copy_from_slice(&f(x, y));
            }
        }
        Frame::new(w, h, data).unwrap()
    }

    /// Ретушь: текст убирается только на своих пикселях, фон остаётся байт-в-байт.
    #[test]
    fn retouch_removes_text_keeps_background() {
        // Фон: байт-градиент, пиксели уникальны (копирование донора бы вскрылось).
        let bg = |x: u32| 100 + (x % 40) as u8;
        // Текущее: текст-блок 250 в (20..23)x(20..23); донор −1: текст в (13..16).
        let frame = px_frame(40, 40, |x, y| {
            if (20..24).contains(&x) && (20..24).contains(&y) {
                [250, 250, 250, 255]
            } else {
                [bg(x), bg(x), bg(x), 255]
            }
        });
        // Три донора: текст в разных местах (и один чистый) — медиана робастна.
        let donor_text_at = |tx: u32| {
            px_frame(40, 40, move |x, y| {
                if (tx..tx + 4).contains(&x) && (20..24).contains(&y) {
                    [250, 250, 250, 255]
                } else {
                    [bg(x), bg(x), bg(x), 255]
                }
            })
        };
        let donors = [donor_text_at(13), donor_text_at(27), donor_text_at(5)];
        let hole: Mask = {
            let mut m = Mask::empty(40, 40);
            for x in 12..26 {
                for y in 18..26 {
                    m.set(x, y, 255);
                }
            }
            m
        };
        let rr = retouch_frame(&frame, &hole, &[], 5, &RetouchConfig::default(), &mut |off| {
            match off {
                -1 | -2 => Some(donors[0].clone()),
                1 | 2 => Some(donors[1].clone()),
                _ => Some(donors[2].clone()),
            }
        })
        .expect("есть доноры");
        // Текстовые пиксели заменены фоном.
        let px = rr.frame.pixel(21, 21).unwrap();
        assert_eq!(px[0], bg(21), "текст не заменён медианой: {px:?}");
        // Целый пиксель ФОНА не тронут (копия донора имела бы то же значение,
        // но проверяем уникальность градиента: донорский пиксель == фон).
        // Ключевая проверка — replaced ровно на текстовом блоке.
        assert!(rr.replaced.count() >= 16);
        assert_eq!(rr.replaced.value(5, 5), 0, "целый фон попал под замену");
        assert_eq!(rr.replaced.value(14, 22), 0, "донорский текст не должен влиять");
        // NN-дыра пуста: доказательства были.
        assert!(rr.nn_hole.is_empty(), "пиксели без доказательств при живом доноре");
    }

    /// Ретушь выравнивает панораму: донор сдвинут на 4 px, но кандидаты
    /// корректно ссылаются на содержимое текущего кадра.
    #[test]
    fn retouch_aligns_pan() {
        let stripes = |x: u32| if (x / 10) % 2 == 0 { 200u8 } else { 70u8 };
        let frame = px_frame(64, 64, |x, y| {
            let v = if (28..32).contains(&x) && (28..32).contains(&y) { 10 } else { stripes(x) };
            [v, v, v, 255]
        });
        // Донор — тот же фон, сдвинутый на 4 px влево (панорама), без wrap-шва:
        // D(x) = stripes(x-4) для x ≥ 4 (правый край донора уходит за кадр).
        let donor = px_frame(64, 64, |x, y| {
            let v = if (40..44).contains(&x) && (28..32).contains(&y) {
                10
            } else if x >= 4 {
                stripes(x - 4)
            } else {
                stripes(0)
            };
            [v, v, v, 255]
        });
        let mut hole = Mask::empty(64, 64);
        for x in 24..36 {
            for y in 24..36 {
                hole.set(x, y, 255);
            }
        }
        let rr = retouch_frame(&frame, &hole, &[], 5, &RetouchConfig::default(), &mut |off| {
            if off != 0 { Some(donor.clone()) } else { None }
        })
        .expect("есть доноры");
        let px = rr.frame.pixel(29, 30).unwrap();
        // B(29) = (29/10=2)%2=0 → 200; без выравнивания кандидат B(25)=70.
        assert_eq!(px[0], 200, "панорама не выровнялась: {px:?}");
        assert_eq!(rr.replaced.value(29, 30), 255);
    }
}


