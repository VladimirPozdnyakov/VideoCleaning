//! Классический инпейнтинг — алгоритм Telea (упрощённая реализация).
//!
//! Схема:
//! 1. Строим поле расстояний `T` для пикселей дыры методом
//!    Fast Marching (эквивалент BFS-дистанции от границы дыры, Dijkstra с весом 1).
//! 2. Заполняем пиксели дыры в порядке роста T. Для каждого пикселя
//!    считаем взвешенное среднее по окну радиуса `radius`, учитывая:
//!    - близость (1/d²),
//!    - «уровень» 1/(1+T(q)) (предпочесть пиксели ближе к границе),
//!    - направленный член — выравнивание с градиентом T (структура края).
//!
//! Упрощения против оригинала (Telea 2004 / OpenCV):
//! - дистанция целочисленная (без квадратичной развязки),
//! - градиент T по доступным сосоедей с односторонними разностями,
//! - вес направления floor через epsilon.
//! Этого достаточно для текста/штрихов на 2D-анимации; основной режим —
//! нейросетевой (M5).

use std::collections::BTreeSet;

use rayon::prelude::*;
use vc_core::{Frame, Mask};

const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;
const INF: u32 = u32::MAX;

/// Поле расстояний и порядок заполнения пикселей дыры.
struct Field {
    /// Финальные флаги: KNOWN — известен (заполнен), else INSIDE.
    flags: Vec<u8>,
    /// Дистанция от границы дыры (в целых пикселях), 0 для границы.
    t: Vec<u32>,
    /// Индексы пикселей дыры в порядке возрастания T.
    order: Vec<u32>,
}

fn build_field(mask: &Mask) -> Field {
    let w = mask.width as usize;
    let h = mask.height as usize;
    let n = w * h;

    let mut flags = vec![0u8; n];
    let mut t = vec![0u32; n];
    for (i, &m) in mask.bits.iter().enumerate() {
        if m != 0 {
            flags[i] = INSIDE;
            t[i] = INF;
        }
    }

    let mut queue: BTreeSet<(u32, usize)> = BTreeSet::new();

    // Стартовая полоса: пиксели дыры, соседние с известными.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if flags[i] != INSIDE {
                continue;
            }
            let bound = (x > 0 && flags[i - 1] == KNOWN)
                || (x + 1 < w && flags[i + 1] == KNOWN)
                || (y > 0 && flags[i - w] == KNOWN)
                || (y + 1 < h && flags[i + w] == KNOWN);
            if bound {
                flags[i] = BAND;
                t[i] = 0;
                queue.insert((0, i));
            }
        }
    }

    // Fast Marching (Dijkstra, вес дуги 1): pop min → расширение.
    let mut order: Vec<u32> = Vec::new();
    while let Some(&(tcur, i)) = queue.first() {
        queue.remove(&(tcur, i));
        if flags[i] != BAND {
            continue; // ленивая дедупликация повторных вставок
        }
        flags[i] = KNOWN;
        order.push(i as u32);

        let x = (i % w) as i32;
        let y = (i / w) as i32;
        let step = tcur + 1;
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            if flags[j] == INSIDE && step < t[j] {
                t[j] = step;
                flags[j] = BAND;
                queue.insert((step, j));
            }
        }
    }

    Field { flags, t, order }
}

/// Заполняет один канал (R/G/B) по полю и порядку. Возвращает копию канала.
fn fill_channel(
    data: &[u8],
    t: &[u32],
    flags_final: &[u8],
    order: &[u32],
    w: usize,
    h: usize,
    radius: usize,
    ch: usize,
) -> Vec<u8> {
    let mut out = data.to_vec();
    // filled: рабочий флаг «этот пиксель уже можно использовать как источник».
    let mut filled = flags_final.to_vec();
    let r = radius as i32;
    let eps = 1e-3f32;
    let w_i = w as i32;
    let h_i = h as i32;

    for &pi in order {
        let p = pi as usize;
        let x = (p % w) as i32;
        let y = (p / w) as i32;

        // Градиент T в точке p — к доступным (известным) соседям.
        let tval = |q: usize| t[q] as f32;
        let elig = |q: usize| filled[q] != INSIDE;
        let mut gx = 0.0f32;
        let mut gy = 0.0f32;
        if x + 1 < w_i && y >= 0 {
            let right = (y as usize) * w + (x as usize + 1);
            let left = (y as usize) * w + (x as usize - 1).max(0);
            let lr = x - 1 >= 0 && elig(left);
            let rr = x + 1 < w_i && elig(right);
            match (lr, rr) {
                (true, true) => gx = (tval(right) - tval(left)) * 0.5,
                (false, true) => gx = tval(right) - tval(p),
                (true, false) => gx = tval(p) - tval(left),
                (false, false) => {}
            }
        }
        if y + 1 < h_i && y >= 0 {
            let up = (y as usize - 1).max(0) * w + x as usize;
            let down = (y as usize + 1).min(h - 1) * w + x as usize;
            let ur = y - 1 >= 0 && elig(up);
            let dr = y + 1 < h_i && elig(down);
            match (ur, dr) {
                (true, true) => gy = (tval(down) - tval(up)) * 0.5,
                (false, true) => gy = tval(down) - tval(p),
                (true, false) => gy = tval(p) - tval(up),
                (false, false) => {}
            }
        }
        let g_len = (gx * gx + gy * gy).sqrt();

        // Взвешенное среднее по окну.
        let mut sum = 0.0f32;
        let mut wsum = 0.0f32;
        for dy in -r..=r {
            let ny = y + dy;
            if ny < 0 || ny >= h_i {
                continue;
            }
            for dx in -r..=r {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                if nx < 0 || nx >= w_i {
                    continue;
                }
                let q = (ny as usize) * w + nx as usize;
                if filled[q] == INSIDE {
                    continue;
                }
                let d2 = (dx * dx + dy * dy) as f32;
                let dist = d2.sqrt();
                let dirx = dx as f32 / dist;
                let diry = dy as f32 / dist;
                let dot = if g_len > 1e-6 {
                    (dirx * gx + diry * gy) / g_len
                } else {
                    0.0
                };
                let dot = dot.abs().min(1.0);
                let level = 1.0 + t[q] as f32;
                let wt = (dot + eps) / (d2 * level);
                sum += wt * out[4 * q + ch] as f32;
                wsum += wt;
            }
        }

        let v = if wsum > 0.0 {
            (sum / wsum).round().clamp(0.0, 255.0) as u8
        } else {
            out[4 * p + ch] // окно пустое — оставляем как есть (крайний случай)
        };
        out[4 * p + ch] = v;
        filled[p] = KNOWN;
    }
    out
}

/// Инпейнтинг кадра по маске (255 = пустить, 0 = оставить).
/// Альфа-канал не трогаем.
pub fn telea_inpaint(frame: &Frame, mask: &Mask, radius: u32) -> Frame {
    debug_assert_eq!(frame.width, mask.width);
    debug_assert_eq!(frame.height, mask.height);
    if mask.is_empty() {
        return frame.clone();
    }
    let field = build_field(mask);
    if field.order.is_empty() {
        // Маска без известного окружения (например, весь кадр) — отдаём как есть.
        return frame.clone();
    }
    let w = mask.width as usize;
    let h = mask.height as usize;

    // Три канала параллельно.
    let channels: Vec<Vec<u8>> = (0..3)
        .into_par_iter()
        .map(|ch| {
            fill_channel(
                &frame.data,
                &field.t,
                &field.flags,
                &field.order,
                w,
                h,
                radius as usize,
                ch,
            )
        })
        .collect();

    let mut data = frame.data.clone();
    for (ch, out) in channels.into_iter().enumerate() {
        for (idx, v) in out.into_iter().enumerate() {
            if idx % 4 == ch {
                data[idx] = v;
            }
        }
    }
    Frame { width: frame.width, height: frame.height, data }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vc_core::Frame;

    #[test]
    fn flat_image_inpaints_to_same_color() {
        let frame = Frame::filled(9, 9, [100, 150, 200, 255]);
        let mut mask = Mask::empty(9, 9);
        for y in 3..6 {
            for x in 3..6 {
                mask.set(x, y, 255);
            }
        }
        let out = telea_inpaint(&frame, &mask, 5);
        for y in 0..9 {
            for x in 0..9 {
                assert_eq!(
                    out.pixel(x, y),
                    Some([100, 150, 200, 255]),
                    "x={x}, y={y}"
                );
            }
        }
    }

    #[test]
    fn empty_mask_is_passthrough() {
        let frame = Frame::filled(5, 5, [1, 2, 3, 255]);
        let mask = Mask::empty(5, 5);
        let out = telea_inpaint(&frame, &mask, 3);
        assert_eq!(out, frame);
    }

    #[test]
    fn full_frame_mask_is_passthrough() {
        let frame = Frame::filled(5, 5, [7, 7, 7, 255]);
        let mask = {
            let mut m = Mask::empty(5, 5);
            for y in 0..5 {
                for x in 0..5 {
                    m.set(x, y, 255);
                }
            }
            m
        };
        let out = telea_inpaint(&frame, &mask, 3);
        assert_eq!(out, frame);
    }

    #[test]
    fn gradient_continues_through_hole() {
        // Горизонтальный градиент 4+2x; в центре дыра.
        let w = 48u32;
        let h = 16u32;
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = (4 + 2 * x).min(255) as u8;
                data.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let frame = Frame::new(w, h, data).unwrap();

        let mut mask = Mask::empty(w, h);
        for y in 6..10 {
            for x in 16..32 {
                mask.set(x, y, 255);
            }
        }
        let out = telea_inpaint(&frame, &mask, 6);

        for y in 6..10 {
            for x in 16..32 {
                let expected = (4 + 2 * x).min(255) as u8;
                let got = out.pixel(x, y).unwrap()[0];
                let err = got.abs_diff(expected);
                assert!(
                    err <= 24,
                    "x={x} y={y}: ожидалось ~{expected}, получено {got}"
                );
            }
        }
    }

    #[test]
    fn alpha_channel_untouched() {
        // Кадр с альфой != 255: инпейнтинг не должен её менять.
        let mut data = Vec::with_capacity(9 * 9 * 4);
        for _ in 0..9 * 9 {
            data.extend_from_slice(&[10, 20, 30, 77]);
        }
        let frame = Frame::new(9, 9, data).unwrap();
        let mut mask = Mask::empty(9, 9);
        mask.set(4, 4, 255);
        let out = telea_inpaint(&frame, &mask, 3);
        assert_eq!(out.pixel(4, 4).unwrap()[3], 77);
        assert_eq!(out.pixel(0, 0).unwrap()[3], 77);
    }
}
