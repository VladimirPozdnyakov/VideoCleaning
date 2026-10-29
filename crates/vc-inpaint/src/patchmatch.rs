//! PatchMatch — перенос патчей из известной области в дыру
//! («content-aware fill», Barnes et al. 2009, упрощённая версия).
//!
//! Для каждого пикселя дыры поддерживается поле ближайших соседей (NNF):
//! координата исходного патча. Проходы пропагации +
//! случайного поиска минимизируют_PATCH разницу патчей (SAD по известным
//! пикселям), затем голосование перекрывающимися патчами строит заливку.
//! Для аниме с плоскими цветами/оговорёнными текстурами результат резкий и
//! фазово-точный, в отличие от нейро-блюра; чистый Rust, детерминированный
//! seed, без сторонних бинарников.

use vc_core::{Frame, Mask};

#[derive(Debug, Clone)]
pub struct PatchMatchConfig {
    /// Сторона квадратного патча (нечётная).
    pub patch: u32,
    /// Проходов пропагации + случайного поиска.
    pub iters: u16,
    /// Seed ГПСЧ (детерминированность).
    pub seed: u64,
}

impl Default for PatchMatchConfig {
    fn default() -> Self {
        Self { patch: 9, iters: 5, seed: 0x5EED }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1).max(1) as u64) as i64
    }
}

/// Заливает дыру патчами из известной области кадра.
pub fn fill(frame: &Frame, hole: &Mask, cfg: &PatchMatchConfig) -> Frame {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let r = (cfg.patch / 2) as usize;
    let known = |i: usize| hole.bits[i] == 0;

    // RGB по-канально.
    let n = w * h;
    let mut red = vec![0f32; n];
    let mut green = vec![0f32; n];
    let mut blue = vec![0f32; n];
    for (i, chunk) in frame.data.chunks_exact(4).enumerate() {
        red[i] = chunk[0] as f32;
        green[i] = chunk[1] as f32;
        blue[i] = chunk[2] as f32;
    }

    // Валидные центры исходных патчей: патч целиком в известной области.
    // Если таких нет — уменьшаем радиус (в пределе патч-точка).
    let valid_centers = |rad: usize| -> Vec<usize> {
        let mut v = Vec::new();
        if h < 2 * rad + 1 || w < 2 * rad + 1 {
            return v;
        }
        for y in rad..h - 2 * rad {
            for x in rad..w - 2 * rad {
                let mut ok = true;
                'probe: for dy in 0..=rad * 2 {
                    for dx in 0..=rad * 2 {
                        if !known((y + dy) * w + (x + dx)) {
                            ok = false;
                            break 'probe;
                        }
                    }
                }
                if ok {
                    v.push(y * w + x);
                }
            }
        }
        v
    };
    let mut rad = r;
    let mut srcs = valid_centers(rad);
    while srcs.len() < 8 && rad > 0 {
        rad -= 1;
        srcs = valid_centers(rad);
    }
    if srcs.is_empty() {
        // Искать нечего — вернуть кадр без изменений.
        return frame.clone();
    }
    let eff_r = rad; // фактический рабочий радиус патча

    // Пиксели дыры.
    let hole_px: Vec<usize> = (0..n).filter(|&i| !known(i)).collect();
    if hole_px.is_empty() {
        return frame.clone();
    }

    // Сравнение патчей по известным пикселям оригинала.
    let cost = |pi: usize, qi: usize| -> f32 {
        let (px, py) = (pi % w, pi / w);
        let (qx, qy) = (qi % w, qi / w);
        let mut sum = 0f32;
        let mut cnt = 0u32;
        for dy in 0..=eff_r * 2 {
            for dx in 0..=eff_r * 2 {
                let tx = px as i64 + dx as i64 - eff_r as i64;
                let ty = py as i64 + dy as i64 - eff_r as i64;
                let sx = qx as i64 + dx as i64 - eff_r as i64;
                let sy = qy as i64 + dy as i64 - eff_r as i64;
                if tx < 0 || ty < 0 || tx >= w as i64 || ty >= h as i64 {
                    continue;
                }
                if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                    continue;
                }
                let t = (ty as usize) * w + (tx as usize);
                let s = (sy as usize) * w + (sx as usize);
                if !known(t) || !known(s) {
                    continue;
                }
                let d = (red[t] - red[s]).abs() + (green[t] - green[s]).abs() + (blue[t] - blue[s])
                    .abs();
                sum += d;
                cnt += 1;
            }
        }
        if cnt == 0 {
            f32::MAX / 4.0
        } else {
            sum / cnt as f32
        }
    };

    let mut rng = Rng(cfg.seed | 1);
    // NNF: (источник, цена) для каждого пикселя дыры.
    let mut nnf: Vec<(usize, f32)> = Vec::with_capacity(hole_px.len());
    for &_p in &hole_px {
        let q = srcs[rng.below(srcs.len() as u64) as usize];
        let c = cost(_p, q);
        nnf.push((q, c));
    }
    let diag = w.max(h).next_power_of_two() as i64;

    for it in 0..cfg.iters {
        let forward = it % 2 == 0;
        let seq: Vec<usize> = if forward {
            (0..hole_px.len()).collect()
        } else {
            (0..hole_px.len()).rev().collect()
        };
        // Сосед в направлении скана для пропагации.
        let step: i64 = if forward { 1 } else { -1 };
        // Индексационная карта пиксель дыры → позиция в списке.
        let mut index_of: Vec<i64> = vec![-1; n];
        for (k, &p) in hole_px.iter().enumerate() {
            index_of[p] = k as i64;
        }
        for &k in &seq {
            let p = hole_px[k];
            let (px, py) = (p % w, p / w);
            let mut best_q = nnf[k].0;
            let mut best_c = nnf[k].1;

            // Пропагация: кандидат соседа, сдвинутый на наш шаг.
            for (dx, dy) in [(step, 0i64), (0, step)] {
                let nx = px as i64 + dx;
                let ny = py as i64 + dy;
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let ni = (ny as usize) * w + (nx as usize);
                if index_of[ni] < 0 {
                    continue;
                }
                let (nq, _nc) = nnf[index_of[ni] as usize];
                // Если сосед (p+dx, p+dy) использует источник q, для p
                // кандидатом будет q, смещённый так же, как сосед смещён от p.
                let cand = nq as i64 + dx * step + dy * step;
                if cand < 0 || cand >= n as i64 {
                    continue;
                }
                let cand = cand as usize;
                let (cx, cy) = (cand % w, cand / w);
                if cx < eff_r || cy < eff_r || cx + eff_r >= w || cy + eff_r >= h {
                    continue;
                }
                if !srcs.contains(&cand) {
                    continue;
                }
                let c = cost(p, cand);
                if c < best_c {
                    best_c = c;
                    best_q = cand;
                }
            }

            // Случайный поиск вокруг текущего источника с ужимающимся радиусом.
            let (bx, by) = ((best_q % w) as i64, (best_q / w) as i64);
            let mut radius = diag;
            loop {
                let qx = bx + rng.range(-radius, radius);
                let qy = by + rng.range(-radius, radius);
                if qx >= eff_r as i64
                    && qy >= eff_r as i64
                    && (qx as usize) + eff_r < w
                    && (qy as usize) + eff_r < h
                {
                    let qi = (qy as usize) * w + (qx as usize);
                    if srcs.contains(&qi) {
                        let c = cost(p, qi);
                        if c < best_c {
                            best_c = c;
                            best_q = qi;
                        }
                    }
                }
                if radius <= 1 {
                    break;
                }
                radius /= 2;
            }
            nnf[k].0 = best_q;
            nnf[k].1 = best_c;
        }
    }

    // Голосование: каждый патч дыры предлагает значения своего источника.
    let mut acc = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    let mut wsum = vec![0f32; n];
    for (k, &p) in hole_px.iter().enumerate() {
        let (q, c) = nnf[k];
        let weight = 1.0 / (c + 1.0);
        let (px, py) = (p % w, p / w);
        let (qx, qy) = (q % w, q / w);
        for dy in 0..=eff_r * 2 {
            for dx in 0..=eff_r * 2 {
                let tx = px as i64 + dx as i64 - eff_r as i64;
                let ty = py as i64 + dy as i64 - eff_r as i64;
                let sx = qx as i64 + dx as i64 - eff_r as i64;
                let sy = qy as i64 + dy as i64 - eff_r as i64;
                if tx < 0 || ty < 0 || tx >= w as i64 || ty >= h as i64 {
                    continue;
                }
                if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                    continue;
                }
                let t = (ty as usize) * w + (tx as usize);
                let s = (sy as usize) * w + (sx as usize);
                if !known(t) || known(s) {
                    continue;
                }
                let _ = known(t); // цель必须是 дыра
                if hole.bits[t] == 0 {
                    continue; // не дыра — не нашей заботы
                }
                wsum[t] += weight;
                acc[0][t] += weight * red[s];
                acc[1][t] += weight * green[s];
                acc[2][t] += weight * blue[s];
            }
        }
    }

    let mut data = frame.data.clone();
    for &t in &hole_px {
        if wsum[t] <= 0.0 {
            // не покрыто голосами — берём центра первого валидного патча
            let s = srcs[0];
            for (c, ch) in [red.clone(), green.clone(), blue.clone()].iter().enumerate() {
                let v = ch[s].clamp(0.0, 255.0).round() as u8;
                data[t * 4 + c] = v;
            }
            continue;
        }
        for c in 0..3 {
            let v = (acc[c][t] / wsum[t]).clamp(0.0, 255.0).round() as u8;
            data[t * 4 + c] = v;
        }
    }
    Frame { width: frame.width, height: frame.height, data }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vc_core::Mask;

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

    fn hole_rect(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> Mask {
        let mut m = Mask::empty(w, h);
        for y in y0..=y1 {
            for x in x0..=x1 {
                m.set(x, y, 255);
            }
        }
        m
    }

    /// Плоская заливка: цвет точно восстанавливается.
    #[test]
    fn flat_filled_exactly() {
        let frame = px_frame(48, 48, |_, _| [90, 160, 40, 255]);
        let hole = hole_rect(48, 48, 16, 16, 31, 31);
        let out = fill(&frame, &hole, &PatchMatchConfig::default());
        for y in 18..30 {
            for x in 18..30 {
                assert_eq!(out.pixel(x, y).unwrap(), [90, 160, 40, 255], "({x},{y})");
            }
        }
        // Вне дыры — байт-в-байт.
        assert_eq!(out.pixel(0, 0).unwrap(), frame.pixel(0, 0).unwrap());
    }

    /// Шахматка продолжается внутрь дыры у кромок (фаза не поломана).
    #[test]
    fn texture_continues_at_edges() {
        let bg = |x: u32, y: u32| if (x / 8 + y / 8) % 2 == 0 { 230u8 } else { 40u8 };
        let frame = px_frame(64, 64, |x, y| [bg(x, y), bg(x, y), bg(x, y), 255]);
        let hole = hole_rect(64, 64, 24, 24, 39, 39);
        let cfg = PatchMatchConfig { patch: 9, iters: 6, ..Default::default() };
        let out = fill(&frame, &hole, &cfg);
        // Кромочные дырявые пиксели (соседние с известными) обязаны совпасть
        // с продолжением фазы фона.
        let checks = [
            (24usize, 26usize),
            (26, 24),
            (39, 36),
            (36, 39),
        ];
        for (x, y) in checks {
            let expect = bg(x as u32, y as u32);
            let got = out.pixel(x as u32, y as u32).unwrap()[0];
            assert!(
                (got as i32 - expect as i32).abs() <= 60,
                "({x},{y}): ожидали ≈{expect}, получили {got}"
            );
        }
    }
}
