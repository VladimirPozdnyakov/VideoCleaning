//! Детектор текста для аниме-оверлеев и хардсабов.
//!
//! Пайплайн (по мотивам SWT/MSER-семейства, упрощённо):
//! 1. локальное среднее яркости (box-blur через интегральное изображение);
//! 2. бинаризация отклонений в обе стороны (тёмный текст на светлом и наоборот);
//! 3. связанные компоненты «чернил»;
//! 4. эвристики размера/пропорций/заполнения;
//! 5. группировка штрихов в строки (горизонтальная и вертикальная, юнион-финд);
//! 6. дедупликация пересечений, отсортированный список `Detection`.
//!
//! Результат — предложения: пользователь подтверждает/правит/отменяет их в GUI.

use vc_core::Frame;

/// Настройки детекции.
#[derive(Debug, Clone)]
pub struct DetectOptions {
    /// Порог локального контраста для «чернил» (шкала 0..=255).
    pub contrast: f32,
    /// Минимальная высота строки (px).
    pub min_char_h: u32,
    /// Максимальная высота строки (px); 0 = 30% высоты кадра.
    pub max_char_h: u32,
    /// Множитель межштрихового промежутка (доля от высоты строки).
    pub group_gap: f32,
    /// Радиус локального среднего (px); 0 = авто (2×макс. высота строки).
    pub local_radius: u32,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            contrast: 48.0,
            min_char_h: 8,
            max_char_h: 0,
            group_gap: 2.0,
            local_radius: 0,
        }
    }
}

/// Найденная текстовая область (кадровые координаты, включительно).
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    /// 0..1 — «похожесть на текст»: плотность чернил в боксе.
    pub confidence: f32,
}

impl Detection {
    pub fn width(&self) -> u32 {
        self.x1 - self.x0 + 1
    }
    pub fn height(&self) -> u32 {
        self.y1 - self.y0 + 1
    }
}

const INF: u32 = u32::MAX;

/// Яркость кадра (шкала 0..=255).
fn luma(frame: &Frame) -> (Vec<f32>, u32, u32) {
    let (w, h) = (frame.width, frame.height);
    let mut l = vec![0f32; (w * h) as usize];
    for (i, chunk) in frame.data.chunks_exact(4).enumerate() {
        l[i] =
            (299 * chunk[0] as u32 + 587 * chunk[1] as u32 + 114 * chunk[2] as u32) as f32 / 1000.0;
    }
    (l, w, h)
}

/// Локальное среднее яркости радиуса `r` (общая площадь (2r+1)², границы
/// усечены) через интегральное изображение.
fn box_mean(l: &[f32], w: u32, h: u32, r: u32) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    let iw = wu + 1;
    let mut ii = vec![0f64; iw * (hu + 1)];
    for y in 0..hu {
        let mut row = 0f64;
        for x in 0..wu {
            row += l[y * wu + x] as f64;
            ii[(y + 1) * iw + x + 1] = ii[y * iw + x + 1] + row;
        }
    }
    let rf = r as i64;
    let mut out = vec![0f32; wu * hu];
    for y in 0..hu {
        let y0 = (y as i64 - rf).max(0);
        let y1 = (y as i64 + rf).min(hu as i64 - 1);
        for x in 0..wu {
            let x0 = (x as i64 - rf).max(0);
            let x1 = (x as i64 + rf).min(wu as i64 - 1);
            let area = ((y1 - y0 + 1) * (x1 - x0 + 1)) as f64;
            let s = ii[(y1 as usize + 1) * iw + x1 as usize + 1]
                - ii[y0 as usize * iw + x1 as usize + 1]
                - ii[(y1 as usize + 1) * iw + x0 as usize]
                + ii[y0 as usize * iw + x0 as usize];
            out[y * wu + x] = (s / area) as f32;
        }
    }
    out
}

/// Компонента «чернил».
pub(crate) struct Component {
    /// Пиксельные индексы (для отладки/гистограмм).
    #[allow(dead_code)]
    pub pixels: Vec<usize>,
    pub count: u32,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

/// Связанные компоненты по маске (8-связность).
fn flood_components(mask: &[bool], w: u32, h: u32) -> Vec<Component> {
    let n = (w * h) as usize;
    let mut seen = vec![false; n];
    let mut comps = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..n {
        if seen[start] || !mask[start] {
            continue;
        }
        seen[start] = true;
        stack.clear();
        stack.push(start);
        let mut pixels = Vec::new();
        let (mut x0, mut y0, mut x1, mut y1) = (INF, INF, 0, 0);
        while let Some(p) = stack.pop() {
            pixels.push(p);
            let px = (p as u32) % w;
            let py = (p as u32) / w;
            x0 = x0.min(px);
            y0 = y0.min(py);
            x1 = x1.max(px);
            y1 = y1.max(py);
            let cx = px as i64;
            let cy = py as i64;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let ni = ((ny * w as i64) + nx) as usize;
                    if !seen[ni] && mask[ni] {
                        seen[ni] = true;
                        stack.push(ni);
                    }
                }
            }
        }
        if pixels.is_empty() {
            continue;
        }
        comps.push(Component {
            count: pixels.len() as u32,
            x0,
            y0,
            x1,
            y1,
            pixels,
        });
    }
    comps
}

/// Чернильная компонента похожа на штрих/букву.
fn plausible_stroke(c: &Component, w: u32, h: u32, max_char_h: u32) -> bool {
    let bw = c.x1 - c.x0 + 1;
    let bh = c.y1 - c.y0 + 1;
    // Микроточки-шум и объекты-гиганты отсекаются.
    if c.count < 4 || bh < 3 {
        return false;
    }
    if bh > max_char_h || bw > h * 2 {
        return false;
    }
    // Не должен занимать больше полкадра ни по одному измерению.
    if bw > w / 2 && bh > h / 2 {
        return false;
    }
    // Заполненность: кольца/рамки/дуги дают низкую плотность.
    if (c.count as f32 / (bw * bh) as f32) < 0.16 {
        return false;
    }
    // Одиночный пиксель-столбом — не штрих: пропорция не проверяем обратную,
    // подчёркивания и вертикали легитимны.
    let _ = w;
    true
}

fn union_find_link(boxes: &[[u32; 4]], overlap_frac: f32, gap_factor: f32) -> Vec<usize> {
    // Объединяет боксы в группы по критерию «одна линия»:
    // перекрытие по высоте ≥ overlap_frac·мин↑Высоты и промежуток
    // ≤ gap_factor·полусумма высот.
    let n = boxes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut Vec<usize>, i: usize) -> usize {
        let mut i = i;
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for a in 0..n {
        for b in a + 1..n {
            let (ax0, ay0, ax1, ay1) = (boxes[a][0], boxes[a][1], boxes[a][2], boxes[a][3]);
            let (bx0, by0, bx1, by1) = (boxes[b][0], boxes[b][1], boxes[b][2], boxes[b][3]);
            let ah = (ay1 - ay0 + 1) as f32;
            let bh = (by1 - by0 + 1) as f32;
            let ov_top = ay0.max(by0);
            let ov_bot = ay1.min(by1);
            let v_overlap = if ov_bot >= ov_top {
                (ov_bot - ov_top + 1) as f32
            } else {
                0.0
            };
            if v_overlap < overlap_frac * ah.min(bh) {
                continue;
            }
            let gap_x = if bx0 as i64 - ax1 as i64 > 0 {
                (bx0 - ax1) as f32
            } else if ax0 as i64 - bx1 as i64 > 0 {
                (ax0 - bx1) as f32
            } else {
                0.0
            };
            if gap_x <= gap_factor * 0.5 * (ah + bh) {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    parent
}

fn merge_groups(boxes: &[[u32; 4]], parent: &[usize]) -> Vec<([u32; 4], usize)> {
    fn root(parent: &[usize], mut i: usize) -> usize {
        while parent[i] != i {
            i = parent[i];
        }
        i
    }
    let mut groups: Vec<([u32; 4], usize)> = parent.iter().map(|_| ([INF, INF, 0, 0], 0usize)).collect();
    let mut root_list: Vec<usize> = Vec::new();
    for (i, b) in boxes.iter().enumerate() {
        let r = root(parent, i);
        let g = &mut groups[r];
        if g.1 == 0 {
            root_list.push(r);
        }
        g.0[0] = g.0[0].min(b[0]);
        g.0[1] = g.0[1].min(b[1]);
        g.0[2] = g.0[2].max(b[2]);
        g.0[3] = g.0[3].max(b[3]);
        g.1 += 1;
    }
    root_list.into_iter().map(|r| groups[r]).collect()
}

fn trace_enabled() -> bool {
    std::env::var("VC_TRACE").is_ok()
}

/// Полный проход детекции текста по кадру.
pub fn detect_text(frame: &Frame, opts: &DetectOptions) -> Vec<Detection> {
    let (l, w, h) = luma(frame);
    let max_char_h = if opts.max_char_h == 0 {
        (h as f32 * 0.35) as u32
    } else {
        opts.max_char_h
    };
    let radius = if opts.local_radius == 0 {
        (max_char_h * 2).max(12)
    } else {
        opts.local_radius
    };

    let mean = box_mean(&l, w, h, radius);
    let n = (w * h) as usize;
    let mut dark = vec![false; n];
    let mut light = vec![false; n];
    for i in 0..n {
        if mean[i] - l[i] >= opts.contrast {
            dark[i] = true;
        }
        if l[i] - mean[i] >= opts.contrast {
            light[i] = true;
        }
    }

    let comps = flood_components(&dark, w, h)
        .into_iter()
        .chain(flood_components(&light, w, h));
    let mut boxes: Vec<[u32; 4]> = Vec::new();
    let mut box_areas: Vec<u32> = Vec::new();
    for c in comps {
        if !plausible_stroke(&c, w, h, max_char_h) {
            if trace_enabled() {
                eprintln!(
                    "trace: отфильтрован штрих {:?}",
                    (c.x0, c.y0, c.x1, c.y1, c.count)
                );
            }
            continue;
        }
        boxes.push([c.x0, c.y0, c.x1, c.y1]);
        box_areas.push(c.count);
    }
    if trace_enabled() {
        eprintln!("trace: штрихов после фильтра={}", boxes.len());
    }
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut cands: Vec<[u32; 4]> = Vec::new();
    for vertical in [false, true] {
        let tmp: Vec<[u32; 4]> = boxes
            .iter()
            .map(|b| {
                let [x0, y0, x1, y1] = *b;
                if vertical {
                    [y0, x0, y1, x1]
                } else {
                    [x0, y0, x1, y1]
                }
            })
            .collect();
        let parent = union_find_link(&tmp, 0.4, opts.group_gap);
        if trace_enabled() {
            eprintln!(
                "trace: vertical={vertical} групп штрихов={}",
                merge_groups(&tmp, &parent).len()
            );
        }
        let _ = &box_areas; // площади используются ниже, в дедупликации
        for (b, ncomp) in merge_groups(&tmp, &parent) {
            if ncomp < 2 {
                if trace_enabled() {
                    eprintln!("trace: скип-синглтон {:?} v={vertical}", (b[0], b[1], b[2], b[3]));
                }
                continue; // одинокий штрих не слово
            }
            let (bw, bh) = (b[2].saturating_sub(b[0]) + 1, b[3].saturating_sub(b[1]) + 1);
            let (w_row, h_row) = if vertical { (bh, bw) } else { (bw, bh) };
            if h_row < opts.min_char_h || h_row > max_char_h {
                if trace_enabled() {
                    eprintln!("trace: скип-габариты {h_row}∈[{},{}]", opts.min_char_h, max_char_h);
                }
                continue;
            }
            if (w_row as f32 / h_row as f32) < 0.4 {
                continue;
            }
            if vertical {
                cands.push([b[1], b[0], b[3], b[2]]);
            } else {
                cands.push(b);
            }
        }
    }

    // Дедупликация пересекающихся боксов по IoU + подсчёт уверенности.
    cands.sort_by_key(|b| (b[0], b[1]));
    let mut out: Vec<Detection> = Vec::new();
    'outer: for b in cands {
        let mut ink_area = 0u32;
        for (i, [sx0, sy0, sx1, sy1]) in boxes.iter().enumerate() {
            if *sx0 >= b[0] && *sx1 <= b[2] && *sy0 >= b[1] && *sy1 <= b[3] {
                ink_area += box_areas[i];
            }
        }
        let bw = (b[2] - b[0] + 1) as f32;
        let bh = (b[3] - b[1] + 1) as f32;
        let area = bw * bh;
        // Чернила в боксе: доля площади под штрихами (буквы ~15–45%).
        let density = ink_area as f32 / area;
        if density < 0.025 {
            continue;
        }
        for d in &out {
            let iou = {
                let (ax0, ay0, ax1, ay1) = (d.x0 as i64, d.y0 as i64, d.x1 as i64, d.y1 as i64);
                let (bx0, by0, bx1, by1) = (b[0] as i64, b[1] as i64, b[2] as i64, b[3] as i64);
                let ix = (ax0.max(bx0), ax1.min(bx1));
                let iy = (ay0.max(by0), ay1.min(by1));
                let iw = (ix.1 - ix.0).max(0) as f32;
                let ih = (iy.1 - iy.0).max(0) as f32;
                let inter = iw * ih;
                let ua = ((ax1 - ax0) * (ay1 - ay0)) as f32;
                let ub = ((bx1 - bx0) * (by1 - by0)) as f32;
                inter / (ua + ub - inter + 1e-9)
            };
            if iou > 0.6 {
                continue 'outer;
            }
        }
        let conf = (density / 0.2).clamp(0.1, 1.0);
        out.push(Detection {
            x0: b[0],
            y0: b[1],
            x1: b[2],
            y1: b[3],
            confidence: conf,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_from(w: u32, h: u32, draw: impl Fn(u32, u32) -> [u8; 4]) -> Frame {
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let p = draw(x, y);
                let i = (y as usize * w as usize + x as usize) * 4;
                data[i..i + 4].copy_from_slice(&p);
            }
        }
        Frame::new(w, h, data).unwrap()
    }

    /// Четыре вертикальных тёмных штриха (псевдо-слово) на светлом фоне.
    fn word_frame() -> Frame {
        frame_from(64, 32, |x, y| {
            let bar = x % 8 < 2;
            let ink = bar && y >= 8 && y < 24;
            if ink {
                [20, 20, 20, 255]
            } else {
                [230, 230, 230, 255]
            }
        })
    }

    fn test_opts() -> DetectOptions {
        // Тестовые кадры мелкие (32 px высоты), авто-потолок 35% мал.
        DetectOptions { max_char_h: 24, ..Default::default() }
    }

    #[test]
    fn detects_word_bars() {
        let frame = word_frame();
        let dets = detect_text(&frame, &test_opts());
        assert!(!dets.is_empty(), "должны найти штрихи");
        // Суммарный bbox покрывает все штрихи (x 0..=31, y 8..=23).
        let x0 = dets.iter().map(|d| d.x0).min().unwrap();
        let x1 = dets.iter().map(|d| d.x1).max().unwrap();
        let y0 = dets.iter().map(|d| d.y0).min().unwrap();
        let y1 = dets.iter().map(|d| d.y1).max().unwrap();
        assert!(x0 <= 2, "x0={x0}");
        assert!(x1 >= 30, "x1={x1}");
        assert!(y0 <= 9 && y1 >= 22, "y0={y0} y1={y1}");
    }

    /// Светлый текст на тёмном фоне — вторая полярность.
    #[test]
    fn detects_light_on_dark() {
        let frame = frame_from(64, 32, |x, y| {
            let ink = x % 8 < 2 && y >= 8 && y < 24;
            if ink {
                [240, 240, 240, 255]
            } else {
                [30, 30, 30, 255]
            }
        });
        let dets = detect_text(&frame, &test_opts());
        assert!(!dets.is_empty(), "светлые штрихи должны найтись");
    }

    #[test]
    fn flat_frame_no_detection() {
        let frame = frame_from(64, 32, |_, _| [128, 128, 128, 255]);
        for opts in [
            DetectOptions::default(),
            DetectOptions {
                contrast: 10.0,
                ..Default::default()
            },
        ] {
            assert!(detect_text(&frame, &opts).is_empty());
        }
    }

    /// Крупный мягкий объект (затухающий круг) не должен попасть в текст.
    #[test]
    fn big_blob_not_detected() {
        let frame = frame_from(96, 96, |x, y| {
            let dx = (x as f32 - 48.0) / 44.0;
            let dy = (y as f32 - 48.0) / 44.0;
            let r = (dx * dx + dy * dy).sqrt();
            let v = (r.max(0.0).min(1.0) * 255.0) as u8;
            [v, v, v, 255]
        });
        let dets = detect_text(&frame, &DetectOptions::default());
        for d in dets {
            assert!(
                d.width() < 20 || d.height() < 20,
                "гигантский объект слипся в «текст»: {:?}",
                (d.x0, d.y0, d.x1, d.y1)
            );
        }
    }

    /// Два слова на одной строке: мелкие слова могут слиться — проверяем
    /// что детекция покрывает их, и главное — плоские зоны вокруг не дают мусора.
    #[test]
    fn two_words_single_line() {
        let mut ink_x = vec![];
        for base in [10u32, 40u32] {
            for k in 0..5u32 {
                ink_x.push(base + k * 3);
            }
        }
        let frame = frame_from(96, 32, |x, y| {
            let col = ink_x.contains(&x);
            let ink = col && y >= 10 && y < 22;
            if ink {
                [15, 15, 15, 255]
            } else {
                [235, 235, 235, 255]
            }
        });
        let dets = detect_text(&frame, &test_opts());
        assert!(!dets.is_empty());
        for d in &dets {
            assert!(d.height() <= 20, "высота детекции impractical: {d:?}");
        }
    }
}
