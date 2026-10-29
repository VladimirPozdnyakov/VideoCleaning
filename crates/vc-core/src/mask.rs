//! Маски удаления: фигуры, диапазоны кадров и растеризация в битмап.
//!
//! Координаты фигур — в пикселях кадра (f32), их же хранит документ.
//! Для обработки растрируются в `Mask` — u8-битмап (0/255).

pub type MaskId = u64;

/// Маркер «до конца видео» для `end`.
pub const UNTIL_END: u64 = u64::MAX;

/// Фигура маски в пикселях кадра.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Shape {
    /// Прямоугольник по двум вершинам (порядок не важен).
    Rect { corners: [[f32; 2]; 2] },
    /// Полигон — замкнутая область из вершин (even-odd заливка).
    Polygon { vertices: Vec<[f32; 2]> },
    /// Кисть: последовательность точек штриха + радиус.
    Brush { points: Vec<[f32; 2]>, radius: f32 },
}

/// Одна маска: фигура + диапазон кадров [start, end] включительно.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MaskRegion {
    pub id: MaskId,
    pub shape: Shape,
    pub start: u64,
    pub end: u64,
}

impl MaskRegion {
    pub fn active_at(&self, frame: u64) -> bool {
        self.start <= frame && frame <= self.end
    }
}

/// Битмап-маска кадра: 0 — чисто, 255 — удалять.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub bits: Vec<u8>,
}

impl Mask {
    pub fn empty(width: u32, height: u32) -> Self {
        Self { width, height, bits: vec![0; (width as usize) * (height as usize)] }
    }

    pub fn value(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.bits[y as usize * self.width as usize + x as usize]
    }

    pub fn set(&mut self, x: u32, y: u32, v: u8) {
        if x < self.width && y < self.height {
            self.bits[y as usize * self.width as usize + x as usize] = v;
        }
    }

    /// Побитовый union (побайтовый максимум u8 0/255).
    pub fn union_with(&mut self, other: &Mask) {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a = (*a).max(*b);
        }
    }

    /// Пересечение: пиксель включён, если непуст в обеих масках.
    pub fn intersected(&self, other: &Mask) -> Mask {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        let mut bits = self.bits.clone();
        for (a, b) in bits.iter_mut().zip(other.bits.iter()) {
            *a = if *a != 0 && *b != 0 { 255 } else { 0 };
        }
        Mask { width: self.width, height: self.height, bits }
    }

    /// Есть ли хоть один непустой пиксель.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&b| b == 0)
    }

    /// Количество непустых пикселей.
    pub fn count(&self) -> u32 {
        self.bits.iter().filter(|&&b| b != 0).count() as u32
    }

    /// Морфологическая дилатация на `px` пикселей (квадратный элемент,
    /// сепарабельно: max по строке, затем по столбцу; сложность O(w·h·px)).
    pub fn dilated(&self, px: u32) -> Mask {
        if px == 0 {
            return self.clone();
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let r = px as usize;
        // Горизонтальный max с раздвижкой по обе стороны.
        let mut tmp = vec![0u8; w * h];
        for y in 0..h {
            let row = &self.bits[y * w..(y + 1) * w];
            let out = &mut tmp[y * w..(y + 1) * w];
            for x in 0..w {
                let lo = x.saturating_sub(r);
                let hi = (x + r + 1).min(w);
                out[x] = row[lo..hi].iter().copied().max().unwrap_or(0);
            }
        }
        // Вертикальный max.
        let mut out = vec![0u8; w * h];
        for x in 0..w {
            for y in 0..h {
                let lo = y.saturating_sub(r);
                let hi = (y + r + 1).min(h);
                let mut m = 0u8;
                for yy in lo..hi {
                    m = m.max(tmp[yy * w + x]);
                }
                out[y * w + x] = m;
            }
        }
        Mask { width: self.width, height: self.height, bits: out }
    }

    /// Пересечение маски с рамкой интереса (для отладки/тестов).
    pub fn bbox(&self) -> Option<((u32, u32), (u32, u32))> {
        let mut min = (u32::MAX, u32::MAX);
        let mut max = (0u32, 0u32);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.value(x, y) != 0 {
                    min = (min.0.min(x), min.1.min(y));
                    max = (max.0.max(x), max.1.max(y));
                }
            }
        }
        if min == (u32::MAX, u32::MAX) { None } else { Some((min, max)) }
    }
}

/// Растеризует все активные на `frame` маски в один битмап.
pub fn rasterize(regions: &[MaskRegion], frame: u64, width: u32, height: u32) -> Mask {
    let mut mask = Mask::empty(width, height);
    for region in regions.iter().filter(|r| r.active_at(frame)) {
        raster_shape(region.shape.clone(), width, height, &mut mask);
    }
    mask
}

fn raster_shape(shape: Shape, width: u32, height: u32, mask: &mut Mask) {
    match shape {
        Shape::Rect { corners } => {
            let [[x0, y0], [x1, y1]] = corners;
            let (x0, x1) = (x0.min(x1), x0.max(x1));
            let (y0, y1) = (y0.min(y1), y0.max(y1));
            fill_rect(RectI {
                x0: x0.max(0.0).floor() as i64,
                y0: y0.max(0.0).floor() as i64,
                x1: x1.min(width as f32).floor() as i64,
                y1: y1.min(height as f32).floor() as i64,
            }, width, mask);
        }
        Shape::Polygon { vertices } => {
            raster_polygon(&vertices, width, height, mask);
        }
        Shape::Brush { points, radius } => {
            for w in points.windows(2) {
                raster_segment(w[0], w[1], radius, width, height, mask);
            }
            // Отдельные точки или замыкание последнего отрезка.
            for &p in &points {
                raster_segment(p, p, radius, width, height, mask);
            }
        }
    }
}

/// Целочисленный прямоугольник для заливки.
#[derive(Clone, Copy, Debug)]
pub struct RectI {
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
}

fn fill_rect(r: RectI, width: u32, mask: &mut Mask) {
    for y in r.y0..=r.y1 {
        if y < 0 {
            continue;
        }
        for x in r.x0..=r.x1 {
            if x < 0 {
                continue;
            }
            mask.set(x as u32, y as u32, 255);
        }
    }
    let _ = width;
}

fn raster_polygon(vertices: &[[f32; 2]], width: u32, height: u32, mask: &mut Mask) {
    if vertices.len() < 3 {
        return;
    }
    let mut y_lo = f32::MAX;
    let mut y_hi = f32::MIN;
    for v in vertices {
        y_lo = y_lo.min(v[1]);
        y_hi = y_hi.max(v[1]);
    }
    y_lo = y_lo.max(0.0).floor();
    y_hi = y_hi.min(height as f32).floor();

    let n = vertices.len();
    for y in y_lo as u32..y_hi as u32 {
        let cy = y as f32 + 0.5;
        let mut xs: Vec<f32> = Vec::new();
        for i in 0..n {
            let a = vertices[i];
            let b = vertices[(i + 1) % n];
            if (a[1] <= cy && b[1] > cy) || (b[1] <= cy && a[1] > cy) {
                let t = (cy - a[1]) / (b[1] - a[1]);
                xs.push(a[0] + t * (b[0] - a[0]));
            }
        }
        xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        for pair in xs.chunks_exact(2) {
            let (xa, xb) = (pair[0].max(0.0).floor() as u32, pair[1].min(width as f32).ceil() as u32);
            for x in xa..xb.min(width) {
                mask.set(x, y, 255);
            }
        }
    }
}

fn raster_segment(p0: [f32; 2], p1: [f32; 2], radius: f32, width: u32, height: u32, mask: &mut Mask) {
    let r = radius.max(0.0);
    let (x_min, x_max) = (
        (p0[0].min(p1[0]) - r).max(0.0).floor() as u32,
        (p0[0].max(p1[0]) + r).min(width as f32).floor() as u32,
    );
    let (y_min, y_max) = (
        (p0[1].min(p1[1]) - r).max(0.0).floor() as u32,
        (p0[1].max(p1[1]) + r).min(height as f32).floor() as u32,
    );
    let r2 = r * r;
    let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let d2 = dx * dx + dy * dy;
    for y in y_min..=y_max.min(height.saturating_sub(1)) {
        for x in x_min..=x_max.min(width.saturating_sub(1)) {
            let cx = x as f32 + 0.5;
            let cy = y as f32 + 0.5;
            // Расстояние до отрезка.
            let t = if d2 > 0.0 {
                ((cx - p0[0]) * dx + (cy - p0[1]) * dy) / d2
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            let proj_x = p0[0] + t * dx;
            let proj_y = p0[1] + t * dy;
            let dist2 = (cx - proj_x) * (cx - proj_x) + (cy - proj_y) * (cy - proj_y);
            if dist2 <= r2 {
                mask.set(x, y, 255);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 12;
    const H: u32 = 12;

    /// Дилатация: одна точка → квадрат (2r+1)²; пустая маска не растёт.
    #[test]
    fn dilate_point_grows_square() {
        let mut m = Mask::empty(W, H);
        m.set(5, 5, 255);
        let d = m.dilated(2);
        assert_eq!(d.count(), 25, "точка должна стать квадратом 5×5");
        assert_eq!(d.value(3, 3), 255);
        assert_eq!(d.value(7, 7), 255);
        assert_eq!(d.value(2, 5), 0);
        // Границы кадра обрезаются.
        let mut e = Mask::empty(W, H);
        e.set(0, 0, 255);
        assert_eq!(e.dilated(3).count(), 16, "4×4 в углу, а не 7×7");
    }

    #[test]
    fn rect_mask_outline() {
        let region = MaskRegion {
            id: 1,
            shape: Shape::Rect { corners: [[2.0, 2.0], [7.0, 7.0]] },
            start: 0,
            end: UNTIL_END,
        };
        let mask = rasterize(&[region], 0, W, H);
        assert_eq!(mask.value(3, 3), 255);
        assert_eq!(mask.value(7, 7), 255);
        assert_eq!(mask.value(8, 8), 0);
        assert_eq!(mask.value(2, 1), 0);
    }

    #[test]
    fn brush_circle_and_segment() {
        let dot = MaskRegion {
            id: 1,
            shape: Shape::Brush { points: vec![[5.0, 5.0]], radius: 2.0 },
            start: 0,
            end: UNTIL_END,
        };
        let m = rasterize(&[dot.clone()], 0, W, H);
        assert_eq!(m.value(5, 5), 255);
        assert_eq!(m.value(7, 7), 0); // dist sqrt(8) > 2

        let stroke = MaskRegion {
            id: 2,
            shape: Shape::Brush { points: vec![[0.0, 5.0], [10.0, 5.0]], radius: 1.5 },
            start: 0,
            end: UNTIL_END,
        };
        let m = rasterize(&[stroke], 0, W, H);
        assert_eq!(m.value(5, 5), 255);
        assert_eq!(m.value(5, 7), 0); // вертикальное расстояние 2 > 1.5
    }

    #[test]
    fn polygon_triangle_even_odd() {
        let tri = MaskRegion {
            id: 1,
            shape: Shape::Polygon {
                vertices: vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
            },
            start: 0,
            end: UNTIL_END,
        };
        let m = rasterize(&[tri], 0, W, H);
        assert_eq!(m.value(1, 1), 255);
        assert_eq!(m.value(9, 9), 0); // вне гипотенузы x+y=10
        assert_eq!(m.value(10, 10), 0);
    }

    #[test]
    fn time_span_excludes_future() {
        let region = MaskRegion {
            id: 1,
            shape: Shape::Rect { corners: [[0.0, 0.0], [3.0, 3.0]] },
            start: 10,
            end: 20,
        };
        assert!(!region.active_at(5));
        assert!(region.active_at(10));
        assert!(region.active_at(20));
        assert!(!region.active_at(21));

        // Маска пустая вне диапазона.
        let m = rasterize(&[region], 0, W, H);
        assert!(m.is_empty());
    }

    #[test]
    fn union_and_count() {
        let mut a = Mask::empty(4, 4);
        let mut b = Mask::empty(4, 4);
        a.set(1, 1, 255);
        b.set(2, 2, 255);
        a.union_with(&b);
        assert_eq!(a.count(), 2);
        assert_eq!(a.value(1, 1), 255);
        assert_eq!(a.value(2, 2), 255);
    }
}
