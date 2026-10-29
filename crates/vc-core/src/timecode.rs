//! Таймкод: индекс кадра плюс fps источника.
//!
//! Используется как единая «валюта времени» во всей программе:
//! таймлайн, маски (диапазоны кадров) и экспорт работают в кадрах,
//! а не в секундах, чтобы не копить погрешности при округлениях.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timecode {
    pub frame_index: u64,
    pub fps: f64,
}

impl Timecode {
    pub fn new(frame_index: u64, fps: f64) -> Self {
        Self { frame_index, fps: if fps > 0.0 { fps } else { 1.0 } }
    }

    /// Позиция кадра в секундах.
    pub fn seconds(&self) -> f64 {
        self.frame_index as f64 / self.fps
    }

    /// Секунды → индекс кадра (округление к ближайшему).
    pub fn from_seconds(seconds: f64, fps: f64) -> Self {
        let idx = (seconds.max(0.0) * fps).round() as u64;
        Self::new(idx, fps)
    }

    /// Следующий/предыдущий кадр (не уходит в минус).
    pub fn step(self, delta: i64) -> Self {
        let next = self.frame_index as i64 + delta;
        Self::new(next.max(0) as u64, self.fps)
    }

    /// Формат `HH:MM:SS::FF`.
    pub fn format(&self) -> String {
        let fps = self.fps.max(1.0);
        let total_frames = self.frame_index as f64;
        let whole_seconds = (total_frames / fps).floor() as u64;
        let frames = (total_frames - whole_seconds as f64 * fps).floor() as u64;
        let h = whole_seconds / 3600;
        let m = (whole_seconds % 3600) / 60;
        let s = whole_seconds % 60;
        format!("{h:02}:{m:02}:{s:02}::{frames:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_25fps() {
        let tc = Timecode::new(125, 25.0);
        assert_eq!(tc.format(), "00:00:05::00");
    }

    #[test]
    fn ntsc_framerate_no_drift() {
        let tc = Timecode::new(2997, 29.97);
        assert_eq!(tc.format(), "00:01:40::00");
    }
}
