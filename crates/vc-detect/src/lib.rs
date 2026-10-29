//! vc-detect — автоопределение текста (SWT + MSER + эвристики).

pub mod swt;

/// Порог уверенности детекции (0..=1).
pub const CONFIDENCE_THRESHOLD: f32 = 0.5;
