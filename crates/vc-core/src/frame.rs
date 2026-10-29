//! Кадр видео: RGBA8-буфер фиксированного размера.
//!
//! RGBA выбран как внутренний формат: FFmpeg отдаёт его через rawvideo-пайп
//! без конверсии, egui принимает его напрямую, алгоритмы обработки не зависят
//! от чрома-субдискретизации.

/// RGBA8-кадр. `data.len() == width * height * 4`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub enum FrameError {
    /// Размер данных не соответствует width * height * 4.
    LengthMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::LengthMismatch { expected, got } => {
                write!(f, "размер буфера кадра неверен: ожидалось {expected} байт, получено {got}")
            }
        }
    }
}

impl std::error::Error for FrameError {}

impl Frame {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self, FrameError> {
        let expected = Self::byte_size_for(width, height);
        if data.len() != expected {
            return Err(FrameError::LengthMismatch { expected, got: data.len() });
        }
        Ok(Self { width, height, data })
    }

    /// Создаёт кадр, заполненный одним цветом (для тестов/заглушек).
    pub fn filled(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        let mut data = vec![0u8; Self::byte_size_for(width, height)];
        for px in data.chunks_exact_mut(4) {
            px.copy_from_slice(&rgba);
        }
        Self { width, height, data }
    }

    pub fn byte_size_for(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize) * 4
    }

    pub fn byte_size(&self) -> usize {
        Self::byte_size_for(self.width, self.height)
    }

    /// RGBA-вектор пикселя, или `Err` при выходе за границы.
    /// `x >= width || y >= height` → возвращает `None`.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let off = (y as usize * self.width as usize + x as usize) * 4;
        Some([self.data[off], self.data[off + 1], self.data[off + 2], self.data[off + 3]])
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filled_frame_matches_buffer_size() {
        let f = Frame::filled(3, 2, [1, 2, 3, 255]);
        assert_eq!(f.data.len(), 24);
        assert_eq!(f.pixel(2, 1), Some([1, 2, 3, 255]));
    }

    #[test]
    fn wrong_length_is_rejected() {
        let err = Frame::new(2, 2, vec![0u8; 15]).unwrap_err();
        assert!(matches!(err, FrameError::LengthMismatch { .. }));
    }
}
