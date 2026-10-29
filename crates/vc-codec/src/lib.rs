//! vc-codec — работа с видео: метаданные, декодирование кадров.
//!
//! `Backend` — абстракция поверх внешнего декодера. Реализация по умолчанию —
//! системный FFmpeg CLI (только через subprocess-пайпы, без привязок к C).
//! Позже чисто-растовый декодер просто реализует тот же трейт.

pub mod export;
pub mod ffmpeg;
pub mod player;

pub use ffmpeg::{FfmpegBackend, FfmpegFrameReader};
pub use player::{FrameCache, Player};

use std::path::{Path, PathBuf};
use thiserror::Error;
use vc_core::{Frame, Timecode};

/// Метаданные видеофайла.
#[derive(Clone, Debug)]
pub struct VideoMetadata {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_secs: Option<f64>,
    pub frame_count: Option<u64>,
    pub has_audio: bool,
    pub codec: Option<String>,
}

impl VideoMetadata {
    pub fn default_timecode(&self) -> Timecode {
        Timecode::new(0, self.fps)
    }
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("файл не найден: {0}")]
    NotFound(PathBuf),
    #[error("не удалось найти ffmpeg/ffprobe в PATH")]
    ToolMissing,
    #[error("ffprobe: {0}")]
    Probe(String),
    #[error("ffmpeg: {0}")]
    Decode(String),
    #[error("некорректный кадр от декодера: {0}")]
    Frame(#[from] vc_core::FrameError),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// Поток кадров открытого файла (последовательное чтение).
pub trait FrameReader {
    /// Размер кадров в потоке (ширина, высота).
    fn size(&self) -> (u32, u32);
    /// Читает следующий кадр, либо `None` по EOF.
    fn read_frame(&mut self) -> Result<Option<Frame>, CodecError>;
}

/// Декодер. `seek` — позиция начала чтения, если бэкенд её поддерживает.
pub trait CodecBackend {
    fn probe(&self, path: &Path) -> Result<VideoMetadata, CodecError>;
    fn open(&self, path: &Path, seek_secs: Option<f64>)
        -> Result<Box<dyn FrameReader + Send>, CodecError>;
    /// Путь к CLI-кодировщику (для экспорта); по умолчанию — ffmpeg из PATH.
    fn ffmpeg_path(&self) -> std::path::PathBuf {
        std::path::PathBuf::from("ffmpeg")
    }
}
