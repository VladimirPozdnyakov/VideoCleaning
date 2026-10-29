//! vc-core — доменные типы VideoCleaning: кадры, таймкоды, (позже) маски и таймлайны.

pub mod frame;
pub mod mask;
pub mod timecode;

pub use frame::{Frame, FrameError};
pub use mask::{rasterize, Mask, MaskId, MaskRegion, Shape, UNTIL_END};
pub use timecode::Timecode;
