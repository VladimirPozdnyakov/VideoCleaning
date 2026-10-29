//! vc-inpaint — ядро скрытия (инпейнтинга).
//!
//! Пайплайн (по плану):
//! 1. `background` — восстановление фона по соседним кадрам (быстро, для статики).
//! 2. `nn` — нейросетевой инпейнтинг через burn (основной режим).
//! 3. `telea` — классический алгоритм (fallback, уже реализован).
//!
//! M3: реализован Telea; API инпейнтинга стабилизирован так, чтобы остальные
//! крейты не зависели от деталей конкретного алгоритма.

pub mod background;
pub mod lama;
pub mod nn;
pub mod patchmatch;
pub mod pipeline;
pub mod telea;

/// Стратегии скрытия, которые выбирает пайплайн.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Восстановление по соседним кадрам (M4).
    Background,
    /// Нейросетевой инпейнтинг (основной, M5).
    Neural,
    /// Классический Telea.
    Telea,
}

/// Стратегия по умолчанию.
pub fn default_strategy() -> Strategy {
    Strategy::Background
}
