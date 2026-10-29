//! Пайплайн скрытия: выбор стратегии для кадра.
//!
//! Стратегии:
//! - `Background` (основная для 2D-анимации): фон из соседних кадров,
//!   остаток закрывает Telea.
//! - `Neural`: заготовка M5, пока сводится к Telea.
//! - `Telea`: чистый классический алгоритм.

use vc_core::{Frame, MaskRegion};

pub use crate::background::MAX_NEIGHBOR_SPAN;

use crate::Strategy;

/// Итог инпейнтинга с метаданными (для отчёта в UI).
pub struct InpaintResult {
    pub frame: Frame,
    pub strategy_used: Strategy,
    /// Сколько пикселей дыры фон не покрыл (дотянул Telea).
    pub uncovered_px: u32,
    /// Относительные индексы кадров, из которых собран фон.
    pub background_from: Vec<i64>,
    /// Чем именно дописаны пиксели без временных доказательств.
    pub filled_by: FillEngine,
}

/// Движок дозаполнения «спорных» пикселей (нет соседей-доказательств).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillEngine {
    /// Предобученная LaMa (AnimeMangaInpainting ONNX).
    Lama,
    /// Content-aware fill (PatchMatch).
    PatchMatch,
    /// Откат: Telea.
    Telea,
}

/// Инпейнтинг кадра по маскам с подгрузкой соседей через замыкание.
///
/// `lookup(offset)` должен вернуть кадр `frame_index + offset`, если он
/// доступен. Так вызывающий (кэш игрока) не отдаёт гигабайты клонов:
/// кадры клонируются лениво, только реально использованные.
/// `dilate_px` — расширение маски перед заливкой (закрывает антиалиасинг
/// и свечение текста за краем нарисованной фигуры).
/// `None` — маска пуста: кадр менять не нужно.
pub fn inpaint_frame_dynamic<F>(
    frame: &Frame,
    regions: &[MaskRegion],
    frame_index: u64,
    radius: u32,
    strategy: Strategy,
    lookup: F,
) -> Option<InpaintResult>
where
    F: FnMut(i64) -> Option<Frame>,
{
    inpaint_frame_dynamic_opts(
        frame,
        regions,
        frame_index,
        radius,
        strategy,
        0,
        lookup,
    )
}

/// Версия с расширением маски (`dilate_px` > 0 — морфологическая дилатация).
pub fn inpaint_frame_dynamic_opts<F>(
    frame: &Frame,
    regions: &[MaskRegion],
    frame_index: u64,
    radius: u32,
    strategy: Strategy,
    dilate_px: u32,
    lookup: F,
) -> Option<InpaintResult>
where
    F: FnMut(i64) -> Option<Frame>,
{
    let mask_raw = vc_core::rasterize(regions, frame_index, frame.width, frame.height);
    let mask = if dilate_px > 0 {
        mask_raw.dilated(dilate_px)
    } else {
        mask_raw
    };
    if mask.is_empty() {
        return None;
    }
    inpaint_masked(frame, &mask, regions, frame_index, radius, strategy, lookup)
}

/// Ядро: маска уже растеризована (и расширена).
fn inpaint_masked<F>(
    frame: &Frame,
    mask: &vc_core::Mask,
    regions: &[MaskRegion],
    frame_index: u64,
    radius: u32,
    strategy: Strategy,
    mut lookup: F,
) -> Option<InpaintResult>
where
    F: FnMut(i64) -> Option<Frame>,
{
    match strategy {
        Strategy::Background => {
            // M5c.1: ретушь — только пиксели, отличающиеся от выровненного
            // временного медианного фона; остальное остаётся исходным.
            match crate::background::retouch_frame(
                frame,
                &mask,
                regions,
                frame_index,
                &crate::background::RetouchConfig::default(),
                &mut lookup,
            ) {
                Some(rr) => {
                    let uncovered_px = rr.nn_hole.count();
                    let out = if uncovered_px == 0 {
                        rr.frame
                    } else {
                        crate::telea::telea_inpaint(&rr.frame, &rr.nn_hole, radius)
                    };
                    Some(InpaintResult {
                        frame: out,
                        strategy_used: Strategy::Background,
                        uncovered_px,
                        background_from: rr.used,
                        filled_by: FillEngine::Telea,
                    })
                }
                None => {
                    // Нет соседей — прежний путь (донор-филл + Telea).
                    let (filled, uncovered, used) = crate::background::fill_from_neighbors(
                        frame,
                        &mask,
                        regions,
                        frame_index,
                        MAX_NEIGHBOR_SPAN,
                        &mut lookup,
                    );
                    let uncovered_px = uncovered.count();
                    let out = if uncovered_px == 0 {
                        filled
                    } else {
                        crate::telea::telea_inpaint(&filled, &uncovered, radius)
                    };
                    Some(InpaintResult {
                        frame: out,
                        strategy_used: Strategy::Background,
                        uncovered_px,
                        background_from: used,
                        filled_by: FillEngine::Telea,
                    })
                }
            }
        }
        // M5c: нейростратегия — ретушь по кадрам + NN только на «спорных»
        // пикселях (без временных доказательств и похожих на штрихи текста).
        Strategy::Neural => {
            let cfg = crate::nn::NeuralConfig::default();
            // В юнит-тестах реальную предобученную модель не вызываем.
            #[cfg(test)]
            let cfg = crate::nn::NeuralConfig { use_lama: false, ..cfg };
            let retouch_cfg = crate::background::RetouchConfig {
                static_to_hole: true,
                ..Default::default()
            };
            match crate::background::retouch_frame(
                frame,
                &mask,
                regions,
                frame_index,
                &retouch_cfg,
                &mut lookup,
            ) {
                Some(rr) => {
                    // NN-дыра: при доступной LaMa — весь регион пользователя:
                    // LaMa перерисовывает дыру из окружения целиком
                    // (полупрозрачный текст на светлом фоне иначе оставлял
                    // недорисованные штрихи → каша). Без модели — прежняя
                    // узкая «спорная» маска (старые тесты).
                    let hole = if cfg.use_lama && crate::lama::available() {
                        mask.clone()
                    } else {
                        let ink = crate::background::ink_mask(frame, 48, 2);
                        rr.nn_hole.intersected(&ink)
                    };
                    if hole.count() > 0 {
                        // Порядок заимствован из VSR (video-subtitle-remover):
                        // для анимации предобученная LaMa даёт лучший результат.
                        // Fallback-цепочка: PatchMatch → Telea.
                        let mut engine = FillEngine::PatchMatch;
                        let lama_out = if cfg.use_lama {
                            match crate::lama::repaint(&rr.frame, &hole) {
                                Some(out) => {
                                    engine = FillEngine::Lama;
                                    Some(out)
                                }
                                None => None,
                            }
                        } else {
                            None
                        };
                        let painted = lama_out.or_else(|| {
                            Some(crate::patchmatch::fill(
                                &rr.frame,
                                &hole,
                                &crate::patchmatch::PatchMatchConfig::default(),
                            ))
                        });
                        match painted {
                            Some(out) => Some(InpaintResult {
                                frame: out,
                                strategy_used: Strategy::Neural,
                                uncovered_px: 0,
                                background_from: rr.used,
                                filled_by: engine,
                            }),
                            None => {
                                let out = crate::telea::telea_inpaint(&rr.frame, &hole, radius);
                                Some(InpaintResult {
                                    frame: out,
                                    strategy_used: Strategy::Neural,
                                    uncovered_px: 0,
                                    background_from: rr.used,
                                    filled_by: FillEngine::Telea,
                                })
                            }
                        }
                    } else {
                        Some(InpaintResult {
                            frame: rr.frame,
                            strategy_used: Strategy::Neural,
                            uncovered_px: 0,
                            background_from: rr.used,
                            filled_by: FillEngine::PatchMatch,
                        })
                    }
                }
                None => {
                    // Нет соседей: self-supervised NN (если влезает) → фон → Telea.
                    if crate::nn::hole_fits(&mask, frame, &cfg) {
                        match crate::nn::neural_inpaint(frame, &mask, &cfg, frame_index) {
                            Some(out) => Some(InpaintResult {
                                frame: out,
                                strategy_used: Strategy::Neural,
                                uncovered_px: 0,
                                background_from: Vec::new(),
                                filled_by: FillEngine::PatchMatch,
                            }),
                            None => {
                                let out = crate::telea::telea_inpaint(frame, &mask, radius);
                                Some(InpaintResult {
                                    frame: out,
                                    strategy_used: Strategy::Telea,
                                    uncovered_px: 0,
                                    background_from: Vec::new(),
                                    filled_by: FillEngine::Telea,
                                })
                            }
                        }
                    } else {
                        let (filled, uncovered, used) = crate::background::fill_from_neighbors(
                            frame,
                            &mask,
                            regions,
                            frame_index,
                            MAX_NEIGHBOR_SPAN,
                            &mut lookup,
                        );
                        let uncovered_px = uncovered.count();
                        let out = if uncovered_px == 0 {
                            filled
                        } else {
                            crate::telea::telea_inpaint(&filled, &uncovered, radius)
                        };
                        Some(InpaintResult {
                            frame: out,
                            strategy_used: Strategy::Background,
                            uncovered_px,
                            background_from: used,
                            filled_by: FillEngine::Telea,
                        })
                    }
                }
            }
        }
        Strategy::Telea => {
            let out = crate::telea::telea_inpaint(frame, &mask, radius);
            Some(InpaintResult {
                frame: out,
                strategy_used: Strategy::Telea,
                uncovered_px: 0,
                background_from: Vec::new(),
                filled_by: FillEngine::Telea,
            })
        }
    }
}

/// Упрощённый вызов без соседей (чистый Telea).
/// Оставлен для тестов и резервного пути без кэша.
pub fn inpaint_frame(
    frame: &Frame,
    regions: &[MaskRegion],
    frame_index: u64,
    radius: u32,
    strategy: Strategy,
) -> Option<Frame> {
    inpaint_frame_dynamic(frame, regions, frame_index, radius, strategy, |_| None)
        .map(|r| r.frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vc_core::{MaskRegion, Shape, UNTIL_END};

    #[test]
    fn no_regions_no_change_single() {
        let frame = Frame::filled(4, 4, [5, 5, 5, 255]);
        assert!(inpaint_frame(&frame, &[], 0, 3, Strategy::Telea).is_none());
    }

    #[test]
    fn single_frame_telea_still_works() {
        let frame = Frame::filled(9, 9, [40, 40, 40, 255]);
        let regions = vec![MaskRegion {
            id: 1,
            shape: Shape::Rect { corners: [[3.0, 3.0], [5.0, 5.0]] },
            start: 2,
            end: UNTIL_END,
        }];
        let out = inpaint_frame(&frame, &regions, 5, 4, Strategy::Telea).unwrap();
        assert_eq!(out.pixel(4, 4), Some([40, 40, 40, 255]));
    }

    /// Нейростратегия (M5c.1): «фотошопная» ретушь — заменяются ТОЛЬКО
    /// текстовые пиксели, целые байты фона сохраняются дословно.
    #[test]
    fn neural_keeps_background_bytes() {
        let bg = |x: u32| 100 + (x % 40) as u8;

        let frame = {
            let mut data = vec![0u8; 40 * 40 * 4];
            for y in 0..40u32 {
                for x in 0..40u32 {
                    let text = (20..24).contains(&x) && (20..24).contains(&y);
                    let v = if text { 250u8 } else { bg(x) };
                    let i = ((y * 40 + x) * 4) as usize;
                    data[i] = v;
                    data[i + 1] = v;
                    data[i + 2] = v;
                    data[i + 3] = 255;
                }
            }
            Frame::new(40, 40, data).unwrap()
        };
        let donors = vec![
            {
                let mut data = vec![0u8; 40 * 40 * 4];
                for y in 0..40u32 {
                    for x in 0..40u32 {
                        let text = (13..17).contains(&x) && (20..24).contains(&y);
                        let v = if text { 250u8 } else { bg(x) };
                        let i = ((y * 40 + x) * 4) as usize;
                        data[i] = v;
                        data[i + 1] = v;
                        data[i + 2] = v;
                        data[i + 3] = 255;
                    }
                }
                Frame::new(40, 40, data).unwrap()
            },
            {
                let mut data = vec![0u8; 40 * 40 * 4];
                for y in 0..40u32 {
                    for x in 0..40u32 {
                        let text = (27..31).contains(&x) && (20..24).contains(&y);
                        let v = if text { 250u8 } else { bg(x) };
                        let i = ((y * 40 + x) * 4) as usize;
                        data[i] = v;
                        data[i + 1] = v;
                        data[i + 2] = v;
                        data[i + 3] = 255;
                    }
                }
                Frame::new(40, 40, data).unwrap()
            },
            {
                let mut data = vec![0u8; 40 * 40 * 4];
                for y in 0..40u32 {
                    for x in 0..40u32 {
                        let v = bg(x);
                        let i = ((y * 40 + x) * 4) as usize;
                        data[i] = v;
                        data[i + 1] = v;
                        data[i + 2] = v;
                        data[i + 3] = 255;
                    }
                }
                Frame::new(40, 40, data).unwrap()
            },
        ];
        let regions = vec![MaskRegion {
            id: 1,
            shape: Shape::Rect { corners: [[12.0, 18.0], [25.0, 25.0]] },
            start: 0,
            end: UNTIL_END,
        }];
        let out = inpaint_frame_dynamic(&frame, &regions, 5, 0, Strategy::Neural, &mut |off: i64| {
            if off != 0 {
                Some(donors[(off.unsigned_abs() as usize).min(2)].clone())
            } else {
                None
            }
        })
        .map(|r| r.frame)
        .unwrap();
        // Текст заменён фоном кадра.
        assert_eq!(out.pixel(21, 21), Some([bg(21), bg(21), bg(21), 255]));
        // Целые пиксели фона сохранены байт-в-байт.
        assert_eq!(out.pixel(5, 5), frame.pixel(5, 5));
        assert_eq!(out.pixel(1, 38), frame.pixel(1, 38));
    }
}
