//! Правая панель: список масок и их свойства (диапазон кадров).

use eframe::egui;
use egui::Button;
use vc_core::{MaskId, MaskRegion, Shape, UNTIL_END};

use crate::app::{region_color, shape_summary, total_frames, VcApp};

/// Редактируемое беззнаковое число с ограничениями (DragValue по i64).
/// Возвращает новое значение, если оно изменилось; эффекты — у вызывающего.
fn drag_bare(ui: &mut egui::Ui, value: &mut u64, min: u64, max: u64) -> Option<u64> {
    let before = *value;
    let mut v = before.min(i64::MAX as u64) as i64;
    let res = ui.add(
        egui::DragValue::new(&mut v)
            .range(min as i64..=max as i64)
            .speed(0.2),
    );
    res.changed().then(|| {
        *value = v.clamp(min as i64, max as i64) as u64;
        *value
    })
}

/// Применяет новую пару (start, end) с корректной записью истории:
/// откат к старым значениям → undo-снапшот → новые значения → инвалидация превью.
fn apply_range_edit(app: &mut VcApp, id: MaskId, old: (u64, u64), new: (u64, u64)) {
    if old == new {
        return;
    }
    {
        let r = app.doc.regions.iter_mut().find(|r| r.id == id).expect("region exists");
        r.start = old.0;
        r.end = old.1;
    }
    app.push_undo();
    let r = app.doc.regions.iter_mut().find(|r| r.id == id).expect("region exists");
    r.start = new.0;
    r.end = new.1;
}

pub(crate) fn masks_panel(app: &mut VcApp, ctx: &egui::Context) {
    egui::SidePanel::right("vc_masks").show(ctx, |ui| {
        ui.add_space(6.0);
        ui.heading(egui::RichText::new("Маски").strong());
        ui.add_space(4.0);

        if app.doc.regions.is_empty() {
            ui.label(
                egui::RichText::new(
                    "Масок пока нет.\nВозьмите инструмент слева и нарисуйте на кадре.",
                )
                .weak(),
            );
        }

        let ids: Vec<_> = app.doc.regions.iter().map(|r| r.id).collect();
        for &id in &ids {
            // Копия региона на момент отрисовки строки (старые значения для undo).
            let Some(region) = app.doc.regions.iter().find(|r| r.id == id).cloned() else {
                continue;
            };
            let selected = app.selection == Some(id);
            let summary = shape_summary(&region.shape);
            let c = region_color(id);

            let (card_rect, card_resp) = ui
                .allocate_exact_size(
                    egui::vec2(ui.available_width(), if selected { 132.0 } else { 42.0 }),
                    egui::Sense::click(),
                );
            // Карточка: фон + цветовой маркер слева.
            ui.painter().rect_filled(
                card_rect,
                6.0,
                if selected {
                    egui::Color32::from_rgb(0x2A, 0x30, 0x3E)
                } else {
                    egui::Color32::from_rgb(0x20, 0x23, 0x2B)
                },
            );
            ui.painter().rect_filled(
                egui::Rect::from_min_size(card_rect.min, egui::vec2(4.0, card_rect.height()))
                    .shrink(0.0),
                2.0,
                if selected { c } else { c.gamma_multiply(0.6) },
            );
            if card_resp.clicked() {
                app.selection = if selected { None } else { Some(id) };
            }
            ui.allocate_ui_at_rect(card_rect, |ui| {
                // Верхняя строка карточки.
                ui.set_clip_rect(card_rect);
                ui.horizontal(|ui| {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(format!("Маска {id} · {summary}")).size(13.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(Button::new("🗑").small())
                            .on_hover_text("Удалить маску (Del)")
                            .clicked()
                        {
                            app.push_undo();
                            app.doc.remove(id);
                            app.selection = None;
                        }
                    });
                });
                if !selected {
                    return;
                }
                ui.add_space(6.0);
                let total = app.meta.as_ref().map_or(1, total_frames);
                ui.label(egui::RichText::new(range_text(&region, total)).weak().size(12.0));
                ui.add_space(2.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label("от");
                    let mut rs = region.start;
                    let new_start = drag_bare(ui, &mut rs, 0, total.saturating_sub(1));
                    if let Some(v) = new_start {
                        apply_range_edit(app, id, (region.start, region.end), (v, region.end));
                    }
                    ui.label("до");
                    let is_inf = region.end == UNTIL_END;
                    if is_inf {
                                ui.label("∞");
                                if ui.small_button("кадры: от текущего").clicked() {
                                    apply_range_edit(
                                        app,
                                        id,
                                        (region.start, region.end),
                                        (region.start, app.current.min(total.saturating_sub(1))),
                                    );
                                }
                                if ui.small_button("только этот кадр").clicked() {
                                    apply_range_edit(
                                        app,
                                        id,
                                        (region.start, region.end),
                                        (app.current, app.current),
                                    );
                                }
                            } else {
                                let mut re = region.end;
                                let new_end = drag_bare(ui, &mut re, region.start, total.saturating_sub(1));
                                if let Some(v) = new_end {
                                    apply_range_edit(app, id, (region.start, region.end), (region.start, v));
                                }
                                if ui.small_button("до конца видео").clicked() {
                                    apply_range_edit(
                                        app,
                                        id,
                                        (region.start, region.end),
                                        (region.start, UNTIL_END),
                                    );
                                }
                            }
                });
            });
            ui.add_space(6.0);
        }
    });
}

fn range_text(region: &MaskRegion, total: u64) -> String {
    let end = if region.end == UNTIL_END {
        "∞".to_string()
    } else if region.end + 1 >= total {
        format!("{} (конец)", total - 1)
    } else {
        format!("{}", region.end)
    };
    format!("кадры {start} … {end}", start = region.start)
}

#[allow(dead_code)]
fn unused_shape(_s: &Shape) {
    let _ = 0u64 as MaskId;
}
