//! Нижняя панель: таймлайн с маркерами диапазонов масок.

use eframe::egui;
use vc_core::Timecode;

use crate::app::{fps_of, region_color, total_frames, VcApp};

pub(crate) fn timeline_panel(app: &mut VcApp, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("vc_timeline").show(ctx, |ui| {
        ui.add_space(6.0);

        // Дорожка маркеров над слайдером.
        let Some(meta) = app.meta.clone() else {
            ui.label("таймлайн недоступен без видео");
            return;
        };
        let total = total_frames(&meta);
        let max_idx = total.saturating_sub(1);
        if max_idx == 0 {
            ui.label("видео из одного кадра");
            return;
        }
        let fps = fps_of(&meta);
        let duration = total as f64 / fps.max(1e-6) as f64;

        // --- Ряд управления: кнопки навигации + слайдер + таймкод. ---
        ui.horizontal(|ui| {
            if ui
                .add(egui::Button::new("‹‹").small())
                .on_hover_text("предыдущая граница диапазона масок")
                .clicked()
            {
                app.jump_mask_boundary(false);
            }
            if ui
                .add(egui::Button::new("«").small())
                .on_hover_text("−25 кадров (Shift+←)")
                .clicked()
            {
                app.step(-25);
            }
            if ui
                .add(egui::Button::new("<").small())
                .on_hover_text("−1 кадр (←)")
                .clicked()
            {
                app.step(-1);
            }
            let before = app.current;
            ui.add(
                egui::Slider::new(&mut app.current, 0..=max_idx)
                    .text("кадр")
                    .integer(),
            );
            if app.current != before {
                // Драг слайдера: пауза + явный seek.
                app.stop_playback();
                if let Some(p) = &app.player {
                    p.seek(app.current);
                }
            }
            if ui
                .add(egui::Button::new(">").small())
                .on_hover_text("+1 кадр (→)")
                .clicked()
            {
                app.step(1);
            }
            if ui
                .add(egui::Button::new("»").small())
                .on_hover_text("+25 кадров (Shift+→)")
                .clicked()
            {
                app.step(25);
            }
            if ui
                .add(egui::Button::new("››").small())
                .on_hover_text("следующая граница диапазона масок")
                .clicked()
            {
                app.jump_mask_boundary(true);
            }
            ui.label(
                egui::RichText::new(Timecode::new(app.current, fps).format()).monospace(),
            );
        });

        ui.add_space(4.0);

        // --- Дорожка маркеров масок (клик/скраб/колесо). ---
        let (track_rect, track_resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 14.0),
            egui::Sense::click_and_drag(),
        );
        // Рамка дорожки.
        ui.painter().rect_filled(
            track_rect,
            3.0,
            egui::Color32::from_rgb(0x24, 0x27, 0x2F),
        );
        ui.painter().rect_stroke(
            track_rect,
            3.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x2E, 0x33, 0x3E)),
            egui::StrokeKind::Inside,
        );

        // Секундные метки, адаптивный шаг.
        {
            let wdt = track_rect.width();
            let mut sec_step = 1.0f64;
            while (duration / sec_step) * (wdt as f64) < 150.0 {
                sec_step *= 2.0;
            }
            let mut major = 0i64;
            loop {
                let sec = major as f64 * sec_step;
                let frame = (sec * fps as f64).round();
                if sec > duration || frame >= total as f64 {
                    break;
                }
                let x = track_rect.left()
                    + (frame as f32 / (max_idx + 1) as f32) * wdt;
                let is_major = sec as i64 % 5 == 0;
                let tick_color = if is_major {
                    egui::Color32::from_rgb(0x3E, 0x44, 0x52)
                } else {
                    egui::Color32::from_rgb(0x2C, 0x30, 0x3A)
                };
                ui.painter().vline(
                    x,
                    track_rect.min.y + if is_major { 0.0 } else { 4.0 }..=track_rect.max.y,
                    egui::Stroke::new(1.0, tick_color),
                );
                if is_major {
                    let label = if sec_step >= 60.0 && (sec as i64) % 60 != 0 {
                        format!("{}м{}с", sec as i64 / 60, sec as i64 % 60)
                    } else if sec_step >= 60.0 {
                        format!("{}м", sec as i64 / 60)
                    } else {
                        format!("{}с", sec as i64)
                    };
                    ui.painter().text(
                        egui::pos2(x + 3.0, track_rect.min.y + 1.0),
                        egui::Align2::LEFT_TOP,
                        label,
                        egui::FontId::proportional(10.0),
                        egui::Color32::from_rgb(0x7A, 0x82, 0x94),
                    );
                }
                major += 1;
                if major > 2000 {
                    break;
                }
            }
        }

        // Диапазоны масок.
        for region in app.doc.regions.iter() {
            let start = region.start.min(max_idx) as f32;
            let end = if region.end == vc_core::UNTIL_END {
                max_idx as f32 + 1.0
            } else {
                (region.end.min(max_idx) + 1) as f32
            };
            let t = (end - start).max(1.0) / (max_idx + 1) as f32;
            let x0 = track_rect.left() + (start / (max_idx + 1) as f32) * track_rect.width();
            let w = t * track_rect.width();
            let sel = app.selection == Some(region.id);
            let c = region_color(region.id);
            let fill = if sel { c } else { c.gamma_multiply(0.55) };
            ui.painter().rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(x0, track_rect.min.y + 3.5),
                    egui::vec2(w, 7.0),
                ),
                2.0,
                fill,
            );
        }

        // Курсор текущего кадра.
        let cursor_x = |idx: u64| {
            track_rect.left()
                + (idx.min(max_idx) as f32 / (max_idx + 1) as f32) * track_rect.width()
        };
        ui.painter().vline(
            cursor_x(app.current),
            track_rect.min.y..=track_rect.max.y,
            egui::Stroke::new(1.8_f32, egui::Color32::from_rgb(0xF0, 0xF4, 0xFF)),
        );
        ui.painter().circle_filled(
            egui::pos2(cursor_x(app.current), track_rect.min.y),
            2.5,
            egui::Color32::from_rgb(0xF0, 0xF4, 0xFF),
        );

        // Взаимодействие: клик — прыжок, драг — скраб, колесо — шаг.
        let frame_at = |pos: egui::Pos2| -> u64 {
            let t = ((pos.x - track_rect.left()) / track_rect.width()).clamp(0.0, 0.9999);
            ((t * (max_idx + 1) as f32).floor() as u64).min(max_idx)
        };
        if track_resp.dragged() {
            if let Some(pos) = track_resp.interact_pointer_pos() {
                let idx = frame_at(pos);
                if idx != app.current {
                    app.stop_playback();
                    let delta = idx as i64 - app.current as i64;
                    // Скраб: сетевой seek на каждое смещение, без стоп-кадра.
                    app.goto(app.current.saturating_add_signed(delta).min(max_idx));
                }
            }
        } else if track_resp.clicked() {
            if let Some(pos) = track_resp.interact_pointer_pos() {
                let idx = frame_at(pos);
                app.stop_playback();
                app.goto(idx);
            }
        } else if track_resp.hovered() {
            // Подсказка под курсором.
            if let Some(pos) = track_resp.hover_pos() {
                let idx = frame_at(pos);
                let tc = Timecode::new(idx, fps).format();
                let x = cursor_x(idx);
                ui.painter().text(
                    egui::pos2(
                        (x - 26.0).max(track_rect.left()),
                        track_rect.min.y - 16.0,
                    ),
                    egui::Align2::LEFT_CENTER,
                    format!("{tc} · кадр {idx}"),
                    egui::FontId::proportional(12.0),
                    egui::Color32::from_rgb(0xD8, 0xE0, 0xEE),
                );
            }
            // Колесо: ±1 кадр, с Shift — ±25.
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll.abs() > 0.5 {
                let d = if scroll > 0.0 { -1 } else { 1 };
                let d = if ui.input(|i| i.modifiers.shift) { d * 25 } else { d };
                app.step(d);
            }
        }

        ui.add_space(6.0);
    });
}
