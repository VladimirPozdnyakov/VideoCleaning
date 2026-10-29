//! Левая панель: инструменты рисования масок.

use eframe::egui;
use egui::Button;

use crate::app::{Tool, VcApp};

const HELP_TEXT: &str = "ЛКМ по кадру — начать рисовать
Enter / дважды клик — завершить полигон
Esc / ПКМ — отменить рисование
Del — удалить выделенную маску
Ctrl+Z / Ctrl+Shift+Z — отмена / возврат
Space — играть, ←/→ — ±1 кадр
Shift+←/→ — ±25 кадров
Ручки у углов ректа — растянуть маску";

pub(crate) fn tool_panel(app: &mut VcApp, ctx: &egui::Context) {
    egui::SidePanel::left("vc_tools").resizable(false).show(ctx, |ui| {
        ui.add_space(6.0);
        ui.heading(egui::RichText::new("Инструменты").strong());
        ui.add_space(2.0);

        // Палитра инструментов: иконки с подсказками.
        let tools: [(&str, &str, Tool); 4] = [
            ("👆", "Осмотр: выбор, перетаскивание\nи растяжение масок", Tool::Pointer),
            ("⬛", "Прямоугольник — рисовать рамку", Tool::Rect),
            ("🎨", "Кисть — закрасить область", Tool::Brush),
            ("⬟", "Полигон — маска по вершинам", Tool::Polygon),
        ];
        egui::Grid::new("tool_grid")
            .num_columns(2)
            .spacing([6.0, 6.0])
            .show(ui, |ui| {
                for (icon, title, tool) in tools {
                    let selected = app.tool == tool;
                    let btn = Button::new(egui::RichText::new(icon).size(17.0))
                        .selected(selected)
                        .min_size(egui::vec2(44.0, 34.0));
                    let resp = ui.add(btn);
                    if resp.clicked() {
                        app.tool = tool;
                        app.drawing = None;
                    }
                    resp.on_hover_text(title);
                    ui.end_row();
                }
            });

        if app.tool == Tool::Brush {
            ui.add(
                egui::Slider::new(&mut app.brush_radius, 2.0..=120.0)
                    .text("радиус кисти"),
            );
        }

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(2.0);

        ui.checkbox(&mut app.show_masks, "Показывать маски");
        ui.checkbox(&mut app.preview_mode, "Превью эффекта");

        ui.add_space(2.0);
        ui.collapsing("Стратегия удаления", |ui| {
            for (s, label) in [
                (vc_inpaint::Strategy::Neural, "Умное удаление (ретушь + нейросеть)"),
                (vc_inpaint::Strategy::Background, "Фон (кадры-доноры)"),
                (vc_inpaint::Strategy::Telea, "Telea (быстрый)"),
            ] {
                if ui.add(egui::RadioButton::new(app.preview_strategy == s, label))
                    .clicked() && app.preview_strategy != s
                {
                    app.preview_strategy = s;
                    app.preview_src = None;
                    app.preview_rx = None;
                    app.preview_frame = None;
                    app.preview_info = None;
                }
            }
            if app.preview_strategy == vc_inpaint::Strategy::Neural {
                let (dot_color, msg) = if vc_inpaint::lama::available() {
                    (
                        egui::Color32::from_rgb(0x6E, 0xD9, 0x7E),
                        "LaMa готова (ONNX Runtime)",
                    )
                } else {
                    (
                        egui::Color32::from_rgb(0xE8, 0x6E, 0x6E),
                        "LaMa не найдена — патчи.\nФайл: models/lama-manga-dynamic.onnx",
                    )
                };
                ui.horizontal(|ui| {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 3.5, dot_color);
                    ui.small(msg.split('\n').next().unwrap_or(""));
                });
                ui.small(msg.split('\n').nth(1).unwrap_or(""));
            }
        });

        ui.add_space(2.0);
        ui.collapsing("Свойства масок", |ui| {
            let dilate_slider = ui
                .add(egui::Slider::new(&mut app.mask_dilate, 0..=12).text("расширение, px"));
            if dilate_slider.changed() {
                app.preview_src = None;
                app.preview_rx = None;
                app.preview_frame = None;
                app.preview_info = None;
            }
            if ui
                .add_enabled(
                    app.meta.is_some(),
                    Button::new("🔍 Найти текст на кадре"),
                )
                .clicked()
            {
                app.detect_text_now();
            }
            ui.small("добавит ректы-подсказки — правьте вручную");
        });

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add_enabled(!app.redo.is_empty(), Button::new("Вернуть").small())
                .clicked()
            {
                app.redo();
            }
            if ui
                .add_enabled(!app.undo.is_empty(), Button::new("Отменить").small())
                .clicked()
            {
                app.undo();
            }
        });

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);
        ui.collapsing("Клавиши", |ui| {
            ui.small(HELP_TEXT);
        });
        ui.add_space(6.0);
    });
}
