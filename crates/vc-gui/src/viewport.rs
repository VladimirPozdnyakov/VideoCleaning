//! Центральная панель: показ кадра, инструменты рисования и оверлей масок.

use eframe::egui;
use egui::{Color32, CursorIcon, Rect, Sense, Shape as EguiShape, Stroke, TextureOptions, Vec2};

use crate::app::{Drawing, INPAINT_RADIUS, PREFILL_BACKWARD_MAX, RectResize, Tool, VcApp};
use vc_core::{Frame, MaskRegion, Shape};

/// Прямоугольник кадра на экране по доступной области (с сохранением пропорций).
/// (оставлен как публичный хелпер для будущих панелей/превью)
#[allow(dead_code)]
pub(crate) fn frame_display_rect(available: egui::Vec2, w: u32, h: u32) -> egui::Rect {
    let size = egui::Vec2::new(w as f32, h as f32);
    let scale = (available.x / size.x).min(available.y / size.y).min(1.0);
    let show = size * scale;
    // Центрируем по доступной области.
    egui::Rect::from_min_size(
        egui::pos2(
            (available.x - show.x) * 0.5,
            (available.y - show.y) * 0.5,
        ),
        show,
    )
}

fn frame_to_screen(rect: Rect, w: u32, _h: u32, p: [f32; 2]) -> egui::Pos2 {
    let scale = rect.width() / w as f32;
    egui::pos2(rect.min.x + p[0] * scale, rect.min.y + p[1] * scale)
}

fn screen_to_frame(rect: Rect, w: u32, h: u32, pos: egui::Pos2) -> [f32; 2] {
    let scale = rect.width() / w as f32;
    let x = ((pos.x - rect.min.x) / scale).clamp(0.0, w as f32 - 1.0);
    let y = ((pos.y - rect.min.y) / scale).clamp(0.0, h as f32 - 1.0);
    [x, y]
}

/// Оверлей выбранной маски.
pub(crate) fn draw_region(
    painter: &egui::Painter,
    rect: Rect,
    w: u32,
    h: u32,
    region: &MaskRegion,
    selected: bool,
) {
    let base = crate::app::region_color(region.id);
    let (fill, stroke) = if selected {
        (
            Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 45),
            Stroke::new(3.0_f32, Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 255)),
        )
    } else {
        (
            Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 35),
            Stroke::new(2.0_f32, Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 220)),
        )
    };
    match &region.shape {
        Shape::Rect { corners } => {
            let a = frame_to_screen(rect, w, h, corners[0]);
            let b = frame_to_screen(rect, w, h, corners[1]);
            let r = egui::Rect::from_two_pos(a, b);
            painter.rect_filled(r, 0.0, fill);
            painter.rect_stroke(r, 0.0, stroke, egui::StrokeKind::Outside);
        }
        Shape::Polygon { vertices } => {
            if vertices.len() < 2 {
                return;
            }
            let pts: Vec<egui::Pos2> = vertices
                .iter()
                .map(|&v| frame_to_screen(rect, w, h, v))
                .collect();
            painter.add(EguiShape::convex_polygon(
                pts.clone(),
                fill,
                stroke,
            ));
        }
        Shape::Brush { points, radius } => {
            // Толщина — 2*радиус в пикселях кадра, масштабируем в экран.
            let width = (2.0 * radius.max(0.5)) * (rect.width() / w as f32);
            let stroke = Stroke::new(width.max(1.0), fill);
            if points.len() == 1 {
                let c = frame_to_screen(rect, w, h, points[0]);
                painter.circle_filled(c, (width / 2.0).max(1.0), fill);
            } else {
                for pair in points.windows(2) {
                    let a = frame_to_screen(rect, w, h, pair[0]);
                    let b = frame_to_screen(rect, w, h, pair[1]);
                    painter.line_segment([a, b], stroke);
                }
            }
        }
    }
}

fn draw_preview(app: &VcApp, painter: &egui::Painter, rect: Rect, w: u32, h: u32) {
    let Some(d) = &app.drawing else { return };
    let ghost = Stroke::new(2.0_f32, Color32::WHITE);
    let ghost_fill = Color32::from_rgba_unmultiplied(255, 255, 255, 60);
    match d.tool {
        Tool::Rect => {
            if let Some(&start) = d.points.first() {
                let a = frame_to_screen(rect, w, h, start);
                let b = frame_to_screen(rect, w, h, d.cur);
                let r = egui::Rect::from_two_pos(a, b);
                painter.rect_filled(r, 0.0, ghost_fill);
                painter.rect_stroke(r, 0.0, ghost, egui::StrokeKind::Outside);
            }
        }
        Tool::Brush => {
            let width = (2.0 * app.brush_radius.max(0.5)) * (rect.width() / w as f32);
            let s = Stroke::new(width.max(1.0), ghost_fill);
            if d.points.len() == 1 {
                painter.circle_filled(frame_to_screen(rect, w, h, d.points[0]), (width / 2.0).max(1.0), ghost_fill);
            }
            for pair in d.points.windows(2) {
                painter.line_segment(
                    [
                        frame_to_screen(rect, w, h, pair[0]),
                        frame_to_screen(rect, w, h, pair[1]),
                    ],
                    s,
                );
            }
            painter.circle_filled(frame_to_screen(rect, w, h, d.cur), (width / 2.0).max(1.0), ghost_fill);
        }
        Tool::Polygon => {
            if !d.points.is_empty() {
                let pts: Vec<egui::Pos2> =
                    d.points.iter().map(|&v| frame_to_screen(rect, w, h, v)).collect();
                for pair in pts.windows(2) {
                    painter.line_segment([pair[0], pair[1]], ghost);
                }
                painter.line_segment(
                    [pts[pts.len() - 1], frame_to_screen(rect, w, h, d.cur)],
                    Stroke::new(1.5_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 140)),
                );
                // Маркер первой точки: если курсор рядом — значит замыкание.
                let first = frame_to_screen(rect, w, h, d.points[0]);
                painter.circle_filled(first, 4.0, Color32::YELLOW);
            }
        }
        Tool::Pointer => {}
    }
}

impl VcApp {
    pub(crate) fn central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(meta) = self.meta.clone() else {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() * 0.45);
                    ui.label("Нет открытого видео");
                    ui.label("Нажмите «Открыть…» и выберите файл");
                });
                return;
            };
            let (fw, fh) = (meta.width, meta.height);

            let frame_opt = self.cache.lock().unwrap().get_cloned(self.current);
            // Кадр в пути от декодера: держим на экране последний показанный
            // (анти-мигание); «декодирование…» — только если показывать нечего.
            let have_frame = frame_opt.is_some();
            let frame = match frame_opt {
                Some(f) => f,
                None => match self.last_frame.clone() {
                    Some(last) => {
                        // Seek здесь НЕ шлём: goto/тики уже заказали его один
                        // раз, а повторные Seek убивают последовательное
                        // чтение декодера (переоткрытие ffmpeg → лайвлок).
                        ctx.request_repaint_after(std::time::Duration::from_millis(30));
                        last
                    }
                    None => {
                        ui.vertical_centered(|ui| {
                            ui.add_space(ui.available_height() * 0.45);
                            if self.player.is_some() {
                                ui.label("⏳ декодирование…");
                            } else {
                                ui.label("Нет открытого видео");
                            }
                        });
                        if let Some(p) = &self.player {
                            p.seek(self.current);
                        }
                        ctx.request_repaint_after(std::time::Duration::from_millis(30));
                        return;
                    }
                },
            };

            // Превью эффекта: при включённом режиме кадр заменяется на
            // результат инпейнтинга (пересчёт — при смене кадра/масок/кэша).
            // Во время воспроизведения НЕ считаем: ретушь+LaMa дороже 41мс
            // даже на GPU — расчёт на каждом кадре душит декодер и даёт
            // периодические застой; превью пересчитывается на паузе.
            if self.preview_mode && !self.playing {
                let rev = self.cache.lock().unwrap().revision();
                let pkey =
                    (self.current, self.doc_rev, rev, self.preview_strategy as u8);
                // Во время тяги ручки ресайза пересчёт не спамим: документ
                // меняется каждый кадр — по отпускании честно пересчитается.
                if self.preview_src != Some(pkey) && self.resize.is_none() {
                    self.compute_preview();
                    self.preview_src = Some(pkey);
                }
                // Достаём готовый фоновый расчёт (если есть и ключ ещё жив).
                if let Some(rx) = &self.preview_rx {
                    if let Ok((k0, k1, k2, k3, info, res_frame)) = rx.try_recv() {
                        let ok = (k0, k1, k2, k3) == pkey;
                        self.preview_rx = None;
                        self.preview_info = if info.is_empty() { None } else { Some(info) };
                        if ok {
                            self.preview_frame = Some(res_frame);
                        } else {
                            // Ключ устарел — честно пересчитать.
                            self.preview_src = None;
                        }
                    }
                }
            }
            let display_frame = if self.preview_mode && !self.playing {
                self.preview_frame.clone().unwrap_or(frame)
            } else {
                frame
            };
            // Запоминаем показанный кадр: им замещаем экран, пока декодер
            // догоняет (кэш пуст на скрабе/перемотке).
            self.last_frame = Some(display_frame.clone());

            // Ключ текстуры: кадр + ревизия документа (превью перезаливается при изменении масок).
            // Важно: при показе last_frame (кэш пуст) текстуру НЕ помечаем
            // как текущий кадр — иначе честный кадр, когда декодер его
            // отдаст, не перезалился бы и дёрганье сменилось бы заморозкой.
            let need_upload = self.tex.is_none()
                || (have_frame
                    && (self.tex_frame != Some(self.current) || self.tex_rev != self.doc_rev));
            if need_upload {
                let img = crate::app::to_color_image(&display_frame);
                match self.tex.take() {
                    Some(mut tex) => {
                        tex.set(img, TextureOptions::NEAREST);
                        self.tex = Some(tex);
                    }
                    None => {
                        self.tex = Some(ctx.load_texture("vc_frame", img, TextureOptions::NEAREST));
                    }
                }
                if have_frame {
                    // Честный кадр — запоминаем ключ. При stale ключ НЕ
                    // обновляем: честный кадр, когда декодер его отдаст,
                    // перезальёт текстуру (иначе — заморозка кадра).
                    self.tex_frame = Some(self.current);
                    self.tex_rev = self.doc_rev;
                }
            }
            let tex = self.tex.as_ref().unwrap();
            let available = ui.available_size();
            let src = tex.size_vec2();
            let scale = (available.x / src.x).min(available.y / src.y).min(1.0);
            let size = Vec2::new(src.x * scale, src.y * scale);
            let rect = egui::Rect::from_min_size(
                ui.min_rect().min + (available - size) * 0.5,
                size,
            );

            // Картинка кадра.
            ui.painter().image(
                tex.id(),
                rect,
                Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );

            // Подпись стратегии превью поверх ключа.
            if let (true, Some(info)) = (self.preview_mode, &self.preview_info) {
                ui.painter().text(
                    rect.left_top() + egui::vec2(8.0, 8.0),
                    egui::Align2::LEFT_TOP,
                    format!("⏵ {info}"),
                    egui::FontId::proportional(15.0),
                    Color32::YELLOW,
                );
            }

            // Оверлей масок.
            if self.show_masks {
                let painter = ui.painter().clone();
                for region in self.doc.regions.iter().filter(|r| r.active_at(self.current)) {
                    draw_region(
                        &painter,
                        rect,
                        fw,
                        fh,
                        region,
                        self.selection == Some(region.id),
                    );
                }
            }

            // Превью недорисованной фигуры.
            draw_preview(self, ui.painter(), rect, fw, fh);

            // Взаимодействие.
            let response = ui.allocate_rect(rect, Sense::click_and_drag());
            match self.tool {
                Tool::Pointer => {
                    // Ручки углов выбранного прямоугольника (изменение размера).
                    let handles: Vec<(egui::Pos2, bool, bool)> = {
                        let sel = self
                            .selection
                            .and_then(|id| {
                                self.doc
                                    .regions
                                    .iter()
                                    .find(|r| r.id == id && r.active_at(self.current))
                                    .map(|r| r.shape.clone())
                            });
                        match sel {
                            Some(Shape::Rect { corners }) => {
                                let (x0, x1) = (
                                    corners[0][0].min(corners[1][0]),
                                    corners[0][0].max(corners[1][0]),
                                );
                                let (y0, y1) = (
                                    corners[0][1].min(corners[1][1]),
                                    corners[0][1].max(corners[1][1]),
                                );
                                vec![
                                    (frame_to_screen(rect, fw, fh, [x0, y0]), true, true),
                                    (frame_to_screen(rect, fw, fh, [x1, y0]), false, true),
                                    (frame_to_screen(rect, fw, fh, [x1, y1]), false, false),
                                    (frame_to_screen(rect, fw, fh, [x0, y1]), true, false),
                                ]
                            }
                            _ => Vec::new(),
                        }
                    };
                    let painter = ui.painter();
                    for (pos, _, _) in &handles {
                        let r = egui::Rect::from_center_size(*pos, egui::vec2(9.0, 9.0));
                        painter.rect_filled(
                            r,
                            1.5,
                            Color32::from_rgba_unmultiplied(255, 255, 255, 230),
                        );
                        painter.rect_stroke(
                            r,
                            0.0,
                            Stroke::new(1.2_f32, Color32::BLACK),
                            egui::StrokeKind::Outside,
                        );
                    }
                    let handle_hit =
                        |pos: egui::Pos2| handles.iter().find(|(p, _, _)| p.distance(pos) < 10.0);
                    if self.resize.is_some() {
                        // Тянем ручку: форма меняется на лету, отдаём при отпускании.
                        if response.dragged() {
                            if let Some(pos) = response.interact_pointer_pos() {
                                let q = screen_to_frame(rect, fw, fh, pos);
                                self.resize_rect_to(q, fw, fh);
                            }
                        }
                        if response.drag_stopped() {
                            self.resize = None;
                        }
                    } else {
                        // Курсор-стрелка растяжения при наведении на ручку.
                        if let Some(pos) = response.hover_pos() {
                            if let Some((_, mx, my)) = handle_hit(pos) {
                                let icon = if *mx == *my {
                                    CursorIcon::ResizeNwSe
                                } else {
                                    CursorIcon::ResizeNeSw
                                };
                                ui.ctx().set_cursor_icon(icon);
                            }
                        }
                        if response.drag_started() {
                            if let Some(pos) = response.interact_pointer_pos() {
                                if let Some((_, mx, my)) = handle_hit(pos) {
                                    self.push_undo();
                                    self.resize = Some(RectResize {
                                        id: self.selection.unwrap_or(0),
                                        min_x: *mx,
                                        min_y: *my,
                                    });
                                    // Мгновенный отклик: ручка тянется в точку старта.
                                    self.resize_rect_to(
                                        screen_to_frame(rect, fw, fh, pos),
                                        fw,
                                        fh,
                                    );
                                } else {
                                    self.select_at(screen_to_frame(rect, fw, fh, pos));
                                }
                            }
                        }
                    }
                }
                Tool::Rect => {
                    if response.drag_started() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let fp = screen_to_frame(rect, fw, fh, pos);
                            self.drawing = Some(Drawing {
                                tool: Tool::Rect,
                                points: vec![fp],
                                cur: fp,
                            });
                        }
                    }
                    if let Some(d) = &mut self.drawing {
                        if d.tool == Tool::Rect && response.dragged() {
                            if let Some(pos) = response.interact_pointer_pos() {
                                d.cur = screen_to_frame(rect, fw, fh, pos);
                            }
                        }
                    }
                    if response.drag_stopped() {
                        let ok = match &self.drawing {
                            Some(d) => {
                                d.tool == Tool::Rect
                                    && (d.cur[0] - d.points[0][0]).abs()
                                        + (d.cur[1] - d.points[0][1]).abs()
                                        > 3.0
                            }
                            None => false,
                        };
                        if ok {
                            self.commit_drawing(true);
                        }
                    }
                }
                Tool::Brush => {
                    if response.drag_started() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let fp = screen_to_frame(rect, fw, fh, pos);
                            self.drawing = Some(Drawing {
                                tool: Tool::Brush,
                                points: vec![fp],
                                cur: fp,
                            });
                        }
                    }
                    if let Some(d) = &mut self.drawing {
                        if d.tool == Tool::Brush && response.dragged() {
                            if let Some(pos) = response.interact_pointer_pos() {
                                let fp = screen_to_frame(rect, fw, fh, pos);
                                // Добавляем точку только если она дальше ~1 кадрового px.
                                let last = d.points.last().copied().unwrap_or(fp);
                                let dx = fp[0] - last[0];
                                let dy = fp[1] - last[1];
                                if dx * dx + dy * dy > 1.0 {
                                    d.points.push(fp);
                                }
                            }
                        }
                    }
                    if response.drag_stopped() {
                        let ok = match &self.drawing {
                            Some(d) => d.tool == Tool::Brush && !d.points.is_empty(),
                            None => false,
                        };
                        if ok {
                            self.commit_drawing(true);
                        }
                    }
                }
                Tool::Polygon => {
                    if response.clicked() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let fp = screen_to_frame(rect, fw, fh, pos);
                            let mut finish = false;
                            match &mut self.drawing {
                                None => {
                                    self.drawing = Some(Drawing {
                                        tool: Tool::Polygon,
                                        points: vec![fp],
                                        cur: fp,
                                    });
                                }
                                Some(d) => {
                                    // Клик по первой точке замыкает (если ≥3 вершин).
                                    let first = d.points[0];
                                    let dx = fp[0] - first[0];
                                    let dy = fp[1] - first[1];
                                    if d.points.len() >= 3 && dx * dx + dy * dy < 64.0 {
                                        finish = true;
                                    } else {
                                        d.points.push(fp);
                                    }
                                }
                            }
                            if finish {
                                self.commit_drawing(true);
                            }
                        }
                    }
                    if response.double_clicked() {
                        let ok = match &self.drawing {
                            Some(d) => d.tool == Tool::Polygon && d.points.len() >= 3,
                            None => false,
                        };
                        if ok {
                            self.commit_drawing(true);
                        }
                    }
                    if response.secondary_clicked() {
                        // ПКМ — отмена полигона.
                        self.drawing = None;
                    }
                }
            }

            // Курсор зависит от инструмента (и от активной тяги ручки).
            let cursor = if let Some(rz) = self.resize {
                if rz.min_x == rz.min_y {
                    CursorIcon::ResizeNwSe
                } else {
                    CursorIcon::ResizeNeSw
                }
            } else {
                match self.tool {
                    Tool::Pointer => CursorIcon::Default,
                    Tool::Rect | Tool::Brush | Tool::Polygon => CursorIcon::Crosshair,
                }
            };
            ui.ctx().set_cursor_icon(cursor);

            self.display_rect = Some(rect);
        });
    }

    /// Сборка превью: подтягиваем соседей из кэша, заказываем backward-prefill,
    /// вызываем пайплайн Background/Telea и запоминаем подпись стратегии.
    pub(crate) fn compute_preview(&mut self) {
        // На playback пересчёт не имеет смысла (дороже кадра) и вреден:
        // prefill_backward переоткрывает декодер, задушая playback.
        if self.playing {
            return;
        }
        self.preview_frame = None;
        self.preview_info = Some("расчёт превью…".to_string());

        // Кадр-донор нужен всегда.
        let Some(frame) = self.cache.lock().unwrap().get_cloned(self.current) else {
            return;
        };

        // Ближайшее начало активных масок: чтобы подтянуть кадры до него.
        if let Some(min_start) = self
            .doc
            .regions
            .iter()
            .filter(|r| r.active_at(self.current))
            .map(|r| r.start)
            .min()
        {
            if let Some(p) = &self.player {
                let count = ((self.current - min_start) as u32).min(PREFILL_BACKWARD_MAX);
                if count > 0 && self.last_prefill != Some((min_start, count)) {
                    p.prefill_backward(self.current, count);
                    self.last_prefill = Some((min_start, count));
                }
            }
        }

        let cache = std::sync::Arc::clone(&self.cache);
        let current = self.current;
        let regions = self.doc.regions.clone();
        let lookup = move |off: i64| -> Option<Frame> {
            let idx = (current as i64 + off).max(0) as u64;
            cache.lock().unwrap().get_cloned(idx)
        };
        let strategy = self.preview_strategy;
        let rev = self.cache.lock().unwrap().revision();
        let key = (self.current, self.doc_rev, rev, strategy as u8);
        let mask_dilate = self.mask_dilate;

        // Долгие стратегии (LaMa/NN) рисуем в фоне: UI не блокируем.
        let (tx, rx) = std::sync::mpsc::channel();
        self.preview_rx = Some(rx);
        std::thread::Builder::new()
            .name("vc-preview".into())
            .spawn(move || {
                let result = vc_inpaint::pipeline::inpaint_frame_dynamic_opts(
                    &frame,
                    &regions,
                    current,
                    INPAINT_RADIUS,
                    strategy,
                    mask_dilate,
                    lookup,
                );
                let payload = match result {
                    Some(res) => (key.0, key.1, key.2, key.3, describe_result(strategy, &res), res.frame),
                    None => (key.0, key.1, key.2, key.3, String::new(), {
                        // Нечего показывать — кадр не меняется.
                        frame.clone()
                    }),
                };
                let _ = tx.send(payload);
            })
            .expect("поток превью");
    }
}

/// Человекочитаемый итог стратегии.
fn describe_result(wanted: vc_inpaint::Strategy, res: &vc_inpaint::pipeline::InpaintResult) -> String {
    match res.strategy_used {
        vc_inpaint::Strategy::Background => {
            let used_str: Vec<String> =
                res.background_from.iter().map(|o| o.to_string()).collect();
            let from = used_str.join(", ");
            let unpx = res.uncovered_px;
            if unpx == 0 {
                format!("ретушь: фон из кадров [{from}]")
            } else {
                format!("ретушь: фон [{from}] + Telea ({unpx}px)")
            }
        }
        vc_inpaint::Strategy::Neural => {
            let from = res
                .background_from
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let fill = match res.filled_by {
                vc_inpaint::pipeline::FillEngine::Lama => "LaMa (предобученная модель)",
                vc_inpaint::pipeline::FillEngine::PatchMatch => "патчи (content-aware fill)",
                vc_inpaint::pipeline::FillEngine::Telea => "Telea",
            };
            if res.background_from.is_empty() {
                format!("умное удаление: {fill}")
            } else {
                format!("умное удаление: ретушь по кадрам [{from}] + {fill}")
            }
        }
        _ => {
            if wanted == vc_inpaint::Strategy::Neural {
                format!("ретушь недоступна → Telea (радиус {INPAINT_RADIUS})")
            } else {
                format!("Telea (радиус {INPAINT_RADIUS})")
            }
        }
    }
}
