//! Главное окно VideoCleaning (M2: маски на кадре, undo/redo).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, Rect, TextureHandle};
use vc_codec::{CodecBackend, FfmpegBackend, FrameCache, Player, VideoMetadata};
use vc_core::{Frame, MaskId, MaskRegion, Shape, Timecode, UNTIL_END};

use crate::masks_panel::masks_panel;
use crate::timeline::timeline_panel;
use crate::tool_panel::tool_panel;

/// Сообщение от рабочего потока экспорта в GUI.
pub(crate) enum ExportMsg {
    /// (готово, всего) — доля полосы.
    Progress(u64, u64),
    /// (готово, всего) — счётчик кадров текущего этапа.
    Frames(u64, u64),
    /// Подпись этапа (сайдкар видео-модели).
    Phase(String),
    /// Готово: (кадров всего, кадров обработано, отменено).
    Done(u64, u64, bool),
    /// Ошибка.
    Failed(String),
}

/// Управление запущенным экспортом.
pub(crate) struct ExportHandle {
    pub(crate) rx: Receiver<ExportMsg>,
    pub(crate) cancel: Arc<AtomicBool>,
}

/// Лимит кэша по байтам (≈ 60 кадров 1080p RGBA).
pub(crate) const CACHE_CAP: usize = 512 * 1024 * 1024;
/// Глубина истории изменений масок.
pub(crate) const UNDO_LIMIT: usize = 100;
/// Радиус ядра классического инпейнтинга (Telea) для превью.
pub(crate) const INPAINT_RADIUS: u32 = 5;
/// Максимальный backward-prefill (кадров до начала маски, для фона).
pub(crate) const PREFILL_BACKWARD_MAX: u32 = 240;

// ---------------------------------------------------------------------------
// Инструменты и документ
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Tool {
    #[default]
    Pointer,
    Rect,
    Brush,
    Polygon,
}

/// Документ: все маски + счётчик id.
#[derive(Clone, Debug, Default)]
pub(crate) struct Doc {
    pub regions: Vec<MaskRegion>,
    pub next_id: MaskId,
}

impl Doc {
    /// Добавляет маску, возвращает её id.
    pub(crate) fn add(&mut self, shape: Shape, start: u64, end: u64) -> MaskId {
        let id = self.next_id;
        self.next_id += 1;
        self.regions.push(MaskRegion { id, shape, start, end });
        id
    }

    pub(crate) fn remove(&mut self, id: MaskId) {
        self.regions.retain(|r| r.id != id);
    }
}

/// Ещё недорисованная фигура (координаты — в пикселях кадра).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Drawing {
    pub tool: Tool,
    /// Rect: [start, cur]; Brush: точки штриха; Polygon: вершины.
    pub points: Vec<[f32; 2]>,
    /// Текущее положение указателя (не учитывается для Brush, там фиксирован points).
    pub cur: [f32; 2],
}

/// Изменение размера готового прямоугольника: какая угловая ручка тянется
/// (сторона по каждой оси прижимается к минимуму или максимуму).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RectResize {
    pub id: MaskId,
    /// Ручка стоит на минимуме по X (иначе — на максимуме).
    pub min_x: bool,
    /// Ручка стоит на минимуме по Y (иначе — на максимуме).
    pub min_y: bool,
}

// ---------------------------------------------------------------------------
// Приложение
// ---------------------------------------------------------------------------

pub struct VcApp {
    pub(crate) backend: FfmpegBackend,
    pub(crate) player: Option<Player>,
    pub(crate) cache: Arc<Mutex<FrameCache>>,
    pub(crate) meta: Option<VideoMetadata>,
    /// Текущий показываемый кадр.
    pub(crate) current: u64,
    /// Воспроизведение: таймер-тикер; кадры подтягивает фоновый декодер.
    pub(crate) playing: bool,
    pub(crate) next_tick: Option<Instant>,
    /// С какого момента декодер не поставляет следующий кадр при игре
    /// (застой: тики ждут кэш, чтобы не убегать от декодера seek'ами).
    pub(crate) stall_since: Option<Instant>,
    /// Текстура последнего показанного кадра.
    pub(crate) tex: Option<TextureHandle>,
    pub(crate) tex_frame: Option<u64>,
    /// Ревизия документа, при которой построена текстура (для превью).
    pub(crate) tex_rev: u64,
    /// Последний показанный кадр: держим на экране, пока декодер
    /// догоняет текущий (анти-мигание «декодирование…»).
    pub(crate) last_frame: Option<Frame>,
    /// Документ масок.
    pub(crate) doc: Doc,
    pub(crate) selection: Option<MaskId>,
    pub(crate) tool: Tool,
    pub(crate) drawing: Option<Drawing>,
    /// Активное изменение размера выбранного прямоугольника (ручка угла).
    pub(crate) resize: Option<RectResize>,
    pub(crate) brush_radius: f32,
    pub(crate) show_masks: bool,
    /// Экранный прямоугольник показанного кадра (для маппинга координат).
    pub(crate) display_rect: Option<Rect>,
    pub(crate) undo: Vec<Doc>,
    pub(crate) redo: Vec<Doc>,
    /// Ревизия документа — инвалидирует превью при любом изменении масок.
    pub(crate) doc_rev: u64,
    /// Режим превью: показывать кадр с инпейнтингом вместо оригинала.
    pub(crate) preview_mode: bool,
    /// Стратегия превью (M5): фон / нейросеть / Telea.
    pub(crate) preview_strategy: vc_inpaint::Strategy,
    pub(crate) preview_frame: Option<Frame>,
    /// (кадр, ревизия, ревизия кэша) для которого построено `preview_frame`.
    pub(crate) preview_src: Option<(u64, u64, u64, u8)>,
    /// Подпись применённой стратегии превью.
    pub(crate) preview_info: Option<String>,
    /// Фоновый расчёт превью: приёмник (ключ, подпись, кадр).
    pub(crate) preview_rx:
        Option<std::sync::mpsc::Receiver<(u64, u64, u64, u8, String, Frame)>>,
    /// Последний отправленный backward-prefill (антидребезг).
    pub(crate) last_prefill: Option<(u64, u32)>,
    pub(crate) status: String,
    pub(crate) error: Option<String>,

    // --- M7: экспорт и проект -------------------------------------------------
    /// Открыт ли диалог экспорта.
    pub(crate) export_dialog: bool,
    /// Путь назначения (в текстовом поле диалога).
    pub(crate) export_out: String,
    /// Качество x264 из диалога.
    pub(crate) export_crf: u32,
    /// Запущенный экспорт (канал для прогресса/итога + флаг отмены).
    pub(crate) export_handle: Option<ExportHandle>,
    /// Отменено ли текущее экспортное задание.
    pub(crate) export_cancelled: bool,
    /// Успешно ли экспорт завершился (для финального статуса).
    pub(crate) export_failed_msg: Option<String>,
    /// Прогресс: готово кадров.
    pub(crate) export_done: u64,
    /// Прогресс: всего кадров (оценка).
    pub(crate) export_total: u64,
    /// Подпись текущего этапа экспорта (для сайдкара видео-модели).
    pub(crate) export_phase: String,
    /// Счётчик кадров этапа: (готово, всего).
    pub(crate) export_frames: (u64, u64),
    /// Момент старта экспорта (для часов/оценки остатка).
    pub(crate) export_started: Option<std::time::Instant>,
    /// Экспортировать видео-моделью (ProPainter, требуется tools/venv).
    pub(crate) export_use_video_model: bool,
    /// Расширение маски в px (закрывает свечение/антиалиасинг текста).
    pub(crate) mask_dilate: u32,
}

impl Default for VcApp {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn total_frames(meta: &VideoMetadata) -> u64 {
    meta.frame_count
        .or_else(|| meta.duration_secs.map(|d| (d * meta.fps).round() as u64))
        .unwrap_or(1)
        .max(1)
}

pub(crate) fn fps_of(meta: &VideoMetadata) -> f64 {
    meta.fps.max(1.0)
}

/// Палитра масок (по id — стабильный цвет).
pub(crate) fn region_color(id: MaskId) -> Color32 {
    const PALETTE: [Color32; 5] = [
        Color32::from_rgb(230, 60, 60),
        Color32::from_rgb(60, 190, 80),
        Color32::from_rgb(70, 130, 240),
        Color32::from_rgb(230, 190, 50),
        Color32::from_rgb(190, 80, 220),
    ];
    PALETTE[(id as usize) % PALETTE.len()]
}

/// Человекочитаемый краткий итог фигуры.
pub(crate) fn shape_summary(shape: &Shape) -> String {
    match shape {
        Shape::Rect { corners } => {
            let [[x0, y0], [x1, y1]] = corners;
            format!("прямоуг ({:.0},{:.0})–({:.0},{:.0})", x0, y0, x1, y1)
        }
        Shape::Polygon { vertices } => format!("полигон, {} верш.", vertices.len()),
        Shape::Brush { points, radius } => {
            format!("кисть, {} т., r={:.0}", points.len(), radius)
        }
    }
}

impl VcApp {
    pub fn new() -> Self {
        Self {
            backend: FfmpegBackend::default(),
            player: None,
            cache: Arc::new(Mutex::new(FrameCache::new(CACHE_CAP))),
            meta: None,
            current: 0,
            playing: false,
            next_tick: None,
            stall_since: None,
            tex: None,
            tex_frame: None,
            tex_rev: 0,
            last_frame: None,
            doc: Doc::default(),
            selection: None,
            tool: Tool::Pointer,
            drawing: None,
            resize: None,
            brush_radius: 12.0,
            show_masks: true,
            display_rect: None,
            undo: Vec::new(),
            redo: Vec::new(),
            doc_rev: 0,
            preview_mode: false,
            preview_strategy: vc_inpaint::Strategy::Neural,
            preview_frame: None,
            preview_src: None,
            preview_info: None,
            preview_rx: None,
            last_prefill: None,
            status: "Откройте видеофайл".into(),
            error: None,
            export_dialog: false,
            export_out: String::new(),
            export_crf: 18,
            export_handle: None,
            export_cancelled: false,
            export_failed_msg: None,
            export_done: 0,
            export_total: 0,
            export_phase: String::new(),
            export_frames: (0, 0),
            export_started: None,
            export_use_video_model: false,
            mask_dilate: 3,
        }
    }

    /// Загрузка файла: метаданные + фоновый декодер.
    fn open_video(&mut self, path: std::path::PathBuf) {
        self.error = None;
        self.tex = None;
        self.tex_frame = None;
        self.tex_rev = 0;
        self.last_frame = None;
        self.playing = false;
        self.next_tick = None;
        // Новое видео = новый документ.
        self.doc = Doc::default();
        self.selection = None;
        self.drawing = None;
        self.undo.clear();
        self.redo.clear();
        self.doc_rev += 1;
        self.preview_frame = None;
        self.preview_src = None;
        self.preview_rx = None;
        match self.backend.probe(&path) {
            Ok(meta) => {
                self.status = format!(
                    "Загружено: {} × {} @ {:.2} fps{}",
                    meta.width,
                    meta.height,
                    meta.fps,
                    if meta.has_audio { " (со звуком)" } else { "" }
                );
                self.current = 0;
                let player = Player::start(FfmpegBackend::default(), path, &meta, CACHE_CAP);
                player.seek(0);
                self.cache = player.cache();
                self.player = Some(player);
                self.meta = Some(meta);
            }
            Err(e) => {
                self.player = None;
                self.error = Some(e.to_string());
                self.status = "Ошибка".into();
            }
        }
    }

    /// Прыжок на кадр. Seek фоновому декодеру — только если кадра нет в кэше.
    pub(crate) fn goto(&mut self, idx: u64) {
        let meta = match &self.meta {
            Some(m) => m,
            None => return,
        };
        let idx = idx.min(total_frames(meta).saturating_sub(1));
        if self.current == idx && self.tex.is_some() {
            return;
        }
        // Фокус кэша всегда = просматриваемый кадр: иначе во время
        // последовательного воспроизведения (без seek'ов) focus остаётся на
        // старой позиции, вытеснение удаляет кадры вокруг текущего, и
        // воспроизведение регулярно упирается в «застой» с re-seek'ом.
        let cached = {
            let mut c = self.cache.lock().unwrap();
            let has = c.contains(idx);
            c.set_focus(idx);
            has
        };
        self.current = idx;
        if !cached {
            if let Some(p) = &self.player {
                p.seek(idx);
            }
        }
    }

    /// Ручное перемещение: останавливает воспроизведение.
    pub(crate) fn step(&mut self, delta: i64) {
        let cur = self.current as i64 + delta;
        self.stop_playback();
        self.goto(cur.max(0) as u64);
    }

    /// Прыжок к ближайшей границе диапазона масок (start/end) в направлении.
    /// Поиск ближайшего кадра-границы, строго отличного от текущего.
    pub(crate) fn jump_mask_boundary(&mut self, forward: bool) {
        let Some(meta) = self.meta.clone() else {
            return;
        };
        let total = total_frames(&meta);
        let max_idx = total.saturating_sub(1);
        // Все границы: start каждого региона и конец (для UNTIL — конец видео).
        let mut edges: Vec<u64> = Vec::new();
        for r in &self.doc.regions {
            if r.start > 0 && r.start < max_idx {
                edges.push(r.start);
            }
            let e = if r.end == vc_core::UNTIL_END {
                max_idx
            } else {
                r.end.min(max_idx)
            };
            if e < max_idx {
                edges.push(e + 1);
            }
        }
        edges.retain(|&e| e != self.current);
        edges.sort_unstable();
        let target = if forward {
            edges.into_iter().find(|&e| e > self.current)
        } else {
            edges.into_iter().rev().find(|&e| e < self.current)
        };
        if let Some(idx) = target {
            self.goto(idx);
        } else {
            self.status = format!("границ масок {} нет", if forward { "дальше" } else { "раньше" });
        }
    }

    pub(crate) fn stop_playback(&mut self) {
        if self.playing {
            self.playing = false;
            self.next_tick = None;
            self.stall_since = None;
        }
    }

    pub(crate) fn toggle_play(&mut self) {
        if self.meta.is_none() {
            return;
        }
        if self.playing {
            self.stop_playback();
        } else {
            self.playing = true;
            self.next_tick = Some(Instant::now());
        }
    }

    /// Тик воспроизводителя (по строгим интервалам 1/fps).
    fn tick_playback(&mut self, ctx: &egui::Context) {
        if !self.playing {
            return;
        }
        let fps = self.meta.as_ref().map_or(25.0, |m| fps_of(m));
        let step = Duration::from_secs_f64(1.0 / fps);
        let total = self.meta.as_ref().map_or(1, total_frames);
        let t = match self.next_tick {
            Some(t) => t,
            None => return,
        };
        let now = Instant::now();
        if now >= t {
            if self.current + 1 >= total {
                self.current = total.saturating_sub(1);
                self.playing = false;
                self.next_tick = None;
            } else {
                let next_cached = self.cache.lock().unwrap().contains(self.current + 1);
                if next_cached {
                    // Кадр готов — обычное продвижение.
                    self.goto(self.current + 1);
                    self.next_tick = Some(t + step);
                    self.stall_since = None;
                } else {
                    // Декодер не успел: НЕ убегаем вперёд и НЕ дёргаем seek
                    // (каждый Seek в player.rs убивает последовательное
                    // чтение и переоткрывает ffmpeg → лайвлок «зависло»).
                    // Ждём кэш; принудительный re-seek — только при долгом
                    // полном застое (мёртвый декодер), как восстановление.
                    let since = *self.stall_since.get_or_insert(now);
                    if now.duration_since(since) > Duration::from_millis(300) {
                        if let Some(p) = &self.player {
                            p.seek(self.current + 1);
                        }
                        self.next_tick = Some(now + step);
                        self.stall_since = None;
                    } else {
                        ctx.request_repaint_after(Duration::from_millis(16));
                        return;
                    }
                }
            }
        }
        ctx.request_repaint_after(t.saturating_duration_since(now));
    }

    // -----------------------------------------------------------------------
    // Документ / история
    // -----------------------------------------------------------------------

    pub(crate) fn push_undo(&mut self) {
        self.undo.push(self.doc.clone());
        if self.undo.len() > UNDO_LIMIT {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.doc_rev += 1;
    }

    pub(crate) fn undo(&mut self) {
        if let Some(prev) = self.undo.pop() {
            self.redo.push(std::mem::replace(&mut self.doc, prev));
            self.selection = None;
            self.doc_rev += 1; // превью и текстура должны перестроиться
        }
    }

    pub(crate) fn redo(&mut self) {
        if let Some(next) = self.redo.pop() {
            self.undo.push(std::mem::replace(&mut self.doc, next));
            self.selection = None;
            self.doc_rev += 1; // превью и текстура должны перестроиться
        }
    }

    /// M6: автоопределение текста на текущем кадре.    ///
    /// Все найденные области добавляются одной пачкой (один undo) как
    /// прямоугольные маски от текущего кадра до конца (хардсаб/оверлей живёт
    /// долго); пользователь затем правит/удаляет их обычными средствами.
    pub(crate) fn detect_text_now(&mut self) {
        if self.meta.is_none() {
            self.status = "Сначала откройте видео".into();
            return;
        }
        let Some(frame) = self.cache.lock().unwrap().get_cloned(self.current) else {
            self.status = "Кадр ещё не готов".into();
            return;
        };
        let opts = vc_detect::swt::DetectOptions::default();
        let dets = vc_detect::swt::detect_text(&frame, &opts);
        if dets.is_empty() {
            self.status = "Текст не найден".into();
            return;
        }
        self.push_undo();
        let start = self.current;
        for d in &dets {
            self.doc.add(
                Shape::Rect {
                    corners: [[d.x0 as f32, d.y0 as f32], [d.x1 as f32, d.y1 as f32]],
                },
                start,
                UNTIL_END,
            );
        }
        self.preview_src = None;
        self.preview_rx = None;
        self.status = format!(
            "Найдено областей: {} — правьте или удаляйте вручную",
            dets.len()
        );
    }

    // -----------------------------------------------------------------------
    // M7: проект и экспорт
    // -----------------------------------------------------------------------

    /// Сохранение проекта (.vcproj): источник + все маски, JSON.
    pub(crate) fn save_project(&mut self) {
        let Some(p) = rfd::FileDialog::new()
            .set_title("Сохранить проект")
            .add_filter("Проект VideoCleaning", &["vcproj"])
            .set_file_name("проект.vcproj")
            .save_file()
        else {
            return;
        };
        let source = self
            .meta
            .as_ref()
            .map(|m| m.path.display().to_string())
            .unwrap_or_default();
        let project = serde_json::json!({
            "version": 1,
            "source": source,
            "regions": self.doc.regions,
        });
        match serde_json::to_string_pretty(&project)
            .map_err(|e| e.to_string())
            .and_then(|s| std::fs::write(&p, s).map_err(|e| e.to_string()))
        {
            Ok(()) => self.status = format!("Проект сохранён: {}", p.display()),
            Err(e) => self.error = Some(format!("Сохранение проекта: {e}")),
        }
    }

    /// Загрузка проекта (.vcproj): маски + опционально открыть видео.
    pub(crate) fn load_project(&mut self) {
        let Some(p) = rfd::FileDialog::new()
            .set_title("Загрузить проект")
            .add_filter("Проект VideoCleaning", &["vcproj"])
            .pick_file()
        else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(&p) else {
            self.error = Some(format!("Не удалось прочитать {}", p.display()));
            return;
        };
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(format!("Проект не JSON: {e}"));
                return;
            }
        };
        let mut saved_regions: Vec<MaskRegion> = Default::default();
        if let Some(arr) = v.get("regions").and_then(|r| r.as_array()) {
            saved_regions = arr
                .iter()
                .filter_map(|r| serde_json::from_value(r.clone()).ok())
                .collect();
        }
        let source_path = v
            .get("source")
            .and_then(|s| s.as_str())
            .map(PathBuf::from);
        self.push_undo();
        let next_id = saved_regions.iter().map(|r| r.id + 1).max().unwrap_or(0);
        self.doc.regions = saved_regions;
        self.doc.next_id = next_id.max(1);
        self.selection = None;
        // Открываем видео из проекта, если оно другое.
        if let Some(src) = source_path {
            let differs = self
                .meta
                .as_ref()
                .map(|m| m.path != src)
                .unwrap_or(true);
            if differs && src.exists() {
                self.open_video(src.clone());
            }
        }
        self.status = format!(
            "Проект загружен: {} масок ({})",
            self.doc.regions.len(),
            p.display()
        );
    }

    /// Старт экспортирования в фоне.
    pub(crate) fn start_export(&mut self) {
        let Some(meta) = self.meta.clone() else {
            return;
        };
        if self.export_handle.is_some() {
            self.status = "Экспорт уже идёт".into();
            return;
        }
        let out = PathBuf::from(&self.export_out);
        if out.as_os_str().is_empty() {
            self.status = "Укажите путь назначения".into();
            return;
        }
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.export_handle = Some(ExportHandle {
            rx,
            cancel: Arc::clone(&cancel),
        });
        self.export_dialog = false;
        self.export_cancelled = false;
        self.export_failed_msg = None;
        self.export_phase = String::new();
        self.export_frames = (0, 0);
        self.export_started = Some(std::time::Instant::now());
        self.status = format!("Экспорт запущен → {}", out.display());

        let backend = self.backend.clone();
        let regions = self.doc.regions.clone();
        let strategy = self.preview_strategy;
        let radius = INPAINT_RADIUS;
        let crf = self.export_crf;
        let dilate = self.mask_dilate;
        let use_ai = self.export_use_video_model;
        std::thread::spawn(move || {
            if use_ai {
                crate::export_ai::run_video_export_threaded(&meta, regions, dilate, crf, out, &cancel, &tx);
                return;
            }
            let mut ex = vc_codec::export::Exporter::new(&backend, &meta, cancel, |done, total| {
                let _ = tx.send(ExportMsg::Progress(done, total));
            });
            ex.regions = regions;
            ex.strategy = strategy;
            ex.radius = radius;
            ex.config = vc_codec::export::ExportConfig {
                crf,
                mask_dilate: dilate,
                ..Default::default()
            };
            let res = ex.run(meta.path.as_path(), &out);
            match res {
                Ok(s) => {
                    let _ = tx.send(ExportMsg::Done(s.frames_in, s.frames_modified, s.cancelled));
                }
                Err(e) => {
                    let _ = tx.send(ExportMsg::Failed(e.to_string()));
                }
            }
        });
    }

    /// Опрос рабочего потока экспорта (каждый кадр GUI).
    pub(crate) fn poll_export(&mut self) {
        let mut msgs: Vec<ExportMsg> = Vec::new();
        let finished = if let Some(h) = &self.export_handle {
            loop {
                match h.rx.try_recv() {
                    Ok(m) => msgs.push(m),
                    Err(TryRecvError::Empty) => break false,
                    Err(TryRecvError::Disconnected) => break true,
                }
            }
        } else {
            false
        };
        let mut done_stats: Option<(u64, u64, bool)> = None;
        for m in msgs {
            match m {
                ExportMsg::Progress(done, total) => {
                    self.export_done = done;
                    self.export_total = total;
                    // Обычный экспорт шлёт кадры прямо в Progress.
                    if !self.export_use_video_model {
                        self.export_frames = (done, total);
                    }
                }
                ExportMsg::Frames(done, total) => self.export_frames = (done, total),
                ExportMsg::Phase(ph) => self.export_phase = ph,
                ExportMsg::Done(a, b, c) => done_stats = Some((a, b, c)),
                ExportMsg::Failed(e) => {
                    self.export_failed_msg = Some(e.clone());
                    self.error = Some(format!("Экспорт: {e}"));
                }
            }
        }
        if done_stats.is_some() || finished {
            if let Some((a, b, c)) = done_stats {
                if c {
                    self.status = format!("Экспорт отменён ({a}/{b} кадров записано)");
                } else {
                    self.status = format!("Экспорт готов: {b} кадров обработано из {a} ✓");
                }
            }
            self.export_handle = None;
        }
    }

    /// Тянет угловую ручку выбранного прямоугольника в точку кадра `q`
    /// (координаты клампятся к границам кадра).
    pub(crate) fn resize_rect_to(&mut self, q: [f32; 2], frame_w: u32, frame_h: u32) {
        let Some(rz) = self.resize else { return };
        let qx = q[0].clamp(0.0, frame_w as f32 - 1.0);
        let qy = q[1].clamp(0.0, frame_h as f32 - 1.0);
        let Some(region) = self
            .doc
            .regions
            .iter_mut()
            .find(|r| r.id == rz.id)
        else {
            return;
        };
        let Shape::Rect { corners } = &mut region.shape else { return };
        // пара замыканий: обновляем одну сторону по оси (мин или макс).
        fn set_side(stored: &mut [[f32; 2]; 2], axis: usize, take_min: bool, val: f32) {
            let imin = if stored[0][axis] <= stored[1][axis] { 0usize } else { 1 };
            stored[if take_min { imin } else { 1 - imin }][axis] = val;
        }
        set_side(corners, 0, rz.min_x, qx);
        set_side(corners, 1, rz.min_y, qy);
        self.doc_rev += 1;
    }

    /// Фиксация недорисованной фигуры как новой маски.
    pub(crate) fn commit_drawing(&mut self, span_from_current: bool) {
        let Some(d) = self.drawing.take() else { return };
        let frame = self.current;
        let shape = match d.tool {
            Tool::Rect => {
                let Some(&start) = d.points.first() else { return };
                Shape::Rect { corners: [start, d.cur] }
            }
            Tool::Brush => {
                if d.points.is_empty() {
                    return;
                }
                Shape::Brush {
                    points: d.points.clone(),
                    radius: self.brush_radius.max(1.0),
                }
            }
            Tool::Polygon => {
                if d.points.len() < 3 {
                    return;
                }
                Shape::Polygon { vertices: d.points.clone() }
            }
            Tool::Pointer => return,
        };
        let (start, end) = if span_from_current { (frame, UNTIL_END) } else { (frame, frame) };
        self.push_undo();
        let id = self.doc.add(shape, start, end);
        self.selection = Some(id);
    }

    /// Клик инструментом «Осмотр»: выделить верхнюю маску под курсором.
    pub(crate) fn select_at(&mut self, frame_pos: [f32; 2]) {
        let target = self
            .doc
            .regions
            .iter()
            .filter(|r| r.active_at(self.current) && shape_contains(&r.shape, frame_pos))
            .map(|r| r.id)
            .max();
        self.selection = target;
    }

    // -----------------------------------------------------------------------
    // Клавиатурные действия
    // -----------------------------------------------------------------------

    fn handle_keys(&mut self, ctx: &egui::Context) {
        ctx.input(|i| {
            let big = i.modifiers.shift;
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.step(if big { -25 } else { -1 });
            } else if i.key_pressed(egui::Key::ArrowRight) {
                self.step(if big { 25 } else { 1 });
            } else if i.key_pressed(egui::Key::Home) {
                self.step(-i64::MAX / 2);
            } else if i.key_pressed(egui::Key::End) {
                self.step(i64::MAX / 2);
            } else if i.key_pressed(egui::Key::Space) {
                self.toggle_play();
            } else if i.key_pressed(egui::Key::Z) && i.modifiers.ctrl && !i.modifiers.shift {
                self.undo();
            } else if i.key_pressed(egui::Key::Z) && i.modifiers.ctrl && i.modifiers.shift
                || (i.key_pressed(egui::Key::Y) && i.modifiers.ctrl)
            {
                self.redo();
            } else if i.key_pressed(egui::Key::Delete)
                || i.key_pressed(egui::Key::Backspace)
            {
                if let Some(id) = self.selection.take() {
                    self.push_undo();
                    self.doc.remove(id);
                }
            } else if i.key_pressed(egui::Key::Escape) {
                // Отмена недорисованной фигуры.
                self.drawing = None;
            } else if i.key_pressed(egui::Key::Enter) {
                // Завершение полигона.
                if let Some(d) = &self.drawing {
                    if d.tool == Tool::Polygon && d.points.len() >= 3 {
                        self.commit_drawing(true);
                    }
                }
            }
        });
    }
}

impl eframe::App for VcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_style(ctx);
        self.tick_playback(ctx);
        self.poll_export();

        // Сбои фонового декодера светим, если есть.
        if let Some(p) = &self.player {
            if let Some(e) = p.take_error() {
                self.error = Some(e);
            }
        }

        self.handle_keys(ctx);

        tool_panel(self, ctx);

        // Верхняя панель.
        egui::TopBottomPanel::top("vc_top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📂 Открыть…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Видео", &["mp4", "mkv", "webm", "avi", "mov"])
                        .pick_file()
                    {
                        self.open_video(path);
                    }
                }
                ui.separator();
                if ui.button("💾 Проект").clicked() {
                    self.save_project();
                }
                if ui.button("📋 Загрузить").clicked() {
                    self.load_project();
                }
                let exporting = self.export_handle.is_some();
                if ui
                    .add_enabled(self.meta.is_some() && !exporting, egui::Button::new("🎬 Экспорт…"))
                    .clicked()
                {
                    if self.export_out.is_empty() {
                        if let Some(meta) = &self.meta {
                            self.export_out = default_export_name(&meta.path);
                        }
                    }
                    self.export_dialog = true;
                }
                if exporting {
                    let frac = self.export_done as f32 / self.export_total.max(1) as f32;
                    ui.add(
                        egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                            .show_percentage()
                            .desired_width(150.0),
                    );
                    if !self.export_phase.is_empty() {
                        ui.label(
                            egui::RichText::new(&self.export_phase).weak().small(),
                        );
                    }
                    // Строка показателей: кадры этапа, часы, оценка остатка.
                    ui.vertical(|ui| {
                        let (fdone, ftotal) = self.export_frames;
                        if ftotal > 0 {
                            ui.label(
                                egui::RichText::new(format!(
                                    "кадров: {fdone}/{ftotal}"
                                ))
                                .weak()
                                .small(),
                            );
                        }
                        if let Some(t0) = self.export_started {
                            let e = t0.elapsed();
                            let mm = e.as_secs() / 60;
                            let ss = e.as_secs() % 60;
                            if frac > 0.015 && frac < 0.995 {
                                let eta =
                                    e.as_secs_f64() / frac as f64 * (1.0 - frac as f64);
                                ui.label(
                                    egui::RichText::new(format!(
                                        "прошло {mm:02}:{ss:02} · осталось ~{:02}:{:02}",
                                        (eta / 60.0) as u64,
                                        (eta % 60.0) as u64,
                                    ))
                                    .weak()
                                    .small(),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "прошло {mm:02}:{ss:02}"
                                    ))
                                    .weak()
                                    .small(),
                                );
                            }
                        }
                    });
                    if ui.button("✋ Стоп").clicked() {
                        if let Some(h) = &self.export_handle {
                            h.cancel.store(true, Ordering::Relaxed);
                        }
                    }
                }
                if ui
                    .button(if self.playing { "⏸ Пауза" } else { "▶ Играть" })
                    .clicked()
                {
                    self.toggle_play();
                }
                if let Some(meta) = &self.meta {
                    ui.separator();
                    ui.label(format!(
                        "{} × {} · {:.2} fps",
                        meta.width, meta.height, meta.fps
                    ));
                    if let Some(d) = meta.duration_secs {
                        ui.label(format!("длина: {d:.1} сек"));
                    }
                    ui.separator();
                    let tc = Timecode::new(self.current, meta.fps);
                    ui.label(format!(
                        "кадр {} из {} · {}",
                        self.current,
                        total_frames(meta).saturating_sub(1),
                        tc.format()
                    ));
                }
                ui.separator();
                ui.label(&self.status);
                if let Some(err) = &self.error {
                    ui.colored_label(Color32::RED, format!("⚠ {err}"));
                }
            });
        });

        // Окно экспорта (M7): локальные копии значений на время окна.
        let mut out_text = self.export_out.clone();
        let mut crf = self.export_crf;
        let mut use_ai = self.export_use_video_model;
        let mut open = self.export_dialog;
        let strat_label = match self.preview_strategy {
            vc_inpaint::Strategy::Background => "фон (кадры-доноры)",
            vc_inpaint::Strategy::Neural => "умное удаление (ретушь + перерисовка)",
            vc_inpaint::Strategy::Telea => "Telea",
        };
        egui::Window::new("Экспорт")
            .open(&mut open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Назначение:");
                    ui.add(
                        egui::TextEdit::singleline(&mut out_text)
                            .desired_width(300.0)
                            .code_editor(),
                    );
                });
                ui.add(
                    egui::Slider::new(&mut crf, 14..=28)
                        .text("CRF (меньше = качественнее)"),
                );
                ui.separator();
                ui.label(format!("Стратегия: {strat_label} (как в превью)"));
                ui.label(format!("Масок: {}", self.doc.regions.len()));
                ui.add_space(2.0);
                let ai_hint = if std::path::Path::new("./tools/venv/bin/python").exists() {
                    "ProPainter: видео-модель обрабатывает кроп вокруг масок; медленно, но на динамичном фоне заметно лучше"
                } else {
                    "ProPainter не найден (tools/venv) — недоступно"
                };
                ui.add_enabled(
                    std::path::Path::new("./tools/venv/bin/python").exists(),
                    egui::Checkbox::new(&mut use_ai, "Видео-инпейнтинг ProPainter (качественнее, дольше)"),
                )
                .on_hover_text(ai_hint);
                ui.add_space(6.0);
                if ui.button("Старт").clicked() {
                    self.export_out = out_text.clone();
                    self.export_crf = crf;
                    self.export_use_video_model = use_ai;
                    self.start_export();
                }
            });
        self.export_dialog = open;
        self.export_out = out_text;
        self.export_crf = crf;
        self.export_use_video_model = use_ai;

        self.central_panel(ctx);
        masks_panel(self, ctx);
        timeline_panel(self, ctx);

        // Живая перерисовка во время рисования.
        if self.drawing.is_some() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }
}

/// Современная тёмная тема: приглушённые панели, мягкие скругления,
/// индиговый акцент выделения, аккуратный текст.
pub(crate) fn apply_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    style.spacing.indent = 16.0;

    let v = &mut style.visuals;
    v.panel_fill = Color32::from_rgb(0x1A, 0x1C, 0x22);
    v.window_fill = Color32::from_rgb(0x1F, 0x22, 0x29);
    v.window_stroke =
        egui::Stroke::new(1.0, Color32::from_rgb(0x2E, 0x33, 0x3E));
    v.faint_bg_color = Color32::from_rgb(0x20, 0x23, 0x2B);
    v.extreme_bg_color = Color32::from_rgb(0x10, 0x12, 0x16);
    let accent = Color32::from_rgb(0x7C, 0x9F, 0xFF);
    let accent_soft = Color32::from_rgb(0x3A, 0x4A, 0x75);
    v.selection.bg_fill = accent_soft;
    v.selection.stroke = egui::Stroke::new(
        1.0,
        Color32::from_rgb(0xA8, 0xC0, 0xFF),
    );
    v.hyperlink_color = accent;
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.weak_bg_fill = w.bg_fill;
    }
    v.widgets.inactive.bg_fill = Color32::from_rgb(0x26, 0x29, 0x32);
    v.widgets.inactive.weak_bg_fill = v.widgets.inactive.bg_fill;
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x2E, 0x33, 0x40);
    v.widgets.hovered.bg_fill = v.widgets.hovered.weak_bg_fill;
    v.widgets.active.weak_bg_fill = accent_soft;
    v.widgets.active.bg_fill = accent_soft;
    v.widgets.active.fg_stroke =
        egui::Stroke::new(1.3, Color32::from_rgb(0xE8, 0xEE, 0xFF));
    v.window_corner_radius = 8.into();
    v.menu_corner_radius = 6.into();
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = 6.into();
    }
    v.widgets.hovered.fg_stroke =
        egui::Stroke::new(1.2, Color32::from_rgb(0xD8, 0xE0, 0xEE));
    v.widgets.inactive.fg_stroke =
        egui::Stroke::new(1.1, Color32::from_rgb(0xC2, 0xC9, 0xD8));
    ctx.set_style(style);
}

/// Путь экспорта по умолчанию: рядом с источником, `имя_clean.mp4`.
pub(crate) fn default_export_name(src: &std::path::Path) -> String {
    let stem = src
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "video".into());
    src.with_file_name(format!("{stem}_clean.mp4"))
        .display()
        .to_string()
}

/// Проверка попадания кадровой точки в фигуру.
pub(crate) fn shape_contains(shape: &Shape, p: [f32; 2]) -> bool {
    match shape {
        Shape::Rect { corners } => {
            let [[x0, y0], [x1, y1]] = *corners;
            let (mx, xx) = (x0.min(x1), x0.max(x1));
            let (my, xy) = (y0.min(y1), y0.max(y1));
            p[0] >= mx && p[0] <= xx && p[1] >= my && p[1] <= xy
        }
        Shape::Polygon { vertices } => {
            if vertices.len() < 3 {
                return false;
            }
            // even-odd: посчитаем пересечения горизонтального луча.
            let mut inside = false;
            let mut j = vertices.len() - 1;
            for i in 0..vertices.len() {
                let a = vertices[i];
                let b = vertices[j];
                let (ai, bi) = (a[1] > p[1], b[1] > p[1]);
                if ai != bi {
                    let t = (p[1] - a[1]) / (b[1] - a[1]);
                    let x = a[0] + t * (b[0] - a[0]);
                    if x > p[0] {
                        inside = !inside;
                    }
                }
                j = i;
            }
            inside
        }
        Shape::Brush { points, radius } => points.iter().any(|&pt| {
            let dx = p[0] - pt[0];
            let dy = p[1] - pt[1];
            dx * dx + dy * dy <= radius * radius
        }),
    }
}

/// Кадр → egui-картинка (RGB, альфе видео не доверяем).
pub(crate) fn to_color_image(frame: &Frame) -> egui::ColorImage {
    let mut data = frame.data.clone();
    for a in data.iter_mut().skip(3).step_by(4) {
        *a = 255;
    }
    egui::ColorImage::from_rgba_unmultiplied([frame.width as usize, frame.height as usize], &data)
}
