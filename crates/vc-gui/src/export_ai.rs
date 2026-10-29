//! Экспорт через видео-модель (ProPainter): сайдкар python из `tools/`.
//!
//! Архитектура: маски по кадрам растеризует та же `vc_core::rasterize`
//! (сырые u8-файлы), ProPainter получает кроп вокруг масок (память!),
//! результат вставляется в полноразмерные кадры; звук добавляет муксер Rust.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vc_core::{MaskRegion, UNTIL_END};

use crate::app::ExportMsg;

/// Отступ вокруг bbox масок (контекст для потоков/трансформера).
const CONTEXT_PX: i64 = 48;
/// Кадров запаса до/после активного диапазона (доноры).
const FRAMES_MARGIN: u64 = 20;

/// Живой лог экспорта: `<out>.export.log`, строки `[мм:сс] текст` с
/// тайм-метками от старта. Пишется с флашем — можно `tail -f` рядом.
#[derive(Clone)]
struct ExportLog {
    path: PathBuf,
    t0: std::time::Instant,
}

impl ExportLog {
    fn start(
        path: &Path,
        plan: &VideoExportPlan,
        meta: &vc_codec::VideoMetadata,
        crf: u32,
    ) -> Self {
        let mut header = format!(
            "VideoCleaning: экспорт видео-моделью (ProPainter)\n\
             видео: {}\n\
             кадр: {}x{} @ {:.3}fps\n\
             crf: {}\n\
             сайдкар: {}\n\
             лог gpu: {}\n\
             сегменты: {}\n",
            meta.path.display(),
            meta.width,
            meta.height,
            meta.fps,
            crf,
            script_path().display(),
            path.with_extension("vcvideo.log").display(),
            plan.segments.len(),
        );
        for (i, s) in plan.segments.iter().enumerate() {
            let (x, y, cw, ch) = s.crop;
            header.push_str(&format!(
                "  [{:02}] кадры {}…{} ({} шт), кроп x={} y={} {}x{}\n",
                i + 1,
                s.frame_start,
                s.frame_start + s.frame_count - 1,
                s.frame_count,
                x,
                y,
                cw,
                ch,
            ));
        }
        header.push_str("  (кадры между сегментами копируются без обработки)\n---\n");
        let _ = std::fs::write(path, header);
        Self { path: path.to_path_buf(), t0: std::time::Instant::now() }
    }

    /// Строка в лог с тайм-меткой (ошибки записи молча игнорируются:
    /// лог не должен ломать экспорт).
    fn note(&self, msg: &str) {
        let t = self.t0.elapsed().as_secs();
        let line = format!("[{:02}:{:02}] {}\n", t / 60, t % 60, msg);
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&self.path) {
            use std::io::Write;
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
    }

    /// Процент полосы (0..100) — гранулярные строки.
    fn percent(&self, frac: f32, source: &str) {
        let p = (frac * 100.0).round() as i64;
        self.note(&format!("{source}: {p}%"));
    }
}

pub(crate) struct VideoExportPlan {
    /// Сегменты обработки: кластеры масок по времени. Между сегментами
    /// кадры не обрабатываются (копируются как есть).
    pub segments: Vec<ExportSegment>,
    /// Суммарное число обрабатываемых кадров (по всем сегментам).
    pub frames_total: u64,
}

pub(crate) struct ExportSegment {
    /// Область кропа (x, y, w, h), кратная 8, внутри кадра.
    pub crop: (i64, i64, u32, u32),
    /// Диапазон обработки: стартовый кадр и длина (включая ±запас).
    pub frame_start: u64,
    pub frame_count: u64,
}

/// bbox фигуры в пикселях кадра (для кропа).
fn shape_bbox(shape: &vc_core::Shape, w: u32, h: u32) -> (i64, i64, i64, i64) {
    let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    let mut add = |px: f32, py: f32, r: f32| {
        x0 = x0.min((px - r).max(0.0).floor() as i64);
        y0 = y0.min((py - r).max(0.0).floor() as i64);
        x1 = x1.max((px + r).min(w as f32).ceil() as i64);
        y1 = y1.max((py + r).min(h as f32).ceil() as i64);
    };
    match shape {
        vc_core::Shape::Rect { corners } => {
            let [[ax, ay], [bx, by]] = corners;
            let (ax, ay, bx, by) = (*ax, *ay, *bx, *by);
            add(ax.min(bx), ay.min(by), 0.0);
            add(ax.max(bx), ay.max(by), 0.0);
        }
        vc_core::Shape::Polygon { vertices } => {
            for &[x, y] in vertices {
                add(x, y, 0.0);
            }
        }
        vc_core::Shape::Brush { points, radius } => {
            for &[x, y] in points {
                add(x, y, *radius);
            }
        }
    }
    if x1 < x0 || y1 < y0 {
        (0, 0, 0, 0)
    } else {
        (x0, y0, x1, y1)
    }
}

/// План экспорта: маски группируются в кластеры по времени — соседние
/// диапазоны (зазор ≤ 2×запаса) образуют один сегмент со своим кропом
/// (bbox только своих масок + контекст, кратность 8). `None` — нет масок.
pub(crate) fn plan_video_export(
    meta: &vc_codec::VideoMetadata,
    regions: &[MaskRegion],
    dilate: u32,
) -> Option<VideoExportPlan> {
    if regions.is_empty() {
        return None;
    }
    let (w, h) = (meta.width, meta.height);
    let total = meta
        .frame_count
        .unwrap_or_else(|| meta.duration_secs.map_or(1, |d| (d * meta.fps).round() as u64))
        .max(1);
    let last = total.saturating_sub(1);
    let d = dilate as i64;

    // Отдельный коридор на каждую маску: временной диапазон + bbox.
    struct Span {
        lo: u64,
        hi: u64,
        bx0: i64,
        by0: i64,
        bx1: i64,
        by1: i64,
    }
    let mut spans: Vec<Span> = regions
        .iter()
        .map(|r| {
            let (x0, y0, x1, y1) = shape_bbox(&r.shape, w, h);
            Span {
                lo: r.start.min(last),
                hi: if r.end == UNTIL_END { last } else { r.end.min(last) },
                bx0: (x0 - d).max(0),
                by0: (y0 - d).max(0),
                bx1: (x1 + d).min(w as i64),
                by1: (y1 + d).min(h as i64),
            }
        })
        .collect();
    spans.sort_by_key(|s| s.lo);

    // Кластеризация: зазор ≤ 2×запаса ⇒ соседние сегменты делились бы
    // запасными кадрами — сливаем в один.
    let mut clusters: Vec<Span> = Vec::new();
    for s in spans {
        match clusters.last_mut() {
            Some(c) if s.lo <= c.hi + 2 * FRAMES_MARGIN => {
                c.hi = c.hi.max(s.hi);
                c.bx0 = c.bx0.min(s.bx0);
                c.by0 = c.by0.min(s.by0);
                c.bx1 = c.bx1.max(s.bx1);
                c.by1 = c.by1.max(s.by1);
            }
            _ => clusters.push(Span {
                lo: s.lo,
                hi: s.hi,
                bx0: s.bx0,
                by0: s.by0,
                bx1: s.bx1,
                by1: s.by1,
            }),
        }
    }

    let mut segments = Vec::with_capacity(clusters.len());
    let mut frames_total = 0u64;
    for c in clusters {
        let seg_start = c.lo.saturating_sub(FRAMES_MARGIN).min(last);
        let seg_end = (c.hi + FRAMES_MARGIN).min(last);
        // Кроп кластера: bbox только его масок + контекст, кратность 8.
        let cx0 = (c.bx0 - CONTEXT_PX).max(0);
        let cy0 = (c.by0 - CONTEXT_PX).max(0);
        let cx1 = (c.bx1 + CONTEXT_PX).min(w as i64);
        let cy1 = (c.by1 + CONTEXT_PX).min(h as i64);
        let cw = ((cx1 - cx0).max(16) / 8 * 8).max(16);
        let ch = ((cy1 - cy0).max(16) / 8 * 8).max(16);
        // Клэмп внутри кадра.
        let cx0 = cx0.min(w as i64 - cw).max(0);
        let cy0 = cy0.min(h as i64 - ch).max(0);
        let count = seg_end - seg_start + 1;
        frames_total += count;
        segments.push(ExportSegment {
            crop: (cx0, cy0, cw as u32, ch as u32),
            frame_start: seg_start,
            frame_count: count,
        });
    }
    Some(VideoExportPlan { segments, frames_total })
}

fn python_path() -> PathBuf {
    if let Ok(p) = std::env::var("VC_PYTHON") {
        return PathBuf::from(p);
    }
    PathBuf::from("./tools/venv/bin/python")
}

fn script_path() -> PathBuf {
    if let Ok(p) = std::env::var("VC_PROPAINTER_CLI") {
        return PathBuf::from(p);
    }
    PathBuf::from("./tools/propainter/inpaint_cli.py")
}

fn repo_path() -> PathBuf {
    if let Ok(p) = std::env::var("VC_PROPAINTER_REPO") {
        return p.into();
    }
    PathBuf::from("./tools/propainter-repo")
}

fn ffmpeg_bin() -> PathBuf {
    vc_codec::FfmpegBackend::default().ffmpeg
}

// ---------------------------------------------------------------------------
// Обёртка для GUI-потока экспорта (вызывается уже в рабочей нити).
// ---------------------------------------------------------------------------

pub(crate) fn run_video_export_threaded(
    meta: &vc_codec::VideoMetadata,
    regions: Vec<MaskRegion>,
    dilate: u32,
    crf: u32,
    out: PathBuf,
    cancel: &Arc<AtomicBool>,
    tx: &std::sync::mpsc::Sender<ExportMsg>,
) {
    let Some(plan) = plan_video_export(meta, &regions, dilate) else {
        let _ = tx.send(ExportMsg::Failed("нет масок для видео-модели".into()));
        return;
    };
    // Полоса GUI: 0..=1000 шагов, из них сайдкар/поллер шлют дробный прогресс.
    let total = 1000u64;
    let _ = tx.send(ExportMsg::Progress(0, total));
    let progress: Arc<dyn Fn(f32) + Send + Sync> = {
        let tx = tx.clone();
        Arc::new(move |frac: f32| {
            let _ = tx.send(ExportMsg::Progress(
                (frac.clamp(0.0, 1.0) * total as f32) as u64,
                total,
            ));
        })
    };
    let phase: Arc<dyn Fn(&str) + Send + Sync> = {
        let tx = tx.clone();
        Arc::new(move |ph| {
            let _ = tx.send(ExportMsg::Phase(ph.to_string()));
        })
    };
    let frames: Arc<dyn Fn(u64, u64) + Send + Sync> = {
        let tx = tx.clone();
        Arc::new(move |done, total| {
            let _ = tx.send(ExportMsg::Frames(done, total));
        })
    };
    match run_video_export(
        meta,
        &regions,
        &plan,
        &out,
        crf,
        cancel,
        &progress,
        &phase,
        &frames,
    ) {
        Ok(true) => {
            let _ = tx.send(ExportMsg::Done(plan.frames_total, plan.frames_total, false));
        }
        Ok(false) => {
            let _ = tx.send(ExportMsg::Done(0, 0, true));
        }
        Err(e) => {
            let _ = tx.send(ExportMsg::Failed(e));
        }
    }
}

/// Полный экспорт видео-моделью (в рабочем потоке).
/// `progress(frac01)` — общий прогресс 0..1; `phase(text)` — подпись этапа.
/// Ok(true) — готово; Ok(false) — отменено; Err — ошибка.
pub(crate) fn run_video_export(
    meta: &vc_codec::VideoMetadata,
    regions: &[MaskRegion],
    plan: &VideoExportPlan,
    out: &Path,
    crf: u32,
    cancel: &Arc<AtomicBool>,
    progress: &Arc<dyn Fn(f32) + Send + Sync>,
    phase: &Arc<dyn Fn(&str) + Send + Sync>,
    frames: &Arc<dyn Fn(u64, u64) + Send + Sync>,
) -> Result<bool, String> {
    let (mw, mh) = (meta.width, meta.height);
    let frames_total = plan.frames_total.max(1);
    let overall = |v: f32| v.clamp(0.0, 1.0);

    // Живой лог экспорта (можно `tail -f` во время рендера): все фазы,
    // проценты, ошибки и итог с тайм-метками от старта.
    let export_log_path = out.with_extension("export.log");
    let exp_log = ExportLog::start(&export_log_path, plan, meta, crf);

    // Прогресс/фазы оборачиваем: всё, что видит GUI, зеркалим в лог.
    let progress: Arc<dyn Fn(f32) + Send + Sync> = {
        let inner = Arc::clone(progress);
        let log = exp_log.clone();
        Arc::new(move |v| {
            inner(v);
            log.percent(v, "прогресс");
        })
    };
    let phase: Arc<dyn Fn(&str) + Send + Sync> = {
        let inner = Arc::clone(phase);
        let log = exp_log.clone();
        Arc::new(move |ph| {
            inner(ph);
            log.note(&format!("фаза: {ph}"));
        })
    };
    let cancel_note = exp_log.clone();
    // Каталог масок — рядом с назначением (на диске, не в tmpfs ОЗУ).
    let work = out
        .parent()
        .unwrap_or(Path::new("."))
        .join(".vc_ai_work");
    let masks = work.join(format!("masks_{}x{}", meta.width, meta.height));
    std::fs::create_dir_all(&masks).map_err(|e| e.to_string())?;

    phase("Маски по кадрам…");
    let mut done = 0u64;
    for seg in &plan.segments {
        for gf in seg.frame_start..seg.frame_start + seg.frame_count {
            if cancel.load(Ordering::Relaxed) {
                cancel_note.note("ОТМЕНА: на стадии растеризации масок");
                return Ok(false);
            }
            let m = vc_core::rasterize(regions, gf, mw, mh).dilated(2);
            write_bytes(&masks.join(format!("{gf:09}.raw")), &m.bits)?;
            done += 1;
            if done % 16 == 0 {
                frames(done, plan.frames_total);
                progress(overall(done as f32 / frames_total as f32 * 0.06));
            }
        }
    }

    // Сайдкар выдаёт видео-only файл; звук добавит муксер к финалу.
    let video_only = out.with_extension("vcvideo.mp4");
    let script = script_path().canonicalize().unwrap_or_else(|_| script_path());
    let repo = repo_path().canonicalize().unwrap_or_else(|_| repo_path());
    // stderr (tqdm/ошибки ProPainter) — сразу в файл: труба не забьётся,
    // а лог остаётся для диагностики.
    let work_log = out.with_extension("vcvideo.log");
    let log_file = std::fs::File::create(&work_log)
        .map_err(|e| format!("лог сайдкара {}: {e}", work_log.display()))?;
    let mut cmd = Command::new(python_path());
    cmd.arg(script)
        .arg("--video").arg(meta.path.as_path())
        .arg("--masks-dir").arg(&masks);
    for seg in &plan.segments {
        let (sx, sy, scw, sch) = seg.crop;
        cmd.arg("--segment").arg(format!(
            "{}:{}:{}:{}:{}:{}",
            seg.frame_start, seg.frame_count, sx, sy, scw, sch
        ));
    }
    cmd.arg("--out").arg(&video_only)
        .arg("--fps").arg(format!("{:.6}", meta.fps))
        .arg("--crf").arg(crf.to_string())
        .arg("--mask-wh").arg(mw.to_string()).arg(mh.to_string())
        .arg("--repo").arg(repo)
        // Файл состояния прогресса GPU-фазы ("k n": сегмент k из n) — его
        // вычитывает tqdm-поллер, строя глобальный процент по всем сегментам.
        .arg("--state").arg(work.join("painter_state"))
        .arg("--subvideo-length").arg("80")
        .stdout(Stdio::piped())
        .stderr(Stdio::from(log_file));

    let mut child: Child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("запуск сайдкара: {e}");
            exp_log.note(&format!("ОШИБКА: {msg}"));
            return Err(msg);
        }
    };
    let stdout = child.stdout.take().expect("stdout piped");

    // Поллер прогресса GPU-фазы: ProPainter пишет tqdm-проценты в stderr
    // (он уже в лог-файле); поллер их вычитывает и ведёт полосу 6—90%.
    // Поллер прогресса GPU-фазы: ProPainter пишет tqdm-проценты в stderr
    // (он уже в лог-файле); поллер их вычитывает и ведёт полосу 6—90%.
    let finished = Arc::new(AtomicBool::new(false));
    let poller = spawn_tqdm_poller(
        work_log.clone(),
        work.join("painter_state"),
        Arc::clone(&finished),
        Arc::clone(&progress),
        Arc::clone(&phase),
    );
    let mut err_from_stdout: Option<String> = None;
    let log_lines = exp_log.clone();
    reader_loop(&mut child, stdout, cancel.as_ref(), |line: &str| {
        if let Some(rest) = line.strip_prefix("PROGRESS ") {
            let mut it = rest.split_whitespace();
            let tag = it.next().unwrap_or("");
            // Маппинг фаз: extract 0—6% (глобальный по всем сегментам),
            // painter 6—90% (гранулярность от tqdm-поллера), paste 90—100%.
            match tag {
                "extract" => {
                    let f: f32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    phase("Подготовка кропа и масок…");
                    log_lines.note(&format!("сайдкар: extract"));
                    let f = f.clamp(0.0, 1.0);
                    frames((f * frames_total as f32) as u64, frames_total);
                    progress(overall(f * 0.06));
                }
                "painter_run" => {
                    // "painter_run k n" — завершён k-й GPU-сегмент из n.
                    let k: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let n: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
                    let n = n.max(1);
                    phase(&format!(
                        "ProPainter: нейро-инпейнтинг (сегмент {k}/{n})…"
                    ));
                    log_lines.note(&format!("сайдкар: painter_run {k}/{n}"));
                    progress(overall(0.06 + 0.84 * (k as f32 / n as f32)));
                }
                "paste" => {
                    // Границу painter→paste бьём честно: конец всех GPU-сегментов.
                    finished.store(true, Ordering::Relaxed);
                    let f: f32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    let f = f.clamp(0.0, 1.0);
                    phase("Сборка полных кадров и кодирование…");
                    log_lines.note(&format!("сайдкар: paste"));
                    // paste-доля — по кадрам output (python знает total).
                    let out_total = meta
                        .frame_count
                        .unwrap_or(0);
                    if out_total > 0 {
                        frames((f * out_total as f32) as u64, out_total);
                    }
                    progress(overall(0.90 + f * 0.10));
                }
                _ => {}
            }
        } else if let Some(e) = line.strip_prefix("ERROR ") {
            log_lines.note(&format!("ОШИБКА от сайдкара: {e}"));
            err_from_stdout = Some(e.to_string());
        }
    });
    // Остановить поллер tqdм (stderr), затем дожидаться сайдкар.
    finished.store(true, Ordering::Relaxed);
    let _ = poller.join();
    phase("Финализация…");
    let st = child.wait().map_err(|e| e.to_string())?;
    exp_log.note(&format!("сайдкар завершился со статусом {st}"));
    progress(overall(0.98));
    if cancel.load(Ordering::Relaxed) {
        exp_log.note("ОТМЕНА: пользователь прервал экспорт");
        let _ = child.kill();
        let _ = std::fs::remove_file(&video_only);
        let _ = std::fs::remove_dir_all(&work);
        return Ok(false);
    }
    if !st.success() {
        let msg = format!("сайдкар вернул статус {st}; лог: {}", work_log.display());
        exp_log.note(&format!("ОШИБКА: {msg}"));
        return Err(msg);
    }
    if let Some(e) = err_from_stdout {
        let msg = format!("{e}; лог: {}", work_log.display());
        exp_log.note(&format!("ОШИБКА: {msg}"));
        return Err(msg);
    }

    phase("Звук и упаковка…");
    if let Err(e) = mux_audio(meta, &video_only, out) {
        exp_log.note(&format!("ОШИБКА муксера звука: {e}"));
        return Err(e);
    }
    let _ = std::fs::remove_file(&video_only);
    let duration = exp_log.t0.elapsed();
    exp_log.note(&format!(
        "ГОТОВО: {} (за {:02}:{:02})",
        out.display(),
        duration.as_secs() / 60,
        duration.as_secs() % 60
    ));
    Ok(true)
}

/// Поллер stderr-лога сайдкара: вычитывает tqdm-проценты ProPainter
/// (`NN%|…`) и файла состояния (`k n` — идёт сегмент k из n) и строит
/// глобальный процент GPU-фазы: (k + pct/100) / n → полоса 6—90%.
/// Выходит по `finished` (ставится читателем stdout на paste/ошибку/отмену).
fn spawn_tqdm_poller(
    log: PathBuf,
    state: PathBuf,
    finished: Arc<AtomicBool>,
    progress: Arc<dyn Fn(f32) + Send + Sync>,
    phase: Arc<dyn Fn(&str) + Send + Sync>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut last = 0u32;
        let mut sent_any = false;
        while !finished.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(500));
            // tqdm пишет \r-обновления; читаем лог целиком (он небольшой),
            // невалидные байты — lossy.
            let Ok(bytes) = std::fs::read(&log) else {
                continue;
            };
            let text = String::from_utf8_lossy(&bytes);
            for chunk in text.split(['\r', '\n']) {
                let t = chunk.trim_start_matches([' ', '\t']);
                let digits_len = t.chars().take_while(|c| c.is_ascii_digit()).count();
                if digits_len == 0 || digits_len > 3 || !t[digits_len..].starts_with('%') {
                    continue;
                }
                if let Ok(p) = t[..digits_len].parse::<u32>() {
                    if p <= 100 && p >= last {
                        last = p;
                    }
                }
            }
            // Состояние сегментов: "k n" (python пишет перед каждым прогоном).
            if let Ok(st) = std::fs::read_to_string(&state) {
                let mut it = st.split_whitespace();
                let k: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                let n: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
                if n > 0 && k < n {
                    phase(&format!(
                        "ProPainter: нейро-инпейнтинг (сегмент {}/{})…",
                        k + 1,
                        n
                    ));
                    let f = (k as f32 + last as f32 / 100.0) / n as f32;
                    progress(0.06 + f.clamp(0.0, 1.0) * 0.84);
                    sent_any = true;
                } else if n > 0 && last > 0 && !sent_any {
                    // Прогресс есть, состояние ещё не перезаписано: не спешим.
                    sent_any = true;
                    progress(0.06 + (last as f32 / 100.0) / n.max(1) as f32 * 0.84);
                }
            }
        }
    })
}

/// Разбор stdout сайдкара с поддержкой отмены.
fn reader_loop(child: &mut Child, stdout: std::process::ChildStdout, cancel: &AtomicBool, mut on_line: impl FnMut(&str)) {
    let mut reader = std::io::BufReader::new(stdout);
    let mut buf = String::new();
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            break;
        }
        buf.clear();
        match reader.read_line(&mut buf) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let l = buf.trim_end();
                if !l.is_empty() {
                    on_line(l);
                }
            }
            Err(_) => break,
        }
    }
}

fn write_bytes(path: &Path, data: &[u8]) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, data).map_err(|e| e.to_string())
}

fn mux_audio(meta: &vc_codec::VideoMetadata, video_only: &Path, out: &Path) -> Result<(), String> {
    let st = Command::new(ffmpeg_bin())
        .args([
            "-y", "-v", "error",
            "-i", video_only.to_string_lossy().as_ref(),
            "-i", meta.path.to_string_lossy().as_ref(),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy",
        ])
        .arg(out)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| e.to_string())?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("статус {st}"))
    }
}
