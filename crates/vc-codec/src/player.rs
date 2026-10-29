//! Фоновый декодер-стример: отдаёт кадры в кэш, пока UI их читает.
//!
//! Схема:
//! - UI управляет через `Player::seek(frame_index)` и читает `Player::cache()`.
//! - Поток держит один `ffmpeg -ss t` процесс и декодирует кадры вперёд
//!   последовательно. При прыжке — процесс убивается и пересоздаётся.
//! - Кэш — LRU с лимитом по байтам; вытеснение — по дальности от `focus`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use vc_core::Frame;

use crate::{CodecBackend, FrameReader, VideoMetadata};

/// Лимит кэша по байтам (по умолчанию 512 МБ ≈ 60 кадров 1080p RGBA).
pub const DEFAULT_CACHE_CAP_BYTES: usize = 512 * 1024 * 1024;

enum Msg {
    /// Сделать доступным кадр с таким индексом (+ продолжить prefetch вперёд).
    Seek(u64),
    /// Подтянуть в кэш `count` кадров перед `base` (для фонового восстановления).
    PrefillBackward { base: u64, count: u32 },
    Quit,
}

/// Готовые кадры. Потокобезопасен через `Arc<Mutex<…>>`.
pub struct FrameCache {
    map: BTreeMap<u64, Frame>,
    bytes: usize,
    cap_bytes: usize,
    /// Индекс кадра, вокруг которого держим кэш (позиция просмотра).
    focus: u64,
    /// Счётчик изменений (инвалидация UI-обновлений вроде превью).
    rev: u64,
}

impl FrameCache {
    pub fn new(cap_bytes: usize) -> Self {
        Self { map: BTreeMap::new(), bytes: 0, cap_bytes, focus: 0, rev: 0 }
    }

    /// Растёт при каждой вставке — сигнал «кэш обновился».
    pub fn revision(&self) -> u64 {
        self.rev
    }

    pub fn get(&self, idx: u64) -> Option<&Frame> {
        self.map.get(&idx)
    }

    pub fn contains(&self, idx: u64) -> bool {
        self.map.contains_key(&idx)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    pub fn focus(&self) -> u64 {
        self.focus
    }

    /// Крайние ключи и объём кэша (диагностика вытеснения).
    pub fn extents(&self) -> (u64, u64, usize) {
        let mn = self.map.keys().next().copied().unwrap_or(0);
        let mx = self.map.keys().next_back().copied().unwrap_or(0);
        (mn, mx, self.bytes)
    }

    pub fn set_focus(&mut self, idx: u64) {
        self.focus = idx;
    }

    /// Клонирует кадр (для UI, чтобы не держать лок больше одного обращения).
    pub fn get_cloned(&self, idx: u64) -> Option<Frame> {
        self.map.get(&idx).cloned()
    }

    pub fn insert(&mut self, idx: u64, frame: Frame) {
        let size = frame.byte_size();
        self.rev += 1;
        match self.map.entry(idx) {
            std::collections::btree_map::Entry::Occupied(mut e) => {
                self.bytes -= e.get().byte_size();
                e.insert(frame);
                self.bytes += size;
            }
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(frame);
                self.bytes += size;
            }
        }
        self.evict_to_cap();
    }

    /// Вытесняет кадры дальше всех от focus, пока не уложимся в cap.
    fn evict_to_cap(&mut self) {
        while self.bytes > self.cap_bytes && self.map.len() > 1 {
            let first = *self.map.keys().next().unwrap();
            let last = *self.map.keys().next_back().unwrap();
            let f = self.focus as i64;
            let victim = if (first as i64 - f).abs() >= (last as i64 - f).abs() {
                first
            } else {
                last
            };
            if let Some(frame) = self.map.remove(&victim) {
                self.bytes -= frame.byte_size();
            }
        }
    }
}

/// Хэндл фонового декодера.
pub struct Player {
    tx: Sender<Msg>,
    cache: Arc<Mutex<FrameCache>>,
    errors: Arc<Mutex<Option<String>>>,
}

/// Сколько кадров читать вперёд от focus. Без лимита декодер (многократно
/// быстрее реального времени) убегает далеко вперёд; кэш при этом насыщен,
/// каждая дальняя вставка вытесняет сама себя — кадры будущего не сохраняются,
/// и playback упирается в отсутствующий следующий кадр каждые ~размер окна.
const READ_AHEAD: u64 = 8;

impl Player {
    /// Создаёт фоновый поток декодера. `fps` берётся из метаданных.
    pub fn start<B>(backend: B, path: PathBuf, meta: &VideoMetadata, cap_bytes: usize) -> Self
    where
        B: CodecBackend + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let cache = Arc::new(Mutex::new(FrameCache::new(cap_bytes)));
        let errors = Arc::new(Mutex::new(None::<String>));
        let fps = meta.fps.max(0.001);

        let cache_t = Arc::clone(&cache);
        let errors_t = Arc::clone(&errors);

        thread::spawn(move || {
            let mut reader: Option<Box<dyn FrameReader + Send>> = None;
            // Индекс СЛЕДУЮЩЕГО читаемого кадра (соответствует позиции ffmpeg).
            let mut pos = 0u64;
            // После достижения EOF тормозим, пока не прилетят новые команды.
            let mut at_eof = false;
            // Последняя заказанная позиция просмотра (куда вернуться после prefill).
            let mut last_target = 0u64;
            // Активная подкачка назад: сколько кадров осталось дочитать.
            let mut prefill_left: u32 = 0;
            let mut prefill_base: u64 = 0;
            let mut quit = false;
            #[allow(unused_mut)]
            let mut last_insert = std::time::Instant::now();

            loop {
                // 1. Сбор всех команд без блокировки: применяем ПО ПОРЯДКУ —
                // поздняя команда переопределяет раннюю.
                loop {
                    match rx.try_recv() {
                        Ok(Msg::Seek(i)) => {
                            cache_t.lock().unwrap().set_focus(i);
                            reader = None;
                            pos = i;
                            last_target = i;
                            prefill_left = 0; // прыжок отменяет подкачку назад
                            at_eof = false;
                        }
                        Ok(Msg::PrefillBackward { base, count }) => {
                            if std::env::var("VC_TRACE").is_ok() {
                                eprintln!("trace: prefill base={base} count={count}");
                            }
                            last_target = base;
                            prefill_left = count;
                            prefill_base = base;
                            // Цель подкачки — сразу после неё вернуться к base.
                            reader = None;
                            pos = base.saturating_sub(count as u64);
                            at_eof = false;
                        }
                        Ok(Msg::Quit) => {
                            quit = true;
                            break;
                        }
                        Err(_) => break,
                    }
                }
                if quit {
                    break;
                }

                // После EOF ничего нового не заказано — притормозим, не крутя цикл
                // и не переоткрывая ffmpeg впустую.
                if at_eof && prefill_left == 0 && reader.is_none() {
                    thread::sleep(Duration::from_millis(30));
                    continue;
                }

                // Read-ahead: не читать дальше READ_AHEAD кадров от focus.
                // Декодер многократно быстрее realtime; без лимита он убегает
                // далеко вперёд, и его вставки вытесняют сами себя в насыщенном
                // кэше — кадры «будущего» не сохраняются, playback спотыкается.
                // Пауза мягкая: reader остаётся открытым, команды (seek)
                // обрабатываются сразу же.
                if prefill_left == 0 && reader.is_some() {
                    let f = cache_t.lock().unwrap().focus();
                    if pos > f + READ_AHEAD {
                        thread::sleep(Duration::from_millis(8));
                        continue;
                    }
                }

                // 2. Открытие процесса, если не открыт.
                if reader.is_none() {
                    match backend.open(&path, Some(pos as f64 / fps)) {
                        Ok(r) => reader = Some(r),
                        Err(e) => {
                            *errors_t.lock().unwrap() =
                                Some(format!("ffmpeg перезапуск: {e}"));
                            thread::sleep(Duration::from_millis(120));
                            continue;
                        }
                    }
                }

                // 3. Кэш переполнен? Вытесняем; если не помогло — притормаживаем.
                {
                    let c = cache_t.lock().unwrap();
                    if c.bytes() > c.cap_bytes() {
                        // вставка уже вызывает evict; тут просто пауза-оттяжка,
                        // чтобы не глотать системные ресурсы впустую.
                        drop(c);
                        thread::sleep(Duration::from_millis(40));
                        continue;
                    }
                }

                // 4. Чтение следующего кадра.
                let r = reader.as_mut().unwrap();
                match r.read_frame() {
                    Ok(Some(frame)) => {
                        cache_t.lock().unwrap().insert(pos, frame);
                        if std::env::var("VC_TRACE").is_ok() {
                            let now = std::time::Instant::now();
                            let dt = now
                                .checked_duration_since(last_insert)
                                .map(|d| d.as_millis())
                                .unwrap_or(0);
                            last_insert = now;
                            eprintln!("trace: insert pos={pos} dt={dt}ms");
                        }
                        pos += 1;
                        if prefill_left > 0 && pos >= prefill_base {
                            // Подкачка назад завершена — вернуться к позиции просмотра.
                            prefill_left = 0;
                            reader = None;
                            pos = last_target;
                        }
                    }
                    Ok(None) => {
                        reader = None;
                        at_eof = true;
                        if std::env::var("VC_TRACE").is_ok() {
                            eprintln!("trace: eof at pos={pos}");
                        }
                    }
                    Err(e) => {
                        *errors_t.lock().unwrap() = Some(e.to_string());
                        reader = None;
                        thread::sleep(Duration::from_millis(120));
                    }
                }
            }
            // Дроп reader убьёт ffmpeg.
        });

        Self { tx, cache, errors }
    }

    pub fn seek(&self, idx: u64) {
        let _ = self.tx.send(Msg::Seek(idx));
    }

    /// Заказывает фоновую подкачку `count` кадров перед `base`
    /// (для восстановления фона), после чего декодер вернётся к `base`.
    pub fn prefill_backward(&self, base: u64, count: u32) {
        let _ = self.tx.send(Msg::PrefillBackward { base, count });
    }

    pub fn cache(&self) -> Arc<Mutex<FrameCache>> {
        Arc::clone(&self.cache)
    }

    /// Убирает последнее сообщение об ошибке фонового декодера (если было).
    pub fn take_error(&self) -> Option<String> {
        self.errors.lock().unwrap().take()
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Quit);
    }
}
