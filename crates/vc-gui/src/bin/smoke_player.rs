//! Smoke-тест M1: Player-переход и prefetch в фоне, без GUI.
//!
//! Запуск: `cargo run -p vc-gui --bin smoke_player -- /tmp/opencode/vctest/test.mp4`

use std::path::Path;
use std::time::{Duration, Instant};

use vc_codec::{CodecBackend, FfmpegBackend, Player};

fn wait_for(player: &Player, idx: u64, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if player.cache().lock().unwrap().contains(idx) {
            return true;
        }
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/opencode/vctest/test.mp4".into());
    let backend = FfmpegBackend::default();
    let meta = backend.probe(Path::new(&path)).expect("probe");
    println!("probe ok: {}×{} @ {:.2} fps", meta.width, meta.height, meta.fps);

    let player = Player::start(FfmpegBackend::default(), path.into(), &meta, 64 * 1024 * 1024);

    // 1. Стрим с 0 должен сам добраться до constexpr EOF (50 кадров).
    assert!(wait_for(&player, 10, Duration::from_secs(5)), "нет кадра 10 после старта");
    println!("кадр 10 в кэше сразу после старта ✓");

    // 2. Jumpy seek вперёд.
    player.seek(38);
    assert!(wait_for(&player, 38, Duration::from_secs(5)), "нет кадра 38 после seek");
    println!("кадр 38 в кэше после seek ✓");

    // 3. Прыжок назад.
    player.seek(3);
    assert!(wait_for(&player, 3, Duration::from_secs(5)), "нет кадра 3 после обратного seek");
    println!("кадр 3 в кэше после обратного seek ✓");

    // 4. Финальный предпоследний кадр (EOF ещё не прошёл наверняка).
    player.seek(49);
    assert!(wait_for(&player, 48, Duration::from_secs(5)), "нет кадра 48 в самом конце");
    println!("кадр 48 в кэше (близко к концу) ✓");

    // 5. Backward-prefill: закинуть кадры перед текущей позицией.
    player.prefill_backward(40, 10);
    assert!(wait_for(&player, 33, Duration::from_secs(5)), "нет кадра 33 после prefill_backward(40,10)");
    println!("backward-prefill: кадр 33 (перед базой 40) в кэше ✓");
    // После подкачки поток вернулся к базе и продолжает вперёд.
    assert!(wait_for(&player, 44, Duration::from_secs(5)), "нет кадра 44 после возврата к базе");
    println!("после подкачки стрим вернулся к базе и продолжил ✓");

    let cache_arc = player.cache();
    let cache = cache_arc.lock().unwrap();
    println!(
        "итог: в кэше ~{} байт (лимит {}) для файла на 50 кадров",
        cache.bytes(),
        cache.cap_bytes()
    );
    println!("SMOKE PLAYER OK");
}
