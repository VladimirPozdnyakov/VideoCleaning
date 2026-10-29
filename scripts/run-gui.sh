#!/usr/bin/env bash
# Запуск GUI в отдельном systemd scope c изоляцией памяти.
#
# Зачем: systemd-oomd при пике (RAM+zram > 90%) убивает самый «толстый»
# cgroup целиком. Если приложение запущено из терминала VSCode, оно живёт
# в cgroup VSCode — и oomd прибивает VSCode (инциденты 01:06/01:14).
# Отдельный scope с лимитами гарантирует: в худшем случае умрёт сам
# видеоредактор, VSCode и остальные не пострадают.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./target/release/videocleaning
[ -x "$BIN" ] || { echo "нет $BIN — сначала cargo build -p vc-gui --release" >&2; exit 1; }

systemctl --user reset-failed videocleaning.scope 2>/dev/null || true
exec systemd-run --user --scope \
    --unit=videocleaning \
    -p MemoryHigh=6G \
    -p MemoryMax=10G \
    "$BIN" "$@"
