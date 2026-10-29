#!/usr/bin/env python3
"""Сайдкар видео-инпейнтинга (ProPainter) для VideoCleaning.

Пайплайн (несколько сегментов):
  1. Для каждого сегмента (--segment START:COUNT:X:Y:W:H, повторяемый)
     читает кадры из исходного видео (cv2), нарезает кроп вокруг масок
     в crop_<i>.mp4 (размеры кратны 8) и маски
     (сырые u8 из masks-dir, глобальная нумерация <idx>.raw).
  2. Прогоняет ProPainter (repo/inference_propainter.py) по каждому сегменту
     отдельно; для Rust-поллера пишет файл состояния (--state: "k n" =
     завершено k сегментов из n).
  3. Один проход по всему видео: кадры сегментов собирают результат,
     остальные копируются без изменений; на выходе видео-only mp4
     всей длины (звук добавляет Rust-муксер).

Прогресс (stdout, для парсинга Rust):
  PROGRESS extract   <f01>   — глобально по всем сегментам (0..1)
  PROGRESS painter_run <k> <n> — завершён GPU-сегмент k из n
  PROGRESS paste     <f01>   — доля кадров финального кодирования
  PROGRESS done 1.0
Выход: 0 успешен; не-0 — ошибка (stderr).

Запуск:
  inpaint_cli.py --video SRC.mp4 --masks-dir DIR --out OUT.mp4
      --segment 12:120:96:200:800:400 --segment 1000:80:... \
      --mask-wh 1920 1080 [--fps 23.976] [--crf 18] [--state F]
      [--work DIR] [--fp16] [--subvideo-length 80]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager


def ffmpeg():
    # imageio-ffmpeg даёт статический ffmpeg; иначе из PATH.
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


@contextmanager
def open_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"не открыть видео: {path}")
    try:
        yield _Reader(cap, path)
    finally:
        cap.release()


class _Reader:
    """Последовательное чтение с кэшем позиции: обычный cap.read(), когда
    идём подряд; CAP_PROP_POS_FRAMES — только при отскоке."""

    def __init__(self, cap, path):
        self.cap = cap
        self.next_idx = 0  # индекс следующего кадра, который отдаст read()

    def read(self, idx):
        if idx == self.next_idx:
            ok, f = self.cap.read()
            if not ok:
                return None
            self.next_idx += 1
            return f
        import cv2
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
            raise RuntimeError(f"не позиционироваться на кадр {idx}")
        ok, f = self.cap.read()
        if not ok:
            return None
        self.next_idx = idx + 1
        return f


def seek_and_read(reader, idx):
    """Читает кадр idx через _Reader (последовательно или отскоком)."""
    return reader.read(idx)


class Writer:
    """Кодирует BGR-кадры (rawvideo-пайп) в mp4 libx264."""

    def __init__(self, w, h, fps, crf, out_path):
        self.proc = subprocess.Popen(
            [ffmpeg(), "-v", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
             "-an", "-c:v", "libx264", "-preset", "medium",
             "-crf", str(crf), "-pix_fmt", "yuv420p", out_path],
            stdin=subprocess.PIPE,
        )
        self.w, self.h = w, h

    def write(self, bgr):
        if bgr.shape[1] != self.w or bgr.shape[0] != self.h:
            raise RuntimeError(f"кадр {bgr.shape[1]}x{bgr.shape[0]} != {self.w}x{self.h}")
        self.proc.stdin.write(bgr.tobytes())

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg-энкодер вернул {self.proc.returncode}")


def progress(tag, value):
    print(f"PROGRESS {tag} {min(max(value, 0.0), 1.0):.4f}", flush=True)


def write_state(path, k, n):
    try:
        with open(path, "w") as f:
            f.write(f"{k} {n}")
    except Exception:
        pass


def parse_segments(items):
    segs = []
    for it in items:
        parts = it.split(":")
        if len(parts) != 6:
            raise RuntimeError(f"--segment формат START:COUNT:X:Y:W:H, получено: {it}")
        s, c, x, y, w, h = (int(p) for p in parts)
        if c <= 0 or w <= 0 or h <= 0:
            raise RuntimeError(f"--segment: пустой сегмент {it}")
        segs.append({"start": s, "count": c, "x": x, "y": y, "w": w, "h": h})
    if not segs:
        raise RuntimeError("нужен хотя бы один --segment")
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--segment", action="append", required=True,
                    metavar="START:COUNT:X:Y:W:H")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--mask-wh", type=int, nargs=2, metavar=("W", "H"), required=True)
    ap.add_argument("--state", default=None,
                    help="файл состояния GPU-фазы для Rust-поллера ('k n')")
    ap.add_argument("--work", default=None)
    ap.add_argument("--repo", default=os.environ.get(
        "VC_PROPAINTER_REPO",
        os.path.join(os.path.dirname(__file__), "..", "propainter-repo")))
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--subvideo-length", type=int, default=80)
    args = ap.parse_args()

    import cv2
    import numpy as np

    segs = parse_segments(args.segment)
    mw, mh = args.mask_wh
    fps = args.fps
    if fps <= 0:
        cap0 = cv2.VideoCapture(args.video)
        fps = cap0.get(cv2.CAP_PROP_FPS) or 25.0
        cap0.release()
    work = args.work or tempfile.mkdtemp(prefix="vc_paint_")
    os.makedirs(work, exist_ok=True)
    state_path = args.state or os.path.join(work, "painter_state")

    total_extract = sum(s["count"] for s in segs)

    # Ограничение разрешения обработки: большой кроп не влезает в VRAM
    # (conv3d/attention растут ~ W×H). Если превышен — считаем кроп
    # заниженным, а при вставке результат растягивается обратно.
    PROC_LIMIT = 1024  # максимум по длинной стороне кропа
    for s in segs:
        m = max(s["w"], s["h"])
        scale = min(1.0, PROC_LIMIT / m) if m > 0 else 1.0
        s["proc_w"] = max(16, (int(s["w"] * scale) // 8) * 8)
        s["proc_h"] = max(16, (int(s["h"] * scale) // 8) * 8)
        if s["proc_w"] != s["w"] or s["proc_h"] != s["h"]:
            print(f"SW тучный кроп {s['w']}x{s['h']} -> {s['proc_w']}x{s['proc_h']}",
                  flush=True)

    # ---------------- 1/3: кроп видео и масок (по сегментам) ----------------
    progress("extract", 0.0)
    extracted = 0
    with open_video(args.video) as vid_r:
        for si, seg in enumerate(segs):
            x, y = seg["x"], seg["y"]
            # Кратность 8 для ProPainter (Rust уже присылает кратные, но
            # страховка не помешает; нарезка масок согласована ниже).
            crop_w = seg["w"] - seg["w"] % 8
            crop_h = seg["h"] - seg["h"] % 8
            proc_w, proc_h = seg["proc_w"], seg["proc_h"]
            crop_path = os.path.join(work, f"crop_{si}.mp4")
            writer = Writer(proc_w, proc_h, fps, 16, crop_path)
            for local in range(seg["count"]):
                frame = seek_and_read(vid_r, seg["start"] + local)
                if frame is None:
                    raise RuntimeError(f"нет кадра {seg['start'] + local} в видео")
                c = frame[y:y + crop_h, x:x + crop_w]
                if proc_w != crop_w or proc_h != crop_h:
                    c = cv2.resize(c, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                writer.write(c)
                extracted += 1
                if extracted % 20 == 0:
                    progress("extract", extracted / max(1, total_extract))
            writer.close()

            # Маски: global-numbered .raw (u8 0/255), нарезаем кропом.
            crop_mask_dir = os.path.join(work, f"crop_masks_{si}")
            os.makedirs(crop_mask_dir, exist_ok=True)
            for local in range(seg["count"]):
                g = seg["start"] + local
                raw = os.path.join(args.masks_dir, f"{g:09d}.raw")
                if os.path.exists(raw):
                    m = np.fromfile(raw, np.uint8).reshape(mh, mw)
                else:
                    m = np.zeros((mh, mw), np.uint8)
                m = m[y:y + crop_h, x:x + crop_w]
                if proc_w != crop_w or proc_h != crop_h:
                    m = cv2.resize(m, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
                # ProPainter хочет PNG 0/255
                cv2.imwrite(os.path.join(crop_mask_dir, f"{local:06d}.png"),
                            (m > 127).astype(np.uint8) * 255)
            seg["crop_w"], seg["crop_h"] = crop_w, crop_h
    progress("extract", 1.0)

    # ---------------- 2/3: ProPainter по сегментам ----------------
    repo = os.path.abspath(args.repo)
    script = os.path.join(repo, "inference_propainter.py")
    if not os.path.exists(script):
        raise RuntimeError(f"нет {script}")
    env = dict(os.environ)
    # ffmpeg для imageio внутри ProPainter
    env.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg())
    # Меньше фрагментации VRAM → дольше живёт повторный прогон сегментов.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    n = len(segs)
    write_state(state_path, 0, n)
    frames_dirs = []
    for si, seg in enumerate(segs):
        write_state(state_path, si, n)
        out_dir = os.path.join(work, f"pp_{si}")
        cmd = [sys.executable, script,
               "--video", os.path.join(work, f"crop_{si}.mp4"),
               "--mask", os.path.join(work, f"crop_masks_{si}"),
               "--output", out_dir,
               "--mode", "video_inpainting",
               "--save_frames",
               "--subvideo_length", str(args.subvideo_length)]
        # fp16 на GPU: вдвое меньше VRAM и быстрее; на CPU inference сам
        # это игнорирует (use_half=False при device=cpu).
        cmd.append("--fp16")
        print(f"RUN сегмент {si + 1}/{n}: кадры {seg['start']}…"
              f"{seg['start'] + seg['count'] - 1}, "
              f"crop={seg['crop_w']}x{seg['crop_h']}", flush=True)
        proc = subprocess.run(cmd, cwd=repo, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"ProPainter (сегмент {si + 1}/{n}) вернул {proc.returncode}")
        frames_dir = os.path.join(out_dir, f"crop_{si}", "frames")
        if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
            raise RuntimeError(f"ProPainter не отдал кадры в {frames_dir}")
        if len(os.listdir(frames_dir)) != seg["count"]:
            raise RuntimeError(
                f"ProPainter отдал {len(os.listdir(frames_dir))} кадров, "
                f"ждали {seg['count']} (сегмент {si + 1}/{n})")
        frames_dirs.append(frames_dir)
        print(f"PROGRESS painter_run {si + 1} {n}", flush=True)
    write_state(state_path, n, n)

    # ---------------- 3/3: вставка + кодирование (один проход) ----------------
    # ПИШЕМ ВЕСЬ РОЛИК: кадры сегментов собирают результат, остальные
    # копируются без изменений (иначе видео короче аудио — плееры
    # показывают статичный кадр до конца).
    ranges = [
        {"lo": s["start"], "hi": s["start"] + s["count"],
         "x": s["x"], "y": s["y"],
         "cw": s["crop_w"], "ch": s["crop_h"], "dir": fd, "names": sorted(os.listdir(fd))}
        for s, fd in zip(segs, frames_dirs)
    ]
    writer = Writer(mw, mh, fps, args.crf, args.out)
    # Всего кадров в исходнике (для честного прогресса paste).
    cap0 = cv2.VideoCapture(args.video)
    total_hint = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    written = 0
    with open_video(args.video) as vid_r:
        gf = 0
        while True:
            frame = seek_and_read(vid_r, gf)
            if frame is None:
                break
            for r in ranges:
                if r["lo"] <= gf < r["hi"]:
                    res = cv2.imread(os.path.join(r["dir"], r["names"][gf - r["lo"]]))
                    if res is None:
                        raise RuntimeError(f"не читается результат {r['names'][gf - r['lo']]}")
                    if res.shape[1] != r["cw"] or res.shape[0] != r["ch"]:
                        res = cv2.resize(res, (r["cw"], r["ch"]))
                    frame[r["y"]:r["y"] + r["ch"], r["x"]:r["x"] + r["cw"]] = res
            writer.write(frame)
            gf += 1
            written += 1
            if gf % 25 == 0 and total_hint > 0:
                progress("paste", gf / total_hint)
    writer.close()
    for r in ranges:
        if written < r["hi"]:
            raise RuntimeError(
                f"видео кончилось на {written} кадре, обрабатывали до {r['hi']}")
    print("PROGRESS done 1.0", flush=True)
    # work тяжёлый; удаляем сам, если не режим отладки.
    if os.environ.get("VC_PROPAINTER_KEEP_WORK") != "1":
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR {e}", file=sys.stderr, flush=True)
        sys.exit(1)
