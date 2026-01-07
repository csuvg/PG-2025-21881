import os
import re
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_midis = repo_root / "midis"
    default_timestamps = repo_root / "timestamps.json"
    default_raw = repo_root / "audios" / "raw"
    default_seg = repo_root / "audios" / "seg"

    parser = argparse.ArgumentParser(
        description="Download full YouTube audio per MIDI and export per-clip MP3s using timestamps.json",
    )
    parser.add_argument("--mididir", type=Path, default=default_midis,
                        help="Directory containing MIDI files named like Q{label}_{ytid}_{idx}.mid")
    parser.add_argument("--timestamps", type=Path, default=default_timestamps,
                        help="Path to timestamps.json with {ytid: [[start, end, label], ...]}")
    parser.add_argument("--raw-dir", type=Path, default=default_raw,
                        help="Directory to cache downloaded raw full MP3s")
    parser.add_argument("--seg-dir", type=Path, default=default_seg,
                        help="Directory to write segmented per-clip MP3s")
    parser.add_argument("--yt-dlp", dest="ytdlp", default="yt-dlp",
                        help="yt-dlp executable path or name in PATH")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel download/clip jobs (1 = sequential)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading full MP3s if already present")
    parser.add_argument("--ffmpeg-location", dest="ffmpeg_location", type=Path, default=None,
                        help="Directory or full path to ffmpeg (and ffprobe). Passed through to yt-dlp and used for clipping.")
    return parser.parse_args()


def ensure_dirs(paths: list[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


_MIDI_NAME_RE = re.compile(r"^Q(?P<label>[1-4])_+(?P<ytid>[A-Za-z0-9_-]{6,})_(?P<idx>\d+)\.mid$")


def discover_midi_clips(mididir: Path) -> list[dict]:
    results: list[dict] = []
    if not mididir.exists():
        print(f"[ERROR] MIDI directory not found: {mididir}")
        return results
    for entry in mididir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        m = _MIDI_NAME_RE.match(name)
        if not m:
            # not a target midi naming; ignore
            continue
        label = int(m.group("label"))
        ytid = m.group("ytid")
        idx = int(m.group("idx"))
        results.append({
            "path": entry,
            "label": label,
            "ytid": ytid,
            "idx": idx,
        })
    return results


def load_timestamps(ts_path: Path) -> dict:
    if not ts_path.exists():
        print(f"[ERROR] timestamps.json not found: {ts_path}")
        return {}
    with ts_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_ytid_to_indices(midi_items: list[dict]) -> dict[str, set[int]]:
    mapping: dict[str, set[int]] = {}
    for it in midi_items:
        s = mapping.setdefault(it["ytid"], set())
        s.add(it["idx"])
    return mapping


def ytid_to_url(ytid: str) -> str:
    return f"https://www.youtube.com/watch?v={ytid}"


def download_full_audio(ytdlp: str, ytid: str, out_path: Path, ffmpeg_location: Path | None, metrics: dict) -> int:
    if out_path.exists():
        return 0
    url = ytid_to_url(ytid)
    # Base command to extract audio as mp3 at highest quality. Requires ffmpeg/ffprobe.
    base_cmd = [
        ytdlp,
        "--no-playlist",
        "--no-part",
        "--no-continue",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--force-overwrites",
        "-o", str(out_path),
    ]
    if ffmpeg_location is not None:
        base_cmd += ["--ffmpeg-location", str(ffmpeg_location)]

    # Try up to 2 attempts; on retry, clear cache to mitigate HTTP 416.
    attempts = 0
    while attempts < 2:
        attempts += 1
        cmd = base_cmd + [url]
        print(f"[DOWNLOAD] {ytid} -> {out_path.name} (attempt {attempts})")
        try:
            rc = subprocess.call(cmd)
        except FileNotFoundError:
            print("[ERROR] yt-dlp not found. Install from https://github.com/yt-dlp/yt-dlp")
            return 1

        if rc == 0:
            return 0

        # If failed, try cache clear and remove potential leftovers, then retry once
        print(f"[WARN] yt-dlp failed for {ytid} (rc={rc}). Preparing retry...")
        # Clear yt-dlp cache to avoid stale ranges (helps on 416)
        try:
            subprocess.call([ytdlp, "--rm-cache-dir"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        # Remove any existing target file or obvious partials
        try:
            if out_path.exists():
                out_path.unlink()
            # Clean common partials in raw dir for this ytid
            for cand in out_path.parent.glob(f"{ytid}.*.part"):
                try:
                    cand.unlink()
                except Exception:
                    pass
            # Also remove non-mp3 audio remnants (e.g., webm/m4a) that might block conversion
            for ext in ("webm", "m4a", "m4v", "mp4"):
                cand = out_path.parent / f"{ytid}.{ext}"
                if cand.exists():
                    try:
                        cand.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        # Track a specific 416 hint if present (best-effort; cannot inspect stderr easily without Popen)
        metrics["errors_416"] = metrics.get("errors_416", 0) + 1

    return rc


def normalize_hms(hms_str: str) -> str:
    parts = hms_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid H:M:S time string: {hms_str}")
    h, m, s = parts
    h = int(h)
    m = int(m)
    s = int(float(s))
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_seconds(hms_str: str) -> int:
    parts = hms_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid H:M:S time string: {hms_str}")
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + int(float(s))


def seconds_to_hms(total_seconds: int) -> str:
    h = total_seconds // 3600
    rem = total_seconds % 3600
    m = rem // 60
    s = rem % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def clip_with_ffmpeg(input_mp3: Path, start_hms: str, end_hms: str, output_mp3: Path, overwrite: bool, ffmpeg_bin: str) -> int:
    if output_mp3.exists() and not overwrite:
        return 0
    start = normalize_hms(start_hms)
    end = normalize_hms(end_hms)
    dur_seconds = hms_to_seconds(end) - hms_to_seconds(start)
    if dur_seconds <= 0:
        print(f"[WARN] Non-positive duration for {input_mp3.name}: {start} -> {end}")
        return 0
    dur = seconds_to_hms(dur_seconds)
    # Re-encode to avoid boundary issues
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", start,
        "-i", str(input_mp3),
        "-t", dur,
        "-ar", "44100",
        "-acodec", "libmp3lame",
        "-y" if overwrite else "-n",
        str(output_mp3),
    ]
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.call(cmd)


def resolve_ffmpeg_bin(ffmpeg_location: Path | None) -> str:
    if ffmpeg_location is None:
        return "ffmpeg"
    # If a directory is provided, append ffmpeg(.exe) depending on platform
    if ffmpeg_location.is_dir():
        candidate = ffmpeg_location / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        return str(candidate)
    # If a file is provided, assume it's the ffmpeg binary
    return str(ffmpeg_location)


def main() -> int:
    args = parse_args()

    ensure_dirs([args.raw_dir, args.seg_dir])

    start_time = time.time()
    metrics = {
        "downloads_attempted": 0,
        "downloads_succeeded": 0,
        "downloads_skipped": 0,
        "downloads_failed": 0,
        "errors_416": 0,
        "clips_attempted": 0,
        "clips_succeeded": 0,
        "clips_skipped_exists": 0,
        "clips_skipped_missing_audio": 0,
        "clips_failed": 0,
    }

    midi_items = discover_midi_clips(args.mididir)
    if not midi_items:
        print("[INFO] No MIDI clips discovered. Nothing to do.")
        return 0

    print(f"[*] Found {len(midi_items)} MIDI clips in {args.mididir}")
    stamps = load_timestamps(args.timestamps)
    if not stamps:
        print("[ERROR] Empty or missing timestamps.json; aborting.")
        return 1

    # Build set of YT IDs we actually need (based on present MIDIs)
    ytid_to_indices = build_ytid_to_indices(midi_items)
    print(f"[*] Unique YouTube IDs targeted: {len(ytid_to_indices)}")

    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_location)

    # Download audios
    for ytid in sorted(ytid_to_indices.keys()):
        raw_mp3 = args.raw_dir / f"{ytid}.mp3"
        if args.skip_download and raw_mp3.exists():
            metrics["downloads_skipped"] += 1
            continue
        metrics["downloads_attempted"] += 1
        rc = download_full_audio(args.ytdlp, ytid, raw_mp3, args.ffmpeg_location, metrics)
        if rc != 0:
            print(f"[ERROR] Failed to download {ytid}, skipping its clips.")
            metrics["downloads_failed"] += 1
            continue
        # Validate output; in some failure cases with missing ffmpeg, yt-dlp leaves source (e.g., webm)
        if raw_mp3.exists():
            metrics["downloads_succeeded"] += 1
        else:
            print(f"[WARN] Expected MP3 not found for {ytid}; attempting recovery")
            # If a source container exists, try converting manually
            fallback_src = None
            for ext in ("webm", "m4a", "mp4"):
                cand = args.raw_dir / f"{ytid}.{ext}"
                if cand.exists():
                    fallback_src = cand
                    break
            if fallback_src is not None:
                conv_cmd = [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel", "error",
                    "-i", str(fallback_src),
                    "-ar", "44100",
                    "-acodec", "libmp3lame",
                    "-y",
                    str(raw_mp3),
                ]
                rc2 = subprocess.call(conv_cmd)
                if rc2 == 0 and raw_mp3.exists():
                    metrics["downloads_succeeded"] += 1
                else:
                    metrics["downloads_failed"] += 1
            else:
                metrics["downloads_failed"] += 1

    # Clip per MIDI (index)
    total_written = 0
    for item in midi_items:
        ytid = item["ytid"]
        idx = item["idx"]
        raw_mp3 = args.raw_dir / f"{ytid}.mp3"
        if not raw_mp3.exists():
            print(f"[WARN] Raw audio missing for {ytid}, skip MIDI index {idx}")
            metrics["clips_skipped_missing_audio"] += 1
            continue
        if ytid not in stamps:
            print(f"[WARN] YouTube ID {ytid} not in timestamps.json, skipping")
            continue
        clip_list = stamps[ytid]
        if idx < 0 or idx >= len(clip_list):
            print(f"[WARN] Index {idx} out of range for {ytid} (clips={len(clip_list)})")
            continue

        start_hms, end_hms, q_label = clip_list[idx]
        out_name = f"Q{q_label}_{ytid}_{idx}.mp3"
        out_path = args.seg_dir / out_name
        if out_path.exists() and not args.overwrite:
            metrics["clips_skipped_exists"] += 1
            continue
        metrics["clips_attempted"] += 1
        rc = clip_with_ffmpeg(raw_mp3, start_hms, end_hms, out_path, overwrite=args.overwrite, ffmpeg_bin=ffmpeg_bin)
        if rc == 0 and out_path.exists():
            total_written += 1
            metrics["clips_succeeded"] += 1
        else:
            metrics["clips_failed"] += 1

    elapsed = time.time() - start_time
    metrics["elapsed_seconds"] = int(elapsed)
    print("[*] Done.")
    print(f"  - MIDI clips discovered: {len(midi_items)}")
    print(f"  - Unique YouTube IDs: {len(ytid_to_indices)}")
    print(f"  - Downloads: attempted={metrics['downloads_attempted']} succeeded={metrics['downloads_succeeded']} skipped={metrics['downloads_skipped']} failed={metrics['downloads_failed']} 416_hints={metrics['errors_416']}")
    print(f"  - Clips: attempted={metrics['clips_attempted']} succeeded={metrics['clips_succeeded']} skipped_exists={metrics['clips_skipped_exists']} skipped_missing_audio={metrics['clips_skipped_missing_audio']} failed={metrics['clips_failed']}")
    print(f"  - Elapsed: {metrics['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)


