#!/usr/bin/env python3
"""
Label a DJ set recording with its tracklist.

Reads the tracklist CSV exported from djay (Title, Artist, Time, BPM, Key,
Start Time in UTC, ...) and converts each track's Start Time into a timestamp
inside the recording.

Usage:
    python3 label_tracks.py [path/to/set.m4a] [path/to/tracklist.csv] [--refine] [--first-track "Title|Artist"]

By default the CSV times are used as-is. With --refine, each one is also
nudged (+/- 30s) toward the biggest audio change nearby.

The recording start time is read from the filename djay produces
("Recording 9-29-26, 8.55.00 PM.m4a"). If that fails you'll be asked for it.

If the first CSV track starts well after the recording does, a row is added
at 00:00: "Unknown track (not in CSV)", or the track given with --first-track.

Output:
    tracklist.txt        detailed list (BPM, key, how each time was derived)
    tracklist.csv        Number, Timestamp, Title, Artist, BPM, Key
    tracklist_share.txt  clean "timestamp  Artist - Title" list for pasting into notes
"""

import csv
import datetime as dt
import glob
import os
import re
import shutil
import sys

import librosa
import numpy as np

from detect_transitions import HOP_LENGTH, SR, fmt_time, prepare_audio

SEARCH_SECONDS = 30       # how far from the CSV time to look for the audible mix point
WINDOW_SECONDS = 6        # audio compared before/after each candidate point
STEP_SECONDS = 0.5        # candidate spacing
MIN_PROMINENCE = 1.5      # peak novelty / median novelty needed to trust the refinement
UNLOGGED_GAP_SECONDS = 30  # gap before the first CSV track that suggests an unlogged track


def parse_recording_start(path):
    """Parse 'Recording 9-29-26, 8.55.00 PM' (local time) into an aware UTC datetime."""
    m = re.search(r"(\d+)-(\d+)-(\d+),\s*(\d+)\.(\d+)\.(\d+)\s*([AP]M)", os.path.basename(path))
    if not m:
        return None
    mo, d, y, h, mi, s, ampm = m.groups()
    local = dt.datetime.strptime(f"{mo}-{d}-{y} {h}:{mi}:{s} {ampm}", "%m-%d-%y %I:%M:%S %p")
    return local.astimezone(dt.timezone.utc)


def ask_recording_start():
    answer = input("Couldn't read the start time from the filename. "
                   "Enter it as local 'YYYY-MM-DD HH:MM:SS' (24h): ").strip()
    local = dt.datetime.strptime(answer, "%Y-%m-%d %H:%M:%S")
    return local.astimezone(dt.timezone.utc)


def read_tracks(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_start"] = dt.datetime.strptime(r["Start Time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return sorted(rows, key=lambda r: r["_start"])


def find_csv():
    here = os.path.dirname(os.path.abspath(__file__))
    matches = sorted(m for m in glob.glob(os.path.join(here, "*.csv"))
                     if os.path.basename(m) != "tracklist.csv")  # skip our own output
    if len(matches) == 1:
        return matches[0]
    return None


def refine(wav_path, t_csv, total_dur):
    """Return (refined_time, prominence). Falls back to t_csv if no clear peak."""
    lo = max(0.0, t_csv - SEARCH_SECONDS - WINDOW_SECONDS)
    hi = min(total_dur, t_csv + SEARCH_SECONDS + WINDOW_SECONDS)
    if hi - lo < 2 * WINDOW_SECONDS + 2:
        return t_csv, 0.0

    y, _ = librosa.load(wav_path, sr=SR, offset=lo, duration=hi - lo, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_LENGTH)
    cent = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LENGTH)
    cent = cent / (cent.max() + 1e-9)
    feats = np.vstack([chroma, cent])

    fps = SR / HOP_LENGTH
    w = int(WINDOW_SECONDS * fps)
    cands, novelty = [], []
    t = max(lo + WINDOW_SECONDS, t_csv - SEARCH_SECONDS)
    end = min(hi - WINDOW_SECONDS, t_csv + SEARCH_SECONDS)
    while t <= end:
        c = int((t - lo) * fps)
        left = feats[:, c - w:c].mean(axis=1)
        right = feats[:, c:c + w].mean(axis=1)
        cands.append(t)
        novelty.append(np.linalg.norm(left - right))
        t += STEP_SECONDS

    if not cands:
        return t_csv, 0.0
    novelty = np.array(novelty)
    best = int(np.argmax(novelty))
    prominence = novelty[best] / (np.median(novelty) + 1e-9)
    if prominence < MIN_PROMINENCE:
        return t_csv, prominence
    return cands[best], prominence


def share_title(entry):
    """Title for public lists: unlogged track becomes 'ID?'; leading track numbers, remaster tags and trailing BPM tags are dropped."""
    if entry["csv"] is None:
        return "ID?"
    title = re.sub(r"^\d{1,2}\s+", "", entry["title"])
    title = re.sub(r"\s*[-(\[]\s*(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?\s*[)\]]?\s*$", "", title, flags=re.I)
    return re.sub(r"\s+\d{2,3}$", "", title)  # trailing BPM tag, e.g. "... (Sickmix Intro) 127"


def main():
    argv = sys.argv[1:]
    first_track = None
    if "--first-track" in argv:
        i = argv.index("--first-track")
        first_track = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    args = [a for a in argv if a != "--refine"]
    do_refine = "--refine" in argv

    audio = args[0] if len(args) >= 1 else \
        input("Enter the path to your DJ set audio file: ").strip().strip("'\"")
    audio = os.path.expanduser(audio)
    if not os.path.isfile(audio):
        print(f"File not found: {audio}")
        sys.exit(1)

    csv_path = args[1] if len(args) >= 2 else find_csv()
    if not csv_path:
        csv_path = input("Enter the path to the tracklist CSV: ").strip().strip("'\"")
    csv_path = os.path.expanduser(csv_path)
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    rec_start = parse_recording_start(audio) or ask_recording_start()
    tracks = read_tracks(csv_path)

    wav_path, tmp_dir = prepare_audio(audio)
    try:
        total_dur = librosa.get_duration(path=wav_path)
        print(f"Recording: {total_dur / 60:.1f} min, {len(tracks)} tracks in CSV\n")

        entries = []
        first_offset = (tracks[0]["_start"] - rec_start).total_seconds()
        if first_offset > UNLOGGED_GAP_SECONDS:
            note = f"plays 00:00 to ~{fmt_time(first_offset)}"
            if first_track:
                title, _, artist = first_track.partition("|")
                entries.append({"time": 0.0, "csv": 0.0, "title": title.strip(), "artist": artist.strip(),
                                "meta": "", "note": f"added manually, {note}"})
            else:
                entries.append({"time": 0.0, "csv": None, "title": "Unknown track (not in CSV)",
                                "artist": "", "meta": "", "note": note})

        for r in tracks:
            t_csv = (r["_start"] - rec_start).total_seconds()
            if not 0 <= t_csv <= total_dur:
                print(f"Skipping '{r['Title']}': start time falls outside the recording")
                continue
            t_ref = t_csv
            note = "CSV time"
            if do_refine:
                t_ref, _ = refine(wav_path, t_csv, total_dur)
                note = f"CSV {fmt_time(t_csv)}, " + (
                    f"refined {t_ref - t_csv:+.0f}s" if t_ref != t_csv else "unrefined")
            entries.append({"time": t_ref, "csv": t_csv, "title": r["Title"], "artist": r["Artist"],
                            "meta": f"{r['BPM']} BPM, {r['Key']}", "note": note})
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    lines = ["# Tracklist with timestamps in the recording",
             f"# Source: {os.path.basename(csv_path)}", ""]
    for i, e in enumerate(entries, 1):
        who = f" - {e['artist']}" if e["artist"] else ""
        meta = f" ({e['meta']})" if e["meta"] else ""
        lines.append(f"{i:2d}. {fmt_time(e['time'])}  {e['title']}{who}{meta}  [{e['note']}]")
    print("\n".join(lines))

    with open("tracklist.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open("tracklist.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Number", "Timestamp", "Title", "Artist", "BPM", "Key"])
        for i, e in enumerate(entries, 1):
            bpm, key = (e["meta"].replace(" BPM", "").split(", ") + ["", ""])[:2] if e["meta"] else ("", "")
            w.writerow([i, fmt_time(e["time"]), share_title(e), e["artist"], bpm, key])

    with open("tracklist_share.txt", "w", encoding="utf-8") as f:
        for e in entries:
            who = f"{e['artist']} - " if e["artist"] else ""
            f.write(f"{fmt_time(e['time'])}  {who}{share_title(e)}\n")

    print("\nSaved to tracklist.txt, tracklist.csv, tracklist_share.txt")


if __name__ == "__main__":
    main()
