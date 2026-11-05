#!/usr/bin/env python3
"""
DJ set transition detector.

Detects likely track-transition points in a long DJ mix by looking for
significant changes in spectral content, tempo, and energy over time.

Usage:
    python3 detect_transitions.py [path/to/set.aiff] [num_tracks]

Notes:
- This does NOT identify songs. It just flags timestamps where the audio
  changes enough that a transition probably happened. You match these
  timestamps to your tracklist order by ear.
- Works on long files by streaming in blocks (won't load 900MB into RAM).
- Output: a list of candidate transition timestamps (mm:ss) plus a rough
  confidence score, written to transitions.txt
"""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy as np
import librosa
import soundfile as sf

# ---- Tunable parameters ----
SR = 22050              # downsample rate for analysis (speed)
HOP_LENGTH = 512
FRAME_SECONDS = 30      # analyze in 30s blocks for memory efficiency
OVERLAP_SECONDS = 5     # overlap between blocks so we don't miss transitions at edges
MIN_GAP_BETWEEN_TRANSITIONS = 30  # seconds; merge nearby detections

def load_block(path, offset_sec, duration_sec, sr=SR):
    y, _ = librosa.load(path, sr=sr, offset=offset_sec, duration=duration_sec, mono=True)
    return y

def get_total_duration(path):
    return librosa.get_duration(path=path)

def prepare_audio(path):
    """Return (path_to_read, temp_dir_or_None).

    Formats libsndfile can read (wav, aiff, flac...) are used as-is. Anything
    else (m4a, mp3, ...) is decoded once with ffmpeg to a mono WAV at SR, since
    the audioread fallback re-decodes from the start for every block (very slow).
    """
    try:
        sf.info(path)
        return path, None
    except Exception:
        pass

    if shutil.which("ffmpeg") is None:
        print("This format needs ffmpeg for fast decoding (brew install ffmpeg).")
        print("Falling back to slow decoding...")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        return path, None

    tmp_dir = tempfile.mkdtemp(prefix="dj_transitions_")
    wav_path = os.path.join(tmp_dir, "decoded.wav")
    print("Decoding to temporary WAV with ffmpeg ...")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), wav_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"ffmpeg failed to decode the file:\n{result.stderr}")
        sys.exit(1)
    return wav_path, tmp_dir

def analyze(path):
    total_dur = get_total_duration(path)
    print(f"Total duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")

    times = []
    chroma_flux = []
    spectral_flux = []
    rms_vals = []
    tempo_vals = []

    step = FRAME_SECONDS - OVERLAP_SECONDS
    offset = 0.0

    prev_chroma_mean = None
    prev_spec_mean = None

    while offset < total_dur:
        dur = min(FRAME_SECONDS, total_dur - offset)
        y = load_block(path, offset, dur)

        if len(y) < SR * 2:  # skip tiny trailing blocks
            break

        # Chroma (harmonic/key content)
        chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_LENGTH)
        chroma_mean = chroma.mean(axis=1)

        # Spectral centroid + bandwidth (timbre/energy distribution)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LENGTH)
        spec_mean = spec_cent.mean()

        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH).mean()

        # Tempo estimate
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP_LENGTH)
            tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
        except Exception:
            tempo_val = np.nan

        times.append(offset)
        rms_vals.append(rms)
        tempo_vals.append(tempo_val)

        if prev_chroma_mean is not None:
            c_flux = np.linalg.norm(chroma_mean - prev_chroma_mean)
            s_flux = abs(spec_mean - prev_spec_mean)
        else:
            c_flux = 0.0
            s_flux = 0.0

        chroma_flux.append(c_flux)
        spectral_flux.append(s_flux)

        prev_chroma_mean = chroma_mean
        prev_spec_mean = spec_mean

        if dur < FRAME_SECONDS:
            break
        offset += step

    return {
        "times": np.array(times),
        "chroma_flux": np.array(chroma_flux),
        "spectral_flux": np.array(spectral_flux),
        "rms": np.array(rms_vals),
        "tempo": np.array(tempo_vals),
    }

def find_transitions(data, num_tracks=None):
    times = data["times"]
    chroma_flux = data["chroma_flux"]
    spectral_flux = data["spectral_flux"]
    rms = data["rms"]
    tempo = data["tempo"]

    # Normalize each signal 0-1
    def norm(x):
        x = np.nan_to_num(x)
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else x

    c_flux_n = norm(chroma_flux)
    s_flux_n = norm(spectral_flux)

    # Tempo change (abs diff between consecutive blocks)
    tempo_diff = np.abs(np.diff(np.nan_to_num(tempo)))
    tempo_diff = np.insert(tempo_diff, 0, 0)
    tempo_diff_n = norm(tempo_diff)

    # RMS dips often happen at transitions (mixdowns, breakdowns)
    rms_n = norm(rms)
    rms_dip = norm(-np.gradient(rms_n))  # negative slope = energy dropping

    # Combined score
    score = (
        0.35 * c_flux_n +
        0.25 * s_flux_n +
        0.25 * tempo_diff_n +
        0.15 * rms_dip
    )

    if num_tracks:
        # Known track count: take the (num_tracks - 1) strongest local peaks,
        # keeping them at least MIN_GAP_BETWEEN_TRANSITIONS apart.
        peaks = [(times[i], score[i]) for i in range(1, len(score) - 1)
                 if score[i] >= score[i-1] and score[i] >= score[i+1]]
        chosen = []
        for t, sc in sorted(peaks, key=lambda x: -x[1]):
            if len(chosen) >= num_tracks - 1:
                break
            if all(abs(t - ct) >= MIN_GAP_BETWEEN_TRANSITIONS for ct, _ in chosen):
                chosen.append((t, sc))
        return sorted(chosen), score, times

    # Find local peaks above threshold
    threshold = np.mean(score) + 1.0 * np.std(score)
    candidates = []
    for i in range(1, len(score) - 1):
        if score[i] > threshold and score[i] >= score[i-1] and score[i] >= score[i+1]:
            candidates.append((times[i], score[i]))

    # Merge candidates that are too close together (keep highest score)
    merged = []
    for t, s in sorted(candidates, key=lambda x: x[0]):
        if merged and t - merged[-1][0] < MIN_GAP_BETWEEN_TRANSITIONS:
            if s > merged[-1][1]:
                merged[-1] = (t, s)
        else:
            merged.append((t, s))

    return merged, score, times

def fmt_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = m // 60
    m = m % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = input("Enter the path to your DJ set audio file: ").strip().strip("'\"")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        num_tracks = int(sys.argv[2])
    else:
        answer = input("How many tracks are in the set? (press Enter if unknown): ").strip()
        num_tracks = int(answer) if answer.isdigit() else None

    print(f"Analyzing {path} ... (this may take a while for long files)")
    read_path, tmp_dir = prepare_audio(path)
    try:
        data = analyze(read_path)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    transitions, score, times = find_transitions(data, num_tracks)

    print(f"\nFound {len(transitions)} candidate transition points:\n")
    out_lines = []
    out_lines.append("# Candidate transition points (approximate)")
    out_lines.append("# Format: timestamp  |  confidence (0-1, higher = more likely a real transition)")
    out_lines.append("")
    for t, s in transitions:
        line = f"{fmt_time(t)}  |  confidence: {s:.2f}"
        print(line)
        out_lines.append(line)

    with open("transitions.txt", "w") as f:
        f.write("\n".join(out_lines))

    print("\nSaved to transitions.txt")
    print("\nNext step: listen around each timestamp (+/- 10-15s) and match")
    print("to your tracklist in order to label which song starts there.")

if __name__ == "__main__":
    main()
