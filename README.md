# dj-tools

Two small Python scripts for working with recordings of DJ sets:

- **`detect_transitions.py`** finds likely track-change points in a mix by analyzing the audio. It flags *when* a transition probably happened; it does not identify songs.
- **`label_tracks.py`** converts a tracklist exported from djay into timestamps for your recording, and writes lists you can paste into notes or YouTube/SoundCloud chapters.

Use the tracklist tool when you have a CSV. Use the transition detector when you don't.

## Features

- Detects transition candidates in long mixes (AIFF, WAV, and FLAC are read directly; M4A and MP3 are decoded with ffmpeg).
- Streams audio in 30-second blocks, so large files don't need to fit in memory.
- Optionally takes the known track count and returns the strongest transition points.
- Labels a djay tracklist CSV with timestamps and handles an unlogged opening track.
- Cleans titles for public lists by removing track numbers, remaster tags, and trailing BPM tags.

## Requirements

- Python 3.9+ (developed on Python 3.13)
- Python packages: `librosa`, `numpy`, `scipy`, `soundfile`, `audioread`
- [ffmpeg](https://ffmpeg.org/) (optional, but strongly recommended for M4A and MP3). On macOS: `brew install ffmpeg`. Without it, compressed files still work but decoding is very slow.

## Installation

```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run both scripts from inside the activated environment. Output files are written to the directory you run the script from, not to the project folder.

## Detecting transitions

```bash
python3 detect_transitions.py path/to/set.aiff [num_tracks]
```

Both arguments are optional. If you leave them out, the script prompts for them. Dragging a file from Finder into the terminal pastes its path, with quotes and `~` handled.

If you give the number of tracks, the script returns up to `num_tracks - 1` of the strongest peaks. Otherwise it uses a statistical threshold to pick candidates.

Results are printed and saved to `transitions.txt`:

```
# Candidate transition points (approximate)
# Format: timestamp  |  confidence (0-1, higher = more likely a real transition)

10:50  |  confidence: 0.71
14:35  |  confidence: 0.64
```

**Next step:** listen 10–15 seconds on either side of each timestamp and match them to your tracklist in order.

### How it works

Each 30-second block (with 5 seconds of overlap) is reduced to four measurements:

| Signal | What it captures |
|---|---|
| Chroma change | Harmonic/key content |
| Spectral centroid change | Timbre and brightness |
| Tempo change | BPM shifts between blocks |
| Energy dip | Volume falling, common during mixes and breakdowns |

These are normalized and combined into one score (weights 0.35, 0.25, 0.25, and 0.15). Peaks closer than 30 seconds are merged, keeping the highest-scoring one.

### Limitations

- Timestamps are approximate. Block resolution is about 25 seconds, so always verify by ear.
- Long-form mixes, blends, and tracks with similar keys and tempos can produce false positives or miss transitions.
- Tempo estimates on mixed audio are rough.
- Long files take a while to analyze, since the whole recording is processed.

## Labeling a set from a djay tracklist

Export the tracklist CSV from djay. It must include these columns: `Title`, `Artist`, `BPM`, `Key`, and `Start Time`.

```bash
python3 label_tracks.py path/to/set.m4a path/to/tracklist.csv
```

If the folder contains exactly one CSV file (other than the `tracklist.csv` this tool writes), you can omit the second argument.

| Option | Effect |
|---|---|
| `--refine` | Nudges each timestamp by up to ±30 seconds toward the nearest audio change. Off by default because it did not clearly improve results in testing. |
| `--first-track "Title\|Artist"` | Names the track that plays before the first CSV entry. |

The recording's start time is read from djay's filename, such as `Recording 9-29-26, 8.55.00 PM.m4a`, in local time. If the filename doesn't match, the script asks for it.

If the first CSV track starts more than 30 seconds after the recording does, the script assumes an unlogged track starts at `00:00`. It appears as `Unknown track (not in CSV)` until you name it with `--first-track`.

### Output files

| File | Contents |
|---|---|
| `tracklist.txt` | Detailed list with BPM, key, and how each timestamp was calculated |
| `tracklist.csv` | `Number, Timestamp, Title, Artist, BPM, Key`. Opens in Excel, Numbers, and Google Sheets. |
| `tracklist_share.txt` | Clean `timestamp  Artist - Title` list for YouTube, SoundCloud, or notes |

Existing copies of these files are overwritten.

**Tips**

- djay filenames contain a non-breaking space before AM/PM. If typing the path gives "File not found", drag the file into the terminal instead.
- `label_tracks.py` imports from `detect_transitions.py`, so keep both scripts in the same folder.

## Tuning

Parameters at the top of `detect_transitions.py`:

| Name | Default | Effect |
|---|---|---|
| `SR` | 22050 | Analysis sample rate. Lower is faster. |
| `FRAME_SECONDS` | 30 | Block length |
| `OVERLAP_SECONDS` | 5 | Overlap between blocks |
| `MIN_GAP_BETWEEN_TRANSITIONS` | 30 | Minimum seconds between reported transitions |

Without a track count, the threshold is the mean plus one standard deviation of the combined score, set in `find_transitions()`. Lower it to get more candidates or raise it for fewer.

## Project layout

```
detect_transitions.py   Transition detector (also used by label_tracks.py)
label_tracks.py         Tracklist labeler for djay CSV exports
```

## Contributing

Issues and pull requests are welcome. Please don't include audio recordings or personal tracklists in issues or pull requests.

## License

MIT. See [LICENSE](LICENSE).
