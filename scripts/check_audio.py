"""Armenian audio check (Phase 7).

    python scripts/check_audio.py              # how many signs have an Armenian file, which are missing / broken
    python scripts/check_audio.py --all        # list every missing sign
    python scripts/check_audio.py --play 20    # play 20 random words as the app does (file or English fallback)

Files: assets/audio/hy/<sign>.wav (16-bit mono WAV), recorded with scripts/record_audio.py.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import audio, config  # noqa: E402
from asl.vocab import load_vocab  # noqa: E402

SHOW_MISSING = 20
PLAY_TIMEOUT_S = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="list every missing sign")
    ap.add_argument("--play", type=int, default=0, metavar="N", help="play N random words")
    ap.add_argument("--seed", type=int, default=None, help="random seed for --play")
    args = ap.parse_args()

    vocab = load_vocab()
    missing = audio.missing_signs(vocab)
    present = [s for s in sorted(vocab) if s not in missing]
    broken = {s: problem for s in present if (problem := audio.check_file(audio.audio_path(s)))}
    print(f"Armenian audio: {len(present)} of {len(vocab)} signs have a file in {config.AUDIO_DIR}")
    print(f"missing: {len(missing)}" + (" (the app speaks the English word for them)" if missing else ""))
    shown = missing if args.all else missing[:SHOW_MISSING]
    if shown:
        print("  " + ", ".join(shown) + (" ..." if len(shown) < len(missing) else ""))
    for sign, problem in broken.items():
        print(f"BROKEN {sign}.wav: {problem}")

    if args.play:
        rng = random.Random(args.seed)
        player = audio.AudioPlayer(vocab, on_error=lambda d: print(f"  ERROR: {d}"))
        for sign in rng.sample(sorted(vocab), min(args.play, len(vocab))):
            kind, _ = player.source(sign)
            print(f"  {sign:12s} {vocab[sign]['hy']:20s} {'Armenian file' if kind == 'wav' else 'English (pyttsx3)'}")
            player.play_word(sign)
            player.idle.wait(PLAY_TIMEOUT_S)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
