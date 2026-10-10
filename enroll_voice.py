"""
Jaddu Voice Assistant - Voice Enrollment CLI.

Records 3-5 natural speech samples from the owner via the microphone,
extracts speaker embeddings with resemblyzer, and saves the mean profile
to voice_profile.npy.

Usage:
    python enroll_voice.py              # default: 4 samples x 3 seconds
    python enroll_voice.py --samples 5 --duration 3.5 --threshold 0.72

Run this BEFORE starting the assistant. Speak in a normal, natural tone -
the same way you would talk to Jaddu during the day.
"""

import argparse
import sys

import numpy as np
import sounddevice as sd

from speaker_verifier import SAMPLE_RATE, DEFAULT_THRESHOLD, SpeakerVerifier

MIN_SPEECH_RMS = 110.0  # sensitive enough for normal speech above ~58 ambient room noise


def record_sample(duration: float) -> bytes:
    """Records `duration` seconds of 16 kHz mono int16 PCM audio with countdown."""
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print(f"[Mic Status] {status}")
        frames.append(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=4000,
        dtype="int16",
        channels=1,
        callback=callback,
    ) as stream:
        # Warm the stream up so the mic is ready
        sd.sleep(300)
        total_seconds = int(duration)
        print("  [Recording: ", end="", flush=True)
        for rem in range(total_seconds, 0, -1):
            print(f"{rem}s... ", end="", flush=True)
            sd.sleep(1000)
        rem_ms = int((duration - total_seconds) * 1000)
        if rem_ms > 0:
            sd.sleep(rem_ms)
        print("Done!]", flush=True)

    return b"".join(frames)


def calculate_rms(audio_bytes: bytes) -> float:
    data = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data.astype(np.float64)))))


def main():
    parser = argparse.ArgumentParser(description="Enroll the owner voice for Jaddu")
    parser.add_argument("--samples", type=int, default=4, help="Number of voice samples (3-5 recommended)")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds to record per sample")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Cosine similarity threshold used for verification later")
    parser.add_argument("--profile", default="voice_profile.npy", help="Output profile path")
    args = parser.parse_args()

    print("=" * 64)
    print("  Jaddu Voice Assistant - Owner Voice Enrollment")
    print("=" * 64)
    print(f"  Target profile : {args.profile}")
    print(f"  Samples        : {args.samples} x {args.duration}s")
    print(f"  Threshold      : {args.threshold}")
    print()
    print("  Tips:")
    print("  * Record in a quiet room, no background TV/music.")
    print("  * Speak naturally and clearly - nothing forced.")
    print("  * Say short command-like phrases, e.g.")
    print("    'Hello Jaddu, system status batao' or 'VS Code kholo'.")
    print("=" * 64)

    verifier = SpeakerVerifier(profile_path=args.profile, threshold=args.threshold)

    if verifier.profile_exists:
        print(f"\n[!] Existing profile found: {verifier.profile_exists}")
        print(f"    Created: {verifier.enrollment_metadata.get('created_at')} "
              f"({verifier.enrollment_metadata.get('n_samples')} samples)")
        ans = input("    Overwrite it? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted. Existing profile kept.")
            return

    print("\nLet's capture your voice. Prepare to speak after pressing Enter.")
    input("Press Enter when ready for the first sample...")

    embeddings = []
    attempts = 0
    while len(embeddings) < args.samples and attempts < args.samples * 3:
        attempts += 1
        print(f"\n  >> Sample {len(embeddings) + 1}/{args.samples}:")
        audio = record_sample(args.duration)

        rms = calculate_rms(audio)
        print(f"    RMS energy: {rms:7.1f}")
        if rms < MIN_SPEECH_RMS:
            print("    [!] Very low audio (quiet). Please speak up / closer to the mic.")
            input("    Press Enter to re-record this sample...")
            continue

        try:
            emb = verifier.embed(audio)
        except ValueError as e:
            print(f"    [!] {e} - re-record this sample.")
            input("    Press Enter to re-record this sample...")
            continue
        except Exception as e:
            print(f"    [!] Embedding failed: {e} - re-record this sample.")
            input("    Press Enter to re-record this sample...")
            continue

        embeddings.append(emb)
        print(f"    [OK] Sample {len(embeddings)}/{args.samples} embedded "
              f"(RMS {rms:.0f})")
        if len(embeddings) < args.samples:
            input(f"\n    Press Enter when ready for sample {len(embeddings) + 1}/{args.samples}...")

    if len(embeddings) < args.samples:
        print("\nNot enough valid samples captured. Enrollment aborted.")
        sys.exit(1)

    # Mean embedding -> normalized enrollment profile
    mean_emb, similarities = verifier.self_test(embeddings)
    verifier.save_profile(mean_emb, n_samples=len(embeddings), threshold=args.threshold)

    print("\n" + "=" * 64)
    print("  Enrollment complete!")
    print("=" * 64)
    print(f"  Profile saved  : {args.profile}")
    print(f"  Samples used   : {len(embeddings)}")
    for i, sim in enumerate(similarities, 1):
        print(f"    Sample {i} self-similarity: {sim:.3f} (vs threshold {args.threshold})")
    min_sim = min(similarities)
    print(f"\n  Recommendation: keep threshold below your lowest self-score ({min_sim:.3f})")
    print("  If a stranger voice keeps passing, RAISE the threshold (e.g. 0.78).")
    print("  If your own voice keeps getting rejected, LOWER it (e.g. 0.65).")
    print("\n  Next step: run `python main.py` and say 'Hello Jaddu'.")

    # Optional live self-test
    try:
        answer = input("\nDo a quick live voice check now? [y/N]: ").strip().lower()
        if answer == "y":
            input("Press Enter, then say 'Hello Jaddu' ...")
            audio = record_sample(2.5)
            ok, sim = verifier.verify(audio)
            print(f"    Verification similarity: {sim:.3f} -> "
                  f"{'MATCH (owner voice confirmed)' if ok else 'REJECTED (try again / re-enroll)'}")
    except KeyboardInterrupt:
        print("\nSkipped live check.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEnrollment cancelled by user.")
        sys.exit(1)