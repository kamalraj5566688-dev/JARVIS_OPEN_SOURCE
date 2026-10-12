"""
Jaddu Voice Assistant - Speaker Verification (Voice Biometrics) module.

Extracts 256-D speaker embeddings from raw 16 kHz mono int16 PCM audio using
the resemblyzer voice encoder (pretrained weights bundled inside the package,
so inference is 100% offline). Voice verification is then performed by cosine
similarity against the enrolled owner profile (voice_profile.npy).

Usage:
    from speaker_verifier import SpeakerVerifier
    verifier = SpeakerVerifier(profile_path="voice_profile.npy", threshold=0.72)
    ok, similarity = verifier.verify(audio_bytes)   # (is_owner, score)
"""

import os
from datetime import datetime

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_PROFILE_PATH = "voice_profile.npy"
DEFAULT_THRESHOLD = 0.62
EMBEDDING_SIZE = 256
MIN_AUDIO_SAMPLES = SAMPLE_RATE * 2 // 5  # ~0.4s minimum for a stable embedding


class SpeakerVerifier:
    """Voice biometric gatekeeper for the Jaddu assistant."""

    def __init__(self, profile_path=DEFAULT_PROFILE_PATH, threshold=DEFAULT_THRESHOLD):
        self.profile_path = profile_path
        self.threshold = float(threshold)
        self._encoder = None
        self._preprocess_wav = None
        self._enrolled = self._load_profile() if os.path.exists(profile_path) else None

    # ------------------------------------------------------------------ #
    # Backend / model management
    # ------------------------------------------------------------------ #
    def _get_encoder(self):
        """Lazily initialise the resemblyzer VoiceEncoder on CPU (offline)."""
        if self._encoder is None:
            from resemblyzer import preprocess_wav
            from resemblyzer.voice_encoder import VoiceEncoder

            self._preprocess_wav = preprocess_wav
            self._encoder = VoiceEncoder(device="cpu", verbose=False)
        return self._encoder

    def _load_profile(self):
        """Reads the enrolled voice profile from disk. Returns dict or None."""
        try:
            data = np.load(self.profile_path, allow_pickle=True)
            obj = data.item() if data.ndim == 0 else data
            if isinstance(obj, dict) and "embedding" in obj:
                emb = np.asarray(obj["embedding"], dtype=np.float32)
                if emb.shape == (EMBEDDING_SIZE,):
                    return obj
                return None
            # Legacy support: a bare 256-D numpy array
            arr = np.asarray(obj, dtype=np.float32).reshape(-1)
            if arr.shape == (EMBEDDING_SIZE,):
                return {"embedding": arr}
            return None
        except Exception as e:
            print(f"[SpeakerVerifier] Failed to load voice profile: {e}")
            return None
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def profile_exists(self) -> bool:
        return self._enrolled is not None

    @property
    def enrollment_metadata(self) -> dict:
        if not self._enrolled:
            return {}
        return {
            "created_at": self._enrolled.get("created_at", "unknown"),
            "n_samples": self._enrolled.get("n_samples", 1),
            "threshold": self._enrolled.get("threshold", self.threshold),
        }

    def preprocess(self, audio_bytes: bytes) -> np.ndarray:
        """Converts raw int16 PCM to an encoder-ready float waveform [-1, 1]."""
        wav = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if wav.size < MIN_AUDIO_SAMPLES:
            raise ValueError(
                f"Audio too short for a reliable embedding ({wav.size} samples, "
                f"need >= {MIN_AUDIO_SAMPLES})"
            )
        self._get_encoder()
        return self._preprocess_wav(wav, source_sr=SAMPLE_RATE)

    def embed(self, audio_bytes: bytes) -> np.ndarray:
        """Returns a 256-D L2-normalised speaker embedding for the audio."""
        wav = self.preprocess(audio_bytes)
        return np.asarray(self._get_encoder().embed_utterance(wav), dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.ravel().astype(np.float64)
        b = b.ravel().astype(np.float64)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / norm)

    def score(self, audio_bytes: bytes) -> float:
        """Cosine similarity of the audio against the enrolled profile."""
        if not self.profile_exists:
            return 0.0
        emb = self.embed(audio_bytes)
        enrolled = np.asarray(self._enrolled["embedding"], dtype=np.float32)
        return self.cosine_similarity(emb, enrolled)

    def verify(self, audio_bytes: bytes):
        """
        Verifies whether the audio belongs to the enrolled owner.

        Returns (bool is_owner, float similarity).
        """
        if not self.profile_exists:
            print("[SpeakerVerifier] No enrolled voice profile present.")
            return False, 0.0
        try:
            sim = self.score(audio_bytes)
        except ValueError as e:
            print(f"[SpeakerVerifier] {e}")
            return False, 0.0
        except Exception as e:
            print(f"[SpeakerVerifier] Verification error: {e}")
            return False, 0.0
        return sim >= self.threshold, sim

    # ------------------------------------------------------------------ #
    # Enrollment helpers
    # ------------------------------------------------------------------ #
    def save_profile(self, embedding, n_samples=1, threshold=None, extra=None):
        """Persists the enrolled owner embedding to voice_profile.npy."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape != (EMBEDDING_SIZE,):
            raise ValueError(f"Embedding must be {EMBEDDING_SIZE}-D, got {emb.shape}")
        record = {
            "embedding": emb,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_samples": int(n_samples),
            "threshold": float(threshold or self.threshold),
        }
        if extra:
            record.update(extra)
        np.save(self.profile_path, record)
        self._enrolled = record
        print(f"[SpeakerVerifier] Voice profile saved to {self.profile_path}")

    def self_test(self, samples):
        """
        Returns per-sample cosine similarities against a mean enrollment
        embedding so the user can sanity-check the chosen threshold.
        """
        mean = np.mean([np.asarray(s, dtype=np.float32) for s in samples], axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-9)
        return mean, [self.cosine_similarity(mean, s) for s in samples]