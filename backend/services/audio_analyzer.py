"""
AUDIO analyzer.

Real operations:
  * load audio. Preferred: soundfile/librosa. Fallback: pure-numpy decoder for
    uncompressed WAV (no ffmpeg / no extra dependency required).
  * duration, sample rate, RMS energy, zero-crossing rate, dynamic range
  * a real STFT magnitude spectrogram (numpy FFT) -> mel-like band energies
  * spectral centroid / rolloff / flatness style features
  * a simple "synthetic-ness" heuristic: unusually clean silence, clipped
    peaks, flat spectral envelope, or unnatural periodicity

MP3/other compressed formats need soundfile+librosa OR ffmpeg. If neither is
available we return a clear, honest error instead of guessing.
"""

from __future__ import annotations

import os
import struct
import wave
from typing import Optional

import numpy as np

from backend.services.base import (
    Finding, ModalityResult, finalize, make_result,
    ENGINE_HEURISTIC, ENGINE_MODEL,
)
from backend.utils.helpers import clamp, risk_to_label
from backend.config import SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD, EMBEDDING_DIM

try:
    import soundfile as sf
    SF_AVAILABLE = True
except Exception:  # pragma: no cover
    SF_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:  # pragma: no cover
    LIBROSA_AVAILABLE = False


def _load_with_soundfile(path: str):
    data, sr = sf.read(path, always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr), "soundfile"


def _load_with_librosa(path: str):
    data, sr = librosa.load(path, sr=None, mono=True)
    return np.asarray(data, dtype=np.float32), int(sr), "librosa"


def _load_wav_numpy(path: str):
    """Pure-stdlib WAV decoder (PCM only). Works with zero extra packages."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
    elif sampwidth == 1:
        dtype = np.uint8
        scale = 128.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if dtype == np.uint8:
        data = data - 128.0
    data = data / scale
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), int(framerate), "numpy-wav"


def _load_audio(path: str):
    errors = []
    for loader in (_load_with_soundfile if SF_AVAILABLE else None,
                   _load_with_librosa if LIBROSA_AVAILABLE else None):
        if loader is None:
            continue
        try:
            return loader(path)
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    try:
        return _load_wav_numpy(path)
    except Exception as exc:
        errors.append(f"numpy-wav: {exc}")
    raise RuntimeError("; ".join(errors) if errors else "no audio decoder available")


def _stft_magnitude(signal: np.ndarray, n_fft: int = 1024, hop: int = 512) -> np.ndarray:
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, signal.size - n_fft + 1, hop):
        frame = signal[start:start + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    if not frames:
        return np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
    return np.asarray(frames, dtype=np.float32).T  # (freq, time)


def _mel_band_energies(power: np.ndarray, sr: int, bands: int = 24) -> np.ndarray:
    """Very small triangular mel-ish filterbank implemented in numpy."""
    n_freq = power.shape[0]
    freqs = np.linspace(0, sr / 2.0, n_freq)
    # mel scale
    mel = 2595.0 * np.log10(1.0 + freqs / 700.0)
    mel_max = mel[-1] if mel[-1] > 0 else 1.0
    edges = np.linspace(0, mel_max, bands + 2)
    energy = np.zeros(bands, dtype=np.float32)
    for b in range(bands):
        lo, center, hi = edges[b], edges[b + 1], edges[b + 2]
        filt = np.zeros(n_freq, dtype=np.float32)
        left = (mel >= lo) & (mel <= center)
        right = (mel >= center) & (mel <= hi)
        if center > lo:
            filt[left] = (mel[left] - lo) / (center - lo)
        if hi > center:
            filt[right] = (hi - mel[right]) / (hi - center)
        energy[b] = float(np.mean(power.T @ filt)) if filt.any() else 0.0
    return energy


def analyze(path: str, meta: dict) -> ModalityResult:
    result = make_result("audio", meta, engine=ENGINE_HEURISTIC)

    try:
        signal, sr, loader = _load_audio(path)
    except Exception as exc:
        result.error = (f"Could not decode audio: {exc}. "
                        "Install soundfile/librosa or provide an uncompressed WAV.")
        result.quality = "unusable"
        result.findings.append(Finding(
            "decode_failed", "Audio could not be decoded",
            "No available decoder could read this audio file.", "high", "missing"))
        return finalize(result, EMBEDDING_DIM, meta.get("filename", "audio"))

    if signal.size == 0:
        result.error = "Audio decoded but contained no samples."
        result.quality = "unusable"
        result.findings.append(Finding(
            "empty", "Empty audio", "The audio track is empty.", "high", "missing"))
        return finalize(result, EMBEDDING_DIM, meta.get("filename", "audio"))

    duration = float(signal.size) / float(sr) if sr else 0.0
    rms = float(np.sqrt(np.mean(signal ** 2)) + 1e-9)
    peak = float(np.max(np.abs(signal)) + 1e-9)
    zcr = float(np.mean(np.abs(np.diff(np.sign(signal)))) / 2.0)
    # clipped-sample ratio
    clipped = float(np.mean(np.abs(signal) > 0.99))
    # dynamic range in dB
    dyn_db = float(20.0 * np.log10((peak + 1e-9) / (rms + 1e-9)))

    spectrum = _stft_magnitude(signal)
    power = spectrum ** 2
    mel = _mel_band_energies(power, sr)
    mel_norm = mel / (mel.max() + 1e-9)

    freqs = np.linspace(0, sr / 2.0, spectrum.shape[0])
    mag_mean = spectrum.mean(axis=1)
    total = mag_mean.sum() + 1e-9
    centroid = float((freqs * mag_mean).sum() / total)
    # spectral flatness (geometric/arithmetic mean of the mel bands)
    gmean = float(np.exp(np.mean(np.log(mel + 1e-9))))
    amean = float(np.mean(mel) + 1e-9)
    flatness = clamp(gmean / amean)
    # silence ratio
    frame_rms = np.sqrt(np.mean(signal[: (signal.size // 1024) * 1024 or signal.size]
                                 .reshape(-1, 1024) ** 2, axis=1)) if signal.size >= 1024 else np.array([rms])
    silence_ratio = float(np.mean(frame_rms < 0.005))

    # --- synthetic / manipulation indicators (heuristic) ------------------
    silence_ind = clamp((silence_ratio - 0.3) / 0.4)       # unnaturally clean silence
    clip_ind = clamp(clipped * 20.0)                        # hard clipping
    flat_ind = clamp((flatness - 0.4) / 0.4)                # flat spectrum -> synthetic
    dyn_ind = clamp((10.0 - dyn_db) / 10.0) if dyn_db < 10 else 0.0  # over-compressed
    zcr_ind = clamp(abs(zcr - 0.08) / 0.2)

    indicators = {
        "clipping_indicator": round(clip_ind, 4),
        "silence_anomaly": round(silence_ind, 4),
        "spectral_flatness_indicator": round(flat_ind, 4),
        "dynamic_range_indicator": round(dyn_ind, 4),
        "zcr_anomaly": round(zcr_ind, 4),
    }
    risk = (0.25 * clip_ind + 0.20 * silence_ind + 0.25 * flat_ind
            + 0.15 * dyn_ind + 0.15 * zcr_ind)

    model_used = False
    try:
        from backend.services.video_analyzer import CV2_AVAILABLE  # noqa: F401
    except Exception:
        pass
    from backend.config import MODE
    if MODE == "model":
        try:
            from backend.models.registry import predict_individual
            model_score = predict_individual(
                "audio", {**indicators, "duration": duration, "rms": rms}, path)
            if model_score is not None:
                risk = float(model_score)
                model_used = True
                result.engine = ENGINE_MODEL
        except Exception:
            model_used = False

    result.score = clamp(risk)
    result.label = risk_to_label(result.score, SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD)
    result.confidence = clamp(0.5 + 0.4 * min(1.0, duration / 3.0) - 0.2 * (sr < 8000))

    result.features = {
        "sample_rate": sr,
        "duration_seconds": round(duration, 3),
        "channels": 1,
        "loader": loader,
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "zero_crossing_rate": round(zcr, 5),
        "clipped_ratio": round(clipped, 6),
        "dynamic_range_db": round(dyn_db, 3),
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_flatness": round(flatness, 4),
        "silence_ratio": round(silence_ratio, 4),
        "mel_bands": [round(float(x), 5) for x in mel_norm.tolist()],
        **indicators,
    }
    result.detail = {
        "model_applied": model_used,
        "mel_bands": [round(float(x), 5) for x in mel_norm.tolist()],
        "decoder": loader,
    }

    if clip_ind > 0.5:
        result.findings.append(Finding(
            "clipping", "Audio clipping",
            "A notable fraction of samples are at full scale, which can mask edits.",
            "low", "contradicts"))
    if flat_ind > 0.5:
        result.findings.append(Finding(
            "flat_spectrum", "Flat spectral envelope",
            "The spectrum is unusually flat, sometimes seen in synthetic speech.",
            "medium", "contradicts"))
    if silence_ratio > 0.5:
        result.findings.append(Finding(
            "silence", "Large silent portions",
            "Much of the audio is near-silent; content may be missing.", "low", "missing"))
    if result.score <= AUTHENTIC_THRESHOLD:
        result.findings.append(Finding(
            "audio_ok", "No strong audio anomaly",
            "Audio-level heuristics did not surface strong manipulation indicators.",
            "info", "supports"))

    result.warnings.append(
        "Audio indicators are DEMO / HEURISTIC; they do not prove synthetic speech.")
    return finalize(result, EMBEDDING_DIM, meta.get("filename", "audio"))
