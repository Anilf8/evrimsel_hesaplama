"""
dataset_presets.py
==================

İki şeyi sağlar:

  1. IDMTLoader:  IDMT-SMT-Guitar WAV dosyalarını yükler.
                  Dataset yoksa zengin harmonikli yapay sinyal üretir.

  2. PRESETS:     5 hedef preset — parametre uzayının farklı bölgelerini
                  temsil eden bilinen ground truth x* vektörleri.
                  Algoritmalar bu noktaları geri keşfetmeye çalışır.

5 preset:  Clean, Crunch, Heavy, Ambient, Random
"""

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

from core.parameter_space import N_PARAMS, PARAM_NAMES


PRESETS: Dict[str, Optional[Dict[str, float]]] = {
    "clean": {
        "dist_drive":     0.10,  "dist_tone":      0.50,
        "eq_bass_gain":   0.55,  "eq_mid_gain":    0.50,  "eq_treble_gain": 0.55,
        "rev_room":       0.20,  "rev_wet":        0.15,  "rev_damping":    0.50,
        "delay_time":     0.10,  "delay_feedback": 0.10,  "delay_mix":      0.10,
        "comp_threshold": 0.50,  "comp_ratio":     0.20,
        "comp_attack":    0.30,  "comp_release":   0.40,
        "output_gain":    0.50,
    },
    "crunch": {
        "dist_drive":     0.40,  "dist_tone":      0.55,
        "eq_bass_gain":   0.55,  "eq_mid_gain":    0.60,  "eq_treble_gain": 0.55,
        "rev_room":       0.25,  "rev_wet":        0.20,  "rev_damping":    0.50,
        "delay_time":     0.15,  "delay_feedback": 0.15,  "delay_mix":      0.15,
        "comp_threshold": 0.40,  "comp_ratio":     0.35,
        "comp_attack":    0.20,  "comp_release":   0.30,
        "output_gain":    0.50,
    },
    "heavy": {
        "dist_drive":     0.85,  "dist_tone":      0.65,
        "eq_bass_gain":   0.65,  "eq_mid_gain":    0.35,  "eq_treble_gain": 0.65,
        "rev_room":       0.20,  "rev_wet":        0.20,  "rev_damping":    0.60,
        "delay_time":     0.10,  "delay_feedback": 0.10,  "delay_mix":      0.10,
        "comp_threshold": 0.20,  "comp_ratio":     0.60,
        "comp_attack":    0.10,  "comp_release":   0.20,
        "output_gain":    0.55,
    },
    "ambient": {
        "dist_drive":     0.20,  "dist_tone":      0.50,
        "eq_bass_gain":   0.50,  "eq_mid_gain":    0.50,  "eq_treble_gain": 0.55,
        "rev_room":       0.85,  "rev_wet":        0.70,  "rev_damping":    0.30,
        "delay_time":     0.60,  "delay_feedback": 0.50,  "delay_mix":      0.50,
        "comp_threshold": 0.50,  "comp_ratio":     0.25,
        "comp_attack":    0.40,  "comp_release":   0.50,
        "output_gain":    0.50,
    },
    # 'random' özel — get_preset_vector() içinde dinamik üretilir
    "random": None,
}

PRESET_NAMES: List[str] = ["clean", "crunch", "heavy", "ambient", "random"]


def get_preset_vector(name: str, seed: int = 0) -> np.ndarray:
    """
    Preset adından 16 boyutlu normalize ground truth vektörü x* döndürür.

    Args:
        name: Preset adı (clean / crunch / heavy / ambient / random).
        seed: 'random' preset için tohum.

    Returns:
        16 boyutlu float32 vektör, [0,1] aralığında.
    """
    if name == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, N_PARAMS).astype(np.float32)

    if name not in PRESETS or PRESETS[name] is None:
        raise ValueError(f"Bilinmeyen preset: {name}")

    spec = PRESETS[name]
    x = np.zeros(N_PARAMS, dtype=np.float32)
    for i, pname in enumerate(PARAM_NAMES):
        if pname not in spec:
            raise KeyError(f"'{name}' presetinde {pname} eksik")
        x[i] = spec[pname]
    return x


class IDMTLoader:
    """
    IDMT-SMT-Guitar yükleyici.

    dataset_root verilirse o klasörde recursive WAV taraması yapar.
    Aksi halde demo modu (yapay E2 harmonikleri) devreye girer.
    """

    DEMO_F0_HZ: float = 82.41  # E2

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        sample_rate: int = 22050,
        verbose: bool = True,
    ):
        self.sr      = sample_rate
        self.root    = dataset_root
        self.files:  List[str] = []
        self.verbose = verbose

        if dataset_root and os.path.isdir(dataset_root):
            self._scan_files()

        if verbose:
            if self.is_available():
                print(f"[IDMT] {len(self.files)} dosya bulundu  ({dataset_root})")
            else:
                print("[IDMT] Dataset yok — demo modu aktif")

    def _scan_files(self) -> None:
        patterns = [
            os.path.join(self.root, "**", "*.wav"),
            os.path.join(self.root, "**", "*.WAV"),
        ]
        for pat in patterns:
            self.files.extend(glob.glob(pat, recursive=True))
        self.files = sorted(set(self.files))

    def is_available(self) -> bool:
        return len(self.files) > 0

    def load_demo(self, duration: float = 1.0, seed: int = 0) -> np.ndarray:
        """
        Yapay gitar sesi — 5 harmonik + hafif gürültü + ADSR zarfı.
        Dataset olmadan da deneylerin çalışmasını sağlar.
        """
        rng = np.random.default_rng(seed)
        n   = int(self.sr * duration)
        t   = np.linspace(0.0, duration, n, dtype=np.float32)
        f0  = self.DEMO_F0_HZ

        sig = (
            0.50 * np.sin(2.0 * np.pi * 1.0 * f0 * t) +
            0.30 * np.sin(2.0 * np.pi * 2.0 * f0 * t) +
            0.15 * np.sin(2.0 * np.pi * 3.0 * f0 * t) +
            0.08 * np.sin(2.0 * np.pi * 4.0 * f0 * t) +
            0.04 * np.sin(2.0 * np.pi * 5.0 * f0 * t) +
            0.02 * rng.standard_normal(n).astype(np.float32)
        )

        # ADSR envelope (basit)
        env = np.ones(n, dtype=np.float32)
        a, r = int(0.01 * self.sr), int(0.20 * self.sr)
        if a > 0:
            env[:a]  = np.linspace(0.0, 1.0, a, dtype=np.float32)
        if r > 0 and r < n:
            env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)

        return np.clip(sig * env, -1.0, 1.0).astype(np.float32)

    def load_random_file(
        self, duration: float = 1.0, seed: int = 0
    ) -> Tuple[np.ndarray, str]:
        """Dataset'ten rastgele bir dosya. Yoksa demo döner."""
        if not self.is_available():
            return self.load_demo(duration, seed), "demo_signal"

        rng  = np.random.default_rng(seed)
        path = str(rng.choice(self.files))
        try:
            audio, file_sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)
            n = int(self.sr * duration)
            if len(audio) >= n:
                start = int(rng.integers(0, len(audio) - n))
                audio = audio[start : start + n]
            else:
                audio = np.pad(audio, (0, n - len(audio)))
            return audio.astype(np.float32), os.path.basename(path)
        except Exception as e:
            if self.verbose:
                print(f"[Uyarı] {path} okunamadı: {e}")
            return self.load_demo(duration, seed), "demo_fallback"

    def get_fixed_dry(self, duration: float = 1.0, seed: int = 0) -> np.ndarray:
        """
        Deneyler için SABİT kuru sinyal.
        Tüm algoritmalar bu sinyalle aynı girdiyi kullanır — adil karşılaştırma.
        """
        dry, _ = self.load_random_file(duration=duration, seed=seed)
        return dry