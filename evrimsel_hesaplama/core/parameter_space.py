"""
parameter_space.py
==================

16 boyutlu sürekli arama uzayı tanımı.

Tüm evrimsel algoritmalar normalize edilmiş vektörlerle ([0, 1]) çalışır.
Sadece Pedalboard'a verilirken `denormalize()` ile gerçek aralıklara çevrilir.
Bu, algoritmaların parametre ölçeklerinden bağımsız çalışmasını sağlar.

Notasyon: x ∈ [0, 1]^16 normalize vektör, p ∈ ℝ^16 gerçek değer vektörü.
"""

from typing import Dict, List, Tuple
import numpy as np


PARAM_SPEC: Dict[str, Tuple[float, float]] = {
    # Distortion — kazanç ve ton
    "dist_drive":     (0.0,  50.0),   # dB
    "dist_tone":      (0.0,   1.0),   # 0 = koyu, 1 = parlak

    # 3-bant EQ
    "eq_bass_gain":   (-12.0, 12.0),  # dB  — Low-shelf 200 Hz
    "eq_mid_gain":    (-12.0, 12.0),  # dB  — Peak     1000 Hz
    "eq_treble_gain": (-12.0, 12.0),  # dB  — High-shelf 4000 Hz

    # Reverb
    "rev_room":       (0.0,   1.0),
    "rev_wet":        (0.0,   0.8),
    "rev_damping":    (0.0,   1.0),

    # Delay
    "delay_time":     (0.01,  0.75),  # saniye
    "delay_feedback": (0.0,   0.8),
    "delay_mix":      (0.0,   0.7),

    # Compressor
    "comp_threshold": (-40.0, 0.0),   # dBFS
    "comp_ratio":     (1.0,   20.0),
    "comp_attack":    (1.0,   100.0), # ms
    "comp_release":   (10.0,  500.0), # ms

    # Çıkış kazancı
    "output_gain":    (-12.0, 6.0),   # dB
}


PARAM_NAMES:  List[str]                  = list(PARAM_SPEC.keys())
PARAM_BOUNDS: List[Tuple[float, float]]  = list(PARAM_SPEC.values())
N_PARAMS:     int                        = len(PARAM_NAMES)


def denormalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize edilmiş [0,1]^16 vektörü gerçek değer aralıklarına dönüştürür.

    Parametre değeri i için:  p_i = lo_i + clip(x_i, 0, 1) * (hi_i - lo_i)
    """
    if len(x) != N_PARAMS:
        raise ValueError(f"x boyutu {N_PARAMS} olmalı, {len(x)} verildi")
    p = np.zeros(N_PARAMS, dtype=np.float32)
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        p[i] = lo + float(np.clip(x[i], 0.0, 1.0)) * (hi - lo)
    return p


def normalize(p: np.ndarray) -> np.ndarray:
    """Gerçek değer vektörünü [0,1]^16'ya geri çevirir (denormalize'ın tersi)."""
    x = np.zeros(N_PARAMS, dtype=np.float32)
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        rng = hi - lo
        x[i] = (p[i] - lo) / rng if rng > 0 else 0.0
    return np.clip(x, 0.0, 1.0)


def get_param_idx(name: str) -> int:
    """Parametre adından ona ait indeksi döndürür."""
    return PARAM_NAMES.index(name)


def random_vector(rng: np.random.Generator) -> np.ndarray:
    """[0,1] uniform rastgele 16 boyutlu vektör."""
    return rng.uniform(0.0, 1.0, N_PARAMS).astype(np.float32)