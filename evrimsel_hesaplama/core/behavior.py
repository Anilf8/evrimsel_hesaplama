"""
behavior.py
===========

Davranış tanımlayıcı (behavior descriptor).

MAP-Elites'in arşivinde "her hücre neyi temsil ediyor?" sorusunun cevabı.
Bir ses sinyalinden 2 boyutlu bir davranış vektörü çıkarıyoruz:

    b(y) = (spectral_centroid, zero_crossing_rate)

Bu boyutlar müzikal olarak yorumlanabilir:
  - Spectral centroid: parlaklık (bright / dark)
  - Zero-crossing rate: sertlik (smooth / harsh)

Davranış uzayı 20×20 hücreye bölünür. Bir parametre vektörünün
hangi hücreye düştüğü, ürettiği sesin özelliklerine bağlıdır —
parametre değerlerine değil. Bu, MAP-Elites'in temel fikridir.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import librosa


@dataclass(frozen=True)
class BehaviorSpace:
    """
    Davranış uzayı tanımı: 2D ızgara, sınırlar ile.

    Sınırlar amprik olarak seçildi — IDMT demo sinyali üzerinde
    rastgele parametrelerle yapılan ön deneyler bu aralıkları kapsadı.
    Çok dar tutmak hücrelerin boş kalmasına, çok geniş tutmak
    çözünürlük kaybına yol açar.
    """
    grid_size:       int   = 20
    centroid_min_hz: float = 200.0
    centroid_max_hz: float = 8000.0
    zcr_min:         float = 0.0
    zcr_max:         float = 0.40

    @property
    def n_cells(self) -> int:
        return self.grid_size ** 2


def compute_behavior(signal: np.ndarray, sample_rate: int = 22050) -> Tuple[float, float]:
    """
    Bir ses sinyalinden 2 boyutlu davranış vektörü çıkar.

    Returns:
        (centroid_hz, zcr) — ikisi de skaler.
    """
    sig = np.asarray(signal, dtype=np.float32)
    mx  = float(np.max(np.abs(sig)))
    if mx > 1e-6:
        sig = sig / mx
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=sig, sr=sample_rate)))
    zcr      = float(np.mean(librosa.feature.zero_crossing_rate(sig)))
    return centroid, zcr


def behavior_to_cell(
    centroid_hz: float,
    zcr:         float,
    space:       BehaviorSpace,
) -> Tuple[int, int]:
    """Davranış değerlerini ızgara hücre indekslerine eşle."""
    G  = space.grid_size
    xn = (centroid_hz - space.centroid_min_hz) / (space.centroid_max_hz - space.centroid_min_hz)
    yn = (zcr         - space.zcr_min)         / (space.zcr_max         - space.zcr_min)
    xi = int(np.clip(xn * (G - 1), 0, G - 1))
    yi = int(np.clip(yn * (G - 1), 0, G - 1))
    return xi, yi