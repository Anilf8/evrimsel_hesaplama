"""
map_elites.py
=============

MAP-Elites — Quality Diversity tabanlı evrimsel algoritma.

Referans: Mouret & Clune (2015) "Illuminating Search Spaces by Mapping Elites"
          arXiv:1504.04909

Klasik EA'lardan temel farkı: tek bir optimum aramaz, arama uzayının
**haritasını çıkarır**. Her davranış bölgesinin en iyi temsilcisini
bir arşivde saklar.

Operatörler:
  - Seçim     : arşivden uniform rastgele ebeveyn
  - Varyasyon : Gauss mutasyonu, σ=0.08
  - Yerleşim  : davranış hücresine yerleştir, eski elite'i geçtiyse değiştir

ÖNEMLİ — fitness değerlendirme bütçesi sayımı:
  MAP-Elites'in 'fitness' fonksiyonu kullanıcının verdiği fitness'a ek olarak
  davranış vektörünü de hesaplamak zorunda. Bu nedenle bizim implementasyonda
  EvaluatorWithBehavior özel bir sarmalayıcı kullanılıyor — base.Evaluator'a
  arşiv yerleşimi için ek bilgi geçirir.
"""

from typing import Callable, Dict, Optional, Tuple
import numpy as np

from core import (
    N_PARAMS,
    BehaviorSpace, behavior_to_cell,
)
from .base import Algorithm, Evaluator, BudgetExhausted


class MAPElites(Algorithm):
    """
    Vanilla MAP-Elites.

    Bütçe yönetimi:
      İlk n_init birey rastgele üretilir ve arşive yerleştirilir.
      Geri kalan bütçe mutasyon iterasyonları için kullanılır.

    Bu sınıf 'fitness_fn'in lazım olduğu kadar küçük ek bilgiyi
    (davranış vektörü) hesaplamak için behavior_fn'i de alır.
    Her iki fonksiyon da aynı ses sinyalinden hesaplandığı için
    behavior_fn cache mekanizması ile birleştirilmiş olabilir
    (bu detay runner tarafında halledilir).
    """

    name = "MAPElites"

    def __init__(
        self,
        behavior_fn:    Callable[[np.ndarray], Tuple[float, float]],
        behavior_space: BehaviorSpace = None,
        n_init:         int           = 200,
        mutation_sigma: float         = 0.08,
    ):
        super().__init__()
        self.behavior_fn    = behavior_fn
        self.behavior_space = behavior_space or BehaviorSpace()
        self.n_init         = n_init
        self.mutation_sigma = mutation_sigma

        # Arşiv: cell (i,j) → (params, fitness, behavior)
        self.archive: Dict[Tuple[int, int], Tuple[np.ndarray, float, Tuple[float, float]]] = {}

    # ────────────────────────────────────────────────────────────────
    def _try_insert(
        self,
        x:        np.ndarray,
        fitness:  float,
    ) -> None:
        """
        Bir bireyi arşive yerleştirmeyi dene.
        Hücre boşsa yerleştir; doluysa fitness daha iyiyse değiştir.
        """
        b = self.behavior_fn(x)                 # (centroid, zcr)
        cell = behavior_to_cell(b[0], b[1], self.behavior_space)

        existing = self.archive.get(cell)
        if existing is None or fitness > existing[1]:
            self.archive[cell] = (x.copy(), fitness, b)

    # ────────────────────────────────────────────────────────────────
    def _select_parent(self, rng: np.random.Generator) -> np.ndarray:
        """Arşivden rastgele bir elite seç."""
        keys = list(self.archive.keys())
        key  = keys[int(rng.integers(0, len(keys)))]
        return self.archive[key][0]

    # ────────────────────────────────────────────────────────────────
    def _mutate(
        self,
        x:   np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """İzotropik Gauss mutasyonu — tüm boyutlara uygulanır."""
        delta = rng.normal(0.0, self.mutation_sigma, size=self.n_params)
        return np.clip(x + delta.astype(np.float32), 0.0, 1.0)

    # ────────────────────────────────────────────────────────────────
    def run(
        self,
        evaluator: Evaluator,
        rng:       np.random.Generator,
    ) -> None:
        # FAZ 1: Başlangıç — n_init rastgele birey
        for _ in range(self.n_init):
            if evaluator.exhausted():
                return
            x = rng.uniform(0.0, 1.0, self.n_params).astype(np.float32)
            f = evaluator(x)
            self._try_insert(x, f)

        # FAZ 2: Mutasyon döngüsü
        while not evaluator.exhausted():
            if not self.archive:
                # Korumalı: arşiv boşsa (init sırasında bütçe bittiyse)
                break
            parent = self._select_parent(rng)
            child  = self._mutate(parent, rng)
            try:
                f = evaluator(child)
            except BudgetExhausted:
                break
            self._try_insert(child, f)

    # ────────────────────────────────────────────────────────────────
    def build_result(self, evaluator, seed, extra=None):
        """RunResult'a arşiv istatistiklerini ekle."""
        archive_info = {
            "n_cells":   len(self.archive),
            "coverage":  len(self.archive) / self.behavior_space.n_cells,
            "qd_score":  sum(fit for _, fit, _ in self.archive.values()),
            "archive":   {                                  # cell → fitness
                cell: fit for cell, (_, fit, _) in self.archive.items()
            },
        }
        extra = {**(extra or {}), "map_elites": archive_info}
        return super().build_result(evaluator, seed, extra)