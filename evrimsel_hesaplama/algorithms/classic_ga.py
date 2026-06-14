"""
classic_ga.py
=============

Klasik Genetic Algorithm — sürekli uzay için reel-değerli kodlama.

Operatörler (sürekli optimizasyon için literatür standardı):

  Seçim       : Turnuva seçimi (k=3)
                  - Fitness ölçeklemesine duyarsız
                  - Yerel optimum baskısını kontrol etmek kolay

  Çaprazlama  : BLX-α (Blend Crossover, α=0.5)
                  c_i = x¹_i + u·(x²_i - x¹_i),   u ~ U(-α, 1+α)
                  - α büyük = daha fazla keşif
                  - α küçük = daha fazla sömürü (exploit)

  Mutasyon    : Gaussian, polynomial olasılıkla
                  x_i ← clip(x_i + N(0, σ²), 0, 1),   olasılık p_m = 1/n

  Elitizm     : En iyi 2 birey her nesilde değişmeden geçer

Referans:  Eshelman & Schaffer (1993), Goldberg (1989), Deb & Beyer (2001).
"""

from typing import Tuple
import numpy as np

from .base import Algorithm, Evaluator


class ClassicGA(Algorithm):
    """
    Generational GA, sabit popülasyon boyutu.

    Bir nesil = pop_size birey değerlendirmesi.
    10.000 bütçe / 100 popülasyon = ~100 nesil.
    """

    name = "ClassicGA"

    def __init__(
        self,
        pop_size:        int   = 100,
        tournament_size: int   = 3,
        crossover_rate:  float = 0.9,
        crossover_alpha: float = 0.5,
        mutation_sigma:  float = 0.1,
        n_elites:        int   = 2,
        mutation_rate:   float = None,    # None → 1/n_params
    ):
        super().__init__()
        self.pop_size        = pop_size
        self.tournament_size = tournament_size
        self.crossover_rate  = crossover_rate
        self.crossover_alpha = crossover_alpha
        self.mutation_sigma  = mutation_sigma
        self.n_elites        = n_elites
        self.mutation_rate   = (mutation_rate if mutation_rate is not None
                                else 1.0 / self.n_params)

    # ────────────────────────────────────────────────────────────────
    # Yardımcı operatörler
    # ────────────────────────────────────────────────────────────────

    def _tournament_select(
        self,
        population: np.ndarray,
        fitnesses:  np.ndarray,
        rng:        np.random.Generator,
    ) -> np.ndarray:
        """k bireyden en iyisini seç (higher fitness = better)."""
        idx = rng.integers(0, len(population), self.tournament_size)
        winner = idx[np.argmax(fitnesses[idx])]
        return population[winner].copy()

    def _blx_alpha_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        BLX-α blend crossover.
        c_i = p1_i + u·(p2_i - p1_i),  u ~ U(-α, 1+α)

        Sonuç ebeveynlerin oluşturduğu doğrudan parçasının dışına da çıkabilir,
        bu keşfi destekler.
        """
        a = self.crossover_alpha
        u = rng.uniform(-a, 1.0 + a, size=self.n_params).astype(np.float32)
        c1 = p1 + u * (p2 - p1)
        # İkinci çocuk: simetrik
        u2 = rng.uniform(-a, 1.0 + a, size=self.n_params).astype(np.float32)
        c2 = p2 + u2 * (p1 - p2)
        return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)

    def _gaussian_mutate(
        self,
        x:   np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Her gen, mutation_rate olasılıkla N(0, σ²) ile pertürbe edilir."""
        mask  = rng.random(self.n_params) < self.mutation_rate
        delta = rng.normal(0.0, self.mutation_sigma, size=self.n_params)
        x = x + mask.astype(np.float32) * delta.astype(np.float32)
        return np.clip(x, 0.0, 1.0)

    # ────────────────────────────────────────────────────────────────
    # Ana akış
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        evaluator: Evaluator,
        rng:       np.random.Generator,
    ) -> None:
        # 1. Başlangıç popülasyonu
        population = rng.uniform(0.0, 1.0,
                                 (self.pop_size, self.n_params)
                                 ).astype(np.float32)
        fitnesses  = evaluator.evaluate_batch(population)

        # 2. Nesiller
        while not evaluator.exhausted():
            # Elitleri sakla (best n_elites bireyler)
            elite_idx = np.argsort(-fitnesses)[: self.n_elites]
            elites    = population[elite_idx].copy()

            # Yeni nesil
            new_pop = []
            new_pop.extend(elites)

            while len(new_pop) < self.pop_size:
                # Seçim
                p1 = self._tournament_select(population, fitnesses, rng)
                p2 = self._tournament_select(population, fitnesses, rng)

                # Çaprazlama
                if rng.random() < self.crossover_rate:
                    c1, c2 = self._blx_alpha_crossover(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutasyon
                c1 = self._gaussian_mutate(c1, rng)
                c2 = self._gaussian_mutate(c2, rng)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            new_pop = np.array(new_pop, dtype=np.float32)

            # Değerlendir (yeni nesilde elitler için yeniden hesap gereksiz
            #               ama tutarlılık için tüm popülasyonu değerlendiriyoruz —
            #               bütçe sayımı doğru olsun diye)
            # Elitleri yeniden değerlendirmiyoruz: ilk n_elites birey aynı,
            # fitness değerleri bilinen elite_fitness değerlerinden alınır.
            new_fitnesses = np.empty(self.pop_size, dtype=np.float64)
            new_fitnesses[: self.n_elites] = fitnesses[elite_idx]

            # Geri kalanı evaluator ile değerlendir
            for i in range(self.n_elites, self.pop_size):
                if evaluator.exhausted():
                    break
                new_fitnesses[i] = evaluator(new_pop[i])

            # Bütçe nesil ortasında biterse: dış while döngüsü
            # bir sonraki iterasyonda False değerlendirip çıkar.
            # Initialize edilmemiş fitness değerleri o noktada zaten okunmaz.

            population = new_pop
            fitnesses  = new_fitnesses