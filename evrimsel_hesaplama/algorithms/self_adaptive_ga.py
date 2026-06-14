"""
self_adaptive_ga.py
===================

Self-Adaptive Genetic Algorithm — mutasyon σ'sı genomla birlikte evrilir.

Referans:
  Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms
  and interval-schemata. Foundations of Genetic Algorithms 2.

  Beyer, H.-G., & Schwefel, H.-P. (2002). Evolution strategies — A
  comprehensive introduction. Natural Computing, 1, 3-52.

═══════════════════════════════════════════════════════════════════
KLASIK GA vs SELF-ADAPTIVE GA — Temel Fark
═══════════════════════════════════════════════════════════════════

Klasik GA (mevcut sistemde):
  genom    = [x_1, x_2, ..., x_16]      # sadece çözüm parametreleri
  mutasyon = x_yeni = x + N(0, σ²)      # σ sabit (0.1)

Self-Adaptive GA:
  genom    = [x_1, ..., x_16, σ_1, ..., σ_16]  # σ'lar da gen!
  mutasyon = σ_yeni = σ * exp(τ' * N₀ + τ * N_i)
             x_yeni = x + σ_yeni * N(0,1)

Avantajları:
  1. Her parametre kendi mutasyon ölçeğini öğrenir
     (distortion hassas → küçük σ, compressor toleranslı → büyük σ)

  2. Arama ilerledikçe σ'lar otomatik küçülür (fine-tuning)

  3. CMA-ES'in basit bir yaklaşımıdır — kovaryans matrisi olmadan

═══════════════════════════════════════════════════════════════════
EVRİM HESAPLAMA TEORİSİ — Adaptasyon Formülü
═══════════════════════════════════════════════════════════════════

Schwefel'in log-normal kuralı:

    σ_i' = σ_i * exp(τ' * N(0,1) + τ_i * N_i(0,1))

Burada:
    τ'  = 1 / sqrt(2n)        # global öğrenme oranı
    τ_i = 1 / sqrt(2*sqrt(n)) # birey-özel öğrenme oranı
    N(0,1) tüm boyutlar için aynı (global)
    N_i(0,1) her boyut için bağımsız (yerel)

n=16 için:
    τ'  ≈ 0.177
    τ_i ≈ 0.354

σ'nın küçük olması arama yarıçapını daraltır (exploitation),
büyük olması genişletir (exploration). exp() ile çarpım çünkü
σ pozitif kalmalı.
"""

from typing import Tuple
import numpy as np

from .base import Algorithm, Evaluator


class SelfAdaptiveGA(Algorithm):
    """
    Sürekli reel-değerli self-adaptive GA.

    Genotip yapısı:
        genom = (x, σ)
        x.shape = (n_params,)   — parametre değerleri  [0,1]
        σ.shape = (n_params,)   — mutasyon adım boyutları (pozitif)

    σ başlangıç değeri: 0.1 (klasik GA ile aynı σ'dan başla)
    σ alt sınır: 1e-4 (σ → 0 önle, donma yaşanmasın)
    σ üst sınır: 0.5 (kontrolsüz büyümeyi engelle)
    """

    name = "SelfAdaptiveGA"

    SIGMA_MIN = 1e-4
    SIGMA_MAX = 0.5

    def __init__(
        self,
        pop_size:        int   = 100,
        tournament_size: int   = 3,
        crossover_rate:  float = 0.9,
        crossover_alpha: float = 0.5,
        sigma_init:      float = 0.1,
        n_elites:        int   = 2,
    ):
        super().__init__()
        self.pop_size        = pop_size
        self.tournament_size = tournament_size
        self.crossover_rate  = crossover_rate
        self.crossover_alpha = crossover_alpha
        self.sigma_init      = sigma_init
        self.n_elites        = n_elites

        # Schwefel öğrenme oranları
        n = self.n_params
        self.tau_global = 1.0 / np.sqrt(2.0 * n)
        self.tau_local  = 1.0 / np.sqrt(2.0 * np.sqrt(n))

    # ────────────────────────────────────────────────────────────────
    # Yardımcılar
    # ────────────────────────────────────────────────────────────────

    def _new_individual(self, rng: np.random.Generator) -> dict:
        """Rastgele bir başlangıç bireyi: (x, σ) çifti."""
        return {
            "x":     rng.uniform(0.0, 1.0, self.n_params).astype(np.float32),
            "sigma": np.full(self.n_params, self.sigma_init, dtype=np.float32),
        }

    def _tournament_select(
        self,
        pop:       list,
        fitnesses: np.ndarray,
        rng:       np.random.Generator,
    ) -> dict:
        """k bireyden en iyisini seç (higher fitness = better)."""
        idx = rng.integers(0, len(pop), self.tournament_size)
        winner = idx[np.argmax(fitnesses[idx])]
        # Derin kopya — σ vektörünün paylaşılmasını önler
        return {
            "x":     pop[winner]["x"].copy(),
            "sigma": pop[winner]["sigma"].copy(),
        }

    def _crossover(
        self,
        p1:  dict,
        p2:  dict,
        rng: np.random.Generator,
    ) -> Tuple[dict, dict]:
        """
        Hem x hem σ üzerinde BLX-α çaprazlama uygula.
        σ'lar da çaprazlamadan etkilenir — eşleşen ebeveynlerin
        arama ölçeği yavru bireylere aktarılır.
        """
        a = self.crossover_alpha

        # x üzerinde BLX-α
        u = rng.uniform(-a, 1.0 + a, size=self.n_params).astype(np.float32)
        c1_x = p1["x"] + u * (p2["x"] - p1["x"])
        u2   = rng.uniform(-a, 1.0 + a, size=self.n_params).astype(np.float32)
        c2_x = p2["x"] + u2 * (p1["x"] - p2["x"])

        # σ üzerinde aritmetik ortalama (daha güvenli — log-uzayda)
        # Geometric mean kullanıyoruz: sqrt(σ1 * σ2)
        c1_sigma = np.sqrt(p1["sigma"] * p2["sigma"])
        c2_sigma = c1_sigma.copy()

        return (
            {"x": np.clip(c1_x, 0.0, 1.0), "sigma": c1_sigma},
            {"x": np.clip(c2_x, 0.0, 1.0), "sigma": c2_sigma},
        )

    def _mutate(self, ind: dict, rng: np.random.Generator) -> dict:
        """
        Schwefel log-normal self-adaptasyon kuralı:
            σ_yeni = σ * exp(τ' * N(0,1) + τ * N_i(0,1))
            x_yeni = x + σ_yeni * N(0,1)
        """
        # σ güncelle (log-normal)
        n0   = float(rng.standard_normal())                # global
        n_i  = rng.standard_normal(self.n_params).astype(np.float32)  # yerel
        new_sigma = ind["sigma"] * np.exp(
            self.tau_global * n0 + self.tau_local * n_i
        )
        new_sigma = np.clip(new_sigma, self.SIGMA_MIN, self.SIGMA_MAX)

        # x güncelle (Gauss, adaptif σ ile)
        noise = rng.standard_normal(self.n_params).astype(np.float32)
        new_x = ind["x"] + new_sigma * noise
        new_x = np.clip(new_x, 0.0, 1.0)

        return {"x": new_x, "sigma": new_sigma}

    # ────────────────────────────────────────────────────────────────
    # Ana akış
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        evaluator: Evaluator,
        rng:       np.random.Generator,
    ) -> None:
        # 1. Başlangıç popülasyonu
        population = [self._new_individual(rng) for _ in range(self.pop_size)]
        fitnesses  = np.array(
            [evaluator(ind["x"]) for ind in population],
            dtype=np.float64,
        )

        # 2. Nesil döngüsü
        while not evaluator.exhausted():
            # Elitleri sakla
            elite_idx     = np.argsort(-fitnesses)[: self.n_elites]
            elite_inds    = [
                {"x": population[i]["x"].copy(),
                 "sigma": population[i]["sigma"].copy()}
                for i in elite_idx
            ]
            elite_fitness = fitnesses[elite_idx].copy()

            # Yeni nesil
            new_pop = list(elite_inds)
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses, rng)
                p2 = self._tournament_select(population, fitnesses, rng)

                # Çaprazlama
                if rng.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2, rng)
                else:
                    c1, c2 = (
                        {"x": p1["x"].copy(), "sigma": p1["sigma"].copy()},
                        {"x": p2["x"].copy(), "sigma": p2["sigma"].copy()},
                    )

                # Mutasyon (her zaman uygulanır — self-adaptive'in temeli)
                c1 = self._mutate(c1, rng)
                c2 = self._mutate(c2, rng)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            # Yeni nesli değerlendir
            new_fitnesses = np.empty(self.pop_size, dtype=np.float64)
            new_fitnesses[: self.n_elites] = elite_fitness

            for i in range(self.n_elites, self.pop_size):
                if evaluator.exhausted():
                    break
                new_fitnesses[i] = evaluator(new_pop[i]["x"])

            population = new_pop
            fitnesses  = new_fitnesses

    # ────────────────────────────────────────────────────────────────
    def build_result(self, evaluator, seed, extra=None):
        """RunResult'a son popülasyonun σ istatistiklerini ekle."""
        # Bu noktada population erişimi yok — extra'ya sadece yapılandırma ekle
        extra = {
            **(extra or {}),
            "self_adaptive_ga": {
                "tau_global": float(self.tau_global),
                "tau_local":  float(self.tau_local),
                "sigma_init": float(self.sigma_init),
            },
        }
        return super().build_result(evaluator, seed, extra)