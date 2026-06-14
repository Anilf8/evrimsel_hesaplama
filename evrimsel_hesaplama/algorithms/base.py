"""
base.py
=======

Tüm evrimsel algoritmaların kalıtacağı soyut taban sınıf.

Tasarım kararları:
  1. Tek bir 'run()' metodu — kara kutu arayüz.
  2. Her algoritma kendi history (yakınsama eğrisi) listesini doldurur.
  3. Fitness değerlendirmesi bütçeye sayılır — kapanış adil karşılaştırma için.
  4. Evaluator sarmalayıcı, sayaç ve geçmiş kayıt mantığını gizler.

Bu yapıyı seçtik çünkü:
  - Random / GA / MAP-Elites / CMA-ES'in dış arayüzü aynı olur.
  - Runner kodu tek bir 'algorithm.run()' ile her birini çalıştırır.
  - Yakınsama eğrisi tek noktadan toplandığı için karşılaştırma temiz.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np

from core import N_PARAMS


# ────────────────────────────────────────────────────────────────────
# Sonuç veri yapısı
# ────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Bir algoritma koşumunun tüm çıktısını barındırır."""

    algorithm_name: str
    seed: int
    best_x: np.ndarray
    best_fitness: float                  # Daha yüksek = daha iyi (fitness konvansiyonu)
    best_loss: float                     # Daha düşük = daha iyi (raw MR-STFT)
    n_evals: int
    history_evals:   List[int]           = field(default_factory=list)
    history_best:    List[float]         = field(default_factory=list)
    extra: dict                          = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.algorithm_name:14s} | seed={self.seed:2d} | "
            f"evals={self.n_evals:5d} | "
            f"best_loss={self.best_loss:.5f} | "
            f"best_fitness={self.best_fitness:+.5f}"
        )


# ────────────────────────────────────────────────────────────────────
# Evaluator — fitness değerlendirmesi + bütçe sayacı
# ────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Fitness değerlendirme katmanı.

    Görevleri:
      1. fitness_fn'i çağırmak (kullanıcı tanımlı kara kutu)
      2. Her çağrıyı sayaç ile saymak
      3. En iyi çözümü takip etmek
      4. Belirli aralıklarda yakınsama geçmişine kayıt eklemek
      5. Budget aşıldığında StopIteration fırlatmak

    Bu sınıf algoritmaların 'kaç değerlendirme yaptım?' kafa karışıklığını
    çözer — algoritma kodları sadece evaluator(x) çağırır, gerisi otomatik.
    """

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        budget: int = 10_000,
        record_every: int = 100,
        higher_is_better: bool = True,
    ):
        self.fitness_fn       = fitness_fn
        self.budget           = budget
        self.record_every     = record_every
        self.higher_is_better = higher_is_better

        # Durum
        self.n_evals: int = 0
        self.best_x: Optional[np.ndarray]   = None
        self.best_fitness: float            = -np.inf if higher_is_better else np.inf

        # Yakınsama geçmişi
        self.history_evals: List[int]   = []
        self.history_best:  List[float] = []

    # ------------------------------------------------------------------

    def is_better(self, a: float, b: float) -> bool:
        return a > b if self.higher_is_better else a < b

    def remaining(self) -> int:
        return self.budget - self.n_evals

    def exhausted(self) -> bool:
        return self.n_evals >= self.budget

    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> float:
        """Bir bireyi değerlendir, en iyiyi güncelle, geçmişe kayıt at."""
        if self.exhausted():
            raise BudgetExhausted(f"Bütçe doldu: {self.budget}")

        f = float(self.fitness_fn(x))
        self.n_evals += 1

        # En iyi güncelle
        if self.is_better(f, self.best_fitness):
            self.best_fitness = f
            self.best_x = x.copy()

        # Periyodik kayıt
        if self.n_evals == 1 or self.n_evals % self.record_every == 0 \
                or self.exhausted():
            self.history_evals.append(self.n_evals)
            self.history_best.append(self.best_fitness)

        return f

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Birden çok bireyi sırayla değerlendir, fitness dizisini döndür."""
        return np.array([self(x) for x in X], dtype=np.float64)


class BudgetExhausted(Exception):
    """Fitness değerlendirme bütçesi tükendiğinde fırlatılır."""
    pass


# ────────────────────────────────────────────────────────────────────
# Soyut Algorithm sınıfı
# ────────────────────────────────────────────────────────────────────

class Algorithm(ABC):
    """
    Tüm evrimsel algoritmaların kalıttığı soyut taban.

    Alt sınıflar şu sözleşmeyi yerine getirir:
      - name özniteliği: string identifier
      - run(evaluator, rng) metodu: algoritma mantığı

    Runner şu şekilde kullanır:
        algo  = ClassicGA(pop_size=100)
        eval_ = Evaluator(fitness_fn, budget=10000)
        rng   = np.random.default_rng(seed)
        algo.run(eval_, rng)
        result = algo.build_result(seed)
    """

    name: str = "abstract"
    n_params: int = N_PARAMS

    # ------------------------------------------------------------------
    @abstractmethod
    def run(
        self,
        evaluator: Evaluator,
        rng: np.random.Generator,
    ) -> None:
        """Algoritma ana döngüsü. evaluator(x) çağrıları yapar."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def build_result(
        self,
        evaluator: Evaluator,
        seed: int,
        extra: Optional[dict] = None,
    ) -> RunResult:
        """Koşum bittikten sonra evaluator'dan RunResult inşa eder."""
        best_fitness = evaluator.best_fitness
        # Raw loss = -fitness (higher_is_better=True olduğu varsayımıyla)
        best_loss = (-best_fitness if evaluator.higher_is_better
                     else best_fitness)
        return RunResult(
            algorithm_name = self.name,
            seed           = seed,
            best_x         = evaluator.best_x.copy() if evaluator.best_x is not None
                             else np.zeros(self.n_params),
            best_fitness   = best_fitness,
            best_loss      = best_loss,
            n_evals        = evaluator.n_evals,
            history_evals  = list(evaluator.history_evals),
            history_best   = list(evaluator.history_best),
            extra          = extra or {},
        )

    # ------------------------------------------------------------------
    def safe_run(
        self,
        evaluator: Evaluator,
        rng: np.random.Generator,
    ) -> None:
        """Bütçe biterse temiz çıkış sağlayan run() sarmalayıcı."""
        try:
            self.run(evaluator, rng)
        except BudgetExhausted:
            pass

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<Algorithm name='{self.name}' n_params={self.n_params}>"