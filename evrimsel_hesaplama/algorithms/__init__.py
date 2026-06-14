"""
Algoritma paketi.

İçerik:
  - base:               Algorithm soyut sınıfı, Evaluator, RunResult
  - random_search:      RandomSearch (baseline)
  - classic_ga:         ClassicGA (turnuva + BLX-α + Gauss mutasyon)
  - self_adaptive_ga:   SelfAdaptiveGA (R1 — Schwefel log-normal σ adaptasyonu)
  - map_elites:         MAPElites (Quality Diversity)
  - cma_es:             CMAES (adaptif kovaryans)
  - cma_me:             CMAME (R3 — hibrit MAP-Elites + CMA-ES)
"""

from .base import (
    Algorithm,
    Evaluator,
    RunResult,
    BudgetExhausted,
)
from .random_search     import RandomSearch
from .classic_ga        import ClassicGA
from .self_adaptive_ga  import SelfAdaptiveGA
from .map_elites        import MAPElites
from .cma_es            import CMAES
from .cma_me            import CMAME

__all__ = [
    "Algorithm", "Evaluator", "RunResult", "BudgetExhausted",
    "RandomSearch", "ClassicGA", "SelfAdaptiveGA",
    "MAPElites", "CMAES", "CMAME",
]