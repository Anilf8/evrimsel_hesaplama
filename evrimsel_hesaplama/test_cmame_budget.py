"""
test_cmame_budget.py
====================

CMA-ME'nin Adım 7'de MAPElites'i tam yenememesinin sebebini araştır.

Hipotez: Düşük bütçede CMA-ES emitter'ları kovaryans öğrenemeden restart
         oluyor. Yüksek bütçede CMA-ME öne geçer.

Test: 2 bütçe (2000, 5000) × 2 algoritma × 3 seed.
"""

import warnings, time
import numpy as np
warnings.filterwarnings("ignore")

from core import (
    EffectChain, MultiResolutionSTFTLoss, IDMTLoader,
    get_preset_vector, compute_behavior, BehaviorSpace,
)
from algorithms import Evaluator, MAPElites, CMAME

SR = 22050
BUDGETS = [2000, 5000]
N_SEEDS = 3

loader = IDMTLoader(None, SR, verbose=False)
chain  = EffectChain(SR)
loss   = MultiResolutionSTFTLoss()
dry    = loader.get_fixed_dry(0.7, seed=0)
target = chain.process(dry, get_preset_vector("crunch", seed=0))
loss.cache_target(target)

def fitness_fn(x):
    return -loss.compute_fast(chain.process(dry, x))

_bc = {}
def behavior_fn(x):
    k = x.tobytes()
    if k not in _bc:
        _bc[k] = compute_behavior(chain.process(dry, x), SR)
    return _bc[k]


print()
print("=" * 70)
print(" CMA-ME bütçe duyarlılık testi")
print("=" * 70)

t_start = time.time()
results = {}

for budget in BUDGETS:
    results[budget] = {"MAPElites": [], "CMA-ME": []}
    print(f"\n--- Bütçe: {budget} ---")

    for s in range(N_SEEDS):
        # MAPElites
        _bc.clear()
        a = MAPElites(behavior_fn=behavior_fn,
                      behavior_space=BehaviorSpace(grid_size=20),
                      n_init=200, mutation_sigma=0.08)
        e = Evaluator(fitness_fn, budget=budget, record_every=500)
        a.safe_run(e, np.random.default_rng(s))
        r = a.build_result(e, seed=s)
        results[budget]["MAPElites"].append({
            "loss": r.best_loss,
            "cov":  r.extra["map_elites"]["coverage"],
            "qd":   r.extra["map_elites"]["qd_score"],
        })

        # CMA-ME — daha uzun emitter ömrü için sigma0 düşür
        _bc.clear()
        a = CMAME(behavior_fn=behavior_fn,
                  behavior_space=BehaviorSpace(grid_size=20),
                  n_init=200, sigma0=0.15)
        e = Evaluator(fitness_fn, budget=budget, record_every=500)
        a.safe_run(e, np.random.default_rng(s))
        r = a.build_result(e, seed=s)
        results[budget]["CMA-ME"].append({
            "loss": r.best_loss,
            "cov":  r.extra["map_elites"]["coverage"],
            "qd":   r.extra["map_elites"]["qd_score"],
        })

        print(f"  seed={s}: "
              f"ME(qd={results[budget]['MAPElites'][-1]['qd']:.0f}, "
              f"cov={results[budget]['MAPElites'][-1]['cov']*100:.0f}%) | "
              f"CMA-ME(qd={results[budget]['CMA-ME'][-1]['qd']:.0f}, "
              f"cov={results[budget]['CMA-ME'][-1]['cov']*100:.0f}%)")

print()
print("=" * 70)
print(" Özet — medyan QD-Score (yüksek = iyi)")
print("=" * 70)
print(f"  {'Bütçe':>6s} | {'MAPElites':>12s} | {'CMA-ME':>12s} | Kazanan")
print("  " + "-" * 50)
for budget in BUDGETS:
    me_qd = np.median([r["qd"] for r in results[budget]["MAPElites"]])
    ce_qd = np.median([r["qd"] for r in results[budget]["CMA-ME"]])
    winner = "CMA-ME" if ce_qd > me_qd else "MAPElites"
    print(f"  {budget:>6d} | {me_qd:>12.1f} | {ce_qd:>12.1f} | {winner}")

print()
print(f"  {'Bütçe':>6s} | {'ME cov':>12s} | {'CMA-ME cov':>12s} | Kazanan")
print("  " + "-" * 50)
for budget in BUDGETS:
    me_cov = np.median([r["cov"] for r in results[budget]["MAPElites"]])
    ce_cov = np.median([r["cov"] for r in results[budget]["CMA-ME"]])
    winner = "CMA-ME" if ce_cov > me_cov else "MAPElites"
    print(f"  {budget:>6d} | {me_cov*100:>11.1f}% | {ce_cov*100:>11.1f}% | {winner}")

print()
print(f"Toplam süre: {(time.time()-t_start)/60:.1f} dk")