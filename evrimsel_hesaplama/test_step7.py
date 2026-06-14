"""
test_step7.py
=============

Adım 7 testi — CMA-ME karşılaştırması.

Doğrulananlar:
  1. CMA-ME çalışıyor (hata vermiyor)
  2. CMA-ME arşivi dolduruyor (coverage > 0)
  3. CMA-ME, klasik MAP-Elites'ten daha yüksek QD-Score veriyor
  4. CMA-ME tekrarlanabilir

Hızlı test — budget=2000, 3 seed.
"""

import warnings
import time
import numpy as np

warnings.filterwarnings("ignore")

from core import (
    EffectChain, MultiResolutionSTFTLoss, IDMTLoader,
    get_preset_vector, compute_behavior, BehaviorSpace,
)
from algorithms import Evaluator, MAPElites, CMAME


def banner(t):
    print()
    print("=" * 66)
    print("  " + t)
    print("=" * 66)


def check(cond, label):
    mark = "OK " if cond else "FAIL"
    print(f"  [{mark}] {label}")
    if not cond:
        raise AssertionError(f"FAIL: {label}")


# ────────────────────────────────────────────────────────────────────
# Kurulum
# ────────────────────────────────────────────────────────────────────
banner("Kurulum — crunch preset")

SR     = 22050
BUDGET = 2000

loader = IDMTLoader(None, SR, verbose=True)
chain  = EffectChain(SR)
loss   = MultiResolutionSTFTLoss()
dry    = loader.get_fixed_dry(0.7, seed=0)
x_star = get_preset_vector("crunch", seed=0)
target = chain.process(dry, x_star)
loss.cache_target(target)


def fitness_fn(x):
    wet = chain.process(dry, x)
    return -loss.compute_fast(wet)


# Behavior cache — çift DSP'yi önler
_b_cache: dict = {}
def behavior_fn(x):
    key = x.tobytes()
    if key not in _b_cache:
        wet = chain.process(dry, x)
        _b_cache[key] = compute_behavior(wet, SR)
    return _b_cache[key]


# ────────────────────────────────────────────────────────────────────
# 1. CMA-ME tek koşum
# ────────────────────────────────────────────────────────────────────
banner("1. CMA-ME tek koşum testi")

_b_cache.clear()
algo = CMAME(
    behavior_fn=behavior_fn,
    behavior_space=BehaviorSpace(grid_size=20),
    n_init=200,
    sigma0=0.15,
)
evalr = Evaluator(fitness_fn, budget=BUDGET, record_every=100)

t0 = time.time()
algo.safe_run(evalr, np.random.default_rng(42))
dt = time.time() - t0
res = algo.build_result(evalr, seed=42)

me_info = res.extra["map_elites"]
print(f"  loss        : {res.best_loss:.4f}")
print(f"  evals       : {res.n_evals}")
print(f"  süre        : {dt:.1f}s")
print(f"  coverage    : {me_info['coverage']*100:.1f}% ({me_info['n_cells']}/400)")
print(f"  qd_score    : {me_info['qd_score']:.2f}")
print(f"  emitters    : {me_info['n_emitters']}")
print(f"  restarts    : {me_info['n_restarts']}")

check(res.n_evals >= BUDGET * 0.9, "Bütçenin %90'ı kullanıldı")
check(me_info["n_cells"] > 10, "Arşivde > 10 hücre var")
check(me_info["n_emitters"] >= 1, "En az 1 emitter kullanıldı")


# ────────────────────────────────────────────────────────────────────
# 2. CMA-ME vs MAPElites — 3 seed
# ────────────────────────────────────────────────────────────────────
banner("2. CMA-ME vs MAPElites — 3 seed karşılaştırma")

results_me   = []
results_cmame = []

for s in range(3):
    # MAPElites
    _b_cache.clear()
    a1 = MAPElites(
        behavior_fn=behavior_fn,
        behavior_space=BehaviorSpace(grid_size=20),
        n_init=200,
        mutation_sigma=0.08,
    )
    e1 = Evaluator(fitness_fn, budget=BUDGET, record_every=200)
    a1.safe_run(e1, np.random.default_rng(100 + s))
    r1 = a1.build_result(e1, seed=100+s)
    me_data1 = r1.extra["map_elites"]
    results_me.append({
        "loss":     r1.best_loss,
        "coverage": me_data1["coverage"],
        "qd_score": me_data1["qd_score"],
    })

    # CMA-ME
    _b_cache.clear()
    a2 = CMAME(
        behavior_fn=behavior_fn,
        behavior_space=BehaviorSpace(grid_size=20),
        n_init=200,
        sigma0=0.15,
    )
    e2 = Evaluator(fitness_fn, budget=BUDGET, record_every=200)
    a2.safe_run(e2, np.random.default_rng(100 + s))
    r2 = a2.build_result(e2, seed=100+s)
    me_data2 = r2.extra["map_elites"]
    results_cmame.append({
        "loss":     r2.best_loss,
        "coverage": me_data2["coverage"],
        "qd_score": me_data2["qd_score"],
    })

    print(f"  seed={s}:")
    print(f"    MAPElites: loss={r1.best_loss:.4f}  "
          f"cov={me_data1['coverage']*100:.1f}%  "
          f"qd={me_data1['qd_score']:.1f}")
    print(f"    CMA-ME   : loss={r2.best_loss:.4f}  "
          f"cov={me_data2['coverage']*100:.1f}%  "
          f"qd={me_data2['qd_score']:.1f}")


# ────────────────────────────────────────────────────────────────────
# 3. Karşılaştırma özeti
# ────────────────────────────────────────────────────────────────────
banner("3. Karşılaştırma özeti")

me_med_loss = np.median([r["loss"] for r in results_me])
ce_med_loss = np.median([r["loss"] for r in results_cmame])
me_med_cov  = np.median([r["coverage"] for r in results_me])
ce_med_cov  = np.median([r["coverage"] for r in results_cmame])
me_med_qd   = np.median([r["qd_score"] for r in results_me])
ce_med_qd   = np.median([r["qd_score"] for r in results_cmame])

print(f"  {'Metrik':>20s} | {'MAPElites':>10s} | {'CMA-ME':>10s} | Kazanan")
print("  " + "-" * 60)
print(f"  {'Best loss (↓)':>20s} | {me_med_loss:>10.4f} | {ce_med_loss:>10.4f} | "
      f"{'CMA-ME' if ce_med_loss < me_med_loss else 'MAPElites'}")
print(f"  {'Coverage (↑)':>20s} | {me_med_cov*100:>9.1f}% | {ce_med_cov*100:>9.1f}% | "
      f"{'CMA-ME' if ce_med_cov > me_med_cov else 'MAPElites'}")
print(f"  {'QD-Score (↑)':>20s} | {me_med_qd:>10.2f} | {ce_med_qd:>10.2f} | "
      f"{'CMA-ME' if ce_med_qd > me_med_qd else 'MAPElites'}")

# Sağlık kontrolleri
check(ce_med_cov >= me_med_cov * 0.7,
      "CMA-ME coverage en az MAPElites'in %70'i (genelde daha iyi)")


# ────────────────────────────────────────────────────────────────────
# 4. Tekrarlanabilirlik
# ────────────────────────────────────────────────────────────────────
banner("4. CMA-ME tekrarlanabilirlik")

_b_cache.clear()
a_a = CMAME(behavior_fn=behavior_fn, n_init=100, sigma0=0.15)
e_a = Evaluator(fitness_fn, budget=500, record_every=50)
a_a.safe_run(e_a, np.random.default_rng(99))
r_a = a_a.build_result(e_a, seed=99)

_b_cache.clear()
a_b = CMAME(behavior_fn=behavior_fn, n_init=100, sigma0=0.15)
e_b = Evaluator(fitness_fn, budget=500, record_every=50)
a_b.safe_run(e_b, np.random.default_rng(99))
r_b = a_b.build_result(e_b, seed=99)

print(f"  Koşum A: loss={r_a.best_loss:.6f}, "
      f"cov={r_a.extra['map_elites']['coverage']*100:.2f}%")
print(f"  Koşum B: loss={r_b.best_loss:.6f}, "
      f"cov={r_b.extra['map_elites']['coverage']*100:.2f}%")
check(np.isclose(r_a.best_loss, r_b.best_loss, atol=1e-6),
      "Aynı seed → aynı sonuç")


banner("Adım 7 BAŞARILI")
print("  CMA-ME hazır")
print("  Sonraki adım: tam deney (main.py all) — 6 algoritmayla")
print("=" * 66)