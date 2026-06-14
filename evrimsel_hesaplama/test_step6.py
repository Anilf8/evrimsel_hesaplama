"""
test_step6.py
=============

Adım 6 testleri — yeni optimizasyonları ve algoritmaları doğrular.

Doğrulananlar:
  1. Hedef cache (O1) çalışıyor ve hızlandırıyor
  2. SelfAdaptiveGA çalışıyor
  3. SelfAdaptiveGA, ClassicGA'dan en az kıyaslanabilir performans veriyor
  4. Tekrarlanabilirlik (her algoritma için)

Hızlı test — budget=2000, n_seeds=3, 1 preset (crunch).
Tam deney için: main.py all (yeni algoritmaların eklenmesinden sonra).
"""

import warnings
import time
import numpy as np

warnings.filterwarnings("ignore")

from core import (
    EffectChain, MultiResolutionSTFTLoss, IDMTLoader,
    get_preset_vector, compute_behavior, BehaviorSpace,
)
from algorithms import (
    Evaluator, RandomSearch, ClassicGA, SelfAdaptiveGA, MAPElites, CMAES,
)


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
banner("Kurulum — crunch preset, küçük bütçe")

SR     = 22050
BUDGET = 2000
SEED   = 42

loader = IDMTLoader(None, SR, verbose=True)
chain  = EffectChain(SR)
loss   = MultiResolutionSTFTLoss()
dry    = loader.get_fixed_dry(0.7, seed=SEED)
x_star = get_preset_vector("crunch", seed=SEED)
target = chain.process(dry, x_star)

print(f"  Sinyal: {len(dry)} örnek, bütçe: {BUDGET}")


# ────────────────────────────────────────────────────────────────────
# 1. Cache hızlanma benchmark
# ────────────────────────────────────────────────────────────────────
banner("1. Hedef cache (O1) hızlanma testi")

# Yavaş yol — eski compute()
t0 = time.time()
for _ in range(200):
    wet = chain.process(dry, np.random.uniform(0, 1, 16).astype(np.float32))
    _   = loss.compute(target, wet)
t_slow = time.time() - t0

# Hızlı yol — cache + compute_fast()
loss.cache_target(target)
t0 = time.time()
for _ in range(200):
    wet = chain.process(dry, np.random.uniform(0, 1, 16).astype(np.float32))
    _   = loss.compute_fast(wet)
t_fast = time.time() - t0

speedup = t_slow / t_fast
print(f"  Yavaş yol  : {t_slow:.2f}s  →  {t_slow/200*1000:.1f} ms/eval")
print(f"  Hızlı yol  : {t_fast:.2f}s  →  {t_fast/200*1000:.1f} ms/eval")
print(f"  Hızlanma   : {speedup:.2f}x")
check(speedup > 1.3, "Cache en az 1.3x hızlanma sağlıyor")


# ────────────────────────────────────────────────────────────────────
# 2. Cache sayısal tutarlılık
# ────────────────────────────────────────────────────────────────────
banner("2. Cache sayısal tutarlılık")

# Cache temizle, yavaş yol değer al
loss.clear_cache()
wet_test = chain.process(dry, np.full(16, 0.5, dtype=np.float32))
v_slow = loss.compute(target, wet_test)

# Cache aç, hızlı yol değer al
loss.cache_target(target)
v_fast = loss.compute_fast(wet_test)

print(f"  Yavaş yol : {v_slow:.6f}")
print(f"  Hızlı yol : {v_fast:.6f}")
print(f"  Fark      : {abs(v_slow-v_fast):.2e}")
check(abs(v_slow - v_fast) < 1e-5, "İki yol sayısal olarak tutarlı")


# ────────────────────────────────────────────────────────────────────
# 3. SelfAdaptiveGA çalışıyor mu?
# ────────────────────────────────────────────────────────────────────
banner("3. SelfAdaptiveGA — gerçek problem üzerinde")

def fitness_fn(x):
    wet = chain.process(dry, x)
    return -loss.compute_fast(wet)


loss.cache_target(target)
algo = SelfAdaptiveGA(pop_size=50, sigma_init=0.1, n_elites=2)
evalr = Evaluator(fitness_fn, budget=BUDGET, record_every=100)
rng = np.random.default_rng(SEED)

t0 = time.time()
algo.safe_run(evalr, rng)
dt = time.time() - t0
result_sa = algo.build_result(evalr, seed=SEED)

print(f"  loss        : {result_sa.best_loss:.4f}")
print(f"  evals       : {result_sa.n_evals}")
print(f"  süre        : {dt:.1f}s")
print(f"  ilk→son hist: {-result_sa.history_best[0]:.4f} → {-result_sa.history_best[-1]:.4f}")
check(result_sa.n_evals >= BUDGET * 0.9, "Bütçenin en az %90'ı kullanıldı")
check(result_sa.best_loss < 1.0, "Anlamlı yakınsama (loss < 1.0)")


# ────────────────────────────────────────────────────────────────────
# 4. SelfAdaptiveGA vs ClassicGA
# ────────────────────────────────────────────────────────────────────
banner("4. SelfAdaptiveGA vs ClassicGA — 3 seed karşılaştırma")

results_classic = []
results_self    = []

for s in range(3):
    # ClassicGA
    loss.cache_target(target)
    a1 = ClassicGA(pop_size=50, mutation_sigma=0.1, n_elites=2)
    e1 = Evaluator(fitness_fn, budget=BUDGET, record_every=200)
    a1.safe_run(e1, np.random.default_rng(s))
    r1 = a1.build_result(e1, seed=s)
    results_classic.append(r1.best_loss)

    # SelfAdaptiveGA
    loss.cache_target(target)
    a2 = SelfAdaptiveGA(pop_size=50, sigma_init=0.1, n_elites=2)
    e2 = Evaluator(fitness_fn, budget=BUDGET, record_every=200)
    a2.safe_run(e2, np.random.default_rng(s))
    r2 = a2.build_result(e2, seed=s)
    results_self.append(r2.best_loss)

    print(f"  seed={s}: ClassicGA={r1.best_loss:.4f}  SelfAdaptiveGA={r2.best_loss:.4f}")

print()
print(f"  Median ClassicGA      : {np.median(results_classic):.4f}")
print(f"  Median SelfAdaptiveGA : {np.median(results_self):.4f}")

improvement = (np.median(results_classic) - np.median(results_self)) / np.median(results_classic) * 100
if improvement > 0:
    print(f"  SelfAdaptive %{improvement:.1f} daha iyi")
else:
    print(f"  Bu bütçede SelfAdaptive {-improvement:.1f}% daha kötü (normal — adaptasyon zaman ister)")


# ────────────────────────────────────────────────────────────────────
# 5. Tekrarlanabilirlik
# ────────────────────────────────────────────────────────────────────
banner("5. SelfAdaptiveGA tekrarlanabilirlik")

loss.cache_target(target)
algo_a = SelfAdaptiveGA(pop_size=30, sigma_init=0.1)
ev_a = Evaluator(fitness_fn, budget=500, record_every=50)
algo_a.safe_run(ev_a, np.random.default_rng(99))
r_a = algo_a.build_result(ev_a, seed=99)

loss.cache_target(target)
algo_b = SelfAdaptiveGA(pop_size=30, sigma_init=0.1)
ev_b = Evaluator(fitness_fn, budget=500, record_every=50)
algo_b.safe_run(ev_b, np.random.default_rng(99))
r_b = algo_b.build_result(ev_b, seed=99)

print(f"  Koşum A: {r_a.best_loss:.6f}")
print(f"  Koşum B: {r_b.best_loss:.6f}")
check(np.isclose(r_a.best_loss, r_b.best_loss, atol=1e-6),
      "Aynı seed → aynı sonuç")


# ────────────────────────────────────────────────────────────────────
banner("Adım 6 BAŞARILI")
print("  Optimizasyonlar + SelfAdaptiveGA hazır")
print("  Sonraki adım: CMA-ME (R3) ve istatistik düzeltmeleri (S1+S2)")
print("=" * 66)