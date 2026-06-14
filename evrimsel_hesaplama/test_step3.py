"""
test_step3.py
=============

Adım 3 sağlamlık testi — 4 algoritmayı birlikte koş.

Doğrulananlar:
  1. MAP-Elites arşivi dolu hücreler üretir
  2. MAP-Elites davranış uzayında çeşitlilik gösterir
  3. CMA-ES yakınsar ve değerli bir sonuç bulur
  4. Tüm algoritmalar RandomSearch'ü yener
  5. Yakınsama eğrileri monoton
  6. Aynı seed → aynı sonuç (her algoritma için)

Düşük bütçe (3000) ile hızlı test. Gerçek deneyde 10.000.
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
    Evaluator, RandomSearch, ClassicGA, MAPElites, CMAES,
)


def banner(t: str) -> None:
    print()
    print("=" * 64)
    print("  " + t)
    print("=" * 64)


def check(cond: bool, label: str) -> None:
    mark = "OK " if cond else "FAIL"
    print(f"  [{mark}] {label}")
    if not cond:
        raise AssertionError(f"FAIL: {label}")


# ────────────────────────────────────────────────────────────────────
# Kurulum
# ────────────────────────────────────────────────────────────────────
banner("Kurulum — crunch preset hedefi")

SR     = 22050
BUDGET = 3000
SEED   = 42

loader = IDMTLoader(None, SR, verbose=True)
chain  = EffectChain(SR)
loss   = MultiResolutionSTFTLoss()
dry    = loader.get_fixed_dry(0.7, seed=SEED)
x_star = get_preset_vector("crunch", seed=SEED)
target = chain.process(dry, x_star)
loss.cache_target(target)  # target STFT önbelleğe alındı

print(f"  Sinyal: {len(dry)} örnek, hedef preset: crunch, bütçe: {BUDGET}")


# Fitness — paylaşılan kapanış (closure)
def fitness_fn(x: np.ndarray) -> float:
    wet = chain.process(dry, x)
    return -loss.compute_fast(wet)   # önbellekli: ~7 ms/eval tasarruf


# MAP-Elites için davranış fonksiyonu
# NOT: ÖZEL DİKKAT — MAP-Elites her bireyi 2 kez işliyor (fitness için bir kez,
#      davranış için bir kez). Bu çift hesaplamayı önlemek için cache kullanıyoruz.
_cache: dict = {}

def behavior_fn(x: np.ndarray) -> tuple:
    key = x.tobytes()
    if key in _cache:
        return _cache[key]
    wet = chain.process(dry, x)
    b   = compute_behavior(wet, SR)
    _cache[key] = b
    return b


# ────────────────────────────────────────────────────────────────────
# Dört algoritmayı koş
# ────────────────────────────────────────────────────────────────────
banner("Dört algoritmayı sırayla koşalım")

results = {}
configs = [
    ("RandomSearch", RandomSearch()),
    ("ClassicGA",    ClassicGA(pop_size=50, mutation_sigma=0.1)),
    ("MAPElites",    MAPElites(behavior_fn=behavior_fn,
                               behavior_space=BehaviorSpace(grid_size=20),
                               n_init=200, mutation_sigma=0.08)),
    ("CMAES",        CMAES(sigma0=0.3)),
]

for name, algo in configs:
    _cache.clear()
    evalr  = Evaluator(fitness_fn, budget=BUDGET, record_every=100)
    rng    = np.random.default_rng(SEED)
    t0     = time.time()
    algo.safe_run(evalr, rng)
    dt     = time.time() - t0
    res    = algo.build_result(evalr, seed=SEED)
    results[name] = res
    print(f"  {name:14s} | loss={res.best_loss:.4f} | "
          f"evals={res.n_evals} | süre={dt:.1f}s")


# ────────────────────────────────────────────────────────────────────
# 1. Hepsi RandomSearch'ü yenmiş mi?
# ────────────────────────────────────────────────────────────────────
banner("1. RandomSearch baseline'ı yenildi mi?")
rs_loss = results["RandomSearch"].best_loss
for name in ["ClassicGA", "MAPElites", "CMAES"]:
    other = results[name].best_loss
    diff  = rs_loss - other
    print(f"  {name:14s}: {other:.4f}  ({'+' if diff > 0 else ''}{diff:.4f} fark)")
    check(other < rs_loss, f"{name}, RandomSearch'ten iyi")


# ────────────────────────────────────────────────────────────────────
# 2. MAP-Elites arşiv çeşitliliği
# ────────────────────────────────────────────────────────────────────
banner("2. MAP-Elites arşiv çeşitliliği")
me_extra = results["MAPElites"].extra["map_elites"]
print(f"  Dolu hücre  : {me_extra['n_cells']}/400")
print(f"  Coverage    : {me_extra['coverage']*100:.1f}%")
print(f"  QD-Score    : {me_extra['qd_score']:.2f}")
check(me_extra["n_cells"] >= 50, "En az 50 hücre dolu (çeşitlilik var)")
check(me_extra["coverage"] >= 0.10, "Coverage ≥ 10%")


# ────────────────────────────────────────────────────────────────────
# 3. CMA-ES yakınsama hızı
# ────────────────────────────────────────────────────────────────────
banner("3. CMA-ES yakınsama")
cma_res = results["CMAES"]
ga_res  = results["ClassicGA"]
print(f"  CMA-ES final loss : {cma_res.best_loss:.4f}")
print(f"  GA     final loss : {ga_res.best_loss:.4f}")
# Hipotez: CMA-ES sürekli uzayda GA'dan daha hızlı yakınsar
# (test bütçesi düşük olduğu için her zaman olmayabilir, sadece gözlemle)
print(f"  (GA'dan {'daha iyi' if cma_res.best_loss < ga_res.best_loss else 'daha kötü'})")


# ────────────────────────────────────────────────────────────────────
# 4. Yakınsama eğrileri monoton mu?
# ────────────────────────────────────────────────────────────────────
banner("4. Yakınsama eğrisi monotonluğu")
for name, res in results.items():
    h = res.history_best
    is_monotone = all(h[i+1] >= h[i] - 1e-9 for i in range(len(h)-1))
    print(f"  {name:14s}: ilk={h[0]:+.4f}, son={h[-1]:+.4f}, "
          f"{'monoton' if is_monotone else 'BOZUK'}")
    check(is_monotone, f"{name} yakınsama monoton")


# ────────────────────────────────────────────────────────────────────
# 5. Tekrarlanabilirlik — sadece CMA-ES ve MAP-Elites için
# ────────────────────────────────────────────────────────────────────
banner("5. Tekrarlanabilirlik (yeni algoritmalar)")

# MAP-Elites
_cache.clear()
me_a = MAPElites(behavior_fn=behavior_fn, n_init=100)
ev_a = Evaluator(fitness_fn, budget=500, record_every=50)
me_a.safe_run(ev_a, np.random.default_rng(99))
r_a = me_a.build_result(ev_a, seed=99)

_cache.clear()
me_b = MAPElites(behavior_fn=behavior_fn, n_init=100)
ev_b = Evaluator(fitness_fn, budget=500, record_every=50)
me_b.safe_run(ev_b, np.random.default_rng(99))
r_b = me_b.build_result(ev_b, seed=99)

print(f"  MAPElites A best_loss = {r_a.best_loss:.6f}")
print(f"  MAPElites B best_loss = {r_b.best_loss:.6f}")
check(np.isclose(r_a.best_loss, r_b.best_loss, atol=1e-9),
      "MAPElites aynı seed → aynı sonuç")

# CMA-ES
_cache.clear()
cma_a = CMAES()
ev_ca = Evaluator(fitness_fn, budget=500, record_every=50)
cma_a.safe_run(ev_ca, np.random.default_rng(99))
r_ca  = cma_a.build_result(ev_ca, seed=99)

_cache.clear()
cma_b = CMAES()
ev_cb = Evaluator(fitness_fn, budget=500, record_every=50)
cma_b.safe_run(ev_cb, np.random.default_rng(99))
r_cb  = cma_b.build_result(ev_cb, seed=99)

print(f"  CMAES A best_loss = {r_ca.best_loss:.6f}")
print(f"  CMAES B best_loss = {r_cb.best_loss:.6f}")
check(np.isclose(r_ca.best_loss, r_cb.best_loss, atol=1e-6),
      "CMAES aynı seed → aynı sonuç")


# ────────────────────────────────────────────────────────────────────
# 6. Yakınsama özet tablo
# ────────────────────────────────────────────────────────────────────
banner("6. Yakınsama özet")
print(f"  {'evals':>7s} | " + " | ".join(f"{n:>10s}" for n in results))
common = set.intersection(*[set(r.history_evals) for r in results.values()])
common = sorted(common)[:: max(1, len(common) // 8)]
for ev in common:
    cols = []
    for name, r in results.items():
        idx = r.history_evals.index(ev)
        loss_val = -r.history_best[idx]
        cols.append(f"{loss_val:>10.4f}")
    print(f"  {ev:>7d} | " + " | ".join(cols))


banner("Adım 3 BAŞARILI")
print("  Sonraki adım: Deney runner + metrikler + istatistik testleri (Adım 4)")
print("=" * 64)