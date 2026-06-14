"""
test_step2.py
=============

Adım 2 sağlamlık testi.

Doğrulananlar:
  1. Algorithm soyut sınıfı doğru çalışıyor
  2. Evaluator bütçeyi takip ediyor, en iyiyi güncelliyor
  3. RandomSearch çalışıp anlamlı bir sonuç üretiyor
  4. ClassicGA RandomSearch'ten daha iyi sonuç buluyor
  5. Yakınsama eğrisi monoton (en iyi fitness sadece artar)
  6. Aynı seed → aynı sonuç (tekrarlanabilirlik)

Düşük bütçe ile (2000 değerlendirme) hızlı test ediyoruz.
Gerçek deneylerde 10.000 kullanılacak.
"""

import warnings
import time
import numpy as np

warnings.filterwarnings("ignore")

from core import (
    EffectChain,
    MultiResolutionSTFTLoss,
    IDMTLoader,
    get_preset_vector,
)
from algorithms import (
    Evaluator,
    RandomSearch,
    ClassicGA,
)


def banner(title: str) -> None:
    print()
    print("=" * 64)
    print("  " + title)
    print("=" * 64)


def check(cond: bool, label: str) -> None:
    mark = "OK " if cond else "FAIL"
    print(f"  [{mark}] {label}")
    if not cond:
        raise AssertionError(f"Test başarısız: {label}")


# ────────────────────────────────────────────────────────────────────
# Test setup — gerçek bir hedef ses üret
# ────────────────────────────────────────────────────────────────────

banner("Adım 2 — Test kurulumu")
SR     = 22050
BUDGET = 2000      # Adım 2 testi için düşük, gerçek deneyde 10000
SEED   = 42

loader = IDMTLoader(dataset_root=None, sample_rate=SR, verbose=True)
chain  = EffectChain(sample_rate=SR)
loss   = MultiResolutionSTFTLoss()

# Sabit kuru sinyal
dry = loader.get_fixed_dry(duration=0.7, seed=SEED)
print(f"  Kuru sinyal: {len(dry)} örnek (~{len(dry)/SR:.2f}s)")

# Ground truth preset — 'crunch'
x_star = get_preset_vector("crunch", seed=SEED)
target = chain.process(dry, x_star)
loss.cache_target(target)  # target STFT önbelleğe alındı
print(f"  Hedef preset: crunch, target len = {len(target)}")


def fitness_fn(x: np.ndarray) -> float:
    """Bir parametre vektörünü çevirir, hedeften uzaklığı negatife alır."""
    wet = chain.process(dry, x)
    return -loss.compute_fast(wet)   # önbellekli: ~7 ms/eval tasarruf


# Sanity: ground truth'un fitness'ı 0'a çok yakın olmalı
gt_fitness = fitness_fn(x_star)
print(f"  Ground truth fitness  = {gt_fitness:+.6f}  (mükemmel = 0)")
check(gt_fitness > -1e-3, "Ground truth fitness yaklaşık 0")


# ────────────────────────────────────────────────────────────────────
# 1. RandomSearch testi
# ────────────────────────────────────────────────────────────────────

banner("1. RandomSearch")
algo_rs    = RandomSearch()
eval_rs    = Evaluator(fitness_fn, budget=BUDGET, record_every=100)
rng_rs     = np.random.default_rng(SEED)

t0 = time.time()
algo_rs.safe_run(eval_rs, rng_rs)
elapsed_rs = time.time() - t0

result_rs = algo_rs.build_result(eval_rs, seed=SEED)
print(f"  {result_rs.summary()}")
print(f"  Süre: {elapsed_rs:.1f}s, kayıt noktası sayısı: {len(result_rs.history_evals)}")
check(result_rs.n_evals == BUDGET, "Tam BUDGET kadar değerlendirme")
check(result_rs.best_x is not None,   "Bir best_x bulundu")
check(result_rs.best_fitness > -np.inf, "Fitness güncellendi")


# ────────────────────────────────────────────────────────────────────
# 2. ClassicGA testi
# ────────────────────────────────────────────────────────────────────

banner("2. ClassicGA")
algo_ga = ClassicGA(
    pop_size=50,
    tournament_size=3,
    crossover_rate=0.9,
    crossover_alpha=0.5,
    mutation_sigma=0.1,
    n_elites=2,
)
eval_ga = Evaluator(fitness_fn, budget=BUDGET, record_every=100)
rng_ga  = np.random.default_rng(SEED)

t0 = time.time()
algo_ga.safe_run(eval_ga, rng_ga)
elapsed_ga = time.time() - t0

result_ga = algo_ga.build_result(eval_ga, seed=SEED)
print(f"  {result_ga.summary()}")
print(f"  Süre: {elapsed_ga:.1f}s, kayıt noktası: {len(result_ga.history_evals)}")
check(result_ga.n_evals <= BUDGET, "Bütçe sınırına uyuldu")
check(result_ga.n_evals >= 0.9 * BUDGET, "Bütçenin en az %90'ı kullanıldı")


# ────────────────────────────────────────────────────────────────────
# 3. Karşılaştırma — GA, RandomSearch'ten iyi mi?
# ────────────────────────────────────────────────────────────────────

banner("3. GA vs RandomSearch")
print(f"  RandomSearch best_loss = {result_rs.best_loss:.4f}")
print(f"  ClassicGA    best_loss = {result_ga.best_loss:.4f}")
print(f"  Fark               = {result_rs.best_loss - result_ga.best_loss:+.4f}")
check(result_ga.best_loss < result_rs.best_loss,
      "GA, RandomSearch'ten daha düşük loss buldu")


# ────────────────────────────────────────────────────────────────────
# 4. Yakınsama eğrileri monoton mu?
# ────────────────────────────────────────────────────────────────────

banner("4. Yakınsama eğrisi monotonluğu")
for name, res in [("RandomSearch", result_rs), ("ClassicGA", result_ga)]:
    h = res.history_best
    is_monotone = all(h[i+1] >= h[i] - 1e-9 for i in range(len(h)-1))
    print(f"  {name:14s}: ilk={h[0]:.4f}, son={h[-1]:.4f}, "
          f"{'monoton' if is_monotone else 'BOZUK'}")
    check(is_monotone, f"{name} yakınsama eğrisi monoton")


# ────────────────────────────────────────────────────────────────────
# 5. Tekrarlanabilirlik
# ────────────────────────────────────────────────────────────────────

banner("5. Tekrarlanabilirlik")
algo_rep = ClassicGA(pop_size=30)
eval_rep = Evaluator(fitness_fn, budget=300, record_every=50)
algo_rep.safe_run(eval_rep, np.random.default_rng(77))
res_a = algo_rep.build_result(eval_rep, seed=77)

algo_rep2 = ClassicGA(pop_size=30)
eval_rep2 = Evaluator(fitness_fn, budget=300, record_every=50)
algo_rep2.safe_run(eval_rep2, np.random.default_rng(77))
res_b = algo_rep2.build_result(eval_rep2, seed=77)

print(f"  Koşum A best_loss = {res_a.best_loss:.6f}")
print(f"  Koşum B best_loss = {res_b.best_loss:.6f}")
check(np.isclose(res_a.best_loss, res_b.best_loss, atol=1e-9),
      "Aynı seed → aynı sonuç")
check(np.allclose(res_a.best_x, res_b.best_x, atol=1e-9),
      "Aynı seed → aynı best_x")


# ────────────────────────────────────────────────────────────────────
# 6. Yakınsama eğrisi şekli — birkaç kayıt noktası göster
# ────────────────────────────────────────────────────────────────────

banner("6. Yakınsama eğrisi (özet)")
print(f"  {'evals':>8s}  {'Random loss':>14s}  {'GA loss':>14s}")
common = sorted(set(result_rs.history_evals) & set(result_ga.history_evals))
sample_points = common[:: max(1, len(common) // 8)]
for ev in sample_points:
    i_rs = result_rs.history_evals.index(ev)
    i_ga = result_ga.history_evals.index(ev)
    loss_rs = -result_rs.history_best[i_rs]
    loss_ga = -result_ga.history_best[i_ga]
    print(f"  {ev:>8d}  {loss_rs:>14.4f}  {loss_ga:>14.4f}")


# ────────────────────────────────────────────────────────────────────
banner("Adım 2 BAŞARILI")
print("  Sonraki adım: MAP-Elites + CMA-ES (Adım 3)")
print("=" * 64)