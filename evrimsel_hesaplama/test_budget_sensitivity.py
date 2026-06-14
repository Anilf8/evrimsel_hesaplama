"""
test_budget_sensitivity.py
==========================

Adım 6 sonuçlarındaki anomaliyi araştır:
"Self-Adaptive GA neden ClassicGA'dan kötü?"

Hipotez: Düşük bütçede SA yenik, yüksek bütçede yakalar/geçer.

Test:
  3 farklı bütçe (1500, 3000, 6000) × 3 seed × 2 algoritma
  = 18 koşum, ~6-8 dakika

Çıktı: Bütçe-performans tablosu — adaptasyon hipotezini doğrular.
"""

import warnings
import time
import numpy as np

warnings.filterwarnings("ignore")

from core import (
    EffectChain, MultiResolutionSTFTLoss, IDMTLoader, get_preset_vector,
)
from algorithms import Evaluator, ClassicGA, SelfAdaptiveGA


SR = 22050
SEED_BASE = 100
BUDGETS = [1500, 3000, 6000]
N_SEEDS = 3


# Kurulum
loader = IDMTLoader(None, SR, verbose=False)
chain  = EffectChain(SR)
loss   = MultiResolutionSTFTLoss()
dry    = loader.get_fixed_dry(0.7, seed=0)
x_star = get_preset_vector("crunch", seed=0)
target = chain.process(dry, x_star)

def fitness_fn(x):
    wet = chain.process(dry, x)
    return -loss.compute_fast(wet)


print()
print("=" * 70)
print(" Bütçe duyarlılık testi — ClassicGA vs SelfAdaptiveGA")
print("=" * 70)
print()

results = {}  # results[budget] = {"ClassicGA": [...], "SelfAdaptiveGA": [...]}

t_start = time.time()

for budget in BUDGETS:
    results[budget] = {"ClassicGA": [], "SelfAdaptiveGA": []}
    print(f"--- Bütçe: {budget} ---")

    for seed in range(N_SEEDS):
        # ClassicGA
        loss.cache_target(target)
        a = ClassicGA(pop_size=50, mutation_sigma=0.1, n_elites=2)
        e = Evaluator(fitness_fn, budget=budget, record_every=200)
        a.safe_run(e, np.random.default_rng(SEED_BASE + seed))
        r = a.build_result(e, seed=SEED_BASE + seed)
        results[budget]["ClassicGA"].append(r.best_loss)

        # SelfAdaptiveGA
        loss.cache_target(target)
        a = SelfAdaptiveGA(pop_size=50, sigma_init=0.1, n_elites=2)
        e = Evaluator(fitness_fn, budget=budget, record_every=200)
        a.safe_run(e, np.random.default_rng(SEED_BASE + seed))
        r = a.build_result(e, seed=SEED_BASE + seed)
        results[budget]["SelfAdaptiveGA"].append(r.best_loss)

        c_loss = results[budget]["ClassicGA"][-1]
        s_loss = results[budget]["SelfAdaptiveGA"][-1]
        print(f"  seed={seed}: Classic={c_loss:.4f}  SelfAdaptive={s_loss:.4f}  "
              f"{'classic_wins' if c_loss < s_loss else 'sa_wins'}")

# Özet tablo
print()
print("=" * 70)
print(" Bütçe-performans tablosu (median loss)")
print("=" * 70)
print(f"  {'Bütçe':>6s} | {'ClassicGA':>10s} | {'SelfAdaptive':>12s} | {'Fark':>10s} | Kazanan")
print("  " + "-" * 65)

for budget in BUDGETS:
    c_med = np.median(results[budget]["ClassicGA"])
    s_med = np.median(results[budget]["SelfAdaptiveGA"])
    diff = (s_med - c_med) / c_med * 100  # positive = SA daha kötü
    winner = "ClassicGA" if c_med < s_med else "SelfAdaptive"
    print(f"  {budget:>6d} | {c_med:>10.4f} | {s_med:>12.4f} | "
          f"{diff:>+9.1f}% | {winner}")

print()
print(f"Toplam süre: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} dk)")

# Hipotezi değerlendir
print()
print("=" * 70)
print(" Hipotez değerlendirmesi:")
print("=" * 70)

c_low = np.median(results[BUDGETS[0]]["ClassicGA"])
s_low = np.median(results[BUDGETS[0]]["SelfAdaptiveGA"])
c_high = np.median(results[BUDGETS[-1]]["ClassicGA"])
s_high = np.median(results[BUDGETS[-1]]["SelfAdaptiveGA"])

gap_low = s_low - c_low
gap_high = s_high - c_high

print(f"  Düşük bütçe ({BUDGETS[0]:>5d}): SA - Classic = {gap_low:+.4f}")
print(f"  Yüksek bütçe ({BUDGETS[-1]:>5d}): SA - Classic = {gap_high:+.4f}")

if gap_high < gap_low:
    print()
    print("  HİPOTEZ DOĞRU: Bütçe arttıkça SA aradaki farkı kapatıyor.")
    if gap_high < 0:
        print("  Hatta yüksek bütçede SA, Classic'i geçiyor!")
elif gap_high > gap_low:
    print()
    print("  HİPOTEZ ŞÜPHELİ: Yüksek bütçede de SA hala arkada.")
    print("  σ adaptasyon parametrelerini ayarlamak gerekebilir.")
else:
    print()
    print("  Sonuçlar belirsiz — daha fazla seed gerekebilir.")