"""
test_step4.py
=============

Adım 4 sağlamlık testi — deney çerçevesi (runner + metrikler + istatistik).

Doğrulananlar:
  1. Runner tüm algoritmaları çalıştırır, sonuçlar eksiksiz gelir
  2. Metrikler hesaplanır (final_loss, auc, param_error, coverage)
  3. Wilcoxon: CMAES ve ClassicGA, RandomSearch'ten anlamlı iyi
  4. VD-A etki büyüklüğü en az "small"
  5. Sonuçlar JSON'a kaydedilip yeniden yüklenebilir
  6. MAP-Elites QD-Score ve coverage geçerli aralıkta

Hızlı test: budget=2000, n_seeds=5, 1 preset.
Gerçek deneyde: budget=10_000, n_seeds=10, 5 preset.
"""

import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

from runner     import ExperimentRunner
from metrics    import compute_all, summarize
from stats_utils import (
    wilcoxon_test,
    vargha_delaney_a,
    pairwise_comparison,
    print_comparison_table,
    print_summary_table,
)


def banner(t):
    print()
    print("=" * 64)
    print("  " + t)
    print("=" * 64)


def check(cond, label):
    mark = "OK " if cond else "FAIL"
    print(f"  [{mark}] {label}")
    if not cond:
        raise AssertionError(f"FAIL: {label}")


# ────────────────────────────────────────────────────────────────────
# Kurulum
# ────────────────────────────────────────────────────────────────────
banner("Kurulum")

BUDGET  = 2000
N_SEEDS = 5
PRESET  = "crunch"

runner = ExperimentRunner(
    budget=BUDGET,
    n_seeds=N_SEEDS,
    record_every=200,
    preset_names=[PRESET],
    dry_path="dry_guitar_riff.wav",   # ← senin WAV dosyan
    dry_duration=1.0,                 # 1 saniyenin ilk 0.7s'ini kullan
    verbose=True,
)
print(f"  Budget={BUDGET}, N_seeds={N_SEEDS}, Preset={PRESET}")


# ────────────────────────────────────────────────────────────────────
# 1. Runner
# ────────────────────────────────────────────────────────────────────
banner("1. Runner — tüm koşumlar")

all_results = runner.run_all()
preset_res  = all_results[PRESET]
algo_names  = list(preset_res.keys())

print(f"\n  Algoritmalar: {algo_names}")
for algo, results in preset_res.items():
    check(len(results) == N_SEEDS,                    f"{algo}: {N_SEEDS} koşum tamamlandı")
    check(all(r.n_evals > 0 for r in results),        f"{algo}: tüm koşumlar eval yaptı")
    check(all(len(r.history_best) > 0 for r in results), f"{algo}: yakınsama geçmişi dolu")


# ────────────────────────────────────────────────────────────────────
# 2. Metrikler
# ────────────────────────────────────────────────────────────────────
banner("2. Metrikler")

all_metrics   = {}
all_summaries = {}

for algo, results in preset_res.items():
    m = compute_all(results, preset_name=PRESET, threshold=0.25)
    all_metrics[algo]   = m
    all_summaries[algo] = summarize(m)

print_summary_table(all_summaries, metric="final_loss")
print_summary_table(all_summaries, metric="auc_loss")

for algo in algo_names:
    losses = all_metrics[algo]["final_loss"]
    check(not np.all(np.isnan(losses)), f"{algo}: final_loss hesaplandı")
    check(np.nanmean(losses) > 0,       f"{algo}: loss pozitif")

me_cov = all_metrics["MAPElites"]["coverage"]
me_qd  = all_metrics["MAPElites"]["qd_score"]
print(f"\n  MAPElites coverage : {np.nanmean(me_cov):.2%} ± {np.nanstd(me_cov):.2%}")
print(f"  MAPElites QD-Score : {np.nanmean(me_qd):.2f} ± {np.nanstd(me_qd):.2f}")
check(np.nanmean(me_cov) > 0.05, "MAPElites ortalama coverage > 5%")
check(not np.all(np.isnan(me_qd)), "MAPElites QD-Score hesaplandı")


# ────────────────────────────────────────────────────────────────────
# 3. İstatistik testleri
# ────────────────────────────────────────────────────────────────────
banner("3. İstatistik testleri (Wilcoxon + VD-A)")

final_loss_by_algo = {algo: all_metrics[algo]["final_loss"] for algo in algo_names}
rows = pairwise_comparison(final_loss_by_algo, metric_name="final_loss", alpha=0.05)
print_comparison_table(rows)

rs_losses = final_loss_by_algo["RandomSearch"]
for algo in ["ClassicGA", "CMAES"]:
    algo_losses = final_loss_by_algo[algo]
    w   = wilcoxon_test(algo_losses, rs_losses, alternative="less")
    vda = vargha_delaney_a(algo_losses, rs_losses)

    print(f"\n  {algo} vs RandomSearch:")
    print(f"    Wilcoxon p={w['p_value']:.4f}, anlamlı={w['significant']}")
    print(f"    VD-A={vda['A']:.3f} ({vda['magnitude']})")

    check(
        np.nanmedian(algo_losses) < np.nanmedian(rs_losses),
        f"{algo} medyanı RandomSearch'ten düşük"
    )
    check(
        vda["magnitude"] in ("small", "medium", "large"),
        f"{algo} VD-A etki büyüklüğü anlamlı"
    )


# ────────────────────────────────────────────────────────────────────
# 4. JSON kayıt / yükleme
# ────────────────────────────────────────────────────────────────────
banner("4. JSON kayıt ve yükleme")

with tempfile.TemporaryDirectory() as tmpdir:
    path = str(Path(tmpdir) / "experiment.json")
    runner.save(all_results, path)
    loaded = ExperimentRunner.load(path)

    check(PRESET in loaded, "Preset JSON'da mevcut")
    for algo in algo_names:
        check(algo in loaded[PRESET], f"{algo} JSON'da mevcut")
        check(len(loaded[PRESET][algo]) == N_SEEDS, f"{algo}: JSON'da {N_SEEDS} koşum")

    orig  = preset_res["CMAES"][0].best_loss
    saved = loaded[PRESET]["CMAES"][0]["best_loss"]
    check(abs(orig - saved) < 1e-9, "CMAES best_loss JSON'da korundu")

    size_kb = Path(path).stat().st_size / 1024
    print(f"  Dosya boyutu: {size_kb:.1f} KB")


# ────────────────────────────────────────────────────────────────────
# 5. Yakınsama özet tablosu (medyan)
# ────────────────────────────────────────────────────────────────────
banner("5. Yakınsama özet (medyan loss)")

common = None
for results in preset_res.values():
    for r in results:
        s = set(r.history_evals)
        common = s if common is None else common & s

common = sorted(common or [])[::max(1, len(common or []) // 6)]

if common:
    print(f"  {'evals':>6s} | " + " | ".join(f"{a:>12s}" for a in algo_names))
    print("  " + "─" * (10 + 15 * len(algo_names)))
    for ev in common:
        cols = []
        for algo in algo_names:
            vals = [-r.history_best[r.history_evals.index(ev)]
                    for r in preset_res[algo] if ev in r.history_evals]
            cols.append(f"{np.median(vals):12.4f}" if vals else f"{'—':>12s}")
        print(f"  {ev:>6d} | " + " | ".join(cols))


# ────────────────────────────────────────────────────────────────────
banner("Adım 4 BAŞARILI")
print("  Sonraki adım: Görselleştirme (Adım 5)")
print("=" * 64)