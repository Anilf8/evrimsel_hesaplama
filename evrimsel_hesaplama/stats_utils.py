"""
statistics.py
=============

İstatistik testleri — algoritma karşılaştırması için.

Mimari görseldeki testler:
  - Wilcoxon işaretli sıra testi  : İki algoritmanın medyanı aynı mı?
  - Vargha-Delaney A (VD-A)       : Etki büyüklüğü (effect size)

Neden bu testler?
  - Wilcoxon: parametrik olmayan → normallik varsayımı yok.
    EC sonuçları genellikle çarpık dağılır, t-testi uygunsuz.
  - VD-A: "A algoritması B'den ne kadar daha iyi?"
    0.5 = fark yok, >0.5 = A daha iyi, <0.5 = B daha iyi.
    p-değerinden daha yorumlanabilir.

Referans:
  Wilcoxon (1945), Vargha & Delaney (2000),
  Arcuri & Briand (2011) "A Practical Guide for Using Statistical Tests
  to Assess Randomized Algorithms in Software Engineering"
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ────────────────────────────────────────────────────────────────────
# Wilcoxon işaretli sıra testi
# ────────────────────────────────────────────────────────────────────

def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "two-sided",
    alpha:       float = 0.05,
) -> Dict[str, float | bool | str]:
    """
    İki algoritmanın metrik dağılımlarını Wilcoxon testi ile karşılaştır.

    a, b: Karşılaştırılacak metrik dizileri (örn. final_loss, n_seeds uzunlukta)
    alternative: "two-sided" | "less" | "greater"
      "less"    → H1: a medyanı < b medyanı  (a daha iyi)
      "greater" → H1: a medyanı > b medyanı  (b daha iyi)

    Döndürür:
      statistic : W istatistiği
      p_value   : ham p-değeri
      significant: p < alpha mı?
      direction : "a_better" | "b_better" | "no_difference"

    NOT: n < 5 için test güvenilmez, uyarı verilir.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # NaN'ları çıkar (eşli olarak)
    mask  = ~(np.isnan(a) | np.isnan(b))
    a, b  = a[mask], b[mask]
    n     = len(a)

    result = {
        "n":           n,
        "statistic":   np.nan,
        "p_value":     np.nan,
        "significant": False,
        "direction":   "insufficient_data",
        "alpha":       alpha,
    }

    if n < 5:
        result["direction"] = "insufficient_data (n<5)"
        return result

    # Farklar sıfırsa (tüm değerler eşit) test yapılamaz
    diffs = a - b
    if np.all(diffs == 0):
        result["direction"] = "identical"
        result["p_value"]   = 1.0
        return result

    try:
        stat, p = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    except ValueError as e:
        result["direction"] = f"error: {e}"
        return result

    significant = bool(p < alpha)
    # Yön: medyan karşılaştırması
    if not significant:
        direction = "no_difference"
    elif np.median(a) < np.median(b):
        direction = "a_better"   # loss için küçük = iyi
    else:
        direction = "b_better"

    result.update({
        "statistic":   float(stat),
        "p_value":     float(p),
        "significant": significant,
        "direction":   direction,
    })
    return result


# ────────────────────────────────────────────────────────────────────
# Vargha-Delaney A (etki büyüklüğü)
# ────────────────────────────────────────────────────────────────────

def vargha_delaney_a(
    a: np.ndarray,
    b: np.ndarray,
) -> Dict[str, float | str]:
    """
    Vargha-Delaney A etki büyüklüğü.

    A(a,b) = P(a < b) + 0.5 · P(a == b)

    Yorumlama (loss metriği, küçük = iyi):
      A = 1.0 → a her zaman b'den daha iyi (küçük)
      A = 0.5 → fark yok
      A = 0.0 → b her zaman a'dan daha iyi

    Büyüklük sınıflandırması (Vargha & Delaney 2000):
      |A - 0.5| < 0.06  → "negligible"  (ihmal edilebilir)
      |A - 0.5| < 0.14  → "small"       (küçük)
      |A - 0.5| < 0.21  → "medium"      (orta)
      else              → "large"        (büyük)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n_a, n_b = len(a), len(b)

    if n_a == 0 or n_b == 0:
        return {"A": np.nan, "magnitude": "insufficient_data", "n_a": n_a, "n_b": n_b}

    # Tüm a[i] vs b[j] çiftleri — vektörize
    # U istatistiğinden hesapla: daha hızlı
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="less")
    A = float(u_stat) / float(n_a * n_b)

    # Büyüklük sınıflandırması
    diff = abs(A - 0.5)
    if diff < 0.06:
        magnitude = "negligible"
    elif diff < 0.14:
        magnitude = "small"
    elif diff < 0.21:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "A":         A,
        "magnitude": magnitude,
        "n_a":       n_a,
        "n_b":       n_b,
    }


# ────────────────────────────────────────────────────────────────────
# Çoklu karşılaştırma: tüm algoritma çiftleri
# ────────────────────────────────────────────────────────────────────

def pairwise_comparison(
    metrics_by_algo: Dict[str, np.ndarray],
    metric_name:     str = "final_loss",
    alpha:           float = 0.05,
) -> List[Dict]:
    """
    Tüm algoritma çiftleri için Wilcoxon + VD-A hesapla.

    metrics_by_algo: {algo_name: metrik_dizisi}
    metric_name: sadece raporlama için etiket

    Döndürür: her çift için sonuç listesi (sıralı, küçükten büyüğe medyan)
    """
    algo_names = list(metrics_by_algo.keys())
    rows = []

    for a_name, b_name in combinations(algo_names, 2):
        a_vals = metrics_by_algo[a_name]
        b_vals = metrics_by_algo[b_name]

        w   = wilcoxon_test(a_vals, b_vals)
        vda = vargha_delaney_a(a_vals, b_vals)

        rows.append({
            "metric":    metric_name,
            "algo_a":    a_name,
            "algo_b":    b_name,
            "median_a":  float(np.nanmedian(a_vals)),
            "median_b":  float(np.nanmedian(b_vals)),
            "wilcoxon_p":  w["p_value"],
            "significant": w["significant"],
            "direction":   w["direction"],
            "VD_A":        vda["A"],
            "magnitude":   vda["magnitude"],
        })

    return rows


# ────────────────────────────────────────────────────────────────────
# Özet tablo yazdırma
# ────────────────────────────────────────────────────────────────────

def print_comparison_table(rows: List[Dict]) -> None:
    """
    pairwise_comparison() çıktısını okunabilir tablo olarak yazdır.

    Örnek çıktı:
      ClassicGA  vs  CMAES      | p=0.043 * | A=0.72 large  | CMAES daha iyi
    """
    print(f"\n{'─'*72}")
    print(f"  {'Çift':30s} | {'p':>8s} | {'VD-A':>6s} {'Büyüklük':10s} | Sonuç")
    print(f"{'─'*72}")

    for r in rows:
        sig_mark = "*" if r["significant"] else " "
        p_str    = f"{r['wilcoxon_p']:.4f}{sig_mark}" if not np.isnan(r["wilcoxon_p"]) else "  NaN "
        A_str    = f"{r['VD_A']:.3f}" if not np.isnan(r.get("VD_A", np.nan)) else "  NaN"
        pair     = f"{r['algo_a']:12s} vs {r['algo_b']:12s}"

        # Yön açıklaması
        d = r["direction"]
        if d == "a_better":
            verdict = f"{r['algo_a']} daha iyi"
        elif d == "b_better":
            verdict = f"{r['algo_b']} daha iyi"
        else:
            verdict = d

        print(f"  {pair:30s} | p={p_str:7s} | A={A_str} {r.get('magnitude','?'):10s} | {verdict}")

    print(f"{'─'*72}")
    print("  * p < alpha anlamlı fark")


def print_summary_table(
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "final_loss",
) -> None:
    """
    Her algoritmanın özet istatistiklerini tablo olarak yazdır.

    summaries: {algo_name: summarize(metrics) çıktısı}
    """
    print(f"\n  Metrik: {metric}")
    print(f"  {'Algoritma':15s} | {'Median':>8s} | {'Mean':>8s} | {'Std':>8s} | n")
    print(f"  {'─'*55}")
    for algo, s in summaries.items():
        m = s.get(metric, {})
        print(
            f"  {algo:15s} | "
            f"{m.get('median', np.nan):8.4f} | "
            f"{m.get('mean',   np.nan):8.4f} | "
            f"{m.get('std',    np.nan):8.4f} | "
            f"{m.get('n', 0)}"
        )