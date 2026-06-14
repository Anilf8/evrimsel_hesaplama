"""
metrics.py
==========

Deney metrikleri — algoritma karşılaştırması için sayısal ölçütler.

Mimari görseldeki metrikler:
  - QD-Score   : MAP-Elites arşiv kalitesi × çeşitlilik
  - Coverage   : Arşivde dolu hücre oranı
  - Parametre hatası : En iyi çözümün ground truth'a uzaklığı

Ek metrikler (yakınsama analizi için):
  - AUC (Alan altı eğri)  : Yakınsama hızı ölçütü
  - Eval-to-threshold     : Belirli bir loss eşiğine kaç eval'de ulaşıldı
  - Final best loss       : Bütçe bitimindeki en iyi değer

Tüm fonksiyonlar List[RunResult] alır ve dict döndürür.
Bu yapı istatistik modülüyle doğrudan çalışır.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from algorithms.base import RunResult
from core import get_preset_vector, N_PARAMS


# ────────────────────────────────────────────────────────────────────
# Temel metrikler
# ────────────────────────────────────────────────────────────────────

def final_losses(results: List[RunResult]) -> np.ndarray:
    """
    Her koşumun final (bütçe sonu) best_loss değerlerini döndür.

    Daha düşük = daha iyi. İstatistik testleri bu dizi üzerinde çalışır.
    """
    return np.array([r.best_loss for r in results], dtype=np.float64)


def auc_loss(results: List[RunResult], normalize: bool = True) -> np.ndarray:
    """
    Yakınsama eğrisinin altındaki alan (AUC) — her koşum için.

    Düşük AUC = hızlı yakınsama (erken iyi sonuç bulundu).

    history_best değerleri fitness (negatif loss) cinsinden gelir.
    Burada loss = -fitness'a çevirip trapez integrali alınır.

    normalize=True → eval sayısına böl, ölçek bağımsız karşılaştırma.
    """
    aucs = []
    for r in results:
        if len(r.history_evals) < 2:
            aucs.append(np.nan)
            continue
        evals  = np.array(r.history_evals, dtype=np.float64)
        losses = -np.array(r.history_best, dtype=np.float64)  # fitness → loss

        # NumPy 2.0+ uyumluluğu
        try:
            area = np.trapezoid(losses, evals)
        except AttributeError:
            area = np.trapz(losses, evals)

        if normalize:
            area /= (evals[-1] - evals[0]) if evals[-1] > evals[0] else 1.0
        aucs.append(area)
    return np.array(aucs, dtype=np.float64)


def evals_to_threshold(
    results:   List[RunResult],
    threshold: float,
) -> np.ndarray:
    """
    Her koşumda loss < threshold eşiğine kaç eval'de ulaşıldı?

    Eşiğe hiç ulaşılamazsa np.nan döner.
    Düşük değer = hızlı yakınsama.

    threshold: loss cinsinden (örn. 0.20 → %20 loss altı)
    """
    counts = []
    for r in results:
        found = np.nan
        for ev, fit in zip(r.history_evals, r.history_best):
            loss = -fit   # fitness → loss
            if loss <= threshold:
                found = float(ev)
                break
        counts.append(found)
    return np.array(counts, dtype=np.float64)


def parameter_error(
    results:     List[RunResult],
    preset_name: str,
    seed:        int = 0,
) -> np.ndarray:
    """
    Her koşumun bulduğu best_x ile ground truth x* arasındaki L2 mesafesi.

    Normalize edilmiş [0,1]^16 uzayında hesaplanır.
    Düşük = ground truth'a yakın çözüm bulundu.

    NOT: Bu metrik yalnızca gerçek bir ground truth olduğunda anlamlıdır.
    Demo modunda x_star rastgele üretildiğinden referans olarak kullanılabilir
    ama yorumlanırken dikkatli olunmalı.
    """
    x_star = get_preset_vector(preset_name, seed=seed).astype(np.float64)
    errors = []
    for r in results:
        if r.best_x is None or len(r.best_x) != N_PARAMS:
            errors.append(np.nan)
            continue
        err = np.linalg.norm(r.best_x.astype(np.float64) - x_star)
        errors.append(err)
    return np.array(errors, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────
# MAP-Elites özel metrikleri
# ────────────────────────────────────────────────────────────────────

def qd_scores(results: List[RunResult]) -> np.ndarray:
    """
    MAP-Elites QD-Score — her koşum için arşivdeki fitness toplamı.

    Yüksek QD-Score = hem kaliteli hem çeşitli çözümler bulundu.

    NOT: fitness = -loss olduğundan QD-Score genellikle negatiftir.
    Karşılaştırmada "daha az negatif = daha iyi" yorumu yapılır.
    """
    scores = []
    for r in results:
        me = r.extra.get("map_elites", {})
        scores.append(float(me.get("qd_score", np.nan)))
    return np.array(scores, dtype=np.float64)


def coverages(results: List[RunResult]) -> np.ndarray:
    """
    MAP-Elites coverage — dolu hücre oranı [0, 1].

    Yüksek coverage = davranış uzayının daha büyük bölümü keşfedildi.
    """
    covs = []
    for r in results:
        me = r.extra.get("map_elites", {})
        covs.append(float(me.get("coverage", np.nan)))
    return np.array(covs, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────
# Toplu metrik hesaplama
# ────────────────────────────────────────────────────────────────────

def compute_all(
    results:     List[RunResult],
    preset_name: str,
    threshold:   float = 0.20,
) -> Dict[str, np.ndarray]:
    """
    Tüm metrikleri tek seferde hesapla.

    Döndürür: metrik_adı → değerler dizisi (n_seeds uzunlukta)

    Kullanım:
        metrics = compute_all(results["ClassicGA"], "crunch")
        print(metrics["final_loss"].mean())
    """
    m: Dict[str, np.ndarray] = {
        "final_loss":        final_losses(results),
        "auc_loss":          auc_loss(results),
        "evals_to_threshold": evals_to_threshold(results, threshold),
        "param_error":       parameter_error(results, preset_name),
    }

    # MAP-Elites özel metrikleri — diğer algoritmalar için NaN olur
    m["qd_score"] = qd_scores(results)
    m["coverage"] = coverages(results)

    return m


def summarize(metrics: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Metrik dizilerini özet istatistiklere çevir (mean, std, median).

    Kullanım:
        summary = summarize(compute_all(results, "crunch"))
        # summary["final_loss"] → {"mean": 0.18, "std": 0.02, "median": 0.17}
    """
    out = {}
    for name, arr in metrics.items():
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            out[name] = {"mean": np.nan, "std": np.nan, "median": np.nan, "n": 0}
        else:
            out[name] = {
                "mean":   float(np.mean(valid)),
                "std":    float(np.std(valid, ddof=1) if len(valid) > 1 else 0.0),
                "median": float(np.median(valid)),
                "n":      int(len(valid)),
            }
    return out