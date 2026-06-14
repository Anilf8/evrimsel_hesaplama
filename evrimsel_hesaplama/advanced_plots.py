"""
advanced_plots.py
=================

Makaleyi sağlamlaştıran 3 ileri grafik. Literatür temelli:

  G1 — QD-Score Evrimi          (Shier 2025, Fig.3 standardı)
  G3 — Algoritma × Preset Matrisi (5 preseti tek figürde özet)
  G4 — Critical Difference Diyagramı (Demšar 2006, EC altın standardı)

Kullanım:
    python advanced_plots.py                       # results/experiment.json okur
    python advanced_plots.py --json path/to.json   # özel yol
    python advanced_plots.py --output figs/         # çıktı klasörü

Her grafik mevcut grafiklerle aynı stilde (serif font, makale kalitesi).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────
# Ortak ayarlar
# ────────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "RandomSearch":   "#999999",
    "ClassicGA":      "#4c72b0",
    "SelfAdaptiveGA": "#dd8452",
    "MAPElites":      "#55a868",
    "CMAES":          "#c44e52",
    "CMA-ME":         "#8172b3",
}

ALGO_MARKERS = {
    "RandomSearch":   "o",
    "ClassicGA":      "s",
    "SelfAdaptiveGA": "^",
    "MAPElites":      "D",
    "CMAES":          "v",
    "CMA-ME":         "P",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════
# G1 — QD-Score Evrimi
# ════════════════════════════════════════════════════════════════════

def plot_qd_score_evolution(data, preset="crunch", save_path="qd_evolution.png"):
    """
    QD-tabanlı algoritmaların (MAPElites, CMA-ME) QD-Score'unun
    değerlendirme sayısına göre evrimi.

    NOT: Bu grafik için runner'ın her kayıt noktasında QD-Score'u
    kaydetmesi gerekir. Mevcut sistemde sadece FINAL QD-Score var.
    Bu yüzden iki seçenek sunuyoruz:
      A) Eğer history_qd varsa: tam evrim eğrisi
      B) Yoksa: final QD-Score bar grafiği (fallback)
    """
    preset_data = data.get(preset, {})
    qd_algos = ["MAPElites", "CMA-ME"]
    available = [a for a in qd_algos if a in preset_data and preset_data[a]]

    if not available:
        print(f"[G1] {preset} için QD algoritması bulunamadı, atlanıyor")
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # history_qd var mı kontrol et
    sample_run = preset_data[available[0]][0]
    has_qd_history = "history_qd" in sample_run.get("extra", {}).get("map_elites", {})

    if has_qd_history:
        # ── Tam evrim eğrisi ──
        for algo in available:
            runs = preset_data[algo]
            evals = runs[0]["history_evals"]
            qd_curves = []
            for run in runs:
                qd_hist = run["extra"]["map_elites"]["history_qd"]
                qd_curves.append(qd_hist)
            qd_arr = np.array(qd_curves)
            mean_qd = np.mean(qd_arr, axis=0)
            std_qd  = np.std(qd_arr, axis=0)

            ax.plot(evals, mean_qd, marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "black"), linewidth=2,
                    markersize=6, label=algo, markevery=max(1, len(evals)//8))
            ax.fill_between(evals, mean_qd - std_qd, mean_qd + std_qd,
                            color=ALGO_COLORS.get(algo, "black"), alpha=0.15)

        ax.set_xlabel("Fitness Değerlendirme Sayısı")
        ax.set_ylabel("QD-Score (Yüksek = Daha İyi)")
        ax.set_title(f"QD-Score Evrimi ({preset.capitalize()} Preset)",
                     fontweight="bold")
    else:
        # ── Fallback: final QD-Score bar grafiği ──
        print(f"[G1] history_qd bulunamadı — final QD-Score bar grafiği üretiliyor")
        print(f"     (Tam evrim için runner'a QD-Score kaydı eklenebilir)")

        algos_with_qd = []
        qd_means = []
        qd_stds  = []
        for algo in available:
            runs = preset_data[algo]
            qds = [r["extra"]["map_elites"]["qd_score"] for r in runs]
            algos_with_qd.append(algo)
            qd_means.append(np.mean(qds))
            qd_stds.append(np.std(qds))

        colors = [ALGO_COLORS.get(a, "gray") for a in algos_with_qd]
        bars = ax.bar(algos_with_qd, qd_means, yerr=qd_stds, capsize=6,
                      color=colors, alpha=0.85, edgecolor="black", linewidth=1)

        for bar, val in zip(bars, qd_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha="center",
                    va="bottom" if val > 0 else "top", fontsize=10,
                    fontweight="bold")

        ax.set_ylabel("QD-Score (Yüksek = Daha İyi)")
        ax.set_title(f"Final QD-Score Karşılaştırması ({preset.capitalize()})",
                     fontweight="bold")

    if has_qd_history:
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[G1] Kaydedildi: {save_path}")


# ════════════════════════════════════════════════════════════════════
# G3 — Algoritma × Preset Isı Haritası
# ════════════════════════════════════════════════════════════════════

def plot_algo_preset_matrix(data, save_path="algo_preset_matrix.png",
                            metric="final_loss"):
    """
    6 algoritma × N preset matrisi. Her hücre o kombinasyonun
    medyan final loss değeri. Düşük = iyi (koyu renk).

    Tek bakışta "hangi algoritma hangi efektte iyi" görünür.
    """
    presets = list(data.keys())
    # Algoritma sırası — sabit
    algo_order = ["RandomSearch", "ClassicGA", "SelfAdaptiveGA",
                  "MAPElites", "CMAES", "CMA-ME"]
    # Sadece veride var olanları al
    algos = [a for a in algo_order
             if any(a in data[p] for p in presets)]

    # Matris doldur — medyan loss
    matrix = np.full((len(algos), len(presets)), np.nan)
    for j, preset in enumerate(presets):
        for i, algo in enumerate(algos):
            if algo in data[preset] and data[preset][algo]:
                losses = [r["best_loss"] for r in data[preset][algo]]
                matrix[i, j] = np.median(losses)

    setup_style()
    fig, ax = plt.subplots(figsize=(max(7, len(presets)*1.5),
                                    max(5, len(algos)*0.7)))

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

    # Eksen etiketleri
    ax.set_xticks(range(len(presets)))
    ax.set_xticklabels([p.capitalize() for p in presets])
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)

    # Her hücreye değeri yaz + en iyiyi işaretle
    for j in range(len(presets)):
        col = matrix[:, j]
        if np.all(np.isnan(col)):
            continue
        best_i = np.nanargmin(col)  # en düşük loss = en iyi
        for i in range(len(algos)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            # En iyiyi kalın + işaret
            is_best = (i == best_i)
            txt = f"{val:.3f}" + (" *" if is_best else "")
            # Renk arka plana göre kontrast
            color = "white" if val > np.nanmean(matrix) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9,
                    fontweight="bold" if is_best else "normal")

    ax.set_xlabel("Preset (Efekt Türü)", fontsize=11)
    ax.set_ylabel("Algoritma", fontsize=11)
    ax.set_title("Algoritma × Preset Performans Matrisi\n"
                 "(Medyan MR-STFT Loss — düşük/yeşil = iyi, * = sütun en iyisi)",
                 fontweight="bold", fontsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MR-STFT Loss", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[G3] Kaydedildi: {save_path}")


# ════════════════════════════════════════════════════════════════════
# G4 — Critical Difference Diyagramı
# ════════════════════════════════════════════════════════════════════

def plot_critical_difference(data, save_path="critical_difference.png",
                             alpha=0.05):
    """
    Demšar (2006) Critical Difference diyagramı.

    Friedman testi ile algoritmaların ortalama sıralarını hesaplar,
    Nemenyi post-hoc ile istatistiksel olarak ayırt edilemeyenleri
    kalın çizgiyle birleştirir.

    Her (preset, seed) kombinasyonu bir "veri seti" olarak ele alınır.
    Bu, EC literatürünün standart karşılaştırma yöntemidir.
    """
    try:
        import scikit_posthocs as sp
        from scipy.stats import friedmanchisquare, rankdata
    except ImportError:
        print("[G4] scikit-posthocs veya scipy eksik — atlanıyor")
        print("     pip install scikit-posthocs")
        return

    presets = list(data.keys())
    algo_order = ["RandomSearch", "ClassicGA", "SelfAdaptiveGA",
                  "MAPElites", "CMAES", "CMA-ME"]
    algos = [a for a in algo_order
             if all(a in data[p] for p in presets)]

    if len(algos) < 3:
        print(f"[G4] En az 3 algoritma gerekli, {len(algos)} bulundu — atlanıyor")
        return

    # Her (preset, seed) bir satır, her algoritma bir sütun → loss matrisi
    # Aynı seed sayısı varsayımı
    rows = []
    for preset in presets:
        n_seeds = min(len(data[preset][a]) for a in algos)
        for s in range(n_seeds):
            row = []
            for algo in algos:
                # Seed'e göre eşleştir (sırada olduğunu varsay)
                row.append(data[preset][algo][s]["best_loss"])
            rows.append(row)

    loss_matrix = np.array(rows)  # shape: (n_datasets, n_algos)
    n_datasets = loss_matrix.shape[0]

    # Friedman testi
    stat, p_value = friedmanchisquare(*[loss_matrix[:, i]
                                        for i in range(len(algos))])
    print(f"[G4] Friedman testi: χ²={stat:.3f}, p={p_value:.4e}")
    if p_value >= alpha:
        print(f"[G4] UYARI: Friedman anlamlı değil (p≥{alpha}). "
              f"Algoritmalar arası fark zayıf olabilir.")

    # Ortalama sıralar (düşük loss = iyi → rank 1)
    ranks = np.array([rankdata(loss_matrix[i, :]) for i in range(n_datasets)])
    avg_ranks = ranks.mean(axis=0)

    # Nemenyi critical difference
    # CD = q_alpha * sqrt(k(k+1)/(6N))
    # q_alpha (Studentized range / sqrt(2)) — k algoritma için
    k = len(algos)
    # Nemenyi q değerleri (alpha=0.05) — k=2..10
    q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
                     6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_alpha = q_alpha_table.get(k, 2.850)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_datasets))
    print(f"[G4] Critical Difference (CD) = {cd:.3f}  (N={n_datasets} veri seti)")

    # ── Çizim ──
    setup_style()
    fig, ax = plt.subplots(figsize=(10, max(3, k * 0.5)))

    # Algoritmaları ortalama sıraya göre sırala
    order = np.argsort(avg_ranks)
    sorted_algos = [algos[i] for i in order]
    sorted_ranks = avg_ranks[order]

    min_rank = 1
    max_rank = k

    # Üst eksen — rank skalası
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(0, k + 2)
    ax.invert_xaxis()  # rank 1 (en iyi) solda

    # Rank ekseni çizgisi
    ax.plot([min_rank, max_rank], [k + 1, k + 1], "k-", linewidth=1.5)
    for r in range(min_rank, max_rank + 1):
        ax.plot([r, r], [k + 1, k + 1.15], "k-", linewidth=1)
        ax.text(r, k + 1.4, str(r), ha="center", fontsize=10)

    # Her algoritmayı yerleştir
    for idx, (algo, rank) in enumerate(zip(sorted_algos, sorted_ranks)):
        y = k - idx
        # Çizgi: rank pozisyonundan algoritma etiketine
        ax.plot([rank, rank], [k + 1, y], color=ALGO_COLORS.get(algo, "gray"),
                linewidth=1.5)
        side = -1 if idx < k / 2 else 1
        label_x = (min_rank - 0.3) if idx < k / 2 else (max_rank + 0.3)
        ax.plot([rank, label_x], [y, y],
                color=ALGO_COLORS.get(algo, "gray"), linewidth=1.5)
        ha = "right" if idx < k / 2 else "left"
        ax.text(label_x + (0.05 if idx >= k/2 else -0.05), y,
                f"{algo} ({rank:.2f})", ha=ha, va="center", fontsize=10,
                fontweight="bold")

    # CD bar — sol üstte göster
    cd_y = k + 1.55
    ax.plot([max_rank, max_rank - cd], [cd_y, cd_y], "k-", linewidth=2.5)
    ax.plot([max_rank, max_rank], [cd_y - 0.08, cd_y + 0.08], "k-", linewidth=1.5)
    ax.plot([max_rank - cd, max_rank - cd], [cd_y - 0.08, cd_y + 0.08],
            "k-", linewidth=1.5)
    ax.text(max_rank - cd/2, cd_y + 0.18, f"CD = {cd:.2f}",
            ha="center", fontsize=9)

    # Ayırt edilemeyen grupları kalın çizgiyle birleştir
    # (ortalama sıraları CD'den yakın olanlar)
    clique_y = 0.5
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= cd:
                # Bu ikisi ayırt edilemez — kalın çizgi
                y_level = clique_y + i * 0.18
                ax.plot([sorted_ranks[i], sorted_ranks[j]],
                        [y_level, y_level], "k-", linewidth=3, alpha=0.6)
                break  # her grup için bir çizgi yeterli

    ax.axis("off")
    ax.set_title(f"Critical Difference Diyagramı (Nemenyi, α={alpha})\n"
                 f"Friedman p={p_value:.2e}, N={n_datasets} veri seti — "
                 f"sol = daha iyi",
                 fontweight="bold", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[G4] Kaydedildi: {save_path}")


# ════════════════════════════════════════════════════════════════════
# Ana akış
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="İleri makale grafikleri")
    parser.add_argument("--json", default="results/experiment.json",
                        help="Deney JSON dosyası")
    parser.add_argument("--output", default="results",
                        help="Çıktı klasörü")
    parser.add_argument("--preset", default=None,
                        help="QD evrimi için preset (varsayılan: ilk preset)")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"HATA: {json_path} bulunamadı.")
        print("Önce 'python main.py all' veya 'python main.py experiment' çalıştır.")
        return

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = load_data(json_path)
    presets = list(data.keys())
    print(f"Yüklenen presetler: {presets}")
    print()

    # G1 — her preset için QD evrimi
    qd_preset = args.preset or presets[0]
    plot_qd_score_evolution(data, preset=qd_preset,
                            save_path=str(out / f"qd_evolution_{qd_preset}.png"))

    # G3 — tüm presetleri kapsayan matris
    plot_algo_preset_matrix(data, save_path=str(out / "algo_preset_matrix.png"))

    # G4 — tüm presetleri kullanan kritik fark
    plot_critical_difference(data, save_path=str(out / "critical_difference.png"))

    print()
    print(f"Tüm ileri grafikler hazır: {out}/")


if __name__ == "__main__":
    main()