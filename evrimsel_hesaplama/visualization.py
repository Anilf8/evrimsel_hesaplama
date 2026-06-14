"""
visualization.py
================

Adım 5 — Deney sonuçlarını okuyarak makale kalitesinde grafikler üretir.

Üretilen Grafikler:
  1. Yakınsama Eğrisi (Convergence Plot) - Algoritmaların hız karşılaştırması
  2. Kutu Grafikleri (Boxplots) - Mouret & Clune (2015) tarzı metrik karşılaştırmaları
  3. MAP-Elites Isı Haritası (Heatmap) - Özellik uzayının aydınlatılması
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Makale kalitesi için genel ayarlar
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'


def load_data(filepath="results/experiment.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────
# 1. Yakınsama Eğrileri (Convergence Curve)
# ────────────────────────────────────────────────────────────────────
def plot_convergence(data, preset="crunch", save_path="results/convergence.png"):
    """Evrimsel algoritmaların standart yakınsama eğrisi (Loss vs Evals)"""
    plt.figure(figsize=(8, 5))

    colors = {"RandomSearch": "gray", "ClassicGA": "blue", "MAPElites": "green", "CMAES": "red"}

    preset_data = data.get(preset, {})
    for algo_name, runs in preset_data.items():
        if not runs: continue

        # Tüm seed'ler için evals ve best_loss geçmişini al
        # Not: Evaluator history_best'i fitness olarak saklıyor, loss = -fitness
        all_losses = []
        evals = runs[0]["history_evals"]  # X ekseni hepsi için aynı

        for run in runs:
            losses = [-f for f in run["history_best"]]
            all_losses.append(losses)

        all_losses = np.array(all_losses)
        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)

        plt.plot(evals, mean_loss, label=algo_name, color=colors.get(algo_name, "black"), linewidth=2)
        plt.fill_between(evals, mean_loss - std_loss, mean_loss + std_loss,
                         color=colors.get(algo_name, "black"), alpha=0.15)

    plt.title(f"Yakınsama Eğrisi ({preset.capitalize()} Preset)")
    plt.xlabel("Değerlendirme Sayısı (Evaluations)")
    plt.ylabel("MR-STFT Kaybı (Düşük = Daha İyi)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Yakınsama eğrisi kaydedildi: {save_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────
# 2. Kutu Grafikleri (Mouret & Clune 2015 Stili Boxplots)
# ────────────────────────────────────────────────────────────────────
def plot_boxplots(data, preset="crunch", save_path="results/boxplots.png"):
    """Algoritmaların final performans ve metriklerinin istatistiksel dağılımı"""
    preset_data = data.get(preset, {})

    # Verileri seaborn için uygun formata (uzun format) çevir
    plot_data = []
    for algo_name, runs in preset_data.items():
        for run in runs:
            plot_data.append({
                "Algoritma": algo_name,
                "Final Loss": run["best_loss"],
            })

    plt.figure(figsize=(6, 5))

    # Liste → DataFrame (yeni seaborn API gereği)
    df = pd.DataFrame(plot_data)

    # Renk paleti (algoritma sırasına göre)
    algo_order = list(preset_data.keys())
    palette = {
        "RandomSearch": "#cccccc",
        "ClassicGA":    "#4c72b0",
        "MAPElites":    "#55a868",
        "CMAES":        "#c44e52",
    }
    # Sadece veride bulunan algoritmaların renklerini al
    colors = [palette.get(a, "#888888") for a in algo_order]

    # Mouret & Clune makalesindeki Fig 3 ve Fig 5 stiline benzer kutu grafiği
    sns.boxplot(
        data=df, x="Algoritma", y="Final Loss",
        order=algo_order,
        hue="Algoritma", legend=False,
        palette=colors,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 7},
    )

    # Bireysel veri noktalarını üstüne ekle (strip plot)
    sns.stripplot(
        data=df, x="Algoritma", y="Final Loss",
        order=algo_order, color="black", alpha=0.4, size=4, jitter=True,
    )

    plt.title("Küresel Performans Karşılaştırması (Final Loss)")
    plt.ylabel("MR-STFT Kaybı (Düşük = Daha İyi)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Kutu grafiği kaydedildi: {save_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────
# 3. MAP-Elites Isı Haritası (Feature-Performance Map)
# ────────────────────────────────────────────────────────────────────
def plot_map_elites_heatmap(data, preset="crunch", grid_size=20, save_path="results/map_elites_heatmap.png"):
    """
    Mouret & Clune (2015) makalesindeki Fig. 3 ve 5 altındaki ikonik
    Özellik-Performans (Feature-Performance) haritaları.
    """
    preset_data = data.get(preset, {})
    if "MAPElites" not in preset_data:
        print("[HATA] MAPElites verisi bulunamadı.")
        return

    # En iyi kapsamı (coverage) yapan koşumu bulalım
    best_run = None
    best_coverage = -1
    for run in preset_data["MAPElites"]:
        cov = run["extra"]["map_elites"]["coverage"]
        if cov > best_coverage:
            best_coverage = cov
            best_run = run

    archive = best_run["extra"]["map_elites"]["archive"]

    # 2D Grid oluştur (Boş hücreler NaN olacak)
    grid = np.full((grid_size, grid_size), np.nan)

    # JSON key'leri string olarak saklar, örn: "(12, 15)" -> tuple'a çevir
    for cell_str, fitness in archive.items():
        # "(x, y)" formatından x ve y'yi çek
        cleaned = cell_str.strip("()").split(",")
        x, y = int(cleaned[0]), int(cleaned[1])
        # Fitness -> Loss (Görselleştirmede loss kullanmak daha iyi olabilir)
        loss = -float(fitness)
        grid[y, x] = loss  # Y satır, X sütun olarak matrise yerleştir

    plt.figure(figsize=(7, 6))

    # Mouret & Clune makalesinde jet/turbo benzeri renk paletleri kullanılmıştır.
    # Düşük loss (iyi) için soğuk renkler, yüksek loss (kötü) için sıcak renkler.
    ax = sns.heatmap(grid, cmap="viridis_r", cbar_kws={'label': 'MR-STFT Loss (Düşük = İyi)'},
                     square=True, xticklabels=5, yticklabels=5)

    # Eksenleri ters çevir ki [0,0] sol alt köşe olsun
    ax.invert_yaxis()

    plt.title(f"MAP-Elites Çözüm Uzayı Aydınlatması (Coverage: %{best_coverage * 100:.1f})")

    # behavior_space.py dosyanıza göre eksen isimleri
    plt.xlabel("Spectral Centroid İndeksi (Parlaklık)")
    plt.ylabel("ZCR İndeksi (Sertlik)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Isı haritası kaydedildi: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Sonuçların bulunduğu JSON dosyasının yolu
    data_file = "results/experiment.json"

    # Klasörü oluştur
    Path("results").mkdir(parents=True, exist_ok=True)

    try:
        data = load_data(data_file)
        # Sadece crunch preseti için örnek çizimler
        plot_convergence(data, preset="crunch")
        plot_boxplots(data, preset="crunch")
        plot_map_elites_heatmap(data, preset="crunch", grid_size=20)
        print("\nTüm grafikler başarıyla üretildi! Makalede kullanıma hazırdır.")
    except FileNotFoundError:
        print(f"Hata: {data_file} bulunamadı. Lütfen önce runner.py'yi çalıştırın.")