# Evrimsel Hesaplama ile Gitar Efekt Optimizasyonu

Gitar efekt zinciri parametrelerini evrimsel algoritmalarla otomatik optimize eden bir araştırma framework'ü. Altı farklı algoritmanın 5 hedef preset üzerindeki performansını karşılaştırır.

## Algoritmalar

| Algoritma | Açıklama |
|-----------|----------|
| `RandomSearch` | Temel referans (rastgele arama) |
| `ClassicGA` | Klasik genetik algoritma (turnuva seçimi + BLX-α çaprazlama) |
| `SelfAdaptiveGA` | Parametre başına uyarlanabilir mutasyon oranlı GA |
| `MAPElites` | Quality Diversity — MAP-Elites ızgara arşivi |
| `CMA-ES` | Kovaryans Matrisi Adaptasyonu Evrim Stratejisi |
| `CMA-ME` | Hibrit: MAP-Elites + CMA-ES |

## Arama Uzayı

16 boyutlu sürekli parametre uzayı: distorsiyon (drive, tone), 3-bantlı EQ, reverb, delay, kompresör, çıkış kazancı.

Fitness fonksiyonu: Multi-Resolution STFT Loss (hedef preset ile üretilen sesin algısal benzerliği).

## Kurulum

```bash
git clone https://github.com/KULLANICI_ADI/evrimsel_hesaplama.git
cd evrimsel_hesaplama
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Kullanım

```bash
# Tam deney + görselleştirme (200 koşum, ~3 saat)
python main.py all

# Sadece deney
python main.py experiment

# Mevcut sonuçlardan grafik üret
python main.py visualize

# Hızlı test (1 preset, 3 seed, küçük bütçe)
python main.py experiment --presets crunch --seeds 3 --budget 1000

# Kendi WAV dosyanızla
python main.py all --wav my_riff.wav --duration 1.0

# Çıktı klasörünü değiştir
python main.py all --output results/run_01/
```

Gelişmiş grafikler (Critical Difference diyagramı, QD-Score evrimi):

```bash
python advanced_plots.py
python advanced_plots.py --json results/experiment.json --output figs/
```

## Proje Yapısı

```
evrimsel_hesaplama/
├── main.py               # Ana giriş noktası
├── runner.py             # Çok koşumlu deney yöneticisi
├── visualization.py      # Yakınsama eğrileri, boxplot, heatmap
├── advanced_plots.py     # QD-Score, algoritma×preset matrisi, CD diyagramı
├── metrics.py            # Deney metrik hesaplamaları
├── core/
│   ├── parameter_space.py   # 16-boyutlu parametre tanımları
│   ├── effect_chain.py      # Pedalboard DSP zinciri
│   ├── fitness.py           # Multi-Resolution STFT Loss
│   ├── dataset_presets.py   # 5 hedef preset + IDMT loader
│   └── behavior.py          # QD davranış uzayı (MAP-Elites için)
└── algorithms/
    ├── base.py              # Soyut temel sınıf
    ├── random_search.py
    ├── classic_ga.py
    ├── self_adaptive_ga.py
    ├── map_elites.py
    ├── cma_es.py
    └── cma_me.py
```

## Deney Parametreleri

| Parametre | Değer |
|-----------|-------|
| Bütçe | 10.000 değerlendirme |
| Seed sayısı | 10 (her kombinasyon için) |
| Preset sayısı | 5 (clean, crunch, heavy, ambient, random) |
| Toplam koşum | 200+ |
| Örnekleme hızı | 22.050 Hz |

## Gereksinimler

Python 3.10+. Bağımlılıklar için `requirements.txt`.

Opsiyonel: Critical Difference diyagramı için `scikit-posthocs` (`pip install scikit-posthocs`).
