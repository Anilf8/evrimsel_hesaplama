"""
main.py
=======

Tüm deney pipeline'ını tek komuttan çalıştırır.

Üç komut modu:

  1. python main.py experiment   →  Deneyi koş ve JSON kaydet
  2. python main.py visualize    →  Mevcut JSON'dan grafikleri üret
  3. python main.py all          →  Hem deneyi koş hem grafikleri üret

Yaygın senaryolar:

  # Tam deney (200 koşum, ~3 saat)
  python main.py all

  # Sadece görselleştirme (önceden koşulmuş veriyle)
  python main.py visualize

  # Hızlı test (sadece crunch, az koşum)
  python main.py experiment --presets crunch --seeds 3 --budget 1000

  # Farklı WAV dosyası ile
  python main.py all --wav my_riff.wav --duration 1.0

  # Çıktı klasörünü değiştir
  python main.py all --output results/run_01/
"""

import argparse
import sys
import time
from pathlib import Path

from runner import ExperimentRunner


# ────────────────────────────────────────────────────────────────────
# Varsayılan ayarlar
# ────────────────────────────────────────────────────────────────────

DEFAULT_WAV       = "dry_guitar_riff.wav"
DEFAULT_OUTPUT    = "results"
DEFAULT_JSON_NAME = "experiment.json"
DEFAULT_PRESETS   = ["clean", "crunch", "heavy", "ambient", "random"]
DEFAULT_BUDGET    = 10_000
DEFAULT_SEEDS     = 10
DEFAULT_DURATION  = 0.7


# ────────────────────────────────────────────────────────────────────
# Komut: experiment — sadece deneyi koş
# ────────────────────────────────────────────────────────────────────

def cmd_experiment(args) -> str:
    """Deneyi koşar, JSON dosyasına kaydeder, dosya yolunu döndürür."""
    print()
    print("=" * 66)
    print("  DENEY BAŞLIYOR")
    print("=" * 66)
    print(f"  WAV dosyası : {args.wav}")
    print(f"  Süre        : {args.duration}s")
    print(f"  Presetler   : {args.presets}")
    print(f"  Bütçe       : {args.budget} fitness değerlendirmesi")
    print(f"  Seed sayısı : {args.seeds}")
    print(f"  Çıktı       : {args.output}/")

    n_total = len(args.presets) * 4 * args.seeds
    print(f"  Toplam koşum: {n_total}")
    print()

    # WAV varsa kullan, yoksa demo
    wav_path = args.wav if Path(args.wav).exists() else None
    if wav_path is None:
        print(f"  ! UYARI: {args.wav} bulunamadı, demo sinyal kullanılacak")
        print()

    runner = ExperimentRunner(
        budget       = args.budget,
        n_seeds      = args.seeds,
        record_every = args.record_every,
        preset_names = args.presets,
        dry_path     = wav_path,
        dry_duration = args.duration,
        verbose      = True,
    )

    t0 = time.perf_counter()
    all_results = runner.run_all()
    dt = time.perf_counter() - t0

    # Kaydet
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / DEFAULT_JSON_NAME
    runner.save(all_results, str(json_path))

    print()
    print("=" * 66)
    print(f"  DENEY TAMAMLANDI")
    print(f"  Süre        : {dt/60:.1f} dakika")
    print(f"  JSON çıktı  : {json_path}")
    print("=" * 66)

    return str(json_path)


# ────────────────────────────────────────────────────────────────────
# Komut: visualize — JSON'dan grafik üret
# ────────────────────────────────────────────────────────────────────

def cmd_visualize(args, json_path: str = None) -> None:
    """Mevcut JSON dosyasını okur, grafikleri kaydeder."""
    # visualization modülünü gecikmeli import et — matplotlib yavaş yüklenir
    from visualization import (
        load_data,
        plot_convergence,
        plot_boxplots,
        plot_map_elites_heatmap,
    )

    # JSON yolu belli değilse default'u kullan
    if json_path is None:
        json_path = str(Path(args.output) / DEFAULT_JSON_NAME)

    if not Path(json_path).exists():
        print(f"HATA: {json_path} bulunamadı.")
        print(f"      Önce 'python main.py experiment' çalıştır.")
        sys.exit(1)

    print()
    print("=" * 66)
    print("  GÖRSELLEŞTİRME")
    print("=" * 66)
    print(f"  JSON kaynak : {json_path}")
    print(f"  Çıktı klasör: {args.output}/")
    print()

    data = load_data(json_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hangi presetler var?
    available_presets = list(data.keys())
    print(f"  Bulunan presetler: {available_presets}")
    print()

    # Her preset için 3 grafik üret
    for preset in available_presets:
        print(f"  --- {preset} ---")
        plot_convergence(
            data, preset=preset,
            save_path=str(output_dir / f"convergence_{preset}.png"),
        )
        plot_boxplots(
            data, preset=preset,
            save_path=str(output_dir / f"boxplots_{preset}.png"),
        )
        # MAP-Elites heatmap — sadece MAPElites koşumu varsa
        if "MAPElites" in data[preset] and data[preset]["MAPElites"]:
            plot_map_elites_heatmap(
                data, preset=preset, grid_size=20,
                save_path=str(output_dir / f"heatmap_{preset}.png"),
            )

    print()
    print("=" * 66)
    print(f"  TÜM GRAFİKLER HAZIR — {output_dir}/")
    print("=" * 66)


# ────────────────────────────────────────────────────────────────────
# Komut: all — deney + görselleştirme
# ────────────────────────────────────────────────────────────────────

def cmd_all(args) -> None:
    """Önce deneyi koş, sonra grafikleri üret."""
    json_path = cmd_experiment(args)
    cmd_visualize(args, json_path=json_path)


# ────────────────────────────────────────────────────────────────────
# Argparse kurulumu
# ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Gitar efektleri evrimsel optimizasyon — deney pipeline'ı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py all                              Tam deney + grafikler
  python main.py experiment --seeds 3 --budget 1000   Hızlı test
  python main.py visualize                        Sadece grafikler
  python main.py all --presets crunch heavy       2 preset
        """,
    )

    p.add_argument(
        "command",
        choices=["experiment", "visualize", "all"],
        help="Hangi adımı çalıştır",
    )

    # Genel parametreler
    p.add_argument("--wav",      default=DEFAULT_WAV,
                   help=f"Kuru WAV dosyası (varsayılan: {DEFAULT_WAV})")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help=f"Saniye cinsinden sinyal süresi (varsayılan: {DEFAULT_DURATION})")
    p.add_argument("--output",   default=DEFAULT_OUTPUT,
                   help=f"Çıktı klasörü (varsayılan: {DEFAULT_OUTPUT})")

    # Deney parametreleri
    p.add_argument("--presets",  nargs="+", default=DEFAULT_PRESETS,
                   help=f"Çalıştırılacak presetler (varsayılan: {DEFAULT_PRESETS})")
    p.add_argument("--budget",   type=int, default=DEFAULT_BUDGET,
                   help=f"Algoritma başına bütçe (varsayılan: {DEFAULT_BUDGET})")
    p.add_argument("--seeds",    type=int, default=DEFAULT_SEEDS,
                   help=f"Koşum tekrarı (varsayılan: {DEFAULT_SEEDS})")
    p.add_argument("--record-every", type=int, default=200,
                   help="Yakınsama eğrisi kayıt aralığı (varsayılan: 200)")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "experiment":
        cmd_experiment(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()