"""
Quick verification — gerçekten WAV mı yüklendi?

İki test:
  1. IDMTLoader.get_fixed_dry() ile yüklenen demo sinyal
  2. _load_dry_from_wav() ile yüklenen WAV (eğer dosya varsa)

Aynıysa bug var, farklıysa her şey doğru çalışıyor.
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np

from runner import ExperimentRunner

# Test 1: dry_path YOK — demo modu
runner_demo = ExperimentRunner(
    budget=100, n_seeds=1, preset_names=["crunch"],
    dry_path=None, verbose=False,
)
demo_signal = runner_demo.dry

# Test 2: WAV varsa kullan
WAV_PATH = "dry_guitar_riff.wav"
if not os.path.exists(WAV_PATH):
    print(f"UYARI: {WAV_PATH} bulunamadı, sadece demo testi yapılıyor")
    print(f"Demo sinyal istatistikleri:")
    print(f"  uzunluk : {len(demo_signal)}")
    print(f"  ilk 5   : {demo_signal[:5]}")
    print(f"  RMS     : {np.sqrt(np.mean(demo_signal**2)):.4f}")
    print(f"  max abs : {np.max(np.abs(demo_signal)):.4f}")
else:
    runner_wav = ExperimentRunner(
        budget=100, n_seeds=1, preset_names=["crunch"],
        dry_path=WAV_PATH, dry_duration=0.7, verbose=False,
    )
    wav_signal = runner_wav.dry

    print()
    print("=" * 60)
    print("Demo sinyal (yapay):")
    print(f"  uzunluk : {len(demo_signal)}")
    print(f"  ilk 5   : {demo_signal[:5]}")
    print(f"  RMS     : {np.sqrt(np.mean(demo_signal**2)):.4f}")
    print(f"  max abs : {np.max(np.abs(demo_signal)):.4f}")
    print()
    print("WAV sinyal (senin riff):")
    print(f"  uzunluk : {len(wav_signal)}")
    print(f"  ilk 5   : {wav_signal[:5]}")
    print(f"  RMS     : {np.sqrt(np.mean(wav_signal**2)):.4f}")
    print(f"  max abs : {np.max(np.abs(wav_signal)):.4f}")
    print()

    diff = np.max(np.abs(demo_signal[:len(wav_signal)] - wav_signal[:len(demo_signal)]))
    print(f"İki sinyal arası max fark: {diff:.4f}")
    if diff < 0.01:
        print(">>> SORUN: Sinyaller aynı, WAV yüklenmemiş!")
    else:
        print(">>> İYİ: Sinyaller farklı — WAV doğru yüklendi.")
    print("=" * 60)