"""
WAV dosyasının enerji profili — nerede sessiz, nerede dolu?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import librosa

WAV_PATH = "dry_guitar_riff.wav"
SR_TARGET = 22050

# Dosyayı yükle
audio, file_sr = sf.read(WAV_PATH, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if file_sr != SR_TARGET:
    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR_TARGET)

print(f"Dosya: {WAV_PATH}")
print(f"  Toplam uzunluk: {len(audio)} örnek ({len(audio)/SR_TARGET:.2f}s)")
print(f"  Sample rate   : {file_sr} Hz → {SR_TARGET} Hz")
print(f"  Max abs       : {np.max(np.abs(audio)):.4f}")
print(f"  RMS           : {np.sqrt(np.mean(audio**2)):.4f}")
print()

# 100 ms'lik pencerelerle RMS hesapla
hop = int(0.1 * SR_TARGET)  # 100 ms
print("Zaman penceresi başına RMS (100 ms aralıklarla):")
print(f"  {'zaman':>8s}  {'RMS':>8s}  {'durum'}")
print("  " + "-" * 35)
for i, start in enumerate(range(0, len(audio) - hop, hop)):
    frame = audio[start:start+hop]
    rms = float(np.sqrt(np.mean(frame**2)))
    # Görsel bar
    bar_len = int(rms * 60)
    bar = "█" * bar_len if bar_len > 0 else ""
    status = "(sessiz)" if rms < 0.01 else "(düşük)" if rms < 0.05 else "(dolu)"
    print(f"  {start/SR_TARGET:6.2f}s  {rms:.4f}  {bar} {status}")

# İlk dolu sample'ın yerini bul (RMS eşik ≥ 0.01)
print()
threshold = 0.01
small_hop = int(0.005 * SR_TARGET)  # 5 ms
first_active = None
for start in range(0, len(audio) - small_hop, small_hop):
    if np.sqrt(np.mean(audio[start:start+small_hop]**2)) >= threshold:
        first_active = start / SR_TARGET
        break

if first_active is not None:
    print(f"İlk aktif örnek (~RMS > 0.01): {first_active*1000:.0f} ms")
    if first_active > 0.05:
        print(f"  → Önerilen kırpma: ilk {first_active*1000:.0f} ms'i atla")
        print(f"  → Etkin sinyal süresi: {(len(audio)/SR_TARGET - first_active):.2f}s")
else:
    print("Sinyal hiçbir noktada aktif değil görünüyor!")