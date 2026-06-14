"""
profile_bottleneck.py
=====================

Hangi bileşen yavaş, ölçelim. Tek bir fitness değerlendirmesinde
geçen süreyi bileşenlere ayırıyoruz.
"""
import warnings, time
import numpy as np

warnings.filterwarnings("ignore")

from core import (
    EffectChain, MultiResolutionSTFTLoss, IDMTLoader, get_preset_vector,
    N_PARAMS,
)

SR    = 22050
N     = 50  # Kaç değerlendirme zamanlanacak

loader = IDMTLoader(None, SR, verbose=False)
chain  = EffectChain(SR)
loss   = MultiResolutionSTFTLoss()

dry    = loader.get_fixed_dry(0.7, seed=42)
x_star = get_preset_vector("crunch", seed=42)
target = chain.process(dry, x_star)

print(f"Sinyal uzunluğu: {len(dry)} örnek ({len(dry)/SR:.2f}s)")
print(f"N = {N} değerlendirme zamanlanıyor")
print()

# Test 1: Sadece efekt zinciri
t0 = time.time()
for _ in range(N):
    x = np.random.uniform(0, 1, N_PARAMS).astype(np.float32)
    wet = chain.process(dry, x)
t_chain = time.time() - t0
print(f"  Sadece efekt zinciri:  {t_chain:.2f}s  →  {t_chain/N*1000:.1f} ms/eval")

# Test 2: Sadece fitness (aynı sinyal ile)
t0 = time.time()
for _ in range(N):
    _ = loss.compute(target, wet)
t_fitness = time.time() - t0
print(f"  Sadece MR-STFT loss:   {t_fitness:.2f}s  →  {t_fitness/N*1000:.1f} ms/eval")

# Test 3: Birlikte (gerçek senaryo)
t0 = time.time()
for _ in range(N):
    x = np.random.uniform(0, 1, N_PARAMS).astype(np.float32)
    wet = chain.process(dry, x)
    _ = loss.compute(target, wet)
t_total = time.time() - t0
print(f"  Toplam (chain+fitness): {t_total:.2f}s  →  {t_total/N*1000:.1f} ms/eval")

print()
print(f"2000 eval beklenen süre: {t_total/N*2000:.0f}s  ({t_total/N*2000/60:.1f} dk)")
print()

# Hangisi darboğaz?
if t_fitness > t_chain * 2:
    print("DARBOĞAZ: MR-STFT loss — STFT'yi vektörleştirmek gerek")
elif t_chain > t_fitness * 2:
    print("DARBOĞAZ: Pedalboard efekt zinciri")
else:
    print("İkisi de payını alıyor")