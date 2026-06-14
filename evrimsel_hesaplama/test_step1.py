"""
test_step1.py
=============

Adım 1 sağlamlık testi.

Çekirdek altyapının uçtan uca çalıştığını doğrular:
  1. Parametre uzayı 16 boyutlu mu?
  2. denormalize / normalize tersinir mi?
  3. IDMTLoader demo sinyali üretebiliyor mu?
  4. EffectChain Pedalboard işliyor mu?
  5. MR-STFT loss anlamlı sıralama veriyor mu?
  6. 5 preset benzersiz sesler üretiyor mu?

Bu test başarılıysa Adım 2'ye geçilebilir.
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore")

from core import (
    N_PARAMS, PARAM_NAMES, denormalize, normalize, random_vector,
    EffectChain, MultiResolutionSTFTLoss,
    PRESET_NAMES, get_preset_vector, IDMTLoader,
)


def banner(title: str) -> None:
    print()
    print("=" * 60)
    print("  " + title)
    print("=" * 60)


def check(condition: bool, label: str) -> None:
    mark = "OK " if condition else "FAIL"
    print(f"  [{mark}] {label}")
    if not condition:
        raise AssertionError(f"Test başarısız: {label}")


banner("Adım 1 testi — Çekirdek altyapı")

# ─────────────────────────────────────────────────────────────
# 1. Parametre uzayı
# ─────────────────────────────────────────────────────────────
banner("1. Parametre uzayı")
print(f"  Toplam boyut: {N_PARAMS}")
check(N_PARAMS == 16, "16 parametre tanımlı")
print(f"  İlk 4 parametre: {PARAM_NAMES[:4]}")

rng = np.random.default_rng(42)
x_norm = random_vector(rng)
print(f"  Rastgele x ∈ [0,1]^16, ilk 3: {x_norm[:3]}")
check(x_norm.min() >= 0.0 and x_norm.max() <= 1.0, "x değerleri [0,1] içinde")

p_real = denormalize(x_norm)
x_back = normalize(p_real)
check(np.allclose(x_norm, x_back, atol=1e-6), "denormalize ∘ normalize = identity")

# ─────────────────────────────────────────────────────────────
# 2. Dataset yükleyici (demo modu)
# ─────────────────────────────────────────────────────────────
banner("2. Dataset yükleyici (demo modu)")
loader = IDMTLoader(dataset_root=None, sample_rate=22050, verbose=True)
dry = loader.get_fixed_dry(duration=1.0, seed=42)
print(f"  Kuru sinyal: {len(dry)} örnek (~{len(dry)/22050:.2f}s)")
print(f"  Aralık: [{dry.min():+.3f}, {dry.max():+.3f}]")
check(len(dry) == 22050, "1 saniyelik sinyal üretildi")
check(np.max(np.abs(dry)) <= 1.0, "Sinyal [-1, 1] aralığında")
check(not np.allclose(dry, 0.0), "Sinyal boş değil")

# ─────────────────────────────────────────────────────────────
# 3. Efekt zinciri
# ─────────────────────────────────────────────────────────────
banner("3. Pedalboard efekt zinciri")
chain = EffectChain(sample_rate=22050)
wet = chain.process(dry, x_norm)
print(f"  Islak sinyal: {len(wet)} örnek")
print(f"  Aralık: [{wet.min():+.3f}, {wet.max():+.3f}]")
check(len(wet) == len(dry), "Islak ve kuru uzunlukları eşit")
check(np.max(np.abs(wet)) <= 1.0, "Çıkış [-1, 1] aralığında")
check(not np.allclose(wet, dry, atol=1e-4), "Efekt sinyali değiştirmiş")

# ─────────────────────────────────────────────────────────────
# 4. MR-STFT loss
# ─────────────────────────────────────────────────────────────
banner("4. MR-STFT loss")
loss_fn = MultiResolutionSTFTLoss()
loss_self  = loss_fn.compute(dry, dry)
loss_wet   = loss_fn.compute(dry, wet)
print(f"  Loss(dry, dry)  = {loss_self:.6f}   <- sıfıra yakın olmalı")
print(f"  Loss(dry, wet)  = {loss_wet:.6f}   <- pozitif olmalı")
check(loss_self < 1e-3, "Aynı sinyal için loss ≈ 0")
check(loss_wet > loss_self, "Farklı sinyal için loss daha yüksek")

# ─────────────────────────────────────────────────────────────
# 5. Hedef presetler
# ─────────────────────────────────────────────────────────────
banner("5. Hedef presetler (ground truth x*)")
preset_signals = {}
for name in PRESET_NAMES:
    x_star = get_preset_vector(name, seed=42)
    target = chain.process(dry, x_star)
    loss_to_dry = loss_fn.compute(dry, target)
    preset_signals[name] = target
    print(f"  {name:10s}: loss(dry, target) = {loss_to_dry:.4f}")
    check(x_star.shape == (N_PARAMS,), f"{name} preset şekli doğru")

# ─────────────────────────────────────────────────────────────
# 6. Presetler birbirinden farklı sesler üretiyor mu?
# ─────────────────────────────────────────────────────────────
banner("6. Presetler benzersiz mi?")
clean   = preset_signals["clean"]
heavy   = preset_signals["heavy"]
ambient = preset_signals["ambient"]

d_clean_heavy   = loss_fn.compute(clean,   heavy)
d_clean_ambient = loss_fn.compute(clean,   ambient)
d_heavy_ambient = loss_fn.compute(heavy,   ambient)

print(f"  clean ↔ heavy    : {d_clean_heavy:.4f}")
print(f"  clean ↔ ambient  : {d_clean_ambient:.4f}")
print(f"  heavy ↔ ambient  : {d_heavy_ambient:.4f}")
check(d_clean_heavy   > 0.5, "clean ve heavy belirgin farklı")
check(d_clean_ambient > 0.5, "clean ve ambient belirgin farklı")
check(d_heavy_ambient > 0.5, "heavy ve ambient belirgin farklı")

# ─────────────────────────────────────────────────────────────
# 7. Reproducibility — aynı seed aynı sonucu üretiyor mu?
# ─────────────────────────────────────────────────────────────
banner("7. Tekrarlanabilirlik")
dry1 = loader.get_fixed_dry(duration=1.0, seed=123)
dry2 = loader.get_fixed_dry(duration=1.0, seed=123)
check(np.allclose(dry1, dry2), "Aynı seed → aynı kuru sinyal")

x1 = get_preset_vector("random", seed=99)
x2 = get_preset_vector("random", seed=99)
check(np.allclose(x1, x2), "Aynı seed → aynı random preset")

# ─────────────────────────────────────────────────────────────
banner("Adım 1 BAŞARILI — Adım 2'ye geçilebilir")
print("  Sonraki adım: algorithm_base.py + random_search.py + classic_ga.py")
print("=" * 60)