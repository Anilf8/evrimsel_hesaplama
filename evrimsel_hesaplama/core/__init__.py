"""
Guitar EC Benchmark — Çekirdek altyapı paketi.

Bu paket evrimsel algoritmaların temel yapı taşlarını içerir:

    - Parametre uzayı (16 boyut)
    - Pedalboard efekt zinciri (DSP)
    - Multi-Resolution STFT loss (fitness)
    - IDMT dataset yükleyici + 5 hedef preset (ground truth)

Tüm algoritmalar bu modüller üzerinden çalışır.
"""

from .parameter_space import (
    PARAM_SPEC,
    PARAM_NAMES,
    PARAM_BOUNDS,
    N_PARAMS,
    denormalize,
    normalize,
    get_param_idx,
    random_vector,
)
from .effect_chain  import EffectChain
from .fitness       import MultiResolutionSTFTLoss
from .dataset_presets import (
    PRESETS,
    PRESET_NAMES,
    get_preset_vector,
    IDMTLoader,
)
from .behavior import (
    BehaviorSpace,
    compute_behavior,
    behavior_to_cell,
)

__all__ = [
    "PARAM_SPEC", "PARAM_NAMES", "PARAM_BOUNDS", "N_PARAMS",
    "denormalize", "normalize", "get_param_idx", "random_vector",
    "EffectChain",
    "MultiResolutionSTFTLoss",
    "PRESETS", "PRESET_NAMES", "get_preset_vector", "IDMTLoader",
    "BehaviorSpace", "compute_behavior", "behavior_to_cell",
]