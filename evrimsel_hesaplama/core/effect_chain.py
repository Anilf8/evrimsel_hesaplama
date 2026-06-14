"""
effect_chain.py
===============

Spotify Pedalboard tabanlı gitar efekt zinciri.

Sinyal akışı (sabit sıra — endüstride standart):

    Compressor → EQ (3 bant) → Distortion → Tone → Delay → Reverb → Gain

Sıralama önemli: önce compresyon yapılır, sonra EQ ile renklendirilir,
sonra distortion uygulanır, son olarak mekansal efektler (delay, reverb).
"""

import numpy as np
from pedalboard import (
    Pedalboard,
    Compressor,
    Distortion,
    Delay,
    Reverb,
    LowShelfFilter,
    PeakFilter,
    HighShelfFilter,
    Gain,
)

from core.parameter_space import denormalize, get_param_idx


class EffectChain:
    """
    Bir parametre vektörünü alır, kuru sinyale Pedalboard ile uygular,
    ıslak sinyali döndürür. Tüm algoritmalar bu sınıf üzerinden gider —
    yani 4 algoritma da aynı DSP arka ucunu kullanır (adil karşılaştırma).
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate

    def build(self, params_norm: np.ndarray) -> Pedalboard:
        """Normalize vektörden Pedalboard nesnesi oluşturur."""
        p = denormalize(params_norm)
        i = get_param_idx

        # dist_tone ∈ [0,1] → 3500 Hz high-shelf gain ∈ [-12, +12] dB
        # 0 → koyu (yüksek frekansları azalt), 1 → parlak (yüksekleri artır)
        tone_gain_db = (p[i("dist_tone")] - 0.5) * 24.0

        return Pedalboard([
            # 1. Compressor — dinamiği yönet
            Compressor(
                threshold_db = float(p[i("comp_threshold")]),
                ratio        = float(p[i("comp_ratio")]),
                attack_ms    = float(p[i("comp_attack")]),
                release_ms   = float(p[i("comp_release")]),
            ),
            # 2-4. EQ — 3 bant şekillendirme
            LowShelfFilter(
                cutoff_frequency_hz = 200.0,
                gain_db             = float(p[i("eq_bass_gain")]),
            ),
            PeakFilter(
                cutoff_frequency_hz = 1000.0,
                gain_db             = float(p[i("eq_mid_gain")]),
                q                   = 1.0,
            ),
            HighShelfFilter(
                cutoff_frequency_hz = 4000.0,
                gain_db             = float(p[i("eq_treble_gain")]),
            ),
            # 5. Distortion — doğrusal olmayan bozma
            Distortion(
                drive_db = float(p[i("dist_drive")]),
            ),
            # 6. Tone — distortion sonrası parlaklık kontrolü
            HighShelfFilter(
                cutoff_frequency_hz = 3500.0,
                gain_db             = float(tone_gain_db),
            ),
            # 7. Delay
            Delay(
                delay_seconds = float(p[i("delay_time")]),
                feedback      = float(p[i("delay_feedback")]),
                mix           = float(p[i("delay_mix")]),
            ),
            # 8. Reverb
            Reverb(
                room_size = float(p[i("rev_room")]),
                wet_level = float(p[i("rev_wet")]),
                damping   = float(p[i("rev_damping")]),
            ),
            # 9. Çıkış kazancı
            Gain(
                gain_db = float(p[i("output_gain")]),
            ),
        ])

    def process(
        self,
        dry: np.ndarray,
        params_norm: np.ndarray,
        normalize_output: bool = True,
    ) -> np.ndarray:
        """
        Kuru sinyale efekt zincirini uygular.

        Args:
            dry: Tek kanallı float32 ses dizisi.
            params_norm: 16 boyutlu normalize parametre vektörü.
            normalize_output: True ise çıkışı [-0.99, 0.99] aralığına sıkıştır.

        Returns:
            İşlenmiş float32 ses dizisi.
        """
        chain = self.build(params_norm)
        wet = chain(dry.astype(np.float32), self.sr)
        if normalize_output:
            mx = float(np.max(np.abs(wet)))
            if mx > 1.0:
                wet = wet / mx * 0.99
        return wet.astype(np.float32)