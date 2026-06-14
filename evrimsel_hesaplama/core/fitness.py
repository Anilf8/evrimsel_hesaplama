"""
fitness.py
==========

Multi-Resolution STFT loss — perceptual ses benzerliği için literatür standardı.

OPTİMİZASYON v2:
  - Hedef sinyalin STFT'leri sadece BİR KEZ hesaplanır (cache_target)
  - Sonraki her eval'de sadece tahmin sinyalin STFT'leri hesaplanır
  - Beklenen hızlanma: ~2x

İki sinyali 5 farklı FFT pencere boyutunda karşılaştırır. Tek bir
çözünürlüğe göre çok daha sağlam: hem zaman hem frekans alanındaki
farkları yakalar.

Referans:
    Engel et al. (2020) DDSP: Differentiable Digital Signal Processing.
    Yamamoto et al. (2019) Parallel WaveGAN.

Bu sınıf "loss" döndürür — daha düşük değer = daha iyi eşleşme.
Algoritma tarafında:  fitness = -loss  (maksimizasyon için)
"""

from typing import List, Optional, Dict
import numpy as np


class MultiResolutionSTFTLoss:
    """
    5 ölçek MR-STFT loss.

    Her ölçek için 2 metrik birleştirilir:
      1. Spectral convergence:    ||S_t - S_p||_F / ||S_t||_F
      2. Log magnitude:           ||log|S_t| - log|S_p|||_1

    Sonuç:  ortalama(2 metriğin toplamı, 5 ölçek üzerinden)

    Performans:
      - compute(target, pred): klasik API — her seferinde her ikisinin STFT'sini hesaplar
      - cache_target(target) + compute_fast(pred): hedef bir kez, sonra sadece tahmin
        ~2x hızlanma — özellikle 10.000+ değerlendirmeli deneyler için kritik
    """

    DEFAULT_FFT_SIZES: List[int] = [64, 128, 256, 512, 1024]

    def __init__(
        self,
        fft_sizes: List[int] = None,
        hop_ratio: float = 0.25,
        eps: float = 1e-7,
    ):
        self.fft_sizes = fft_sizes if fft_sizes is not None else self.DEFAULT_FFT_SIZES
        self.hop_ratio = hop_ratio
        self.eps = eps

        # Hedef sinyal cache — cache_target() ile doldurulur
        self._target_cache: Optional[Dict[int, np.ndarray]] = None
        self._target_log_cache: Optional[Dict[int, np.ndarray]] = None
        self._target_norm_cache: Optional[Dict[int, float]] = None
        self._target_length: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────
    # Vektörleştirilmiş STFT — Adım 2'de optimize ettiğimiz versiyon
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _stft_magnitude(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
        """
        Vektörleştirilmiş STFT — magnitude döndürür.
        sliding_window_view + batch FFT.
        """
        x = np.asarray(x, dtype=np.float32)
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))

        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
        window = np.hanning(n_fft).astype(np.float32)
        spec   = np.abs(np.fft.rfft(frames * window, axis=1))
        return spec.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────
    # Cache yönetimi
    # ─────────────────────────────────────────────────────────────────

    def cache_target(self, target: np.ndarray) -> None:
        """
        Hedef sinyalin STFT'lerini önceden hesapla ve sakla.

        Tek bir hedef için çok sayıda tahmin karşılaştırılacaksa
        (örn. 10.000 fitness değerlendirmesi) ÇOK ÖNEMLİ optimizasyon.
        """
        target = np.asarray(target, dtype=np.float32)
        self._target_cache      = {}
        self._target_log_cache  = {}
        self._target_norm_cache = {}
        self._target_length     = len(target)

        for n_fft in self.fft_sizes:
            hop = max(1, int(n_fft * self.hop_ratio))
            T   = self._stft_magnitude(target, n_fft, hop)
            self._target_cache[n_fft]      = T
            self._target_log_cache[n_fft]  = np.log(T + self.eps)
            self._target_norm_cache[n_fft] = float(np.linalg.norm(T, "fro"))

    def clear_cache(self) -> None:
        """Cache'i temizle (yeni hedefe geçerken)."""
        self._target_cache      = None
        self._target_log_cache  = None
        self._target_norm_cache = None
        self._target_length     = None

    @property
    def is_cached(self) -> bool:
        return self._target_cache is not None

    # ─────────────────────────────────────────────────────────────────
    # Hızlı yol: cache_target() çağrıldıktan sonra
    # ─────────────────────────────────────────────────────────────────

    def compute_fast(self, pred: np.ndarray) -> float:
        """
        Hedef cache kullanılarak hızlı loss hesabı.

        cache_target() çağrılmamışsa hata fırlatır.
        """
        if not self.is_cached:
            raise RuntimeError(
                "compute_fast() öncesinde cache_target() çağırın. "
                "Veya cache gerektirmeyen compute(target, pred) kullanın."
            )

        pred = np.asarray(pred, dtype=np.float32)
        # Uzunlukları eşitle
        n = min(self._target_length, len(pred))
        if n < 64:
            return float("inf")
        pred = pred[:n]

        losses = []
        for n_fft in self.fft_sizes:
            hop = max(1, int(n_fft * self.hop_ratio))
            P   = self._stft_magnitude(pred, n_fft, hop)
            T   = self._target_cache[n_fft]

            # Boyutları eşitle (hop sebepli uzunluk farkı olabilir)
            min_frames = min(T.shape[0], P.shape[0])
            T_use   = T[:min_frames]
            P_use   = P[:min_frames]
            T_norm  = self._target_norm_cache[n_fft]   # önceden hesaplanmış
            T_log   = self._target_log_cache[n_fft][:min_frames]

            # Spectral convergence
            spec_conv = np.linalg.norm(T_use - P_use, "fro") / (T_norm + self.eps)

            # Log magnitude
            log_mag = float(np.mean(np.abs(
                T_log - np.log(P_use + self.eps)
            )))

            losses.append(float(spec_conv + log_mag))

        return float(np.mean(losses))

    # ─────────────────────────────────────────────────────────────────
    # Yavaş yol — geriye uyumluluk
    # ─────────────────────────────────────────────────────────────────

    def _single_resolution_loss(
        self, target: np.ndarray, pred: np.ndarray, n_fft: int,
    ) -> float:
        """Tek bir FFT boyutunda spectral convergence + log magnitude."""
        hop = max(1, int(n_fft * self.hop_ratio))
        T = self._stft_magnitude(target, n_fft, hop)
        P = self._stft_magnitude(pred,   n_fft, hop)

        norm_T = float(np.linalg.norm(T, "fro"))
        spec_conv = np.linalg.norm(T - P, "fro") / (norm_T + self.eps)

        log_mag = float(np.mean(np.abs(
            np.log(T + self.eps) - np.log(P + self.eps)
        )))
        return float(spec_conv + log_mag)

    def compute(self, target: np.ndarray, pred: np.ndarray) -> float:
        """
        Klasik API — hedef cache kullanmaz, her seferinde her ikisinin STFT'sini
        hesaplar. Cache eksikse veya geriye uyumluluk gerekiyorsa kullan.
        """
        target = np.asarray(target, dtype=np.float32)
        pred   = np.asarray(pred,   dtype=np.float32)
        n = min(len(target), len(pred))
        if n < 64:
            return float("inf")
        target, pred = target[:n], pred[:n]

        losses = [
            self._single_resolution_loss(target, pred, nfft)
            for nfft in self.fft_sizes
        ]
        return float(np.mean(losses))

    def as_fitness(self, target: np.ndarray, pred: np.ndarray) -> float:
        """Loss'u maksimize edilebilir bir fitness'a çevirir."""
        return -self.compute(target, pred)