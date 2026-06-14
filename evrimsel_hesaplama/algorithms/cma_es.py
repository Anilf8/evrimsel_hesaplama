"""
cma_es.py
=========

CMA-ES — Covariance Matrix Adaptation Evolution Strategy.

Referans:
  Hansen, N. & Ostermeier, A. (2001)
  "Completely Derandomized Self-Adaptation in Evolution Strategies"
  Evolutionary Computation, 9(2), 159-195.

  Hansen, N. (2016) "The CMA Evolution Strategy: A Tutorial"
  arXiv:1604.00772  ← implementasyonun temel başvurusu

CMA-ES nedir?
  Gaussian örnekleme tabanlı kara-kutu optimizasyon algoritması.
  Her nesilde bir dağılım N(m, σ²·C) kullanarak λ aday üretir,
  en iyi μ tanesini seçer ve bu seçime göre:
    - ortalama vektörünü (m) günceller
    - adım büyüklüğünü (σ) adapte eder    → CSA (Cumulative Step-size Adaptation)
    - kovaryans matrisini (C) günceller   → CMA

Neden CMA-ES?
  GA ve RandomSearch'ten farkı: koordinat eksenlerinden bağımsız.
  Parametreler arasındaki ilişkiyi (kovaryans) öğrenir.
  İkinci-dereceden yüzeyler için quadratic convergence gösterir.
  16 boyutlu bu problem için pratik en iyi seçeneklerden biridir.

Bu proje CMA-ES'i pycma kütüphanesi OLMADAN saf NumPy ile uygular.
Sebep: bütçe sayımını ve seed kontrolünü Evaluator'da tutmak için.
pycma kendi iç sayacını kullanır ve base.Evaluator ile entegrasyonu güçleşir.

Terminoloji (Hansen tutorial'dan):
  m   : dağılım ortalaması (mean), n_params boyutlu
  σ   : adım büyüklüğü (step-size / sigma)
  C   : kovaryans matrisi, (n_params × n_params)
  λ   : nesil başına aday sayısı (population size)
  μ   : seçilen en iyi aday sayısı (parent count)
  w_i : ağırlık vektörü (w_1 ≥ w_2 ≥ … ≥ w_μ > 0)
  p_σ : adım-büyüklüğü yol birikim vektörü (evolution path for σ)
  p_c : kovaryans güncelleme yol birikim vektörü (evolution path for C)
  μ_eff: etkin seçim kütlesi (effective selection mass)
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from core import N_PARAMS
from .base import Algorithm, Evaluator, BudgetExhausted


class CMAES(Algorithm):
    """
    Saf NumPy CMA-ES implementasyonu.

    Sınır yönetimi:
      Arama uzayı [0, 1]^16 ile sınırlıdır.
      CMA-ES doğası gereği sınırsız uzayda çalışır.
      Bu implementasyon sınır dışı adayları kesmez (clip etmez),
      onun yerine sınır dışına çıkan bireyleri evaluator'a göndermeden
      önce yansıtır (mirror boundary). Bu:
        a) Dağılımın sınırda çökmesini (boundary stagnation) önler.
        b) Kovaryans güncellemelerinin bias almasını azaltır.
        c) Evaluator bütçesini boşa harcamaz.

    Yakınsama kriteri:
      σ < sigma_tol veya bütçe dolduğunda durur.
      (Erken durdurmak ilerleyen adımda opsiyonel olarak eklenebilir.)

    Parametreler:
      sigma0      : Başlangıç adım büyüklüğü. [0,1] uzayında 0.2–0.5 arası
                    önerilir. Küçük sigma → yerel arama, büyük sigma → keşif.
      popsize     : λ (nesil başına aday). None → Hansen'ın 4+⌊3·ln(n)⌋ formülü.
      mu_frac     : μ/λ oranı (None → 0.5). Seçim baskısını belirler.
      mean0       : Başlangıç ortalama. None → [0,1]^n uniform rastgele.
      sigma_tol   : σ bu değerin altına düşünce dur (yakınsama kriteri).
      max_iter    : Maksimum nesil sayısı (None → bütçeyle sınırlı).
    """

    name = "CMAES"

    def __init__(
        self,
        sigma0:    float         = 0.3,
        popsize:   Optional[int] = None,
        mu_frac:   float         = 0.5,
        mean0:     Optional[np.ndarray] = None,
        sigma_tol: float         = 1e-8,
        max_iter:  Optional[int] = None,
    ):
        super().__init__()
        self.sigma0    = sigma0
        self.popsize   = popsize
        self.mu_frac   = mu_frac
        self.mean0     = mean0
        self.sigma_tol = sigma_tol
        self.max_iter  = max_iter

    # ────────────────────────────────────────────────────────────────
    # Yardımcılar
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _mirror_boundary(x: np.ndarray) -> np.ndarray:
        """
        [0,1] sınırı için yansıtma (mirror/reflection).

        Sınır dışına çıkan değerleri katla-geri-yansıt şeklinde işler:
          0.0 → 0.0  (sınır üzerinde)
          1.1 → 0.9  (üst sınır yansıması)
         -0.1 → 0.1  (alt sınır yansıması)
          1.6 → 0.4  (çift yansıma: önce 1.0'dan, sonra 0.0'dan)

        Bu yöntem clip'ten daha iyidir çünkü:
          - Sınıra yapışma (stagnation) oluşturmaz.
          - Kovaryans matrisinin sınırda bias kazanmasını engeller.
        """
        # Periyodun 2 olduğu testere dalgası dönüşümü
        x = x % 2.0                  # [0, 2) aralığına getir
        x = np.where(x > 1.0, 2.0 - x, x)   # üst yarıyı yansıt
        return x.astype(np.float32)

    # ────────────────────────────────────────────────────────────────
    # CMA-ES parametre başlatma
    # ────────────────────────────────────────────────────────────────

    def _init_hyperparams(self, n: int):
        """
        Hansen (2016) Tablo 1'deki standart CMA-ES hiperparametrelerini hesapla.

        n : problem boyutu (= N_PARAMS = 16)

        Döndürür: (λ, μ, w, μ_eff, c_σ, d_σ, c_c, c_1, c_μ, chi_n)
        """
        # λ: nesil başına aday sayısı
        lam = self.popsize if self.popsize else 4 + int(np.floor(3 * np.log(n)))

        # μ: kaç en iyi birey seçilir
        mu = max(1, int(np.floor(lam * self.mu_frac)))

        # Ağırlıklar: w_i = ln(μ+0.5) - ln(i)  (i=1,...,μ)
        # Pozitif, azalan, sonra normalize → toplamları 1
        raw_w = np.array([np.log(mu + 0.5) - np.log(i + 1)
                          for i in range(mu)], dtype=np.float64)
        w = raw_w / raw_w.sum()        # normalize: Σw_i = 1

        # μ_eff: etkin seçim kütlesi
        # μ_eff = (Σw_i)² / Σw_i²  — tek birey seçilseydi 1, tümü eşit ağırlıklıysa μ
        mu_eff = 1.0 / np.sum(w ** 2)

        # ── Adım büyüklüğü kontrolü (CSA) ──
        # c_σ: p_σ birikim hızı (düşük → yavaş güncelleme, yüksek → hızlı)
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        # d_σ: adım büyüklüğü sönümleme faktörü
        d_sigma = (1.0
                   + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0)
                   + c_sigma)

        # ── Kovaryans matrisi güncelleme (CMA) ──
        # c_c: p_c birikim hızı (rank-1 için)
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        # c_1: rank-1 güncelleme ağırlığı
        c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        # c_μ: rank-μ güncelleme ağırlığı
        c_mu = min(
            1.0 - c_1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff)
        )

        # χ_n: n boyutlu standart normalin beklenen normu
        # E[||N(0,I)||] ≈ √n · (1 - 1/(4n) + 1/(21n²))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        return lam, mu, w, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_n

    # ────────────────────────────────────────────────────────────────
    # Ana döngü
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        evaluator: Evaluator,
        rng:       np.random.Generator,
    ) -> None:
        """
        CMA-ES ana döngüsü.

        Her nesil:
          1. λ aday örnekle: x_k = m + σ · N(0, C)
          2. Sınır yansıtma uygula
          3. Hepsini değerlendir (evaluator çağrısı)
          4. En iyi μ'yu fitness'a göre sırala
          5. m'yi ağırlıklı ortalama ile güncelle
          6. p_σ, p_c yollarını biriktir
          7. σ'yı CSA ile adapte et
          8. C'yi rank-1 + rank-μ güncellemesiyle adapte et
          9. Durma kriteri kontrol et (σ_tol veya bütçe)
        """
        n = self.n_params   # 16

        # ── Hiperparametreler ──
        lam, mu, w, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_n = \
            self._init_hyperparams(n)

        # ── Başlangıç durumu ──
        if self.mean0 is not None:
            m = self.mean0.copy().astype(np.float64)
        else:
            # [0.3, 0.7] aralığında başla: arama uzayının ortasında,
            # sınırlardan uzak. Bu kovaryans güncellemelerinin erken
            # sıkışmasını önler.
            m = rng.uniform(0.3, 0.7, n).astype(np.float64)

        sigma = float(self.sigma0)          # adım büyüklüğü

        C   = np.eye(n, dtype=np.float64)   # kovaryans matrisi (başlangıç: birim matris)
        p_c = np.zeros(n, dtype=np.float64) # kovaryans yol birikim vektörü
        p_s = np.zeros(n, dtype=np.float64) # adım büyüklüğü yol birikim vektörü

        # Eigendecomposition cache: C = B · D² · Bᵀ
        # B: eigenvektör matrisi, D: eigendeğerlerin karekökü (diagonal)
        # Her nesil yeniden hesaplamak yerine aralıklı güncelle.
        B  = np.eye(n, dtype=np.float64)
        D  = np.ones(n, dtype=np.float64)   # D[i] = √λ_i
        BD = B * D                           # = B @ diag(D)  örnekleme için

        # Eigendecomposition güncelleme aralığı:
        # Hansen önerisi: her max(1, ⌊1/(10·n·(c1+cmu))⌋) nesilde bir.
        # Sık güncelleme = maliyetli ama tutarlı; seyrek = hızlı ama hatalı.
        eigen_interval = max(1, int(np.floor(1.0 / (10.0 * n * (c_1 + c_mu)))))

        gen = 0  # nesil sayacı

        while not evaluator.exhausted():
            # ── Durma kriteri: σ çok küçüldü ──
            if sigma < self.sigma_tol:
                break

            # ── Durma kriteri: maksimum iterasyon ──
            if self.max_iter is not None and gen >= self.max_iter:
                break

            gen += 1

            # ── 1. Aday örnekleme ──
            # z_k ~ N(0, I),  y_k = B·D·z_k ~ N(0, C),  x_k = m + σ·y_k
            # BD = B @ diag(D) önceki iterasyondan hazır.
            z = rng.standard_normal((lam, n))   # (λ, n)
            y = z @ BD.T                         # (λ, n)  — y_k = BD · z_k
            x_raw = m + sigma * y               # (λ, n)  — gerçek aday

            # ── 2. Sınır yansıtma ──
            x_cand = np.apply_along_axis(
                self._mirror_boundary, 1, x_raw
            ).astype(np.float64)

            # ── 3. Değerlendirme ──
            fitnesses = np.empty(lam, dtype=np.float64)
            for k in range(lam):
                if evaluator.exhausted():
                    # Bütçe nesil ortasında bitti:
                    # değerlendirilen kısmi nesli sıralayacak kadar
                    # veri yoksa çık. Evaluator.best_x zaten en iyiyi tutar.
                    return
                fitnesses[k] = evaluator(x_cand[k].astype(np.float32))

            # ── 4. Sıralama: en iyi μ aday ──
            # higher_is_better=True olduğu için büyük fitness = iyi
            order     = np.argsort(-fitnesses)   # azalan sıra
            elite_idx = order[:mu]               # ilk μ tanesi seçildi

            # Seçilen z ve y vektörleri (m'den sapma, normalize edilmiş uzayda)
            z_elite = z[elite_idx]               # (μ, n)
            y_elite = y[elite_idx]               # (μ, n)

            # ── 5. Mean güncelleme ──
            # m_new = Σ w_i · x_{i:λ}   (ağırlıklı ortalama)
            # Eşdeğer: m_new = m + σ · Σ w_i · y_{i:λ}
            y_w  = (w[:, np.newaxis] * y_elite).sum(axis=0)   # (n,)
            m_new = m + sigma * y_w

            # ── 6. Evolution paths (yol birikim vektörleri) ──

            # p_σ güncelleme (CSA için):
            # p_σ ← (1-c_σ)·p_σ + √(c_σ·(2-c_σ)·μ_eff) · B·(Σ w_i·z_{i:λ})
            z_w   = (w[:, np.newaxis] * z_elite).sum(axis=0)  # (n,) normalize arama yönü
            Bz_w  = B @ z_w                                    # (n,) normalize edilmiş yön
            p_s   = ((1.0 - c_sigma) * p_s
                     + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * Bz_w)

            # h_σ (heaviside indikatörü): p_σ normu çok büyükse rank-1 baskıla
            # Normu beklenenden büyükse dağılım henüz oturmamıştır.
            p_s_norm = float(np.linalg.norm(p_s))
            h_sigma  = int(
                p_s_norm / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * gen))
                < (1.4 + 2.0 / (n + 1.0)) * chi_n
            )

            # p_c güncelleme (kovaryans rank-1 için):
            # p_c ← (1-c_c)·p_c + h_σ · √(c_c·(2-c_c)·μ_eff) · (Σ w_i·y_{i:λ})
            p_c = ((1.0 - c_c) * p_c
                   + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w)

            # ── 7. Adım büyüklüğü (σ) güncelleme — CSA ──
            # σ ← σ · exp(c_σ/d_σ · (||p_σ||/χ_n - 1))
            # ||p_σ|| < χ_n → σ küçül (yerel arama)
            # ||p_σ|| > χ_n → σ büyü  (keşif)
            sigma = sigma * np.exp(
                (c_sigma / d_sigma) * (p_s_norm / chi_n - 1.0)
            )
            # σ patlamaması için güvenlik klipsi
            sigma = float(np.clip(sigma, 1e-10, 1.0))

            # ── 8. Kovaryans matrisi güncelleme ──
            # C ← (1 - c_1 - c_μ)·C
            #       + c_1·(p_c·p_cᵀ + (1-h_σ)·c_c·(2-c_c)·C)   ← rank-1
            #       + c_μ · Σ w_i · y_{i:λ}·y_{i:λ}ᵀ            ← rank-μ

            # Rank-1 terimi
            rank1 = np.outer(p_c, p_c)

            # Rank-μ terimi: Σ w_i · y_i · y_iᵀ
            rank_mu = sum(w[i] * np.outer(y_elite[i], y_elite[i])
                          for i in range(mu))

            # δ(h_σ): h_σ=0 olduğunda C'ye düzeltme (nümerik stabilite)
            delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c)

            C = ((1.0 - c_1 - c_mu) * C
                 + c_1 * (rank1 + delta_h * C)
                 + c_mu * rank_mu)

            # Simetri zorla (nümerik hata birikimi nedeniyle C asimetrik olabilir)
            C = 0.5 * (C + C.T)

            # ── 9. Eigendecomposition güncelleme (aralıklı) ──
            # C = B · diag(D²) · Bᵀ
            # Örnekleme: N(0, C) = B · diag(D) · N(0, I)
            if gen % eigen_interval == 0:
                # Simetrik eigendecomposition: her zaman gerçek özdeğer garantili
                try:
                    eigenvalues, B = np.linalg.eigh(C)
                    # Nümerik negatif özdeğerleri sıfırla (C kesinlikle pozitif semi-definite)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    D  = np.sqrt(eigenvalues)   # D[i] = √λ_i
                    BD = B * D                  # BD[i,j] = B[i,j] * D[j]
                except np.linalg.LinAlgError:
                    # Eğer C sıfırlanamıyorsa (sayısal kararsızlık),
                    # önceki B ve D'yi koru — bir nesil atla.
                    pass

            # Mevcut ortalamayı güncelle
            m = m_new

        # ────────────────────────────────────────────────────────────
        # Döngü bitti: bütçe doldu veya sigma_tol aşıldı.
        # evaluator.best_x ve evaluator.best_fitness zaten tutuldu.
        # build_result() base sınıfından devralınır.
        # ────────────────────────────────────────────────────────────