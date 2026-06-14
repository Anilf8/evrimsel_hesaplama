"""
cma_me.py
=========

CMA-ME — Covariance Matrix Adaptation MAP-Elites.

Referans:
    Fontaine, M. C., Togelius, J., Nikolaidis, S., & Hoover, A. K. (2020).
    "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space."
    GECCO 2020.

═══════════════════════════════════════════════════════════════════
HİBRİT FİKİR
═══════════════════════════════════════════════════════════════════

Klasik MAP-Elites:
    Arşivden rastgele elite seç → Gaussian σ=0.08 ile mutasyon → arşive ekle.
    Sabit σ, yön bilgisi yok.

CMA-ME:
    Arşivden rastgele elite seç → CMA-ES emitter'ı bu elite üzerinde başlat
    → λ çocuk örnekle → arşive ekle → kovaryans matrisini iyileşmeye göre güncelle
    → emitter restart edilene kadar devam et.

Sonuç: Hem MAP-Elites'in çeşitlilik koruyan arşivi var,
       hem CMA-ES'in adaptif arama yönü var.

═══════════════════════════════════════════════════════════════════
EMITTER RESTART KOŞULLARI (Fontaine et al. 2020 §3.3)
═══════════════════════════════════════════════════════════════════

CMA-ES emitter durmuşsa (σ çok küçük veya pycma stop()):
    → Arşivden yeni rastgele elite seç
    → Yeni CMA-ES instance başlat
    → Tekrar devam et

Bu, lokal optimumlardan kurtulup keşfe geri dönmeyi sağlar.

═══════════════════════════════════════════════════════════════════
ÖDÜLLENDİRME — İYİLEŞME SIRALAMASI
═══════════════════════════════════════════════════════════════════

Klasik CMA-ES çocukları FITNESS'a göre sıralar.
CMA-ME çocukları İYİLEŞMEYE göre sıralar:
    Δ(x_child) = f(x_child) - f(arşivdeki_komşusu)

Eğer çocuk arşivde yeni hücre dolduruyorsa: yüksek ödül.
Eğer mevcut hücreyi geçiyorsa: orta ödül.
Eğer geçmiyorsa: düşük ödül.

Bu sayede CMA-ES "arşivi doldur" yönünde optimize olur, sadece tek
fitness peak'i aramak yerine.
"""

from typing import Callable, Dict, List, Optional, Tuple
import warnings
import numpy as np
import cma

from core import (
    N_PARAMS,
    BehaviorSpace, behavior_to_cell,
)
from .base import Algorithm, Evaluator, BudgetExhausted


class CMAME(Algorithm):
    """
    CMA-ME — CMA-ES emitter'lı MAP-Elites.

    Arşiv ve davranış uzayı MAP-Elites ile aynı.
    Mutasyon yerine CMA-ES örnekleme dağılımı kullanılır.
    """

    name = "CMA-ME"

    def __init__(
        self,
        behavior_fn:    Callable[[np.ndarray], Tuple[float, float]],
        behavior_space: BehaviorSpace = None,
        n_init:         int           = 200,
        sigma0:         float         = 0.15,
        emitter_popsize: int          = None,    # None → pycma default
        verbose:        int           = -9,
    ):
        super().__init__()
        self.behavior_fn     = behavior_fn
        self.behavior_space  = behavior_space or BehaviorSpace()
        self.n_init          = n_init
        self.sigma0          = sigma0
        self.emitter_popsize = emitter_popsize
        self.verbose         = verbose

        # Arşiv: cell (i,j) → (params, fitness, behavior)
        self.archive: Dict[Tuple[int, int], Tuple[np.ndarray, float, Tuple[float, float]]] = {}

        # İstatistikler
        self.n_emitters_used = 0
        self.n_restarts      = 0

    # ────────────────────────────────────────────────────────────────
    # MAP-Elites arşiv mekanizması (MAPElites'tan kopyalandı)
    # ────────────────────────────────────────────────────────────────

    def _try_insert(self, x: np.ndarray, fitness: float) -> Tuple[bool, float]:
        """
        Bir bireyi arşive yerleştirmeyi dene.

        Returns:
            (was_improvement: bool, improvement_amount: float)
            - was_improvement: hücre boşsa veya fitness artıysa True
            - improvement_amount: yeni_fitness - eski_fitness (eski yoksa fitness)
        """
        b = self.behavior_fn(x)
        cell = behavior_to_cell(b[0], b[1], self.behavior_space)

        existing = self.archive.get(cell)
        if existing is None:
            # Yeni hücre — büyük ödül
            self.archive[cell] = (x.copy(), fitness, b)
            return True, fitness  # max ödül: tüm fitness değeri

        if fitness > existing[1]:
            # Mevcut hücreyi geçti — küçük ödül (sadece fark)
            old_fitness = existing[1]
            self.archive[cell] = (x.copy(), fitness, b)
            return True, fitness - old_fitness

        # Geçemedi
        return False, 0.0

    def _select_random_elite(self, rng: np.random.Generator) -> np.ndarray:
        """Arşivden rastgele bir elite seç (CMA-ES'in başlangıç ortalaması olacak)."""
        keys = list(self.archive.keys())
        key  = keys[int(rng.integers(0, len(keys)))]
        return self.archive[key][0]

    # ────────────────────────────────────────────────────────────────
    # CMA-ES emitter
    # ────────────────────────────────────────────────────────────────

    def _create_emitter(
        self,
        x0:   np.ndarray,
        seed: int,
    ) -> cma.CMAEvolutionStrategy:
        """
        Bir x0 noktası etrafında yeni CMA-ES emitter başlat.

        Durma kriterleri GEVŞETİLDİ — Fontaine 2020'de emitter sadece
        gerçekten durgunlaştığında restart eder. pycma'nın varsayılan
        tolfun/tolx kriterleri çok agresif, erken restart'a yol açıyor.
        Bu yüzden onları gevşetiyoruz; restart kararını biz veriyoruz
        (arşive katkı kalmadığında).
        """
        opts = {
            "bounds":      [0.0, 1.0],
            "seed":        seed,
            "verbose":     self.verbose,
            "verb_log":    0,
            "verb_disp":   0,
            # Durma kriterlerini gevşet — biz kendi restart mantığımızı kullanacağız
            "tolfun":      1e-12,   # fonksiyon değeri toleransı (çok küçük = geç dur)
            "tolfunhist":  1e-12,
            "tolx":        1e-12,   # x toleransı (çok küçük = geç dur)
            "tolstagnation": 999999,  # durgunluk kontrolünü kapat
            "maxiter":     999999,  # iterasyon limitini kaldır — bütçe kontrol eder
        }
        if self.emitter_popsize is not None:
            opts["popsize"] = self.emitter_popsize

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cma.CMAEvolutionStrategy(
                x0.tolist(),
                self.sigma0,
                opts,
            )

    # ────────────────────────────────────────────────────────────────
    # Ana akış
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        evaluator: Evaluator,
        rng:       np.random.Generator,
    ) -> None:
        # FAZ 1: Başlangıç — n_init rastgele birey
        # CMA-ME'nin de arşivde başlangıç çözümlerine ihtiyacı var
        for _ in range(self.n_init):
            if evaluator.exhausted():
                return
            x = rng.uniform(0.0, 1.0, self.n_params).astype(np.float32)
            try:
                f = evaluator(x)
            except BudgetExhausted:
                return
            self._try_insert(x, f)

        if not self.archive:
            return

        # FAZ 2: CMA-ES emitter döngüsü
        # Restart kriteri: emitter ardışık PATIENCE nesil boyunca arşive
        # hiç katkı yapmazsa (improvement=0) restart edilir.
        PATIENCE = 15   # ardışık katkısız nesil sayısı

        while not evaluator.exhausted():
            # Yeni emitter başlat — arşivden rastgele bir elite ortalama olarak
            x0 = self._select_random_elite(rng)
            seed = int(rng.integers(1, 2**31 - 1))

            try:
                es = self._create_emitter(x0, seed)
            except Exception:
                # CMA-ES başlatma hatası (nadir) — yeni elite dene
                continue

            self.n_emitters_used += 1
            stagnation_counter = 0   # ardışık katkısız nesil sayacı

            # Bu emitter, ya bütçe biter ya da PATIENCE nesil katkısız kalana kadar çalışır
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                while not evaluator.exhausted() and not es.stop():
                    # λ aday üret
                    candidates = es.ask()
                    if not candidates:
                        break

                    # Adayları değerlendir VE iyileşme miktarlarını topla
                    improvements: List[float] = []
                    bütçe_tükendi = False
                    generation_improved = False  # bu nesilde katkı oldu mu?

                    for c in candidates:
                        if evaluator.exhausted():
                            bütçe_tükendi = True
                            break
                        x_arr = np.clip(np.asarray(c, dtype=np.float32), 0.0, 1.0)
                        try:
                            f = evaluator(x_arr)
                        except BudgetExhausted:
                            bütçe_tükendi = True
                            break

                        # ÖDÜLLENDİRME: arşivde iyileşme
                        improved, delta = self._try_insert(x_arr, f)
                        if improved:
                            generation_improved = True
                        # CMA-ES MINIMIZE eder; biz iyileşmeyi maksimize ediyoruz
                        improvements.append(-delta if improved else 0.0)

                    if bütçe_tükendi or len(improvements) < len(candidates):
                        break

                    # CMA-ES'i iyileşme sıralamasına göre güncelle
                    es.tell(candidates, improvements)

                    # Durgunluk takibi — bu nesil katkı yaptı mı?
                    if generation_improved:
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

                    # PATIENCE nesil katkısız → restart
                    if stagnation_counter >= PATIENCE:
                        break

            # Emitter durdu — restart
            self.n_restarts += 1

    # ────────────────────────────────────────────────────────────────
    def build_result(self, evaluator, seed, extra=None):
        """RunResult'a CMA-ME özel istatistikleri ekle."""
        archive_info = {
            "n_cells":   len(self.archive),
            "coverage":  len(self.archive) / self.behavior_space.n_cells,
            "qd_score":  sum(fit for _, fit, _ in self.archive.values()),
            "archive":   {                          # cell → fitness
                cell: fit for cell, (_, fit, _) in self.archive.items()
            },
            "n_emitters": self.n_emitters_used,
            "n_restarts": self.n_restarts,
        }
        extra = {**(extra or {}), "map_elites": archive_info}
        return super().build_result(evaluator, seed, extra)