"""
runner.py
=========

Deney çerçevesi — çok koşumlu deney yöneticisi.

Görev:
  Her algoritma × her preset × N_SEEDS kombinasyonu için:
    1. Evaluator kur (aynı fitness_fn, farklı seed)
    2. Algoritmayı çalıştır
    3. RunResult'u topla
    4. Sonuçları diske kaydet (JSON)

Mimari kararlar:
  - 200 koşum = 4 algoritma × 5 preset × 10 seed
  - Her koşum tamamen bağımsız → paralel çalışmaya hazır
  - Seed kontrolü deterministik: seed = preset_idx * 100 + run_idx
  - Hata yakalama: tek koşum çökerse diğerleri devam eder
  - İlerleme: tqdm veya basit print (tqdm opsiyonel)

Kullanım:
    runner = ExperimentRunner(budget=10_000, n_seeds=10)
    results = runner.run_all()
    runner.save(results, "results/experiment_01.json")
"""

from __future__ import annotations

from algorithms import RandomSearch, ClassicGA, SelfAdaptiveGA, MAPElites, CMAES, CMAME
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf
import librosa

from algorithms import Evaluator, RunResult
from algorithms.base import Algorithm
from core import (
    EffectChain, MultiResolutionSTFTLoss,
    IDMTLoader, get_preset_vector,
    PRESET_NAMES, compute_behavior, BehaviorSpace,
)
from algorithms import RandomSearch, ClassicGA, MAPElites, CMAES ,CMAME


# ────────────────────────────────────────────────────────────────────
# Sabitler
# ────────────────────────────────────────────────────────────────────

BUDGET      = 10_000   # Gerçek deney bütçesi
N_SEEDS     = 10       # Her kombinasyon için kaç bağımsız koşum
RECORD_EVERY = 200     # Yakınsama geçmişi kayıt aralığı
SR          = 22_050   # Örnekleme hızı


# ────────────────────────────────────────────────────────────────────
# Yardımcı: fitness ve davranış fonksiyonları
# ────────────────────────────────────────────────────────────────────

def build_fitness_fn(
    chain:  EffectChain,
    loss:   MultiResolutionSTFTLoss,
    dry:    np.ndarray,
    target: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """
    Verilen hedef sinyal için fitness closure'ı döndürür.
    cache_target() çağrısı burada yapılır → her preset için bir kez.
    """
    loss.cache_target(target)

    def fitness_fn(x: np.ndarray) -> float:
        wet = chain.process(dry, x)
        return -loss.compute_fast(wet)

    return fitness_fn


def build_behavior_fn(
    chain: EffectChain,
    dry:   np.ndarray,
) -> Callable[[np.ndarray], tuple]:
    """
    MAPElites için davranış fonksiyonu.
    Basit cache ile çift DSP hesabını önler.
    """
    _cache: dict = {}

    def behavior_fn(x: np.ndarray) -> tuple:
        key = x.tobytes()
        if key not in _cache:
            wet = chain.process(dry, x)
            _cache[key] = compute_behavior(wet, SR)
        return _cache[key]

    return behavior_fn


# ────────────────────────────────────────────────────────────────────
# Algoritma fabrikası
# ────────────────────────────────────────────────────────────────────

def make_algorithms(behavior_fn: Callable) -> Dict[str, Algorithm]:
    return {
        "RandomSearch":   RandomSearch(),
        "ClassicGA":      ClassicGA(pop_size=100, mutation_sigma=0.1),
        "SelfAdaptiveGA": SelfAdaptiveGA(pop_size=100, sigma_init=0.1),  # ← YENİ
        "MAPElites":      MAPElites(
                              behavior_fn=behavior_fn,
                              behavior_space=BehaviorSpace(grid_size=20),
                              n_init=200, mutation_sigma=0.08,
                          ),
        "CMAES":          CMAES(sigma0=0.3),
        # runner.py make_algorithms() içine:
        "CMA-ME": CMAME(behavior_fn=behavior_fn,
                        behavior_space=BehaviorSpace(grid_size=20),
                        n_init=200, sigma0=0.15),
    }

# ────────────────────────────────────────────────────────────────────
# Sonuç serileştirme
# ────────────────────────────────────────────────────────────────────

def result_to_dict(r: RunResult) -> dict:
    """RunResult → JSON-serileştirilebilir sözlük."""
    return {
        "algorithm_name": r.algorithm_name,
        "seed":           r.seed,
        "best_fitness":   float(r.best_fitness),
        "best_loss":      float(r.best_loss),
        "n_evals":        r.n_evals,
        "history_evals":  r.history_evals,
        "history_best":   [float(v) for v in r.history_best],
        # best_x: liste olarak sakla (16 float)
        "best_x":         r.best_x.tolist() if r.best_x is not None else [],
        # extra: MAP-Elites arşiv istatistikleri vs.
        "extra":          _serialize_extra(r.extra),
    }


def _serialize_extra(extra: dict) -> dict:
    """extra dict içindeki numpy dizilerini ve tuple key'leri JSON'a çevir (Rekürsif)."""
    out = {}
    for k, v in extra.items():
        # JSON standardı gereği tüm anahtarları kesinlikle string yapıyoruz
        str_k = str(k)

        if isinstance(v, dict):
            # Sözlük içinde sözlük varsa, fonksiyonu tekrar çağır (rekürsiyon)
            out[str_k] = _serialize_extra(v)
        elif isinstance(v, np.ndarray):
            out[str_k] = v.tolist()
        elif isinstance(v, np.floating):
            out[str_k] = float(v)
        elif isinstance(v, np.integer):
            out[str_k] = int(v)
        else:
            out[str_k] = v

    return out

# ────────────────────────────────────────────────────────────────────
# Ana runner sınıfı
# ────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Çok koşumlu deney yöneticisi.

    Toplam koşum sayısı = len(preset_names) × n_algo × n_seeds
    Varsayılan: 5 preset × 4 algo × 10 seed = 200 koşum

    Her koşum:
      - Bağımsız RNG (seed deterministik)
      - Bağımsız Evaluator (bütçe sıfırdan başlar)
      - Hata yakalanır, koşum atlanır (diğerleri etkilenmez)
    """

    def __init__(
        self,
        budget:       int           = BUDGET,
        n_seeds:      int           = N_SEEDS,
        record_every: int           = RECORD_EVERY,
        preset_names: List[str]     = None,
        dataset_root: Optional[str] = None,
        dry_path:     Optional[str] = None,
        dry_duration: float         = 0.7,
        verbose:      bool          = True,
    ):
        self.budget       = budget
        self.n_seeds      = n_seeds
        self.record_every = record_every
        self.preset_names = preset_names or list(PRESET_NAMES)
        self.verbose      = verbose

        # DSP altyapısı
        self.loader = IDMTLoader(dataset_root, SR, verbose=verbose)
        self.chain  = EffectChain(SR)
        self.loss   = MultiResolutionSTFTLoss()

        # Kuru sinyal kaynağı — öncelik sırası:
        #   1. dry_path verildiyse: doğrudan WAV dosyası
        #   2. dataset_root + IDMTLoader'da dosya varsa: rastgele bir IDMT örneği
        #   3. demo modu (yapay sinyal)
        if dry_path is not None:
            self.dry = self._load_dry_from_wav(dry_path, dry_duration)
        else:
            self.dry = self.loader.get_fixed_dry(duration=dry_duration, seed=0)

    # ──────────────────────────────────────────────────────────────

    def _load_dry_from_wav(self, path: str, duration: float = 0.7) -> np.ndarray:
        """
        Tek bir WAV dosyasını kuru sinyal olarak yükle.

        İşlemler:
          1. soundfile ile oku
          2. Stereo ise mono'ya indir
          3. Sample rate ≠ SR ise librosa.resample
          4. İstenen süreye kırp veya pad et
          5. Peak amplitüdü 0.9'a normalize (loss tutarlılığı için)
        """
        audio, file_sr = sf.read(path, dtype="float32")

        # Stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Sample rate eşitle
        if file_sr != SR:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
        audio, _ = librosa.effects.trim(audio, top_db=30)

        # Süreyi ayarla
        n = int(duration * SR)
        if len(audio) >= n:
            audio = audio[:n]
        else:
            audio = np.pad(audio, (0, n - len(audio)))

        # Peak normalize
        mx = float(np.max(np.abs(audio)))
        if mx > 1e-6:
            audio = audio / mx * 0.9

        self._log(
            f"  Kuru sinyal: {path}\n"
            f"    {len(audio)} örnek, {len(audio)/SR:.2f}s @ {SR} Hz"
        )
        return audio.astype(np.float32)

    # ──────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ──────────────────────────────────────────────────────────────

    def run_single(
        self,
        algo:       Algorithm,
        fitness_fn: Callable,
        seed:       int,
        tag:        str = "",
    ) -> Optional[RunResult]:
        """
        Tek bir (algoritma, seed) kombinasyonunu çalıştır.

        Hata olursa None döner ve devam eder.
        """
        try:
            evaluator = Evaluator(
                fitness_fn,
                budget=self.budget,
                record_every=self.record_every,
            )
            rng = np.random.default_rng(seed)
            t0  = time.perf_counter()
            algo.safe_run(evaluator, rng)
            dt  = time.perf_counter() - t0

            result = algo.build_result(evaluator, seed=seed)

            if self.verbose:
                print(
                    f"  {tag:30s} | seed={seed:3d} | "
                    f"loss={result.best_loss:.4f} | "
                    f"evals={result.n_evals:5d} | "
                    f"{dt:.1f}s"
                )
            return result

        except Exception as e:
            print(f"  [HATA] {tag} seed={seed}: {e}")
            if self.verbose:
                traceback.print_exc()
            return None

    # ──────────────────────────────────────────────────────────────

    def run_preset(self, preset_name: str) -> Dict[str, List[RunResult]]:
        """
        Bir preset için tüm algoritma × seed kombinasyonlarını çalıştır.

        Döndürür: {algo_name: [RunResult, ...]}  (n_seeds uzunlukta)
        """
        self._log(f"\n{'='*64}")
        self._log(f"  Preset: {preset_name}")
        self._log(f"{'='*64}")

        # Hedef sinyal — bu preset için bir kez hesapla
        x_star  = get_preset_vector(preset_name, seed=0)
        target  = self.chain.process(self.dry, x_star)
        fitness_fn   = build_fitness_fn(self.chain, self.loss, self.dry, target)
        behavior_fn  = build_behavior_fn(self.chain, self.dry)

        preset_results: Dict[str, List[RunResult]] = {}

        for seed_idx in range(self.n_seeds):
            # Deterministik seed: preset indeksi * 1000 + seed sırası
            preset_idx = self.preset_names.index(preset_name)
            seed = preset_idx * 1000 + seed_idx

            # Her seed için taze algoritmalar (durum sıfırlanır)
            algos = make_algorithms(behavior_fn)

            for algo_name, algo in algos.items():
                tag = f"{preset_name}/{algo_name}"
                result = self.run_single(algo, fitness_fn, seed, tag)
                if result is not None:
                    preset_results.setdefault(algo_name, []).append(result)

        return preset_results

    # ──────────────────────────────────────────────────────────────

    def run_all(self) -> Dict[str, Dict[str, List[RunResult]]]:
        """
        Tüm preset × algoritma × seed kombinasyonlarını çalıştır.

        Döndürür: {preset_name: {algo_name: [RunResult, ...]}}

        Toplam koşum = len(presets) × n_algo × n_seeds
        """
        n_total = len(self.preset_names) * 4 * self.n_seeds
        self._log(f"Deney başlıyor — toplam {n_total} koşum")
        self._log(f"  Budget={self.budget}, N_seeds={self.n_seeds}")

        all_results: Dict[str, Dict[str, List[RunResult]]] = {}
        t_start = time.perf_counter()

        for preset_name in self.preset_names:
            all_results[preset_name] = self.run_preset(preset_name)

        dt_total = time.perf_counter() - t_start
        self._log(f"\nToplam süre: {dt_total/60:.1f} dakika")

        return all_results

    # ──────────────────────────────────────────────────────────────

    def save(
        self,
        all_results: Dict[str, Dict[str, List[RunResult]]],
        path: str = "results/experiment.json",
    ) -> None:
        """Tüm sonuçları JSON olarak kaydet."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        serialized = {}
        for preset, algo_dict in all_results.items():
            serialized[preset] = {}
            for algo, results in algo_dict.items():
                serialized[preset][algo] = [result_to_dict(r) for r in results]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

        self._log(f"Sonuçlar kaydedildi: {path}")

    @staticmethod
    def load(path: str) -> dict:
        """Kaydedilmiş JSON sonuçlarını yükle."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)