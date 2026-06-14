"""
random_search.py
================

Random Search — EC karşılaştırmalarında zorunlu alt sınır (baseline).

Hafızasız stokastik arama:
  - Her iterasyonda parametre uzayından bağımsız bir örnek çekilir
  - Önceki örneklerden hiçbir bilgi taşınmaz
  - Ne seçim, ne çaprazlama, ne mutasyon

Neden burada? "Eğer önerdiğin algoritma random search'ten daha iyi
değilse, hiçbir şey yapmıyor demektir." EC literatüründe altın kural.
"""

import numpy as np

from core import N_PARAMS
from .base import Algorithm, Evaluator


class RandomSearch(Algorithm):
    """
    Uniform örnekleme: x ~ U([0,1]^16) her iterasyonda.

    Yakınsama garantisi sonsuz iterasyonda mevcuttur ama pratikte
    16 boyutlu sürekli uzayda çok yavaştır. Diğer EC algoritmaları
    bu yöntemi anlamlı şekilde yenmek zorunda.
    """

    name = "RandomSearch"

    def __init__(self):
        super().__init__()

    def run(
        self,
        evaluator: Evaluator,
        rng: np.random.Generator,
    ) -> None:
        while not evaluator.exhausted():
            x = rng.uniform(0.0, 1.0, self.n_params).astype(np.float32)
            evaluator(x)