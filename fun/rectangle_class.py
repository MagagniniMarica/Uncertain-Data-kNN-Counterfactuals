# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini 

Class Rectangle
"""
from dataclasses import dataclass
from typing import Dict, Tuple#, Iterable, Mapping
# from itertools import product


@dataclass
class Rettangolo:
    features: Tuple[str, ...]                 # ("f1", "f2", ...)
    x: Dict[str, float]                       # {f: center}
    hs: Dict[str, float]                      # {f: half-size}
    label: str = ""
    name: str = ""

    def __post_init__(self):
        # Features non vuote e senza duplicati
        if len(set(self.features)) != len(self.features):
            raise ValueError("features contiene duplicati")

        # Stesse chiavi (almeno) di features
        missing_x = [f for f in self.features if f not in self.x]
        missing_h = [f for f in self.features if f not in self.hs]
        if missing_x or missing_h:
            raise ValueError(f"Chiavi mancanti: x={missing_x}, hs={missing_h}")

        # (opzionale) vieta chiavi extra
        extra_x = set(self.x) - set(self.features)
        extra_h = set(self.hs) - set(self.features)
        if extra_x or extra_h:
            raise ValueError(f"Chiavi extra non ammesse: x={sorted(extra_x)}, hs={sorted(extra_h)}")

        # half-size non negativo
        bad = [f for f in self.features if self.hs[f] < 0]
        if bad:
            raise ValueError(f"Half-size negativo per: {bad}")

    @property
    def dim(self) -> int:
        return len(self.features)

    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        """
        Restituisce {f1: (min_f1, max_f1), ..., fJ: (min_fJ, max_fJ)} nell'ordine features
        """
        return {f : (self.x[f] - self.hs[f], self.x[f] + self.hs[f]) for f in self.features}

    # def corners(self) -> Tuple[Tuple[float, ...], ...]:
    #     """
    #     Restituisce i 2^J vertici come tuple di float nell'ordine features
    #     """
    #     b = self.bounds()
    #     return tuple(
    #         tuple(interval[i] for interval, i in zip(b, choice))
    #         for choice in product([0, 1], repeat=self.dim)
    #     )

    # def p_in_R(self, point: Mapping[str, float] | Iterable[float], *, strict: bool = False) -> bool:
    #     """
    #     point può essere:
    #       - mapping: {"f1": v1, ...}
    #       - iterabile: (v1, v2, ...) nell'ordine self.features
    #     """
    #     if isinstance(point, Mapping):
    #         missing = [f for f in self.features if f not in point]
    #         if missing:
    #             raise ValueError(f"Point (dict) senza chiavi: {missing}")
    #         getter = lambda f: float(point[f])
    #     else:
    #         vals = tuple(point)
    #         if len(vals) != self.dim:
    #             raise ValueError("Il punto ha dimensione errata")
    #         getter = lambda f, _it=iter(vals): next(_it)  # consumiamo in ordine features

    #     if strict:
    #         return all(abs(getter(f) - self.x[f]) < self.hs[f] for f in self.features)
    #     else:
    #         return all(abs(getter(f) - self.x[f]) <= self.hs[f] for f in self.features)
