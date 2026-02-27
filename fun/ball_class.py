# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 17:07:16 2026

@author: Marica Magagnini

Ball class
"""

from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class ball:
    features: Tuple[str, ...]                 # ("f1", "f2", ...)
    x: Dict[str, float]                       # {f: center} coordinates
    r: float                                  # raggio
    label: str = ""
    name: str = ""
    
    
    @property
    def dim(self) -> int:
        """Numero di dimensioni J"""
        return len(self.features)