#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


#  @jitclass
class FSketch:
    """
    FSketch(x: np.ndarray, categorical_lengths: np.ndarray, p: int, d: int, random_state=None)
    procedure INITIALIZE
         Choose random mapping ρ : {1, . . . n} → {1, . . . d}
         Choose some prime p
         Choose n random numbers R = r1, . . . , rn with each ri ∈ {0, . . . p − 1}
    end procedure

    procedure CREATESKETCH(x ∈ {0, 1, . . . c}n)
        Create empty sketch φ(x) = 0d
        for i = 1 . . . n do
            j = ρ(i)
            φj (x) = (φj (x) + xi · ri) mod p
        end for
        return φ(x)
    end procedure
    """

    def __init__(
        self,
        x: np.ndarray,
        c_max: np.int64,
        p: np.int64,
        d: np.int64,
        random_state=None,
    ):
        self.n: np.int64 = x.shape[1]
        self.data: np.ndarray = x
        self.d: np.int64 = d
        self.prime: np.int64 = p
        self.sketch: np.ndarray = np.empty(shape=(x.shape[0], self.d), dtype=np.int16)
        self.cmax: np.int64 = c_max
        if random_state:
            self.random_state = random_state
            self.rng = np.random.default_rng(self.random_state)
        else:
            self.random_state = np.random.randint(np.random.randint(255))
            self.rng = np.random.default_rng(self.random_state)
        self.cmap = self.rng.integers(low=0, high=self.d, size=self.n)
        self.rand_coef_ = self.rng.integers(low=0, high=self.prime, size=self.n)

    def create_sketch(self) -> np.ndarray:
        for i in np.arange(self.n):
            j = self.cmap[i]
            try:
                self.sketch[:, j] = self.sketch[:, j] + (
                    self.data[:, i] * self.rand_coef_[i]
                )
                self.sketch[:, j] = self.sketch[:, j] % self.prime
            except IndexError as e:
                print(i, j)
                print(self.cmap)
                return "Failed"
        return self.sketch


if __name__ == "__main__":
    dsaa = np.random.randint(879, size=(54655, 452))
    fs = FSketch(dsaa, 500, 319, 50, random_state=42)
    we = fs.create_sketch()
