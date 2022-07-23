#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def isPrime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(np.sqrt(n) + 1), 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True


def nextPrime(N):
    if N <= 1:
        return 2
    prime = N
    found = False
    while not found:
        prime = prime + 1
        if isPrime(prime):
            found = True
    return prime


def Hamming_distance(a, b):
    "Funtion to calculate hamming distance between array a and b"
    ham = 0
    for i in range(a.shape[1]):
        if a[:, i] != b[:, i]:
            ham += 1
    return ham


def FSketch_Ham_Estimate(a, b, d, p):
    """'Documentation:
    Parameters:
      a: scipy sparse scr array
      b: sipy sparse scr  array

      d: actual dimesion (dimension before reduction)

      p: prime number next to the maximum element in X
    Returns:
      integer values
    """
    k = a.shape[1]
    ham = Hamming_distance(a, b)
    if ham / (k * (1 - 1 / p)) < 1:
        return round(np.log(1 - ham / (k * (1 - 1 / p))) / np.log(1 - 1 / k))
    else:
        return ham


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
        d: np.int64,
        p: np.int64 = None,
        random_state=None,
    ):
        self.n: np.int64 = x.shape[1]
        self.data: np.ndarray = x
        self.d: np.int64 = d
        if p:
            self.prime: np.int64 = p
        else:
            self.prime: np.int64 = nextPrime(c_max)
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
            self.sketch[:, j] = self.sketch[:, j] + (
                self.data[:, i] * self.rand_coef_[i]
            )
            self.sketch[:, j] = self.sketch[:, j] % self.prime
        return self.sketch
