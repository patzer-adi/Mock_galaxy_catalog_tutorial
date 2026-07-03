# 🌌 MockTutorial — A Complete Beginner's Guide

> **Your goal:** Understand this entire codebase from scratch, then rebuild it yourself.
>
> **Who this is for:** Someone comfortable with Python, but with no background in cosmology, physics, or advanced mathematics.
>
> This document is structured like a **textbook**. We build from zero — starting with the biggest picture, gradually zooming in on every function, equation, and physical concept.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The Big Picture — How Cosmologists Study the Universe](#2-the-big-picture)
3. [Background Concepts You Need First](#3-background-concepts)
   - 3.1 [What is a Galaxy?](#31-what-is-a-galaxy)
   - 3.2 [Dark Matter and Halos](#32-dark-matter-and-halos)
   - 3.3 [The Halo Model](#33-the-halo-model)
   - 3.4 [HOD — Halo Occupation Distribution](#34-hod--halo-occupation-distribution)
   - 3.5 [Mock Catalogs](#35-mock-catalogs)
   - 3.6 [The Two-Point Correlation Function ξ(r)](#36-the-two-point-correlation-function)
   - 3.7 [The Projected Correlation Function wₚ(rₚ)](#37-the-projected-correlation-function)
4. [Mathematics You Need](#4-mathematics-you-need)
   - 4.1 [Probability and Random Variables](#41-probability-and-random-variables)
   - 4.2 [The Error Function (erf)](#42-the-error-function-erf)
   - 4.3 [The Poisson Distribution](#43-the-poisson-distribution)
   - 4.4 [Logarithms and Powers of 10](#44-logarithms-and-powers-of-10)
   - 4.5 [Inverse Transform Sampling](#45-inverse-transform-sampling)
   - 4.6 [NFW Profile](#46-nfw-profile)
   - 4.7 [Pair Counting and KD-Trees](#47-pair-counting-and-kd-trees)
5. [File Overview](#5-file-overview)
   - 5.1 [MockTutorial.ipynb](#51-mocktutorialipynb)
   - 5.2 [out_1.parents (the halo catalog)](#52-out_1parents)
6. [Code Walkthrough — Class by Class, Function by Function](#6-code-walkthrough)
   - 6.1 [Imports](#61-imports)
   - 6.2 [Class: HODFits](#62-class-hodfits)
   - 6.3 [Class: HODFunctions (extends HODFits)](#63-class-hodfunctions)
   - 6.4 [Class: Mocker (extends HODFunctions)](#64-class-mocker)
   - 6.5 [Class: TwoPointCorrelationFunctionPeriodic](#65-class-twopointcorrelationfunctionperiodic)
7. [The Complete Pipeline — Data Flow Diagram](#7-the-complete-pipeline)
8. [Running the Code — What the Output Means](#8-running-the-code)
9. [How to Rebuild This From Scratch](#9-how-to-rebuild-this-from-scratch)
10. [Glossary of Symbols](#10-glossary-of-symbols)
11. [Chapter Summaries and Exercises](#11-chapter-summaries-and-exercises)

---

## 1. What is this project?

This project is a **mock galaxy catalog generator**. In plain language:

- Scientists have a simulation of the universe. In it, invisible clumps of matter called **dark matter halos** have formed.
- We want to ask: *"If galaxies live inside these halos, where would those galaxies be, and how bright would they be?"*
- We use a statistical rule called the **HOD (Halo Occupation Distribution)** to decide.
- The result is a fake but realistic list of galaxies — their positions and luminosities (brightnesses).
- We then measure how **clustered** these fake galaxies are using the **two-point correlation function**.
- We compare our result to real galaxy survey data to check if our model is correct.

**Why do we do this?** Because we cannot directly observe dark matter. But we CAN observe galaxies. If our model correctly predicts how real galaxies cluster, we have learned something about dark matter.

---

## 2. The Big Picture

Imagine trying to understand how a city is organized. You can't go to every house — there are too many. Instead, you take an aerial photograph and count how many houses are within various distances of each other. That pattern of clustering tells you something about how people organize themselves.

Cosmologists do the same thing with galaxies:

1. **Run a simulation** — a supercomputer simulates the entire universe from the Big Bang forward. Dark matter collapses into **halos** (clumps).
2. **Populate halos with galaxies** — use a statistical rule (HOD) to decide which halos contain galaxies, and how many.
3. **Measure clustering** — count how often pairs of galaxies appear at various separations. This is the correlation function.
4. **Compare to observations** — does our fake (mock) catalog cluster like real galaxies do?

This notebook implements steps 2 and 3.

---

## 3. Background Concepts

### 3.1 What is a Galaxy?

A galaxy is a gravitationally bound system of stars, gas, dust, and dark matter — ranging from a few thousand to hundreds of billions of stars.

In this code, a galaxy has:
- **Position** (x, y, z) — where it is in space (in Mpc/h)
- **Luminosity** (Mr) — how bright it is, measured in the r-band (red light). Mr is given in **magnitudes**: a negative scale where more negative = brighter. Mr = -20.5 is the threshold here.

### 3.2 Dark Matter and Halos

Dark matter is matter we cannot see. It doesn't emit or absorb light. But it has mass and gravity, and makes up ~85% of all matter in the universe.

**Dark matter halos** form when gravity pulls dark matter together into dense clumps. These halos are where galaxies form. Think of halos as the "nests" where galaxies live.

Key halo properties used in this code:
| Property | Symbol | Description |
|---|---|---|
| Halo mass | m200b | Mass enclosed within a region 200× the background density (solar masses / h) |
| Position | x, y, z | Center of the halo in Mpc/h |
| Virial radius | rvir | Approximate "size" of the halo |
| Scale radius | rs | Inner characteristic radius of the halo's density profile |
| Concentration | c | Ratio rvir/rs — how concentrated the halo is |
| Particle ID | pid | -1 means it's a "parent" halo, not a subhalo |

### 3.3 The Halo Model

The Halo Model is a framework that says:
> All matter (and all galaxies) lives inside dark matter halos.

So to understand the universe's structure, understand halos.

Halos have a characteristic density profile — the **NFW profile** (named after Navarro, Frenk & White 1996):

```
ρ(r) = ρ₀ / [ (r/rs)(1 + r/rs)² ]
```

Where:
- `ρ(r)` = density at radius r from the halo center
- `ρ₀` = central density (a normalization constant)
- `rs` = scale radius
- `r` = distance from the center

**Intuition:** The density is very high at the center, drops off in a predictable way, and falls steeply at large radii.

The NFW profile predicts where satellites (secondary galaxies) are located inside a halo.

### 3.4 HOD — Halo Occupation Distribution

The HOD is a statistical recipe that answers: **how many galaxies does a halo of mass M contain?**

It says:
- Each halo has **at most one central galaxy** (living at the center of the halo)
- Each halo can have **zero or more satellite galaxies** (orbiting inside the halo)

#### Central Galaxy Occupation
The probability that a halo of mass M hosts a central galaxy brighter than luminosity threshold L is:

```
⟨Ncen⟩(M) = (1/2) × [1 + erf( (log₁₀(M) - log₁₀(Mmin)) / σ_lgM )]
```

This is a smooth step function. It means:
- Very low-mass halos: almost never have a central galaxy (probability ≈ 0)
- Very high-mass halos: almost always have one (probability ≈ 1)
- Mmin = the "half-way" mass where 50% of halos are occupied
- σ_lgM = how sharp or smooth the transition is

**Where does erf come from?** See Section 4.2.

#### Satellite Galaxy Occupation
The mean number of satellites in a halo of mass M is:

```
⟨Nsat⟩(M) = ((M - M₀) / M₁)^α   [if M > M₀, else 0]
```

Where:
- M₀ = minimum halo mass for any satellites
- M₁ = characteristic mass where ⟨Nsat⟩ ≈ 1
- α = power law slope

**Intuition:** Bigger halos have more satellites. It's a power law — double the mass, get roughly 2^α satellites.

The actual number of satellites in any halo is drawn from a **Poisson distribution** with mean ⟨Nsat⟩.

### 3.5 Mock Catalogs

A **mock catalog** is a simulated dataset that looks like a real observational survey. We create it by:
1. Taking a dark matter simulation (the halo catalog)
2. Populating it with fake galaxies using the HOD
3. Assigning properties (luminosity, position)

The advantage: we know the truth (we set up the simulation), so we can test our analysis methods.

### 3.6 The Two-Point Correlation Function

The **two-point correlation function** ξ(r) answers:

> How much more likely is it to find a galaxy at a distance r from another galaxy, compared to a random (uniform) distribution?

If galaxies were completely random, ξ(r) = 0 everywhere.

In reality:
- At small scales (< 1 Mpc/h): ξ(r) >> 1 — galaxies cluster strongly
- At large scales (> 100 Mpc/h): ξ(r) ≈ 0 — galaxies look random

**Mathematical definition:**

The excess probability of finding a galaxy in volume dV at distance r from another galaxy is:
```
dP = n̄ × [1 + ξ(r)] × dV
```

Where n̄ is the mean number density of galaxies.

**How we measure it** (Peebles-Hauser estimator):
```
ξ̂(r) = DD/RR - 1
```

Where:
- DD = normalized count of data-data galaxy pairs at separation r
- RR = expected pairs for a random distribution

### 3.7 The Projected Correlation Function

Real surveys suffer from **redshift space distortions** — galaxies appear shifted in the line-of-sight direction due to their peculiar velocities. To avoid this, we project along the line of sight:

```
wₚ(rₚ) = 2 ∫₀^πmax ξ(√(rₚ² + π²)) dπ
```

Where:
- rₚ = separation perpendicular to the line of sight
- π = separation along the line of sight
- πmax = maximum line-of-sight separation to integrate over

This is what `tpcf` calculates with `proj=1` in the code.

---

## 4. Mathematics You Need

### 4.1 Probability and Random Variables

A **random variable** is a quantity that can take different values randomly. For example, rolling a die gives values 1-6.

A **probability distribution** tells you how likely each value is.

**Uniform distribution**: All values equally likely. In Python: `np.random.rand()` gives uniform values between 0 and 1.

**Cumulative Distribution Function (CDF)**: The probability that the variable is less than or equal to x.

For a distribution with CDF F(x), if you want to sample from it:
1. Generate a uniform random number u between 0 and 1
2. Find x such that F(x) = u
This is called **inverse transform sampling** (see Section 4.5).

### 4.2 The Error Function (erf)

The error function appears everywhere in the HOD. Here's where it comes from.

**Start with the Normal (Gaussian) distribution:**

A Gaussian is a bell-shaped distribution:
```
P(x) = (1/√(2π)) × exp(-x²/2)
```

It describes quantities that have many small random effects added together (Central Limit Theorem).

**The CDF of a Gaussian** (probability that x < some value X) is:

```
Φ(X) = (1/2)[1 + erf(X/√2)]
```

where erf is the error function:
```
erf(z) = (2/√π) ∫₀^z exp(-t²) dt
```

**Key properties of erf:**
- erf(0) = 0
- erf(+∞) = +1
- erf(-∞) = -1
- erf(-z) = -erf(z)

**So (1/2)[1 + erf(z)]** gives values between 0 and 1, making it ideal as a probability.

When z is very negative → result ≈ 0
When z = 0 → result = 0.5
When z is very positive → result ≈ 1

This is exactly the shape we want for the central galaxy probability.

In Python: `scipy.special.erf(z)`

### 4.3 The Poisson Distribution

The Poisson distribution describes the number of events in a fixed interval when events occur randomly at a known average rate.

If the mean is λ, the probability of getting exactly k events is:
```
P(k) = λᵏ × e^(-λ) / k!
```

**In this code:** If the mean number of satellites in a halo is ⟨Nsat⟩, the actual count drawn for each halo is a Poisson random variable with mean ⟨Nsat⟩.

**Example:** If ⟨Nsat⟩ = 2.3, then:
- P(0) = e^(-2.3) ≈ 0.10
- P(1) = 2.3 × e^(-2.3) ≈ 0.23
- P(2) = 2.3² × e^(-2.3) / 2 ≈ 0.27
- P(3) = ... etc.

In Python: `np.random.poisson(mean_Nsat)`

### 4.4 Logarithms and Powers of 10

**log₁₀(x)** = the power to which 10 must be raised to get x.
- log₁₀(100) = 2  (because 10² = 100)
- log₁₀(10⁴²) = 42

Halo masses are huge (10¹¹ to 10¹⁵ solar masses), so scientists work with log₁₀(M). In the code, `lgm` means log₁₀(M).

**Converting:** `mhalo = 10**lgm`

### 4.5 Inverse Transform Sampling

This is how we assign random luminosities to galaxies.

**The problem:** We know the probability distribution of luminosities (from the HOD). We want to generate random luminosities that follow this distribution.

**The solution:**
1. Compute the CDF: P(>L|M) = probability of luminosity > L given halo mass M
2. Generate a uniform random number u between 0 and 1
3. Find L such that P(>L|M) = u

**Why does this work?** The CDF maps any probability distribution to a uniform [0,1] distribution. Inverting it maps uniform random numbers back to the original distribution.

In the code, this is done numerically via interpolation:
```python
Mr_cen = np.interp(u, PcenL, Mrvals)
```

### 4.6 NFW Profile

**The Navarro-Frenk-White (NFW) density profile:**
```
ρ(r) = ρ_crit × δ_c / [(r/rs) × (1 + r/rs)²]
```

The **mass enclosed within radius r** is:
```
M(r) ∝ ln(1 + r/rs) - (r/rs)/(1 + r/rs)
```

This function appears directly in `Mencl_nfw(x)` where x = r/rs.

To sample satellite positions from this profile:
1. Build the CDF: Px = M(x)/M(xmax)
2. Sample radius r using inverse transform (see Section 4.5)
3. Sample direction angles isotropically (uniformly on the sphere)

**Sampling directions uniformly on a sphere:**
```python
phi = 2*pi * rand()                     # azimuthal angle [0, 2π]
cos_theta = 2*rand() - 1               # polar angle: uniform in cos(θ)
sin_theta = sqrt(1 - cos_theta**2)
```

Note: We sample uniformly in cos(θ), NOT in θ. This is because the surface area element on a sphere is sin(θ)dθdφ = -d(cos θ)dφ — equal areas correspond to equal intervals in cos(θ).

Then convert to Cartesian:
```
x = r × sin(θ) × sin(φ)
y = r × sin(θ) × cos(φ)
z = r × cos(θ)
```

### 4.7 Pair Counting and KD-Trees

**The problem:** We have N galaxies. We want to count all pairs with separation between r₁ and r₂.

**Naïve approach:** Compare every pair — O(N²) operations. For N = 80,000 galaxies, that's ~3 billion comparisons. Too slow.

**Better approach: KD-Trees**

A KD-Tree (K-Dimensional Tree) is a data structure that organizes points in space hierarchically. It allows finding all neighbors within a distance r in much less than O(N²) time — typically O(N log N).

`scipy.spatial.cKDTree` — the `c` means it's implemented in C for speed.

```python
tree = cKDTree(positions, boxsize=Lbox)
counts = tree.count_neighbors(other_tree, radii)
```

The `boxsize` parameter enables **periodic boundary conditions** — if a pair crosses the edge of the box, it wraps around. This simulates an infinite universe.

---

## 5. File Overview

### 5.1 MockTutorial.ipynb

**What it is:** A Jupyter notebook (interactive Python document) that contains all the code.

**Purpose:** Teaches and demonstrates the full pipeline of creating a mock galaxy catalog and measuring its clustering.

**Position in pipeline:** The entire pipeline lives here. It defines classes and runs them step-by-step.

**Inputs:** The file `halos/out_1.parents` (halo catalog)

**Outputs:**
- `centrals` — structured array of central galaxies with positions and luminosities
- `satellites` — structured array of satellite galaxies
- Plots of luminosity function, mass function, and correlation functions

**Dependencies:** numpy, scipy, pandas, matplotlib, gc

---

### 5.2 out_1.parents

**What it is:** A text file containing data about dark matter halos from a cosmological N-body simulation.

**Format:** Space-delimited text, with rows for each halo and columns for properties.

**Key columns:**
| Column | Type | Description |
|---|---|---|
| ID | int | Unique halo identifier |
| pid | int | Parent halo ID. -1 means it's a "host" halo (not inside another halo) |
| m200b | float | Halo mass in M☉/h within overdensity 200× background |
| x, y, z | float | Position in Mpc/h |
| vx, vy, vz | float | Velocity in km/s |
| rvir | float | Virial radius |
| rs | float | Scale radius (comoving kpc/h) |
| mvir, m200c, mCustom, ... | float | Mass measured with different overdensity definitions |

**Simulation parameters:**
- Box size: 300 Mpc/h (a cube 300 Mpc/h on each side)
- Omega_m = 0.276 (matter density fraction)
- Redshift: z = 0 (today)
- Total halos: 2,941,105
- After filtering (mass cut + parent only): 458,324 halos

---

## 6. Code Walkthrough

### 6.1 Imports

```python
import numpy as np
import scipy.special as sysp
import scipy.spatial as syspat
import sys
import pandas as pd
import gc
import matplotlib.pyplot as plt
```

| Library | Purpose |
|---|---|
| numpy | Fast numerical arrays and math |
| scipy.special | Special functions including erf |
| scipy.spatial | Spatial data structures, especially cKDTree for pair counting |
| pandas | Reading tabular data from files |
| gc | Garbage collector — manually freeing memory |
| matplotlib | Plotting |

**Why gc (garbage collection)?**
This code handles millions of objects. Python normally manages memory automatically, but for large arrays, explicitly calling `gc.collect()` forces Python to free unused memory immediately, preventing crashes on machines with limited RAM.

---

### 6.2 Class: HODFits

**File:** MockTutorial.ipynb (Cell 2)

**Purpose:** Stores the HOD parameters as polynomial/special fits, based on the paper by Paul, Pahwa, Paranjape (2019).

**Why this class exists:** The HOD parameters (Mmin, σ_lgM, M1, M0, α) vary with the luminosity threshold Mr. Rather than look them up in a table, the authors fit smooth functions of Mr and store those fit coefficients here.

**Inputs to the class:** None (parameters are hardcoded from the paper)

**Outputs:** Smooth functions for the 5 HOD parameters

---

#### `__init__(self)`

**Purpose:** Stores the best-fit polynomial coefficients from the paper.

```python
self.bf = {'lgMmin': np.array([12.33, -0.85, 0.19]),
           'siglgM': np.array([0.44, -0.16, 0.3]),
           'lgM1':   np.array([13.52, -0.72, 0.16]),
           'lgM0':   np.array([12.24, -0.54, 0.0]),
           'alpha':  np.array([1.16, -0.20, 0.10])}
```

Each parameter has 3 coefficients [a0, a1, a2]. They are fit as either:
- Polynomial: `param = a0 + a1*x + a2*x²`   (where x = Mr + 20.5)
- Error function: `param = a0 + a1*erf(x/a2)` (only for σ_lgM)

**Why x = Mr + 20.5?** It shifts the coordinate so that Mr = -20.5 maps to x = 0. This centers the fit around the threshold of interest and makes the polynomial coefficients more interpretable.

---

#### `hod_fit(self, param, Mr)`

**Purpose:** Returns the value of a single HOD parameter at a given luminosity threshold Mr.

**Inputs:**
- `param` — string: one of 'lgMmin', 'siglgM', 'lgM1', 'lgM0', 'alpha'
- `Mr` — float or array: luminosity threshold in magnitudes (negative: -23.5 to -20.5)

**Output:** Float or array — the value of that HOD parameter

**Algorithm:**
```python
x = Mr + 20.5
if param != 'siglgM':
    out = a0 + a1*x + a2*x**2   # quadratic polynomial
else:
    out = a0 + a1*erf(x/a2)     # error function fit
```

**Why a different formula for σ_lgM?** σ_lgM is constrained to be positive and can't be well described by a polynomial over the full Mr range. The erf functional form naturally captures its smooth bounded behavior.

---

#### `hod_fit_deriv(self, param, Mr)`

**Purpose:** Returns the derivative d(param)/d(Mr) — how fast the HOD parameter changes as the luminosity threshold changes.

**Why derivatives?** Some downstream calculations (e.g., assigning luminosities via the luminosity function dN/dMr) require knowing how rapidly the HOD changes with Mr.

**Algorithm:**
```python
if param != 'siglgM':
    d/dx[a0 + a1*x + a2*x**2] = a1 + 2*a2*x
else:
    d/dx[a0 + a1*erf(x/a2)] = (a1/a2)*(2/√π)*exp(-(x/a2)²)
```

The second formula uses the standard rule: d/dz[erf(z)] = (2/√π)exp(-z²), applied with the chain rule.

---

### 6.3 Class: HODFunctions (extends HODFits)

**File:** MockTutorial.ipynb (Cell 2)

**Purpose:** Implements the actual HOD functions — the probability of hosting a central galaxy, and the mean number of satellites.

**Inherits from:** HODFits — gains access to all fit coefficients and `hod_fit()`.

**Pipeline position:** These functions are the mathematical heart of the HOD. Everything in `Mocker` calls these.

---

#### `erf_arg(self, Mr, lgm)`

**Purpose:** Computes the argument of the error function: (log₁₀(M) - log₁₀(Mmin)) / σ_lgM

This is a "convenience function" — it calculates something needed repeatedly by other methods.

**Physics:** The argument measures how many "sigmas" the halo mass M is above the minimum halo mass Mmin (in log space).

**Why log space?** Because halo masses span many orders of magnitude (10¹⁰ to 10¹⁵ M☉), it's natural to work in log₁₀(M). The width of the HOD step function σ_lgM is a width in log space.

**Handling arrays:**
```python
# If both Mr and lgm are arrays, the result has shape (lgm.size, Mr.size)
lgm_arr = np.outer(lgm, np.ones(Mr.size))
```

`np.outer(a, b)` creates a 2D array where element [i,j] = a[i]*b[j]. This is needed when you want to evaluate the HOD for every combination of halo mass and luminosity threshold simultaneously — a vectorized "broadcasting" approach.

---

#### `fcen_thresh(self, Mr, lgm)`

**Purpose:** Probability that a halo of mass 10^lgm has a central galaxy brighter than threshold Mr.

**Formula:**
```
⟨Ncen⟩(>L|M) = (1/2)[1 + erf((log₁₀(M) - log₁₀(Mmin)) / σ_lgM)]
```

**Physics:** This is a smooth step function (S-shaped curve). Halos much more massive than Mmin → probability ≈ 1. Halos much less massive → probability ≈ 0.

**Input:**
- Mr — luminosity threshold (magnitudes)
- lgm — log₁₀(M/M☉) of the halo

**Output:** Probability ∈ [0, 1]

---

#### `Nsat_thresh(self, Mr, lgm)`

**Purpose:** Mean number of satellites brighter than threshold Mr in a halo of mass 10^lgm.

**Formula:**
```
⟨Nsat⟩(>L|M) = ((M - M₀) / M₁)^α   [if M > M₀, else 0]
```

**Physics:** Power law behavior — more massive halos have more satellites. M₀ is a minimum threshold mass (halos below M₀ have no satellites). M₁ and α control the normalization and slope.

**Code detail — handling the M < M₀ case:**
```python
Dm = (mhalo_arr - M0) / M1
Dm[Dm < 0.0] = 0.0    # Set negative values to zero
out = Dm**alpha
```

Setting the negative values to zero before raising to power α avoids numerical issues (e.g., raising a negative number to a fractional power gives NaN).

---

### 6.4 Class: Mocker (extends HODFunctions)

**File:** MockTutorial.ipynb (Cell 3)

**Purpose:** The main class that coordinates the entire pipeline — reading data, applying the HOD, assigning properties, and outputting the mock catalog.

**Inherits from:** HODFunctions → HODFits

**Think of it as:** The "director" that calls every other function in the right order.

---

#### `__init__(self)`

**Purpose:** Sets all the simulation parameters and initializes the random number generator.

**Key parameters explained:**

```python
self.halo_cat = 'halos/out_1.parents'   # Path to halo catalog file
self.redshift = 0.0                       # We're looking at the universe "today"
self.Om = 0.276                           # Ω_m: fraction of universe's energy density in matter
self.Lbox = 300.0                         # Box size in Mpc/h
self.massdef = 'm200b'                    # Which mass column to use
self.mmin = 2e11                          # Minimum halo mass to consider [M☉/h]
self.Mrmax = -20.5                        # Luminosity threshold [magnitudes]
self.Dvir = 200*self.Om                   # Overdensity for virial definition (relative to background)
self.rhoc = 2.7754e11                     # Critical density [(M☉/h)/(Mpc/h)³]
self.rhoc_z = self.rhoc * self.EHub(0.0)**2  # Critical density at z=0
self.rng = np.random.RandomState(seed=42) # Random number generator with fixed seed
```

**What is Ω_m (Om)?**
The universe's total energy density includes:
- Matter (dark + normal): ~27%
- Dark energy: ~68%
- Radiation: ~5%

Ω_m = 0.276 means matter is 27.6% of the total.

**What is the critical density (ρ_crit)?**
The density the universe would need to be "flat" (Euclidean geometry). Measured in (M☉/h) per (Mpc/h)³.

**What is m200b?**
The halo mass within the radius where the density is 200× the **background** density (not critical density). "b" stands for "background."

**Fixed random seed (`seed=42`):**
Setting a fixed seed makes the code reproducible. Everyone running the code with the same input gets the same "random" mock catalog. This is essential for science.

**Luminosity grid:**
```python
dMr = 0.001
nMr = int((23.5 + self.Mrmax) / dMr)  # = 3000 steps
self.Mrvals = np.linspace(-23.5, self.Mrmax, nMr)  # from -23.5 to -20.5
```

This creates a fine grid of magnitudes for numerical integration when assigning luminosities via inverse transform sampling.

---

#### `EHub(self, z)`

**Purpose:** Computes the dimensionless Hubble parameter E(z) = H(z)/H₀.

**Formula:**
```
E(z) = √(Ωm × (1+z)³ + (1 - Ωm))
```

**Physics:** The Hubble parameter H(z) describes how fast the universe is expanding at redshift z.

- The term (1+z)³ is for matter (its density dilutes as volume grows as (1+z)⁻³)
- The term (1-Ωm) = Ω_Λ is for dark energy (constant density — Einstein's cosmological constant)

At z=0: E(0) = √(0.276 + 0.724) = √1.0 = 1.0 exactly.

Used to calculate `rhoc_z` — the critical density at any redshift.

---

#### `read_this(self)`

**Purpose:** Reads the halo catalog file into memory.

**Implementation:**
```python
halos = pd.read_csv(self.halo_cat, dtype=self.halodatatype, names=...,
                    comment='#', delim_whitespace=True, header=None).to_records()
```

- `pd.read_csv`: pandas reads the file as a table
- `comment='#'`: skip lines starting with #
- `delim_whitespace=True`: columns are separated by spaces
- `dtype=self.halodatatype`: each column is read with the correct type
- `.to_records()`: converts to numpy structured array (faster access)

**Why a structured array?** It allows accessing columns by name: `halos['m200b']` instead of `halos[:, 5]`. More readable and less error-prone.

---

#### `prep_halo_data(self)`

**Purpose:** Reads and **filters** the halo catalog. Returns only halos that are "parent" halos (not subhalos) above a mass threshold.

**Algorithm:**
```python
halos = self.read_this()
cond_clean = (halos[self.massdef] >= self.mmin)  # mass cut
cond_clean = cond_clean & (halos['pid'] == -1)    # parent halos only
halos = halos[cond_clean]
```

**Why filter on `pid == -1`?** In a simulation, smaller halos can be embedded inside larger ones. The `pid` (parent ID) tells you if a halo is a "subhalo" (inside another). `pid == -1` means it's a top-level parent halo. The HOD is designed for parent halos — you don't want to double-count by assigning galaxies to both parent and subhalo.

**Why the mass cut (mmin = 2×10¹¹ M☉/h)?** Halos below this mass are too small to host galaxies brighter than Mr = -20.5. The HOD gives essentially zero probability for them, so we exclude them to save time and memory.

**Result:** Of 2,941,105 total objects, 458,324 pass the filter.

---

#### `occupy_halos(self, halo_mass)`

**Purpose:** Decides which halos get a central galaxy using the HOD probability.

**Algorithm — Monte Carlo acceptance:**
```python
fcen_min = self.fcen_thresh(self.Mrmax, np.log10(halo_mass))  # occupation probability
u = self.rng.rand(halo_mass.size)                               # uniform random numbers
occupy[u > fcen_min] = False                                    # reject if u > probability
```

**What is Monte Carlo acceptance?**
If a halo has probability p of being occupied:
- Generate a uniform random number u ∈ [0, 1]
- If u ≤ p → occupied (accept)
- If u > p → not occupied (reject)

On average, p fraction of halos will be occupied. This is the standard Monte Carlo method.

**Result:** 67,102 of 458,324 halos are occupied (~14.6%).

---

#### `assign_centrals(self, halos)`

**Purpose:** For each occupied halo, creates a central galaxy record with properties.

**Structured array created:**
```python
centrals = np.zeros(halos.size, dtype=[
    ('haloid', 'int'),
    ('lgm', 'double'),      # log₁₀(halo mass)
    ('rvir', 'double'),     # virial radius [comoving Mpc/h]
    ('con', 'double'),      # concentration c = rvir/rs
    ('x', 'double'), ('y', 'double'), ('z', 'double'),
    ('Mr', 'double'),       # luminosity [magnitudes]
    ('Nsat', 'int')         # number of satellites
])
```

**Virial radius calculation:**
```python
rvir = (3 * halos[massdef] / (4 * pi * Dvir * rhoc_z))**(1/3.)
```

This comes from the definition: M = (4/3)π r³ × Δ_vir × ρ_crit
Solving for r: r = (3M / (4π Δ_vir ρ_crit))^(1/3)

**Comoving vs physical:** The simulation stores positions in comoving coordinates. The virial radius is stored in comoving Mpc/h: `rvir_comoving = rvir_physical × (1+z)`. At z=0, they're the same.

**Concentration:**
```python
con = rvir / (rs * 1e-3 / (1+z))   # rs is in comoving kpc/h, convert to Mpc/h
```

Concentration c = rvir / rs measures how "concentrated" a halo is. Higher c → more centrally concentrated halo.

**Then calls:** `assign_central_luminosity()` and `get_Poisson()`.

---

#### `assign_central_luminosity(self, lgm_halos)`

**Purpose:** Assigns a luminosity (Mr value) to each central galaxy.

**Physics:** The luminosity of the central galaxy is NOT fixed — it's drawn from the conditional luminosity function (CLF). The CLF says: given that a halo of mass M has a central galaxy brighter than Mr_threshold, what luminosity does it actually have?

**The CLF:**
```
P_cen(>L | M) = fcen(>L|M) / fcen(>L_min|M)
```

This is the cumulative probability that the central galaxy is brighter than L, conditioned on it being brighter than L_min.

**Algorithm — Inverse transform sampling:**
```python
for h in range(lgm_halos.size):
    PcenL = fcen_thresh(Mrvals, lgm_halos[h]) / fcen_thresh(Mrmax, lgm_halos[h])
    # PcenL[i] = P(>Mrvals[i] | halo h)
    
    u = rng.rand()  # uniform random number
    
    # Find the two magnitudes bracketing u in the CDF
    Mr_upp = Mrvals[PcenL > u][0]    # brightest Mr with probability > u
    Mr_low = Mrvals[PcenL <= u][-1]  # faintest Mr with probability ≤ u
    
    # Choose the fainter one (enforce nesting: central fainter than threshold)
    Mr_cen[h] = max(Mr_upp, Mr_low)  # max = dimmer (magnitudes: bigger = dimmer)
```

**Why "enforce nesting"?** We want the central galaxy to be no brighter than the threshold Mr_max = -20.5. Choosing the larger (dimmer) of the two bracketing values ensures this.

**Note:** The code loops over halos explicitly. The comment says this is "not much slower than vectorised calc, but more memory-efficient." For ~67,000 halos, memory efficiency matters.

---

#### `get_Poisson(self, lgm_halos)`

**Purpose:** Assigns each central galaxy a number of satellite galaxies, drawn from a Poisson distribution.

```python
mean_Nsat = self.Nsat_thresh(self.Mrmax, lgm_halos)  # ⟨Nsat⟩ for each halo
Nsat_values = self.rng.poisson(mean_Nsat)              # random Poisson draw
```

**Key point:** Even if ⟨Nsat⟩ = 0.01, the Poisson draw can still give 1 satellite occasionally (with probability 1%). This is correct physics — rare but possible.

---

#### `assign_satellites(self, centrals)`

**Purpose:** For all halos with Nsat > 0, creates satellite galaxy records with properties.

**Algorithm — Three separate loops:**
1. **Loop 1:** Assign halo ID, log mass, concentration to each satellite
2. **Loop 2:** Assign positions using NFW profile sampling
3. **Loop 3:** Assign luminosities using inverse transform sampling

**Why three separate loops instead of one?** The author chose to separate concerns. It's easier to read and debug. The performance is similar.

**Indexing trick:**
```python
s_lo = 0
for h in range(centrals.size):
    s_hi = s_lo + centrals['Nsat'][h]
    sl = np.s_[s_lo:s_hi]  # slice object [s_lo:s_hi]
    satellites['haloid'][sl] = centrals['haloid'][h]
    s_lo = s_hi
```

The satellites array is flat (1D). Satellites belonging to halo h occupy slots [s_lo:s_hi]. This is a memory-efficient way to store variable-length satellite lists per halo.

---

#### `assign_satellite_luminosities(self, centrals, h)`

**Purpose:** Assigns luminosities to all satellites of halo h.

```python
PsatL = Nsat_thresh(Mrvals, lgmhalo) / Nsat_thresh(Mrmax, lgmhalo)
u = rng.rand(Nsat_h)
Mr_sat = np.array([Mrvals[(PsatL >= u[s])][0] for s in range(Nsat_h)])
```

**Physics:** Satellite luminosities come from the satellite CLF, which is a renormalized version of Nsat(>L|M).

**Inverse transform sampling:** For each satellite s, find the brightest Mr where Psat(>Mr|M) ≥ u[s]. This samples from the cumulative distribution.

---

#### `assign_satellite_pos(self, centrals, h)`

**Purpose:** Places satellites at positions drawn from the NFW profile around the central galaxy.

```python
sat_pos = self.gen_NFW_profile(Nsat_h, cvir=con, Rvir=rvir, rng=self.rng)
sx = sat_pos[:,0] + centrals['x'][h]   # add halo center position
sy = sat_pos[:,1] + centrals['y'][h]
sz = sat_pos[:,2] + centrals['z'][h]

# Periodic boundary conditions
sx = sx % self.Lbox
sy = sy % self.Lbox
sz = sz % self.Lbox
```

**Periodic boundary conditions:** When a satellite's position goes beyond the box edge (e.g., x > 300), wrap it back: x = x % 300. This simulates an infinite periodic universe.

---

#### `gen_NFW_profile(self, Nsat, cvir, Rvir, rng, include_cen=False, clean_up=False)`

**Purpose:** Generates 3D positions distributed according to the NFW profile.

```python
# Sample angular positions uniformly on sphere
phi = 2*pi * rng.rand(Nsat)
cos_theta = 2*rng.rand(Nsat) - 1
sin_theta = np.sqrt(1 - cos_theta**2)

# Sample radii from NFW profile (via gen_rsamp)
rsamp = self.gen_rsamp(Nsat, cvir=cvir, Rvir=Rvir, rng=rng)

# Convert to Cartesian
x = rsamp * sin_theta * sin(phi)
y = rsamp * sin_theta * cos(phi)
z = rsamp * cos_theta
```

**Output shape:** (Nsat, 3) — one (x,y,z) per satellite, relative to the halo center.

---

#### `gen_rsamp(self, Nsat, cvir, Rvir, rng, clean_up=False)`

**Purpose:** Samples radii from the NFW profile using inverse transform sampling.

```python
xmax = 2*cvir          # maximum x = r/rs to consider
rs = Rvir / cvir       # scale radius

xfine = np.linspace(0, xmax, 100000)   # fine grid in x = r/rs
Px = self.Mencl_nfw(xfine)             # mass enclosed as function of x
Px /= self.Mencl_nfw(cvir)             # normalize to [0,1] at rvir

vran = rng.rand(Nsat)                  # uniform random numbers
rsamp = np.interp(vran, Px, xfine)    # inverse transform via interpolation
rsamp *= rs                            # convert from r/rs to physical units
```

**Why xmax = 2*cvir?** The virial radius corresponds to x = cvir (since cvir = rvir/rs). Going to 2*cvir allows satellites to extend slightly beyond the virial radius (physically reasonable).

**Small-x correction:**
```python
ind_small = np.where(xfine < 1e-3)[0]
if ind_small.size:
    Px[ind_small] = xfine**2/2 - 2*xfine**3/3 + 3*xfine**4/4 - 4*xfine**5/5
```

At very small x, `ln(1+x) - x/(1+x)` suffers from numerical cancellation. The Taylor series expansion is used instead for accuracy.

---

#### `Mencl_nfw(self, x)`

**Purpose:** NFW enclosed mass function (normalized).

```
M_enclosed(x) = ln(1+x) - x/(1+x)
```

where x = r/rs.

**Derivation:** Integrating the NFW density profile from 0 to r:
```
M(<r) ∝ ∫₀ʳ [ρ₀ / (r'/rs)(1+r'/rs)²] × 4πr'² dr'
```

After substitution u = r'/rs and integration:
```
M(<r) ∝ ln(1+x) - x/(1+x)   where x = r/rs
```

This function monotonically increases from 0 to M(cvir) and is used as the CDF for radial sampling.

---

#### `mock_it(self)`

**Purpose:** Main execution function — calls everything in the right order.

```python
def mock_it(self):
    halos_all = self.prep_halo_data()        # Read & filter halos
    occupy = self.occupy_halos(halos_all['m200b'])  # Monte Carlo occupation
    halos = halos_all[occupy]               # Keep only occupied halos
    
    centrals = self.assign_centrals(halos)   # Assign central galaxy properties
    satellites = self.assign_satellites(centrals)  # Assign satellite properties
    
    return centrals, satellites, all_masses
```

**Output:** The complete mock catalog — 82,983 galaxies (66,975 centrals + ~16,008 satellites).

---

### 6.5 Class: TwoPointCorrelationFunctionPeriodic

**File:** MockTutorial.ipynb (Cell 4)

**Purpose:** Calculates the two-point correlation function (either monopole ξ(r) or projected wₚ(rₚ)) for a periodic simulation box.

**Pipeline position:** Takes the mock catalog as input, outputs clustering statistics.

---

#### `__init__(self, lgbin, rmin, rmax, nbin, Lbox, proj, pimax)`

**Purpose:** Initializes binning setup.

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| lgbin | 1 | 1 = logarithmic bins, 0 = linear bins |
| rmin | 0.01 | Minimum separation [Mpc/h] |
| rmax | 30 | Maximum separation [Mpc/h] |
| nbin | 15 | Number of bins |
| Lbox | 300.0 | Box size [Mpc/h] |
| proj | 0 | 0 = monopole ξ(r), 1 = projected wₚ(rₚ) |
| pimax | 60.0 | Max line-of-sight separation for projection [Mpc/h] |

**Logarithmic binning:**
```python
rbin = np.logspace(log10(rmin), log10(rmax), nbin+1)
rmid = np.sqrt(rbin[1:] * rbin[:-1])   # geometric mean = bin center in log space
```

**Why logarithmic?** The correlation function spans orders of magnitude in separation. Logarithmic bins give equal resolution in log(r), capturing both small-scale (1-halo term) and large-scale (2-halo term) behavior.

**Why geometric mean for bin center?** In log space, the "center" of a bin [r₁, r₂] is √(r₁ × r₂), not (r₁ + r₂)/2.

**Projected mode setup:**
For projected correlation, the code also sets up:
- `rpbin`, `rpmid`: bins in projected separation rₚ
- `rbin`, `rmid`: finer bins in 3D separation r (for the intermediate monopole calculation)
- `rbin_int`, `rmid_int`: even finer interpolation bins

This layered binning is needed because wₚ requires integrating ξ(r) from rₚ to √(rₚ² + πmax²).

---

#### `pair_counts(self, pos_1, pos_2)`

**Purpose:** Counts pairs of galaxies at each separation bin.

```python
tree_1 = cKDTree(pos_1, boxsize=self.Lbox)  # build KD-tree for dataset 1
tree_2 = cKDTree(pos_2, boxsize=self.Lbox)  # build KD-tree for dataset 2

cum_counts = tree_1.count_neighbors(tree_2, self.rbin, cumulative=True)
bin_counts = np.diff(cum_counts)   # convert cumulative to per-bin counts
```

**`count_neighbors(tree2, r)`:** For each point in tree1, counts how many points in tree2 are within distance r. With `cumulative=True`, gives counts within each radius. `np.diff()` converts to per-bin.

**Periodic boundary:** The `boxsize` parameter tells the KD-tree to use periodic boundary conditions — distances wrap around the box edges.

**Output:** Array of length nbin — raw pair counts in each separation bin.

---

#### `RR_theory(self)`

**Purpose:** Computes the expected number of random-random (RR) pairs analytically.

```
RR = (4π/3)(r_max³ - r_min³) / L_box³
```

**Derivation:** For a completely random distribution in a box of side L:
- Total pairs = N × N (normalized by N²)
- Pairs at separation [r₁, r₂] = (volume of shell) / (box volume)
- Shell volume = (4π/3)(r₂³ - r₁³)

**Why use theory instead of random catalogs?** Usually, you'd generate a large random catalog and count pairs. But in a **periodic box**, the random distribution is perfectly uniform, so you can compute RR analytically. This is exact and saves enormous computation time.

**Validity:** Only valid for rmax < Lbox/2. Otherwise, edge effects become significant.

---

#### `twoPCF(self, pos_data, pos_data2=None)`

**Purpose:** Main function — computes the correlation function.

**Algorithm:**
```python
DD = self.DD_split(pos_data, pos_data2)  # count galaxy pairs
RR = self.RR_theory()                      # expected random pairs
cf_r = DD / RR - 1.0                      # Peebles-Hauser estimator
```

**The Peebles-Hauser estimator:**
```
ξ̂(r) = DD/RR - 1
```

If galaxies are random (uniform), DD = RR → ξ = 0.
If galaxies cluster, DD > RR → ξ > 0.

**If proj=1 (projected mode):**
The code then integrates ξ(r) to get wₚ(rₚ):
```
wₚ(rₚ) ≈ 2 ∫_{rₚ}^{√(rₚ² + πmax²)} ξ(r) × r / √(r² - rₚ²) dr
```

This is the Abel transform, implemented numerically with `np.trapz` (trapezoidal integration).

**Special handling of the lower integration limit:**
```python
Drp = rmid_int[ind[0]] - rpmid[rp]
add_this = np.sqrt(2*Drp/rpmid[rp]) * cf_r_int[ind[0]] * rpmid[rp]
cf[rp] = add_this + 2 * np.trapz(...)
```

Near r = rₚ, the integrand diverges as 1/√(r² - rₚ²). The first term handles this singular behavior analytically, and the trapz handles the rest.

---

#### `DD_only(self, pos_data1, pos_data2=None)`

**Purpose:** Computes normalized pair counts.

**Auto-correlation (pos_data2 is None):**
```python
cf = pair_counts(pos_data1, pos_data1) / ndata / (ndata - 1)
```

Normalize by N(N-1) — total ordered pairs (excluding self-pairs).

**Cross-correlation:**
```python
cf = pair_counts(pos_data1, pos_data2) / ndata1 / ndata2
```

Normalize by N1 × N2.

---

#### `DD_split(self, pos_data, pos_data2=None)`

**Purpose:** Wrapper around `DD_only` that handles large datasets by recursively splitting them in half.

**The problem:** For N > N_SPLIT = 8000 points, `cKDTree` runs slowly or out of memory.

**Binary split algorithm:**
```
DD(A∪B, A∪B) = DD(A,A) + 2×DD(A,B) + DD(B,B)
```

Where A = first half, B = second half.

This is just algebra: every pair in the full set belongs to one of: (A,A), (A,B), or (B,B).

The recursion continues until each chunk is ≤ N_SPLIT.

**Time complexity:** O(N log² N) instead of O(N²).

**Memory efficiency:** Only 2 × N_SPLIT/2 points are held in RAM at once at the deepest level.

---

## 7. The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   PIPELINE OVERVIEW                          │
└─────────────────────────────────────────────────────────────┘

out_1.parents (halo catalog)
        │
        ▼
┌─────────────────────┐
│  prep_halo_data()   │  ← Read file, apply mass cut, keep only pid=-1
│  458,324 halos      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  occupy_halos()     │  ← Monte Carlo: does this halo get a central galaxy?
│  67,102 occupied    │    probability = fcen_thresh(Mr=-20.5, lgm)
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  assign_centrals()                                           │
│  ├── Copy halo ID, position, log mass                       │
│  ├── Compute rvir (from mass + critical density)            │
│  ├── Compute concentration c = rvir/rs                      │
│  ├── assign_central_luminosity() ← inverse transform on CLF │
│  └── get_Poisson()              ← Poisson(⟨Nsat⟩)          │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  assign_satellites()                                         │
│  ├── Loop 1: Copy halo ID, lgm, concentration              │
│  ├── Loop 2: assign_satellite_pos()                         │
│  │     └── gen_NFW_profile()                               │
│  │           └── gen_rsamp() ← inverse transform on NFW    │
│  └── Loop 3: assign_satellite_luminosities() ← CLF         │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  TwoPointCorrelationFunctionPeriodic                         │
│  ├── pair_counts() ← cKDTree                               │
│  ├── RR_theory() ← analytic random-random                  │
│  └── twoPCF() ← DD/RR - 1, then project for wp            │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Plots & Output │
└─────────────────┘
```

---

## 8. Running the Code

**What the notebook does step by step:**

1. **Create `Mocker` instance:** `mm = Mocker()` — sets all parameters.

2. **Read halos:** `halos = mm.prep_halo_data()`
   - Reads 2.9 million objects, keeps 458,324 parent halos above mass cut.

3. **Visualize halo mass function:** Histogram of log₁₀(m200b) on log scale.
   - Shows how many halos exist at each mass. More small halos than large ones.
   - "Occupied" subset shows only halos above the effective HOD threshold.

4. **Run full pipeline:** `centrals, satellites, all_masses = mm.mock_it()`
   - Prints progress messages: number occupied, luminosities assigned, etc.
   - Output: ~82,983 galaxies total (66,975 centrals + ~16,008 satellites)

5. **Plot luminosity function:** Histogram of Mr for centrals and satellites.
   - Shows the distribution of galaxy brightnesses.
   - Satellites are typically fainter than centrals.

6. **Set up correlation function calculators:**
   ```python
   tpcf_real = TwoPointCorrelationFunctionPeriodic(rmin=0.01, rmax=60, nbin=20, Lbox=300, proj=0)
   tpcf = TwoPointCorrelationFunctionPeriodic(rmin=0.1, rmax=40, nbin=20, Lbox=300, proj=1, pimax=40.0)
   ```

7. **Measure projected correlation function:**
   ```python
   wp_cen = tpcf.twoPCF(cpos, cpos)   # centrals auto-correlation
   wp_sat = tpcf.twoPCF(spos, spos)   # satellites auto-correlation
   wp_gal = tpcf.twoPCF(gpos, gpos)   # all galaxies auto-correlation
   ```

8. **Compare to data:** Load observational data (Zehavi 2011) and plot alongside the mock.
   - The mock wₚ should match the observed wₚ if the HOD parameters are correct.

---

## 9. How to Rebuild This From Scratch

This section is your roadmap to recreating the project without looking at the original code.

### Step 1: Understand the HOD parameters
- Read the paper: Paul, Pahwa, Paranjape 2019
- Know the 5 parameters: Mmin, σ_lgM, M1, M0, α
- Know their dependence on Mr

### Step 2: Implement HODFits
```python
class HODFits:
    def __init__(self):
        self.bf = {...}  # store coefficients
    
    def hod_fit(self, param, Mr):
        # Return parameter value at Mr
        x = Mr + 20.5
        a0, a1, a2 = self.bf[param]
        if param == 'siglgM':
            return a0 + a1*erf(x/a2)
        else:
            return a0 + a1*x + a2*x**2
```

### Step 3: Implement HODFunctions
```python
def fcen_thresh(self, Mr, lgm):
    erfarg = (lgm - log10(Mmin(Mr))) / sigma(Mr)
    return 0.5 * (1 + erf(erfarg))

def Nsat_thresh(self, Mr, lgm):
    M0 = 10**hod_fit('lgM0', Mr)
    M1 = 10**hod_fit('lgM1', Mr)
    alpha = hod_fit('alpha', Mr)
    m = 10**lgm
    return max(0, (m - M0)/M1)**alpha
```

### Step 4: Read halo catalog
```python
import pandas as pd
halos = pd.read_csv('out_1.parents', comment='#', 
                    delim_whitespace=True, names=[...])
# Filter: m200b > mmin and pid == -1
```

### Step 5: Implement Monte Carlo occupation
```python
fcen = fcen_thresh(Mr_threshold, log10(halos['m200b']))
u = np.random.rand(len(halos))
occupied = u <= fcen
```

### Step 6: Assign luminosities via inverse transform
```python
Mr_grid = np.linspace(-23.5, -20.5, 3000)
for h in occupied_halos:
    P_cen_L = fcen_thresh(Mr_grid, lgm[h]) / fcen_thresh(-20.5, lgm[h])
    u = np.random.rand()
    Mr_cen[h] = np.interp(u, P_cen_L[::-1], Mr_grid[::-1])
```

### Step 7: Assign Poisson satellite counts
```python
mean_Nsat = Nsat_thresh(-20.5, lgm_halos)
Nsat = np.random.poisson(mean_Nsat)
```

### Step 8: Compute virial radius and concentration
```python
rvir = (3*M / (4*pi * 200*Om * rhoc))**(1/3)
con = rvir / rs
```

### Step 9: Sample NFW positions
```python
def sample_nfw(N, con, rvir):
    rs = rvir / con
    xmax = 2*con
    x_grid = np.linspace(0, xmax, 100000)
    CDF = Mencl_nfw(x_grid) / Mencl_nfw(con)
    u = np.random.rand(N)
    x_samp = np.interp(u, CDF, x_grid)
    r_samp = x_samp * rs
    # Sample angles
    phi = 2*pi * np.random.rand(N)
    cos_theta = 2*np.random.rand(N) - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    return r_samp * np.c_[sin_theta*np.sin(phi), sin_theta*np.cos(phi), cos_theta]
```

### Step 10: Compute 2PCF
```python
from scipy.spatial import cKDTree

def compute_xi(positions, Lbox, rmin, rmax, nbin):
    rbin = np.logspace(np.log10(rmin), np.log10(rmax), nbin+1)
    tree = cKDTree(positions, boxsize=Lbox)
    N = len(positions)
    cum = tree.count_neighbors(tree, rbin, cumulative=True)
    DD = np.diff(cum) / N / (N-1)
    RR = 4*pi/3 * np.diff(rbin**3) / Lbox**3
    return DD/RR - 1
```

---

## 10. Glossary of Symbols

| Symbol | Name | Units | Physical Meaning |
|---|---|---|---|
| M | Halo mass | M☉/h | Total dark matter mass of the halo |
| Mmin | Minimum HOD mass | M☉/h | Mass where 50% of halos host a central |
| σ_lgM | HOD width | dimensionless | Width of the central occupation step function in log M |
| M0 | Satellite cutoff mass | M☉/h | Minimum mass for satellite galaxies |
| M1 | Satellite scale mass | M☉/h | Mass where ⟨Nsat⟩ ≈ 1 |
| α | Satellite slope | dimensionless | Power law index for satellite count |
| Mr | Absolute magnitude (r-band) | magnitudes | Luminosity (more negative = brighter) |
| lgm | log₁₀(M) | dimensionless | Log of halo mass |
| z | Redshift | dimensionless | How much the universe has expanded since light was emitted |
| Ω_m | Matter density parameter | dimensionless | Fraction of total density in matter |
| H₀ | Hubble constant | km/s/Mpc | Current expansion rate of the universe |
| E(z) | Dimensionless Hubble | dimensionless | H(z)/H₀ |
| ρ_crit | Critical density | M☉/h / (Mpc/h)³ | Density for flat universe |
| r_s | Scale radius | Mpc/h | NFW profile parameter |
| r_vir | Virial radius | Mpc/h | "Edge" of the halo |
| c_vir | Concentration | dimensionless | r_vir / r_s |
| ξ(r) | Correlation function | dimensionless | Excess galaxy pairs at separation r |
| wₚ(rₚ) | Projected 2PCF | Mpc/h | Line-of-sight projected correlation |
| Δ_vir | Virial overdensity | dimensionless | 200 × Ω_m for m200b definition |
| erf | Error function | dimensionless | Integral of Gaussian |
| CLF | Conditional Luminosity Function | dimensionless | P(L|M) — luminosity given halo mass |
| HOD | Halo Occupation Distribution | dimensionless | N(M) — how many galaxies in halo of mass M |

---

## 11. Chapter Summaries and Exercises

### Summary: Part I — Physics

The universe contains dark matter halos. Galaxies form inside halos. The HOD statistically describes how many galaxies live in each halo. We use this to create a mock galaxy catalog that we can compare with real surveys.

### Summary: Part II — Mathematics

- erf gives the probability of the central occupation (smooth step function)
- Poisson distribution gives the number of satellite galaxies
- Inverse transform sampling draws luminosities from the CLF
- NFW profile describes where satellites live inside halos
- KD-trees make pair counting efficient
- Two-point correlation function measures galaxy clustering

### Summary: Part III — Code

| Class | Purpose |
|---|---|
| HODFits | Stores polynomial fit coefficients for HOD parameters |
| HODFunctions | Computes ⟨Ncen⟩ and ⟨Nsat⟩ as functions of M and L |
| Mocker | Reads halos, applies HOD, assigns galaxy properties |
| TwoPointCorrelationFunctionPeriodic | Measures galaxy clustering |

### Key Ideas to Remember

1. **Central galaxies sit at halo centers. Satellites orbit inside halos.**
2. **The HOD is probabilistic — each run gives slightly different results (but same seed = same result).**
3. **Luminosities are drawn by inverse transform sampling from the CLF.**
4. **Satellite positions follow the NFW profile.**
5. **The correlation function ξ(r) = DD/RR - 1 measures clustering.**
6. **KD-trees make the O(N²) pair counting tractable.**

### Common Beginner Mistakes

1. Confusing log₁₀ and natural log (ln). The code uses log₁₀ for masses.
2. Forgetting that magnitudes are a reversed scale (more negative = brighter).
3. Forgetting periodic boundary conditions (positions must be wrapped with % Lbox).
4. Not converting rs from comoving kpc/h to Mpc/h before computing concentration.
5. Raising (M - M₀)/M₁ to a power when it's negative — always clip to zero first.

### Exercises

1. **Easy:** Change `Mrmax = -21.0`. How many halos get occupied? Why does the number change?
2. **Medium:** What happens to wₚ(rₚ) if you set α = 2 (more satellites per halo)?
3. **Hard:** Implement the HOD for a different luminosity threshold, say Mr < -22.
4. **Challenge:** Replace the Peebles-Hauser estimator with the Landy-Szalay estimator: ξ = (DD - 2DR + RR)/RR. This requires generating a random catalog.
5. **Expert:** Add velocity information to the mock catalog and compute the 2PCF in redshift space.

### Suggested Reading

- **Cooray & Sheth 2002** — "Halo Models of Large Scale Structure" (review of the halo model)
- **Berlind & Weinberg 2002** — "The Halo Occupation Distribution and the Physics of Galaxy Formation"
- **Zehavi et al. 2011** — "Galaxy Clustering in the Completed SDSS Redshift Survey" (the observational data used here)
- **Paul, Pahwa, Paranjape 2019** — the paper that provides the HOD fit coefficients used in this code
- **Navarro, Frenk & White 1996/1997** — the original NFW profile papers

---

*This README was generated as a teaching document to help beginners understand the complete MockTutorial codebase — from zero physics and math background to being able to rebuild the entire pipeline from scratch.*
# Mock_galaxy_catalog_tutorial
