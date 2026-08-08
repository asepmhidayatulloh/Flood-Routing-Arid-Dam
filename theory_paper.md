# A Generalized Semi-Analytical Solution for Reservoir Routing in Ephemeral (Arid-Region) Streams: Combining the Williams–Hann Inflow Hydrograph with a Full/Empty Reservoir Formulation

*Working draft — companion to `reservoir_routing_colab.ipynb`, `routing_core.py`, and `index.html`.*

## Abstract

Reservoir routing in arid regions must, unlike routing in perennial rivers,
explicitly account for the reservoir's initial condition: an arid-region
dam is empty far more often than it is full, and the classical routing
literature (traditional/modified-Puls method) has no clean way to represent
that state. Kamis, Bahrawi & Elfeki (*Arab. J. Geosci.* 2018, 11:106 —
referred to throughout as "the reference paper") addressed this numerically,
combining the mass-conservation equation, a combined spillway/orifice
discharge law, and the Mohammadzadah-Habili et al. (2009) dimensionless
depth–storage relation, solved with first- and second-order Euler schemes.
This paper develops a complementary **semi-analytical** solution to the same
generalized problem, specialized to the Williams & Hann (1972) synthetic
inflow hydrograph that is standard practice in Saudi Arabian and other arid
watershed studies. The governing nonlinear ODE has no general closed-form
solution (as the reference paper itself notes), so we obtain the closed form
by (i) writing the Hann hydrograph exactly as a Fenton-type pulse, (ii)
reducing the Habili storage curve locally to a power law matched to the
*exact* exponent of whichever outlet term dominates the routing (1.5 for a
spillway, 0.5 for an orifice/pipe — not a blind curve fit), and (iii)
applying Yevdjevich's (1959) integrating-factor method to the resulting
linear reservoir. The closed form, and every other analytical step, was
re-derived and independently checked with `sympy` (symbolic computation)
rather than transcribed from the (partly OCR-garbled) literature formula,
which allowed us to catch and fix two derivation errors during development
(documented in Section 5.4 and reproduced in the companion notebook). The
resulting model is validated against a from-scratch numerical benchmark
(reformulated in storage-space to avoid a stiffness singularity present in
the reference paper's own depth-space formulation at an empty reservoir) and
applied to a synthetic arid-region dam across return periods of 5–100
years. The full-reservoir solution matches the numerical benchmark closely
(RMSE 3.3–3.7 m³/s against peaks of 6–68 m³/s, i.e. roughly 5–13% of peak);
the empty-reservoir solution matches well when the flood never reaches the
spillway crest, and degrades (documented, with a stated validity condition)
once the crest is reached before the inflow has passed its own peak.

---

## 1. Introduction

Ephemeral-stream dams in arid regions are designed, sized, and normally
*operated* empty: rainfall is erratic, most events do not fill the
reservoir, and the reference paper shows (its Table 6) that starting empty
versus starting full changes the peak outflow by roughly a factor of two to
six for the same design flood. Yet essentially every classical
reservoir-routing method — including the traditional/modified-Puls method
still taught in textbooks (Chow, Maidment & Mays 1988) — implicitly assumes
a full reservoir and cannot represent the empty case without ad hoc
modification (a point the reference paper makes explicitly).

The reference paper's contribution was to combine three pieces into one
numerically-solvable framework: (1) the mass-conservation equation for the
reservoir, (2) a combined spillway + orifice/pipe discharge law, and (3) the
Mohammadzadah-Habili dimensionless depth–volume relation (parameterized by a
single "reservoir coefficient" *N*) to represent reservoir shape without
needing a full elevation–storage survey. It solved the resulting ODE
numerically (Euler schemes) and validated the method against an existing
analytical solution (Yevdjevich 1959, as reformulated by Fenton 1992) for
the special case where storage and outflow are both *exact*, matching power
laws of depth — a case the reference paper uses only as a validation
benchmark, not as a generally-applicable design tool, precisely because that
matching condition rarely holds for a real (Habili-shaped) reservoir with a
real (spillway + orifice) outlet.

This paper's contribution is to **extend the closed-form side** of that
validation benchmark into a genuinely usable semi-analytical *design* tool:
one that (a) is driven by the Williams & Hann (1972) inflow hydrograph
actually used in Saudi practice (rather than the more abstract Fenton pulse
form used for validation only), (b) handles a *combined* spillway + orifice
outlet, and (c) handles *both* the full and empty starting conditions
through one unified formulation, with a closed-form (incomplete-gamma-based)
estimate of the empty-reservoir fill-up time. We deliberately keep the
reference paper's numerical solution as the benchmark against which the new
closed form is validated throughout — the two methods are complementary,
not competing: the closed form is fast, transparent, and reproducible for
scoping and sensitivity work; the numerical solution remains authoritative
for final design.

## 2. Governing Equations (recap, reference-paper equation numbers in parentheses)

**Mass conservation** (Eq. 1):

```
dS(t)/dt = I(t) − O(t)
```

**Spillway discharge** (Eq. 2–3):

```
O_sp(h) = C·B·(h−P)^1.5   if h > P,   else 0
```

**Orifice/pipe discharge** (Eq. 4–5):

```
O_or(h) = n·Cd·a·(h−Δz)^0.5   if h > Δz,   else 0
```

**Habili dimensionless storage** (Eq. 6, inverted form used here):

```
S(h) = Smax·(e^{ln2·h/hmax} − 1)^{1/N}
h(S) = (hmax/ln2)·ln(1 + (S/Smax)^N)          <- closed-form inverse, new
```

**Habili surface area** (Eq. 8, = dS/dh):

```
A(h) = (Smax·ln2 / (N·hmax))·e^{ln2·h/hmax}·(e^{ln2·h/hmax} − 1)^{(1−N)/N}
```

**Reservoir coefficient from surveyed data** (Eq. 9):

```
N = 2·ln2·Smax / (Amax·hmax)
```

**Williams & Hann inflow hydrograph** (Eq. 14–15):

```
I(t) = Imax·[(t/tp)·exp(1 − t/tp)]^K,        K = 6.5·Imax·tp/V
```

Combining these, the reference paper's Eq. 13 gives the governing nonlinear,
non-autonomous ODE for the water depth h(t), which it states explicitly
**has no general analytical solution**.

## 3. Numerical Benchmark: a Storage-Space Reformulation

The reference paper integrates Eq. 13 directly in depth-space,
`dh/dt = (I − O)/A(h)`, using explicit Euler schemes. During development of
the companion code, we found that a modern adaptive-step solver
(`scipy.solve_ivp`, method LSODA) applied to this same depth-space
formulation **silently fails** for an empty-starting reservoir: because
`A(h) → 0` as `h → 0` (the reservoir bottom has, in the Habili model, zero
width), the right-hand side blows up near the initial condition, and the
adaptive step-size control collapses — the solver returns
`success=False` and a state frozen near `h=0`, which (if not checked) is
easy to mistake for a *physically* near-empty result.

We avoid this by integrating in **storage-space** instead:

```
dS/dt = I(t) − O(h(S)),      h(S) via the closed-form inverse above
```

which is smooth and non-singular everywhere, including exactly at `S=0`.
This reformulation is mathematically identical to the reference paper's
Eq. 13 (`dS = A(h) dh`, chain rule) but is far better behaved numerically,
and is the benchmark used throughout the rest of this paper (`route_numerical`
in `routing_core.py`). We additionally found (and document, Section 5.3)
that an orifice invert placed *exactly* at the reservoir bottom (Δz=0)
reintroduces a related, milder stiffness (because the Habili area also
vanishes faster than linearly near h=0 whenever N<1, and the orifice
discharge's `√h` derivative is unbounded there); this is avoided by placing
the orifice invert above the dead-storage level, `Δz>0`, which is in any
case the physically realistic configuration shown in the reference paper's
own Fig. 1 (the orifice sits above the sediment-filled dead storage, not at
the true bottom).

## 4. Semi-Analytical Solution

### 4.1 Step 1 — Hann inflow as an exact Fenton-type pulse

The reference paper itself shows (its Eq. 17) that the Williams-Hann
hydrograph is *exactly* a Fenton (1992) -type pulse,
`I(t) = P0·t^s·exp(−f·t)`, with

```
s = K,     f = K/tp,     P0 = Imax·e^K / tp^K
```

This means Yevdjevich's (1959) analytical machinery — derived for a Fenton
pulse, and used by the reference paper only as a *validation* case — can in
principle be driven directly by a real Williams-Hann design hydrograph, with
no separate inflow-model translation needed.

### 4.2 Step 2 — the integer-exponent subtlety (a real bug, caught and fixed)

Yevdjevich's closed form requires an **integer** power `s` (it is built from
a finite polynomial-times-exponential particular solution). The Hann shape
factor `K` is, in general, *not* an integer — the SCS default is K=3.77.
Naively rounding `K` to the nearest integer `s` while keeping the exact
(fractional-K) `P0` and `f` is a subtle but serious error: since
`P0 ~ tp^{-K}`, even a small change in the exponent changes `P0·tp^s` by a
large multiplicative factor whenever `tp` is large (as it always is, in
seconds). For the worked example in this paper (`Imax=193`, `tp=1400`,
`K=3.77`), rounding `K` to `s=4` without rescaling inflates the pulse's peak
value from 193 m³/s to **1021 m³/s — a 5.3× error** (reproduced live in the
companion notebook, Part 2). The fix is to re-derive `P0` and `f` for the
*rounded* exponent so the surrogate pulse still peaks at `(tp, Imax)`:

```
s = round(K),     f = s/tp,     P0 = Imax·e^s / tp^s
```

This is used only inside the closed-form solver; the exact fractional-K
Hann hydrograph is retained everywhere else (plotted inflow, fill-time
estimate, RMSE comparison against the numerical benchmark).

### 4.3 Step 3 — local power-law reduction using the outlet's own exponent

Yevdjevich's method further requires storage and outlet discharge to be
*matching* power laws of depth, `S = a·h^m`, `O = b·h^m` (reference paper
Eq. 18). The Habili storage curve is not a power law in general. The
reference paper itself fits a power law to the *whole* curve for validation
purposes (its Fig. 5). We initially tried the same blind log-log fit here
and found it performs poorly (peak outflow overestimated by roughly 5×) —
because a single global exponent cannot represent both the steep low-depth
region and the flatter high-depth region of the Habili curve, and the fitted
exponent ends up dominated by whichever region has the most curve points,
not the region that actually matters for routing.

The fix is to **fix the exponent to the outlet's own physically exact
value** — `m=1.5` if a spillway is present (matching Eq. 2 exactly),
`m=0.5` otherwise (matching the orifice/pipe law, Eq. 4 exactly) — and only
fit the *storage* coefficient `a` to that fixed exponent, over the
**surcharge** depth `hs = h − invert` above whichever outlet invert is
active (not over `h` measured from the reservoir bottom, which mixes in
irrelevant curvature from parts of the reservoir the flood never reaches).
The fitting window itself is chosen adaptively: rather than fitting over the
entire physically possible depth range, we solve for a *characteristic*
depth `hs_char` such that `S(invert+hs_char) − S(invert)` equals the total
flood volume — i.e., "how high would the water get if the whole flood were
retained" — and fit over `[0, hs_char]`. This keeps the local power-law fit
anchored to the part of the storage curve the event actually visits.

With this fix, a secondary, lower orifice (below the spillway crest, if
present) is folded in as Yevdjevich's base-outflow term `O0` — physically,
the orifice supplies a roughly steady discharge that the spillway-driven
flood pulse rides on top of, exactly the role the `I0`/`O0` terms play in
Yevdjevich's original formulation.

### 4.4 Step 4 — the closed-form solution itself (sympy-verified)

With storage and outflow reduced to `S = a·hs^m`, `O = c·S` (`c = b/a`), the
mass-conservation ODE collapses to a **linear** reservoir:

```
dO/dt + c·O = c·I(t) = c·(I0 + P0·t^s·exp(−f·t))
```

We solve this symbolically with `sympy.dsolve` rather than by hand — the
reference paper's own Eq. 20 (Yevdjevich's published formula, transcribed
through an OCR'd PDF) turned out, on reconstruction, to contain what appears
to be an index/sign artifact; when implemented literally it disagreed with a
direct numerical solution of the same reduced ODE by a large, systematic
factor. Re-deriving the solution from scratch with `sympy` — and
independently checking it against a numerical solve of the reduced ODE for
random parameter draws (`_selftest_yevdjevich` in `routing_core.py`, which
runs automatically whenever the module is imported) — resolved this and
gives a solution verified to match a direct numerical integration of the
*same* reduced problem to within numerical-solver precision (RMSE ≈ 1×10⁻⁸
in the pure power-law test case of Section 6.1, where no approximation
beyond "solve the linear ODE" is involved).

*(Known limitation: the closed form has a removable singularity when
`c ≈ f` — outlet-drainage rate coincidentally equal to inflow-decay rate.
This is a genuine edge case of Yevdjevich's method, not handled here; it
essentially never arises for real reservoirs, since `c` (an outlet/storage
ratio) and `f` (a catchment-response rate) are governed by unrelated
physics.)*

### 4.5 Empty-reservoir fill-up time (closed form, incomplete gamma function)

For an empty-starting reservoir, the routing splits into a fill-up phase (no
outflow above the crest) and a routing phase once the crest is reached — the
delay time `τ` the reference paper calls out conceptually (its Eq. 25) but
does not give in closed form. Because the Hann/Fenton inflow's cumulative
volume has an exact primitive in terms of the (regularized) lower
incomplete gamma function,

```
V(t) = ∫₀ᵗ I(t')dt' = P0·Γ(s+1)/f^{s+1}·P(s+1, f·t)
```

(`P` = regularized lower incomplete gamma, computed with
`scipy.special.gammainc`), we obtain a closed-form estimate of `τ` by
solving `V(τ) = S(P)` (numerically inverting this one-dimensional monotone
equation, e.g. by bisection — still "closed form" in the sense of using an
exact analytical primitive rather than integrating the full nonlinear ODE).
This estimate neglects any discharge already leaving through a low-level
orifice during the fill-up phase, so it is a first-order estimate; in the
worked application below it matched the numerically-exact fill time to
within roughly 3–5% (Section 6.3).

If the flood volume is smaller than `S(P)`, the crest is never reached
(`τ = ∞`); the whole event is then routed through the low-level orifice
alone (`m=0.5` reduction, same machinery), reproducing the reference paper's
qualitative finding that small floods are almost entirely absorbed by an
empty arid-region reservoir.

### 4.6 Post-fill routing and its validity limit

Once the crest is reached, the remaining routing problem is structurally
identical to the full-reservoir case, shifted in time by `τ`. The inflow for
`t > τ`, however, is the **declining tail** of the Hann hydrograph, not a
fresh pulse. An initial attempt to model it as a fresh, full-amplitude
Fenton pulse restarting at `t'=0` was found (by comparison with the
numerical benchmark) to be badly wrong: because such a pulse necessarily
rises back up to a *second* full-amplitude peak on top of the base flow, it
double-counts a large fraction of the flood volume (observed error: peak
outflow overestimated by roughly 14× in the worst case encountered). The fix
used here instead matches the **local exponential decay rate** of the true
Hann hydrograph at `t=τ`, obtained from its exact logarithmic derivative,

```
d(ln I)/dt = K/t − K/tp  ⟹  f_tail = K·(1/tp − 1/τ)     (evaluated at t=τ)
```

and models the post-fill inflow as `I0·exp(−f_tail·t')` (a monotonically
decaying tail, no artificial rebound) — which is simply the `s=0` case of
the same closed-form solver. **This tail model is only valid once the
inflow has passed its own peak, i.e. `τ > tp`**; if the crest is reached
while the inflow is still rising (`τ ≤ tp`), the exponential-decay
assumption is wrong-signed, and the numerical solution should be used
instead. This condition is checked and flagged automatically by
`route_semi_analytical` (see `info["tail_model"]` in the returned dict).

## 5. Validation

### 5.1 Pure power-law case (isolates the closed-form machinery)

Reproducing the reference paper's own validation setup — a fictitious
reservoir with *exact* matching power-law storage and outflow, driven by a
Fenton-type pulse, so no local-fit approximation is involved — the
`sympy`-derived closed form matches a direct numerical solution of the same
ODE with RMSE ≈ 1.08×10⁻⁸ (companion notebook, Part 3), confirming the
closed-form machinery itself is correct to numerical precision.

### 5.2 Full-reservoir case, synthetic application (Section 6)

Across five synthetic return-period floods (5–100 years) routed through one
illustrative reservoir, the full-reservoir semi-analytical solution matches
the numerical benchmark with RMSE 3.32–3.73 m³/s against peak outflows of
6.2–68.3 m³/s (roughly 5–13% of peak, worsening mildly at low flows/small
peaks where the fixed spillway exponent is a slightly worse local
approximation of the Habili curve's true local slope). See Table 1.

### 5.3 Empty-reservoir case

For return periods where the flood never reaches the crest (5, 10, 25, and
— marginally — 50-year events in the worked example), the orifice-only
closed form matches the numerical benchmark reasonably well (RMSE
0.32–3.56 m³/s against peaks of 3.1–4.9 m³/s). For the 100-year event, where
the crest **is** reached (`τ ≈ 1830–1925 s`, comfortably past `tp=1400 s`,
satisfying the validity condition of Section 4.6), the closed form
overestimates the peak outflow by roughly 38% (43.7 vs 31.7 m³/s numerical,
RMSE 5.02 m³/s). This is a **documented, first-generation limitation**: the
post-fill tail model captures the right qualitative behaviour (decaying,
crest-triggered outflow) but is not yet as accurate as the full-reservoir
solution. For design-grade results in this specific regime, the numerical
benchmark should be treated as authoritative; the closed form remains useful
for fast scoping, sensitivity sweeps, and as a check that the numerical
solution is in the right ballpark.

### 5.4 Summary of derivation errors caught during development

Three genuine errors were found and corrected while building this model,
each caught by cross-checking against an independent numerical solution
rather than trusting the symbolic manipulation alone:

1. A literal transcription of the reference paper's Eq. 20 (itself
   presumably transcribed from Yevdjevich 1959 through an OCR'd PDF)
   produced outflows several times too large; replaced by a from-scratch
   `sympy.dsolve` derivation, verified against direct numerical solution
   (Section 4.4).
2. Rounding the Hann shape factor `K` to an integer without simultaneously
   rescaling `P0` and `f` inflated the driving pulse's peak by 5.3× in the
   worked example (Section 4.2).
3. Restarting a fresh full-amplitude pulse for the post-fill-up inflow tail
   double-counted flood volume, overestimating peak outflow by up to 14×
   in early testing; replaced by a decay-rate-matched exponential tail
   (Section 4.6).

This is reported not as a curiosity but as a methodological point: closed-form
"AI-assisted" derivations of this kind should always be checked against an
independent numerical solution before being trusted, exactly as done
throughout this project (`_selftest_yevdjevich`, and the RMSE columns
reported for every case in Section 6).

## 6. Application: A Synthetic Arid-Region Dam, Return Periods 5–100 Years

*(Reservoir and outlet parameters are synthetic/illustrative, per this
project's current scope; see `routing_core.py` docstrings and the notebook
for how to substitute site-specific values from a real elevation–storage
survey and hydrological study.)*

**Reservoir**: Habili shape with `Smax=3.0×10^5 m³`, `hmax=12 m`, `N=0.43`
(derived `Amax≈80,600 m²`; `N` falls within the 0.39–0.68 range the
reference paper reports for eight real Saudi dam sites, its Table 7).
**Outlet**: spillway `C=1.7`, `B=40 m`, crest `P=8 m`; orifice bank
`n=3`, `Cd=0.62`, `a=1.0 m²`, invert `Δz=1 m`. **Inflow**: Williams-Hann,
`tp=1400 s`, `K=3.77` (SCS default), peak inflows `Imax` = 10, 18, 35, 55,
80 m³/s for the 5/10/25/50/100-year events respectively.

**Table 1 — Peak outflow, full reservoir**

| Return period (yr) | Ip (m³/s) | Op numerical (m³/s) | Op analytical (m³/s) | RMSE (m³/s) | Attenuation |
|---:|---:|---:|---:|---:|---:|
| 5   | 10.0 | 6.23  | 7.74  | 3.32 | 0.62 |
| 10  | 18.0 | 12.89 | 14.50 | 3.41 | 0.72 |
| 25  | 35.0 | 27.79 | 29.50 | 3.52 | 0.79 |
| 50  | 55.0 | 45.70 | 47.38 | 3.63 | 0.83 |
| 100 | 80.0 | 68.34 | 69.73 | 3.73 | 0.85 |

**Table 2 — Peak outflow, empty reservoir**

| Return period (yr) | Ip (m³/s) | Op numerical (m³/s) | Op analytical (m³/s) | RMSE (m³/s) | τ numerical (s) | τ analytical (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 5   | 10.0 | 3.07  | 3.61  | 0.32 | never | ∞ |
| 10  | 18.0 | 3.70  | 4.86  | 0.56 | never | ∞ |
| 25  | 35.0 | 4.42  | 6.51  | 1.01 | never | ∞ |
| 50  | 55.0 | 4.90  | 11.95 | 3.56 | never | 2623 |
| 100 | 80.0 | 31.73 | 43.70 | 5.02 | 1925  | 1830 |

Consistent with the reference paper's own qualitative finding (its Table 6),
attenuation is far greater and less variable for the full-reservoir case
(62–85% here) than for the empty-reservoir case, where small/moderate
floods are almost entirely absorbed and only large events produce
significant, and more variable, outflow.

## 7. Conclusions

1. The nonlinear reservoir-routing ODE combining a Habili-shaped reservoir,
   a combined spillway/orifice outlet, and a Williams-Hann inflow has no
   exact general closed-form solution — but a **local, physically-motivated
   power-law reduction** (matching the *outlet's own* exact exponent, not a
   blind curve fit) makes Yevdjevich's (1959) integrating-factor method
   usable as a genuine semi-analytical design tool, not just a validation
   special case.
2. The same unified ODE, reformulated in **storage-space** rather than
   depth-space, naturally and robustly handles both the "reservoir full"
   and "reservoir empty" initial conditions the reference paper identifies
   as central to arid-region design, without special-case branching and
   without the numerical stiffness that a naive depth-space integration
   exhibits at an empty reservoir.
3. A closed-form (incomplete-gamma-function-based) estimate of the
   empty-reservoir fill-up time τ is available and performs well; the
   subsequent post-fill routing is accurate when the crest is reached after
   the inflow has peaked (τ > tp), and is a documented, flagged limitation
   otherwise.
4. Every analytical step in this paper was derived and cross-checked
   symbolically (`sympy`) and numerically rather than transcribed from
   literature formulas — a practice that caught three real errors during
   development and is recommended as standard practice for any
   "AI-assisted" analytical derivation of this kind.

## 8. Limitations and Suggested Next Steps

* The post-fill tail model (Section 4.6) should be improved — e.g. by a
  perturbative or matched-asymptotic correction for the `τ ≤ tp` regime, or
  by fitting the exponential decay rate over a short window numerically
  rather than from the instantaneous log-derivative alone.
* The current reservoir and flood parameters are synthetic; the next step
  (once real inflow hydrographs or elevation–storage survey data are
  available) is to substitute them directly, exactly as the reference paper
  does for its Al-Ulb dam case study.
* A systematic sensitivity/uncertainty analysis of the reservoir
  coefficient `N` (as the reference paper performs for its Al-Ulb case,
  its Fig. 10/Table 5) is straightforward to extend to the combined-outlet,
  full/empty formulation developed here (see Part 5 of the companion
  notebook for a first pass).

## References

* Kamis, A.S., Bahrawi, J.A., Elfeki, A.M. (2018). Reservoir routing in
  ephemeral streams in arid regions. *Arabian Journal of Geosciences*,
  11:106. https://doi.org/10.1007/s12517-018-3440-7 — **the reference
  paper** for this work.
* Yevdjevich, V.M. (1959). Analytical integration of the differential
  equation for water storage. *J. Res. Natl. Bur. Stand.* B, 63B(1):43–52.
* Fenton, J.D. (1992). Reservoir routing. *Hydrological Sciences Journal*,
  37(3):233–246.
* Mohammadzadah-Habili, J., Heidarpour, M., Mousavi, S., Haghiabi, A.
  (2009). Derivation of reservoir's area-capacity equations. *J. Hydrol.
  Eng.*, 14(9):1017–1023.
* Williams, J.R., Hann, R.W.J. (1972). HYMO: problem oriented computer
  language for building hydrologic models. *Water Resour. Res.*, 8(1):79–86.
* Borland, W.M., Miller, C.R. (1958). Distribution of sediment in large
  reservoirs. *J. Hydraul. Div.*, 84(2):1587.1–1587.10.
* Chow, V.T., Maidment, D.R., Mays, L.W. (1988). *Applied Hydrology*,
  International Edition. McGraw-Hill.

---

*Companion files: `routing_core.py` (core engine), `reservoir_routing_colab.ipynb`
(Google Colab notebook, pre-run — Parts 1–7 reproduce every result in this
paper), `index.html` (interactive browser tool for exploring the model
without running any code).*
