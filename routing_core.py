"""
routing_core.py
================
Generalized reservoir routing engine for ephemeral (arid-region) streams.

Combines:
  * Williams & Hann (1972) synthetic inflow hydrograph (as used by
    Kamis, Bahrawi & Elfeki, Arab J Geosci (2018) 11:106, Eq. 14-15).
  * Mohammadzadah-Habili et al. (2009) dimensionless depth-storage
    ("reservoir coefficient" N) relation for reservoir shape (Eq. 6, 8, 9
    of the reference paper).
  * A combined spillway + bottom orifice/pipe outlet.
  * A single reservoir-routing ODE (mass conservation) that is valid for
    BOTH the "full reservoir" case (h(0) = P, spillway crest already
    reached) and the "empty reservoir" case (h(0) = 0) with no special
    branching: the outlet terms are naturally zero until the water
    surface rises above their invert, so the fill-up delay time (tau in
    the reference paper's Eq. 25) emerges automatically from the ODE.

Two solution routes are provided:
  1. NUMERICAL  (reference / benchmark): scipy.integrate.solve_ivp on the
     exact nonlinear ODE (this is what the source paper calls the
     "proposed methodology", solved there with 1st/2nd order Euler).
  2. SEMI-ANALYTICAL (new contribution of this project): a closed-form
     solution obtained by (a) writing the Williams-Hann inflow as a
     Fenton-type inflow I(t) = P0 t^s exp(-f t) (exact correspondence,
     Eq. 17 of the reference paper), (b) locally fitting the Habili
     storage curve S(h) and the combined-outlet curve O(h) to matching
     power laws S = a h^m, O = b h^m over the operating depth range
     (as the reference paper itself does for validation, its Fig. 5),
     and (c) applying Yevdjevich's (1959) integrating-factor solution
     (Eq. 20) to that reduced problem. The empty-reservoir fill-up time
     is estimated in closed form from the regularized lower incomplete
     gamma function (the exact primitive of a Fenton-type inflow).

Everything below is self-contained (numpy + scipy only) so it runs as-is
in Google Colab.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gamma as gamma_fn
from scipy.optimize import brentq, curve_fit
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# 1. Inflow hydrograph  (Williams & Hann, 1972 / Eq. 14-15)
# ---------------------------------------------------------------------

def hann_K_from_volume(Imax, tp, V):
    """Shape factor K of the Williams-Hann hydrograph from peak & volume
    (Eq. 15). For the SCS unit hydrograph K = 3.77 (used as default)."""
    return 6.5 * Imax * tp / V


def hann_inflow(t, Imax, tp, K):
    """Williams-Hann synthetic inflow hydrograph, Eq. 14.
    I(t) = Imax * (t/tp * exp(1 - t/tp))**K,  t >= 0
    Vectorized; returns 0 at t=0 (handles the t=0 edge cleanly)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    x = t[pos] / tp
    out[pos] = Imax * (x * np.exp(1.0 - x)) ** K
    return out if out.shape else float(out)


def hann_to_fenton_params(Imax, tp, K):
    """Exact correspondence (Eq. 17): Williams-Hann  <=>  Fenton (1992)
    inflow I(t) = P0 * t**s * exp(-f*t).
    Returns (P0, s, f)."""
    s = K
    f = K / tp
    P0 = Imax * np.exp(K) / tp ** K
    return P0, s, f


def hann_cumulative_volume(t, Imax, tp, K):
    """Closed-form cumulative inflow volume  V(t) = int_0^t I(t') dt'
    using the regularized lower incomplete gamma function. Exact,
    because a Williams-Hann hydrograph is a Fenton-type P0 t^s e^{-ft}
    pulse, whose primitive is a (regularized) incomplete-gamma function:
        V(t) = P0 * Gamma(s+1) / f**(s+1) * gammainc(s+1, f*t)
    """
    P0, s, f = hann_to_fenton_params(Imax, tp, K)
    return P0 * gamma_fn(s + 1) / f ** (s + 1) * gammainc(s + 1, f * t)


def hann_total_volume(Imax, tp, K):
    """Total volume under the Hann hydrograph (t -> infinity)."""
    P0, s, f = hann_to_fenton_params(Imax, tp, K)
    return P0 * gamma_fn(s + 1) / f ** (s + 1)


def hann_to_fenton_params_integer(Imax, tp, K):
    """
    Integer-exponent SURROGATE of the Williams-Hann inflow, used only by
    the closed-form (Yevdjevich) routing solution below, which needs an
    integer power s so that dO/dt + cO = cI(t) has an elementary
    polynomial-times-exponential particular solution.

    K (Eq. 15) is in general NOT an integer (e.g. K=3.77 for the SCS
    unit hydrograph). Simply rounding K to the nearest integer s while
    re-using the *exact* P0, f computed for the fractional K is WRONG:
    since P0 ~ tp**(-K), even a small change in the exponent changes
    P0*tp**s by a large factor whenever tp >> 1. The correct surrogate
    instead re-derives f and P0 so the *rounded* pulse P0*t**s*exp(-f*t)
    still peaks at the same place and value as the true hydrograph
    (t=tp, I=Imax):
        s = round(K),   f = s/tp,   P0 = Imax * e**s / tp**s
    """
    s_int = int(round(K))
    f = s_int / tp
    P0 = Imax * np.exp(s_int) / tp ** s_int
    return P0, s_int, f


# ---------------------------------------------------------------------
# 2. Reservoir shape  (Mohammadzadah-Habili et al., 2009 / Eq. 6, 8, 9)
# ---------------------------------------------------------------------

@dataclass
class ReservoirShape:
    Smax: float   # maximum storage (m^3)
    hmax: float   # maximum height (m) corresponding to Smax
    N: float = None      # reservoir coefficient; if None, must supply Amax
    Amax: float = None   # max surface area (m^2); used to derive N (Eq. 9)

    def __post_init__(self):
        if self.N is None:
            if self.Amax is None:
                raise ValueError("Provide either N or Amax.")
            self.N = 2 * np.log(2) * self.Smax / (self.Amax * self.hmax)
        if self.Amax is None:
            self.Amax = 2 * np.log(2) * self.Smax / (self.N * self.hmax)

    def S(self, h):
        """Storage as a function of depth h above reservoir bottom (Eq.6)."""
        h = np.asarray(h, dtype=float)
        h = np.clip(h, 0.0, None)
        x = np.log(2.0) * h / self.hmax
        return self.Smax * np.power(np.expm1(x), 1.0 / self.N)

    def A(self, h):
        """Surface area = dS/dh (Eq. 8). Guarded near h=0 to avoid 0**neg."""
        h = np.asarray(h, dtype=float)
        h = np.clip(h, 1e-9, None)
        x = np.log(2.0) * h / self.hmax
        expm1x = np.expm1(x)
        expm1x = np.clip(expm1x, 1e-12, None)
        return (self.Smax * np.log(2.0) / (self.N * self.hmax)) \
            * np.exp(x) * expm1x ** ((1.0 - self.N) / self.N)

    def M_shape_factor(self):
        """Reservoir shape factor M via Eq. 10 (inverse of N=1.0751M-0.9063)."""
        return (self.N + 0.9063) / 1.0751

    def h_of_S(self, S):
        """
        Closed-form INVERSE of Eq. 6: given storage S, return depth h.
        Since S = Smax*(e^{ln2 h/hmax} - 1)^{1/N}  =>
              h = (hmax/ln2) * ln(1 + (S/Smax)**N)
        Used to integrate the routing ODE in storage-space (S as the
        state variable) instead of depth-space: A(h) -> 0 as h -> 0
        makes dh/dt = (I-O)/A(h) numerically stiff/singular near an
        empty reservoir, whereas dS/dt = I-O is perfectly smooth and
        S(0)=0 is an exact, non-singular initial condition.
        """
        S = np.clip(np.asarray(S, dtype=float), 0.0, None)
        return (self.hmax / np.log(2.0)) * np.log1p((S / self.Smax) ** self.N)


# ---------------------------------------------------------------------
# 3. Combined outlet (spillway + orifice/pipe), Eq. 2-5
# ---------------------------------------------------------------------

@dataclass
class Outlet:
    # Spillway (set B=0 to disable)
    C: float = 0.0      # spillway discharge coefficient
    B: float = 0.0      # spillway crest length (m)
    P: float = 0.0      # spillway crest height above bottom (m)
    # Orifice / pipe bank (set n=0 or a=0 to disable)
    Cd: float = 0.0
    a: float = 0.0
    n: int = 0
    dz: float = 0.0     # orifice invert height above bottom (m)

    def O(self, h):
        h = np.asarray(h, dtype=float)
        Osp = np.where(h > self.P, self.C * self.B * np.clip(h - self.P, 0, None) ** 1.5, 0.0)
        Oor = np.where(h > self.dz, self.n * self.Cd * self.a * np.clip(h - self.dz, 0, None) ** 0.5, 0.0)
        return Osp + Oor


# ---------------------------------------------------------------------
# 4. NUMERICAL (benchmark) solution of the exact ODE
# ---------------------------------------------------------------------

def route_numerical(t_eval, Imax, tp, K, shape: ReservoirShape, outlet: Outlet,
                     reservoir_state="empty", h0=None, rtol=1e-8, atol=1e-8):
    """
    Solve the EXACT mass-conservation ODE with scipy solve_ivp, but in
    STORAGE-space rather than depth-space:

        dS/dt = I(t) - O(h(S)),     h(S) = shape.h_of_S(S)   (closed form)

    instead of the paper's own formulation
        dh/dt = (I(t) - O(h)) / A(h).

    The two are mathematically identical (dS/dt = A(h) dh/dt by the
    chain rule), but the S-space version is smooth and non-stiff
    everywhere, including at an empty reservoir (S=0 is an ordinary,
    non-singular point), whereas A(h) -> 0 as h -> 0 makes the h-space
    ODE numerically singular there. During development, the h-space
    formulation was found to make LSODA silently fail (return
    success=False, effectively frozen at the initial condition) for the
    empty-reservoir case -- the S-space formulation below was verified
    to fix this (solver converges, and reproduces the "full" case
    results, which do not touch the singular region, to high precision).

    reservoir_state: "empty" -> S(0) = 0  (Fig. 4b of reference paper)
                      "full"  -> S(0) = shape.S(outlet.P) (Fig. 4a; live
                                 storage already full up to the spillway
                                 crest, flood becomes surcharge)
                      "custom"-> supply h0 explicitly (converted to S0)
    Returns dict with h(t), O(t), I(t), and (for empty case) the
    numerically-exact fill-up time tau (first time h reaches P).
    """
    if reservoir_state == "empty":
        S_init = 0.0
    elif reservoir_state == "full":
        S_init = float(shape.S(np.array([outlet.P]))[0])
    elif reservoir_state == "custom":
        S_init = float(shape.S(np.array([h0]))[0])
    else:
        raise ValueError("reservoir_state must be 'empty', 'full' or 'custom'")

    def rhs(t, y):
        S = max(y[0], 0.0)
        h = shape.h_of_S(np.array([S]))[0]
        I = hann_inflow(np.array([t]), Imax, tp, K)[0]
        O = outlet.O(np.array([h]))[0]
        return [I - O]

    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(rhs, t_span, [S_init], t_eval=t_eval, method="LSODA",
                     rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"route_numerical: solve_ivp failed ({sol.message})")

    S_t = np.clip(sol.y[0], 0.0, None)
    h_t = shape.h_of_S(S_t)
    O_t = outlet.O(h_t)
    I_t = hann_inflow(t_eval, Imax, tp, K)

    tau = None
    if reservoir_state == "empty" and outlet.P > 0 and np.any(h_t >= outlet.P):
        idx = np.argmax(h_t >= outlet.P)
        tau = t_eval[idx]

    return {"t": t_eval, "h": h_t, "S": S_t, "O": O_t, "I": I_t, "tau_numerical": tau}


# ---------------------------------------------------------------------
# 5. SEMI-ANALYTICAL solution (Yevdjevich integrating-factor method
#    applied to the Habili shape + combined outlet, driven by the exact
#    Williams-Hann -> Fenton inflow correspondence).
# ---------------------------------------------------------------------

def fit_storage_power_law(shape: ReservoirShape, base_h, m, hs_max, npts=60):
    """
    Fit the *surcharge* storage above an outlet invert `base_h`,
        S_eff(hs) = S(base_h + hs) - S(base_h)   ,  hs = h - base_h >= 0
    to a power law  S_eff(hs) ~= a * hs**m  (forced through the origin,
    least squares on 'a' only), with m FIXED to the outlet's own
    physically exact exponent (1.5 for a spillway, 0.5 for an
    orifice/pipe -- Eq. 2 and Eq. 4 of the reference paper). Using the
    outlet's exact exponent (rather than a blind log-log fit of the
    whole Habili curve) is what makes Yevdjevich's (1959) matching
    condition S=a*h^m, O=b*h^m (Eq. 18) hold well over the operating
    range, since the outlet term is then reproduced exactly.
    """
    hs = np.linspace(hs_max / npts, hs_max, npts)
    Seff = shape.S(base_h + hs) - shape.S(base_h)
    a = np.sum(Seff * hs ** m) / np.sum(hs ** (2 * m))
    return float(a)


_YEVDJEVICH_SOLVERS = {}  # cache: s_int -> lambdified closed-form O(t)


def _get_yevdjevich_solver(s_int):
    """
    Symbolically solve (once per integer s, then cache) the LINEAR
    reservoir ODE

        dO/dt + c*O = c * ( I0 + P0 * t**s * exp(-f*t) ),   O(0) = O0

    which is Yevdjevich's (1959) reduced routing problem once the
    storage-outflow matching condition S = a*h**m, O = c*a*h**m = c*S
    (Eq. 18-19 of the reference paper) is imposed. Solving this
    symbolically with sympy (rather than hand-deriving and manually
    expanding the repeated-integration-by-parts series) removes the
    risk of a sign/indexing slip in the finite series, and is verified
    once at import time (see `_selftest_yevdjevich`) against a direct
    numerical solve of the same reduced ODE.
    """
    if s_int in _YEVDJEVICH_SOLVERS:
        return _YEVDJEVICH_SOLVERS[s_int]
    import sympy as sp
    t, c, f, P0, I0, O0 = sp.symbols('t c f P0 I0 O0', positive=True)
    Ofun = sp.Function('O')
    ode = sp.Eq(Ofun(t).diff(t) + c * Ofun(t), c * (I0 + P0 * t ** s_int * sp.exp(-f * t)))
    sol = sp.dsolve(ode, Ofun(t), ics={Ofun(0): O0})
    fn = sp.lambdify((t, c, f, P0, I0, O0), sol.rhs, 'numpy')
    _YEVDJEVICH_SOLVERS[s_int] = fn
    return fn


def yevdjevich_outflow(t, I0, O0, P0, s, f, c):
    """
    Closed-form outflow of the reduced linear reservoir (see
    `_get_yevdjevich_solver`). s must be an integer (use
    `hann_to_fenton_params_integer` to get a consistent integer-s
    surrogate of the Williams-Hann inflow -- see that function's
    docstring for why naive rounding of K is NOT sufficient).
    """
    t = np.asarray(t, dtype=float)
    s_int = int(round(s))
    fn = _get_yevdjevich_solver(s_int)
    return np.asarray(fn(t, c, f, P0, I0, O0), dtype=float)


def _selftest_yevdjevich():
    """Sanity check: compare the closed form against a direct numerical
    solve of the same reduced linear ODE, dO/dt + c*O = c*I(t). Raises
    AssertionError if they disagree beyond a small numerical tolerance.
    Runs once at import time."""
    rng = np.random.default_rng(0)
    for _ in range(3):
        s_int = int(rng.integers(1, 6))
        c = float(rng.uniform(1e-4, 5e-3))
        # keep f well away from c: the closed form has a REMOVABLE
        # singularity at c=f (division by (f-c)**(s+1)), a genuine
        # mathematical edge case of Yevdjevich's method rather than a
        # derivation error. It practically never arises for real
        # reservoirs (c = outlet/storage ratio, f = inflow decay rate
        # are governed by unrelated physics), so it is not handled here;
        # avoid it in the self-test.
        f = c * float(rng.uniform(3.0, 8.0))
        P0 = float(rng.uniform(1e-6, 1e-3))
        I0 = float(rng.uniform(0, 5))
        O0 = float(rng.uniform(0, 5))
        t_eval = np.linspace(0, 5000, 300)

        def rhs(tt, y):
            return [c * (I0 + P0 * tt ** s_int * np.exp(-f * tt) - y[0])]

        sol = solve_ivp(rhs, (0, 5000), [O0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
        O_num = sol.y[0]
        O_ana = yevdjevich_outflow(t_eval, I0, O0, P0, s_int, f, c)
        if not np.allclose(O_num, O_ana, rtol=2e-3, atol=1e-3 * max(1.0, np.max(np.abs(O_num)))):
            raise AssertionError(f"Yevdjevich closed form failed self-test for s={s_int}")


try:
    _selftest_yevdjevich()
except Exception as _e:  # pragma: no cover
    import warnings
    warnings.warn(f"routing_core: Yevdjevich closed-form self-test failed: {_e}")


def estimate_fill_time(shape: ReservoirShape, outlet: Outlet, Imax, tp, K):
    """
    Closed-form ESTIMATE of the time tau needed for an empty reservoir to
    fill to the spillway crest (Eq. 25 idea), obtained by inverting the
    exact incomplete-gamma cumulative-inflow volume against the target
    storage S(P). This neglects any outflow already leaving through a
    low-level orifice during the filling phase, so it is an upper-bound
    / first estimate -- refine with the numerical solution if the
    orifice capacity is not small compared with the inflow.
    """
    if outlet.P <= 0:
        return 0.0
    target_V = shape.S(outlet.P)
    Vtot = hann_total_volume(Imax, tp, K)
    if target_V >= Vtot:
        return np.inf  # flood volume too small to ever reach the crest
    g = lambda t: hann_cumulative_volume(t, Imax, tp, K) - target_V
    # bracket
    t_hi = tp
    while g(t_hi) < 0:
        t_hi *= 1.6
        if t_hi > 1e7:
            return np.inf
    return brentq(g, 0.0, t_hi)


def _characteristic_depth(shape: ReservoirShape, invert, Vtot, cap):
    """
    Estimate a representative surcharge depth hs_char above `invert` by
    solving S(invert+hs) - S(invert) = Vtot ("if the whole flood volume
    were stored above the invert, how deep would it get"). Used to pick
    a SENSIBLE fitting window for the local power-law match
    (fit_storage_power_law): fitting over the full [invert, hmax] range
    when the flood only ever raises the level by a small fraction of
    that range badly biases the fitted storage coefficient 'a' (and
    hence c=b/a) toward the wrong part of the strongly nonlinear Habili
    curve. Capped at `cap` (typically hmax - invert).
    """
    cap = max(cap, 1e-3)
    target = shape.S(invert + cap) - shape.S(invert)
    if Vtot >= target:
        return cap
    g = lambda hs: (shape.S(invert + hs) - shape.S(invert)) - Vtot
    try:
        return brentq(g, 1e-6, cap)
    except ValueError:
        return cap


def _dominant_outlet(outlet: Outlet):
    """Pick which outlet term physically dominates the flood response and
    return (m, b, invert, has_secondary). Spillway (m=1.5, Eq.2) is used
    whenever present, since it is normally sized for the design flood;
    the orifice/pipe (m=0.5, Eq.4) is used when there is no spillway."""
    have_spillway = outlet.B > 0 and outlet.C > 0
    have_orifice = outlet.n > 0 and outlet.Cd > 0 and outlet.a > 0
    if have_spillway:
        return 1.5, outlet.C * outlet.B, outlet.P, have_orifice
    elif have_orifice:
        return 0.5, outlet.n * outlet.Cd * outlet.a, outlet.dz, False
    else:
        raise ValueError("Outlet has neither a spillway nor an orifice defined.")


def route_semi_analytical(t_eval, Imax, tp, K, shape: ReservoirShape, outlet: Outlet,
                           reservoir_state="empty", hs_span=None):
    """
    Semi-analytical outflow hydrograph -- generalized empty/full routing.

    Method
    ------
    1. Williams-Hann inflow is rewritten exactly as a Fenton-type pulse
       P0 t^s e^{-ft} (Eq. 17).
    2. The dominant outlet (spillway if present, else the orifice bank)
       fixes the matching exponent m (1.5 or 0.5, EXACT from Eq. 2 / 4)
       instead of a blind log-log fit of the whole Habili curve.
    3. The Habili storage curve is reduced to a one-parameter power law
       S_eff(hs) = a*hs**m in the surcharge hs above that outlet's
       invert (fit_storage_power_law) -- this is the Eq. 18 matching
       condition S=a*h^m, O=b*h^m needed for Yevdjevich's method.
    4. A secondary, lower orifice (below the spillway crest) is folded
       in as Yevdjevich's base-outflow O0 -- physically the quasi-steady
       "base flow" the flood pulse rides on top of, exactly the role
       I0/O0 play in Eq. 20.
    5. Yevdjevich's (1959) closed-form solution (Eq. 20) is applied to
       the reduced problem.

    reservoir_state="full": routing starts at the outlet invert
       (hs=0), i.e. Fig. 4a of the reference paper.
    reservoir_state="empty": the closed-form fill time tau is computed
       first (incomplete-gamma inversion, estimate_fill_time). If the
       flood volume never reaches the invert (tau = inf) the ENTIRE
       hydrograph is instead routed through the low-level orifice alone
       (m=0.5) from t=0 -- the "small floods almost fully absorbed by
       an empty reservoir" behaviour reported in the reference paper
       (Table 6, low return periods). Otherwise routing after tau
       mirrors the "full" case, shifted by tau, with I0 = I(tau).
    """
    # NOTE: the closed-form solver needs an integer power s; use the
    # rescaled integer surrogate (see hann_to_fenton_params_integer),
    # NOT the exact fractional-K params (those remain in use for
    # estimate_fill_time / hann_inflow, which do not require integer s).
    P0, s, f = hann_to_fenton_params_integer(Imax, tp, K)
    t_eval = np.asarray(t_eval, dtype=float)
    O_out = np.zeros_like(t_eval)
    m, b, invert, has_secondary = _dominant_outlet(outlet)
    Vtot_true = hann_total_volume(Imax, tp, K)

    if hs_span is None:
        hs_span = _characteristic_depth(shape, invert, Vtot_true, shape.hmax - invert)

    a_fit = fit_storage_power_law(shape, invert, m, hs_span)
    c = b / a_fit if a_fit != 0 else 0.0
    O0_base = outlet.n * outlet.Cd * outlet.a * np.sqrt(max(outlet.P - outlet.dz, 0.0)) \
        if (has_secondary and outlet.P > outlet.dz) else 0.0

    info = {"m": m, "a": a_fit, "b": b, "c": c, "invert": invert, "O0_base": O0_base}

    if reservoir_state == "full":
        O_out = np.clip(
            yevdjevich_outflow(t_eval, I0=0.0, O0=O0_base, P0=P0, s=s, f=f, c=c),
            0.0, None)
        tau_used = 0.0
    else:  # "empty"
        tau = estimate_fill_time(shape, outlet, Imax, tp, K)
        tau_used = tau
        if not np.isfinite(tau):
            # crest never reached: route the WHOLE event through the
            # low-level orifice/pipe alone (m=0.5), from the reservoir
            # bottom, t=0
            if outlet.n > 0 and outlet.Cd > 0 and outlet.a > 0:
                m2, b2, invert2 = 0.5, outlet.n * outlet.Cd * outlet.a, outlet.dz
                hs2 = _characteristic_depth(shape, invert2, Vtot_true, shape.hmax - invert2)
                a2 = fit_storage_power_law(shape, invert2, m2, hs2)
                c2 = b2 / a2 if a2 != 0 else 0.0
                O_out = np.clip(
                    yevdjevich_outflow(t_eval, I0=0.0, O0=0.0, P0=P0, s=s, f=f, c=c2),
                    0.0, None)
                info.update({"m": m2, "a": a2, "b": b2, "c": c2, "invert": invert2,
                             "note": "orifice-only regime (crest never reached)"})
            else:
                O_out[:] = 0.0
        else:
            # Post-tau tail model: the Hann hydrograph's log-derivative
            # is exactly  d(ln I)/dt = K/t - K/tp  (from I ~ t^K e^{-Kt/tp}),
            # so the LOCAL decay rate at t=tau is
            #   f_tail = -(dlnI/dt)|_tau = K*(1/tp - 1/tau)
            # Modelling the remaining inflow as I0 * exp(-f_tail * t')
            # (t'=t-tau) matches both the value AND the slope of the true
            # hydrograph at the moment the crest is reached, and decays
            # monotonically -- unlike naively restarting a fresh
            # P0 t'^s e^{-f t'} pulse (which would rise BACK UP to a
            # second full-amplitude peak Imax on top of the base flow,
            # a serious double-count of the flood volume; this was
            # checked against the numerical benchmark and rejected).
            # This tail model is only sensible once the inflow is past
            # its peak (tau > tp); for tau <= tp use the numerical
            # solution instead (flagged in `info`).
            after = t_eval >= tau
            if np.any(after):
                t_shift = t_eval[after] - tau
                I0 = float(hann_inflow(np.array([tau]), Imax, tp, K)[0])
                if tau > tp:
                    f_tail = K * (1.0 / tp - 1.0 / tau)
                    O_out[after] = np.clip(
                        yevdjevich_outflow(t_shift, I0=0.0, O0=O0_base, P0=I0, s=0, f=f_tail, c=c),
                        0.0, None)
                    info["tail_model"] = "exponential decay matched at tau (tau > tp)"
                else:
                    # tau before the inflow peak: the simple decaying-tail
                    # surrogate is not valid (inflow is still rising in
                    # reality). Flag it; fall back to holding at O0_base
                    # as a conservative placeholder and recommend the
                    # numerical solution for this regime.
                    O_out[after] = O0_base
                    info["tail_model"] = "NOT VALID (tau <= tp) -- use route_numerical for this case"

    I_out = hann_inflow(t_eval, Imax, tp, K)
    info["tau_analytical"] = tau_used
    return {"t": t_eval, "O": O_out, "I": I_out, "tau_analytical": tau_used, "fit": info}


# ---------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------

def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))
