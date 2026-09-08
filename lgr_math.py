"""
LGR Math Core Module
Contains all the mathematical and control theory logic required to build the Root Locus (LGR).
"""

import numpy as np
from fractions import Fraction

def format_frac(val, tol=1e-5):
    if abs(val - round(val)) < tol:
        return f"{int(round(val))}"
    frac = Fraction(float(val)).limit_denominator(1000)
    if frac.denominator == 1:
        return f"{frac.numerator}"
    if frac.numerator < 0:
        return f"-{abs(frac.numerator)}/{frac.denominator}"
    return f"{frac.numerator}/{frac.denominator}"

import sympy as sp
from typing import List, Tuple, Dict, Any


def parse_coeffs(text: str) -> List[float]:
    """Parses a space-separated string of numbers into a list of floats."""
    try:
        if not text.strip():
            return []
        return [float(x) for x in text.replace(",", ".").strip().split()]
    except ValueError:
        return None


def format_poly_latex(coeffs: List[float], var: str = "s") -> str:
    """Formats polynomial coefficients into a LaTeX string representation."""
    if len(coeffs) == 0:
        return "0"
    terms = []
    degree = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        if c == 0 and degree > 0:
            degree -= 1
            continue

        term = ""
        if c < 0:
            term += " - " if len(terms) > 0 else "-"
        else:
            term += " + " if len(terms) > 0 else ""

        c_abs = abs(c)
        if c_abs != 1 or degree == 0:
            term += f"{format_frac(c_abs)}"

        if degree > 0:
            if degree == 1:
                term += var
            else:
                term += f"{var}^{{{degree}}}"

        terms.append(term)
        degree -= 1

    if not terms:
        return "0"
    return "".join(terms).lstrip(" + ")


def format_factored_latex(roots: np.ndarray) -> str:
    """Formats a list of roots into a factored LaTeX polynomial string."""
    if len(roots) == 0:
        return "1"

    counts = {}
    for r in roots:
        r_rounded = np.round(r, 4)
        found = False
        for k in counts.keys():
            if np.isclose(r_rounded, k, atol=1e-4):
                counts[k] += 1
                found = True
                break
        if not found:
            counts[r_rounded] = 1

    terms = []
    for r, count in counts.items():
        if np.isreal(r):
            r_val = np.real(r)
            if r_val == 0:
                base = "s"
            elif r_val > 0:
                base = f"(s - {format_frac(r_val)})"
            else:
                base = f"(s + {format_frac(-r_val)})"
        else:
            real_part = np.real(r)
            imag_part = np.imag(r)
            sign = "+" if imag_part > 0 else "-"
            base = f"(s - ({format_frac(real_part)} {sign} {format_frac(abs(imag_part))}j))"

        if count > 1:
            terms.append(f"{base}^{{{count}}}")
        else:
            terms.append(base)

    return "".join(terms)


def get_plot_limits(
    poles: np.ndarray, zeros: np.ndarray, extra_points: List[complex] = None
) -> Tuple[float, float, float, float]:
    """Calculates appropriate spatial boundaries (xmin, xmax, ymin, ymax) for the plotting window."""
    points = (
        [np.real(p) + 1j * np.imag(p) for p in poles]
        + [np.real(z) + 1j * np.imag(z) for z in zeros]
        + (extra_points if extra_points else [])
    )

    if not points:
        return -5.0, 5.0, -5.0, 5.0

    x_coords = [np.real(p) for p in points]
    y_coords = [np.imag(p) for p in points]

    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)

    xspan = xmax - xmin
    yspan = ymax - ymin

    if xspan == 0:
        xmin -= 2
        xmax += 2
        xspan = 4
    if yspan == 0:
        ymin -= 2
        ymax += 2
        yspan = 4

    margin_x = max(1.0, xspan * 0.2)
    margin_y = max(1.0, yspan * 0.2)

    return xmin - margin_x, xmax + margin_x, ymin - margin_y, ymax + margin_y


def find_breakaway_points(
    N_coeffs: List[float], D_coeffs: List[float]
) -> List[Tuple[complex, float]]:
    """Calculates breakaway and break-in points for the root locus. Returns list of (s_val, K_val)."""
    D_poly = np.poly1d(D_coeffs)
    N_poly = np.poly1d(N_coeffs)
    D_der = np.polyder(D_poly)
    N_der = np.polyder(N_poly)

    eq = np.polysub(np.polymul(D_poly, N_der), np.polymul(N_poly, D_der))
    roots = np.roots(eq)

    valid_points = []
    for r in roots:
        N_val = np.polyval(N_poly, r)
        D_val = np.polyval(D_poly, r)
        if abs(N_val) < 1e-10:
            continue
        K_val = -D_val / N_val

        if np.isreal(r) and abs(np.imag(r)) < 1e-5:
            r_real = np.real(r)
            if np.isreal(K_val) and np.real(K_val) > 0:
                valid_points.append((r_real, np.real(K_val)))
        else:
            if abs(np.imag(K_val)) < 1e-5 and np.real(K_val) > 0:
                valid_points.append((r, np.real(K_val)))

    return valid_points


def build_routh_hurwitz(N_coeffs: List[float], D_coeffs: List[float]) -> Dict[str, Any]:
    """Generates the Routh-Hurwitz table and calculates imaginary axis crossing frequencies."""
    s_sym, K_sym = sp.symbols("s K")
    D_sym = sum(c * s_sym ** (len(D_coeffs) - 1 - i) for i, c in enumerate(D_coeffs))
    N_sym = sum(c * s_sym ** (len(N_coeffs) - 1 - i) for i, c in enumerate(N_coeffs))
    char_eq = D_sym + K_sym * N_sym

    char_poly = sp.Poly(char_eq, s_sym)
    char_coeffs_sym = char_poly.all_coeffs()
    degree = len(char_coeffs_sym) - 1

    routh_table = []
    row0 = char_coeffs_sym[0::2]
    row1 = char_coeffs_sym[1::2]

    if len(row1) < len(row0):
        row1.append(0)

    routh_table.append(row0)
    routh_table.append(row1)

    for i in range(2, degree + 1):
        prev_row = routh_table[i - 1]
        prev_prev_row = routh_table[i - 2]

        new_row = []
        for j in range(len(prev_row) - 1):
            num = (
                prev_row[0] * prev_prev_row[j + 1] - prev_prev_row[0] * prev_row[j + 1]
            )
            den = prev_row[0]
            if den != 0:
                val = sp.cancel(num / den)
            else:
                val = sp.Integer(0)
            new_row.append(val)

        new_row.append(sp.Integer(0))
        routh_table.append(new_row)

    omega_vals = []
    crossings_data = []
    for i in range(2, degree + 1):
        try:
            row_expr = routh_table[i][0]
            if row_expr == 0:
                continue
            solutions = sp.solve(row_expr, K_sym)
            for sol in solutions:
                if sol.is_real and sol > 0:
                    aux_row = routh_table[i - 1]
                    power = degree - (i - 1)
                    aux_eq = 0
                    aux_eq_sym = 0
                    for j, val in enumerate(aux_row):
                        if power - 2 * j >= 0:
                            aux_eq += val.subs(K_sym, sol) * s_sym ** (power - 2 * j)
                            aux_eq_sym += val * s_sym ** (power - 2 * j)

                    roots_aux = sp.roots(aux_eq, s_sym)
                    omega_list = []
                    for r, mult in roots_aux.items():
                        if abs(sp.re(r)) < 1e-5 and sp.im(r) > 0:
                            omega_vals.append(float(sp.im(r)))
                            omega_list.append(float(sp.im(r)))

                    if omega_list:
                        crossings_data.append(
                            {
                                "s_power": degree - i,
                                "row_expr": row_expr,
                                "k_crit": float(sol),
                                "aux_eq_sym": aux_eq_sym,
                                "aux_eq_sub": aux_eq,
                                "omegas": omega_list,
                            }
                        )
        except:
            pass

    return {
        "table": routh_table,
        "crossings": omega_vals,
        "crossings_data": crossings_data,
        "degree": degree,
        "row0_len": len(row0),
    }


def simulate_root_locus(
    D_coeffs: List[float], N_coeffs: List[float], nP: int, nZ: int, points: int = 2500, extra_K: List[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulates the positions of the roots for logarithmically spaced values of K.
    Returns (K_vec, all_roots)."""
    K_vec = np.logspace(-4, 4, points)
    K_vec = np.insert(K_vec, 0, 0)
    if extra_K:
        K_vec = np.concatenate([K_vec, extra_K])
    K_vec = np.sort(np.unique(K_vec))

    all_roots = []
    prev_r = None
    ls = max(nP, nZ)

    for k in K_vec:
        char_eq_coeffs = np.polyadd(D_coeffs, np.polymul(N_coeffs, [k]))
        r = np.roots(char_eq_coeffs)

        # Pad missing roots at infinity
        if len(r) < ls:
            r = np.concatenate((r, np.full(ls - len(r), np.nan + 1j * np.nan)))

        if prev_r is not None:
            idx = []
            r_copy = list(r)
            for pr in prev_r:
                if np.isnan(pr):
                    dists = [np.inf if not np.isnan(cr) else 0 for cr in r_copy]
                    if 0 in dists:
                        idx.append(r_copy.pop(np.argmin(dists)))
                    else:
                        dists = [abs(cr) if not np.isnan(cr) else 0 for cr in r_copy]
                        idx.append(r_copy.pop(np.argmax(dists)))
                else:
                    dists = [
                        abs(pr - cr) if not np.isnan(cr) else np.inf for cr in r_copy
                    ]
                    idx.append(r_copy.pop(np.argmin(dists)))
            r = np.array(idx)
        else:
            r = np.array(sorted(r, key=lambda x: (np.isnan(x), np.angle(x))))

        all_roots.append(r)
        prev_r = r

    return K_vec, np.array(all_roots)
