"""
LGR Math Core Module
Contains all the mathematical and control theory logic required to build the Root Locus (LGR).
"""

import numpy as np
import sympy as sp
from fractions import Fraction
from typing import List, Tuple, Dict, Any


def format_frac(val, tol=1e-5):
    import math
    try:
        fval = float(val)
        if math.isinf(fval):
            return "\\infty" if fval > 0 else "-\\infty"
        if math.isnan(fval):
            return "NaN"
        if abs(fval - round(fval)) < tol:
            return f"{int(round(fval))}"
        return f"{fval:.3g}"
    except Exception:
        pass
    try:
        if abs(val - round(val)) < tol:
            return f"{int(round(val))}"
        return f"{float(val):.3g}"
    except:
        return str(val)

def format_complex_frac(val, tol=1e-5):
    r = float(np.real(val))
    i = float(np.imag(val))

    if abs(i) < tol:
        return format_frac(r, tol)

    i_str = "j" if abs(abs(i) - 1.0) < tol else f"{abs(i):.3g}j"
    sign = "+" if i > 0 else "-"

    if abs(r) < tol:
        return f"{'-' if i < 0 else ''}{i_str}"
    return f"{format_frac(r, tol)} {sign} {i_str}"


def poly_coeffs_to_sym(coeffs: List[float], var_sym: sp.Symbol) -> sp.Expr:
    """Converts a polynomial coefficient list into a clean SymPy expression."""
    expr = sp.Integer(0)
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-12:
            continue
        try:
            frac = Fraction(float(c)).limit_denominator(10000)
            if abs(float(frac) - c) < 1e-6:
                c_sym = sp.Rational(frac.numerator, frac.denominator)
            else:
                c_sym = sp.Float(c)
        except Exception:
            c_sym = sp.Float(c)
        expr += c_sym * (var_sym**power)
    return expr


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


def calculate_breakaway_details(
    N_coeffs: List[float], D_coeffs: List[float]
) -> Dict[str, Any]:
    """Calculates all symbolic, polynomial, and numerical steps for breakaway/break-in points."""
    s_sym = sp.symbols("s")
    D_sym = poly_coeffs_to_sym(D_coeffs, s_sym)
    N_sym = poly_coeffs_to_sym(N_coeffs, s_sym)
    if N_sym == 0:
        N_sym = sp.Integer(1)

    D_der_sym = sp.diff(D_sym, s_sym)
    N_der_sym = sp.diff(N_sym, s_sym)

    # U(s) = D'(s)*N(s) - D(s)*N'(s)
    U_sym = sp.expand(D_der_sym * N_sym - D_sym * N_der_sym)

    # Simplified or monic polynomial for presentation:
    U_simp = U_sym
    U_content = sp.Integer(1)
    if U_sym != 0:
        try:
            poly_U = sp.Poly(U_sym, s_sym)
            if poly_U.degree() > 0:
                content, prim = sp.primitive(U_sym)
                # Ensure positive leading coefficient for presentation
                if sp.Poly(prim, s_sym).LC() < 0:
                    prim = -prim
                    content = -content
                U_simp = prim
                U_content = content
        except Exception:
            U_simp = U_sym

    D_poly = np.poly1d(D_coeffs)
    N_poly = np.poly1d(N_coeffs)
    D_der_np = np.polyder(D_poly)
    N_der_np = np.polyder(N_poly)

    eq_np = np.polysub(np.polymul(D_poly, N_der_np), np.polymul(N_poly, D_der_np))
    if eq_np.order > 0:
        roots = np.roots(eq_np)
        roots = sorted(roots, key=lambda r: (-np.real(r), -np.imag(r)))
    else:
        roots = np.array([])

    poles = np.roots(D_coeffs) if len(D_coeffs) > 1 else np.array([])
    zeros = np.roots(N_coeffs) if len(N_coeffs) > 1 else np.array([])
    real_crit = sorted(
        [float(np.real(p)) for p in poles if abs(np.imag(p)) < 1e-5]
        + [float(np.real(z)) for z in zeros if abs(np.imag(z)) < 1e-5],
        reverse=True,
    )

    # Second derivative polynomial:
    # dK/ds = - U(s) / N(s)^2
    # At U(s*) = 0: d2K/ds2 = - U'(s*) / [N(s*)]^2
    U_np = np.polysub(np.polymul(D_der_np, N_poly), np.polymul(D_poly, N_der_np))
    U_der_np = np.polyder(U_np)

    candidates = []
    valid_points = []

    for idx, r in enumerate(roots):
        is_r_real = abs(np.imag(r)) < 1e-5
        r_val = float(np.real(r)) if is_r_real else complex(r)

        N_val = complex(np.polyval(N_poly, r))
        D_val = complex(np.polyval(D_poly, r))

        if abs(N_val) < 1e-10:
            continue

        K_val = -D_val / N_val
        K_is_real = abs(np.imag(K_val)) < 1e-5
        K_real = float(np.real(K_val))
        K_imag = float(np.imag(K_val))

        U_der_val = complex(np.polyval(U_der_np, r))
        d2K_val = -U_der_val / (N_val**2)
        d2K_real = float(np.real(d2K_val)) if is_r_real else None

        count_right = 0
        on_real_lgr = False
        if is_r_real:
            count_right = sum(1 for c in real_crit if c > r_val)
            on_real_lgr = count_right % 2 == 1

        is_valid = False
        reason_invalid = ""
        classification = "invalid"

        if is_r_real:
            if not K_is_real:
                reason_invalid = (
                    "O ganho K possui parte imaginária não nula (não é um número real)."
                )
            elif K_real <= 0:
                reason_invalid = f"O ganho K = {format_frac(K_real)} é menor ou igual a zero (pertence ao LGR complementar, com K < 0)."
            elif not on_real_lgr:
                reason_invalid = f"O ponto não pertence aos segmentos do LGR no eixo real (possui {count_right} polos e zeros à direita, número par)."
            else:
                is_valid = True
                valid_points.append((r_val, K_real))
                if d2K_real < -1e-7:
                    classification = "breakaway"
                elif d2K_real > 1e-7:
                    classification = "breakin"
                else:
                    classification = "inflection"
        else:
            if K_is_real and K_real > 0:
                is_valid = True
                valid_points.append((r, K_real))
                classification = "complex"
            else:
                reason_invalid = f"O ganho K = {format_complex_frac(K_val)} não é um número real positivo."

        candidates.append(
            {
                "index": idx + 1,
                "s_val": r_val,
                "is_real": is_r_real,
                "D_val": D_val,
                "N_val": N_val,
                "K_val": K_val,
                "K_real": K_real,
                "K_imag": K_imag,
                "K_is_real": K_is_real,
                "is_valid": is_valid,
                "reason_invalid": reason_invalid,
                "classification": classification,
                "d2K_val": d2K_val,
                "d2K_real": d2K_real,
                "count_right": count_right,
                "on_real_lgr": on_real_lgr,
            }
        )

    return {
        "D_sym": D_sym,
        "N_sym": N_sym,
        "D_der_sym": D_der_sym,
        "N_der_sym": N_der_sym,
        "U_sym": U_sym,
        "U_simp": U_simp,
        "U_content": U_content,
        "poly_order": eq_np.order,
        "is_constant_deriv": eq_np.order == 0,
        "candidates": candidates,
        "valid_points": valid_points,
    }


def find_breakaway_points(
    N_coeffs: List[float], D_coeffs: List[float]
) -> List[Tuple[complex, float]]:
    """Calculates breakaway and break-in points for the root locus. Returns list of (s_val, K_val)."""
    details = calculate_breakaway_details(N_coeffs, D_coeffs)
    return details["valid_points"]


def build_routh_hurwitz(N_coeffs: List[float], D_coeffs: List[float]) -> Dict[str, Any]:
    """Generates the Routh-Hurwitz table and calculates imaginary axis crossing frequencies,
    including detailed step-by-step formulation of table elements, critical gain deduction,
    auxiliary equation resolution, and analytical proofs."""
    s_sym = sp.symbols("s")
    K_sym = sp.symbols("K")
    w_sym = sp.symbols("omega", real=True)
    D_sym = sum(c * s_sym ** (len(D_coeffs) - 1 - i) for i, c in enumerate(D_coeffs))
    N_sym = sum(c * s_sym ** (len(N_coeffs) - 1 - i) for i, c in enumerate(N_coeffs))
    char_eq = D_sym + K_sym * N_sym

    char_poly = sp.Poly(char_eq, s_sym)
    char_coeffs_sym = char_poly.all_coeffs()
    degree = len(char_coeffs_sym) - 1

    routh_table = []
    row0 = char_coeffs_sym[0::2]
    row1 = char_coeffs_sym[1::2]

    while len(row1) < len(row0):
        row1.append(sp.Integer(0))

    routh_table.append(row0)
    routh_table.append(row1)

    routh_steps = []
    for i in range(2, degree + 1):
        prev_row = routh_table[i - 1]
        prev_prev_row = routh_table[i - 2]
        power = degree - i

        new_row = []
        for j in range(len(prev_row) - 1):
            piv = prev_row[0]
            a12 = prev_prev_row[j + 1] if (j + 1) < len(prev_prev_row) else sp.Integer(0)
            a11 = prev_prev_row[0]
            a22 = prev_row[j + 1] if (j + 1) < len(prev_row) else sp.Integer(0)
            num = piv * a12 - a11 * a22
            den = piv
            if den != 0:
                val = sp.cancel(num / den)
            else:
                val = sp.Integer(0)
            new_row.append(val)

            routh_steps.append(
                {
                    "row_power": power,
                    "col_idx": j,
                    "pivot": piv,
                    "a11": a11,
                    "a12": a12,
                    "a22": a22,
                    "num": num,
                    "den": den,
                    "val": val,
                    "val_together": sp.together(val),
                }
            )

        new_row.append(sp.Integer(0))
        routh_table.append(new_row)

    first_col = []
    for i in range(degree + 1):
        power = degree - i
        expr = routh_table[i][0]
        first_col.append(
            {
                "power": power,
                "expr": expr,
                "expr_together": sp.together(expr),
                "has_K": expr.has(K_sym),
            }
        )

    poles = np.roots(D_coeffs) if len(D_coeffs) > 1 else np.array([])
    zeros = np.roots(N_coeffs) if len(N_coeffs) > 1 else np.array([])

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
                        c_r = complex(r)
                        if abs(c_r.real) < 1e-5 and c_r.imag > 0:
                            omega_vals.append(float(c_r.imag))
                            omega_list.append(float(c_r.imag))

                    if not omega_list:
                        sol_s = sp.solve(aux_eq, s_sym)
                        for r in sol_s:
                            c_r = complex(r)
                            if abs(c_r.real) < 1e-5 and c_r.imag > 0:
                                val_w = float(c_r.imag)
                                if val_w not in omega_list:
                                    omega_vals.append(val_w)
                                    omega_list.append(val_w)

                    if omega_list:
                        t_expr = sp.together(row_expr)
                        num_k, den_k = sp.fraction(t_expr)
                        k_solve_steps = {
                            "row_expr": row_expr,
                            "together": t_expr,
                            "num": num_k,
                            "den": den_k,
                            "k_crit": float(sol),
                            "k_crit_sym": sol,
                        }

                        aux_poly = sp.Poly(aux_eq, s_sym)
                        aux_deg = aux_poly.degree()
                        aux_coeffs = aux_poly.all_coeffs()
                        aux_solve_steps = {
                            "degree": aux_deg,
                            "coeffs": aux_coeffs,
                            "eq": aux_eq,
                        }

                        direct_proofs = []
                        vector_proofs = []
                        P_at_k = char_eq.subs(K_sym, sol)
                        P_at_k_poly = sp.Poly(P_at_k, s_sym)
                        p_coeffs = P_at_k_poly.all_coeffs()
                        p_deg = len(p_coeffs) - 1

                        for w in omega_list:
                            s0 = 1j * w
                            terms_breakdown = []
                            real_sum = 0.0
                            imag_sum = 0.0
                            for idx, c in enumerate(p_coeffs):
                                p_pow = p_deg - idx
                                term_val = complex(c) * (s0 ** p_pow)
                                real_sum += term_val.real
                                imag_sum += term_val.imag
                                terms_breakdown.append(
                                    {
                                        "coeff": float(c),
                                        "power": p_pow,
                                        "term_val": term_val,
                                    }
                                )

                            P_jw_sym = P_at_k.subs(s_sym, sp.I * w_sym).expand()
                            re_sym = sp.re(P_jw_sym)
                            im_sym = sp.im(P_jw_sym)

                            direct_proofs.append(
                                {
                                    "w": w,
                                    "P_at_k": P_at_k,
                                    "P_jw_sym": P_jw_sym,
                                    "re_sym": re_sym,
                                    "im_sym": im_sym,
                                    "terms_breakdown": terms_breakdown,
                                    "real_sum": real_sum,
                                    "imag_sum": imag_sum,
                                }
                            )

                            angles_p = [
                                float(np.degrees(np.angle(s0 - p))) for p in poles
                            ]
                            angles_z = [
                                float(np.degrees(np.angle(s0 - z))) for z in zeros
                            ]
                            sum_p = sum(angles_p)
                            sum_z = sum(angles_z)
                            phase_raw = sum_z - sum_p
                            phase_norm = (phase_raw + 180) % 360 - 180
                            dist_p = [float(abs(s0 - p)) for p in poles]
                            dist_z = [float(abs(s0 - z)) for z in zeros]
                            prod_p = float(np.prod(dist_p)) if dist_p else 1.0
                            prod_z = float(np.prod(dist_z)) if dist_z else 1.0
                            k_calc = prod_p / prod_z if prod_z != 0 else float("inf")

                            vector_proofs.append(
                                {
                                    "w": w,
                                    "angles_p": angles_p,
                                    "angles_z": angles_z,
                                    "sum_p": sum_p,
                                    "sum_z": sum_z,
                                    "phase_raw": phase_raw,
                                    "phase_norm": phase_norm,
                                    "dist_p": dist_p,
                                    "dist_z": dist_z,
                                    "prod_p": prod_p,
                                    "prod_z": prod_z,
                                    "k_calc": k_calc,
                                }
                            )

                        crossings_data.append(
                            {
                                "s_power": degree - i,
                                "row_expr": row_expr,
                                "k_crit": float(sol),
                                "k_crit_sym": sol,
                                "k_solve_steps": k_solve_steps,
                                "aux_power": power,
                                "aux_eq_sym": aux_eq_sym,
                                "aux_eq_sub": aux_eq,
                                "aux_solve_steps": aux_solve_steps,
                                "omegas": omega_list,
                                "direct_proofs": direct_proofs,
                                "vector_proofs": vector_proofs,
                            }
                        )
        except Exception:
            pass

    no_crossing_proof = None
    if not crossings_data:
        try:
            P_jw_sym = char_eq.subs(s_sym, sp.I * w_sym).expand()
            re_sym = sp.re(P_jw_sym)
            im_sym = sp.im(P_jw_sym)
            no_crossing_proof = {
                "char_eq": char_eq,
                "P_jw_sym": P_jw_sym,
                "re_sym": re_sym,
                "im_sym": im_sym,
            }
        except Exception:
            pass

    return {
        "table": routh_table,
        "crossings": omega_vals,
        "crossings_data": crossings_data,
        "degree": degree,
        "row0_len": len(row0),
        "char_eq": char_eq,
        "char_poly": char_poly,
        "routh_steps": routh_steps,
        "first_col": first_col,
        "no_crossing_proof": no_crossing_proof,
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


def calculate_departure_arrival_angles(
    poles: np.ndarray, zeros: np.ndarray
) -> Dict[str, Any]:
    """Calculates departure angles from complex poles and arrival angles at complex zeros,
    including detailed vector breakdown from every other singularity, step-by-step algebraic
    formula evaluation, and infinitesimal perturbation proofs."""
    poles = np.array(poles, dtype=complex)
    zeros = np.array(zeros, dtype=complex)

    complex_poles = []
    for p in poles:
        if abs(np.imag(p)) > 1e-5 and not any(np.isclose(p, cp) for cp in complex_poles):
            complex_poles.append(p)

    complex_zeros = []
    for z in zeros:
        if abs(np.imag(z)) > 1e-5 and not any(np.isclose(z, cz) for cz in complex_zeros):
            complex_zeros.append(z)

    pole_details = []
    for cp in complex_poles:
        m = sum(1 for p in poles if np.isclose(cp, p))
        other_poles = [p for p in poles if not np.isclose(cp, p)]

        vecs_p = []
        for p in other_poles:
            v = cp - p
            ang = float(np.degrees(np.angle(v)))
            vecs_p.append(
                {
                    "pole": p,
                    "vector": v,
                    "re": float(np.real(v)),
                    "im": float(np.imag(v)),
                    "mag": float(abs(v)),
                    "angle_deg": ang,
                }
            )

        vecs_z = []
        for z in zeros:
            v = cp - z
            ang = float(np.degrees(np.angle(v)))
            vecs_z.append(
                {
                    "zero": z,
                    "vector": v,
                    "re": float(np.real(v)),
                    "im": float(np.imag(v)),
                    "mag": float(abs(v)),
                    "angle_deg": ang,
                }
            )

        sum_p = sum(v["angle_deg"] for v in vecs_p)
        sum_z = sum(v["angle_deg"] for v in vecs_z)

        branches = []
        for q in range(m):
            raw = ((2 * q + 1) * 180.0 - sum_p + sum_z) / m
            norm = (raw + 180.0) % 360.0 - 180.0
            a360 = norm if norm >= 0 else norm + 360.0

            # Perturbation proof at s_test = cp + eps * exp(j * norm)
            eps = 1e-4
            s_test = cp + eps * np.exp(1j * np.radians(norm))
            phase_raw = sum(
                np.degrees(np.angle(s_test - z)) for z in zeros
            ) - sum(np.degrees(np.angle(s_test - p)) for p in poles)
            phase_norm = (phase_raw + 180.0) % 360.0 - 180.0
            dist_p = [abs(s_test - p) for p in poles]
            dist_z = [abs(s_test - z) for z in zeros]
            k_test = (
                float(np.prod(dist_p) / np.prod(dist_z))
                if dist_z
                else float(np.prod(dist_p))
            )
            is_valid = bool(
                np.isclose(abs(phase_norm), 180.0, atol=1.0) and k_test > 0
            )

            branches.append(
                {
                    "q": q,
                    "raw": raw,
                    "norm": norm,
                    "a360": a360,
                    "proof": {
                        "eps": eps,
                        "s_test": s_test,
                        "phase_norm": phase_norm,
                        "k_test": k_test,
                        "is_valid": is_valid,
                    },
                }
            )

        pole_details.append(
            {
                "pole": cp,
                "conj_pole": np.conj(cp),
                "multiplicity": m,
                "vecs_p": vecs_p,
                "vecs_z": vecs_z,
                "sum_p": sum_p,
                "sum_z": sum_z,
                "branches": branches,
            }
        )

    zero_details = []
    for cz in complex_zeros:
        m = sum(1 for z in zeros if np.isclose(cz, z))
        other_zeros = [z for z in zeros if not np.isclose(cz, z)]

        vecs_p = []
        for p in poles:
            v = cz - p
            ang = float(np.degrees(np.angle(v)))
            vecs_p.append(
                {
                    "pole": p,
                    "vector": v,
                    "re": float(np.real(v)),
                    "im": float(np.imag(v)),
                    "mag": float(abs(v)),
                    "angle_deg": ang,
                }
            )

        vecs_z = []
        for z in other_zeros:
            v = cz - z
            ang = float(np.degrees(np.angle(v)))
            vecs_z.append(
                {
                    "zero": z,
                    "vector": v,
                    "re": float(np.real(v)),
                    "im": float(np.imag(v)),
                    "mag": float(abs(v)),
                    "angle_deg": ang,
                }
            )

        sum_p = sum(v["angle_deg"] for v in vecs_p)
        sum_z = sum(v["angle_deg"] for v in vecs_z)

        branches = []
        for q in range(m):
            raw = ((2 * q + 1) * 180.0 + sum_p - sum_z) / m
            norm = (raw + 180.0) % 360.0 - 180.0
            a360 = norm if norm >= 0 else norm + 360.0

            # Perturbation proof at s_test = cz + eps * exp(j * norm)
            eps = 1e-4
            s_test = cz + eps * np.exp(1j * np.radians(norm))
            phase_raw = sum(
                np.degrees(np.angle(s_test - z)) for z in zeros
            ) - sum(np.degrees(np.angle(s_test - p)) for p in poles)
            phase_norm = (phase_raw + 180.0) % 360.0 - 180.0
            dist_p = [abs(s_test - p) for p in poles]
            dist_z = [abs(s_test - z) for z in zeros]
            k_test = (
                float(np.prod(dist_p) / np.prod(dist_z))
                if dist_z
                else float(np.prod(dist_p))
            )
            is_valid = bool(
                np.isclose(abs(phase_norm), 180.0, atol=1.0) and k_test > 0
            )

            branches.append(
                {
                    "q": q,
                    "raw": raw,
                    "norm": norm,
                    "a360": a360,
                    "proof": {
                        "eps": eps,
                        "s_test": s_test,
                        "phase_norm": phase_norm,
                        "k_test": k_test,
                        "is_valid": is_valid,
                    },
                }
            )

        zero_details.append(
            {
                "zero": cz,
                "conj_zero": np.conj(cz),
                "multiplicity": m,
                "vecs_p": vecs_p,
                "vecs_z": vecs_z,
                "sum_p": sum_p,
                "sum_z": sum_z,
                "branches": branches,
            }
        )

    return {
        "has_complex": bool(complex_poles or complex_zeros),
        "complex_poles": complex_poles,
        "complex_zeros": complex_zeros,
        "pole_details": pole_details,
        "zero_details": zero_details,
    }


def evaluate_test_point_details(
    s0: complex,
    poles: np.ndarray,
    zeros: np.ndarray,
    D_coeffs: List[float],
    N_coeffs: List[float],
    atol_deg: float = 5.0,
) -> Dict[str, Any]:
    """Calculates all pedagogical and mathematical details for a test point s0:
    - Vectors and angles from every open-loop pole and zero
    - Angle criterion summation and modulo normalization
    - Pertinence status to LGR and angular deficiency
    - Magnitude condition evaluation for gain K
    - Step-by-step characteristic polynomial substitution P(s0, K) = D(s0) + K*N(s0)
    - Analytical proof verification (residual check and required complex gain).
    """
    s0 = complex(s0)
    poles = np.array(poles, dtype=complex)
    zeros = np.array(zeros, dtype=complex)
    D_coeffs = [float(c) for c in D_coeffs]
    N_coeffs = [float(c) for c in N_coeffs]

    # Coincidence detection
    coincident_pole = None
    coincident_pole_idx = None
    for i, p in enumerate(poles):
        if np.isclose(s0, p, atol=1e-5):
            coincident_pole = p
            coincident_pole_idx = i + 1
            break

    coincident_zero = None
    coincident_zero_idx = None
    for j, z in enumerate(zeros):
        if np.isclose(s0, z, atol=1e-5):
            coincident_zero = z
            coincident_zero_idx = j + 1
            break

    vecs_p = []
    for i, p in enumerate(poles):
        v = s0 - p
        re = float(np.real(v))
        im = float(np.imag(v))
        dist = float(abs(v))
        ang = float(np.degrees(np.angle(v))) if dist > 1e-9 else 0.0
        ang_360 = (ang % 360.0 + 360.0) % 360.0
        vecs_p.append(
            {
                "index": i + 1,
                "pole": p,
                "vector": v,
                "delta_sigma": re,
                "delta_omega": im,
                "dist": dist,
                "angle_deg": ang,
                "angle_360": ang_360,
            }
        )

    vecs_z = []
    for j, z in enumerate(zeros):
        w = s0 - z
        re = float(np.real(w))
        im = float(np.imag(w))
        dist = float(abs(w))
        ang = float(np.degrees(np.angle(w))) if dist > 1e-9 else 0.0
        ang_360 = (ang % 360.0 + 360.0) % 360.0
        vecs_z.append(
            {
                "index": j + 1,
                "zero": z,
                "vector": w,
                "delta_sigma": re,
                "delta_omega": im,
                "dist": dist,
                "angle_deg": ang,
                "angle_360": ang_360,
            }
        )

    angles_p = [vp["angle_deg"] for vp in vecs_p]
    angles_z = [vz["angle_deg"] for vz in vecs_z]
    sum_p = sum(angles_p)
    sum_z = sum(angles_z)
    total_angle = sum_z - sum_p
    normalized_angle = (total_angle % 360.0 + 360.0) % 360.0
    angle_norm_180 = (total_angle + 180.0) % 360.0 - 180.0

    if coincident_pole is not None or coincident_zero is not None:
        is_lgr = True
    else:
        is_lgr = bool(
            np.isclose(normalized_angle, 180.0, atol=atol_deg)
            or np.isclose(abs(angle_norm_180), 180.0, atol=atol_deg)
        )

    # Angular deficiency: phi_def = (180 - total_angle) mod 360
    angular_deficiency_360 = (180.0 - total_angle) % 360.0
    angular_deficiency_signed = (
        angular_deficiency_360
        if angular_deficiency_360 <= 180.0
        else angular_deficiency_360 - 360.0
    )

    # Passo 12: Distances and Gain K
    dist_p = [vp["dist"] for vp in vecs_p]
    dist_z = [vz["dist"] for vz in vecs_z]
    prod_p = float(np.prod(dist_p)) if dist_p else 1.0
    prod_z = float(np.prod(dist_z)) if dist_z else 1.0

    d_lead = abs(D_coeffs[0]) if D_coeffs else 1.0
    n_lead = abs(N_coeffs[0]) if N_coeffs else 1.0
    scale_factor = (n_lead / d_lead) if d_lead != 0 else 1.0

    if coincident_pole is not None:
        K_val = 0.0
        K_geom = 0.0
    elif coincident_zero is not None:
        K_val = float("inf")
        K_geom = float("inf")
    else:
        K_geom = prod_p / prod_z if prod_z != 0 else float("inf")
        K_val = (
            (prod_p / (scale_factor * prod_z))
            if (scale_factor * prod_z != 0)
            else float("inf")
        )

    # Characteristic polynomial evaluation D(s0) and N(s0)
    deg_d = len(D_coeffs) - 1
    terms_d = []
    d_val_acc = 0.0 + 0.0j
    for i, c in enumerate(D_coeffs):
        power = deg_d - i
        s_pow = s0**power
        term_val = c * s_pow
        d_val_acc += term_val
        terms_d.append(
            {
                "coeff": float(c),
                "power": power,
                "s_pow": complex(s_pow),
                "term_val": complex(term_val),
            }
        )

    deg_n = len(N_coeffs) - 1
    terms_n = []
    n_val_acc = 0.0 + 0.0j
    for j, c in enumerate(N_coeffs):
        power = deg_n - j
        s_pow = s0**power
        term_val = c * s_pow
        n_val_acc += term_val
        terms_n.append(
            {
                "coeff": float(c),
                "power": power,
                "s_pow": complex(s_pow),
                "term_val": complex(term_val),
            }
        )

    D_s0 = d_val_acc
    N_s0 = n_val_acc

    # Characteristic polynomial P(s0, K) = D(s0) + K * N(s0)
    if np.isfinite(K_val):
        P_s0 = D_s0 + K_val * N_s0
        residual = abs(P_s0)
    else:
        P_s0 = complex(float("inf"), float("inf"))
        residual = float("inf")

    # Required K to make D(s0) + K * N(s0) = 0 => K_req = -D(s0) / N(s0)
    if abs(N_s0) > 1e-12:
        K_req = -D_s0 / N_s0
    else:
        K_req = None

    scale_mag = max(
        abs(D_s0),
        abs(K_val * N_s0) if np.isfinite(K_val) else 1.0,
        1.0,
    )
    rel_residual = residual / scale_mag if np.isfinite(residual) else float("inf")
    proof_verified = bool(
        residual < 1e-2 or (is_lgr and rel_residual < 0.05)
    )

    return {
        "s0": s0,
        "s0_real": float(np.real(s0)),
        "s0_imag": float(np.imag(s0)),
        "is_pole": coincident_pole is not None,
        "coincident_pole": coincident_pole,
        "coincident_pole_idx": coincident_pole_idx,
        "is_zero": coincident_zero is not None,
        "coincident_zero": coincident_zero,
        "coincident_zero_idx": coincident_zero_idx,
        "vecs_p": vecs_p,
        "vecs_z": vecs_z,
        "angles_p": angles_p,
        "angles_z": angles_z,
        "sum_p": sum_p,
        "sum_z": sum_z,
        "total_angle": total_angle,
        "normalized_angle": normalized_angle,
        "angle_norm_180": angle_norm_180,
        "is_lgr": is_lgr,
        "angular_deficiency_360": angular_deficiency_360,
        "angular_deficiency_signed": angular_deficiency_signed,
        "dist_p": dist_p,
        "dist_z": dist_z,
        "prod_p": prod_p,
        "prod_z": prod_z,
        "scale_factor": scale_factor,
        "K": K_val,
        "K_geom": K_geom,
        "D_coeffs": D_coeffs,
        "N_coeffs": N_coeffs,
        "terms_d": terms_d,
        "terms_n": terms_n,
        "D_s0": D_s0,
        "N_s0": N_s0,
        "P_s0": P_s0,
        "residual": residual,
        "proof_verified": proof_verified,
        "K_req": K_req,
    }
