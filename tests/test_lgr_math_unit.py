"""
Testes unitários isolados para o módulo lgr_math.py.
Cobre formatação, parsing, conversão simbólica, limites de plot,
cálculo de breakaway, Routh-Hurwitz e simulação de lugar das raízes.
"""

import numpy as np
import sympy as sp
import pytest

from lgr_math import (
    format_frac,
    format_complex_frac,
    parse_coeffs,
    format_poly_latex,
    format_factored_latex,
    poly_coeffs_to_sym,
    get_plot_limits,
    find_breakaway_points,
    calculate_breakaway_details,
    build_routh_hurwitz,
    simulate_root_locus,
)


def test_format_frac_integers_and_floats():
    assert format_frac(2.0) == "2"
    assert format_frac(-4.0) == "-4"
    assert format_frac(0.0) == "0"
    assert format_frac(116.565051) == "116.565"
    assert format_frac(-18.4349488) == "-18.4349"
    assert format_frac(float("inf")) == "\\infty"
    assert format_frac(float("-inf")) == "-\\infty"
    assert format_frac(float("nan")) == "NaN"


def test_format_complex_frac():
    assert format_complex_frac(3.0) == "3"
    assert format_complex_frac(0.0) == "0"
    assert format_complex_frac(1j) == "j"
    assert format_complex_frac(-1j) == "-j"
    assert format_complex_frac(2.5j) == "2.5j"
    assert format_complex_frac(-2.5j) == "-2.5j"
    assert format_complex_frac(1 + 1j) == "1 + j"
    assert format_complex_frac(-2 - 3j) == "-2 - 3j"
    assert format_complex_frac(0.5 - 1.25j) == "0.5 - 1.25j"


def test_parse_coeffs():
    assert parse_coeffs("1 4 0") == [1.0, 4.0, 0.0]
    assert parse_coeffs("  1,5   -2.25  3 ") == [1.5, -2.25, 3.0]
    assert parse_coeffs("") == []
    assert parse_coeffs("   ") == []
    assert parse_coeffs("abc") is None
    assert parse_coeffs("1 2 invalid") is None


def test_format_poly_latex():
    assert format_poly_latex([]) == "0"
    assert format_poly_latex([1, 4, 3]) == "s^{2} + 4s + 3"
    assert format_poly_latex([1, 0, 0]) == "s^{2} + 0"
    assert format_poly_latex([-1, 2, -3]) == "-s^{2} + 2s - 3"
    assert format_poly_latex([5]) == "5"


def test_format_factored_latex():
    assert format_factored_latex(np.array([])) == "1"
    roots_real = np.array([0.0, -4.0])
    res = format_factored_latex(roots_real)
    assert "s" in res
    assert "(s + 4)" in res

    # Raízes repetidas
    repeated = np.array([-2.0, -2.0])
    res_rep = format_factored_latex(repeated)
    assert "(s + 2)^{2}" in res_rep


def test_poly_coeffs_to_sym():
    s = sp.Symbol("s")
    expr = poly_coeffs_to_sym([1, 4, 0], s)
    assert expr == s**2 + 4 * s

    expr_const = poly_coeffs_to_sym([5], s)
    assert expr_const == 5


def test_get_plot_limits():
    poles = np.array([0.0, -4.0])
    zeros = np.array([-2.0])
    extra = [complex(-2, 2)]
    xmin, xmax, ymin, ymax = get_plot_limits(poles, zeros, extra)
    assert xmin < -4.0
    assert xmax > 0.0
    assert ymin <= -1.0
    assert ymax >= 2.0


def test_find_breakaway_and_details():
    # G(s) = K / [s(s+4)] => D = [1, 4, 0], N = [1]
    # dK/ds = 0 => 2s + 4 = 0 => s = -2, K = 4
    details = calculate_breakaway_details([1], [1, 4, 0])
    assert details["is_constant_deriv"] is False
    assert len(details["candidates"]) >= 1

    valid = details["valid_points"]
    assert len(valid) == 1
    s_val, k_val = valid[0]
    assert np.isclose(s_val, -2.0)
    assert np.isclose(k_val, 4.0)

    # find_breakaway_points conveniência
    bp = find_breakaway_points([1], [1, 4, 0])
    assert len(bp) == 1
    assert np.isclose(bp[0][0], -2.0)


def test_build_routh_hurwitz():
    # Sistema estável com cruzamento crítico: G(s) = K / [s(s+1)(s+2)] = K / [s^3 + 3s^2 + 2s]
    # D(s) = [1, 3, 2, 0], N(s) = [1]
    # Linha s^1: (3*2 - 1*K)/3 = 0 => K_crit = 6, aux eq: 3s^2 + 6 = 0 => s^2 = -2 => w = sqrt(2)
    res = build_routh_hurwitz([1], [1, 3, 2, 0])
    assert res["degree"] == 3
    assert len(res["crossings_data"]) >= 1

    cd = res["crossings_data"][0]
    assert np.isclose(float(cd["k_crit"]), 6.0, atol=1e-2)
    assert any(np.isclose(w, np.sqrt(2), atol=1e-2) for w in cd["omegas"])


def test_simulate_root_locus():
    # 2 polos reais
    K_vec, all_roots = simulate_root_locus([1, 4, 0], [1], 2, 0)
    assert len(K_vec) > 10
    assert all_roots.shape[1] == 2
    # No ganho K ~ 0, raízes devem estar próximas dos polos 0 e -4
    init_roots = sorted(all_roots[0], key=lambda r: np.real(r))
    assert np.isclose(init_roots[0], -4.0, atol=1e-2)
    assert np.isclose(init_roots[1], 0.0, atol=1e-2)
