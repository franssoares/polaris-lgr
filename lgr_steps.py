"""
LGR Step-by-Step Educational Presentation Module
Contains rendering functions for the 12 classic Evans Root Locus steps
and the simplified mode items in Streamlit.
"""

import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.colors
from typing import Dict, Any, List, Optional

from lgr_math import (
    format_frac,
    format_complex_frac,
    format_poly_latex,
    format_factored_latex,
    simulate_root_locus,
)
from lgr_plots import (
    create_base_plot,
    add_poles_zeros_traces,
    plot_test_point_vectors,
    generate_fig7_asymptotes,
    generate_fig8_breakaway,
    generate_fig9_crossings,
    generate_fig10_angles,
)


def format_root(r: complex) -> str:
    """Formats a root using 6 significant digits without artificial early truncation."""
    return format_complex_frac(r)


def get_char_poly_latex(D_coeffs: List[float], N_coeffs: List[float]) -> str:
    """Constructs the characteristic polynomial string D(s) + K*N(s) = 0 in LaTeX format."""
    max_len = max(len(D_coeffs), len(N_coeffs))
    D_pad = np.pad(D_coeffs, (max_len - len(D_coeffs), 0))
    N_pad = np.pad(N_coeffs, (max_len - len(N_coeffs), 0))

    terms = []
    for i in range(max_len):
        power = max_len - 1 - i
        d_val = D_pad[i]
        n_val = N_pad[i]

        if d_val == 0 and n_val == 0:
            continue

        term_parts = []
        if d_val != 0:
            term_parts.append(f"{format_frac(d_val)}")
        if n_val != 0:
            sign = (
                "+"
                if n_val > 0 and d_val != 0
                else ("" if n_val > 0 else "-")
            )
            abs_n = abs(n_val)
            n_str = f"{format_frac(abs_n)}" if abs_n != 1 else ""
            term_parts.append(f"{sign}{n_str}K")

        term_coeff = "".join(term_parts).strip()
        if d_val != 0 and n_val != 0:
            term_coeff = f"({term_coeff})"

        if power == 0:
            terms.append(term_coeff)
        elif power == 1:
            if term_coeff == "1":
                terms.append("s")
            elif term_coeff == "-1":
                terms.append("-s")
            else:
                terms.append(f"{term_coeff}s")
        else:
            if term_coeff == "1":
                terms.append(f"s^{{{power}}}")
            elif term_coeff == "-1":
                terms.append(f"-s^{{{power}}}")
            else:
                terms.append(f"{term_coeff}s^{{{power}}}")

    if not terms:
        return "0"
    res = terms[0]
    for t in terms[1:]:
        if t.startswith("-"):
            res += f" - {t[1:]}"
        else:
            res += f" + {t}"
    return res


def get_block_latex(n_lat: str, d_lat: str, is_g: bool = False) -> str:
    """Formats block fractions for G(s) and H(s) in LaTeX notation."""
    if is_g:
        if n_lat == "1" and d_lat == "1":
            return "1"
        if d_lat == "1":
            return n_lat
        return r"\frac{" + n_lat + r"}{" + d_lat + r"}"
    else:
        if n_lat == "1" and d_lat == "1":
            return ""
        if d_lat == "1":
            return r"\cdot " + n_lat
        return r"\cdot \frac{" + n_lat + r"}{" + d_lat + r"}"


def format_eval_poly(poly_coeffs: List[float], s_val: complex) -> str:
    """Builds a LaTeX string showing substitution of s_val into a polynomial."""
    terms = []
    deg = len(poly_coeffs) - 1
    for i, c in enumerate(poly_coeffs):
        if c == 0:
            continue
        power = deg - i
        c_str = format_frac(c)
        if power == 0:
            terms.append(c_str)
        else:
            if power == 1:
                terms.append(f"({c_str})({format_complex_frac(s_val)})")
            else:
                terms.append(f"({c_str})({format_complex_frac(s_val)})^{{{power}}}")
    if not terms:
        return "0"
    return " + ".join(terms).replace("+ -", "- ")


def render_phasor_angles(
    details: List[Dict[str, Any]],
    is_pole: bool,
    poles: np.ndarray,
    zeros: np.ndarray,
) -> None:
    """Renders the step-by-step vector and angle math for departure/arrival angles."""
    if not details:
        return

    kind = "partida" if is_pole else "chegada"
    sing_type = "polos" if is_pole else "zeros"
    sing_char = "p" if is_pole else "z"
    other_char = "z" if is_pole else "p"
    theta_char = r"\theta_d" if is_pole else r"\theta_a"

    st.markdown(f"**Ângulos de {kind} ({sing_type} complexos)**")
    st.markdown("**Fórmula:**")

    if is_pole:
        st.latex(rf"{theta_char} = 180^\circ - \sum_{{j \neq k}} \angle (p_k - p_j) + \sum_j \angle (p_k - z_j)")
    else:
        st.latex(rf"{theta_char} = 180^\circ - \sum_{{j \neq k}} \angle (z_k - z_j) + \sum_j \angle (z_k - p_j)")

    for item in details:
        cp = item["pole"] if is_pole else item["zero"]
        m = item["multiplicity"]
        vecs_same = item["vecs_p"] if is_pole else item["vecs_z"]
        vecs_diff = item["vecs_z"] if is_pole else item["vecs_p"]
        sum_same = item["sum_p"] if is_pole else item["sum_z"]
        sum_diff = item["sum_z"] if is_pole else item["sum_p"]

        sorted_poles = sorted(poles, key=lambda x: (np.real(x), np.imag(x)))
        sorted_zeros = sorted(zeros, key=lambda x: (np.real(x), np.imag(x)))
        if is_pole:
            k_idx = next((i + 1 for i, p in enumerate(sorted_poles) if np.isclose(cp, p)), "k")
        else:
            k_idx = next((i + 1 for i, z in enumerate(sorted_zeros) if np.isclose(cp, z)), "k")

        cp_str = format_complex_frac(cp)
        st.markdown(f"**{'Polo' if is_pole else 'Zero'} ${sing_char}_{{{k_idx}}} = {cp_str}$:**")

        st.markdown(f"**Ângulos dos outros {sing_type}:**")
        if vecs_same:
            for v in vecs_same:
                orig = v["pole"] if is_pole else v["zero"]
                if is_pole:
                    j_idx = next((i + 1 for i, p in enumerate(sorted_poles) if np.isclose(orig, p)), "j")
                else:
                    j_idx = next((i + 1 for i, z in enumerate(sorted_zeros) if np.isclose(orig, z)), "j")

                c_calc = v["vector"]
                ang = v["angle_deg"]
                orig_str = format_complex_frac(orig)
                orig_disp = f"({orig_str})" if "-" in orig_str or "+" in orig_str else orig_str
                st.latex(
                    rf"\angle ({sing_char}_{{{k_idx}}} - {sing_char}_{{{j_idx}}}) = \angle ({cp_str} - {orig_disp}) = \angle ({format_complex_frac(c_calc)}) = {format_frac(ang)}^\circ"
                )
        else:
            st.markdown(f"Não há outros {sing_type}.")

        st.markdown(f"**Ângulos dos {'zeros' if is_pole else 'polos'}:**")
        if vecs_diff:
            for v in vecs_diff:
                orig = v["zero"] if is_pole else v["pole"]
                if is_pole:
                    j_idx = next((i + 1 for i, z in enumerate(sorted_zeros) if np.isclose(orig, z)), "j")
                else:
                    j_idx = next((i + 1 for i, p in enumerate(sorted_poles) if np.isclose(orig, p)), "j")

                c_calc = v["vector"]
                ang = v["angle_deg"]
                orig_str = format_complex_frac(orig)
                orig_disp = f"({orig_str})" if "-" in orig_str or "+" in orig_str else orig_str
                st.latex(
                    rf"\angle ({sing_char}_{{{k_idx}}} - {other_char}_{{{j_idx}}}) = \angle ({cp_str} - {orig_disp}) = \angle ({format_complex_frac(c_calc)}) = {format_frac(ang)}^\circ"
                )
        else:
            st.markdown(f"Não há {'zeros' if is_pole else 'polos'}.")

        st.markdown("**Somatórios:**")
        st.latex(rf"\sum \angle ({sing_char}_{{{k_idx}}} - {sing_char}_j) = {format_frac(sum_same)}^\circ")
        st.latex(rf"\sum \angle ({sing_char}_{{{k_idx}}} - {other_char}_j) = {format_frac(sum_diff)}^\circ")

        st.markdown("**Resultado:**")
        for b in item["branches"]:
            q = b["q"]
            norm = b["norm"]
            theta_sub = rf"\theta_{{d, {k_idx}}}" if is_pole else rf"\theta_{{a, {k_idx}}}"

            if m == 1:
                st.latex(
                    rf"{theta_sub} = 180^\circ - ({format_frac(sum_same)}^\circ) + ({format_frac(sum_diff)}^\circ) = {format_frac(norm)}^\circ"
                )
            else:
                st.latex(
                    rf"q = {q} \implies {theta_sub} = \frac{{180^\circ({2*q+1}) - ({format_frac(sum_same)}^\circ) + ({format_frac(sum_diff)}^\circ)}}{{{m}}} = {format_frac(norm)}^\circ"
                )

            st.markdown(rf"**Conjugado:** ${format_frac(-norm if norm != 0 else 0)}^\circ$")


def render_final_animated_lgr(
    valid_breakaway: List[Any],
    routh_result: Dict[str, Any],
    D_coeffs: List[float],
    N_coeffs: List[float],
    nP: int,
    nZ: int,
    poles: np.ndarray,
    zeros: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    sim_data: Optional[Any] = None,
) -> None:
    """Renders the interactive animated Plotly figure of the complete root locus."""
    if sim_data is not None:
        K_vec, all_roots = sim_data
    else:
        extra_K = []
        if valid_breakaway:
            extra_K.extend([float(k) for _, k in valid_breakaway if k > 0])
        crossings_data = routh_result.get("crossings_data", [])
        if crossings_data:
            extra_K.extend([float(data["k_crit"]) for data in crossings_data if data["k_crit"] > 0])
        K_vec, all_roots = simulate_root_locus(D_coeffs, N_coeffs, nP, nZ, extra_K=extra_K)

    fig_final = create_base_plot(
        poles,
        zeros,
        "Lugar Geométrico das Raízes (LGR) Completo",
        xmin,
        xmax,
        ymin,
        ymax,
    )

    palette = plotly.colors.qualitative.Plotly
    margin_x = (xmax - xmin) * 0.05
    margin_y = (ymax - ymin) * 0.05
    box_xmin, box_xmax = xmin + margin_x, xmax - margin_x
    box_ymin, box_ymax = ymin + margin_y, ymax - margin_y

    branches_x = []
    branches_y = []
    branch_trace_indices = []

    for i in range(all_roots.shape[1]):
        branch = all_roots[:, i]
        vx, vy = [], []
        hit_infinity_idx = None

        for pt in branch:
            if np.isnan(pt):
                continue
            x_p, y_p = np.real(pt), np.imag(pt)
            vx.append(x_p)
            vy.append(y_p)

            if hit_infinity_idx is None and (
                x_p < box_xmin or x_p > box_xmax or y_p < box_ymin or y_p > box_ymax
            ):
                hit_infinity_idx = len(vx) - 1

        branches_x.append(vx)
        branches_y.append(vy)

        c = palette[i % len(palette)]
        if len(vx) > 0:
            fig_final.add_trace(
                go.Scatter(
                    x=[vx[0]],
                    y=[vy[0]],
                    mode="lines",
                    line=dict(color=c, width=3),
                    name=f"Ramo {i+1}",
                )
            )
            branch_trace_indices.append(len(fig_final.data) - 1)

        if len(vx) > 10:
            mid = len(vx) // 4
            fig_final.add_annotation(
                x=vx[mid],
                y=vy[mid],
                ax=vx[mid - 1],
                ay=vy[mid - 1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=c,
            )

        if hit_infinity_idx is not None and hit_infinity_idx >= 1:
            fig_final.add_annotation(
                x=vx[hit_infinity_idx],
                y=vy[hit_infinity_idx],
                ax=vx[hit_infinity_idx - 1],
                ay=vy[hit_infinity_idx - 1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=c,
            )

    if nP != nZ:
        sigma_A = (np.sum(poles) - np.sum(zeros)) / (nP - nZ)
        angles_A = [(2 * q + 1) * 180 / abs(nP - nZ) for q in range(abs(nP - nZ))]
        length_max = max(2.0, max(xmax - xmin, ymax - ymin) * 0.8)
        fig_final.add_trace(
            go.Scatter(
                x=[np.real(sigma_A)],
                y=[0],
                mode="markers",
                marker=dict(symbol="square", size=8, color="orange"),
                name="Centroide",
            )
        )
        for angle in angles_A:
            rad = np.radians(angle)
            dx = length_max * np.cos(rad) * 0.5
            dy = length_max * np.sin(rad) * 0.5
            fig_final.add_trace(
                go.Scatter(
                    x=[np.real(sigma_A), np.real(sigma_A) + dx],
                    y=[0, dy],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dash"),
                    showlegend=False,
                )
            )

    init_x, init_y = [], []
    for i in range(all_roots.shape[1]):
        if branches_x[i]:
            init_x.append(branches_x[i][0])
            init_y.append(branches_y[i][0])

    fig_final.add_trace(
        go.Scatter(
            x=init_x,
            y=init_y,
            mode="markers",
            marker=dict(
                size=12,
                color="white",
                symbol="diamond",
                line=dict(width=2, color="#00b4d8"),
            ),
            name="Raízes Dinâmicas",
            hovertemplate="Raiz: %{x:.4f} + %{y:.4f}j<extra></extra>",
        )
    )

    dynamic_trace_idx = len(fig_final.data) - 1
    step_frames = max(1, len(K_vec) // 100)
    frames = []
    slider_steps = []

    for i in range(0, len(K_vec), step_frames):
        k_val = K_vec[i]
        frame_name = f"frame_{i}"
        frame_data = []

        for b_idx, trace_idx in enumerate(branch_trace_indices):
            limit = min(i + 1, len(branches_x[b_idx]))
            if limit > 0:
                frame_data.append(
                    go.Scatter(
                        x=branches_x[b_idx][:limit], y=branches_y[b_idx][:limit]
                    )
                )
            else:
                frame_data.append(go.Scatter(x=[], y=[]))

        frame_dyn_x, frame_dyn_y = [], []
        for b_idx in range(all_roots.shape[1]):
            limit = min(i, len(branches_x[b_idx]) - 1)
            if limit >= 0:
                frame_dyn_x.append(branches_x[b_idx][limit])
                frame_dyn_y.append(branches_y[b_idx][limit])

        frame_data.append(go.Scatter(x=frame_dyn_x, y=frame_dyn_y))
        frame_traces = branch_trace_indices + [dynamic_trace_idx]

        frames.append(
            go.Frame(data=frame_data, name=frame_name, traces=frame_traces)
        )

        slider_steps.append(
            {
                "args": [
                    [frame_name],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"{k_val:.2f}",
                "method": "animate",
            }
        )

    fig_final.frames = frames
    fig_final.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 60, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16, "color": "white"},
                    "prefix": "Ganho K = ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": slider_steps,
            }
        ],
        height=700,
    )

    st.plotly_chart(fig_final, width="stretch", config={"scrollZoom": True})


# ==============================================================================
# MODALIDADE SIMPLIFICADA (ITEM A e ITEM B)
# ==============================================================================

def render_item_a(
    data: Dict[str, Any],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    length_max: float,
) -> None:
    """Renders Item (a): Esboço do LGR in simplified view mode."""
    N_coeffs = data["N_coeffs"]
    D_coeffs = data["D_coeffs"]
    poles = data["poles"]
    zeros = data["zeros"]
    nP = data["nP"]
    nZ = data["nZ"]
    real_roots = data["real_roots"]
    segment_coords = data["segment_coords"]
    segments = data["segments"]
    sigma_A = data["sigma_A"]
    angles_A = data["angles_A"]
    breakaway_details = data["breakaway_details"]
    candidates = breakaway_details["candidates"]
    is_constant_deriv = breakaway_details["is_constant_deriv"]
    routh_result = data["routh_result"]
    crossings_data = data["crossings_data"]
    dep_arr = data["dep_arr_details"]
    valid_breakaway = data["valid_breakaway"]

    st.markdown("### ITEM (a): ESBOÇO DO LGR")
    st.divider()

    st.markdown("**1. Polinômio característico com K em evidência**")
    char_poly_str = get_char_poly_latex(D_coeffs, N_coeffs)
    num_str = format_poly_latex(N_coeffs)
    den_str = format_poly_latex(D_coeffs)
    st.latex(r"1 + G(s)H(s) = 1 + K \frac{" + num_str + r"}{" + den_str + r"} = 0 \Rightarrow " + char_poly_str + " = 0")
    st.divider()

    st.markdown("**2. Fatoração de P(s)**")
    num_fact = format_factored_latex(zeros)
    den_fact = format_factored_latex(poles)
    K_scale = N_coeffs[0] / D_coeffs[0] if D_coeffs[0] != 0 else 1.0
    K_str = f"{format_frac(K_scale)}" if K_scale != 1.0 else ""
    st.latex(r"P(s) = " + K_str + r"\frac{" + num_fact + r"}{" + den_fact + r"}")
    st.divider()

    st.markdown("**3. Polos e zeros de malha aberta**")
    p_list = [f"p_{{{i+1}}} = {format_root(p)}" for i, p in enumerate(sorted(poles, key=lambda x: (np.real(x), np.imag(x))))]
    z_list = [f"z_{{{i+1}}} = {format_root(z)}" for i, z in enumerate(sorted(zeros, key=lambda x: (np.real(x), np.imag(x))))]
    st.markdown("• Polos: " + (", ".join([f"${p}$" for p in p_list]) if p_list else "Nenhum"))
    st.markdown("• Zeros: " + (", ".join([f"${z}$" for z in z_list]) if z_list else "Nenhum"))
    st.divider()

    st.markdown("**4. Segmentos do eixo real**")
    st.markdown("Aplica-se à esquerda de um número ímpar de polos+zeros no eixo real.")
    fig_p34 = create_base_plot(poles, zeros, "Passos 3 e 4: Polos, Zeros e Eixo Real", xmin, xmax, ymin, ymax)
    for start, end in segment_coords:
        fig_p34.add_trace(go.Scatter(x=[start, end], y=[0, 0], mode="lines", line=dict(color="#00b4d8", width=5), name="LGR Real"))
        if end <= xmin:
            fig_p34.add_annotation(
                x=end, y=0, ax=start, ay=0, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=3, arrowcolor="#00b4d8"
            )
    st.markdown("• Segmentos válidos: " + (", ".join(segments) if segments else "Nenhum"))
    st.plotly_chart(fig_p34, width="stretch", config={"scrollZoom": True})
    st.divider()

    st.markdown("**5. Número de lugares separados (Ramos)**")
    ls = max(nP, nZ)
    st.latex(f"LS = \\max(n_P, n_Z) = \\max({nP}, {nZ}) = {ls}")
    st.divider()

    st.markdown("**6. Simetria**")
    st.markdown("O LGR é simétrico em relação ao eixo real.")
    st.divider()

    st.markdown("**7. Assíntotas**")
    if nP == nZ:
        st.markdown("• Não há assíntotas.")
    else:
        sum_p_str = " + ".join([f"({format_complex_frac(p)})" for p in poles]) or "0"
        sum_z_str = " + ".join([f"({format_complex_frac(z)})" for z in zeros]) or "0"
        sigma_a_val = format_frac(np.real(sigma_A)) if sigma_A is not None else "0"
        st.latex(f"\\sigma_A = \\frac{{\\sum p_i - \\sum z_i}}{{n_P - n_Z}} = \\frac{{[{sum_p_str}] - [{sum_z_str}]}}{{{nP} - {nZ}}} = {sigma_a_val}")
        for q, a in enumerate(angles_A):
            st.latex(rf"\theta_{{{q}}} = \frac{{180^\circ(2({q})+1)}}{{|n_P - n_Z|}} = {format_frac(a)}^\circ")
    fig7 = generate_fig7_asymptotes(poles, zeros, sigma_A, angles_A, xmin, xmax, ymin, ymax, length_max)
    st.plotly_chart(fig7, width="stretch", config={"scrollZoom": True})
    st.divider()

    st.markdown("**8. Pontos de Saída/Entrada**")
    D_sym = breakaway_details["D_sym"]
    N_sym = breakaway_details["N_sym"]
    D_der_sym = breakaway_details["D_der_sym"]
    N_der_sym = breakaway_details["N_der_sym"]
    U_simp = breakaway_details["U_simp"]

    d_latex = sp.latex(D_sym)
    n_latex = sp.latex(N_sym)
    if n_latex == "1":
        st.latex(rf"1) \quad K(s) = -P(s)^{{-1}} = -\left({d_latex}\right)")
        st.latex(r"2) \quad \frac{dK}{ds} = -D'(s) = 0 \implies D'(s) = 0")
    else:
        st.latex(rf"1) \quad K(s) = -P(s)^{{-1}} = -\frac{{{d_latex}}}{{{n_latex}}}")
        st.latex(r"2) \quad \frac{dK}{ds} = -\frac{D'(s)N(s) - D(s)N'(s)}{[N(s)]^2} = 0 \implies D'(s)N(s) - D(s)N'(s) = 0")
        st.latex(rf"({sp.latex(D_der_sym)})({n_latex}) - ({d_latex})({sp.latex(N_der_sym)}) = 0")

    st.markdown("Polinômio resultante e raízes:")
    if is_constant_deriv:
        st.latex(rf"{sp.latex(U_simp)} = 0 \implies \text{{Sem raízes (Não há candidatos)}}")
    else:
        st.latex(rf"{sp.latex(U_simp)} = 0")
        if not candidates:
            st.markdown("• Nenhum candidato encontrado.")
        else:
            for c in candidates:
                idx = c["index"]
                s_str = format_complex_frac(c["s_val"])
                K_str = format_complex_frac(c["K_val"])
                if c["is_valid"]:
                    if c["K_real"] < 1e-5:
                        st.markdown(f"• $s_{{{idx}}} = {s_str}$ ($K(s_{{{idx}}}) = {K_str}$) $\\to$ **Válido (partida trivial)**")
                    elif np.isinf(c["K_real"]) or c["K_real"] > 1e10:
                        st.markdown(f"• $s_{{{idx}}} = {s_str}$ ($K(s_{{{idx}}}) \\to \\infty$) $\\to$ **Válido (chegada trivial)**")
                    else:
                        st.markdown(f"• $s_{{{idx}}} = {s_str}$ ($K(s_{{{idx}}}) = {K_str}$) $\\to$ **Válido**")
                else:
                    st.markdown(f"• $s_{{{idx}}} = {s_str}$ ($K(s_{{{idx}}}) = {K_str}$) $\\to$ **Descartado ({c['reason_invalid']})**")
    st.divider()

    st.markdown("**9. Cruzamento com o eixo imaginário**")
    if not crossings_data:
        st.markdown("• Não cruza o eixo imaginário (para $K > 0$).")
    else:
        for cd in crossings_data:
            k_crit = cd["k_crit"]
            power = cd["s_power"]
            row_expr = cd["row_expr"]
            aux_eq_sub = cd["aux_eq_sub"]
            omegas = cd["omegas"]
            st.markdown(f"• Linha $s^{power}$: ganho crítico $K = {format_frac(k_crit)}$")
            st.latex(rf"\text{{Linha }} s^{{{power}}}: \quad {sp.latex(row_expr)} = 0 \implies K_{{crit}} = {format_frac(k_crit)}")
            st.latex(rf"\text{{Eq. Auxiliar }}(K={format_frac(k_crit)}): \quad A(s) = {sp.latex(sp.together(aux_eq_sub))} = 0")
            for w in omegas:
                st.markdown(f"• Raízes do cruzamento: $s = \\pm {format_frac(w)}j$")

    fig9 = generate_fig9_crossings(poles, zeros, crossings_data, xmin, xmax, ymin, ymax)
    st.plotly_chart(fig9, width="stretch", config={"scrollZoom": True})
    st.divider()

    st.markdown("**10. Ângulos de Partida e Chegada**")
    if not dep_arr.get("has_complex"):
        st.markdown("• Não há polos ou zeros complexos conjugados.")
        st.markdown("**Motivo:** Apenas singularidades com parte imaginária não nula demandam cálculo tangencial.")
        if nP > 0:
            p_str = ", ".join([f"{format_complex_frac(p)}" for p in poles])
            st.latex(rf"p_i \in \{{{p_str}\}} \implies \text{{Im}}(p_i) = 0 \quad \forall p_i")
        if nZ > 0:
            z_str = ", ".join([f"{format_complex_frac(z)}" for z in zeros])
            st.latex(rf"z_j \in \{{{z_str}\}} \implies \text{{Im}}(z_j) = 0 \quad \forall z_j")
    else:
        render_phasor_angles(dep_arr.get("pole_details", []), True, poles, zeros)
        render_phasor_angles(dep_arr.get("zero_details", []), False, poles, zeros)

    fig10 = generate_fig10_angles(poles, zeros, dep_arr.get('has_complex'), dep_arr.get('pole_details', []), dep_arr.get('zero_details', []), xmin, xmax, ymin, ymax)
    st.plotly_chart(fig10, width='stretch', config={'scrollZoom': True})

    st.markdown("**Esboço Final do LGR:**")
    render_final_animated_lgr(valid_breakaway, routh_result, D_coeffs, N_coeffs, nP, nZ, poles, zeros, xmin, xmax, ymin, ymax, sim_data=(data["K_vec"], data["all_roots"]))
    st.divider()


def render_item_b(
    data: Dict[str, Any],
    s0: complex,
    test_details: Dict[str, Any],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    """Renders Item (b): Teste de Ponto Candidato e Ganho K in simplified view mode."""
    poles = data["poles"]
    zeros = data["zeros"]
    N_coeffs = data["N_coeffs"]
    D_coeffs = data["D_coeffs"]

    st.markdown("### ITEM (b): TESTE DE PONTO CANDIDATO E GANHO K")
    s0_str = format_complex_frac(s0)
    s0_disp = f"({s0_str})" if "-" in s0_str or "+" in s0_str else s0_str

    st.markdown("**Condição de pertinência ao LGR:**")
    st.latex(r"\sum \angle(s_0 - z_j) - \sum \angle(s_0 - p_i) = \pm 180^\circ(2q+1)")
    st.markdown(f"**Ponto de teste:** $s_0 = {s0_str}$")

    st.markdown("**Ângulos dos polos ($\\theta_i$):**")
    if not test_details["vecs_p"]:
        st.markdown("*(Não há polos no sistema)*")
        st.latex(r"\sum \theta_i = 0.00^\circ")
    else:
        for vp in test_details["vecs_p"]:
            idx = vp["index"]
            p_str = format_complex_frac(vp["pole"])
            p_disp = f"({p_str})" if "-" in p_str or "+" in p_str else p_str
            v_str = format_complex_frac(vp["vector"])
            a_str = format_frac(vp["angle_deg"])
            st.latex(rf"\theta_{{{idx}}} = \angle(s_0 - p_{{{idx}}}) = \angle({s0_disp} - {p_disp}) = \angle({v_str}) = {a_str}^\circ")
        st.latex(rf"\sum \theta_i = {format_frac(test_details['sum_p'])}^\circ")

    st.markdown("**Ângulos dos zeros ($\\phi_j$):**")
    if not test_details["vecs_z"]:
        st.markdown("*(Não há zeros no sistema)*")
        st.latex(r"\sum \phi_j = 0.00^\circ")
    else:
        for vz in test_details["vecs_z"]:
            idx = vz["index"]
            z_str = format_complex_frac(vz["zero"])
            z_disp = f"({z_str})" if "-" in z_str or "+" in z_str else z_str
            v_str = format_complex_frac(vz["vector"])
            a_str = format_frac(vz["angle_deg"])
            st.latex(rf"\phi_{{{idx}}} = \angle(s_0 - z_{{{idx}}}) = \angle({s0_disp} - {z_disp}) = \angle({v_str}) = {a_str}^\circ")
        st.latex(rf"\sum \phi_j = {format_frac(test_details['sum_z'])}^\circ")

    st.markdown("**Avaliação:**")
    delta_theta = test_details['sum_p'] - test_details['sum_z']
    st.latex(rf"\Delta\theta = \sum \theta_i - \sum \phi_j = {format_frac(test_details['sum_p'])}^\circ - {format_frac(test_details['sum_z'])}^\circ = {format_frac(delta_theta)}^\circ")
    norm_angle = test_details['normalized_angle']
    st.markdown(f"Ângulo normalizado: **{format_frac(norm_angle)}°**")
    is_lgr = test_details["is_lgr"]
    if is_lgr:
        st.success(rf"O ponto pertence ao LGR ($\Delta\theta = {format_frac(delta_theta)}^\circ \approx \pm 180^\circ$)")
    else:
        st.error(rf"O ponto não pertence ao LGR ($\Delta\theta = {format_frac(delta_theta)}^\circ \neq \pm 180^\circ$)")

    st.divider()
    st.markdown("**Cálculo de K**")
    st.markdown("**Fórmula do critério de módulo:**")
    scale_factor = (abs(N_coeffs[0])/abs(D_coeffs[0])) if D_coeffs[0] != 0 else 1.0
    scale_str = f"{format_frac(scale_factor)} \\cdot " if scale_factor != 1.0 else ""
    st.latex(rf"K = \frac{{\prod |s_0 - p_i|}}{{{scale_str}\prod |s_0 - z_j|}}")
    st.markdown(f"**Ponto:** $s_0 = {s0_str}$")

    st.markdown("**Distâncias dos polos:**")
    if not test_details["vecs_p"]:
        st.markdown("*(Não há polos no sistema)*")
        prod_p_str = "1"
    else:
        d_p_strs = []
        for vp in test_details["vecs_p"]:
            idx = vp["index"]
            p_str = format_complex_frac(vp["pole"])
            p_disp = f"({p_str})" if "-" in p_str or "+" in p_str else p_str
            v_str = format_complex_frac(vp["vector"])
            d_str = format_frac(vp["dist"])
            d_p_strs.append(d_str)
            st.latex(rf"|s_0 - p_{{{idx}}}| = |{s0_disp} - {p_disp}| = |{v_str}| = {d_str}")
        st.markdown("**Produto das distâncias dos polos:**")
        prod_p_str = " \\cdot ".join(d_p_strs)
        st.latex(rf"\prod |s_0 - p_i| = {prod_p_str} = {format_frac(test_details['prod_p'])}")

    st.markdown("**Distâncias dos zeros:**")
    if not test_details["vecs_z"]:
        st.markdown("*(Não há zeros no sistema)*")
        prod_z_str = "1"
    else:
        d_z_strs = []
        for vz in test_details["vecs_z"]:
            idx = vz["index"]
            z_str = format_complex_frac(vz["zero"])
            z_disp = f"({z_str})" if "-" in z_str or "+" in z_str else z_str
            v_str = format_complex_frac(vz["vector"])
            d_str = format_frac(vz["dist"])
            d_z_strs.append(d_str)
            st.latex(rf"|s_0 - z_{{{idx}}}| = |{s0_disp} - {z_disp}| = |{v_str}| = {d_str}")
        st.markdown("**Produto das distâncias dos zeros:**")
        prod_z_str = " \\cdot ".join(d_z_strs)
        st.latex(rf"\prod |s_0 - z_j| = {prod_z_str} = {format_frac(test_details['prod_z'])}")

    st.markdown("**Resultado:**")
    K_val = test_details["K"]
    st.latex(rf"K = \frac{{{format_frac(test_details['prod_p'])}}}{{{scale_str}{format_frac(test_details['prod_z'])}}} = {format_frac(K_val)}")

    if is_lgr:
        st.success(f"O ponto pertence ao LGR. **K = {format_frac(K_val)}**")
    else:
        st.warning(f"O ponto não pertence ao LGR. **K = {format_frac(K_val)}** (valor de referência)")

    st.markdown("**Gráfico dos Vetores:**")
    fig_test = plot_test_point_vectors(poles, zeros, s0, test_details, xmin, xmax, ymin, ymax)
    st.plotly_chart(fig_test, width="stretch", config={"scrollZoom": True})


# ==============================================================================
# PASSOS COMPLETOS DO LGR (PASSOS 1 A 12)
# ==============================================================================

def render_step_1_char_eq(data: Dict[str, Any], ng: List[float], dg: List[float], nh: List[float], dh: List[float]) -> None:
    """Passo 1: Equação Característica."""
    N_coeffs = data["N_coeffs"]
    D_coeffs = data["D_coeffs"]
    ng_latex = format_poly_latex(ng if ng else [1])
    dg_latex = format_poly_latex(dg if dg else [1])
    nh_latex = format_poly_latex(nh if nh else [1])
    dh_latex = format_poly_latex(dh if dh else [1])
    num_str = format_poly_latex(N_coeffs)
    den_str = format_poly_latex(D_coeffs)

    st.markdown("**1. Função de Transferência de Malha Fechada (FTMF)**")
    st.markdown("A relação entre a saída e a entrada de um sistema em malha fechada com realimentação negativa é dada por:")
    st.latex(r"T(s) = \frac{G(s)}{1 + G(s)H(s)}")

    st.markdown("**2. Equação Característica e Estabilidade**")
    st.markdown("A estabilidade do sistema é determinada pelos polos da malha fechada, que são as raízes do denominador da FTMF igualado a zero:")
    st.latex(r"1 + G(s)H(s) = 0")

    st.markdown("**3. Substituição das Funções**")
    st.markdown("Substituindo os blocos $G(s)$ e $H(s)$ separadamente na equação característica:")
    g_latex = get_block_latex(ng_latex, dg_latex, True)
    h_latex = get_block_latex(nh_latex, dh_latex, False)
    h_str = rf"\underbrace{{{h_latex}}}_{{H(s)}}" if h_latex == "1" else rf"\underbrace{{\left( {h_latex} \right)}}_{{H(s)}}"
    g_str = rf"\underbrace{{\left( {g_latex} \right)}}_{{G(s)}}"
    st.latex(r"1 + " + g_str + r" \cdot " + h_str + r" = 0")

    st.markdown("**4. Multiplicação das Frações**")
    st.markdown("Agrupando as funções em uma única fração multiplicada (numerador com numerador, denominador com denominador):")
    n_mult = (
        rf"({ng_latex}) \cdot ({nh_latex})"
        if nh_latex != "1" and ng_latex != "1"
        else (ng_latex if nh_latex == "1" else nh_latex)
    )
    d_mult = (
        rf"({dg_latex}) \cdot ({dh_latex})"
        if dh_latex != "1" and dg_latex != "1"
        else (dg_latex if dh_latex == "1" else dh_latex)
    )

    if n_mult == "1" and d_mult == "1":
        st.latex(r"1 + K \cdot 1 = 0")
    elif d_mult == "1":
        st.latex(r"1 + K \left( " + n_mult + r" \right) = 0")
    else:
        st.latex(r"1 + K \frac{" + n_mult + r"}{" + d_mult + r"} = 0")

    st.markdown("**5. Separação de $P(s)$**")
    st.markdown("Agrupando a parte fixa do sistema consolidado como $P(s)$, podemos reescrever o termo de malha aberta como:")
    st.latex(r"G(s)H(s) = K \cdot P(s) = K \frac{" + num_str + r"}{" + den_str + r"}")

    st.markdown("**6. Obtenção da Equação Característica**")
    st.markdown(r"Portanto, obtemos a equação característica final na forma padrão do Lugar Geométrico das Raízes ($1 + K \cdot P(s) = 0$):")
    st.latex(r"\boxed{ 1 + K \frac{" + num_str + r"}{" + den_str + r"} = 0 }")

    char_poly_str = get_char_poly_latex(D_coeffs, N_coeffs)
    st.markdown(r"Ou, multiplicando toda a equação pelo denominador, obtemos o **polinômio característico** ($D(s) + K \cdot N(s) = 0$):")
    st.latex(r"\boxed{ " + char_poly_str + r" = 0 }")


def render_step_2_factored(data: Dict[str, Any]) -> None:
    """Passo 2: Forma fatorada de P(s)."""
    zeros = data["zeros"]
    poles = data["poles"]
    N_coeffs = data["N_coeffs"]
    D_coeffs = data["D_coeffs"]

    num_fact = format_factored_latex(zeros)
    den_fact = format_factored_latex(poles)
    K_scale = N_coeffs[0] / D_coeffs[0] if D_coeffs[0] != 0 else 1.0
    K_str = f"{format_frac(K_scale)}" if K_scale != 1.0 else ""
    st.latex(r"P(s) = " + K_str + r"\frac{" + num_fact + r"}{" + den_fact + r"}")

    z_list = [f"z_{{{i+1}}} = {format_root(z)}" for i, z in enumerate(sorted(zeros, key=lambda x: (np.real(x), np.imag(x))))]
    p_list = [f"p_{{{i+1}}} = {format_root(p)}" for i, p in enumerate(sorted(poles, key=lambda x: (np.real(x), np.imag(x))))]

    st.markdown("**Zeros da malha aberta:** " + (", ".join([f"${z}$" for z in z_list]) if z_list else "Nenhum zero finito."))
    st.markdown("**Polos da malha aberta:** " + (", ".join([f"${p}$" for p in p_list]) if p_list else "Nenhum polo finito."))


def render_step_3_poles_zeros(data: Dict[str, Any], xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Passo 3: Polos e zeros no plano s."""
    fig3 = create_base_plot(data["poles"], data["zeros"], "Polos e Zeros", xmin, xmax, ymin, ymax, show_labels=True)
    st.plotly_chart(fig3, width="stretch", config={"scrollZoom": True})


def render_step_4_real_axis(data: Dict[str, Any], xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Passo 4: Segmentos do eixo real."""
    poles = data["poles"]
    zeros = data["zeros"]
    segments = data["segments"]
    segment_coords = data["segment_coords"]

    fig4 = create_base_plot(poles, zeros, "Mapeamento no Eixo Real", xmin, xmax, ymin, ymax)
    for start, end in segment_coords:
        if end <= xmin:
            fig4.add_annotation(
                x=end, y=0, ax=start, ay=0, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=3, arrowcolor="#00b4d8"
            )

    if segments:
        st.markdown("**Segmentos válidos:** " + r" $\cup$ ".join(segments))
    else:
        st.markdown("Não há segmentos válidos no eixo real.")

    final_x = []
    final_y = []
    for start, end in segment_coords:
        final_x.extend([start, end, None])
        final_y.extend([0, 0, None])

    fig4.add_trace(go.Scatter(x=final_x, y=final_y, mode="lines", line=dict(color="#00b4d8", width=5), name="LGR Real", hoverinfo="skip"))
    fig4.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="line-ns", size=24, line=dict(color="yellow", width=4)), name="Varredura", showlegend=False))

    base_annotations = list(fig4.layout.annotations) if fig4.layout.annotations else []
    hidden_annotations = []
    for ann in base_annotations:
        hidden_ann = ann.to_plotly_json()
        hidden_ann["visible"] = False
        hidden_annotations.append(hidden_ann)

    frames = []
    x_scan_vals = np.linspace(xmax, xmin - (xmax - xmin) * 0.05, 80)
    lgr_trace_idx = len(fig4.data) - 2
    scanner_trace_idx = len(fig4.data) - 1

    for step_x in x_scan_vals:
        frame_x = []
        frame_y = []
        for start, end in segment_coords:
            if step_x <= start:
                curr_end = max(end, step_x)
                frame_x.extend([start, curr_end, None])
                frame_y.extend([0, 0, None])
            else:
                frame_x.extend([None, None, None])
                frame_y.extend([None, None, None])

        frames.append(
            go.Frame(
                data=[go.Scatter(x=frame_x, y=frame_y), go.Scatter(x=[step_x], y=[0])],
                layout=go.Layout(annotations=hidden_annotations),
                traces=[lgr_trace_idx, scanner_trace_idx],
            )
        )

    restored_annotations = []
    for ann in base_annotations:
        restored_ann = ann.to_plotly_json()
        restored_ann["visible"] = True
        restored_annotations.append(restored_ann)

    frames.append(
        go.Frame(
            data=[go.Scatter(x=final_x, y=final_y), go.Scatter(x=[None], y=[None])],
            layout=go.Layout(annotations=restored_annotations),
            traces=[lgr_trace_idx, scanner_trace_idx],
        )
    )

    fig4.frames = frames
    fig4.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]
    )
    st.plotly_chart(fig4, width="stretch", config={"scrollZoom": True})


def render_step_5_branches(data: Dict[str, Any]) -> None:
    """Passo 5: Número de lugares separados (ramos)."""
    nP = data["nP"]
    nZ = data["nZ"]
    st.markdown(r"Sendo $n_P$ o número de polos e $n_Z$ o número de zeros da malha aberta, temos:")
    st.markdown(rf"- $n_P = {nP}$")
    st.markdown(rf"- $n_Z = {nZ}$")
    st.markdown(r"O número de lugares separados (ramos do LGR) é dado por:")
    ls = max(nP, nZ)
    st.latex(r"LS = \max(n_P, n_Z)")
    st.latex(rf"LS = \max({nP}, {nZ}) = {ls}")


def render_step_6_symmetry() -> None:
    """Passo 6: Simetria."""
    st.markdown("O LGR é simétrico em relação ao eixo real.")


def render_step_7_asymptotas(data: Dict[str, Any], xmin: float, xmax: float, ymin: float, ymax: float, length_max: float) -> None:
    """Passo 7: Assíntotas."""
    nP = data["nP"]
    nZ = data["nZ"]
    poles = data["poles"]
    zeros = data["zeros"]
    sigma_A = data["sigma_A"]
    angles_A = data["angles_A"]

    if nP == nZ:
        st.markdown("Não há assíntotas.")
    else:
        st.markdown(r"**Centro das assíntotas ($\sigma_A$):**")
        sum_p_str = " + ".join([f"({format_complex_frac(p)})" for p in poles]) or "0"
        sum_z_str = " + ".join([f"({format_complex_frac(z)})" for z in zeros]) or "0"

        st.latex(r"\sigma_A = \frac{\sum p_i - \sum z_i}{n_P - n_Z}")
        st.latex(rf"\sigma_A = \frac{{[{sum_p_str}] - [{sum_z_str}]}}{{{nP} - {nZ}}} = {format_frac(np.real(sigma_A))}")

        st.markdown(r"**Ângulos das assíntotas ($\theta_k$):**")
        st.latex(r"\theta_k = \frac{(2k + 1) \cdot 180^\circ}{|n_P - n_Z|} \quad \text{para } k = 0, 1, \dots, |n_P - n_Z| - 1")
        for k, angle in enumerate(angles_A):
            st.latex(rf"\theta_{k} = \frac{{(2({k}) + 1) \cdot 180^\circ}}{{{abs(nP - nZ)}}} = {format_frac(angle)}^\circ")

        st.markdown(r"**Cruzamento das assíntotas com o eixo imaginário:**")
        st.markdown(
            r"A equação da reta é $y = \tan(\theta_k) \cdot (x - \sigma_A)$. Quando $x = 0$, temos $y_{cruzamento} = -\sigma_A \cdot \tan(\theta_k)$. Como a assíntota é uma semirreta que parte de $\sigma_A$, ela só cruza de fato se estiver apontando na direção do eixo imaginário."
        )

        for k, angle in enumerate(angles_A):
            if angle % 180 == 90:
                if np.real(sigma_A) == 0:
                    st.latex(rf"\theta_{k} = {format_frac(angle)}^\circ \implies \text{{Assíntota sobre o eixo imaginário}}")
                else:
                    st.latex(rf"\theta_{k} = {format_frac(angle)}^\circ \implies \text{{Assíntota paralela ao eixo imaginário (não cruza)}}")
            else:
                cos_val = np.cos(np.radians(angle))
                t_cross = -np.real(sigma_A) / cos_val if cos_val != 0 else -1
                if t_cross >= 0:
                    cross_y = -np.real(sigma_A) * np.tan(np.radians(angle))
                    if abs(cross_y) < 1e-5:
                        st.latex(rf"\theta_{k} = {format_frac(angle)}^\circ \implies \text{{Assíntota sobre o eixo real}}")
                    else:
                        st.latex(rf"\theta_{k} = {format_frac(angle)}^\circ \implies y_{{cruzamento}} = -({format_frac(np.real(sigma_A))}) \cdot \tan({format_frac(angle)}^\circ) = {format_frac(cross_y)}j")
                else:
                    st.latex(rf"\theta_{k} = {format_frac(angle)}^\circ \implies \text{{A semirreta se afasta do eixo imaginário (não cruza)}}")

        fig7 = generate_fig7_asymptotes(poles, zeros, sigma_A, angles_A, xmin, xmax, ymin, ymax, length_max)
        st.plotly_chart(fig7, width="stretch", config={"scrollZoom": True})


def render_step_8_breakaway(data: Dict[str, Any], xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Passo 8: Pontos de saída/entrada."""
    poles = data["poles"]
    zeros = data["zeros"]
    segment_coords = data["segment_coords"]
    breakaway_details = data["breakaway_details"]
    D_sym = breakaway_details["D_sym"]
    N_sym = breakaway_details["N_sym"]
    D_der_sym = breakaway_details["D_der_sym"]
    N_der_sym = breakaway_details["N_der_sym"]
    U_sym = breakaway_details["U_sym"]
    U_simp = breakaway_details["U_simp"]
    candidates = breakaway_details["candidates"]
    valid_points = breakaway_details["valid_points"]
    is_constant_deriv = breakaway_details["is_constant_deriv"]

    st.markdown(
        r"Os pontos de **saída** (*breakaway*) e de **entrada** (*break-in*) indicam as posições onde ramos do LGR se encontram e se bifurcam. "
        r"No eixo real, esses pontos correspondem a raízes múltiplas da equação característica $1 + K P(s) = 0$, o que coincide com os pontos críticos da função de ganho $K(s)$:"
    )
    st.latex(r"\frac{dK}{ds} = 0")

    st.markdown(r"**1. Expressão Analítica do Ganho $K(s)$:**")
    st.latex(r"K(s) = -\frac{D(s)}{N(s)}")
    d_latex = sp.latex(D_sym)
    n_latex = sp.latex(N_sym)
    if n_latex == "1":
        st.latex(rf"K(s) = -\left({d_latex}\right)")
    else:
        st.latex(rf"K(s) = -\frac{{{d_latex}}}{{{n_latex}}}")

    st.markdown(r"**2. Derivação de $K(s)$ em Relação a $s$ ($\frac{dK}{ds} = 0$):**")
    if n_latex == "1":
        st.latex(r"\frac{dK}{ds} = -D'(s) = 0 \implies D'(s) = 0")
        st.latex(rf"D'(s) = \frac{{d}}{{ds}}\left[ {d_latex} \right] = {sp.latex(D_der_sym)}")
        st.latex(rf"{sp.latex(D_der_sym)} = 0")
    else:
        st.latex(r"\frac{dK}{ds} = -\frac{D'(s) \cdot N(s) - D(s) \cdot N'(s)}{[N(s)]^2} = 0")
        st.latex(rf"D'(s) = \frac{{d}}{{ds}}\left[ {d_latex} \right] = {sp.latex(D_der_sym)}")
        st.latex(rf"N'(s) = \frac{{d}}{{ds}}\left[ {n_latex} \right] = {sp.latex(N_der_sym)}")
        st.latex(r"D'(s) \cdot N(s) - D(s) \cdot N'(s) = 0")
        st.latex(rf"\left({sp.latex(D_der_sym)}\right) \cdot \left({n_latex}\right) - \left({d_latex}\right) \cdot \left({sp.latex(N_der_sym)}\right) = 0")

    st.markdown(r"**3. Equação Polinomial Resultante:**")
    if is_constant_deriv:
        st.markdown(rf"A derivada resulta em uma constante não nula (${sp.latex(U_sym)} \neq 0$). Portanto, a equação $\frac{{dK}}{{ds}} = 0$ não possui soluções no plano finito.")
        st.info("Não existem pontos de saída ou de entrada para este sistema.")
    else:
        eq_expanded_str = f"{sp.latex(U_sym)} = 0"
        eq_simp_str = f"{sp.latex(U_simp)} = 0"
        if U_sym != U_simp and U_simp != 0:
            st.latex(rf"{eq_expanded_str} \iff {eq_simp_str}")
        else:
            st.latex(eq_expanded_str)

        st.markdown(r"**4. Raízes Candidatas ($\sigma_i$):**")
        for cand in candidates:
            idx = cand["index"]
            s_str = format_complex_frac(cand["s_val"])
            st.latex(rf"s_{{{idx}}} = {s_str}")

        st.markdown(r"**5. Análise e Validação de Cada Candidato:**")
        for cand in candidates:
            idx = cand["index"]
            s_str = format_complex_frac(cand["s_val"])
            is_r = cand["is_real"]
            d_val = cand["D_val"]
            n_val = cand["N_val"]
            k_val = cand["K_val"]
            k_real = cand["K_real"]
            k_is_r = cand["K_is_real"]
            d2k_r = cand["d2K_real"]
            is_v = cand["is_valid"]
            c_right = cand["count_right"]
            on_r = cand["on_real_lgr"]
            cls = cand["classification"]

            st.markdown("---")
            st.markdown(f"##### Candidato $s_{{{idx}}} = {s_str}$:")
            d_eval_str = format_complex_frac(d_val)
            n_eval_str = format_complex_frac(n_val)
            k_eval_str = format_complex_frac(k_val)

            if abs(n_val - 1.0) < 1e-5:
                st.latex(rf"K(s_{{{idx}}}) = -D({s_str}) = -\left({d_eval_str}\right) = {k_eval_str}")
            else:
                st.latex(rf"K(s_{{{idx}}}) = -\frac{{D({s_str})}}{{N({s_str})}} = -\frac{{{d_eval_str}}}{{{n_eval_str}}} = {k_eval_str}")

            if is_r:
                paridade = "número ímpar" if on_r else "número par"
                st.markdown(f"À direita de $s = {s_str}$ existem **{c_right}** polos e zeros reais ({paridade}).")
                if k_real > 0 and on_r:
                    st.markdown(rf"Como $K = {format_frac(k_real)} > 0$ e o ponto está em segmento com número ímpar de singularidades à direita, ele **pertence ao LGR**.")
                else:
                    st.markdown(rf"Como $K = {format_frac(k_real)} \le 0$, o ponto **NÃO pertence ao LGR direto**.")
            else:
                if k_is_r and k_real > 0:
                    st.markdown(rf"O ganho $K = {format_frac(k_real)}$ é real e positivo no plano complexo.")
                else:
                    st.markdown(rf"O ganho $K = {k_eval_str}$ não é um número real positivo, logo o ponto **NÃO pertence ao LGR**.")

            if is_v and is_r and d2k_r is not None:
                st.latex(rf"\frac{{d^2K}}{{ds^2}}\Bigg|_{{s = {s_str}}} = {format_frac(d2k_r)}")
                if cls == "breakaway":
                    st.markdown(rf"Como $\frac{{d^2K}}{{ds^2}} = {format_frac(d2k_r)} < 0$, a função $K(s)$ atinge um **máximo local** ao longo do eixo real $\implies$ **Ponto de Saída (*Breakaway*)**.")
                elif cls == "breakin":
                    st.markdown(rf"Como $\frac{{d^2K}}{{ds^2}} = {format_frac(d2k_r)} > 0$, a função $K(s)$ atinge um **mínimo local** ao longo do eixo real $\implies$ **Ponto de Entrada (*Break-in*)**.")

            if is_v:
                tipo_nome = "Ponto de Saída (Breakaway)" if cls == "breakaway" else ("Ponto de Entrada (Break-in)" if cls == "breakin" else "Ponto de Bifurcação")
                st.success(f"**Válido:** {tipo_nome} em $s = {s_str}$ com ganho $K = {format_frac(k_real)}$.")
            else:
                st.warning(f"**Descartado:** {cand['reason_invalid']}")

        st.markdown("---")
        st.markdown(r"**6. Resumo dos Pontos Válidos:**")
        if not valid_points:
            st.info("Nenhum ponto válido para $K > 0$ foi identificado.")
        else:
            summary_rows = []
            for cand in candidates:
                if cand["is_valid"]:
                    tipo = "Saída (*Breakaway*)" if cand["classification"] == "breakaway" else ("Entrada (*Break-in*)" if cand["classification"] == "breakin" else "Bifurcação Complexa")
                    s_f = format_complex_frac(cand["s_val"])
                    k_f = format_frac(cand["K_real"])
                    d2_f = (
                        f"{format_frac(cand['d2K_real'])} (máximo local)"
                        if cand["classification"] == "breakaway"
                        else (
                            f"{format_frac(cand['d2K_real'])} (mínimo local)"
                            if cand["classification"] == "breakin"
                            else "-"
                        )
                    )
                    summary_rows.append(f"| **{tipo}** | $s = {s_f}$ | $K = {k_f}$ | $\\frac{{d^2K}}{{ds^2}} = {d2_f}$ |")

            summary_table = "| Tipo | Coordenada ($s$) | Ganho ($K$) | Critério da 2ª Derivada |\n| :--- | :--- | :--- | :--- |\n" + "\n".join(summary_rows)
            st.markdown(summary_table)

        st.markdown("---")
        st.markdown(r"**7. Visualização Gráfica dos Pontos:**")
        tab_plane, tab_curve = st.tabs(["Localização no Plano s", "Curva de Ganho K(σ) no Eixo Real"])

        with tab_plane:
            fig8 = generate_fig8_breakaway(poles, zeros, segment_coords, candidates, xmin, xmax, ymin, ymax)
            st.plotly_chart(fig8, width="stretch", config={"scrollZoom": True})

        with tab_curve:
            real_valid = [c for c in candidates if c["is_valid"] and c["is_real"]]
            all_reals = [np.real(p) for p in poles if abs(np.imag(p)) < 1e-5] + [np.real(z) for z in zeros if abs(np.imag(z)) < 1e-5] + [c["s_val"] for c in real_valid]
            s_min = min(all_reals) - 1.5 if all_reals else xmin
            s_max = max(all_reals) + 1.5 if all_reals else xmax

            sigma_axis = np.linspace(s_min, s_max, 800)
            N_poly = np.poly1d(data["N_coeffs"])
            D_poly = np.poly1d(data["D_coeffs"])
            n_vals = N_poly(sigma_axis)
            d_vals = D_poly(sigma_axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                k_axis = np.where(np.abs(n_vals) < 1e-6, np.nan, -d_vals / n_vals)
                k_axis = np.where((k_axis >= -50) & (k_axis <= 150), k_axis, np.nan)

            fig_k = go.Figure()
            fig_k.add_trace(go.Scatter(x=sigma_axis, y=k_axis, mode="lines", name="K(σ) = -D(σ)/N(σ)", line=dict(color="#00b4d8", width=3)))
            for c in real_valid:
                s_c = float(c["s_val"])
                k_c = float(c["K_real"])
                cls_c = c["classification"]
                color_c = "#ff007f" if cls_c == "breakaway" else "#00ff88"
                name_c = "Breakaway" if cls_c == "breakaway" else "Break-in"
                fig_k.add_trace(go.Scatter(x=[s_c], y=[k_c], mode="markers+text", marker=dict(size=14, color=color_c, line=dict(width=2, color="white")), text=[f"{name_c}<br>s={format_frac(s_c)}<br>K={format_frac(k_c)}"], textposition="top center", name=f"{name_c} (s={format_frac(s_c)})"))

            fig_k.update_layout(
                title="Comportamento da Função K(σ) no Eixo Real",
                xaxis_title="Eixo Real (σ)",
                yaxis_title="Ganho K(σ)",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="white"),
                xaxis=dict(zeroline=True, zerolinecolor="rgba(255, 255, 255, 0.3)", showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)"),
                yaxis=dict(zeroline=True, zerolinecolor="rgba(255, 255, 255, 0.3)", showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)"),
                height=480,
            )
            st.plotly_chart(fig_k, width="stretch", config={"scrollZoom": True})


def render_step_9_routh_crossings(data: Dict[str, Any], xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Passo 9: Cruzamento com o eixo imaginário."""
    routh_result = data["routh_result"]
    crossings_data = data["crossings_data"]
    poles = data["poles"]
    zeros = data["zeros"]
    char_poly_str = get_char_poly_latex(data["D_coeffs"], data["N_coeffs"])

    st.markdown(
        r"O cruzamento dos ramos do LGR com o eixo imaginário ($s = \pm j\omega$) estabelece a fronteira entre a estabilidade assintótica e a instabilidade do sistema em malha fechada."
    )
    st.latex(char_poly_str + " = 0")

    st.markdown("---")
    st.markdown(r"**1. Construção e Formulação Algébrica da Tabela de Routh:**")
    routh_steps = routh_result.get("routh_steps", [])
    if routh_steps:
        with st.expander("Ver formulação detalhada de cada termo calculado da tabela", expanded=False):
            for step in routh_steps:
                p_pow = step["row_power"]
                c_idx = step["col_idx"]
                piv_str = sp.latex(sp.together(step["pivot"]))
                a11_str = sp.latex(sp.together(step["a11"]))
                a12_str = sp.latex(sp.together(step["a12"]))
                a22_str = sp.latex(sp.together(step["a22"]))
                val_str = sp.latex(sp.together(step["val"]))
                if step["a12"] != 0 or step["a22"] != 0 or step["val"] != 0 or c_idx == 0:
                    st.latex(rf"r_{{s^{{{p_pow}}}, \text{{Col }} {c_idx+1}}} = \frac{{({piv_str}) \cdot ({a12_str}) - ({a11_str}) \cdot ({a22_str})}}{{{piv_str}}} = {val_str}")

    routh_table = routh_result["table"]
    degree = routh_result["degree"]
    row0_len = routh_result["row0_len"]

    md_table = "| $s^i$ | " + " | ".join([f"Col {j+1}" for j in range(row0_len)]) + " |\n"
    md_table += "|" + "|".join(["---" for _ in range(row0_len + 1)]) + "|\n"
    for i in range(degree + 1):
        power = degree - i
        row_str = f"| $s^{power}$ | "
        row_str += " | ".join([f"${sp.latex(sp.together(val))}$" if val != 0 else "$0$" for val in routh_table[i]])
        row_str += " |\n"
        md_table += row_str
    st.markdown(md_table)

    st.markdown("---")
    st.markdown(r"**2. Análise da Primeira Coluna e Determinação do Ganho Crítico ($K_{crítico}$):**")
    if crossings_data:
        for idx_c, cd in enumerate(crossings_data):
            power = cd["s_power"]
            row_expr = cd["row_expr"]
            k_crit = cd["k_crit"]
            aux_power = cd.get("aux_power", power + 1)
            aux_eq_sym = cd["aux_eq_sym"]
            aux_eq_sub = cd["aux_eq_sub"]
            omegas = cd["omegas"]
            k_solve_steps = cd.get("k_solve_steps", {})
            direct_proofs = cd.get("direct_proofs", [])
            vector_proofs = cd.get("vector_proofs", [])

            st.markdown(f"##### Cruzamento #{idx_c + 1} (Linha $s^{power}$):")
            t_expr = k_solve_steps.get("together", sp.together(row_expr))
            num_k = k_solve_steps.get("num", t_expr)
            den_k = k_solve_steps.get("den", sp.Integer(1))

            if den_k != 1 and den_k != -1:
                st.latex(rf"{sp.latex(t_expr)} = 0 \iff \frac{{{sp.latex(num_k)}}}{{{sp.latex(den_k)}}} = 0")
                st.latex(rf"{sp.latex(num_k)} = 0 \implies K_{{crítico}} = {format_frac(k_crit)}")
            else:
                st.latex(rf"{sp.latex(t_expr)} = 0 \implies K_{{crítico}} = {format_frac(k_crit)}")

            st.success(rf"**Ganho Crítico:** $K_{{crítico}} = {format_frac(k_crit)}$")
            st.latex(rf"A(s) = {sp.latex(sp.together(aux_eq_sym))} = 0")
            st.latex(rf"A(s)\Big|_{{K = {format_frac(k_crit)}}} = {sp.latex(sp.together(aux_eq_sub))} = 0")

            for w in omegas:
                st.latex(rf"s^2 + {format_frac(w**2)} = 0 \implies s = \pm {format_frac(w)}j \implies \omega = {format_frac(w)} \text{{ rad/s}}")

            st.info(
                f"**Pontos de Cruzamento no Eixo Imaginário:** "
                + ", ".join([f"$s = \\pm {format_frac(w)}j$ ($\\omega = {format_frac(w)}$ rad/s)" for w in omegas])
                + f" para $K = {format_frac(k_crit)}$."
            )

            # Provas analíticas
            st.markdown("---")
            st.markdown(rf"#### Comprovação Matemática do Resultado (Prova Real):")
            for d_proof in direct_proofs:
                w_val = d_proof["w"]
                r_sum = d_proof["real_sum"]
                i_sum = d_proof["imag_sum"]
                st.latex(rf"P(\pm {format_frac(w_val)}j)\Big|_{{K = {format_frac(k_crit)}}} = {format_frac(r_sum)} + {format_frac(i_sum)}j \approx 0 \quad (\text{{COMPROVADO}})")

            for v_proof in vector_proofs:
                w_val = v_proof["w"]
                ph_norm = v_proof["phase_norm"]
                k_calc = v_proof["k_calc"]
                st.latex(rf"\angle G(j\omega)H(j\omega) = {format_frac(ph_norm)}^\circ \equiv \pm 180^\circ \quad \text{e} \quad K = {format_frac(k_calc)} = K_{{crítico}} \quad (\text{{COMPROVADO}})")
    else:
        st.markdown(r"**Nenhum cruzamento com o eixo imaginário foi identificado para $K > 0$.**")
        first_col = routh_result.get("first_col", [])
        for fc in first_col:
            p = fc["power"]
            expr = fc["expr"]
            has_k = fc["has_K"]
            expr_str = sp.latex(sp.together(expr))
            if not has_k:
                st.latex(rf"s^{{{p}}}: \quad {expr_str} > 0 \quad (\text{{termo constante positivo}})")
            else:
                st.latex(rf"s^{{{p}}}: \quad {expr_str} > 0 \quad (\text{{estritamente positivo para todo }} K > 0)")
        st.markdown("Pelo Teorema de Routh-Hurwitz, todos os polos permanecem no semiplano esquerdo (SPE).")

    fig9 = generate_fig9_crossings(poles, zeros, crossings_data, xmin, xmax, ymin, ymax)
    st.plotly_chart(fig9, width="stretch", config={"scrollZoom": True})


def render_step_10_departure_arrival(data: Dict[str, Any], xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Passo 10: Ângulos de partida/chegada."""
    poles = data["poles"]
    zeros = data["zeros"]
    dep_arr_details = data["dep_arr_details"]
    has_complex = dep_arr_details["has_complex"]
    pole_details = dep_arr_details["pole_details"]
    zero_details = dep_arr_details["zero_details"]

    st.markdown(
        r"Os **ângulos de partida** ($\theta_p$) e **ângulos de chegada** ($\theta_z$) determinam as direções angulares tangenciais com que os ramos do LGR emergem dos polos complexos ou incidem nos zeros complexos."
    )
    if not has_complex:
        st.divider()
        st.info("**Não aplicável:** O sistema não possui polos nem zeros complexos conjugados (Im != 0).")
    else:
        if pole_details:
            render_phasor_angles(pole_details, True, poles, zeros)
        if zero_details:
            st.divider()
            render_phasor_angles(zero_details, False, poles, zeros)

    st.divider()
    st.markdown(r"### Visualização Gráfica dos Vetores e Ângulos de Partida/Chegada:")
    fig10 = generate_fig10_angles(poles, zeros, has_complex, pole_details, zero_details, xmin, xmax, ymin, ymax)
    st.plotly_chart(fig10, width="stretch", config={"scrollZoom": True})


def render_step_11_angle_criterion(
    data: Dict[str, Any],
    s0: complex,
    test_details: Dict[str, Any],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    """Passo 11: Critério de ângulo (s0)."""
    poles = data["poles"]
    zeros = data["zeros"]
    s0_str = format_complex_frac(s0)
    is_lgr = test_details["is_lgr"]
    vecs_p = test_details["vecs_p"]
    vecs_z = test_details["vecs_z"]
    sum_p = test_details["sum_p"]
    sum_z = test_details["sum_z"]
    total_angle = test_details["total_angle"]
    norm_angle = test_details["normalized_angle"]
    angle_180 = test_details["angle_norm_180"]
    defic = test_details["angular_deficiency_signed"]
    is_pole = test_details["is_pole"]
    is_zero = test_details["is_zero"]

    st.markdown(
        r"O **Critério de Ângulo** determina se um ponto de teste genérico $s_0 \in \mathbb{C}$ pertence ou não ao Lugar Geométrico das Raízes direto ($K > 0$):"
    )
    st.latex(r"\angle G(s_0)H(s_0) = \sum_{j=1}^{nZ} \angle(s_0 - z_j) - \sum_{i=1}^{nP} \angle(s_0 - p_i) = \pm 180^\circ(2q+1)")
    st.markdown(rf"Ponto de teste selecionado: **$s_0 = {s0_str}$** ($\sigma_0 = {format_frac(np.real(s0))}$, $\omega_0 = {format_frac(np.imag(s0))}$).")

    st.divider()
    st.markdown(r"### 1. Vetores partindo dos Polos até $s_0$ ($\vec{v}_{p_i} = s_0 - p_i$):")
    if vecs_p:
        p_table = [
            "| Polo Origem ($p_i$) | Vetor $\\vec{v}_{p_i} = s_0 - p_i$ | Componentes $(\\Delta \\sigma, \\Delta \\omega)$ | Distância $\\lVert \\vec{v}_{p_i} \\rVert$ | Ângulo $\\theta_{p_i} = \\operatorname{atan2}(\\Delta \\omega, \\Delta \\sigma)$ |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for vp in vecs_p:
            p_orig = format_complex_frac(vp["pole"])
            v_str = format_complex_frac(vp["vector"])
            ds_str = format_frac(vp["delta_sigma"])
            dw_str = format_frac(vp["delta_omega"])
            mag_str = format_frac(vp["dist"])
            ang_str = rf"{format_frac(vp['angle_deg'])}^\circ"
            p_table.append(f"| $p_{{{vp['index']}}} = {p_orig}$ | ${v_str}$ | $\\Delta\\sigma = {ds_str}, \\Delta\\omega = {dw_str}$ | ${mag_str}$ | ${ang_str}$ |")
        st.markdown("\n".join(p_table))
        sum_p_terms = " + ".join([rf"({format_frac(vp['angle_deg'])}^\circ)" for vp in vecs_p])
        st.latex(rf"\sum_{{i=1}}^{{nP}} \theta_{{p_i}} = {sum_p_terms} = {format_frac(sum_p)}^\circ")
    else:
        st.markdown(r"*(Não há polos no sistema)* $\implies \sum \theta_p = 0^\circ$.")

    st.divider()
    st.markdown(r"### 2. Vetores partindo dos Zeros até $s_0$ ($\vec{w}_{z_j} = s_0 - z_j$):")
    if vecs_z:
        z_table = [
            "| Zero Origem ($z_j$) | Vetor $\\vec{w}_{z_j} = s_0 - z_j$ | Componentes $(\\Delta \\sigma, \\Delta \\omega)$ | Distância $\\lVert \\vec{w}_{z_j} \\rVert$ | Ângulo $\\phi_{z_j} = \\operatorname{atan2}(\\Delta \\omega, \\Delta \\sigma)$ |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for vz in vecs_z:
            z_orig = format_complex_frac(vz["zero"])
            w_str = format_complex_frac(vz["vector"])
            ds_str = format_frac(vz["delta_sigma"])
            dw_str = format_frac(vz["delta_omega"])
            mag_str = format_frac(vz["dist"])
            ang_str = rf"{format_frac(vz['angle_deg'])}^\circ"
            z_table.append(f"| $z_{{{vz['index']}}} = {z_orig}$ | ${w_str}$ | $\\Delta\\sigma = {ds_str}, \\Delta\\omega = {dw_str}$ | ${mag_str}$ | ${ang_str}$ |")
        st.markdown("\n".join(z_table))
        sum_z_terms = " + ".join([rf"({format_frac(vz['angle_deg'])}^\circ)" for vz in vecs_z])
        st.latex(rf"\sum_{{j=1}}^{{nZ}} \phi_{{z_j}} = {sum_z_terms} = {format_frac(sum_z)}^\circ")
    else:
        st.markdown(r"*(Não há zeros no sistema)* $\implies \sum \phi_z = 0^\circ$.")

    st.divider()
    st.markdown(r"### 3. Balanço Angular e Redução Trigonométrica:")
    st.latex(rf"\angle G(s_0)H(s_0) = \sum \phi_z - \sum \theta_p = ({format_frac(sum_z)}^\circ) - ({format_frac(sum_p)}^\circ) = {format_frac(total_angle)}^\circ")
    st.latex(rf"\angle G(s_0)H(s_0) \equiv {format_frac(norm_angle)}^\circ \pmod{{360^\circ}} \quad \left( \text{{ou }} {format_frac(angle_180)}^\circ \in (-180^\circ, 180^\circ] \right)")

    st.divider()
    st.markdown(r"### 4. Veredito de Pertinência ao LGR:")
    if is_pole:
        st.success(rf"**O ponto $s_0 = {s0_str}$ coincide com o polo $p_{{{test_details['coincident_pole_idx']}}}$.** Pertence ao LGR com $K = 0$.")
    elif is_zero:
        st.success(rf"**O ponto $s_0 = {s0_str}$ coincide com o zero $z_{{{test_details['coincident_zero_idx']}}}$.** Pertence ao LGR quando $K \to \infty$.")
    elif is_lgr:
        st.success(rf"**O ponto $s_0 = {s0_str}$ PERTENCE ao Lugar Geométrico das Raízes.** A fase resultante fecha em **${format_frac(norm_angle)}^\circ \approx 180^\circ$**.")
    else:
        st.error(rf"**O ponto $s_0 = {s0_str}$ NÃO PERTENCE ao Lugar Geométrico das Raízes direto.** A fase resultante é **${format_frac(norm_angle)}^\circ \neq 180^\circ$** (divergência de ${format_frac(abs(defic))}^\circ$).")
        st.markdown(r"#### Cálculo da Deficiência Angular ($\Delta \theta$):")
        st.latex(rf"\Delta \theta = 180^\circ - \angle G(s_0)H(s_0) \equiv {format_frac(defic)}^\circ")
        st.info(rf"**Ação de Projeto Recomendada:** Requer compensador com contribuição de fase $\angle G_c(s_0) = {format_frac(defic)}^\circ$.")

    st.divider()
    st.markdown(r"### 5. Visualização Gráfica dos Vetores Partindo das Singularidades até $s_0$:")
    fig11 = plot_test_point_vectors(poles, zeros, s0, test_details, xmin, xmax, ymin, ymax)
    st.plotly_chart(fig11, width="stretch", config={"scrollZoom": True})


def render_step_12_gain_k(
    data: Dict[str, Any],
    s0: complex,
    test_details: Dict[str, Any],
) -> None:
    """Passo 12: Cálculo de K (s0)."""
    D_coeffs = data["D_coeffs"]
    N_coeffs = data["N_coeffs"]
    s0_str = format_complex_frac(s0)
    is_lgr = test_details["is_lgr"]
    is_pole = test_details["is_pole"]
    is_zero = test_details["is_zero"]
    K_val = test_details["K"]
    prod_p = test_details["prod_p"]
    prod_z = test_details["prod_z"]
    dist_p = test_details["dist_p"]
    dist_z = test_details["dist_z"]
    scale_fac = test_details["scale_factor"]
    D_s0 = test_details["D_s0"]
    N_s0 = test_details["N_s0"]
    P_s0 = test_details["P_s0"]
    residual = test_details["residual"]
    K_req = test_details["K_req"]
    defic = test_details["angular_deficiency_signed"]

    st.markdown(
        r"A **Condição de Módulo** determina o valor exato do ganho estático $K$ associado ao ponto $s_0$:"
    )
    st.latex(r"|K \cdot G(s_0)H(s_0)| = 1 \implies K = \frac{\prod_{i=1}^{nP} |s_0 - p_i|}{\prod_{j=1}^{nZ} |s_0 - z_j|}")

    st.divider()
    st.markdown(r"### 1. Substituição e Formulação do Produto das Distâncias:")
    if dist_p:
        str_p_prod = " \\cdot ".join([f"{format_frac(d)}" for d in dist_p])
        st.latex(rf"\prod_{{i=1}}^{{nP}} |s_0 - p_i| = {str_p_prod} = {format_frac(prod_p)}")
    else:
        st.latex(r"\prod_{i=1}^{nP} |s_0 - p_i| = 1 \quad (\text{sem polos finitos})")

    if dist_z:
        str_z_prod = " \\cdot ".join([f"{format_frac(d)}" for d in dist_z])
        st.latex(rf"\prod_{{j=1}}^{{nZ}} |s_0 - z_j| = {str_z_prod} = {format_frac(prod_z)}")
    else:
        st.latex(r"\prod_{j=1}^{nZ} |s_0 - z_j| = 1 \quad (\text{sem zeros finitos})")

    if is_pole:
        st.latex(r"K(s_0) = 0 \quad (\text{ponto coincide com um polo})")
    elif is_zero:
        st.latex(r"K(s_0) \to \infty \quad (\text{ponto coincide com um zero})")
    else:
        if abs(scale_fac - 1.0) > 1e-4:
            st.latex(rf"K = \frac{{1}}{{|K_{{escala}}|}} \cdot \frac{{\prod |s_0 - p_i|}}{{\prod |s_0 - z_j|}} = \frac{{1}}{{{format_frac(scale_fac)}}} \cdot \frac{{{format_frac(prod_p)}}}{{{format_frac(prod_z)}}} = {format_frac(K_val)}")
        else:
            st.latex(rf"K = \frac{{\prod |s_0 - p_i|}}{{\prod |s_0 - z_j|}} = \frac{{{format_frac(prod_p)}}}{{{format_frac(prod_z)}}} = {format_frac(K_val)}")

    st.divider()
    st.markdown(r"### 2. Comprovação Matemática do Resultado (Prova Real):")
    st.latex(r"P(s) = D(s) + K \cdot N(s) = 0")

    d_eval_str = format_eval_poly(D_coeffs, s0)
    n_eval_str = format_eval_poly(N_coeffs, s0)
    st.markdown(r"**Etapa A — Avaliação do Denominador $D(s_0)$:**")
    st.latex(rf"D(s_0) = {d_eval_str} = {format_complex_frac(D_s0)}")

    st.markdown(r"**Etapa B — Avaliação do Numerador $N(s_0)$:**")
    st.latex(rf"N(s_0) = {n_eval_str} = {format_complex_frac(N_s0)}")

    st.markdown(r"**Etapa C — Multiplicação pelo Ganho Calculado $K$:**")
    k_n_val = K_val * N_s0 if np.isfinite(K_val) else float("inf")
    st.latex(rf"K \cdot N(s_0) = ({format_frac(K_val)}) \cdot ({format_complex_frac(N_s0)}) = {format_complex_frac(k_n_val)}")

    st.markdown(r"**Etapa D — Balanço e Soma $P(s_0)$:**")
    st.latex(rf"P(s_0) = D(s_0) + K \cdot N(s_0) = ({format_complex_frac(D_s0)}) + ({format_complex_frac(k_n_val)}) = {format_complex_frac(P_s0)}")

    if is_lgr and np.isfinite(residual):
        st.latex(r"\boxed{ P(s_0) = 0 + 0j = 0 \quad (\text{COMPROVADO}) }")
        st.success(rf"**COMPROVAÇÃO CONCLUÍDA COM SUCESSO:** A equação característica zera para $s = {s0_str}$ com $K = {format_frac(K_val)}$.")
    else:
        st.latex(rf"\boxed{{ P(s_0) = {format_complex_frac(P_s0)} \neq 0 \quad (\text{{Resíduo }} |P(s_0)| = {format_frac(residual)}) }}")
        if K_req is not None:
            st.markdown(r"**Demonstração do Ganho Necessário ($K_{nec}$):**")
            st.latex(rf"K_{{nec}} = -\frac{{D(s_0)}}{{N(s_0)}} = -\frac{{{format_complex_frac(D_s0)}}}{{{format_complex_frac(N_s0)}}} = {format_complex_frac(K_req)}")
            if abs(np.imag(K_req)) > 1e-4:
                st.warning(rf"**Ganho Complexo Não-Realizável:** $\operatorname{{Im}}(K_{{nec}}) = {format_frac(np.imag(K_req))} \neq 0$. Requer compensação de fase de {format_frac(defic)}°.")
            elif np.real(K_req) < 0:
                st.warning(rf"**Ganho Negativo:** $K_{{nec}} = {format_frac(np.real(K_req))} < 0$, pertenceria ao LGR complementar (realimentação positiva).")
        st.error(rf"**COMPROVAÇÃO DE NÃO-PERTINÊNCIA:** O resíduo $|P(s_0)| = {format_frac(residual)} \neq 0$ confirma que $s_0$ não pode ser polo com ganho puramente real positivo.")
