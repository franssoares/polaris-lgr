"""
Polaris LGR — Construção Interativa do Lugar Geométrico das Raízes
Interface web profissional em Streamlit para análise e projeto de sistemas de controle.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, List, Tuple

from lgr_math import (
    parse_coeffs,
    format_frac,
    format_complex_frac,
    get_plot_limits,
    calculate_breakaway_details,
    build_routh_hurwitz,
    simulate_root_locus,
    calculate_departure_arrival_angles,
    evaluate_test_point_details,
)
from lgr_steps import (
    render_item_a,
    render_item_b,
    render_step_1_char_eq,
    render_step_2_factored,
    render_step_3_poles_zeros,
    render_step_4_real_axis,
    render_step_5_branches,
    render_step_6_symmetry,
    render_step_7_asymptotas,
    render_step_8_breakaway,
    render_step_9_routh_crossings,
    render_step_10_departure_arrival,
    render_step_11_angle_criterion,
    render_step_12_gain_k,
    render_final_animated_lgr,
)

st.set_page_config(page_title="Polaris LGR", layout="wide")

PRESETS = {
    "Personalizado": {
        "ng": "1 2",
        "dg": "1 4 0",
        "nh": "1",
        "dh": "1",
        "s0_re": -2.0,
        "s0_im": 2.0,
    },
    "Caso 1: 2 polos reais, sem zeros — G(s) = K / [s(s+4)]": {
        "ng": "1",
        "dg": "1 4 0",
        "nh": "1",
        "dh": "1",
        "s0_re": -2.0,
        "s0_im": 0.0,
    },
    "Caso 2: 2 polos reais, 1 zero real — G(s) = K(s+2) / [s(s+4)]": {
        "ng": "1 2",
        "dg": "1 4 0",
        "nh": "1",
        "dh": "1",
        "s0_re": -1.0,
        "s0_im": 0.0,
    },
    "Caso 3: 3 polos reais com cruzamento jw — G(s) = K / [s(s+2)(s+4)]": {
        "ng": "1",
        "dg": "1 6 8 0",
        "nh": "1",
        "dh": "1",
        "s0_re": 0.0,
        "s0_im": 2.828,
    },
    "Caso 4: Polos complexos conjugados — G(s) = K(s+3) / [s(s^2 + 2s + 2)]": {
        "ng": "1 3",
        "dg": "1 2 2 0",
        "nh": "1",
        "dh": "1",
        "s0_re": -1.0,
        "s0_im": 1.0,
    },
    "Caso 5: Zeros complexos conjugados — G(s) = K(s^2 + 2s + 5) / [s(s+1)(s+2)(s+3)]": {
        "ng": "1 2 5",
        "dg": "1 6 11 6 0",
        "nh": "1",
        "dh": "1",
        "s0_re": -1.0,
        "s0_im": 2.0,
    },
}


@st.cache_data(show_spinner=False)
def compute_lgr_pipeline(
    ng_tuple: Tuple[float, ...],
    dg_tuple: Tuple[float, ...],
    nh_tuple: Tuple[float, ...],
    dh_tuple: Tuple[float, ...],
) -> Dict[str, Any]:
    """Computes all analytical and numerical Root Locus data once with caching."""
    ng_list = list(ng_tuple) if ng_tuple else [1.0]
    dg_list = list(dg_tuple) if dg_tuple else [1.0]
    nh_list = list(nh_tuple) if nh_tuple else [1.0]
    dh_list = list(dh_tuple) if dh_tuple else [1.0]

    N_coeffs = np.polymul(ng_list, nh_list)
    D_coeffs = np.polymul(dg_list, dh_list)

    zeros = np.roots(N_coeffs)
    poles = np.roots(D_coeffs)
    nP = len(poles)
    nZ = len(zeros)

    real_roots = sorted(
        [np.real(r) for r in np.concatenate((poles, zeros)) if abs(np.imag(r)) < 1e-5],
        reverse=True,
    )

    segments = []
    for j in range(0, len(real_roots), 2):
        start = real_roots[j]
        if j + 1 < len(real_roots):
            end = real_roots[j + 1]
            segments.append(f"[{format_frac(end)}, {format_frac(start)}]")
        else:
            segments.append(f"(-∞, {format_frac(start)}]")

    if nP != nZ:
        sigma_A = (np.sum(poles) - np.sum(zeros)) / (nP - nZ)
        angles_A = [(2 * q + 1) * 180 / abs(nP - nZ) for q in range(abs(nP - nZ))]
    else:
        sigma_A = None
        angles_A = []

    breakaway_details = calculate_breakaway_details(list(N_coeffs), list(D_coeffs))
    valid_breakaway = breakaway_details["valid_points"]
    routh_result = build_routh_hurwitz(list(N_coeffs), list(D_coeffs))
    crossings_data = routh_result.get("crossings_data", [])
    omega_vals = routh_result.get("crossings", [])
    dep_arr_details = calculate_departure_arrival_angles(poles, zeros)

    extra_K = []
    if valid_breakaway:
        extra_K.extend([float(k) for _, k in valid_breakaway if k > 0])
    if crossings_data:
        extra_K.extend([float(d["k_crit"]) for d in crossings_data if d["k_crit"] > 0])

    K_vec, all_roots = simulate_root_locus(D_coeffs, N_coeffs, nP, nZ, extra_K=extra_K)

    return {
        "N_coeffs": list(N_coeffs),
        "D_coeffs": list(D_coeffs),
        "zeros": zeros,
        "poles": poles,
        "nP": nP,
        "nZ": nZ,
        "real_roots": real_roots,
        "segments": segments,
        "sigma_A": sigma_A,
        "angles_A": angles_A,
        "breakaway_details": breakaway_details,
        "valid_breakaway": valid_breakaway,
        "routh_result": routh_result,
        "crossings_data": crossings_data,
        "omega_vals": omega_vals,
        "dep_arr_details": dep_arr_details,
        "K_vec": K_vec,
        "all_roots": all_roots,
    }


@st.cache_data(
    show_spinner=False,
    hash_funcs={
        complex: lambda c: (float(c.real), float(c.imag)),
        np.complex128: lambda c: (float(c.real), float(c.imag)),
        np.complex64: lambda c: (float(c.real), float(c.imag)),
    },
)
def compute_cached_test_point(
    s0: complex,
    poles_tuple: Tuple[complex, ...],
    zeros_tuple: Tuple[complex, ...],
    D_tuple: Tuple[float, ...],
    N_tuple: Tuple[float, ...],
    atol_deg: float,
) -> Dict[str, Any]:
    """Evaluates candidate test point details with caching."""
    return evaluate_test_point_details(
        s0,
        np.array(poles_tuple),
        np.array(zeros_tuple),
        list(D_tuple),
        list(N_tuple),
        atol_deg=atol_deg,
    )


def main() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .polaris-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0 1.5rem 0;
            margin-bottom: 2.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header-text {
            display: flex;
            flex-direction: column;
        }
        
        .polaris-header h1 {
            margin: 0;
            padding: 0;
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        .polaris-header p {
            margin: 0;
            padding-top: 0.2rem;
            color: #888888;
            font-size: 0.95rem;
            font-weight: 400;
        }
        
        .github-link {
            display: flex;
            align-items: center;
            color: #ffffff;
            opacity: 0.5;
            transition: opacity 0.3s ease;
            text-decoration: none;
        }
        
        .github-link:hover {
            opacity: 1;
        }
        
        .polaris-footer {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 0 1rem 0;
            margin-top: 5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #777777;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        .footer-highlight {
            color: #00b4d8;
            font-weight: 500;
        }
        
        .footer-name {
            color: #ffffff;
            font-weight: 500;
        }
        
        .footer-credits {
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        
        @media (max-width: 768px) {
            .polaris-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 1.2rem;
            }
        }
        </style>
        
        <div class="polaris-header">
            <div class="header-text">
                <h1>Polaris LGR</h1>
                <p>Construção Interativa do Lugar Geométrico das Raízes</p>
                <p style="font-size: 0.85rem; color: #555555; margin-top: 0.4rem;">Desenvolvido por Franklin Luiz Soares do Nascimento Filho</p>
            </div>
            <a href="https://github.com/franssoares/polaris-lgr" target="_blank" class="github-link" title="Ver repositório no GitHub">
                <svg height="28" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="28" style="fill: currentColor;">
                    <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
                </svg>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Seleção de Predefinições
    preset_choice = st.selectbox(
        "Carregar Sistema Predefinido",
        list(PRESETS.keys()),
        index=0,
        help="Carregue rapidamente sistemas clássicos da teoria de controle ou selecione Personalizado para definir livremente.",
    )
    active_preset = PRESETS[preset_choice]

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        modo = st.radio("Modo de Exibição", ["Completo", "Simplificado"], horizontal=True)
    with col_m2:
        if modo.startswith("Simplificado"):
            tol_deg = st.number_input(
                "Tolerância angular (graus)",
                value=1.0,
                step=0.1,
                help="Usado no Item (b) para decidir se o ponto pertence ao LGR.",
            )
        else:
            tol_deg = 5.0

    st.divider()
    st.latex(r"1 + H(s)G(s) = 0 \Rightarrow 1 + K \cdot P(s) = 0")
    st.latex(r"G(s) = K \frac{N_G(s)}{D_G(s)} \hspace{1.5cm} H(s) = \frac{N_H(s)}{D_H(s)}")

    col1, col2 = st.columns(2)
    with col1:
        ng_str = st.text_input(
            "Numerador de G(s) - NG(s)",
            value=active_preset["ng"],
            key="num_input",
            help="Coeficientes do numerador de G(s) em ordem decrescente, separados por espaço.",
        )
        dg_str = st.text_input(
            "Denominador de G(s) - DG(s)",
            value=active_preset["dg"],
            key="den_input",
            help="Coeficientes do denominador de G(s) em ordem decrescente, separados por espaço.",
        )
    with col2:
        nh_str = st.text_input(
            "Numerador de H(s) - NH(s)",
            value=active_preset["nh"],
            key="nh_input",
            help="Coeficientes do numerador de H(s) em ordem decrescente, separados por espaço.",
        )
        dh_str = st.text_input(
            "Denominador de H(s) - DH(s)",
            value=active_preset["dh"],
            key="dh_input",
            help="Coeficientes do denominador de H(s) em ordem decrescente, separados por espaço.",
        )

    st.markdown("### Ponto de Teste $s_0$")
    col3, col4 = st.columns(2)
    with col3:
        s0_real = st.number_input(r"Parte real ($\sigma$)", value=float(active_preset["s0_re"]))
    with col4:
        s0_imag = st.number_input(r"Parte imaginária ($j\omega$)", value=float(active_preset["s0_im"]))

    st.markdown("### Limites do Gráfico (Opcional)")
    use_limits = st.checkbox("Definir limites manualmente (desativa autoscale)")
    if use_limits:
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            xmin_man = st.number_input("X mín", value=-6.0)
        with col6:
            xmax_man = st.number_input("X máx", value=2.0)
        with col7:
            ymin_man = st.number_input("Y mín", value=-4.0)
        with col8:
            ymax_man = st.number_input("Y máx", value=4.0)

    calc_btn = st.button("Calcular LGR", type="primary")

    if calc_btn:
        st.session_state["has_calculated"] = True

    if st.session_state.get("has_calculated", False):
        ng = parse_coeffs(ng_str)
        dg = parse_coeffs(dg_str)
        nh = parse_coeffs(nh_str)
        dh = parse_coeffs(dh_str)

        if any(x is None for x in [ng, dg, nh, dh]):
            st.error("Erro de sintaxe nos coeficientes. Utilize apenas números separados por espaço.")
            st.stop()

        if not dg or not any(abs(c) > 1e-12 for c in dg):
            st.error("O denominador D_G(s) não pode ser nulo.")
            st.stop()

        if not dh or not any(abs(c) > 1e-12 for c in dh):
            st.error("O denominador D_H(s) não pode ser nulo.")
            st.stop()

        s0 = complex(s0_real, s0_imag)

        with st.spinner("Calculando parâmetros analíticos e gráficos do LGR..."):
            data = compute_lgr_pipeline(
                tuple(ng if ng else [1.0]),
                tuple(dg if dg else [1.0]),
                tuple(nh if nh else [1.0]),
                tuple(dh if dh else [1.0]),
            )

        poles = data["poles"]
        zeros = data["zeros"]
        valid_breakaway = data["valid_breakaway"]
        omega_vals = data["omega_vals"]
        sigma_A = data["sigma_A"]
        nP = data["nP"]
        nZ = data["nZ"]

        extra_points = [s0]
        extra_points.extend([complex(np.real(p[0]), np.imag(p[0])) for p in valid_breakaway])
        extra_points.extend([complex(0, w) for w in omega_vals])
        extra_points.extend([complex(0, -w) for w in omega_vals])
        if sigma_A is not None:
            extra_points.append(complex(np.real(sigma_A), 0))

        if use_limits:
            xmin, xmax, ymin, ymax = xmin_man, xmax_man, ymin_man, ymax_man
        else:
            xmin, xmax, ymin, ymax = get_plot_limits(poles, zeros, extra_points)

        max_dist = max(xmax - xmin, ymax - ymin)
        length_max = max(2.0, max_dist * 0.8)

        # Atualiza coordenadas dos segmentos reais com base nos limites atuais
        real_roots = data["real_roots"]
        segment_coords = []
        for j in range(0, len(real_roots), 2):
            start = real_roots[j]
            if j + 1 < len(real_roots):
                segment_coords.append((start, real_roots[j + 1]))
            else:
                end_plot = xmin - (xmax - xmin) * 0.1
                segment_coords.append((start, end_plot))
        data["segment_coords"] = segment_coords

        test_details = compute_cached_test_point(
            s0,
            tuple(poles),
            tuple(zeros),
            tuple(data["D_coeffs"]),
            tuple(data["N_coeffs"]),
            tol_deg,
        )

        if modo.startswith("Simplificado"):
            st.markdown("### Seleção de Itens:")
            col_a, col_b = st.columns(2)
            with col_a:
                show_item_a = st.checkbox("Item (a) — Esboço do LGR", value=True)
            with col_b:
                show_item_b = st.checkbox("Item (b) — Teste de Ponto", value=True)

            if show_item_a:
                render_item_a(data, xmin, xmax, ymin, ymax, length_max)

            if show_item_b:
                render_item_b(data, s0, test_details, xmin, xmax, ymin, ymax)

        else:
            expand_all = st.checkbox("Expandir todos os passos", value=True)

            with st.expander("Passo 1: Equação característica", expanded=expand_all):
                render_step_1_char_eq(data, ng, dg, nh, dh)

            with st.expander("Passo 2: Forma fatorada de P(s)", expanded=expand_all):
                render_step_2_factored(data)

            with st.expander("Passo 3: Polos e zeros no plano s", expanded=expand_all):
                render_step_3_poles_zeros(data, xmin, xmax, ymin, ymax)

            with st.expander("Passo 4: Segmentos do eixo real", expanded=expand_all):
                render_step_4_real_axis(data, xmin, xmax, ymin, ymax)

            with st.expander("Passo 5: Número de lugares separados (ramos)", expanded=expand_all):
                render_step_5_branches(data)

            with st.expander("Passo 6: Simetria", expanded=expand_all):
                render_step_6_symmetry()

            with st.expander("Passo 7: Assíntotas", expanded=expand_all):
                render_step_7_asymptotas(data, xmin, xmax, ymin, ymax, length_max)

            with st.expander("Passo 8: Pontos de saída/entrada", expanded=expand_all):
                render_step_8_breakaway(data, xmin, xmax, ymin, ymax)

            with st.expander("Passo 9: Cruzamento com o eixo imaginário", expanded=expand_all):
                render_step_9_routh_crossings(data, xmin, xmax, ymin, ymax)

            with st.expander("Passo 10: Ângulos de partida/chegada", expanded=expand_all):
                render_step_10_departure_arrival(data, xmin, xmax, ymin, ymax)

            with st.expander("Passo 11: Critério de ângulo (s0)", expanded=expand_all):
                render_step_11_angle_criterion(data, s0, test_details, xmin, xmax, ymin, ymax)

            with st.expander("Passo 12: Cálculo de K (s0)", expanded=expand_all):
                render_step_12_gain_k(data, s0, test_details)

            st.markdown("## Gráfico Final do LGR")
            render_final_animated_lgr(
                data["valid_breakaway"],
                data["routh_result"],
                data["D_coeffs"],
                data["N_coeffs"],
                nP,
                nZ,
                poles,
                zeros,
                xmin,
                xmax,
                ymin,
                ymax,
                sim_data=(data["K_vec"], data["all_roots"]),
            )

        st.markdown(
            """
            <div class="polaris-footer">
                <div class="footer-credits">
                    <span>Desenvolvido por <span class="footer-name">franssoares</span></span>
                </div>
                <p style="margin: 0.5rem 0 0 0;">Disciplina de Sistemas de Controle &bull; <span class="footer-highlight">Polaris LGR &copy; 2026</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
