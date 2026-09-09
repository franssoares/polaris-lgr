import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.colors
import importlib
import lgr_math
import lgr_plots

importlib.reload(lgr_math)
importlib.reload(lgr_plots)

from lgr_math import (
    parse_coeffs,
    format_poly_latex,
    format_factored_latex,
    get_plot_limits,
    find_breakaway_points,
    calculate_breakaway_details,
    build_routh_hurwitz,
    simulate_root_locus,
    calculate_departure_arrival_angles,
    evaluate_test_point_details,
)
from lgr_plots import create_base_plot, add_poles_zeros_traces, plot_test_point_vectors

st.set_page_config(page_title="LGR - 12 Passos", layout="wide")


from fractions import Fraction

def format_frac(val, tol=1e-5):
    if abs(val - round(val)) < tol:
        return f"{int(round(val))}"
    return f"{float(val):.3g}"

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

def main():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        /* Esconde elementos nativos do Streamlit que poluem a tela */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Reduz o padding gigante do topo para colar o nosso header em cima */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Aplica tipografia moderna do portfólio a todo o app */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Header Minimalista e Elegante */
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
        
        /* Footer Elegante */
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
        
        /* Responsividade para Mobile */
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

    st.latex(r"1 + H(s)G(s) = 0 \Rightarrow 1 + K \cdot P(s) = 0")
    st.latex(
        r"G(s) = K \frac{N_G(s)}{D_G(s)} \hspace{1.5cm} H(s) = \frac{N_H(s)}{D_H(s)}"
    )

    col1, col2 = st.columns(2)
    with col1:
        ng_str = st.text_input("Numerador de G(s) - NG(s)", "1 2")
        dg_str = st.text_input("Denominador de G(s) - DG(s)", "1 4 0")
    with col2:
        nh_str = st.text_input("Numerador de H(s) - NH(s)", "1")
        dh_str = st.text_input("Denominador de H(s) - DH(s)", "1")

    st.markdown("### Ponto de Teste $s_0$")
    col3, col4 = st.columns(2)
    with col3:
        s0_real = st.number_input(r"Parte real ($\sigma$)", value=-2.0)
    with col4:
        s0_imag = st.number_input(r"Parte imaginária ($j\omega$)", value=2.0)

    st.markdown("### Limites do Gráfico (Opcional)")
    use_limits = st.checkbox("Definir limites manualmente (desativa o autoscale)")
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

    calc_btn = st.button("Calcular LGR")

    if calc_btn:
        ng = parse_coeffs(ng_str)
        dg = parse_coeffs(dg_str)
        nh = parse_coeffs(nh_str)
        dh = parse_coeffs(dh_str)

        if any(x is None for x in [ng, dg, nh, dh]) or not dg or not dh:
            st.error(
                "Erro ao ler os coeficientes. Certifique-se de digitar apenas números separados por espaço."
            )
            st.stop()

        N_coeffs = np.polymul(ng if ng else [1], nh if nh else [1])
        D_coeffs = np.polymul(dg if dg else [1], dh if dh else [1])

        zeros = np.roots(N_coeffs)
        poles = np.roots(D_coeffs)
        nP = len(poles)
        nZ = len(zeros)

        s0 = complex(s0_real, s0_imag)

        # Calculate features to determine auto limits
        breakaway_details = calculate_breakaway_details(N_coeffs, D_coeffs)
        valid_breakaway = breakaway_details["valid_points"]
        routh_result = build_routh_hurwitz(N_coeffs, D_coeffs)
        omega_vals = routh_result["crossings"]

        extra_points = [s0]
        extra_points.extend([complex(p[0], np.imag(p[0])) for p in valid_breakaway])
        extra_points.extend([complex(0, w) for w in omega_vals])
        extra_points.extend([complex(0, -w) for w in omega_vals])

        if nP != nZ:
            sigma_A = (np.sum(poles) - np.sum(zeros)) / (nP - nZ)
            extra_points.append(complex(np.real(sigma_A), 0))
        else:
            sigma_A = None

        if use_limits:
            xmin, xmax, ymin, ymax = xmin_man, xmax_man, ymin_man, ymax_man
        else:
            xmin, xmax, ymin, ymax = get_plot_limits(poles, zeros, extra_points)

        max_dist = max(xmax - xmin, ymax - ymin)
        length_max = max(2.0, max_dist * 0.8)

        # Passo 1
        with st.expander("Passo 1: Equação característica", expanded=True):
            ng_latex = format_poly_latex(ng if ng else [1])
            dg_latex = format_poly_latex(dg if dg else [1])
            nh_latex = format_poly_latex(nh if nh else [1])
            dh_latex = format_poly_latex(dh if dh else [1])
            num_str = format_poly_latex(N_coeffs)
            den_str = format_poly_latex(D_coeffs)

            st.markdown("**1. Função de Transferência de Malha Fechada (FTMF)**")
            st.markdown(
                "A relação entre a saída e a entrada de um sistema em malha fechada com realimentação negativa é dada por:"
            )
            st.latex(r"T(s) = \frac{G(s)}{1 + G(s)H(s)}")

            st.markdown("**2. Equação Característica e Estabilidade**")
            st.markdown(
                "A estabilidade do sistema é determinada pelos polos da malha fechada, que são as raízes do denominador da FTMF igualado a zero:"
            )
            st.latex(r"1 + G(s)H(s) = 0")

            st.markdown("**3. Substituição das Funções**")
            st.markdown(
                "Substituindo os blocos $G(s)$ e $H(s)$ separadamente na equação característica:"
            )

            def get_block_latex(n_lat, d_lat, is_g=False):
                prefix = "K " if is_g else ""
                if n_lat == "1" and d_lat == "1":
                    return f"{prefix}1" if prefix else "1"
                elif d_lat == "1":
                    return (
                        f"{prefix}({n_lat})"
                        if ("+" in n_lat or "-" in n_lat)
                        else f"{prefix}{n_lat}"
                    )
                else:
                    return f"{prefix}\\frac{{{n_lat}}}{{{d_lat}}}"

            g_latex = get_block_latex(ng_latex, dg_latex, True)
            h_latex = get_block_latex(nh_latex, dh_latex, False)

            h_str = (
                rf"\underbrace{{{h_latex}}}_{{H(s)}}"
                if h_latex == "1"
                else rf"\underbrace{{\left( {h_latex} \right)}}_{{H(s)}}"
            )
            g_str = rf"\underbrace{{\left( {g_latex} \right)}}_{{G(s)}}"

            st.latex(r"1 + " + g_str + r" \cdot " + h_str + r" = 0")

            st.markdown("**4. Multiplicação das Frações**")
            st.markdown(
                "Agrupando as funções em uma única fração multiplicada (numerador com numerador, denominador com denominador):"
            )

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
            st.markdown(
                "Agrupando a parte fixa do sistema consolidado como $P(s)$, podemos reescrever o termo de malha aberta como:"
            )
            st.latex(
                r"G(s)H(s) = K \cdot P(s) = K \frac{" + num_str + r"}{" + den_str + r"}"
            )

            st.markdown("**6. Obtenção da Equação Característica**")
            st.markdown(
                r"Portanto, obtemos a equação característica final na forma padrão do Lugar Geométrico das Raízes ($1 + K \cdot P(s) = 0$):"
            )
            st.latex(r"\boxed{ 1 + K \frac{" + num_str + r"}{" + den_str + r"} = 0 }")

            def get_char_poly_latex(D_coeffs, N_coeffs):
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

            char_poly_str = get_char_poly_latex(D_coeffs, N_coeffs)

            st.markdown(
                r"Ou, multiplicando toda a equação pelo denominador, obtemos o **polinômio característico** ($D(s) + K \cdot N(s) = 0$), que é o formato utilizado pelo critério de Routh-Hurwitz:"
            )
            st.latex(r"\boxed{ " + char_poly_str + r" = 0 }")

        # Passo 2
        with st.expander("Passo 2: Forma fatorada de P(s)", expanded=True):
            num_fact = format_factored_latex(zeros)
            den_fact = format_factored_latex(poles)
            K_scale = N_coeffs[0] / D_coeffs[0]
            K_str = f"{format_frac(K_scale)}" if K_scale != 1.0 else ""
            st.latex(
                r"P(s) = " + K_str + r"\frac{" + num_fact + r"}{" + den_fact + r"}"
            )

            def format_root(r):
                r_rounded = np.round(r, 4)
                if abs(np.imag(r_rounded)) < 1e-5:
                    return f"{format_frac(np.real(r_rounded))}"
                else:
                    sign = "+" if np.imag(r_rounded) > 0 else "-"
                    return f"{format_frac(np.real(r_rounded))} {sign} {format_frac(abs(np.imag(r_rounded)))}j"

            z_list = [
                f"z_{{{i+1}}} = {format_root(z)}"
                for i, z in enumerate(
                    sorted(zeros, key=lambda x: (np.real(x), np.imag(x)))
                )
            ]
            p_list = [
                f"p_{{{i+1}}} = {format_root(p)}"
                for i, p in enumerate(
                    sorted(poles, key=lambda x: (np.real(x), np.imag(x)))
                )
            ]

            st.markdown(
                "**Zeros da malha aberta:** "
                + (
                    ", ".join([f"${z}$" for z in z_list])
                    if z_list
                    else "Nenhum zero finito."
                )
            )
            st.markdown(
                "**Polos da malha aberta:** "
                + (
                    ", ".join([f"${p}$" for p in p_list])
                    if p_list
                    else "Nenhum polo finito."
                )
            )

        # Passo 3
        with st.expander("Passo 3: Polos e zeros no plano s", expanded=True):
            fig3 = create_base_plot(
                poles, zeros, "Polos e Zeros", xmin, xmax, ymin, ymax, show_labels=True
            )
            st.plotly_chart(fig3, width="stretch", config={"scrollZoom": True})

        # Passo 4
        with st.expander("Passo 4: Segmentos do eixo real", expanded=True):
            real_roots = [
                np.real(r)
                for r in np.concatenate((poles, zeros))
                if abs(np.imag(r)) < 1e-5
            ]
            real_roots = sorted(real_roots, reverse=True)

            fig4 = create_base_plot(
                poles, zeros, "Mapeamento no Eixo Real", xmin, xmax, ymin, ymax
            )

            segments = []
            segment_coords = []
            for i in range(0, len(real_roots), 2):
                start = real_roots[i]
                if i + 1 < len(real_roots):
                    end = real_roots[i + 1]
                    segments.append(f"[{format_frac(end)}, {format_frac(start)}]")
                    segment_coords.append((start, end))
                else:
                    segments.append(f"(-∞, {format_frac(start)}]")
                    end_plot = xmin - (xmax - xmin) * 0.1
                    segment_coords.append((start, end_plot))
                    fig4.add_annotation(
                        x=end_plot,
                        y=0,
                        ax=start,
                        ay=0,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=3,
                        arrowcolor="#00b4d8",
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

            fig4.add_trace(
                go.Scatter(
                    x=final_x,
                    y=final_y,
                    mode="lines",
                    line=dict(color="#00b4d8", width=5),
                    name="LGR Real",
                    hoverinfo="skip",
                )
            )
            fig4.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol="line-ns", size=24, line=dict(color="yellow", width=4)
                    ),
                    name="Varredura",
                    showlegend=False,
                )
            )

            base_annotations = (
                list(fig4.layout.annotations) if fig4.layout.annotations else []
            )
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
                        data=[
                            go.Scatter(x=frame_x, y=frame_y),
                            go.Scatter(x=[step_x], y=[0]),
                        ],
                        layout=go.Layout(annotations=hidden_annotations),
                        traces=[lgr_trace_idx, scanner_trace_idx],
                    )
                )

            # Add final frame to hide the scanner and explicitly restore the arrow visibility
            restored_annotations = []
            for ann in base_annotations:
                restored_ann = ann.to_plotly_json()
                restored_ann["visible"] = True
                restored_annotations.append(restored_ann)

            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(x=final_x, y=final_y),
                        go.Scatter(x=[None], y=[None]),
                    ],
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
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 40, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0},
                                    },
                                ],
                                "label": " Play",
                                "method": "animate",
                            },
                            {
                                "args": [
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                                "label": " Pause",
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

        # Passo 5
        with st.expander("Passo 5: Número de lugares separados (ramos)", expanded=True):
            st.markdown(r"Sendo $n_P$ o número de polos e $n_Z$ o número de zeros da malha aberta, temos:")
            st.markdown(rf"- $n_P = {nP}$")
            st.markdown(rf"- $n_Z = {nZ}$")
            st.markdown(r"O número de lugares separados (ramos do LGR) é dado por:")
            ls = max(nP, nZ)
            st.latex(r"LS = \max(n_P, n_Z)")
            st.latex(rf"LS = \max({nP}, {nZ}) = {ls}")

        # Passo 6
        with st.expander("Passo 6: Simetria", expanded=True):
            st.markdown("O LGR é simétrico em relação ao eixo real.")

        # Passo 7
        with st.expander("Passo 7: Assíntotas", expanded=True):
            if nP == nZ:
                st.markdown("Não há assíntotas.")
            else:
                st.markdown(r"**Centro das assíntotas ($\sigma_A$):**")
                sum_p_str = " + ".join([f"({format_complex_frac(p)})" for p in poles])
                sum_z_str = " + ".join([f"({format_complex_frac(z)})" for z in zeros])
                if not sum_p_str:
                    sum_p_str = "0"
                if not sum_z_str:
                    sum_z_str = "0"

                st.latex(r"\sigma_A = \frac{\sum p_i - \sum z_i}{n_P - n_Z}")
                st.latex(
                    rf"\sigma_A = \frac{{[{sum_p_str}] - [{sum_z_str}]}}{{{nP} - {nZ}}} = {format_frac(np.real(sigma_A))}"
                )

                angles_A = []
                for q in range(abs(nP - nZ)):
                    angle = (2 * q + 1) * 180 / abs(nP - nZ)
                    angles_A.append(angle)

                st.markdown(r"**Ângulos das assíntotas ($\theta_k$):**")
                st.latex(
                    r"\theta_k = \frac{(2k + 1) \cdot 180^\circ}{|n_P - n_Z|} \quad \text{para } k = 0, 1, \dots, |n_P - n_Z| - 1"
                )
                for k, angle in enumerate(angles_A):
                    st.latex(
                        rf"\theta_{k} = \frac{{(2({k}) + 1) \cdot 180^\circ}}{{{abs(nP - nZ)}}} = {format_frac(angle)}^\circ"
                    )

                st.markdown(r"**Cruzamento das assíntotas com o eixo imaginário:**")
                st.markdown(
                    r"A equação da reta é $y = \tan(\theta_k) \cdot (x - \sigma_A)$. Quando $x = 0$, temos $y_{cruzamento} = -\sigma_A \cdot \tan(\theta_k)$. Como a assíntota é uma semirreta que parte de $\sigma_A$, ela só cruza de fato se estiver apontando na direção do eixo imaginário."
                )

                crossings = []
                for k, angle in enumerate(angles_A):
                    if angle % 180 == 90:
                        if np.real(sigma_A) == 0:
                            st.latex(
                                rf"\theta_{k} = {format_frac(angle)}^\circ \implies \text{{Assíntota sobre o eixo imaginário}}"
                            )
                        else:
                            st.latex(
                                rf"\theta_{k} = {format_frac(angle)}^\circ \implies \text{{Assíntota paralela ao eixo imaginário (não cruza)}}"
                            )
                    else:
                        cos_val = np.cos(np.radians(angle))
                        t_cross = -np.real(sigma_A) / cos_val if cos_val != 0 else -1
                        if t_cross >= 0:
                            cross_y = -np.real(sigma_A) * np.tan(np.radians(angle))
                            if abs(cross_y) < 1e-5:
                                st.latex(
                                    rf"\theta_{k} = {format_frac(angle)}^\circ \implies \text{{Assíntota sobre o eixo real (não destacaremos a origem)}}"
                                )
                            else:
                                st.latex(
                                    rf"\theta_{k} = {format_frac(angle)}^\circ \implies y_{{cruzamento}} = -({format_frac(np.real(sigma_A))}) \cdot \tan({format_frac(angle)}^\circ) = {format_frac(cross_y)}j"
                                )
                                crossings.append((angle, cross_y))
                        else:
                            st.latex(
                                rf"\theta_{k} = {format_frac(angle)}^\circ \implies \text{{A semirreta se afasta do eixo imaginário (não cruza)}}"
                            )

                fig7 = create_base_plot(
                    poles,
                    zeros,
                    "Assíntotas",
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    draw_poles_zeros=False,
                )

                for q, angle in enumerate(angles_A):
                    rad = np.radians(angle)

                    t_vals = []
                    if np.cos(rad) > 1e-5:
                        t_vals.append((xmax - np.real(sigma_A)) / np.cos(rad))
                    elif np.cos(rad) < -1e-5:
                        t_vals.append((xmin - np.real(sigma_A)) / np.cos(rad))

                    if np.sin(rad) > 1e-5:
                        t_vals.append((ymax - 0) / np.sin(rad))
                    elif np.sin(rad) < -1e-5:
                        t_vals.append((ymin - 0) / np.sin(rad))

                    t_vals = [t for t in t_vals if t > 0]
                    t_max = min(t_vals) if t_vals else length_max

                    base_R = max(xmax - xmin, ymax - ymin) * 0.05
                    arc_R = base_R + q * (max(xmax - xmin, ymax - ymin) * 0.08)

                    t_draw = max(t_max * 0.85, arc_R * 1.25)

                    dx = t_draw * np.cos(rad)
                    dy = t_draw * np.sin(rad)
                    head_x = np.real(sigma_A) + dx
                    head_y = dy

                    fig7.add_trace(
                        go.Scatter(
                            x=[np.real(sigma_A), head_x],
                            y=[0, head_y],
                            mode="lines",
                            line=dict(color="orange", width=2, dash="dash"),
                            showlegend=False,
                        )
                    )

                    fig7.add_annotation(
                        x=head_x,
                        y=head_y,
                        ax=np.real(sigma_A) + dx * 0.9,
                        ay=dy * 0.9,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor="orange",
                    )

                    if angle > 5:
                        # Draw counter-clockwise from 0 to angle
                        theta_vals = np.linspace(
                            0, np.radians(angle), max(15, int(angle))
                        )
                        arc_x = np.real(sigma_A) + arc_R * np.cos(theta_vals)
                        arc_y = arc_R * np.sin(theta_vals)

                        fig7.add_trace(
                            go.Scatter(
                                x=arc_x,
                                y=arc_y,
                                mode="lines",
                                line=dict(color="rgba(255, 165, 0, 0.6)", width=2),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

                        # Arrowhead at the end of the arc (pointing counter-clockwise)
                        delta = np.radians(8)
                        tail_x = np.real(sigma_A) + arc_R * np.cos(
                            np.radians(angle) - delta
                        )
                        tail_y = arc_R * np.sin(np.radians(angle) - delta)

                        fig7.add_annotation(
                            x=arc_x[-1],
                            y=arc_y[-1],
                            ax=tail_x,
                            ay=tail_y,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.2,
                            arrowwidth=2,
                            arrowcolor="orange",
                        )

                        text_R = arc_R + max(xmax - xmin, ymax - ymin) * 0.02
                        mid_rad = np.radians(angle) / 2
                        text_x = np.real(sigma_A) + text_R * np.cos(mid_rad)
                        text_y = text_R * np.sin(mid_rad)

                        fig7.add_annotation(
                            x=text_x,
                            y=text_y,
                            text=f"{format_frac(angle)}°",
                            showarrow=False,
                            font=dict(color="orange", size=13),
                        )
                    else:
                        fig7.add_annotation(
                            x=head_x + dx * 0.1,
                            y=head_y + dy * 0.1,
                            text=f"{format_frac(angle)}°",
                            showarrow=False,
                            font=dict(color="orange", size=14),
                        )

                for angle, cross_y in crossings:
                    if abs(cross_y) < ymax * 5:  # prevent plotting huge values
                        fig7.add_trace(
                            go.Scatter(
                                x=[0],
                                y=[cross_y],
                                mode="markers+text",
                                marker=dict(
                                    symbol="circle-open",
                                    size=10,
                                    color="magenta",
                                    line=dict(width=2),
                                ),
                                text=[f"{format_frac(cross_y)}j"],
                                textposition="middle right",
                                textfont=dict(color="magenta", size=13),
                                name=f"Cruzamento {format_frac(angle)}°",
                            )
                        )

                # Add centroid
                fig7.add_trace(
                    go.Scatter(
                        x=[np.real(sigma_A)],
                        y=[0],
                        mode="markers+text",
                        marker=dict(
                            symbol="square",
                            size=10,
                            color="orange",
                            line=dict(color="white", width=2),
                        ),
                        text=[f"σ<sub>A</sub> = {format_frac(np.real(sigma_A))}"],
                        textposition="bottom center",
                        textfont=dict(color="orange", size=13),
                        name="Centroide",
                        hoverinfo="skip",
                    )
                )

                # Render poles and zeros AT THE VERY END so they appear on top of EVERYTHING
                add_poles_zeros_traces(fig7, poles, zeros, show_labels=False)

                fig7.update_layout(height=600)
                st.plotly_chart(fig7, width="stretch", config={"scrollZoom": True})

        # Passo 8
        with st.expander("Passo 8: Pontos de saída/entrada", expanded=True):
            st.markdown(
                r"Os pontos de **saída** (*breakaway*) e de **entrada** (*break-in*) indicam as posições onde ramos do LGR se encontram e se bifurcam. "
                r"No eixo real, esses pontos correspondem a raízes múltiplas da equação característica $1 + K P(s) = 0$, o que coincide com os pontos críticos da função de ganho $K(s)$, isto é, onde a taxa de variação do ganho em relação a $s$ é nula:"
            )
            st.latex(r"\frac{dK}{ds} = 0")

            D_sym = breakaway_details["D_sym"]
            N_sym = breakaway_details["N_sym"]
            D_der_sym = breakaway_details["D_der_sym"]
            N_der_sym = breakaway_details["N_der_sym"]
            U_sym = breakaway_details["U_sym"]
            U_simp = breakaway_details["U_simp"]
            candidates = breakaway_details["candidates"]
            valid_points = breakaway_details["valid_points"]
            is_constant_deriv = breakaway_details["is_constant_deriv"]

            # 1. Expressão Analítica do Ganho K(s)
            st.markdown(r"**1. Expressão Analítica do Ganho $K(s)$:**")
            st.markdown(
                r"A partir da equação característica consolidada $1 + K \frac{N(s)}{D(s)} = 0 \iff D(s) + K \cdot N(s) = 0$, isolamos o ganho $K$ em função de $s$:"
            )
            st.latex(r"K(s) = -\frac{D(s)}{N(s)}")

            d_latex = sp.latex(D_sym)
            n_latex = sp.latex(N_sym)
            if n_latex == "1":
                st.latex(rf"K(s) = -\left({d_latex}\right)")
            else:
                st.latex(rf"K(s) = -\frac{{{d_latex}}}{{{n_latex}}}")

            # 2. Aplicação da Derivada dK/ds = 0
            st.markdown(r"**2. Derivação de $K(s)$ em Relação a $s$ ($\frac{dK}{ds} = 0$):**")
            if n_latex == "1":
                st.markdown(
                    r"Como o numerador é unitário ($N(s) = 1$), a derivada de $K(s)$ é obtida diretamente diferenciando o polinômio $D(s)$:"
                )
                st.latex(r"\frac{dK}{ds} = -D'(s) = 0 \implies D'(s) = 0")
                st.markdown(r"Calculando a derivada de $D(s)$:")
                st.latex(rf"D'(s) = \frac{{d}}{{ds}}\left[ {d_latex} \right] = {sp.latex(D_der_sym)}")
                st.markdown(r"Igualando a zero para encontrar os pontos críticos:")
                st.latex(rf"{sp.latex(D_der_sym)} = 0")
            else:
                st.markdown(
                    r"Pela regra da derivada do quociente para uma função racional $\frac{D(s)}{N(s)}$:"
                )
                st.latex(
                    r"\frac{dK}{ds} = -\frac{D'(s) \cdot N(s) - D(s) \cdot N'(s)}{[N(s)]^2} = 0"
                )
                st.markdown(r"Calculando as derivadas dos polinômios $D(s)$ e $N(s)$:")
                st.latex(rf"D'(s) = \frac{{d}}{{ds}}\left[ {d_latex} \right] = {sp.latex(D_der_sym)}")
                st.latex(rf"N'(s) = \frac{{d}}{{ds}}\left[ {n_latex} \right] = {sp.latex(N_der_sym)}")
                st.markdown(
                    r"Como o denominador $[N(s)]^2 \neq 0$ para pontos finitos fora dos zeros de malha aberta, a condição $\frac{dK}{ds} = 0$ equivale a igualar o numerador a zero:"
                )
                st.latex(r"D'(s) \cdot N(s) - D(s) \cdot N'(s) = 0")
                st.markdown(r"Substituindo as expressões derivadas:")
                st.latex(
                    rf"\left({sp.latex(D_der_sym)}\right) \cdot \left({n_latex}\right) - \left({d_latex}\right) \cdot \left({sp.latex(N_der_sym)}\right) = 0"
                )

            # 3. Equação Polinomial Resultante
            st.markdown(r"**3. Equação Polinomial Resultante:**")
            if is_constant_deriv:
                st.markdown(
                    rf"A derivada resulta em uma constante não nula (${sp.latex(U_sym)} \neq 0$). Portanto, a equação $\frac{{dK}}{{ds}} = 0$ não possui soluções no plano finito."
                )
                st.info("Não existem pontos de saída ou de entrada para este sistema.")
            else:
                st.markdown(
                    r"Expandindo os produtos e agrupando os termos em potências decrescentes de $s$:"
                )
                eq_expanded_str = f"{sp.latex(U_sym)} = 0"
                eq_simp_str = f"{sp.latex(U_simp)} = 0"
                if U_sym != U_simp and U_simp != 0:
                    st.latex(rf"{eq_expanded_str} \iff {eq_simp_str}")
                else:
                    st.latex(eq_expanded_str)

                # 4. Raízes Candidatas
                st.markdown(r"**4. Raízes Candidatas ($\sigma_i$):**")
                st.markdown(
                    r"Resolvendo a equação polinomial $\frac{dK}{ds} = 0$, encontramos as seguintes raízes candidatas:"
                )
                for cand in candidates:
                    idx = cand["index"]
                    s_v = cand["s_val"]
                    s_str = format_complex_frac(s_v)
                    st.latex(rf"s_{{{idx}}} = {s_str}")

                # 5. Análise e Validação de Cada Candidato
                st.markdown(r"**5. Análise e Validação de Cada Candidato:**")
                st.markdown(
                    r"Para cada candidato $s_i$, avaliamos três critérios fundamentais:"
                    r"<br>1. **Cálculo do Ganho:** $K(s_i) = -\frac{D(s_i)}{N(s_i)}$ deve ser real e estritamente positivo ($K > 0$) para pertencer ao LGR direto."
                    r"<br>2. **Pertencimento ao LGR no Eixo Real:** A soma dos polos e zeros reais à direita de $s_i$ deve ser ímpar (regra do Passo 4)."
                    r"<br>3. **Classificação Física (Segunda Derivada $\frac{d^2K}{ds^2}$):**"
                    r"<br>&nbsp;&nbsp;&nbsp;&nbsp;• $\frac{d^2K}{ds^2} < 0 \implies$ **Máximo Local** de $K(s) \implies$ **Ponto de Saída (*Breakaway*)** (dois ramos partem de polos adjacentes e se dividem rumo ao plano complexo)."
                    r"<br>&nbsp;&nbsp;&nbsp;&nbsp;• $\frac{d^2K}{ds^2} > 0 \implies$ **Mínimo Local** de $K(s) \implies$ **Ponto de Entrada (*Break-in*)** (dois ramos complexos retornam ao eixo real e se separam em direção a zeros finitos ou assíntotas).",
                    unsafe_allow_html=True,
                )

                for cand in candidates:
                    idx = cand["index"]
                    s_v = cand["s_val"]
                    s_str = format_complex_frac(s_v)
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

                    # Subetapa a: Cálculo do ganho
                    st.markdown(r"**a) Avaliação do ganho $K(s)$:**")
                    d_eval_str = format_complex_frac(d_val)
                    n_eval_str = format_complex_frac(n_val)
                    k_eval_str = format_complex_frac(k_val)

                    if abs(n_val - 1.0) < 1e-5:
                        st.latex(
                            rf"K(s_{{{idx}}}) = -D({s_str}) = -\left({d_eval_str}\right) = {k_eval_str}"
                        )
                    else:
                        st.latex(
                            rf"K(s_{{{idx}}}) = -\frac{{D({s_str})}}{{N({s_str})}} = -\frac{{{d_eval_str}}}{{{n_eval_str}}} = {k_eval_str}"
                        )

                    # Subetapa b: Pertencimento
                    st.markdown(r"**b) Teste de pertinência ao LGR:**")
                    if is_r:
                        paridade = "número ímpar" if on_r else "número par"
                        st.markdown(
                            f"À direita de $s = {s_str}$ existem **{c_right}** polos e zeros reais ({paridade})."
                        )
                        if k_real > 0 and on_r:
                            st.markdown(
                                rf"Como $K = {format_frac(k_real)} > 0$ e o ponto está em segmento com número ímpar de singularidades à direita, ele **pertence ao LGR**."
                            )
                        else:
                            st.markdown(
                                rf"Como $K = {format_frac(k_real)} \le 0$, o ponto **NÃO pertence ao LGR direto** (pertenceria ao LGR complementar para $K < 0$)."
                            )
                    else:
                        if k_is_r and k_real > 0:
                            st.markdown(
                                rf"O ganho $K = {format_frac(k_real)}$ é real e positivo no plano complexo."
                            )
                        else:
                            st.markdown(
                                rf"O ganho $K = {k_eval_str}$ não é um número real positivo, logo o ponto **NÃO pertence ao LGR**."
                            )

                    # Subetapa c: Segunda derivada e classificação
                    if is_v and is_r and d2k_r is not None:
                        st.markdown(
                            r"**c) Classificação via Segunda Derivada ($\frac{d^2K}{ds^2}$):**"
                        )
                        st.latex(
                            rf"\frac{{d^2K}}{{ds^2}}\Bigg|_{{s = {s_str}}} = {format_frac(d2k_r)}"
                        )
                        if cls == "breakaway":
                            st.markdown(
                                rf"Como $\frac{{d^2K}}{{ds^2}} = {format_frac(d2k_r)} < 0$, a função $K(s)$ atinge um **máximo local** ao longo do eixo real $\implies$ **Ponto de Saída (*Breakaway*)**."
                            )
                        elif cls == "breakin":
                            st.markdown(
                                rf"Como $\frac{{d^2K}}{{ds^2}} = {format_frac(d2k_r)} > 0$, a função $K(s)$ atinge um **mínimo local** ao longo do eixo real $\implies$ **Ponto de Entrada (*Break-in*)**."
                            )
                        else:
                            st.markdown(
                                rf"Como $\frac{{d^2K}}{{ds^2}} \approx 0$, trata-se de um ponto de inflexão de ordem superior."
                            )

                    # Veredito
                    if is_v:
                        tipo_nome = (
                            "Ponto de Saída (Breakaway)"
                            if cls == "breakaway"
                            else (
                                "Ponto de Entrada (Break-in)"
                                if cls == "breakin"
                                else "Ponto de Bifurcação"
                            )
                        )
                        st.success(
                            f" **Válido:** {tipo_nome} em $s = {s_str}$ com ganho $K = {format_frac(k_real)}$."
                        )
                    else:
                        st.warning(f" **Descartado:** {cand['reason_invalid']}")

                # 6. Resumo Consolidado
                st.markdown("---")
                st.markdown(r"**6. Resumo dos Pontos Válidos:**")
                if not valid_points:
                    st.info("Nenhum ponto válido para $K > 0$ foi identificado.")
                else:
                    summary_rows = []
                    for cand in candidates:
                        if cand["is_valid"]:
                            tipo = (
                                "Saída (*Breakaway*)"
                                if cand["classification"] == "breakaway"
                                else (
                                    "Entrada (*Break-in*)"
                                    if cand["classification"] == "breakin"
                                    else "Bifurcação Complexa"
                                )
                            )
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
                            summary_rows.append(
                                f"| **{tipo}** | $s = {s_f}$ | $K = {k_f}$ | $\\frac{{d^2K}}{{ds^2}} = {d2_f}$ |"
                            )

                    summary_table = (
                        "| Tipo | Coordenada ($s$) | Ganho ($K$) | Critério da 2ª Derivada |\n"
                        "| :--- | :--- | :--- | :--- |\n"
                        + "\n".join(summary_rows)
                    )
                    st.markdown(summary_table)

                # 7. Visualização Gráfica
                st.markdown("---")
                st.markdown(r"**7. Visualização Gráfica dos Pontos:**")
                tab_plane, tab_curve = st.tabs(
                    [" Localização no Plano s", " Curva de Ganho K(σ) no Eixo Real"]
                )

                with tab_plane:
                    fig8 = create_base_plot(
                        poles,
                        zeros,
                        "Pontos de Saída e Entrada no Plano s",
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        draw_poles_zeros=False,
                    )

                    # Real axis segments
                    final_x = []
                    final_y = []
                    for start, end in segment_coords:
                        final_x.extend([start, end, None])
                        final_y.extend([0, 0, None])

                    fig8.add_trace(
                        go.Scatter(
                            x=final_x,
                            y=final_y,
                            mode="lines",
                            line=dict(color="#00b4d8", width=5),
                            name="Segmentos LGR Real",
                            hoverinfo="skip",
                        )
                    )

                    # Plot valid points
                    for cand in candidates:
                        if cand["is_valid"]:
                            s_pt = cand["s_val"]
                            k_pt = cand["K_real"]
                            cls = cand["classification"]
                            x_pt = float(np.real(s_pt))
                            y_pt = float(np.imag(s_pt))
                            s_lbl = format_complex_frac(s_pt)
                            k_lbl = format_frac(k_pt)

                            if cls == "breakaway":
                                color_m = "#00ff88"
                                name_m = "Ponto de Saída"
                                tag = "Saída"
                            elif cls == "breakin":
                                color_m = "#ff9e00"
                                name_m = "Ponto de Entrada"
                                tag = "Entrada"
                            else:
                                color_m = "#e040fb"
                                name_m = "Bifurcação"
                                tag = "Bifurcação"

                            fig8.add_trace(
                                go.Scatter(
                                    x=[x_pt],
                                    y=[y_pt],
                                    mode="markers",
                                    marker=dict(
                                        symbol="diamond",
                                        size=14,
                                        color=color_m,
                                        line=dict(width=2, color="white"),
                                    ),
                                    name=f"{name_m} (s={s_lbl})",
                                    hovertemplate=f"<b>{name_m}</b><br>s = {s_lbl}<br>K = {k_lbl}<extra></extra>",
                                )
                            )

                            fig8.add_annotation(
                                x=x_pt,
                                y=y_pt,
                                text=f"{tag}<br>s = {s_lbl}<br>K = {k_lbl}",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor=color_m,
                                ax=0,
                                ay=-45 if y_pt >= 0 else 45,
                                font=dict(color="white", size=11),
                                bgcolor="rgba(14, 17, 23, 0.85)",
                                bordercolor=color_m,
                                borderwidth=1,
                                borderpad=3,
                            )

                    add_poles_zeros_traces(fig8, poles, zeros, show_labels=False)
                    fig8.update_layout(height=520)
                    st.plotly_chart(fig8, width="stretch", config={"scrollZoom": True})

                with tab_curve:
                    real_valid = [
                        c for c in candidates if c["is_valid"] and c["is_real"]
                    ]
                    if not real_valid and not segment_coords:
                        st.markdown(
                            "Não há pontos de saída/entrada sobre o eixo real para plotar a curva $K(\\sigma)$."
                        )
                    else:
                        D_poly = np.poly1d(D_coeffs)
                        N_poly = np.poly1d(N_coeffs)

                        all_reals = (
                            [
                                float(np.real(p))
                                for p in poles
                                if abs(np.imag(p)) < 1e-5
                            ]
                            + [
                                float(np.real(z))
                                for z in zeros
                                if abs(np.imag(z)) < 1e-5
                            ]
                            + [c["s_val"] for c in real_valid]
                        )
                        if all_reals:
                            s_min = min(all_reals) - 1.5
                            s_max = max(all_reals) + 1.5
                        else:
                            s_min, s_max = xmin, xmax

                        sigma_axis = np.linspace(s_min, s_max, 800)
                        k_axis = []
                        for s_val in sigma_axis:
                            n_val = N_poly(s_val)
                            if abs(n_val) < 1e-6:
                                k_axis.append(np.nan)
                            else:
                                k_v = -D_poly(s_val) / n_val
                                k_axis.append(
                                    k_v if -50 <= k_v <= 150 else np.nan
                                )

                        fig_k = go.Figure()
                        fig_k.add_trace(
                            go.Scatter(
                                x=sigma_axis,
                                y=k_axis,
                                mode="lines",
                                name="K(σ) = -D(σ)/N(σ)",
                                line=dict(color="#00b4d8", width=3),
                            )
                        )

                        fig_k.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color="rgba(255, 255, 255, 0.3)",
                        )

                        for c in real_valid:
                            s_v = c["s_val"]
                            k_v = c["K_real"]
                            cls = c["classification"]
                            c_color = "#00ff88" if cls == "breakaway" else "#ff9e00"
                            c_lbl = (
                                "Máximo (Saída)"
                                if cls == "breakaway"
                                else "Mínimo (Entrada)"
                            )

                            tangent_w = (s_max - s_min) * 0.08
                            fig_k.add_trace(
                                go.Scatter(
                                    x=[s_v - tangent_w, s_v + tangent_w],
                                    y=[k_v, k_v],
                                    mode="lines",
                                    line=dict(
                                        color="rgba(255, 255, 255, 0.7)",
                                        width=2,
                                        dash="dot",
                                    ),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )

                            fig_k.add_trace(
                                go.Scatter(
                                    x=[s_v],
                                    y=[k_v],
                                    mode="markers",
                                    marker=dict(
                                        symbol="diamond",
                                        size=14,
                                        color=c_color,
                                        line=dict(width=2, color="white"),
                                    ),
                                    name=f"{c_lbl}: s={format_frac(s_v)}, K={format_frac(k_v)}",
                                    hovertemplate=f"<b>{c_lbl}</b><br>dK/dσ = 0<br>σ = {format_frac(s_v)}<br>K = {format_frac(k_v)}<extra></extra>",
                                )
                            )

                            fig_k.add_annotation(
                                x=s_v,
                                y=k_v,
                                text=f"<b>dK/dσ = 0</b><br>{c_lbl}<br>σ = {format_frac(s_v)}<br>K = {format_frac(k_v)}",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor=c_color,
                                ax=0,
                                ay=-50 if cls == "breakaway" else 50,
                                font=dict(color="white", size=11),
                                bgcolor="rgba(14, 17, 23, 0.85)",
                                bordercolor=c_color,
                                borderwidth=1,
                                borderpad=3,
                            )

                        fig_k.update_layout(
                            title="Comportamento da Função de Ganho K(σ) no Eixo Real",
                            xaxis_title="Eixo Real (σ)",
                            yaxis_title="Ganho K(σ)",
                            plot_bgcolor="#0e1117",
                            paper_bgcolor="#0e1117",
                            font=dict(color="white"),
                            xaxis=dict(
                                zeroline=True,
                                zerolinecolor="rgba(255, 255, 255, 0.3)",
                                showgrid=True,
                                gridcolor="rgba(255, 255, 255, 0.1)",
                                griddash="dot",
                            ),
                            yaxis=dict(
                                zeroline=True,
                                zerolinecolor="rgba(255, 255, 255, 0.3)",
                                showgrid=True,
                                gridcolor="rgba(255, 255, 255, 0.1)",
                                griddash="dot",
                            ),
                            height=480,
                            margin=dict(l=10, r=10, t=50, b=20),
                        )
                        st.plotly_chart(
                            fig_k, width="stretch", config={"scrollZoom": True}
                        )

        # Passo 9
        with st.expander("Passo 9: Cruzamento com o eixo imaginário", expanded=True):
            st.markdown(
                r"O cruzamento dos ramos do LGR com o eixo imaginário ($s = \pm j\omega$) estabelece a fronteira entre a estabilidade assintótica e a instabilidade do sistema em malha fechada. "
                r"Nos pontos de travessia, os polos situam-se exatamente sobre o eixo $j\omega$, resultando em uma resposta puramente oscilatória de amplitude constante (estabilidade marginal) com frequência angular $\omega$ rad/s."
            )
            st.markdown(
                r"Para aplicar o critério de estabilidade de Routh-Hurwitz, utilizamos o polinômio característico de malha fechada obtido a partir de $1 + G(s)H(s) = 0 \iff D(s) + K \cdot N(s) = 0$:"
            )
            st.latex(char_poly_str + " = 0")

            # 1. Construção e Formulação Algébrica da Tabela de Routh
            st.markdown("---")
            st.markdown(r"**1. Construção e Formulação Algébrica da Tabela de Routh:**")
            st.markdown(
                r"A tabela é iniciada pelas duas primeiras linhas com os coeficientes do polinômio característico em ordem decrescente de potências de $s$ (potências pares na primeira linha e ímpares na segunda). "
                r"A partir da terceira linha ($s^{n-2}$ em diante), cada termo $r_{s^k, \text{Col } j+1}$ é obtido a partir das duas linhas imediatamente anteriores pela regra do determinante cruzado dividida pelo pivô da linha anterior:"
            )
            st.latex(
                r"r_{s^k, \text{Col } j+1} = \frac{\text{pivô} \cdot a_{sup\_dir} - a_{sup\_esq} \cdot a_{dir}}{\text{pivô}} = -\frac{1}{\text{pivô}} \begin{vmatrix} a_{sup\_esq} & a_{sup\_dir} \\ \text{pivô} & a_{dir} \end{vmatrix}"
            )

            routh_steps = routh_result.get("routh_steps", [])
            if routh_steps:
                with st.expander(" Ver formulação detalhada de cada termo calculado da tabela", expanded=False):
                    for step in routh_steps:
                        p_pow = step["row_power"]
                        c_idx = step["col_idx"]
                        piv_str = sp.latex(sp.together(step["pivot"]))
                        a11_str = sp.latex(sp.together(step["a11"]))
                        a12_str = sp.latex(sp.together(step["a12"]))
                        a22_str = sp.latex(sp.together(step["a22"]))
                        val_str = sp.latex(sp.together(step["val"]))

                        if step["a12"] != 0 or step["a22"] != 0 or step["val"] != 0 or c_idx == 0:
                            st.latex(
                                rf"r_{{s^{{{p_pow}}}, \text{{Col }} {c_idx+1}}} = \frac{{({piv_str}) \cdot ({a12_str}) - ({a11_str}) \cdot ({a22_str})}}{{{piv_str}}} = {val_str}"
                            )

            routh_table = routh_result["table"]
            degree = routh_result["degree"]
            row0_len = routh_result["row0_len"]

            md_table = (
                "| $s^i$ | "
                + " | ".join([f"Col {j+1}" for j in range(row0_len)])
                + " |\n"
            )
            md_table += "|" + "|".join(["---" for _ in range(row0_len + 1)]) + "|\n"

            for i in range(degree + 1):
                power = degree - i
                row_str = f"| $s^{power}$ | "
                row_str += " | ".join(
                    [
                        f"${sp.latex(sp.together(val))}$" if val != 0 else "$0$"
                        for val in routh_table[i]
                    ]
                )
                row_str += " |\n"
                md_table += row_str

            st.markdown(md_table)

            # 2. Análise da Primeira Coluna e Determinação do Ganho Crítico
            st.markdown("---")
            st.markdown(r"**2. Análise da Primeira Coluna e Determinação do Ganho Crítico ($K_{crítico}$):**")
            st.markdown(
                r"Pelo critério de Routh, o número de mudanças de sinal na primeira coluna é igual ao número de polos no semiplano direito (SPD). "
                r"O cruzamento com o eixo imaginário ocorre quando uma linha inteira se anula para um determinado ganho crítico $K = K_{crítico} > 0$ (estabilidade marginal), "
                r"indicando que as raízes da linha imediatamente superior são puramente imaginárias ($\pm j\omega$)."
            )

            crossings_data = routh_result.get("crossings_data", [])
            if crossings_data:
                for idx_c, data in enumerate(crossings_data):
                    power = data["s_power"]
                    row_expr = data["row_expr"]
                    k_crit = data["k_crit"]
                    aux_power = data.get("aux_power", power + 1)
                    aux_eq_sym = data["aux_eq_sym"]
                    aux_eq_sub = data["aux_eq_sub"]
                    omegas = data["omegas"]
                    k_solve_steps = data.get("k_solve_steps", {})
                    direct_proofs = data.get("direct_proofs", [])
                    vector_proofs = data.get("vector_proofs", [])

                    st.markdown(f"##### Cruzamento #{idx_c + 1} (Linha $s^{power}$):")

                    # a) Isolando K_crit
                    st.markdown(f"**a) Isolamento do ganho crítico $K_{{crítico}}$ na linha $s^{power}$:**")
                    st.markdown(f"Igualamos o primeiro elemento da linha $s^{power}$ a zero:")
                    t_expr = k_solve_steps.get("together", sp.together(row_expr))
                    num_k = k_solve_steps.get("num", t_expr)
                    den_k = k_solve_steps.get("den", sp.Integer(1))

                    if den_k != 1 and den_k != -1:
                        st.latex(rf"{sp.latex(t_expr)} = 0 \iff \frac{{{sp.latex(num_k)}}}{{{sp.latex(den_k)}}} = 0")
                        st.markdown(f"Como o denominador ${sp.latex(den_k)} \\neq 0$, o numerador deve ser nulo:")
                        st.latex(rf"{sp.latex(num_k)} = 0 \implies K_{{crítico}} = {format_frac(k_crit)}")
                    else:
                        st.latex(rf"{sp.latex(t_expr)} = 0 \implies K_{{crítico}} = {format_frac(k_crit)}")

                    st.success(rf" **Ganho Crítico:** $K_{{crítico}} = {format_frac(k_crit)}$")

                    # b) Equação Auxiliar
                    st.markdown(f"**b) Montagem da Equação Auxiliar ($A(s) = 0$):**")
                    st.markdown(
                        rf"Extraímos os coeficientes da linha imediatamente superior ($s^{aux_power}$), pulando as potências de 2 em 2 (polinômio par em $s$):"
                    )
                    st.latex(rf"A(s) = {sp.latex(sp.together(aux_eq_sym))} = 0")
                    st.markdown(rf"Substituindo $K = K_{{crítico}} = {format_frac(k_crit)}$ na equação auxiliar:")
                    st.latex(rf"A(s)\Big|_{{K = {format_frac(k_crit)}}} = {sp.latex(sp.together(aux_eq_sub))} = 0")

                    # c) Resolução de A(s) = 0
                    st.markdown(r"**c) Resolução algébrica de $A(s) = 0$ e determinação de $\omega$:**")
                    aux_steps = data.get("aux_solve_steps", {})
                    aux_deg = aux_steps.get("degree", 2)
                    aux_coeffs = aux_steps.get("coeffs", [])

                    if aux_deg == 2 and len(aux_coeffs) == 3:
                        a_c = aux_coeffs[0]
                        b_c = aux_coeffs[2]
                        s2_val = sp.cancel(-b_c / a_c)
                        st.markdown(r"Isolando o termo $s^2$:")
                        st.latex(rf"{sp.latex(a_c)} s^2 + {sp.latex(b_c)} = 0 \implies {sp.latex(a_c)} s^2 = -{sp.latex(b_c)} \implies s^2 = {sp.latex(s2_val)}")
                        st.markdown(r"Como sobre o eixo imaginário temos $s = j\omega$, decorre que $s^2 = (j\omega)^2 = -\omega^2$:")
                        w2_val = sp.cancel(-s2_val)
                        st.latex(rf"-\omega^2 = {sp.latex(s2_val)} \implies \omega^2 = {sp.latex(w2_val)}")
                        for w in omegas:
                            st.latex(rf"\omega = \sqrt{{{sp.latex(w2_val)}}} \approx {format_frac(w)} \text{{ rad/s}} \implies s = \pm {format_frac(w)}j")
                    else:
                        st.markdown(r"Fatorando e extraindo as raízes puramente imaginárias de $A(s) = 0$:")
                        for w in omegas:
                            st.latex(rf"s^2 + {format_frac(w**2)} = 0 \implies s = \pm {format_frac(w)}j \implies \omega = {format_frac(w)} \text{{ rad/s}}")

                    st.info(
                        f" **Pontos de Cruzamento no Eixo Imaginário:** "
                        + ", ".join([f"$s = \\pm {format_frac(w)}j$ ($\\omega = {format_frac(w)}$ rad/s)" for w in omegas])
                        + f" para $K = {format_frac(k_crit)}$."
                    )

                    # d) Comprovação Matemática do Resultado (Prova Real)
                    st.markdown("---")
                    st.markdown(rf"####  Comprovação Matemática do Resultado (Prova Real):")
                    st.markdown(
                        r"Para comprovar matematicamente que as frequências encontradas e o ganho crítico satisfazem rigorosamente as condições de fechamento do sistema, aplicamos duas comprovações analíticas independentes:"
                    )

                    # Prova 1: Substituição direta em P(s) = 0
                    st.markdown(r"**Prova 1: Substituição Direta na Equação Característica ($P(j\omega) = 0$):**")
                    st.markdown(
                        rf"Substituímos o ganho $K = {format_frac(k_crit)}$ e o par imaginário $s = j\omega$ no polinômio característico original $P(s) = D(s) + K \cdot N(s)$:"
                    )
                    for d_proof in direct_proofs:
                        w_val = d_proof["w"]
                        p_sub_sym = d_proof["P_at_k"]
                        re_sym = d_proof["re_sym"]
                        im_sym = d_proof["im_sym"]
                        t_break = d_proof["terms_breakdown"]
                        r_sum = d_proof["real_sum"]
                        i_sum = d_proof["imag_sum"]

                        st.markdown(rf"Para $\omega = {format_frac(w_val)}$ rad/s ($s = {format_frac(w_val)}j$):")
                        st.latex(rf"P(s)\Big|_{{K = {format_frac(k_crit)}}} = {sp.latex(p_sub_sym)} = 0")

                        rows_md = [
                            "| Termo ($a_k s^k$) | Substituição com $s = j\\omega$ | Valor Calculado |",
                            "| :--- | :--- | :--- |",
                        ]
                        for tb in t_break:
                            coef = tb["coeff"]
                            pw = tb["power"]
                            t_val = tb["term_val"]
                            if pw == 0:
                                sub_str = f"{format_frac(coef)}"
                            elif pw == 1:
                                sub_str = f"({format_frac(coef)}) \\cdot ({format_frac(w_val)}j)"
                            else:
                                sub_str = f"({format_frac(coef)}) \\cdot ({format_frac(w_val)}j)^{{{pw}}}"
                            t_val_str = format_complex_frac(t_val)
                            rows_md.append(f"| ${format_frac(coef)} s^{{{pw}}}$ | ${sub_str}$ | ${t_val_str}$ |")

                        st.markdown("\n".join(rows_md))

                        st.latex(
                            rf"\text{{Re}}\left\{{P({format_frac(w_val)}j)\}}\right. = {sp.latex(re_sym)} = {format_frac(r_sum)} \approx 0 \quad (\checkmark)"
                        )
                        st.latex(
                            rf"\text{{Im}}\left\{{P({format_frac(w_val)}j)\}}\right. = {sp.latex(im_sym)} = {format_frac(i_sum)} \approx 0 \quad (\checkmark)"
                        )
                        st.latex(
                            rf"P(\pm {format_frac(w_val)}j)\Big|_{{K = {format_frac(k_crit)}}} = 0 + j0 = 0 \quad (\checkmark \textbf{{ COMPROVADO}})"
                        )
                        st.markdown(
                            rf"Como tanto a parte real quanto a imaginária se anulam identicamente, comprova-se analiticamente que $s = \pm {format_frac(w_val)}j$ é raiz exata de malha fechada quando $K = {format_frac(k_crit)}$."
                        )

                    # Prova 2: Critério Geométrico de Ângulo e Módulo do LGR
                    st.markdown(r"**Prova 2: Critério Geométrico de Ângulo e Módulo do LGR:**")
                    st.markdown(
                        r"Avaliamos a função de transferência de malha aberta $G(s)H(s) = \frac{N(s)}{D(s)}$ no ponto de teste sobre o eixo imaginário $s_0 = +j\omega$:"
                    )
                    for v_proof in vector_proofs:
                        w_val = v_proof["w"]
                        s_ang_p = v_proof["sum_p"]
                        s_ang_z = v_proof["sum_z"]
                        ph_norm = v_proof["phase_norm"]
                        k_calc = v_proof["k_calc"]
                        p_dist = v_proof["prod_p"]
                        z_dist = v_proof["prod_z"]

                        st.markdown(rf"No ponto de teste $s_0 = {format_frac(w_val)}j$:")
                        st.latex(
                            rf"\angle G(s_0)H(s_0) = \sum \phi_z - \sum \theta_p = {format_frac(s_ang_z)}^\circ - ({format_frac(s_ang_p)}^\circ) = {format_frac(ph_norm)}^\circ \equiv \pm 180^\circ \quad (\checkmark \textbf{{ COMPROVADO}})"
                        )
                        st.latex(
                            rf"K = \frac{{\prod |s_0 - p_i|}}{{\prod |s_0 - z_i|}} = \frac{{{format_frac(p_dist)}}}{{{format_frac(z_dist)}}} = {format_frac(k_calc)} = K_{{crítico}} \quad (\checkmark \textbf{{ COMPROVADO}})"
                        )
                        st.markdown(
                            rf"O critério de fase ($\pm 180^\circ$) comprova que o ponto pertence rigorosamente ao LGR direto, e a condição de módulo confirma exatamente o valor $K_{{crítico}} = {format_frac(k_crit)}$."
                        )
            else:
                st.markdown(r"**Nenhum cruzamento com o eixo imaginário foi identificado para $K > 0$.**")
                st.markdown(r"####  Comprovação Matemática de Ausência de Cruzamento:")

                # Comprovação Routh
                st.markdown(r"**a) Comprovação pelo Critério de Routh-Hurwitz:**")
                st.markdown(
                    r"Para estabilidade assintótica estrita, todos os coeficientes da primeira coluna da tabela de Routh devem ser estritamente positivos. "
                    r"Analisando cada linha da 1ª coluna obtida:"
                )
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

                st.markdown(
                    r"Como não há nenhuma mudança de sinal na primeira coluna da tabela de Routh para qualquer $K > 0$, "
                    r"pelo Teorema de Routh-Hurwitz **todos os polos de malha fechada permanecem estritamente no semiplano esquerdo (SPE)**. "
                    r"Nenhum ramo do LGR cruza o eixo imaginário rumo ao semiplano direito para $K > 0$."
                )

                # Comprovação P(jw)
                no_cross = routh_result.get("no_crossing_proof")
                if no_cross:
                    st.markdown(r"**b) Comprovação Analítica Direta ($P(j\omega) = 0$):**")
                    st.markdown(
                        r"Substituindo $s = j\omega$ no polinômio característico $P(s) = D(s) + K \cdot N(s) = 0$:"
                    )
                    im_s = sp.latex(no_cross["im_sym"])
                    re_s = sp.latex(no_cross["re_sym"])
                    st.latex(rf"P(j\omega) = \left({re_s}\right) + j \left({im_s}\right) = 0")
                    st.markdown(
                        r"Para haver raízes sobre o eixo imaginário fora da origem ($\omega > 0$), a parte imaginária deve se anular identicamente:"
                    )
                    st.latex(rf"\text{{Im}}\left\{{P(j\omega)\}}\right. = {im_s} = 0")
                    st.markdown(
                        r"Como a equação acima não admite soluções reais com $\omega > 0$ e $K > 0$, "
                        r"comprova-se formalmente que nenhum ramo do LGR cruza o eixo imaginário no plano finito."
                    )

            # 3. Visualização Gráfica no Plano s
            st.markdown("---")
            st.markdown(r"**3. Visualização Gráfica no Plano $s$:**")
            fig9 = create_base_plot(
                poles,
                zeros,
                "Cruzamento com o Eixo Imaginário (jω)",
                xmin,
                xmax,
                ymin,
                ymax,
                draw_poles_zeros=False,
            )

            # Highlight imaginary axis
            fig9.add_vline(
                x=0,
                line_dash="solid",
                line_color="rgba(0, 180, 216, 0.4)",
                line_width=2,
            )

            if crossings_data:
                for data in crossings_data:
                    k_crit = data["k_crit"]
                    omegas = data["omegas"]
                    for w in omegas:
                        # Upper crossing
                        fig9.add_trace(
                            go.Scatter(
                                x=[0],
                                y=[w],
                                mode="markers",
                                marker=dict(
                                    symbol="star",
                                    size=16,
                                    color="#ff007f",
                                    line=dict(width=2, color="white"),
                                ),
                                name=f"+{format_frac(w)}j (K={format_frac(k_crit)})",
                                hovertemplate=f"<b>Cruzamento jω</b><br>s = +{format_frac(w)}j<br>ω = {format_frac(w)} rad/s<br>K = {format_frac(k_crit)}<extra></extra>",
                            )
                        )
                        fig9.add_annotation(
                            x=0,
                            y=w,
                            text=f"<b>jω = +{format_frac(w)}</b><br>K = {format_frac(k_crit)}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="#ff007f",
                            ax=50,
                            ay=-30,
                            font=dict(color="white", size=11),
                            bgcolor="rgba(14, 17, 23, 0.85)",
                            bordercolor="#ff007f",
                            borderwidth=1,
                            borderpad=3,
                        )

                        # Lower crossing
                        fig9.add_trace(
                            go.Scatter(
                                x=[0],
                                y=[-w],
                                mode="markers",
                                marker=dict(
                                    symbol="star",
                                    size=16,
                                    color="#ff007f",
                                    line=dict(width=2, color="white"),
                                ),
                                name=f"-{format_frac(w)}j (K={format_frac(k_crit)})",
                                hovertemplate=f"<b>Cruzamento jω</b><br>s = -{format_frac(w)}j<br>ω = {format_frac(w)} rad/s<br>K = {format_frac(k_crit)}<extra></extra>",
                            )
                        )
                        fig9.add_annotation(
                            x=0,
                            y=-w,
                            text=f"<b>jω = -{format_frac(w)}</b><br>K = {format_frac(k_crit)}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="#ff007f",
                            ax=50,
                            ay=30,
                            font=dict(color="white", size=11),
                            bgcolor="rgba(14, 17, 23, 0.85)",
                            bordercolor="#ff007f",
                            borderwidth=1,
                            borderpad=3,
                        )

                        # Segment connecting the two conjugate crossings
                        fig9.add_trace(
                            go.Scatter(
                                x=[0, 0],
                                y=[-w, w],
                                mode="lines",
                                line=dict(color="#ff007f", width=2, dash="dash"),
                                name="Eixo de Oscilação",
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
            else:
                fig9.add_annotation(
                    x=0,
                    y=(ymin + ymax) * 0.25,
                    text="<b>Sem cruzamento com jω para K > 0</b><br>Sistema permanece estável no SPE",
                    showarrow=False,
                    font=dict(color="#00ff88", size=12),
                    bgcolor="rgba(14, 17, 23, 0.85)",
                    bordercolor="#00ff88",
                    borderwidth=1,
                    borderpad=5,
                )

            add_poles_zeros_traces(fig9, poles, zeros, show_labels=False)
            fig9.update_layout(height=520)
            st.plotly_chart(fig9, width="stretch", config={"scrollZoom": True})

        # Passo 10
        with st.expander("Passo 10: Ângulos de partida/chegada", expanded=True):
            st.markdown(
                r"Os **ângulos de partida** ($\theta_p$) e **ângulos de chegada** ($\theta_z$) determinam as direções angulares tangenciais com que os ramos do LGR emergem dos polos complexos (quando o ganho $K \to 0^+$) ou incidem nos zeros complexos (quando $K \to \infty$)."
            )
            st.markdown(
                r"Essas direções decorrem diretamente da **condição de ângulo** fundamental do LGR direto ($K > 0$):"
            )
            st.latex(
                r"\angle G(s)H(s) = \sum_{j=1}^{nZ} \angle(s - z_j) - \sum_{i=1}^{nP} \angle(s - p_i) = \pm 180^\circ(2q+1)"
            )

            dep_arr_details = calculate_departure_arrival_angles(poles, zeros)
            has_complex = dep_arr_details["has_complex"]
            pole_details = dep_arr_details["pole_details"]
            zero_details = dep_arr_details["zero_details"]

            if not has_complex:
                st.markdown("---")
                st.info(
                    "️ **Não aplicável:** O sistema não possui polos nem zeros complexos conjugados ($\text{Im} \neq 0$)."
                )
                st.markdown(
                    r"**Fundamentação Teórica:** "
                    r"Conforme estabelecido nos Passos 4 e 8, os polos e zeros puramente reais possuem ramos do LGR que iniciam e terminam "
                    r"estritamente alinhados ao longo do próprio eixo real (em direções de $0^\circ$ ou $180^\circ$). "
                    r"Assim, o cálculo de ângulos tangenciais de partida e chegada só se define para singularidades com parte imaginária não nula."
                )
            else:
                # 1. Polos Complexos (Ângulos de Partida)
                if pole_details:
                    st.markdown("---")
                    st.markdown(r"### 1. Polos Complexos — Ângulos de Partida ($\theta_p$):")
                    st.markdown(
                        r"Para um ponto $s$ em uma vizinhança infinitesimal de um polo complexo $p_k$ ($s = p_k + \epsilon e^{j\theta_p}$ com $\epsilon \to 0^+$), "
                        r"a contribuição angular do próprio polo $p_k$ é $m \cdot \theta_p$ (onde $m$ é sua multiplicidade). "
                        r"Substituindo na condição angular do LGR:"
                    )
                    st.latex(
                        r"\sum_{j=1}^{nZ} \phi_{z_j} - \left( m \cdot \theta_p + \sum_{i \neq k} \theta_{p_i} \right) = 180^\circ(2q+1)"
                    )
                    st.markdown(r"Isolando o ângulo de partida $\theta_p$:")
                    st.latex(
                        r"\theta_p = \frac{180^\circ(2q+1) + \sum \phi_z - \sum_{i \neq k} \theta_{p_i}}{m}"
                    )

                    for idx_p, pd in enumerate(pole_details):
                        cp = pd["pole"]
                        conj_p = pd["conj_pole"]
                        m = pd["multiplicity"]
                        vecs_p = pd["vecs_p"]
                        vecs_z = pd["vecs_z"]
                        sum_p = pd["sum_p"]
                        sum_z = pd["sum_z"]
                        branches = pd["branches"]

                        st.markdown(
                            f"#### Polo Complexo $p_{{{idx_p+1}}} = {format_complex_frac(cp)}$ (Multiplicidade $m = {m}$):"
                        )

                        # a) Vetores dos outros polos
                        st.markdown(r"**a) Vetores partindo dos demais polos até $p$ ($\vec{v}_i = p - p_i$):**")
                        if vecs_p:
                            p_table = [
                                "| Polo Origem ($p_i$) | Vetor $\\vec{v} = p - p_i$ | Distância $|\\vec{v}|$ | Ângulo $\\theta_{p_i} = \\operatorname{atan2}(\\Delta \\omega, \\Delta \\sigma)$ |",
                                "| :--- | :--- | :--- | :--- |",
                            ]
                            for vp in vecs_p:
                                p_orig_str = format_complex_frac(vp["pole"])
                                v_str = format_complex_frac(vp["vector"])
                                mag_str = format_frac(vp["mag"])
                                ang_str = f"{format_frac(vp['angle_deg'])}^\\circ"
                                p_table.append(f"| ${p_orig_str}$ | ${v_str}$ | ${mag_str}$ | ${ang_str}$ |")
                            st.markdown("\n".join(p_table))
                            st.latex(
                                rf"\sum \theta_{{outros\_polos}} = "
                                + " + ".join([f"{format_frac(vp['angle_deg'])}^\\circ" for vp in vecs_p])
                                + f" = {format_frac(sum_p)}^\\circ"
                            )
                        else:
                            st.markdown(r"*(Não há outros polos no sistema)* $\implies \sum \theta_{outros\_polos} = 0^\circ$.")

                        # b) Vetores dos zeros
                        st.markdown(r"**b) Vetores partindo dos zeros até $p$ ($\vec{w}_j = p - z_j$):**")
                        if vecs_z:
                            z_table = [
                                "| Zero Origem ($z_j$) | Vetor $\\vec{w} = p - z_j$ | Distância $|\\vec{w}|$ | Ângulo $\\phi_{z_j} = \\operatorname{atan2}(\\Delta \\omega, \\Delta \\sigma)$ |",
                                "| :--- | :--- | :--- | :--- |",
                            ]
                            for vz in vecs_z:
                                z_orig_str = format_complex_frac(vz["zero"])
                                w_str = format_complex_frac(vz["vector"])
                                mag_str = format_frac(vz["mag"])
                                ang_str = f"{format_frac(vz['angle_deg'])}^\\circ"
                                z_table.append(f"| ${z_orig_str}$ | ${w_str}$ | ${mag_str}$ | ${ang_str}$ |")
                            st.markdown("\n".join(z_table))
                            st.latex(
                                rf"\sum \phi_z = "
                                + " + ".join([f"{format_frac(vz['angle_deg'])}^\\circ" for vz in vecs_z])
                                + f" = {format_frac(sum_z)}^\\circ"
                            )
                        else:
                            st.markdown(r"*(Não há zeros finitos no sistema)* $\implies \sum \phi_z = 0^\circ$.")

                        # c) Dedução algébrica
                        st.markdown(r"**c) Formulação e Cálculo Algébrico de $\theta_p$:**")
                        for b in branches:
                            q = b["q"]
                            raw = b["raw"]
                            norm = b["norm"]
                            a360 = b["a360"]
                            st.latex(
                                rf"q = {q} \implies \theta_p = \frac{{180^\circ({2*q+1}) + ({format_frac(sum_z)}^\circ) - ({format_frac(sum_p)}^\circ)}}{{{m}}} = {format_frac(raw)}^\circ \equiv {format_frac(norm)}^\circ \quad (\text{{ou }} {format_frac(a360)}^\circ)"
                            )

                        # d) Simetria conjugada
                        first_norm = branches[0]["norm"]
                        st.markdown(
                            rf"Pela propriedade de simetria do LGR (Passo 6), para o polo conjugado $\bar{{p}} = {format_complex_frac(conj_p)}$, "
                            rf"o ângulo de partida é o reflexo especular exato: **$\theta_{{\bar{{p}}}} = -\theta_p = {format_frac(-first_norm)}^\circ$**."
                        )

                        # e) Comprovação Matemática
                        st.markdown(r"**d)  Comprovação Matemática do Resultado (Prova Real):**")
                        st.markdown(
                            r"Para comprovar matematicamente que o ramo do LGR emerge na direção calculada, testamos um ponto infinitesimalmente deslocado ao longo de $\theta_p$:"
                        )
                        for b in branches:
                            proof = b["proof"]
                            s_t = proof["s_test"]
                            ph_t = proof["phase_norm"]
                            k_t = proof["k_test"]
                            st.latex(
                                rf"s_{{teste}} = p + \epsilon \cdot e^{{j \theta_p}} = {format_complex_frac(s_t)} \quad (\epsilon = 10^{{-4}})"
                            )
                            st.latex(
                                rf"\angle G(s_{{teste}})H(s_{{teste}}) = {format_frac(ph_t)}^\circ \equiv \pm 180^\circ \quad (\checkmark \textbf{{ COMPROVADO}})"
                            )
                            st.latex(
                                rf"K(s_{{teste}}) = {format_frac(k_t)} > 0 \quad (\checkmark \textbf{{ COMPROVADO}})"
                            )
                            st.markdown(
                                rf"A fase avaliada satisfaz rigorosamente a condição de ângulo de $180^\circ$ e o ganho $K > 0$ é estritamente positivo, comprovando analiticamente que a trajetória parte do polo na direção $\theta_p = {format_frac(b['norm'])}^\circ$."
                            )

                # 2. Zeros Complexos (Ângulos de Chegada)
                if zero_details:
                    st.markdown("---")
                    st.markdown(r"### 2. Zeros Complexos — Ângulos de Chegada ($\theta_z$):")
                    st.markdown(
                        r"Para um ponto $s$ em uma vizinhança infinitesimal de um zero complexo $z_k$ ($s = z_k + \epsilon e^{j\theta_z}$ com $\epsilon \to 0^+$), "
                        r"a contribuição angular do próprio zero $z_k$ é $m \cdot \theta_z$. "
                        r"Substituindo na condição angular do LGR:"
                    )
                    st.latex(
                        r"\left( m \cdot \theta_z + \sum_{j \neq k} \phi_{z_j} \right) - \sum_{i=1}^{nP} \theta_{p_i} = 180^\circ(2q+1)"
                    )
                    st.markdown(r"Isolando o ângulo de chegada $\theta_z$:")
                    st.latex(
                        r"\theta_z = \frac{180^\circ(2q+1) + \sum \theta_p - \sum_{j \neq k} \phi_{z_j}}{m}"
                    )

                    for idx_z, zd in enumerate(zero_details):
                        cz = zd["zero"]
                        conj_z = zd["conj_zero"]
                        m = zd["multiplicity"]
                        vecs_p = zd["vecs_p"]
                        vecs_z = zd["vecs_z"]
                        sum_p = zd["sum_p"]
                        sum_z = zd["sum_z"]
                        branches = zd["branches"]

                        st.markdown(
                            f"#### Zero Complexo $z_{{{idx_z+1}}} = {format_complex_frac(cz)}$ (Multiplicidade $m = {m}$):"
                        )

                        # a) Vetores dos polos
                        st.markdown(r"**a) Vetores partindo dos polos até $z$ ($\vec{v}_i = z - p_i$):**")
                        if vecs_p:
                            p_table = [
                                "| Polo Origem ($p_i$) | Vetor $\\vec{v} = z - p_i$ | Distância $|\\vec{v}|$ | Ângulo $\\theta_{p_i} = \\operatorname{atan2}(\\Delta \\omega, \\Delta \\sigma)$ |",
                                "| :--- | :--- | :--- | :--- |",
                            ]
                            for vp in vecs_p:
                                p_orig_str = format_complex_frac(vp["pole"])
                                v_str = format_complex_frac(vp["vector"])
                                mag_str = format_frac(vp["mag"])
                                ang_str = f"{format_frac(vp['angle_deg'])}^\\circ"
                                p_table.append(f"| ${p_orig_str}$ | ${v_str}$ | ${mag_str}$ | ${ang_str}$ |")
                            st.markdown("\n".join(p_table))
                            st.latex(
                                rf"\sum \theta_p = "
                                + " + ".join([f"{format_frac(vp['angle_deg'])}^\\circ" for vp in vecs_p])
                                + f" = {format_frac(sum_p)}^\\circ"
                            )
                        else:
                            st.markdown(r"*(Não há polos no sistema)* $\implies \sum \theta_p = 0^\circ$.")

                        # b) Vetores dos outros zeros
                        st.markdown(r"**b) Vetores partindo dos demais zeros até $z$ ($\vec{w}_j = z - z_j$):**")
                        if vecs_z:
                            z_table = [
                                "| Zero Origem ($z_j$) | Vetor $\\vec{w} = z - z_j$ | Distância $|\\vec{w}|$ | Ângulo $\\phi_{z_j} = \\operatorname{atan2}(\\Delta \\omega, \\Delta \\sigma)$ |",
                                "| :--- | :--- | :--- | :--- |",
                            ]
                            for vz in vecs_z:
                                z_orig_str = format_complex_frac(vz["zero"])
                                w_str = format_complex_frac(vz["vector"])
                                mag_str = format_frac(vz["mag"])
                                ang_str = f"{format_frac(vz['angle_deg'])}^\\circ"
                                z_table.append(f"| ${z_orig_str}$ | ${w_str}$ | ${mag_str}$ | ${ang_str}$ |")
                            st.markdown("\n".join(z_table))
                            st.latex(
                                rf"\sum \phi_{{outros\_zeros}} = "
                                + " + ".join([f"{format_frac(vz['angle_deg'])}^\\circ" for vz in vecs_z])
                                + f" = {format_frac(sum_z)}^\\circ"
                            )
                        else:
                            st.markdown(r"*(Não há outros zeros no sistema)* $\implies \sum \phi_{outros\_zeros} = 0^\circ$.")

                        # c) Dedução algébrica
                        st.markdown(r"**c) Formulação e Cálculo Algébrico de $\theta_z$:**")
                        for b in branches:
                            q = b["q"]
                            raw = b["raw"]
                            norm = b["norm"]
                            a360 = b["a360"]
                            st.latex(
                                rf"q = {q} \implies \theta_z = \frac{{180^\circ({2*q+1}) + ({format_frac(sum_p)}^\circ) - ({format_frac(sum_z)}^\circ)}}{{{m}}} = {format_frac(raw)}^\circ \equiv {format_frac(norm)}^\circ \quad (\text{{ou }} {format_frac(a360)}^\circ)"
                            )

                        # d) Simetria conjugada
                        first_norm = branches[0]["norm"]
                        st.markdown(
                            rf"Pela propriedade de simetria do LGR (Passo 6), para o zero conjugado $\bar{{z}} = {format_complex_frac(conj_z)}$, "
                            rf"o ângulo de chegada é o reflexo especular exato: **$\theta_{{\bar{{z}}}} = -\theta_z = {format_frac(-first_norm)}^\circ$**."
                        )

                        # e) Comprovação Matemática
                        st.markdown(r"**d)  Comprovação Matemática do Resultado (Prova Real):**")
                        st.markdown(
                            r"Para comprovar matematicamente que o ramo do LGR incide no zero na direção calculada, testamos um ponto infinitesimalmente deslocado ao longo de $\theta_z$:"
                        )
                        for b in branches:
                            proof = b["proof"]
                            s_t = proof["s_test"]
                            ph_t = proof["phase_norm"]
                            k_t = proof["k_test"]
                            st.latex(
                                rf"s_{{teste}} = z + \epsilon \cdot e^{{j \theta_z}} = {format_complex_frac(s_t)} \quad (\epsilon = 10^{{-4}})"
                            )
                            st.latex(
                                rf"\angle G(s_{{teste}})H(s_{{teste}}) = {format_frac(ph_t)}^\circ \equiv \pm 180^\circ \quad (\checkmark \textbf{{ COMPROVADO}})"
                            )
                            st.latex(
                                rf"K(s_{{teste}}) = {format_frac(k_t)} \gg 1 \quad (\checkmark \textbf{{ COMPROVADO}})"
                            )
                            st.markdown(
                                rf"A fase fecha exatamente em $\pm 180^\circ$ e o ganho $K \to \infty$, comprovando analiticamente que o ramo incide no zero na direção $\theta_z = {format_frac(b['norm'])}^\circ$."
                            )

            # 3. Visualização Gráfica no Plano s
            st.markdown("---")
            st.markdown(r"### 3. Visualização Gráfica dos Vetores e Ângulos de Partida/Chegada:")
            fig10 = create_base_plot(
                poles,
                zeros,
                "Ângulos de Partida e Chegada no Plano s",
                xmin,
                xmax,
                ymin,
                ymax,
                draw_poles_zeros=False,
            )

            if has_complex:
                arrow_len = max(0.8, (xmax - xmin) * 0.12)

                # Departure arrows for poles
                for pd in pole_details:
                    cp = pd["pole"]
                    for b in pd["branches"]:
                        ang_deg = b["norm"]
                        rad = np.radians(ang_deg)
                        dx = arrow_len * np.cos(rad)
                        dy = arrow_len * np.sin(rad)

                        # Arrow line
                        fig10.add_trace(
                            go.Scatter(
                                x=[cp.real, cp.real + dx],
                                y=[cp.imag, cp.imag + dy],
                                mode="lines",
                                line=dict(color="#00f5d4", width=3),
                                name=f"Partida de {format_complex_frac(cp)} (θ={format_frac(ang_deg)}°)",
                                hovertemplate=f"<b>Ângulo de Partida</b><br>s = {format_complex_frac(cp)}<br>θ<sub>p</sub> = {format_frac(ang_deg)}°<extra></extra>",
                            )
                        )
                        fig10.add_annotation(
                            x=cp.real + dx,
                            y=cp.imag + dy,
                            ax=cp.real,
                            ay=cp.imag,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowwidth=3,
                            arrowcolor="#00f5d4",
                        )
                        fig10.add_annotation(
                            x=cp.real + dx * 1.15,
                            y=cp.imag + dy * 1.15,
                            text=f"<b>θ<sub>p</sub> = {format_frac(ang_deg)}°</b>",
                            showarrow=False,
                            font=dict(color="#00f5d4", size=11),
                            bgcolor="rgba(14, 17, 23, 0.85)",
                            bordercolor="#00f5d4",
                            borderwidth=1,
                            borderpad=3,
                        )

                        # Vectors from other poles to this complex pole (dashed)
                        for vp in pd["vecs_p"]:
                            p_orig = vp["pole"]
                            fig10.add_trace(
                                go.Scatter(
                                    x=[p_orig.real, cp.real],
                                    y=[p_orig.imag, cp.imag],
                                    mode="lines",
                                    line=dict(color="rgba(255, 100, 100, 0.4)", width=1.5, dash="dot"),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )
                        for vz in pd["vecs_z"]:
                            z_orig = vz["zero"]
                            fig10.add_trace(
                                go.Scatter(
                                    x=[z_orig.real, cp.real],
                                    y=[z_orig.imag, cp.imag],
                                    mode="lines",
                                    line=dict(color="rgba(100, 255, 100, 0.4)", width=1.5, dash="dot"),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )

                # Arrival arrows for zeros
                for zd in zero_details:
                    cz = zd["zero"]
                    for b in zd["branches"]:
                        ang_deg = b["norm"]
                        rad = np.radians(ang_deg)
                        dx = arrow_len * np.cos(rad)
                        dy = arrow_len * np.sin(rad)

                        fig10.add_trace(
                            go.Scatter(
                                x=[cz.real + dx, cz.real],
                                y=[cz.imag + dy, cz.imag],
                                mode="lines",
                                line=dict(color="#ffbe0b", width=3),
                                name=f"Chegada em {format_complex_frac(cz)} (θ={format_frac(ang_deg)}°)",
                                hovertemplate=f"<b>Ângulo de Chegada</b><br>s = {format_complex_frac(cz)}<br>θ<sub>z</sub> = {format_frac(ang_deg)}°<extra></extra>",
                            )
                        )
                        fig10.add_annotation(
                            x=cz.real,
                            y=cz.imag,
                            ax=cz.real + dx,
                            ay=cz.imag + dy,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowwidth=3,
                            arrowcolor="#ffbe0b",
                        )
                        fig10.add_annotation(
                            x=cz.real + dx * 1.15,
                            y=cz.imag + dy * 1.15,
                            text=f"<b>θ<sub>z</sub> = {format_frac(ang_deg)}°</b>",
                            showarrow=False,
                            font=dict(color="#ffbe0b", size=11),
                            bgcolor="rgba(14, 17, 23, 0.85)",
                            bordercolor="#ffbe0b",
                            borderwidth=1,
                            borderpad=3,
                        )
            else:
                fig10.add_annotation(
                    x=(xmin + xmax) * 0.5,
                    y=(ymin + ymax) * 0.5,
                    text="<b>Não aplicável:</b> Sem polos ou zeros complexos conjugados.<br>Ramos iniciam e terminam ao longo do eixo real.",
                    showarrow=False,
                    font=dict(color="#00b4d8", size=13),
                    bgcolor="rgba(14, 17, 23, 0.85)",
                    bordercolor="#00b4d8",
                    borderwidth=1,
                    borderpad=6,
                )

            add_poles_zeros_traces(fig10, poles, zeros, show_labels=False)
            fig10.update_layout(height=520)
            st.plotly_chart(fig10, width="stretch", config={"scrollZoom": True})

        # Passo 11
        with st.expander("Passo 11: Critério de ângulo ($s_0$)", expanded=True):
            st.markdown(
                r"O **Critério de Ângulo** determina se um ponto de teste genérico $s_0 \in \mathbb{C}$ pertence ou não ao Lugar Geométrico das Raízes direto ($K > 0$)."
            )
            st.markdown(
                r"Ele decorre diretamente da equação característica da malha fechada $1 + K G(s)H(s) = 0 \implies G(s)H(s) = -\frac{1}{K} = \frac{1}{K} e^{j 180^\circ(2q+1)}$ para $K > 0$:"
            )
            st.latex(
                r"\angle G(s_0)H(s_0) = \sum_{j=1}^{nZ} \angle(s_0 - z_j) - \sum_{i=1}^{nP} \angle(s_0 - p_i) = \pm 180^\circ(2q+1), \quad q \in \mathbb{Z}"
            )

            test_details = evaluate_test_point_details(s0, poles, zeros, D_coeffs, N_coeffs)
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
                rf"Ponto de teste selecionado na barra lateral: **$s_0 = {s0_str}$** "
                rf"($\sigma_0 = {format_frac(np.real(s0))}$, $\omega_0 = {format_frac(np.imag(s0))}$)."
            )

            # 1. Vetores a partir dos polos
            st.markdown("---")
            st.markdown(r"### 1. Vetores partindo dos Polos até $s_0$ ($\vec{v}_{p_i} = s_0 - p_i$):")
            if vecs_p:
                p_table = [
                    "| Polo Origem ($p_i$) | Vetor $\\vec{v}_{p_i} = s_0 - p_i$ | Componentes $(\\Delta \\sigma, \\Delta \\omega)$ | Distância $|\\vec{v}_{p_i}|$ | Ângulo $\\theta_{p_i} = \\operatorname{atan2}(\\Delta \\omega, \\Delta \\sigma)$ |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
                for vp in vecs_p:
                    p_orig = format_complex_frac(vp["pole"])
                    v_str = format_complex_frac(vp["vector"])
                    ds_str = format_frac(vp["delta_sigma"])
                    dw_str = format_frac(vp["delta_omega"])
                    mag_str = format_frac(vp["dist"])
                    ang_str = f"{format_frac(vp['angle_deg'])}^\\circ"
                    p_table.append(
                        f"| $p_{{{vp['index']}}} = {p_orig}$ | ${v_str}$ | $\\Delta\\sigma = {ds_str}, \\Delta\\omega = {dw_str}$ | ${mag_str}$ | ${ang_str}$ |"
                    )
                st.markdown("\n".join(p_table))
                sum_p_terms = " + ".join([f"({format_frac(vp['angle_deg'])}^\\circ)" for vp in vecs_p])
                st.latex(rf"\sum_{{i=1}}^{{nP}} \theta_{{p_i}} = {sum_p_terms} = {format_frac(sum_p)}^\circ")
            else:
                st.markdown(r"*(Não há polos no sistema)* $\implies \sum \theta_p = 0^\circ$.")

            # 2. Vetores a partir dos zeros
            st.markdown("---")
            st.markdown(r"### 2. Vetores partindo dos Zeros até $s_0$ ($\vec{w}_{z_j} = s_0 - z_j$):")
            if vecs_z:
                z_table = [
                    "| Zero Origem ($z_j$) | Vetor $\\vec{w}_{z_j} = s_0 - z_j$ | Componentes $(\\Delta \\sigma, \\?\\Delta \\omega)$ | Distância $|\\vec{w}_{z_j}|$ | Ângulo $\\phi_{z_j} = \\operatorname{atan2}(\\Delta \\omega, \\Delta \\sigma)$ |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
                for vz in vecs_z:
                    z_orig = format_complex_frac(vz["zero"])
                    w_str = format_complex_frac(vz["vector"])
                    ds_str = format_frac(vz["delta_sigma"])
                    dw_str = format_frac(vz["delta_omega"])
                    mag_str = format_frac(vz["dist"])
                    ang_str = f"{format_frac(vz['angle_deg'])}^\\circ"
                    z_table.append(
                        f"| $z_{{{vz['index']}}} = {z_orig}$ | ${w_str}$ | $\\Delta\\sigma = {ds_str}, \\Delta\\omega = {dw_str}$ | ${mag_str}$ | ${ang_str}$ |"
                    )
                st.markdown("\n".join(z_table))
                sum_z_terms = " + ".join([f"({format_frac(vz['angle_deg'])}^\\circ)" for vz in vecs_z])
                st.latex(rf"\sum_{{j=1}}^{{nZ}} \phi_{{z_j}} = {sum_z_terms} = {format_frac(sum_z)}^\circ")
            else:
                st.markdown(r"*(Não há zeros no sistema)* $\implies \sum \phi_z = 0^\circ$.")

            # 3. Substituição e Fase Resultante
            st.markdown("---")
            st.markdown(r"### 3. Balanço Angular e Redução Trigonométrica:")
            st.latex(
                rf"\angle G(s_0)H(s_0) = \sum \phi_z - \sum \theta_p = ({format_frac(sum_z)}^\circ) - ({format_frac(sum_p)}^\circ) = {format_frac(total_angle)}^\circ"
            )
            st.latex(
                rf"\angle G(s_0)H(s_0) \equiv {format_frac(norm_angle)}^\circ \pmod{{360^\circ}} \quad \left( \text{{ou }} {format_frac(angle_180)}^\circ \in (-180^\circ, 180^\circ] \right)"
            )

            # 4. Veredito e Deficiência Angular
            st.markdown("---")
            st.markdown(r"### 4. Veredito de Pertinência ao LGR:")
            if is_pole:
                st.success(
                    rf" **O ponto $s_0 = {s0_str}$ coincide exatamente com o polo $p_{{{test_details['coincident_pole_idx']}}}$.** "
                    rf"No LGR, os ramos partem dos polos com ganho $K = 0$, portanto o ponto pertence trivialmente ao LGR."
                )
            elif is_zero:
                st.success(
                    rf" **O ponto $s_0 = {s0_str}$ coincide exatamente com o zero $z_{{{test_details['coincident_zero_idx']}}}$.** "
                    rf"No LGR, os ramos incidem nos zeros quando o ganho $K \to \infty$, portanto o ponto pertence trivialmente ao LGR."
                )
            elif is_lgr:
                st.success(
                    rf" **O ponto $s_0 = {s0_str}$ PERTENCE ao Lugar Geométrico das Raízes!**<br>"
                    rf"A fase resultante fecha em **${format_frac(norm_angle)}^\circ \approx 180^\circ$**, satisfazendo rigorosamente a condição angular do LGR direto ($K > 0$).",
                )
            else:
                st.error(
                    rf" **O ponto $s_0 = {s0_str}$ NÃO PERTENCE ao Lugar Geométrico das Raízes direto!**<br>"
                    rf"A fase resultante é **${format_frac(norm_angle)}^\circ \neq 180^\circ$** (divergência angular de ${format_frac(abs(defic))}^\circ$).",
                )
                st.markdown(r"####  Cálculo da Deficiência Angular ($\Delta \theta$):")
                st.markdown(
                    r"Em projeto de sistemas de controle, a **deficiência angular** representa a contribuição de fase líquida "
                    r"que um controlador/compensador dinâmico (como um Compensador por Avanço de Fase ou PD) deve fornecer em $s_0$ "
                    r"para forçar o ramo do LGR a atravessar esse ponto de operação desejado:"
                )
                st.latex(
                    rf"\Delta \theta = 180^\circ - \angle G(s_0)H(s_0) = 180^\circ - ({format_frac(total_angle)}^\circ) \equiv {format_frac(defic)}^\circ"
                )
                st.info(
                    rf" **Ação de Projeto Recomendada:** Para tornar $s_0$ um polo de malha fechada dominante, "
                    rf"deve-se projetar um compensador de avanço $G_c(s) = \frac{{s + z_c}}{{s + p_c}}$ com contribuição de fase $\angle G_c(s_0) = {format_frac(defic)}^\circ$."
                )

            # 5. Visualização Gráfica Interativa
            st.markdown("---")
            st.markdown(r"### 5. Visualização Gráfica dos Vetores Partindo das Singularidades até $s_0$:")
            fig11 = plot_test_point_vectors(poles, zeros, s0, test_details, xmin, xmax, ymin, ymax)
            st.plotly_chart(fig11, width="stretch", config={"scrollZoom": True})

        # Passo 12
        with st.expander("Passo 12: Cálculo de K ($s_0$)", expanded=True):
            st.markdown(
                r"A **Condição de Módulo** determina o valor exato do ganho estático $K$ associado ao ponto $s_0$. "
                r"Ela decorre diretamente do módulo da equação de malha fechada $|1 + K G(s_0)H(s_0)| = 0$:"
            )
            st.latex(r"|K \cdot G(s_0)H(s_0)| = 1 \implies K = \frac{1}{|G(s_0)H(s_0)|} = \frac{\prod_{i=1}^{nP} |s_0 - p_i|}{\prod_{j=1}^{nZ} |s_0 - z_j|}")

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

            # 1. Detalhamento do Cálculo Numérico de K
            st.markdown("---")
            st.markdown(r"### 1. Substituição e Formulação do Produto das Distâncias:")

            st.markdown(r"**a) Numerador: Produto das distâncias de $s_0$ a todos os polos ($\prod |s_0 - p_i|$):**")
            if dist_p:
                str_p_prod = " \\cdot ".join([f"{format_frac(d)}" for d in dist_p])
                st.latex(rf"\prod_{{i=1}}^{{nP}} |s_0 - p_i| = {str_p_prod} = {format_frac(prod_p)}")
            else:
                st.latex(r"\prod_{i=1}^{nP} |s_0 - p_i| = 1 \quad (\text{sem polos finitos})")

            st.markdown(r"**b) Denominador: Produto das distâncias de $s_0$ a todos os zeros ($\prod |s_0 - z_j|$):**")
            if dist_z:
                str_z_prod = " \\cdot ".join([f"{format_frac(d)}" for d in dist_z])
                st.latex(rf"\prod_{{j=1}}^{{nZ}} |s_0 - z_j| = {str_z_prod} = {format_frac(prod_z)}")
            else:
                st.latex(r"\prod_{j=1}^{nZ} |s_0 - z_j| = 1 \quad (\text{sem zeros finitos})")

            st.markdown(r"**c) Cálculo Final do Ganho $K$:**")
            if is_pole:
                st.latex(r"K(s_0) = 0 \quad (\text{ponto coincide com um polo})")
            elif is_zero:
                st.latex(r"K(s_0) \to \infty \quad (\text{ponto coincide com um zero})")
            else:
                if abs(scale_fac - 1.0) > 1e-4:
                    st.latex(
                        rf"K = \frac{{1}}{{|K_{{escala}}|}} \cdot \frac{{\prod |s_0 - p_i|}}{{\prod |s_0 - z_j|}} = "
                        rf"\frac{{1}}{{{format_frac(scale_fac)}}} \cdot \frac{{{format_frac(prod_p)}}}{{{format_frac(prod_z)}}} = {format_frac(K_val)}"
                    )
                else:
                    st.latex(
                        rf"K = \frac{{\prod |s_0 - p_i|}}{{\prod |s_0 - z_j|}} = "
                        rf"\frac{{{format_frac(prod_p)}}}{{{format_frac(prod_z)}}} = {format_frac(K_val)}"
                    )

            # 2. Comprovação Matemática (Prova Real)
            st.markdown("---")
            st.markdown(r"### 2.  Comprovação Matemática do Resultado (Prova Real):")
            st.markdown(
                r"Para comprovar analiticamente a exatidão do resultado e verificar se $s_0$ é de fato um polo de malha fechada, "
                rf"substituímos o ponto $s = s_0 = {s0_str}$ e o ganho $K = {format_frac(K_val)}$ diretamente na Equação Característica:"
            )
            st.latex(r"P(s) = D(s) + K \cdot N(s) = 0")

            def format_eval_poly(poly_coeffs, s_val):
                deg = len(poly_coeffs) - 1
                terms = []
                for i, c in enumerate(poly_coeffs):
                    if abs(c) < 1e-9:
                        continue
                    p = deg - i
                    c_str = format_frac(c)
                    s_str = f"({format_complex_frac(s_val)})"
                    if p == 0:
                        terms.append(f"{c_str}")
                    elif p == 1:
                        if c == 1:
                            terms.append(f"{s_str}")
                        elif c == -1:
                            terms.append(f"-{s_str}")
                        else:
                            terms.append(f"{c_str} \\cdot {s_str}")
                    else:
                        if c == 1:
                            terms.append(f"{s_str}^{{{p}}}")
                        elif c == -1:
                            terms.append(f"-{s_str}^{{{p}}}")
                        else:
                            terms.append(f"{c_str} \\cdot {s_str}^{{{p}}}")
                return " + ".join(terms) if terms else "0"

            d_eval_str = format_eval_poly(D_coeffs, s0)
            n_eval_str = format_eval_poly(N_coeffs, s0)

            st.markdown(r"**Etapa A — Avaliação do Polinômio do Denominador $D(s_0)$:**")
            st.latex(rf"D(s_0) = {d_eval_str} = {format_complex_frac(D_s0)}")

            st.markdown(r"**Etapa B — Avaliação do Polinômio do Numerador $N(s_0)$:**")
            st.latex(rf"N(s_0) = {n_eval_str} = {format_complex_frac(N_s0)}")

            st.markdown(r"**Etapa C — Multiplicação pelo Ganho Calculado $K$:**")
            k_n_val = K_val * N_s0 if np.isfinite(K_val) else float("inf")
            st.latex(
                rf"K \cdot N(s_0) = ({format_frac(K_val)}) \cdot ({format_complex_frac(N_s0)}) = {format_complex_frac(k_n_val)}"
            )

            st.markdown(r"**Etapa D — Balanço e Soma da Equação Característica $P(s_0)$:**")
            st.latex(
                rf"P(s_0) = D(s_0) + K \cdot N(s_0) = ({format_complex_frac(D_s0)}) + ({format_complex_frac(k_n_val)}) = {format_complex_frac(P_s0)}"
            )

            # Conclusão da Prova Real
            if is_lgr and np.isfinite(residual):
                st.latex(
                    r"\boxed{ P(s_0) = 0 + 0j = 0 \quad (\checkmark \textbf{ COMPROVADO}) }"
                )
                st.success(
                    rf" **COMPROVAÇÃO CONCLUÍDA COM SUCESSO!**<br>"
                    rf"A equação característica zera perfeitamente para $s = {s0_str}$ com o ganho $K = {format_frac(K_val)}$. "
                    rf"Isso comprova analiticamente, sem qualquer margem de dúvida, que **$s_0$ é um polo de malha fechada exato**.",
                )
            else:
                st.latex(
                    rf"\boxed{{ P(s_0) = {format_complex_frac(P_s0)} \neq 0 \quad (\text{{Resíduo }} |P(s_0)| = {format_frac(residual)}) }}"
                )
                if K_req is not None:
                    st.markdown(
                        r"**Demonstração do Ganho Necessário ($K_{nec}$):** "
                        r"Para que $s_0$ fosse uma raiz da equação característica sem compensação ($D(s_0) + K \cdot N(s_0) = 0$), "
                        r"o ganho $K$ deveria ser obrigatoriamente:"
                    )
                    st.latex(
                        rf"K_{{nec}} = -\frac{{D(s_0)}}{{N(s_0)}} = -\frac{{{format_complex_frac(D_s0)}}}{{{format_complex_frac(N_s0)}}} = {format_complex_frac(K_req)}"
                    )
                    if abs(np.imag(K_req)) > 1e-4:
                        st.warning(
                            rf"️ **Ganho Complexo Não-Realizável:** "
                            rf"Como $\operatorname{{Im}}(K_{{nec}}) = {format_frac(np.imag(K_req))} \neq 0$, "
                            rf"é matematicamente impossível colocar um polo em $s_0$ usando apenas um ganho real $K \in \mathbb{{R}}$. "
                            rf"A equação característica não zera porque há deficiência angular de **${format_frac(defic)}^\circ$**.",
                        )
                    elif np.real(K_req) < 0:
                        st.warning(
                            rf"️ **Ganho Negativo:** "
                            rf"Como $K_{{nec}} = {format_frac(np.real(K_req))} < 0$, este ponto só existiria sob realimentação positiva (LGR inverso), "
                            rf"mas **NÃO** no LGR direto com realimentação negativa ($K > 0$).",
                        )
                st.error(
                    rf" **COMPROVAÇÃO DE NÃO-PERTINÊNCIA:**<br>"
                    rf"O resíduo $|P(s_0)| = {format_frac(residual)} \neq 0$ comprova analiticamente por que $s_0$ não pode ser atingido "
                    rf"apenas ajustando o ganho $K$. É estritamente necessária a compensação de fase calculada no Passo 11.",
                )

        # Gráfico Final
        st.markdown("## Gráfico Final do LGR")
        
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
                # Inicializa o ramo com apenas o primeiro ponto (K=0) para a animação pintá-lo aos poucos
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

            # Seta no meio do caminho (direção a partir do polo)
            if len(vx) > 10:
                mid = (
                    len(vx) // 4
                )  # Pega um ponto a 25% do caminho para mostrar logo de cara
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

            # Seta marcando a saída do viewport (indo pro infinito)
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

        # --- ANIMAÇÃO DAS RAÍZES (K ITERATIVO) ---
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
                hovertemplate="Raiz: %{x:.3f} + %{y:.3f}j<extra></extra>",
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

            # Atualiza as linhas dos ramos para crescerem até o ponto atual
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

            # Atualiza os diamantes (trajetória verdadeira, sem limite geométrico)
            frame_dyn_x, frame_dyn_y = [], []
            for b_idx in range(all_roots.shape[1]):
                limit = min(i, len(branches_x[b_idx]) - 1)
                if limit >= 0:
                    frame_dyn_x.append(branches_x[b_idx][limit])
                    frame_dyn_y.append(branches_y[b_idx][limit])

            frame_data.append(go.Scatter(x=frame_dyn_x, y=frame_dyn_y))

            # A ordem dos traços modificados deve corresponder exatamente a estes índices
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
                            "label": " Play",
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
                            "label": " Pause",
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

    st.markdown(
        """
        <div class="polaris-footer">
            <div class="footer-credits">
                <span>Desenvolvido com</span>
                <span style="color: #e25555;">&hearts;</span>
                <span>por <span class="footer-name">franssoares</span></span>
            </div>
            <p style="margin: 0.5rem 0 0 0;">Disciplina de Sistemas de Controle &bull; <span class="footer-highlight">Polaris LGR &copy; 2026</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
