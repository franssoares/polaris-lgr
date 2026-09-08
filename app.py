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
    build_routh_hurwitz,
    simulate_root_locus,
)
from lgr_plots import create_base_plot, add_poles_zeros_traces

st.set_page_config(page_title="LGR - 12 Passos", layout="wide")


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
        valid_breakaway = find_breakaway_points(N_coeffs, D_coeffs)
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
                        term_parts.append(f"{d_val:g}")
                    if n_val != 0:
                        sign = (
                            "+"
                            if n_val > 0 and d_val != 0
                            else ("" if n_val > 0 else "-")
                        )
                        abs_n = abs(n_val)
                        n_str = f"{abs_n:g}" if abs_n != 1 else ""
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
            K_str = f"{K_scale:g}" if K_scale != 1.0 else ""
            st.latex(
                r"P(s) = " + K_str + r"\frac{" + num_fact + r"}{" + den_fact + r"}"
            )

            def format_root(r):
                r_rounded = np.round(r, 4)
                if abs(np.imag(r_rounded)) < 1e-5:
                    return f"{np.real(r_rounded):g}"
                else:
                    sign = "+" if np.imag(r_rounded) > 0 else "-"
                    return f"{np.real(r_rounded):g} {sign} {abs(np.imag(r_rounded)):g}j"

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
                    segments.append(f"[{end:g}, {start:g}]")
                    segment_coords.append((start, end))
                else:
                    segments.append(f"(-∞, {start:g}]")
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
                                "label": "▶ Play",
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
                                "label": "⏸ Pause",
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
            ls = max(nP, nZ)
            st.latex(f"LS = \\max({nP}, {nZ}) = {ls}")

        # Passo 6
        with st.expander("Passo 6: Simetria", expanded=True):
            st.markdown("O LGR é simétrico em relação ao eixo real.")

        # Passo 7
        with st.expander("Passo 7: Assíntotas", expanded=True):
            if nP == nZ:
                st.markdown("Não há assíntotas.")
            else:
                st.markdown(r"**Centro das assíntotas ($\sigma_A$):**")
                sum_p_str = " + ".join(
                    [
                        f"({np.real(p):g}{'+' + str(np.imag(p)) + 'j' if np.imag(p)>0 else (str(np.imag(p)) + 'j' if np.imag(p)<0 else '')})"
                        for p in poles
                    ]
                )
                sum_z_str = " + ".join(
                    [
                        f"({np.real(z):g}{'+' + str(np.imag(z)) + 'j' if np.imag(z)>0 else (str(np.imag(z)) + 'j' if np.imag(z)<0 else '')})"
                        for z in zeros
                    ]
                )
                if not sum_p_str:
                    sum_p_str = "0"
                if not sum_z_str:
                    sum_z_str = "0"

                st.latex(r"\sigma_A = \frac{\sum p_i - \sum z_i}{n_P - n_Z}")
                st.latex(
                    rf"\sigma_A = \frac{{[{sum_p_str}] - [{sum_z_str}]}}{{{nP} - {nZ}}} = {np.real(sigma_A):.3g}"
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
                        rf"\theta_{k} = \frac{{(2({k}) + 1) \cdot 180^\circ}}{{{abs(nP - nZ)}}} = {angle:.1f}^\circ"
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
                                rf"\theta_{k} = {angle:.1f}^\circ \implies \text{{Assíntota sobre o eixo imaginário}}"
                            )
                        else:
                            st.latex(
                                rf"\theta_{k} = {angle:.1f}^\circ \implies \text{{Assíntota paralela ao eixo imaginário (não cruza)}}"
                            )
                    else:
                        cos_val = np.cos(np.radians(angle))
                        t_cross = -np.real(sigma_A) / cos_val if cos_val != 0 else -1
                        if t_cross >= 0:
                            cross_y = -np.real(sigma_A) * np.tan(np.radians(angle))
                            if abs(cross_y) < 1e-5:
                                st.latex(
                                    rf"\theta_{k} = {angle:.1f}^\circ \implies \text{{Assíntota sobre o eixo real (não destacaremos a origem)}}"
                                )
                            else:
                                st.latex(
                                    rf"\theta_{k} = {angle:.1f}^\circ \implies y_{{cruzamento}} = -({np.real(sigma_A):.3g}) \cdot \tan({angle:.1f}^\circ) = {cross_y:.3g}j"
                                )
                                crossings.append((angle, cross_y))
                        else:
                            st.latex(
                                rf"\theta_{k} = {angle:.1f}^\circ \implies \text{{A semirreta se afasta do eixo imaginário (não cruza)}}"
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

                    t_draw = t_max * 0.85

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

                    base_R = max(xmax - xmin, ymax - ymin) * 0.05
                    arc_R = base_R + q * (max(xmax - xmin, ymax - ymin) * 0.08)

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
                            text=f"{angle:.1f}°",
                            showarrow=False,
                            font=dict(color="orange", size=13),
                        )
                    else:
                        fig7.add_annotation(
                            x=head_x + dx * 0.1,
                            y=head_y + dy * 0.1,
                            text=f"{angle:.1f}°",
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
                                text=[f"{cross_y:.3g}j"],
                                textposition="middle right",
                                textfont=dict(color="magenta", size=13),
                                name=f"Cruzamento {angle:.1f}°",
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
                        text=[f"σ<sub>A</sub> = {np.real(sigma_A):.3g}"],
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
            if not valid_breakaway:
                st.markdown("Nenhum ponto válido para $K > 0$.")
            else:
                for s_val, k_val in valid_breakaway:
                    s_format = (
                        f"{s_val:.3f}"
                        if np.isreal(s_val)
                        else f"{np.real(s_val):.3f} + {np.imag(s_val):.3f}j"
                    )
                    st.markdown(f"- Ponto $s = {s_format}$ com ganho $K = {k_val:.3g}$")

        # Passo 9
        with st.expander("Passo 9: Cruzamento com o eixo imaginário", expanded=True):
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
                        f"${sp.latex(sp.cancel(val))}$" if val != 0 else "$0$"
                        for val in routh_table[i]
                    ]
                )
                row_str += " |\n"
                md_table += row_str

            st.markdown(md_table)

            crossings_data = routh_result.get("crossings_data", [])
            if crossings_data:
                st.markdown(r"**Como o cruzamento foi calculado:**")
                st.markdown(
                    r"O sistema cruza o eixo imaginário quando ocorre uma oscilação marginalmente estável. Isso se traduz na Tabela de Routh quando uma linha inteira se anula, indicando raízes puramente imaginárias geradas pela linha imediatamente superior (Equação Auxiliar)."
                )

                for data in crossings_data:
                    power = data["s_power"]
                    row_expr = data["row_expr"]
                    k_crit = data["k_crit"]
                    aux_eq_sym = data["aux_eq_sym"]
                    aux_eq_sub = data["aux_eq_sub"]
                    omegas = data["omegas"]

                    st.markdown(f"**1. Isolando $K$ na linha $s^{power}$:**")
                    st.markdown(
                        f"Igualamos o primeiro elemento da linha $s^{power}$ a zero:"
                    )
                    st.latex(
                        f"{sp.latex(sp.cancel(row_expr))} = 0 \\implies K_{{crítico}} = {k_crit:.3g}"
                    )

                    st.markdown(f"**2. Montando a Equação Auxiliar ($A(s)$):**")
                    st.markdown(
                        f"Pegamos os coeficientes da linha logo acima ($s^{power+1}$) e pulamos as potências de 2 em 2:"
                    )
                    st.latex(f"A(s) = {sp.latex(sp.cancel(aux_eq_sym))} = 0")

                    st.markdown(f"**3. Substituindo $K$ e extraindo as raízes:**")
                    st.latex(f"A(s) = {sp.latex(sp.cancel(aux_eq_sub))} = 0")

                    for w in omegas:
                        st.latex(
                            rf"s = \pm {w:.3g}j \implies \omega = {w:.3g} \text{{ rad/s}}"
                        )
            else:
                st.markdown(
                    "Nenhum cruzamento com o eixo imaginário foi identificado para $K > 0$."
                )

        # Passo 10
        with st.expander("Passo 10: Ângulos de partida/chegada", expanded=True):
            complex_poles = []
            for p in poles:
                if abs(np.imag(p)) > 1e-5 and not any(
                    np.isclose(p, cp) for cp in complex_poles
                ):
                    complex_poles.append(p)

            complex_zeros = []
            for z in zeros:
                if abs(np.imag(z)) > 1e-5 and not any(
                    np.isclose(z, cz) for cz in complex_zeros
                ):
                    complex_zeros.append(z)

            if not complex_poles and not complex_zeros:
                st.markdown("Não aplicável (sem polos ou zeros complexos).")
            else:
                for cp in complex_poles:
                    m = sum(1 for p in poles if np.isclose(cp, p))
                    p_angles = [
                        np.degrees(np.angle(cp - p))
                        for p in poles
                        if not np.isclose(cp, p)
                    ]
                    z_angles = [np.degrees(np.angle(cp - z)) for z in zeros]
                    sum_p = sum(p_angles)
                    sum_z = sum(z_angles)

                    st.markdown(
                        f"**Ângulo de Partida do polo $p = {np.real(cp):.3g} {'+' if np.imag(cp)>0 else '-'} {abs(np.imag(cp)):.3g}j$:**"
                    )
                    st.latex(
                        r"\theta_p = \frac{180^\circ(2q+1) + \sum \phi_z - \sum \theta_{outros\_polos}}{m}"
                    )

                    if p_angles:
                        st.latex(
                            rf"\sum \theta_{{outros\_polos}} = "
                            + " + ".join([f"{a:.1f}^\\circ" for a in p_angles])
                            + f" = {sum_p:.1f}^\\circ"
                        )
                    if z_angles:
                        st.latex(
                            rf"\sum \phi_z = "
                            + " + ".join([f"{a:.1f}^\\circ" for a in z_angles])
                            + f" = {sum_z:.1f}^\\circ"
                        )

                    for q in range(m):
                        angle_dep = ((2 * q + 1) * 180 - sum_p + sum_z) / m
                        angle_norm = (angle_dep + 180) % 360 - 180
                        st.latex(
                            rf"q = {q} \implies \theta_p = \frac{{180^\circ({2*q+1}) + {sum_z:.1f}^\circ - ({sum_p:.1f}^\circ)}}{{{m}}} = {angle_norm:.1f}^\circ"
                        )

                for cz in complex_zeros:
                    m = sum(1 for z in zeros if np.isclose(cz, z))
                    z_angles = [
                        np.degrees(np.angle(cz - z))
                        for z in zeros
                        if not np.isclose(cz, z)
                    ]
                    p_angles = [np.degrees(np.angle(cz - p)) for p in poles]
                    sum_z = sum(z_angles)
                    sum_p = sum(p_angles)

                    st.markdown(
                        f"**Ângulo de Chegada no zero $z = {np.real(cz):.3g} {'+' if np.imag(cz)>0 else '-'} {abs(np.imag(cz)):.3g}j$:**"
                    )
                    st.latex(
                        r"\theta_z = \frac{180^\circ(2q+1) + \sum \theta_p - \sum \phi_{outros\_zeros}}{m}"
                    )

                    if p_angles:
                        st.latex(
                            rf"\sum \theta_p = "
                            + " + ".join([f"{a:.1f}^\\circ" for a in p_angles])
                            + f" = {sum_p:.1f}^\\circ"
                        )
                    if z_angles:
                        st.latex(
                            rf"\sum \phi_{{outros\_zeros}} = "
                            + " + ".join([f"{a:.1f}^\\circ" for a in z_angles])
                            + f" = {sum_z:.1f}^\\circ"
                        )

                    for q in range(m):
                        angle_arr = ((2 * q + 1) * 180 - sum_z + sum_p) / m
                        angle_norm = (angle_arr + 180) % 360 - 180
                        st.latex(
                            rf"q = {q} \implies \theta_z = \frac{{180^\circ({2*q+1}) + {sum_p:.1f}^\circ - ({sum_z:.1f}^\circ)}}{{{m}}} = {angle_norm:.1f}^\circ"
                        )

        # Passo 11
        with st.expander("Passo 11: Critério de ângulo ($s_0$)", expanded=True):
            angles_z = [np.degrees(np.angle(s0 - z)) for z in zeros]
            angles_p = [np.degrees(np.angle(s0 - p)) for p in poles]
            total_angle = sum(angles_z) - sum(angles_p)
            normalized_angle = total_angle % 360
            if normalized_angle < 0:
                normalized_angle += 360
            is_lgr = np.isclose(normalized_angle, 180, atol=5.0)

            st.markdown("**Condição de pertinência ao LGR:**")
            st.latex(
                r"\angle G(s_0)H(s_0) = \sum \angle(s_0 - z_j) - \sum \angle(s_0 - p_i) = \pm 180^\circ(2q+1)"
            )

            st.markdown(
                f"Para o ponto de teste $s_0 = {np.real(s0):.3g} {'+' if np.imag(s0)>0 else '-'} {abs(np.imag(s0)):.3g}j$, calculamos os vetores:"
            )

            def fmt_c(c):
                return f"{np.real(c):.3g} {'+' if np.imag(c)>=0 else '-'} {abs(np.imag(c)):.3g}j"

            if len(zeros) > 0:
                st.markdown("**Ângulos a partir dos zeros:**")
                for i, z in enumerate(zeros):
                    st.latex(
                        rf"\angle(s_0 - z_{i+1}) = \angle({fmt_c(s0)} - ({fmt_c(z)})) = \angle({fmt_c(s0 - z)}) = {angles_z[i]:.1f}^\circ"
                    )

            if len(poles) > 0:
                st.markdown("**Ângulos a partir dos polos:**")
                for i, p in enumerate(poles):
                    st.latex(
                        rf"\angle(s_0 - p_{i+1}) = \angle({fmt_c(s0)} - ({fmt_c(p)})) = \angle({fmt_c(s0 - p)}) = {angles_p[i]:.1f}^\circ"
                    )

            st.markdown("**Substituindo na condição:**")

            str_z = (
                "(" + " + ".join([f"{a:.1f}^\\circ" for a in angles_z]) + ")"
                if angles_z
                else r"0^\circ"
            )
            str_p = (
                "(" + " + ".join([f"{a:.1f}^\\circ" for a in angles_p]) + ")"
                if angles_p
                else r"0^\circ"
            )

            st.latex(
                rf"\angle G(s_0)H(s_0) = {str_z} - {str_p} = {total_angle:.1f}^\circ"
            )

            if is_lgr:
                st.success(
                    f"Ponto pertence ao LGR (Ângulo {normalized_angle:.1f}° ≈ 180°)."
                )
            else:
                st.error(
                    f"Ponto NÃO pertence ao LGR (Ângulo {normalized_angle:.1f}° ≠ 180°)."
                )

        # Passo 12
        with st.expander("Passo 12: Cálculo de K ($s_0$)", expanded=True):
            dist_z = [abs(s0 - z) for z in zeros]
            dist_p = [abs(s0 - p) for p in poles]
            K_val = (np.prod(dist_p) if dist_p else 1.0) / (
                np.prod(dist_z) if dist_z else 1.0
            )

            st.markdown(
                "O ganho $K$ no ponto de teste $s_0$ é dado pelo módulo dos vetores:"
            )
            st.latex(r"K = \frac{\prod |s_0 - p_i|}{\prod |s_0 - z_i|}")

            if len(zeros) > 0:
                st.markdown("**Distâncias aos zeros:**")
                for i, z in enumerate(zeros):
                    st.latex(
                        rf"|s_0 - z_{i+1}| = |{fmt_c(s0)} - ({fmt_c(z)})| = |{fmt_c(s0 - z)}| = {dist_z[i]:.3g}"
                    )

            if len(poles) > 0:
                st.markdown("**Distâncias aos polos:**")
                for i, p in enumerate(poles):
                    st.latex(
                        rf"|s_0 - p_{i+1}| = |{fmt_c(s0)} - ({fmt_c(p)})| = |{fmt_c(s0 - p)}| = {dist_p[i]:.3g}"
                    )

            st.markdown("**Substituindo:**")

            str_p = " \\cdot ".join([f"{d:.3g}" for d in dist_p]) if dist_p else "1"
            str_z = " \\cdot ".join([f"{d:.3g}" for d in dist_z]) if dist_z else "1"

            st.latex(rf"K = \frac{{{str_p}}}{{{str_z}}} = {K_val:.3g}")

        # Gráfico Final
        st.markdown("## Gráfico Final do LGR")

        K_vec, all_roots = simulate_root_locus(D_coeffs, N_coeffs, nP, nZ)

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
                            "label": "▶ Play",
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
                            "label": "⏸ Pause",
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
