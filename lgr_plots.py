"""
LGR Plotting Module
Contains configuration and helpers for rendering interactive Plotly charts.
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any
from lgr_math import format_frac, format_complex_frac


def configure_layout(
    fig: go.Figure, title: str, xmin: float, xmax: float, ymin: float, ymax: float
) -> None:
    """Applies standard aesthetic configurations and axis boundaries to a Plotly figure."""
    fig.update_layout(
        title=title,
        xaxis_title=r"Eixo Real (σ)",
        yaxis_title=r"Eixo Imaginário (jω)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.3)",
            zerolinewidth=1,
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.1)",
            gridwidth=1,
            griddash="dot",
            range=[xmin, xmax],
        ),
        yaxis=dict(
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.3)",
            zerolinewidth=1,
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.1)",
            gridwidth=1,
            griddash="dot",
            range=[ymin, ymax],
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(14, 17, 23, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=50, b=20),
        height=500,
        hovermode="closest",
        dragmode="pan",
    )


def add_poles_zeros_traces(
    fig: go.Figure, poles: np.ndarray, zeros: np.ndarray, show_labels: bool = False
) -> None:
    """Adds markers for finite poles (red X) and zeros (green O)."""
    if len(poles) > 0:
        sorted_poles = sorted(poles, key=lambda x: (np.real(x), np.imag(x)))
        text_poles = (
            [f"p<sub>{i+1}</sub>" for i in range(len(sorted_poles))]
            if show_labels
            else None
        )
        mode = "markers+text" if show_labels else "markers"
        fig.add_trace(
            go.Scatter(
                x=[np.real(p) for p in sorted_poles],
                y=[np.imag(p) for p in sorted_poles],
                mode=mode,
                text=text_poles,
                textposition="top center",
                textfont=dict(size=14, color="white"),
                marker=dict(
                    symbol="x", size=12, color="red", line=dict(width=4, color="red")
                ),
                name="Pólos",
                hovertemplate="Pólo: %{x:.3f} + %{y:.3f}j<extra></extra>",
            )
        )
    if len(zeros) > 0:
        sorted_zeros = sorted(zeros, key=lambda x: (np.real(x), np.imag(x)))
        text_zeros = (
            [f"z<sub>{i+1}</sub>" for i in range(len(sorted_zeros))]
            if show_labels
            else None
        )
        mode = "markers+text" if show_labels else "markers"
        fig.add_trace(
            go.Scatter(
                x=[np.real(z) for z in sorted_zeros],
                y=[np.imag(z) for z in sorted_zeros],
                mode=mode,
                text=text_zeros,
                textposition="top center",
                textfont=dict(size=14, color="white"),
                marker=dict(
                    symbol="circle-open",
                    size=12,
                    color="green",
                    line=dict(width=4, color="green"),
                ),
                name="Zeros",
                hovertemplate="Zero: %{x:.3f} + %{y:.3f}j<extra></extra>",
            )
        )


def create_base_plot(
    poles: np.ndarray,
    zeros: np.ndarray,
    title: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    show_labels: bool = False,
    draw_poles_zeros: bool = True,
) -> go.Figure:
    """Creates a configured baseline figure with poles and zeros rendered."""
    fig = go.Figure()
    if draw_poles_zeros:
        add_poles_zeros_traces(fig, poles, zeros, show_labels=show_labels)
    configure_layout(fig, title, xmin, xmax, ymin, ymax)
    return fig


def plot_test_point_vectors(
    poles: np.ndarray,
    zeros: np.ndarray,
    s0: complex,
    test_details: Dict[str, Any],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> go.Figure:
    """Renders an interactive Plotly diagram with vectors from poles and zeros to test point s0."""
    fig = create_base_plot(
        poles,
        zeros,
        "Vetores e Critério de Ângulo para o Ponto de Teste s₀",
        xmin,
        xmax,
        ymin,
        ymax,
        show_labels=True,
    )

    # 1. Vetores dos polos até s0
    for vp in test_details.get("vecs_p", []):
        p = vp["pole"]
        if abs(s0 - p) > 1e-4:
            fig.add_trace(
                go.Scatter(
                    x=[p.real, s0.real],
                    y=[p.imag, s0.imag],
                    mode="lines",
                    line=dict(color="#ff6b6b", width=2, dash="dash"),
                    name=f"Vetor p<sub>{vp['index']}</sub> → s₀ (θ={format_frac(vp['angle_deg'])}°)",
                    hovertemplate=(
                        f"<b>Vetor do Polo p<sub>{vp['index']}</sub></b><br>"
                        f"Origem p: {format_complex_frac(p)}<br>"
                        f"Vetor: {format_complex_frac(vp['vector'])}<br>"
                        f"Módulo: {format_frac(vp['dist'])}<br>"
                        f"Ângulo θ: {format_frac(vp['angle_deg'])}°"
                        f"<extra></extra>"
                    ),
                )
            )
            # Seta apontando para s0
            fig.add_annotation(
                x=s0.real,
                y=s0.imag,
                ax=p.real,
                ay=p.imag,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor="#ff6b6b",
            )
            # Rótulo no ponto médio
            mid_x = (p.real + s0.real) * 0.5
            mid_y = (p.imag + s0.imag) * 0.5
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=f"<b>θ<sub>p{vp['index']}</sub>={format_frac(vp['angle_deg'])}°</b><br>|v|={format_frac(vp['dist'])}",
                showarrow=False,
                font=dict(color="#ff6b6b", size=10),
                bgcolor="rgba(14, 17, 23, 0.85)",
                bordercolor="#ff6b6b",
                borderwidth=1,
                borderpad=2,
            )

    # 2. Vetores dos zeros até s0
    for vz in test_details.get("vecs_z", []):
        z = vz["zero"]
        if abs(s0 - z) > 1e-4:
            fig.add_trace(
                go.Scatter(
                    x=[z.real, s0.real],
                    y=[z.imag, s0.imag],
                    mode="lines",
                    line=dict(color="#51cf66", width=2, dash="dash"),
                    name=f"Vetor z<sub>{vz['index']}</sub> → s₀ (ϕ={format_frac(vz['angle_deg'])}°)",
                    hovertemplate=(
                        f"<b>Vetor do Zero z<sub>{vz['index']}</sub></b><br>"
                        f"Origem z: {format_complex_frac(z)}<br>"
                        f"Vetor: {format_complex_frac(vz['vector'])}<br>"
                        f"Módulo: {format_frac(vz['dist'])}<br>"
                        f"Ângulo ϕ: {format_frac(vz['angle_deg'])}°"
                        f"<extra></extra>"
                    ),
                )
            )
            # Seta apontando para s0
            fig.add_annotation(
                x=s0.real,
                y=s0.imag,
                ax=z.real,
                ay=z.imag,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor="#51cf66",
            )
            # Rótulo no ponto médio
            mid_x = (z.real + s0.real) * 0.5
            mid_y = (z.imag + s0.imag) * 0.5
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=f"<b>ϕ<sub>z{vz['index']}</sub>={format_frac(vz['angle_deg'])}°</b><br>|w|={format_frac(vz['dist'])}",
                showarrow=False,
                font=dict(color="#51cf66", size=10),
                bgcolor="rgba(14, 17, 23, 0.85)",
                bordercolor="#51cf66",
                borderwidth=1,
                borderpad=2,
            )

    # 3. Marcador do ponto s0
    is_lgr = test_details.get("is_lgr", False)
    status_str = "PERTENCE ao LGR" if is_lgr else "NÃO Pertence ao LGR"
    k_val_str = (
        format_frac(test_details.get("K", 0.0))
        if np.isfinite(test_details.get("K", 0.0))
        else "∞"
    )

    fig.add_trace(
        go.Scatter(
            x=[s0.real],
            y=[s0.imag],
            mode="markers+text",
            text=["<b>s₀</b>"],
            textposition="top right",
            textfont=dict(size=14, color="#00f5d4"),
            marker=dict(
                symbol="diamond",
                size=14,
                color="#00f5d4",
                line=dict(width=2, color="white"),
            ),
            name="Ponto de Teste s₀",
            hovertemplate=(
                f"<b>Ponto de Teste s₀</b><br>"
                f"s₀ = {format_complex_frac(s0)}<br>"
                f"Status: {status_str}<br>"
                f"Fase Resultante: {format_frac(test_details.get('normalized_angle', 0.0))}°<br>"
                f"Ganho K: {k_val_str}"
                f"<extra></extra>"
            ),
        )
    )

    # 4. Banner de resumo no gráfico
    if is_lgr:
        banner_text = (
            f"<b>Condição de Ângulo: SATISFEITA (≈ 180°)</b><br>"
            f"s₀ = {format_complex_frac(s0)} Pertence ao LGR | Ganho K = {k_val_str}"
        )
        banner_color = "#2ec4b6"
    else:
        defic = test_details.get("angular_deficiency_signed", 0.0)
        banner_text = (
            f"<b>Condição de Ângulo: NÃO SATISFEITA (≠ 180°)</b><br>"
            f"Fase = {format_frac(test_details.get('normalized_angle', 0.0))}° | Deficiência Angular: Δθ = {format_frac(defic)}°"
        )
        banner_color = "#ff6b6b"

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        text=banner_text,
        showarrow=False,
        font=dict(color=banner_color, size=12),
        bgcolor="rgba(14, 17, 23, 0.9)",
        bordercolor=banner_color,
        borderwidth=1.5,
        borderpad=6,
    )

    fig.update_layout(height=520)
    return fig


def generate_fig7_asymptotes(poles, zeros, sigma_A, angles_A, xmin, xmax, ymin, ymax, length_max):
    fig7 = create_base_plot(poles, zeros, "Assíntotas", xmin, xmax, ymin, ymax, draw_poles_zeros=False)
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
            text=f"θ_{q} = {format_frac(angle)}°",
            showarrow=False,
            font=dict(color="orange", size=12),
            bgcolor="rgba(14, 17, 23, 0.85)",
            bordercolor="orange",
            borderwidth=1,
            borderpad=3,
        )
        theta_start = 0 if rad >= 0 else rad
        theta_end = rad if rad >= 0 else 0
        theta_arc = np.linspace(theta_start, theta_end, 30)
        arc_x = np.real(sigma_A) + arc_R * np.cos(theta_arc)
        arc_y = arc_R * np.sin(theta_arc)
        fig7.add_trace(
            go.Scatter(
                x=arc_x,
                y=arc_y,
                mode="lines",
                line=dict(color="rgba(255, 165, 0, 0.6)", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if sigma_A is not None:
        fig7.add_trace(
            go.Scatter(
                x=[np.real(sigma_A)],
                y=[0],
                mode="markers+text",
                marker=dict(symbol="star", size=14, color="orange", line=dict(width=2, color="white")),
                text=["σ_A"],
                textposition="top center",
                textfont=dict(size=14, color="white"),
                name=f"Centróide (σ_A = {format_frac(np.real(sigma_A))})",
                hovertemplate=f"<b>Centróide</b><br>σ_A = {format_frac(np.real(sigma_A))}<extra></extra>",
            )
        )
    add_poles_zeros_traces(fig7, poles, zeros, show_labels=False)
    fig7.update_layout(height=520)
    return fig7

def generate_fig8_breakaway(poles, zeros, segment_coords, candidates, xmin, xmax, ymin, ymax):
    fig8 = create_base_plot(poles, zeros, "Pontos de Saída e Entrada no Plano s", xmin, xmax, ymin, ymax, draw_poles_zeros=False)
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
                    marker=dict(symbol="diamond", size=14, color=color_m, line=dict(width=2, color="white")),
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
    return fig8

def generate_fig9_crossings(poles, zeros, crossings_data, xmin, xmax, ymin, ymax):
    fig9 = create_base_plot(poles, zeros, "Cruzamento com o Eixo Imaginário (jω)", xmin, xmax, ymin, ymax, draw_poles_zeros=False)
    fig9.add_vline(x=0, line_dash="solid", line_color="rgba(0, 180, 216, 0.4)", line_width=2)
    if crossings_data:
        for data in crossings_data:
            k_crit = data["k_crit"]
            omegas = data["omegas"]
            for w in omegas:
                fig9.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[w],
                        mode="markers",
                        marker=dict(symbol="star", size=16, color="#ff007f", line=dict(width=2, color="white")),
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
                fig9.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[-w],
                        mode="markers",
                        marker=dict(symbol="star", size=16, color="#ff007f", line=dict(width=2, color="white")),
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
    return fig9

def generate_fig10_angles(poles, zeros, has_complex, pole_details, zero_details, xmin, xmax, ymin, ymax):
    fig10 = create_base_plot(poles, zeros, "Ângulos de Partida e Chegada no Plano s", xmin, xmax, ymin, ymax, draw_poles_zeros=False)
    if has_complex:
        arrow_len = max(0.8, (xmax - xmin) * 0.12)
        for pd in pole_details:
            cp = pd["pole"]
            for b in pd["branches"]:
                ang_deg = b["norm"]
                rad = np.radians(ang_deg)
                dx = arrow_len * np.cos(rad)
                dy = arrow_len * np.sin(rad)
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
    return fig10
