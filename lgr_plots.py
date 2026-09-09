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
