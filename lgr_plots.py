"""
LGR Plotting Module
Contains configuration and helpers for rendering interactive Plotly charts.
"""

import plotly.graph_objects as go
import numpy as np
from typing import List


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
