from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PRIMARY_COLOR = "#2563eb"  # blue-600
SUCCESS_COLOR = "#16a34a"  # green-600
WARNING_COLOR = "#ca8a04"  # yellow-600
DANGER_COLOR = "#dc2626"   # red-600
MUTED_COLOR = "#64748b"    # slate-500


def customize_chart_layout(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, Segoe UI, Arial", size=12),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    )
    return fig


def generate_total_forecast_chart(df: pd.DataFrame, show_ci: bool = True) -> go.Figure:
    # Expect columns: date, mean, [p10, p90]
    # Try to infer column names
    dcol = next((c for c in df.columns if c.lower().startswith("date")), "date")
    y_mean = next((c for c in df.columns if c.lower() in ("mean", "forecast", "yhat", "total")), None)
    if y_mean is None:
        # fallback: first numeric column
        num_cols = df.select_dtypes(include="number").columns
        y_mean = num_cols[0] if len(num_cols) else None
    # Prepare dates and labels
    dates = pd.to_datetime(df[dcol])
    month_labels = dates.dt.strftime("%b-%Y")  # e.g., Jan-2025

    fig = go.Figure()
    if y_mean is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=df[y_mean],
            mode="lines+markers",
            name="Mean Forecast",
            line=dict(color=PRIMARY_COLOR, width=2),
            hovertemplate="Bulan: %{customdata[0]}<br>Forecast: %{y:,.0f}<extra></extra>",
            customdata=month_labels.to_numpy().reshape(-1, 1),
        ))
    if show_ci:
        p10 = next((c for c in df.columns if c.lower() in ("p10", "lower", "lo")), None)
        p90 = next((c for c in df.columns if c.lower() in ("p90", "upper", "hi")), None)
        if p10 and p90:
            fig.add_trace(go.Scatter(
                x=dates,
                y=df[p90],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=df[p10],
                fill="tonexty",
                line=dict(width=0),
                name="Confidence",
                fillcolor="rgba(37,99,235,0.15)",
                hoverinfo="skip"
            ))
    fig = customize_chart_layout(fig, title="Total Sales Forecast (24 Months)")
    # Axis formatting: monthly ticks and month-year labels
    fig.update_xaxes(
        tickmode="array",
        tickvals=dates,
        ticktext=month_labels,
        tickangle=-45,
        title_text="Bulan",
        dtick="M1",
    )
    fig.update_yaxes(title_text="Forecast")
    return fig


def generate_top_products_chart(df: pd.DataFrame, month: Optional[str] = None, grouped: bool = False):
    # Expect columns: date, rank, product, category, forecast
    date_col = next((c for c in df.columns if c.lower().startswith("date")), "date")
    rank_col = next((c for c in df.columns if c.lower() == "rank"), "rank")
    prod_col = next((c for c in df.columns if "product" in c.lower()), "product")
    val_col = next((c for c in df.columns if c.lower() in ("forecast", "mean", "p50")), None)
    if not val_col:
        num_cols = df.select_dtypes(include="number").columns
        val_col = num_cols[-1] if len(num_cols) else None

    # Parse and format dates
    dser = pd.to_datetime(df[date_col])
    df = df.copy()
    df[date_col] = dser
    df["month_label_space"] = df[date_col].dt.strftime("%b %Y")  # e.g., Jan 2025

    # Always keep only top-5 per month for visualization
    df = df[df[rank_col] <= 5]

    if month and not grouped:
        # Allow selecting by either raw string or formatted month
        try:
            m_dt = pd.to_datetime(month)
        except Exception:
            # try matching formatted label
            mask = df["month_label_space"].astype(str) == str(month)
            ddf = df[mask]
        else:
            ddf = df[df[date_col] == m_dt]
        ddf = ddf.sort_values(rank_col)
        # Horizontal bar for a single month (still 5 bars)
        fig = px.bar(
            ddf,
            x=val_col,
            y=prod_col,
            orientation="h",
            color=prod_col,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data={prod_col: False},  # we'll use hovertemplate
        )
        fig.update_traces(
            hovertemplate=(
                "Produk: %{y}<br>"
                + "Bulan: %{customdata[0]}<br>"
                + "Forecast: %{x:,.0f}<br>"
                + "Ranking: %{customdata[1]}<extra></extra>"
            ),
            customdata=ddf[["month_label_space", rank_col]].to_numpy(),
            texttemplate="%{x:,.0f}",
            textposition="outside",
        )
        fig = customize_chart_layout(fig, title=f"Top Produk - {ddf['month_label_space'].iloc[0] if len(ddf) else month}")
        fig.update_xaxes(title_text="Forecast")
        fig.update_yaxes(title_text="Produk")
        return fig

    # Grouped across months: five bars per month
    # Sort months and preserve chronological order on x-axis
    df = df.sort_values([date_col, rank_col])
    fig = px.bar(
        df,
        x="month_label_space",
        y=val_col,
        color=prod_col,
        barmode="group",
        category_orders={"month_label_space": sorted(df["month_label_space"].unique(), key=lambda s: pd.to_datetime("01-"+s, format="%d-%b %Y"))},
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={prod_col: False},
    )
    fig.update_traces(
        hovertemplate=(
            "Produk: %{customdata[0]}<br>"
            + "Bulan: %{x}<br>"
            + "Forecast: %{y:,.0f}<br>"
            + "Ranking: %{customdata[1]}<extra></extra>"
        )
    )
    fig.update_traces(customdata=df[[prod_col, rank_col]].to_numpy())
    fig = customize_chart_layout(fig, title="Top Produk per Bulan (Top 5)")
    fig.update_xaxes(title_text="Bulan")
    fig.update_yaxes(title_text="Forecast")
    return fig


def generate_product_detail_chart(hist_df: Optional[pd.DataFrame], fc_df: pd.DataFrame, product_name: str) -> go.Figure:
    fig = go.Figure()
    # Historical
    if hist_df is not None and not hist_df.empty:
        dcol = next((c for c in hist_df.columns if c.lower().startswith("date")), "date")
        ycol = next((c for c in hist_df.columns if c.lower() in ("qty", "quantity", "sales")), None)
        if ycol:
            fig.add_trace(go.Scatter(x=pd.to_datetime(hist_df[dcol]), y=hist_df[ycol], mode="lines",
                                     name="Historis", line=dict(color=MUTED_COLOR, dash="dot")))
    # Forecast
    dcol = next((c for c in fc_df.columns if c.lower().startswith("date")), "date")
    y_mean = next((c for c in fc_df.columns if c.lower() in ("mean", "forecast", "p50")), None)
    if y_mean:
        fig.add_trace(go.Scatter(x=pd.to_datetime(fc_df[dcol]), y=fc_df[y_mean], mode="lines+markers",
                                 name="Forecast", line=dict(color=PRIMARY_COLOR)))
    p10 = next((c for c in fc_df.columns if c.lower() in ("p10", "lower", "lo")), None)
    p90 = next((c for c in fc_df.columns if c.lower() in ("p90", "upper", "hi")), None)
    if p10 and p90:
        fig.add_trace(go.Scatter(x=pd.to_datetime(fc_df[dcol]), y=fc_df[p90], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pd.to_datetime(fc_df[dcol]), y=fc_df[p10], fill="tonexty", line=dict(width=0),
                                 name="Confidence", fillcolor="rgba(37,99,235,0.15)", hoverinfo="skip"))
    return customize_chart_layout(fig, title=f"Forecast Produk: {product_name}")


def generate_diagnostics_charts(diag: pd.DataFrame):
    figs = {}
    if "Historical CV" in diag.columns:
        figs["cv_hist"] = px.histogram(diag, x="Historical CV", nbins=30, title="Distribusi Historical CV")
        customize_chart_layout(figs["cv_hist"])  # mutate
    if "Category" in diag.columns:
        figs["per_cat"] = px.bar(diag.groupby("Category").size().reset_index(name="n"), x="Category", y="n", title="Produk per Kategori")
        customize_chart_layout(figs["per_cat"])  # mutate
    if "Model Used" in diag.columns:
        pie_df = diag["Model Used"].map({True: "LSTM", False: "Fallback"}).value_counts().reset_index()
        pie_df.columns = ["Model", "Count"]
        figs["pie"] = px.pie(pie_df, names="Model", values="Count", title="LSTM vs Fallback")
        customize_chart_layout(figs["pie"])  # mutate
    return figs
