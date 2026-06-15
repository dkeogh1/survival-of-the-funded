"""Interactive Plotly visualizations of the analysis.

Produces standalone HTML in output/figures/ plus a combined index.html:
  1. firm-fate Sankey       -- the 2004 cohort's journey across 8 years
  2. cumulative incidence   -- survival / failure / acquisition over time
  3. hazard-ratio forest    -- original (M&A dropped) vs competing-risks
  4. reproduction dumbbell  -- thesis published HR vs this reproduction
  5. interaction heatmap    -- Friedman's H-statistic, thesis picks marked

Run: uv run python src/visualize.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyreadstat
import plotly.graph_objects as go
import plotly.io as pio
from lifelines import AalenJohansenFitter

import config as C
import build_dataset as B

FIG = C.OUTPUT / "figures"
PALETTE = dict(active="#2a9d8f", failed="#e76f51", acquired="#457b9d",
               dropped="#bdbdbd", protective="#2a9d8f", harmful="#e76f51",
               thesis="#e9c46a", repro="#457b9d")
pio.templates.default = "plotly_white"


# --------------------------------------------------------------- 1. Sankey
def firm_state_matrix() -> pd.DataFrame:
    cols = ["mprid"] + C.STATUS + C.OUT_OF_BUSINESS
    df, _ = pyreadstat.read_dta(str(C.RAW_DTA), usecols=cols)
    status = {w: B._num(df, f"final_status_code_{w}") for w in C.WAVES}
    a10 = {w: B._num(df, f"a10_out_of_business_{w}") for w in C.FOLLOWUP_WAVES}
    states = np.empty((len(df), 8), dtype=object)
    states[:, 0] = "Active"
    for i in range(len(df)):
        absorbed = None
        for w in C.FOLLOWUP_WAVES:
            if absorbed:
                states[i, w] = absorbed
                continue
            st = B._classify_wave(status[w].iat[i], a10[w].iat[i])
            if st == "alive":
                states[i, w] = "Active"
            elif st == "out_of_business":
                states[i, w] = absorbed = "Failed"
            elif st == "sold_merged":
                states[i, w] = absorbed = "Acquired"
            else:
                states[i, w] = "Dropped"      # non-response / ineligible
    return pd.DataFrame(states, columns=[f"y{w}" for w in C.WAVES])


def fig_sankey() -> go.Figure:
    sm = firm_state_matrix()
    order = ["Active", "Failed", "Acquired", "Dropped"]
    color = {"Active": PALETTE["active"], "Failed": PALETTE["failed"],
             "Acquired": PALETTE["acquired"], "Dropped": PALETTE["dropped"]}
    # node index per (wave, state)
    nodes, node_idx, node_color, node_x = [], {}, [], []
    for w in C.WAVES:
        for s in order:
            node_idx[(w, s)] = len(nodes)
            nodes.append(f"{s}")
            node_color.append(color[s])
            node_x.append(w / 7)
    src, tgt, val, link_color = [], [], [], []
    for w in range(7):
        g = sm.groupby([f"y{w}", f"y{w+1}"]).size()
        for (a, b), n in g.items():
            src.append(node_idx[(w, a)]); tgt.append(node_idx[(w + 1, b)])
            val.append(int(n))
            rgb = color[b].lstrip("#")
            link_color.append("rgba(%d,%d,%d,0.35)" % tuple(int(rgb[k:k+2], 16) for k in (0, 2, 4)))
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, color=node_color, x=node_x, pad=18, thickness=16,
                  line=dict(width=0)),
        link=dict(source=src, target=tgt, value=val, color=link_color)))
    fig.update_layout(
        title="Where the 2004 KFS startup cohort ended up (4,928 firms, 8 years)",
        font_size=12, height=560)
    return fig


# ------------------------------------------------ 2. cumulative incidence
def fig_incidence() -> go.Figure:
    df = pd.read_parquet(C.PROCESSED / "extended.parquet")
    grid = np.arange(0, 9)
    cif = {}
    for cause, name in [(1, "Failed"), (2, "Acquired")]:
        ajf = AalenJohansenFitter(calculate_variance=False)
        ajf.fit(df["duration"], df["event_type"], event_of_interest=cause)
        cif[name] = ajf.predict(grid).values
    failed, acq = cif["Failed"], cif["Acquired"]
    alive = 1 - failed - acq
    fig = go.Figure()
    for name, y, col in [("Still active", alive, PALETTE["active"]),
                         ("Acquired (exit)", acq, PALETTE["acquired"]),
                         ("Failed", failed, PALETTE["failed"])]:
        fig.add_trace(go.Scatter(x=grid, y=y * 100, name=name, mode="lines",
                                 stackgroup="one", line=dict(width=0.5, color=col)))
    fig.update_layout(title="Competing-risks outcome of the cohort over time",
                      xaxis_title="Years since founding (2004)",
                      yaxis_title="Share of firms (%)", height=480,
                      yaxis_range=[0, 100], hovermode="x unified")
    return fig


# --------------------------------------------------- helpers for HR figs
def _ci_from_hr_z(hr, z):
    """95% CI for a hazard ratio given HR and z-stat (se = log(HR)/z)."""
    loghr = np.log(hr)
    se = np.where(np.abs(z) > 1e-6, np.abs(loghr / z), np.nan)
    return np.exp(loghr - 1.96 * se), np.exp(loghr + 1.96 * se)


LABELS = {"eqangels": "Angel equity", "eqvc": "Venture capital",
          "eqgovt": "Government equity", "eqfff": "Friends/Family/Fools",
          "debtfin": "Debt financing", "univcompadv": "University partnership",
          "compcompadv": "Company partnership", "patentcompadv": "Patent advantage",
          "foundered": "Founder education", "founderworkexp": "Founder experience",
          "foundage": "Founder age"}


# ------------------------------------------------------- 3. forest plot
def fig_forest() -> go.Figure:
    a = pd.read_csv(C.OUTPUT / "tables" / "cox_A_main.csv").set_index("variable")
    cr = pd.read_csv(C.OUTPUT / "tables" / "cox_causespecific_failure.csv").set_index("variable")
    rows = [v for v in LABELS if v in a.index and v in cr.index]
    fig = go.Figure()
    for src_df, name, col, dy in [(a, "Original (M&A dropped)", PALETTE["harmful"], -0.16),
                                  (cr, "Competing risks", PALETTE["protective"], 0.16)]:
        hr = src_df.loc[rows, "hazard_ratio"].values
        lo, hi = _ci_from_hr_z(hr, src_df.loc[rows, "z"].values)
        y = np.arange(len(rows)) + dy
        fig.add_trace(go.Scatter(
            x=hr, y=y, mode="markers", name=name, marker=dict(color=col, size=9),
            error_x=dict(type="data", symmetric=False, array=hi - hr, arrayminus=hr - lo,
                         color=col, thickness=1.5)))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#888")
    fig.update_layout(
        title="Failure hazard ratios: dropping M&A vs. modeling it as a competing event",
        xaxis_title="Hazard ratio (log scale)  —  <1 protective, >1 harmful",
        xaxis_type="log", height=620,
        yaxis=dict(tickmode="array", tickvals=list(range(len(rows))),
                   ticktext=[LABELS[r] for r in rows]),
        legend=dict(orientation="h", y=1.06))
    return fig


# --------------------------------------------------- 4. reproduction dumbbell
THESIS_HR = {"eqangels": 0.28, "eqvc": 1.03, "eqgovt": 3.62, "eqfff": 0.55,
             "debtfin": 0.13, "univcompadv": 0.68, "compcompadv": 0.175,
             "patentcompadv": 0.51, "foundered": 0.977, "founderworkexp": 0.99,
             "foundage": 0.96}


def fig_dumbbell() -> go.Figure:
    a = pd.read_csv(C.OUTPUT / "tables" / "cox_A_main.csv").set_index("variable")
    rows = [v for v in THESIS_HR if v in a.index]
    rows.sort(key=lambda v: a.loc[v, "hazard_ratio"])
    fig = go.Figure()
    for v in rows:
        th, rp = THESIS_HR[v], a.loc[v, "hazard_ratio"]
        agree = (th < 1) == (rp < 1)
        fig.add_trace(go.Scatter(
            x=[th, rp], y=[LABELS[v], LABELS[v]], mode="lines",
            line=dict(color="#2a9d8f" if agree else "#e76f51", width=2),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[THESIS_HR[v] for v in rows], y=[LABELS[v] for v in rows],
                  mode="markers", name="Thesis (published)",
                  marker=dict(color=PALETTE["thesis"], size=12, line=dict(width=1, color="#777"))))
    fig.add_trace(go.Scatter(x=[a.loc[v, "hazard_ratio"] for v in rows], y=[LABELS[v] for v in rows],
                  mode="markers", name="This reproduction",
                  marker=dict(color=PALETTE["repro"], size=12, line=dict(width=1, color="#777"))))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#888")
    fig.update_layout(
        title="Reproduction scorecard: published hazard ratios vs. this rebuild"
              "<br><sup>green = same direction, red = sign flip (the financing gap)</sup>",
        xaxis_title="Hazard ratio (log scale)", xaxis_type="log", height=560,
        legend=dict(orientation="h", y=1.07))
    return fig


# --------------------------------------------------- 5. interaction heatmap
def fig_heatmap() -> go.Figure:
    H = pd.read_csv(C.OUTPUT / "tables" / "interaction_Hstat.csv")
    feats = pd.unique(H[["feat_a", "feat_b"]].values.ravel())
    feats = sorted(feats, key=lambda f: H[(H.feat_a == f) | (H.feat_b == f)]["H2"].max(),
                   reverse=True)
    M = pd.DataFrame(np.nan, index=feats, columns=feats)
    for _, r in H.iterrows():
        M.loc[r.feat_a, r.feat_b] = r.H2
        M.loc[r.feat_b, r.feat_a] = r.H2
    thesis = {("foundered", "univcompadv"), ("founderexpsameind", "compcompadv"),
              ("founderworkexp", "compcompadv")}
    text = M.copy().astype(object)
    for a in feats:
        for b in feats:
            mark = "★" if (a, b) in thesis or (b, a) in thesis else ""
            text.loc[a, b] = mark
    fig = go.Figure(go.Heatmap(
        z=M.values, x=feats, y=feats, colorscale="YlOrRd",
        text=text.values, texttemplate="%{text}", textfont=dict(size=16, color="#1d3557"),
        colorbar=dict(title="H²"), hovertemplate="%{y} × %{x}<br>H²=%{z:.3f}<extra></extra>"))
    fig.update_layout(
        title="Data-driven interaction strength (Friedman's H²)  —  ★ = thesis's hand-picked term",
        height=620, xaxis=dict(tickangle=40))
    return fig


# --------------------------------------------------------------- assemble
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    figs = [("1_firm_fate_sankey", fig_sankey()),
            ("2_cumulative_incidence", fig_incidence()),
            ("3_hazard_forest", fig_forest()),
            ("4_reproduction_scorecard", fig_dumbbell()),
            ("5_interaction_heatmap", fig_heatmap())]
    divs = []
    for name, fig in figs:
        fig.write_html(FIG / f"{name}.html", include_plotlyjs="cdn")
        divs.append(fig.to_html(full_html=False, include_plotlyjs=False))
        print(f"  wrote output/figures/{name}.html")
    index = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        "<title>Survival of the Funded — figures</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem;color:#222}h1{margin-bottom:.2rem}p.sub{color:#666;margin-top:0}"
        "div.fig{margin:2.5rem 0;border-top:1px solid #eee;padding-top:1rem}</style></head><body>"
        "<h1>Survival of the Funded — interactive figures</h1>"
        "<p class='sub'>Reproduction &amp; extensions of Keogh (JEMI 2021) on the Kauffman Firm Survey.</p>"
        + "".join(f"<div class='fig'>{d}</div>" for d in divs)
        + "</body></html>")
    (FIG / "index.html").write_text(index)
    print(f"  wrote output/figures/index.html  (all {len(figs)} figures in one page)")


if __name__ == "__main__":
    main()
