from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer · EDA",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).with_name("Breast_Cancer.csv")
EXPECTED = [
    "Age", "Race", "Marital Status", "T Stage", "N Stage", "6th Stage",
    "differentiate", "Grade", "A Stage", "Tumor Size", "Estrogen Status",
    "Progesterone Status", "Regional Node Examined", "Reginol Node Positive",
    "Survival Months", "Status",
]
NUMERIC = [
    "Age", "Grade", "Tumor Size", "Regional Node Examined",
    "Reginol Node Positive", "Survival Months",
]
STAGE_ORDER = ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IV"]
COLORS = {"Alive": "#21a179", "Dead": "#e76f51"}
PLOT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#dce6f3", "family": "Inter, sans-serif"},
    "margin": {"l": 12, "r": 12, "t": 28, "b": 12},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
}


@st.cache_data(show_spinner=False)
def load_data(uploaded: bytes | None, last_modified: float) -> pd.DataFrame:
    if uploaded is None:
        frame = pd.read_csv(DATA_PATH)
    else:
        from io import BytesIO
        frame = pd.read_csv(BytesIO(uploaded))
    frame.columns = frame.columns.str.strip()
    missing = sorted(set(EXPECTED) - set(frame.columns))
    if missing:
        raise ValueError("The CSV is missing expected columns: " + ", ".join(missing))
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["Status"] = frame["Status"].astype("string").str.strip().str.title()
    frame["6th Stage"] = frame["6th Stage"].astype("string").str.strip().str.upper()
    frame["T Stage"] = frame["T Stage"].astype("string").str.strip().str.upper()
    frame["N Stage"] = frame["N Stage"].astype("string").str.strip().str.upper()
    frame["Grade"] = frame["Grade"].astype("Int64")
    return frame


def multiselect_filter(label: str, column: str, data: pd.DataFrame) -> list:
    options = sorted(data[column].dropna().unique().tolist(), key=str)
    return st.sidebar.multiselect(label, options, default=options)


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data.copy()
    for label, col in [
        ("Outcome", "Status"), ("Stage", "6th Stage"), ("Race", "Race"),
        ("Marital status", "Marital Status"), ("Differentiation", "differentiate"),
        ("Estrogen receptor", "Estrogen Status"),
        ("Progesterone receptor", "Progesterone Status"),
    ]:
        selected = multiselect_filter(label, col, data)
        if selected:
            filtered = filtered[filtered[col].isin(selected)]
        else:
            filtered = filtered.iloc[0:0]
    for label, col in [("Age", "Age"), ("Follow-up (months)", "Survival Months")]:
        lo, hi = int(data[col].min()), int(data[col].max())
        selected = st.sidebar.slider(label, min_value=lo, max_value=hi, value=(lo, hi))
        filtered = filtered[filtered[col].between(*selected)]
    return filtered


def tidy(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(**PLOT, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#26354a")
    fig.update_yaxes(gridcolor="#25344a", zeroline=False, linecolor="#26354a")
    return fig


def show_chart(fig: go.Figure, height: int = 350) -> None:
    description = fig.layout.title.text or "Interactive chart of the selected patient cohort"
    st.plotly_chart(tidy(fig, height), width="stretch", alt=description)


def metric(label: str, value: str, foot: str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """<style>
    .stApp {background: radial-gradient(ellipse at 15% 0%, #17253a 0%, #0b1220 48%, #09101b 100%); color:#e8eef7;}
    [data-testid="stSidebar"] {background:#0d1726; border-right:1px solid #1e2d42;}
    .block-container {padding-top:2rem; padding-bottom:3rem; max-width:1500px;}
    h1,h2,h3 {letter-spacing:-.025em;}
    .hero {padding:1.5rem 1.8rem; border:1px solid #263750; border-radius:20px; background:linear-gradient(110deg,rgba(35,65,95,.58),rgba(17,29,46,.72)); margin-bottom:1.3rem;}
    .eyebrow {color:#64d8c2; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700;}
    .hero h1 {font-size:2.2rem; margin:.35rem 0 .4rem;}
    .hero p {color:#a9b8cb; margin:0; max-width:900px;}
    .metric {background:linear-gradient(145deg,#142237,#101a2a); border:1px solid #25364d; border-radius:16px; padding:1rem 1.1rem; min-height:112px;}
    .metric-label {font-size:.78rem;color:#9aabc0;letter-spacing:.04em;}
    .metric-value {font-size:1.8rem;font-weight:700;margin:.25rem 0;color:#f4f7fb;}
    .metric-foot {font-size:.75rem;color:#8ea1b8;}
    .insight {border-left:3px solid #64d8c2;background:#122034;padding:.85rem 1rem;border-radius:0 12px 12px 0;color:#c8d5e5;margin:.45rem 0;}
    .stTabs [data-baseweb="tab-list"] {gap:.45rem;}
    .stTabs [data-baseweb="tab"] {background:#111d2e;border-radius:10px 10px 0 0;padding:.6rem 1rem;}
    [data-testid="stDataFrame"] {border:1px solid #263750;border-radius:12px;overflow:hidden;}
    </style>""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## ◉  Cohort controls")
st.sidebar.caption("Filters update every chart, metric, and download.")
uploaded_file = st.sidebar.file_uploader("Explore another CSV", type=["csv"])
try:
    upload_bytes = uploaded_file.getvalue() if uploaded_file else None
    mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0
    raw = load_data(upload_bytes, mtime)
except Exception as exc:
    st.error(f"Could not load this dataset: {exc}")
    st.stop()

filtered = apply_filters(raw)
if filtered.empty:
    st.warning("No patients match these filters. Adjust the cohort controls in the sidebar.")
    st.stop()

total = len(filtered)
alive = int(filtered["Status"].eq("Alive").sum())
dead = int(filtered["Status"].eq("Dead").sum())
alive_rate = alive / total * 100 if total else 0
median_survival = filtered["Survival Months"].median()
median_tumor = filtered["Tumor Size"].median()

st.markdown(
    '<div class="hero"><div class="eyebrow">Clinical cohort explorer · EDA</div>'
    '<h1>Breast cancer outcomes</h1>'
    '<p>Explore how stage, tumor burden, lymph node involvement, receptor status, and patient demographics relate to recorded outcomes in this dataset.</p></div>',
    unsafe_allow_html=True,
)

if raw.duplicated().any():
    st.caption(f"Source data: {len(raw):,} records · {int(raw.duplicated().sum()):,} exact duplicate rows present. All source rows are retained in this dashboard.")
else:
    st.caption(f"Source data: {len(raw):,} records · {raw.shape[1]} fields · all source rows retained.")

cards = st.columns(4)
with cards[0]: metric("PATIENTS IN COHORT", f"{total:,}", f"of {len(raw):,} source records")
with cards[1]: metric("RECORDED ALIVE", f"{alive_rate:.1f}%", f"{alive:,} alive · {dead:,} dead")
with cards[2]: metric("MEDIAN FOLLOW-UP", f"{median_survival:.0f} mo" if pd.notna(median_survival) else "—", "survival months in the selected cohort")
with cards[3]: metric("MEDIAN TUMOR SIZE", f"{median_tumor:.0f} mm" if pd.notna(median_tumor) else "—", "recorded tumor size")

st.markdown("### Explore the cohort")
overview_tab, outcome_tab, tumor_tab, patient_tab, data_tab = st.tabs(
    ["Overview", "Survival & stage", "Tumor & nodes", "Patient profile", "Data explorer"]
)

with overview_tab:
    left, right = st.columns([1.15, 1])
    with left:
        by_stage = filtered.groupby(["6th Stage", "Status"], observed=True).size().rename("Patients").reset_index()
        order = [s for s in STAGE_ORDER if s in by_stage["6th Stage"].unique()]
        order += [s for s in by_stage["6th Stage"].unique() if s not in order]
        fig = px.bar(by_stage, x="6th Stage", y="Patients", color="Status", barmode="stack",
                     category_orders={"6th Stage": order}, color_discrete_map=COLORS,
                     title="Patient count by cancer stage and status")
        show_chart(fig)
    with right:
        status_counts = filtered["Status"].value_counts().rename_axis("Status").reset_index(name="Patients")
        fig = px.pie(status_counts, names="Status", values="Patients", hole=.66,
                     color="Status", color_discrete_map=COLORS, title="Recorded outcome mix")
        fig.update_traces(textinfo="percent+label", marker_line_color="#101a2a", marker_line_width=3)
        show_chart(fig)
    a, b = st.columns(2)
    with a:
        stage_rates = filtered.groupby("6th Stage", observed=True).agg(
            Patients=("Status", "size"), Alive=("Status", lambda s: s.eq("Alive").mean() * 100)
        ).reset_index().sort_values("Alive")
        fig = px.bar(stage_rates, x="Alive", y="6th Stage", orientation="h", color="Alive",
                     color_continuous_scale=["#e76f51", "#e9c46a", "#21a179"],
                     hover_data={"Patients": True, "Alive": ":.1f"}, title="Alive share within each stage")
        fig.update_xaxes(title="Alive in recorded status (%)", range=[0, 100])
        show_chart(fig)
    with b:
        fig = px.histogram(filtered, x="Age", color="Status", marginal="box", nbins=24,
                           color_discrete_map=COLORS, barmode="overlay", opacity=.8,
                           title="Age distribution by recorded outcome")
        show_chart(fig)

    most_common = filtered["6th Stage"].mode()
    if not most_common.empty:
        stage_rows = filtered[filtered["6th Stage"] == most_common.iloc[0]]
        st.markdown(f'<div class="insight"><b>In this cohort:</b> stage {most_common.iloc[0]} is most common ({len(stage_rows):,} patients); the recorded alive share is {alive_rate:.1f}%. These are descriptive patterns, not treatment effects or individual forecasts.</div>', unsafe_allow_html=True)

with outcome_tab:
    left, right = st.columns(2)
    with left:
        surv_stage = filtered.groupby(["6th Stage", "Status"], observed=True)["Survival Months"].median().reset_index()
        order = [s for s in STAGE_ORDER if s in surv_stage["6th Stage"].unique()]
        fig = px.bar(surv_stage, x="6th Stage", y="Survival Months", color="Status", barmode="group",
                     category_orders={"6th Stage": order}, color_discrete_map=COLORS,
                     title="Median recorded survival months by stage and status")
        show_chart(fig)
    with right:
        fig = px.box(filtered, x="6th Stage", y="Survival Months", color="Status", points="outliers",
                     category_orders={"6th Stage": STAGE_ORDER}, color_discrete_map=COLORS,
                     title="Follow-up distribution across stages")
        show_chart(fig)
    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered, x="Survival Months", color="Status", nbins=30, barmode="overlay",
                           opacity=.75, color_discrete_map=COLORS, title="Follow-up month distribution")
        show_chart(fig)
    with right:
        cross = pd.crosstab(filtered["6th Stage"], filtered["Status"], normalize="index") * 100
        fig = px.imshow(cross, text_auto=".1f", color_continuous_scale="Tealgrn", aspect="auto",
                        labels={"x": "Recorded status", "y": "6th Stage", "color": "% within stage"},
                        title="Outcome mix within each stage (%)")
        show_chart(fig)

with tumor_tab:
    left, right = st.columns(2)
    with left:
        fig = px.scatter(filtered, x="Tumor Size", y="Reginol Node Positive", color="Status",
                         size="Regional Node Examined", hover_data=["Age", "6th Stage", "Survival Months"],
                         color_discrete_map=COLORS, opacity=.7,
                         title="Tumor size and positive regional nodes")
        show_chart(fig)
    with right:
        fig = px.box(filtered, x="6th Stage", y="Tumor Size", color="Status", points="outliers",
                     category_orders={"6th Stage": STAGE_ORDER}, color_discrete_map=COLORS,
                     title="Tumor size distribution by stage")
        show_chart(fig)
    left, right = st.columns(2)
    with left:
        fig = px.scatter(filtered, x="Reginol Node Positive", y="Survival Months", color="Status",
                         size="Tumor Size", hover_data=["Age", "6th Stage"], opacity=.7,
                         color_discrete_map=COLORS, title="Positive nodes and recorded follow-up")
        show_chart(fig)
    with right:
        node_summary = filtered.groupby("6th Stage", observed=True).agg(
            Patients=("Reginol Node Positive", "size"),
            Median_positive_nodes=("Reginol Node Positive", "median"),
            Median_examined=("Regional Node Examined", "median"),
        ).reset_index()
        fig = px.bar(node_summary, x="6th Stage", y="Median_positive_nodes", color="Median_positive_nodes",
                     hover_data=["Patients", "Median_examined"], category_orders={"6th Stage": STAGE_ORDER},
                     color_continuous_scale="Tealgrn", title="Median positive nodes by stage")
        fig.update_yaxes(title="Median positive regional nodes")
        show_chart(fig)

    corr_cols = ["Age", "Grade", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]
    corr = filtered[corr_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Numeric feature correlations (Pearson r)")
    show_chart(fig, 430)
    st.caption("Correlation describes linear association in this filtered cohort; it does not establish cause and effect.")

with patient_tab:
    left, right = st.columns(2)
    with left:
        race = filtered.groupby(["Race", "Status"], observed=True).size().rename("Patients").reset_index()
        fig = px.bar(race, x="Race", y="Patients", color="Status", barmode="group", color_discrete_map=COLORS,
                     title="Recorded outcome by race")
        show_chart(fig)
    with right:
        marital = filtered.groupby(["Marital Status", "Status"], observed=True).size().rename("Patients").reset_index()
        fig = px.bar(marital, x="Marital Status", y="Patients", color="Status", barmode="group", color_discrete_map=COLORS,
                     title="Recorded outcome by marital status")
        show_chart(fig)
    left, right = st.columns(2)
    with left:
        receptor = filtered.groupby(["Estrogen Status", "Progesterone Status", "Status"], observed=True).size().rename("Patients").reset_index()
        fig = px.bar(receptor, x="Estrogen Status", y="Patients", color="Status", barmode="group",
                     facet_col="Progesterone Status", color_discrete_map=COLORS,
                     title="Outcome by hormone receptor status")
        show_chart(fig)
    with right:
        grade = filtered.groupby(["Grade", "Status"], observed=True).size().rename("Patients").reset_index()
        fig = px.bar(grade, x="Grade", y="Patients", color="Status", barmode="group", color_discrete_map=COLORS,
                     title="Recorded outcome by tumor grade")
        show_chart(fig)

with data_tab:
    st.markdown("#### Filtered patient records")
    query = st.text_input("Search visible records", placeholder="Search across visible values…")
    view = filtered.copy()
    if query:
        mask = view.astype("string").apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.caption(f"Showing {len(view):,} rows from a filtered cohort of {total:,} patients.")
    st.dataframe(view, width="stretch", hide_index=True, height=480, alt="Patient records in the selected cohort")
    st.download_button("Download filtered records (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="breast_cancer_filtered.csv", mime="text/csv")
    st.markdown("#### Dataset quality and field guide")
    qc = pd.DataFrame({
        "Field": raw.columns,
        "Type": raw.dtypes.astype(str).values,
        "Missing": raw.isna().sum().values,
        "Unique values": raw.nunique(dropna=True).values,
    })
    st.dataframe(qc, width="stretch", hide_index=True, alt="Missing values and unique counts by dataset field")
    st.caption("This dashboard retains all source rows, including outliers, so rare values and advanced-stage records remain visible. The source notebook's blanket IQR filtering can remove clinically relevant observations. Status and survival months are shown as recorded in the CSV; survival curves and causal estimates are not inferred from this dataset.")

st.markdown("---")
st.caption("EDA for exploration only · Group differences can reflect cohort composition and follow-up. This dashboard is not a clinical decision tool.")
