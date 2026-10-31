"""Interactive dashboard for the breast cancer EDA dataset."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer Outcomes | EDA",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "Breast_Cancer.csv"
REQUIRED_COLUMNS = {
    "Age", "Status", "6th Stage", "Survival Months", "Tumor Size",
    "Estrogen Status", "Progesterone Status",
}
COLORS = {"Alive": "#48b89a", "Dead": "#ef7890"}
PLOT_TEMPLATE = "plotly_dark"


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    data = pd.read_csv(BytesIO(file_bytes))
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("string").str.strip()
    for col in ("Age", "Grade", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def chart_layout(fig, height=360):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        title_font_size=16,
        hoverlabel=dict(bgcolor="#15212b", font_color="#f5f7fa"),
    )
    fig.update_xaxes(gridcolor="rgba(160,180,190,.12)", zerolinecolor="rgba(160,180,190,.18)")
    fig.update_yaxes(gridcolor="rgba(160,180,190,.12)", zerolinecolor="rgba(160,180,190,.18)")
    return fig


def pct(value):
    return f"{value:.1f}%" if pd.notna(value) else "—"


with st.sidebar:
    st.markdown("## 🎗️ EDA dashboard")
    st.caption("Breast cancer patient outcomes")
    upload = st.file_uploader("Use a CSV dataset", type=["csv"], help="Upload a compatible file to replace the bundled dataset.")
    try:
        source_bytes = upload.getvalue() if upload else DEFAULT_CSV.read_bytes()
        source_name = upload.name if upload else DEFAULT_CSV.name
        raw = load_csv(source_bytes)
    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")
        st.stop()

    missing_required = REQUIRED_COLUMNS - set(raw.columns)
    if missing_required:
        st.error("This CSV is missing required columns: " + ", ".join(sorted(missing_required)))
        st.stop()

    st.markdown("### Filter the cohort")
    working = raw.copy()
    if "Status" in working:
        status_options = sorted(working["Status"].dropna().unique().tolist())
        statuses = st.multiselect("Outcome", status_options, default=status_options)
        working = working[working["Status"].isin(statuses)]
    if "6th Stage" in working:
        stage_options = sorted(working["6th Stage"].dropna().unique().tolist())
        stages = st.multiselect("Clinical stage", stage_options, default=stage_options)
        working = working[working["6th Stage"].isin(stages)]

    if "Age" in working and working["Age"].notna().any():
        age_min, age_max = int(working["Age"].min()), int(working["Age"].max())
        age_range = st.slider("Age range", age_min, age_max, (age_min, age_max), step=1)
        working = working[working["Age"].between(*age_range) | working["Age"].isna()]
    if "Survival Months" in working and working["Survival Months"].notna().any():
        month_min, month_max = int(working["Survival Months"].min()), int(working["Survival Months"].max())
        if month_min < month_max:
            month_range = st.slider("Follow-up months", month_min, month_max, (month_min, month_max), step=1)
            working = working[working["Survival Months"].between(*month_range) | working["Survival Months"].isna()]

    categorical_filters = [
        ("Race", "Race"), ("Marital Status", "Marital status"),
        ("Estrogen Status", "Estrogen receptor"), ("Progesterone Status", "Progesterone receptor"),
        ("A Stage", "A stage"),
    ]
    for column, label in categorical_filters:
        if column in working:
            choices = sorted(working[column].dropna().unique().tolist())
            selected = st.multiselect(label, choices, default=choices, key=f"filter_{column}")
            working = working[working[column].isin(selected)]

    st.divider()
    st.caption(f"Source: {source_name} · {len(raw):,} rows")


st.caption("PATIENT OUTCOMES · EXPLORATORY ANALYSIS")
st.title("Breast cancer, in context.")
st.write("Explore how clinical stage, tumor characteristics, receptor status and patient demographics relate to recorded outcomes and follow-up time.")

if working.empty:
    st.warning("No records match these filters. Expand the cohort in the sidebar.")
    st.stop()

alive = working[working["Status"].str.casefold() == "alive"]
alive_rate = len(alive) / len(working) * 100
median_months = working["Survival Months"].median()
median_age = working["Age"].median()
de_dup = raw.drop_duplicates()
duplicate_count = len(raw) - len(de_dup)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Patients in view", f"{len(working):,}", border=True)
k2.metric("Recorded alive", pct(alive_rate), border=True)
k3.metric("Recorded dead", f"{(working['Status'].str.casefold() == 'dead').sum():,}", border=True)
k4.metric("Median follow-up", f"{median_months:.0f} mo" if pd.notna(median_months) else "—", border=True)
k5.metric("Median age", f"{median_age:.0f} yr" if pd.notna(median_age) else "—", border=True)

overview_tab, stage_tab, profile_tab, data_tab = st.tabs(["Overview", "Stage & outcomes", "Patient profile", "Explore records"])

with overview_tab:
    left, right = st.columns([1.05, 1.45], gap="large")
    with left:
        counts = working["Status"].value_counts().rename_axis("Status").reset_index(name="Patients")
        fig = px.pie(counts, names="Status", values="Patients", hole=.68, color="Status", color_discrete_map=COLORS,
                     title="Recorded outcome mix")
        fig.update_traces(textposition="inside", textinfo="percent+label", marker=dict(line=dict(color="#10202a", width=3)))
        chart_layout(fig, 340)
        st.plotly_chart(fig, width="stretch", alt="Proportion of patients with each recorded outcome")
    with right:
        fig = px.histogram(working.dropna(subset=["Survival Months"]), x="Survival Months", color="Status",
                           nbins=30, barmode="overlay", opacity=.82, color_discrete_map=COLORS,
                           title="Follow-up months by recorded outcome")
        chart_layout(fig, 340)
        fig.update_layout(xaxis_title="Months", yaxis_title="Patients")
        st.plotly_chart(fig, width="stretch", alt="Distribution of recorded follow-up months by outcome")

    left, right = st.columns(2, gap="large")
    with left:
        age_data = working.dropna(subset=["Age"])
        fig = px.histogram(age_data, x="Age", color="Status", nbins=24, barmode="overlay", opacity=.8,
                           color_discrete_map=COLORS, title="Age distribution")
        chart_layout(fig, 330)
        fig.update_layout(xaxis_title="Age (years)", yaxis_title="Patients")
        st.plotly_chart(fig, width="stretch", alt="Age distribution grouped by recorded outcome")
    with right:
        stage_counts = working.groupby(["6th Stage", "Status"]).size().reset_index(name="Patients")
        stage_order = [s for s in ["IIA", "IIB", "IIIA", "IIIB", "IIIC"] if s in working["6th Stage"].unique()]
        fig = px.bar(stage_counts, x="6th Stage", y="Patients", color="Status", barmode="group",
                     category_orders={"6th Stage": stage_order}, color_discrete_map=COLORS,
                     title="Patient count by clinical stage")
        chart_layout(fig, 330)
        st.plotly_chart(fig, width="stretch", alt="Patient counts for each stage, grouped by recorded outcome")

    stage_totals = working.groupby("6th Stage").size()
    stage_outcomes = working.groupby(["6th Stage", "Status"]).size().unstack(fill_value=0)
    stage_outcomes["Total"] = stage_outcomes.sum(axis=1)
    if "Dead" in stage_outcomes:
        sufficiently_sized = stage_outcomes[stage_outcomes["Total"] >= 20]
        highest_dead_share = (sufficiently_sized["Dead"] / sufficiently_sized["Total"]).idxmax() if not sufficiently_sized.empty else None
    else:
        highest_dead_share = None
    alive_followup = working.loc[working["Status"].str.casefold() == "alive", "Survival Months"].median()
    dead_followup = working.loc[working["Status"].str.casefold() == "dead", "Survival Months"].median()
    with st.container(border=True):
        st.markdown("#### Signals in this filtered cohort")
        signals = st.columns(3)
        if not stage_totals.empty:
            major_stage = stage_totals.idxmax()
            signals[0].markdown(f"**Most represented stage**  \n{major_stage} · {stage_totals[major_stage]:,} patients")
        if highest_dead_share is not None:
            dead_share = stage_outcomes.loc[highest_dead_share, "Dead"] / stage_outcomes.loc[highest_dead_share, "Total"] * 100
            signals[1].markdown(f"**Highest recorded dead share**  \n{highest_dead_share} · {dead_share:.1f}% (n={int(stage_outcomes.loc[highest_dead_share, 'Total']):,})")
        if pd.notna(alive_followup) and pd.notna(dead_followup):
            signals[2].markdown(f"**Median follow-up by outcome**  \nAlive {alive_followup:.0f} mo · Dead {dead_followup:.0f} mo")
        st.caption("Stage ranking excludes groups smaller than 20 records. These are descriptive differences in the selected rows, not adjusted risks or survival estimates.")

    st.info("Outcome labels and follow-up months are descriptive fields in this dataset. They do not provide a time-to-event estimate or adjust for differences in patient characteristics.")

with stage_tab:
    st.subheader("Stage comparisons")
    st.caption("Counts and within-stage outcome shares are both shown to help account for unequal cohort sizes.")
    stage_summary = working.groupby(["6th Stage", "Status"]).size().unstack(fill_value=0)
    stage_summary["Total"] = stage_summary.sum(axis=1)
    stage_summary["Alive share"] = (stage_summary.get("Alive", 0) / stage_summary["Total"] * 100).round(1)
    order = [s for s in ["IIA", "IIB", "IIIA", "IIIB", "IIIC"] if s in stage_summary.index] + [s for s in stage_summary.index if s not in ["IIA", "IIB", "IIIA", "IIIB", "IIIC"]]
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        stage_long = working.groupby(["6th Stage", "Status"]).size().reset_index(name="Patients")
        fig = px.bar(stage_long, x="6th Stage", y="Patients", color="Status", barmode="group",
                     category_orders={"6th Stage": order}, color_discrete_map=COLORS,
                     title="Outcome counts by stage")
        chart_layout(fig, 390)
        st.plotly_chart(fig, width="stretch", alt="Counts of alive and dead outcomes across clinical stages")
    with right:
        if "Alive" in stage_summary:
            fig = px.bar(stage_summary.reset_index(), x="6th Stage", y="Alive share", range_y=[0, 100],
                         category_orders={"6th Stage": order}, title="Recorded alive share within stage",
                         color_discrete_sequence=["#64d6b1"])
            chart_layout(fig, 390)
            fig.update_layout(yaxis_title="Alive share (%)")
            st.plotly_chart(fig, width="stretch", alt="Percentage labeled alive within each clinical stage")
        else:
            st.info("No records with an Alive outcome in this filtered cohort.")

    followup = working.groupby(["6th Stage", "Status"], as_index=False).agg(
        Median_months=("Survival Months", "median"), Patients=("Survival Months", "count")
    )
    fig = px.bar(followup, x="6th Stage", y="Median_months", color="Status", barmode="group",
                 category_orders={"6th Stage": order}, color_discrete_map=COLORS,
                 hover_data=["Patients"], title="Median recorded follow-up by stage and outcome")
    chart_layout(fig, 360)
    fig.update_layout(yaxis_title="Median follow-up (months)")
    st.plotly_chart(fig, width="stretch", alt="Median recorded follow-up by stage and recorded outcome")
    st.dataframe(stage_summary.reindex(order), width="stretch", alt="Patient count and recorded outcome share by clinical stage")

with profile_tab:
    st.subheader("Clinical and demographic patterns")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if "Tumor Size" in working:
            tumor = working.dropna(subset=["Tumor Size"])
            fig = px.box(tumor, x="Status", y="Tumor Size", color="Status", points="outliers",
                         color_discrete_map=COLORS, title="Tumor size by recorded outcome")
            chart_layout(fig, 360)
            fig.update_layout(xaxis_title="Outcome", yaxis_title="Tumor size")
            st.plotly_chart(fig, width="stretch", alt="Tumor size distribution split by recorded outcome")
    with c2:
        receptors = [c for c in ["Estrogen Status", "Progesterone Status"] if c in working]
        if receptors:
            receptor_data = working.melt(id_vars="Status", value_vars=receptors, var_name="Receptor", value_name="Result")
            table = receptor_data.groupby(["Receptor", "Result", "Status"]).size().reset_index(name="Patients")
            fig = px.bar(table, x="Receptor", y="Patients", color="Status", barmode="group",
                         facet_col="Result", color_discrete_map=COLORS, title="Receptor profile and outcome")
            chart_layout(fig, 360)
            fig.update_layout(xaxis_title="Receptor", yaxis_title="Patients")
            st.plotly_chart(fig, width="stretch", alt="Patient outcome counts across receptor status groups")

    c1, c2 = st.columns(2, gap="large")
    for col, column, heading in [(c1, "Race", "Outcome by race"), (c2, "Marital Status", "Outcome by marital status")]:
        if column in working:
            with col:
                grouping = working.groupby([column, "Status"]).size().reset_index(name="Patients")
                fig = px.bar(grouping, x=column, y="Patients", color="Status", barmode="group",
                             color_discrete_map=COLORS, title=heading)
                chart_layout(fig, 340)
                fig.update_layout(xaxis_title="", yaxis_title="Patients")
                st.plotly_chart(fig, width="stretch", alt=heading)

    numeric_candidates = [c for c in ["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"] if c in working]
    if len(numeric_candidates) >= 2:
        st.markdown("#### Numeric relationships")
        x_col = st.selectbox("Horizontal axis", numeric_candidates, index=0, key="scatter_x")
        y_col = st.selectbox("Vertical axis", [c for c in numeric_candidates if c != x_col], index=0, key="scatter_y")
        color_col = st.selectbox("Color points by", [c for c in ["Status", "6th Stage", "Grade"] if c in working], key="scatter_color")
        fig = px.scatter(working, x=x_col, y=y_col, color=color_col, opacity=.65,
                         hover_data=[c for c in ["Age", "6th Stage", "Tumor Size", "Survival Months"] if c in working],
                         title=f"{y_col} vs {x_col}")
        chart_layout(fig, 430)
        st.plotly_chart(fig, width="stretch", alt=f"Scatter plot of {y_col} against {x_col}, colored by {color_col}")

with data_tab:
    st.subheader("Filtered records")
    st.caption(f"Showing {len(working):,} of {len(raw):,} rows. Use the column menu to sort or search.")
    search = st.text_input("Search visible fields", placeholder="For example: IIA, Married, Positive")
    shown = working.copy()
    if search:
        mask = shown.astype("string").apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        shown = shown[mask]
    st.dataframe(shown, width="stretch", hide_index=True, height=470, alt="Filtered breast cancer patient records")
    st.download_button("Download filtered CSV", shown.to_csv(index=False).encode("utf-8"),
                       file_name="breast_cancer_filtered.csv", mime="text/csv")

with st.expander("Data quality & interpretation"):
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Exact duplicate rows", f"{duplicate_count:,}")
    q2.metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
    q3.metric("Columns", f"{raw.shape[1]}")
    q4.metric("Outlier rule", "IQR · numeric")
    st.write("The source EDA applies IQR filtering repeatedly to every numeric field. This dashboard keeps all source rows by default because clinical extremes can be meaningful, and reports IQR flags without removing them.")
    numeric = raw.select_dtypes(include="number")
    iqr_rows = []
    for column in numeric:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = ((values < lower) | (values > upper)).sum()
        iqr_rows.append({"Numeric field": column, "IQR outlier rows": int(flagged), "Lower bound": round(lower, 2), "Upper bound": round(upper, 2)})
    if iqr_rows:
        st.dataframe(pd.DataFrame(iqr_rows), width="stretch", hide_index=True, alt="IQR outlier counts and bounds for numeric columns")
    st.write("Recorded alive share is the proportion labeled Alive in the selected records. Follow-up months are summarized descriptively and should not be interpreted as a causal effect or individual prognosis.")
    if not raw.empty:
        st.download_button("Download source CSV", raw.to_csv(index=False).encode("utf-8"),
                           file_name="breast_cancer_source.csv", mime="text/csv")
