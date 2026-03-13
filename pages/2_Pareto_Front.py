# pages/2_Pareto_Front.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Project Trade‑off Explorer", layout="wide")

st.title("⚖️ Project Trade‑off Explorer (Pareto Front)")
st.caption("Choose engineering projects based on cost, impact, and risk.")

# ---------- Scenario & theory tab ----------
tab_scenario, tab_play, tab_notes = st.tabs(
    ["Scenario & intuition", "Playground", "Notes"]
)

with tab_scenario:
    st.subheader("Real‑life scenario")
    st.write(
        "Imagine you have a fixed budget and several project ideas "
        "(IoT system, AI tool, robotics demo, etc.).\n"
        "Each project has **cost**, **expected impact**, and **risk**."
    )
    st.warning(
        "There is rarely a single 'best' project. "
        "Instead, there is a **frontier of best trade‑offs** (Pareto front)."
    )

    st.markdown("### What is a Pareto front?")
    st.markdown(
        "- A project A is **dominated** by project B if B is **no worse in any objective** "
        "and **better in at least one**.\n"
        "- The **Pareto front** is the set of all non‑dominated projects.\n"
        "- Moving along the front, you **gain** in one objective but **lose** in another."
    )

# ---------- Default dataset ----------
DEFAULT_DATA = {
    "Project": [
        "IoT Smart Garden",
        "Robotics Line Follower",
        "AI Resume Screener",
        "Smart Campus App",
        "FPGA Signal Analyzer",
        "Low‑Cost Drone",
        "Energy Monitoring Dashboard",
        "Voice‑Controlled Home",
        "Autonomous Cart",
        "Predictive Maintenance Model",
    ],
    # lower is better
    "Cost_kINR": [45, 15, 30, 20, 60, 55, 25, 35, 70, 40],
    # higher is better
    "Impact_Score": [80, 50, 85, 75, 70, 90, 65, 78, 88, 82],
    # lower is better
    "Risk_Score": [30, 20, 40, 25, 50, 45, 35, 30, 55, 38],
}

def find_pareto(costs: np.ndarray) -> np.ndarray:
    """
    costs: 2D array where each column is an objective to MINIMIZE.
    Returns boolean mask — True means Pareto optimal.
    """
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        dominated[i] = False
        is_efficient[dominated] = False
    return is_efficient

with tab_play:
    # ---------- Sidebar controls ----------
    with st.sidebar:
        st.header("Data source")
        source = st.radio(
            "Dataset",
            ["Use built‑in project data", "Upload CSV"],
        )
        uploaded = None
        if source == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV file", type=["csv"])
            st.caption("CSV should have at least 3 numeric columns.")

        st.markdown("---")
        st.header("Objectives")
        st.caption("Select two objectives and whether to minimize or maximize.")

    # ---------- Load data ----------
    if source == "Upload CSV" and uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} rows from uploaded file.")
    else:
        df = pd.DataFrame(DEFAULT_DATA)
        if source == "Upload CSV":
            st.info("No file uploaded — using built‑in project dataset.")

    st.subheader("Dataset preview")
    st.dataframe(df, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns in the dataset.")
        st.stop()

    with st.sidebar:
        obj1_col = st.selectbox("Objective 1 (X‑axis)", numeric_cols, index=0)
        obj1_dir = st.radio(
            "Direction 1", ["Minimize", "Maximize"], index=0, horizontal=True
        )

        obj2_col = st.selectbox(
            "Objective 2 (Y‑axis)",
            numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0,
        )
        obj2_dir = st.radio(
            "Direction 2", ["Minimize", "Maximize"], index=1, horizontal=True
        )

        label_col = st.selectbox(
            "Label column",
            ["(none)"] + text_cols,
        )

        color_by = st.selectbox(
            "Color points by (optional)",
            ["(none)"] + numeric_cols,
        )

        run_btn = st.button("Compute Pareto front", type="primary", use_container_width=True)

    st.markdown("---")

    if run_btn:
        v1 = df[obj1_col].values.astype(float)
        v2 = df[obj2_col].values.astype(float)

        # convert to "minimize" space
        c1 = v1 if obj1_dir == "Minimize" else -v1
        c2 = v2 if obj2_dir == "Minimize" else -v2
        costs = np.column_stack([c1, c2])

        mask = find_pareto(costs)
        df["Pareto"] = mask

        pareto_df = df[mask].copy()
        dominated_df = df[~mask].copy()

        # Top metrics
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("Total projects", len(df))
        c2m.metric("Pareto‑optimal projects", int(mask.sum()))
        c3m.metric("Dominated projects", int((~mask).sum()))

        # Two sub‑tabs for visuals
        t_scatter, t_table = st.tabs(["Scatter plot", "Pareto table"])

        with t_scatter:
            st.subheader("Pareto front visualization")

            fig, ax = plt.subplots(figsize=(10, 6))

            # Optional color dimension
            if color_by != "(none)":
                cmap = plt.get_cmap("viridis")
                norm_vals = df[color_by].values.astype(float)
                norm = (norm_vals - norm_vals.min()) / (norm_vals.ptp() + 1e-9)
                colors_all = cmap(norm)
            else:
                colors_all = ["lightgrey"] * len(df)

            # Plot dominated
            ax.scatter(
                dominated_df[obj1_col],
                dominated_df[obj2_col],
                c="lightgrey",
                edgecolors="grey",
                s=70,
                alpha=0.6,
                label="Dominated",
                zorder=1,
            )

            # Plot Pareto
            ax.scatter(
                pareto_df[obj1_col],
                pareto_df[obj2_col],
                c="tab:red",
                edgecolors="black",
                s=120,
                alpha=0.9,
                label="Pareto‑optimal",
                zorder=3,
            )

            # Draw front line (sorted by x)
            pf_sorted = pareto_df.sort_values(by=obj1_col)
            ax.plot(
                pf_sorted[obj1_col],
                pf_sorted[obj2_col],
                "r--",
                linewidth=1.5,
                alpha=0.7,
                zorder=2,
            )

            # Labels on points
            if label_col != "(none)":
                for _, row in pareto_df.iterrows():
                    ax.annotate(
                        str(row[label_col]),
                        (row[obj1_col], row[obj2_col]),
                        xytext=(5, 4),
                        textcoords="offset points",
                        fontsize=8,
                        color="darkred",
                    )

            ax.set_xlabel(f"{obj1_col} [{obj1_dir}]", fontsize=11)
            ax.set_ylabel(f"{obj2_col} [{obj2_dir}]", fontsize=11)
            ax.set_title("Project trade‑offs Pareto front", fontsize=13)
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.legend()

            st.pyplot(fig, use_container_width=True)
            plt.close()

        with t_table:
            st.subheader("Pareto‑optimal project list")
            show_cols = (
                ([label_col] if label_col != "(none)" else []) + [obj1_col, obj2_col]
            )
            st.dataframe(
                pareto_df[show_cols].reset_index(drop=True),
                use_container_width=True,
            )

            # Download only Pareto set
            buf = io.StringIO()
            pareto_df.to_csv(buf, index=False)
            st.download_button(
                "Download Pareto‑optimal projects as CSV",
                data=buf.getvalue(),
                file_name="pareto_projects.csv",
                mime="text/csv",
            )
    else:
        st.info("Set objectives in the sidebar and click **Compute Pareto front**.")

with tab_notes:
    st.subheader("Questions to explore")
    st.markdown(
        "- What changes when you **swap** X and Y objectives?\n"
        "- If you treat **risk as minimize** vs **maximize**, how does the front move?\n"
        "- Which projects would you pick if budget is very strict vs flexible?"
    )
