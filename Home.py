# Home.py
import streamlit as st

st.set_page_config(
    page_title="Optimization Playground",
    layout="wide",
    page_icon="🧪",
)

st.title("🧪 Optimization Playground")
st.caption("Interactive lab for understanding optimization algorithms with real‑life stories.")

tab_overview, tab_algos, tab_how = st.tabs(["Overview", "Algorithms", "How to use"])

with tab_overview:
    st.subheader("Why this playground?")
    st.info(
        "Instead of focusing on formulas, this app focuses on **intuition**.\n"
        "Each page starts with a real‑life scenario, then lets you play with parameters."
    )
    st.markdown("---")
    st.subheader("What you can explore")
    st.markdown(
        "- Gradient‑based search on a 3D surface\n"
        "- Trade‑offs between conflicting project goals\n"
        "- Nature‑inspired search using genetic algorithms\n"
        "- Escaping local minima with simulated annealing"
    )

with tab_algos:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ⚙️ Surface Minimization")
        st.write(
            "Design a system by minimizing a loss surface with Steepest Descent, "
            "Newton, and Conjugate Gradient."
        )
        st.page_link("pages/1_Unconstrained_Minimization.py",
                     label="Open surface minimization →", icon="📉")

        st.markdown("### 🧬 Genetic Scheduling")
        st.write(
            "Use a Genetic Algorithm to build an efficient packing/scheduling plan."
        )
        st.page_link("pages/3_Genetic_Algorithm.py",
                     label="Open GA playground →", icon="🧬")

    with c2:
        st.markdown("### ⚖️ Project Trade‑offs")
        st.write(
            "Compare engineering project ideas based on **cost, impact, and risk** "
            "using Pareto fronts."
        )
        st.page_link("pages/2_Pareto_Front.py",
                     label="Open trade‑off explorer →", icon="📊")

        st.markdown("### 🌡️ Cooling‑based Search")
        st.write(
            "See how simulated annealing schedules tasks by gradually reducing randomness."
        )
        st.page_link("pages/4_Simulated_Annealing.py",
                     label="Open SA explorer →", icon="🌡️")

with tab_how:
    st.subheader("How to use this app in class")
    st.markdown(
        "1. Start with **Project Trade‑offs** to understand Pareto fronts.\n"
        "2. Then explore **Surface Minimization** to see gradient‑based methods.\n"
        "3. Use **GA** and **SA** to compare stochastic search strategies.\n"
        "4. Take screenshots and note your observations for your report."
    )
