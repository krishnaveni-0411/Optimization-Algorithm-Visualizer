# Home.py
import streamlit as st

st.set_page_config(
    page_title="Optimization Playground",
    layout="wide",
    page_icon="🧪",
)

# ---- HEADER -------------------------------------------------------------
st.title("🧪 Optimization Playground")
st.caption("Interactive lab for understanding optimization algorithms with real‑life stories.")

st.markdown(
    """
<b>Created by:</b> <span style="font-size:16px; font-weight:600;">Krishnaveni O_o 2310040013</span>  
<i>“Turning optimization theory into playable experiments.”</i>
""",
    unsafe_allow_html=True,
)

st.markdown("")

# Little fun bar
c1, c2, c3 = st.columns(3)
c1.success("🎯 Focus: Intuition first")
c2.info("🧠 Algorithms: GD, GA, SA, Pareto")
c3.warning("⚙️ Tech: Python · Streamlit · Matplotlib")

st.markdown("---")

tab_overview, tab_algos, tab_how = st.tabs(["Overview", "Algorithms", "How to use"])

with tab_overview:
    st.subheader("Why this playground?")
    st.info(
        "Instead of focusing on formulas, this app focuses on **intuition**.\n"
        "Each page starts with a real‑life scenario, then lets you play with parameters."
    )

    st.markdown("### What you can explore")
    st.markdown(
        "- How gradient‑based methods move on a **loss surface**.\n"
        "- How to balance **conflicting project goals** using Pareto fronts.\n"
        "- How **Genetic Algorithms** evolve solutions over generations.\n"
        "- How **Simulated Annealing** escapes local minima using randomness."
    )

    st.markdown("### Tiny challenge for you 👀")
    st.markdown(
        "- Try to intentionally **break** an algorithm (choose weird parameters).\n"
        "- Then tune it back until it behaves nicely again."
    )

with tab_algos:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ⚙️ Surface Minimization")
        st.write(
            "Design a system by minimizing a loss surface with "
            "**Steepest Descent, Newton, and Conjugate Gradient**."
        )
        st.page_link(
            "pages/1_Unconstrained_Minimization.py",
            label="Open surface minimization →",
            icon="📉",
        )

        st.markdown("### 🧬 Genetic Scheduling")
        st.write(
            "Use a **Genetic Algorithm** to build an efficient "
            "packing / scheduling plan under constraints."
        )
        st.page_link(
            "pages/3_Genetic_Algorithm.py",
            label="Open GA playground →",
            icon="🧬",
        )

    with c2:
        st.markdown("### ⚖️ Project Trade‑offs")
        st.write(
            "Compare engineering project ideas based on **cost, impact, and risk** "
            "using **Pareto fronts**."
        )
        st.page_link(
            "pages/2_Pareto_Front.py",
            label="Open trade‑off explorer →",
            icon="📊",
        )

        st.markdown("### 🌡️ Cooling‑based Search")
        st.write(
            "Watch **Simulated Annealing** schedule tasks by gradually "
            "reducing randomness (cooling)."
        )
        st.page_link(
            "pages/4_Simulated_Annealing.py",
            label="Open SA explorer →",
            icon="🌡️",
        )

with tab_how:
    st.subheader("How to use this app in class")
    st.markdown(
        "1. Start with **Project Trade‑offs** to understand Pareto fronts.\n"
        "2. Then explore **Surface Minimization** to see gradient‑based methods.\n"
        "3. Use **GA** and **SA** to compare stochastic search strategies.\n"
        "4. Take screenshots and write 2–3 lines of observation per page."
    )

    st.markdown("---")
    st.caption("Tip: This app was built as a course mini‑project by Your Name (ECE).")
