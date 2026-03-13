# pages/4_Simulated_Annealing.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

st.set_page_config(page_title="SA Exam Scheduler", layout="wide")

st.title("🌡️ Simulated Annealing Exam Scheduler")
st.caption("Use temperature to escape bad timetables and reduce clashes.")

tab_scenario, tab_play, tab_notes = st.tabs(
    ["Scenario & intuition", "Playground", "Notes"]
)

with tab_scenario:
    st.subheader("Real‑life scenario")
    st.write(
        "You need to schedule several exams into a limited number of time slots.\n"
        "Students are enrolled in different subsets of exams; if two of their exams "
        "land in the **same slot**, that's a clash."
    )
    st.info(
        "Simulated Annealing starts with a random timetable, then keeps making small "
        "changes. At high temperature, it sometimes accepts worse timetables to escape "
        "local minima; as it cools, it becomes more strict."
    )
    st.markdown("### Intuition")
    st.markdown(
        "- Treat the **number of clashes** as an energy/cost.\n"
        "- Early (high T): explore a lot, accept bad moves.\n"
        "- Late (low T): exploit, accept only improvements or tiny bad moves."
    )

# ---------- Problem data ----------
EXAMS = [
    "Mathematics",
    "Physics",
    "Chemistry",
    "English",
    "History",
    "Computer Science",
    "Economics",
    "Biology",
    "Statistics",
    "Geography",
]
STUDENTS = [
    [0, 1, 5],
    [0, 2, 6],
    [1, 3, 7],
    [2, 4, 8],
    [3, 5, 9],
    [0, 4, 7],
    [1, 6, 8],
    [2, 5, 9],
    [3, 6, 0],
    [4, 7, 1],
    [5, 8, 2],
    [6, 9, 3],
    [7, 0, 4],
    [8, 1, 5],
    [9, 2, 6],
    [0, 3, 8],
    [1, 4, 9],
    [2, 7, 5],
    [3, 8, 6],
    [4, 9, 7],
    [0, 5, 2],
    [1, 6, 3],
    [2, 7, 4],
    [3, 8, 0],
    [4, 9, 1],
    [5, 0, 6],
    [6, 1, 7],
    [7, 2, 8],
    [8, 3, 9],
    [9, 4, 0],
]
NUM_EXAMS = len(EXAMS)

# ---------- Playground ----------
with tab_play:
    with st.sidebar:
        st.header("Timetable settings")
        num_slots = st.slider("Number of time slots", 3, 8, 5)

        st.markdown("---")
        st.header("SA parameters")
        init_temp = st.slider("Initial temperature", 10.0, 500.0, 120.0, 10.0)
        cooling_rate = st.slider(
            "Cooling rate", 0.80, 0.999, 0.995, 0.001, format="%.3f"
        )
        min_temp = st.select_slider(
            "Min temperature", options=[0.001, 0.01, 0.1, 1.0], value=0.1
        )
        max_iter = st.slider("Max iterations", 500, 10000, 5000, 500)
        seed = st.number_input("Random seed", value=42, step=1)

        st.markdown("---")
        show_initial = st.checkbox("Show initial random timetable", value=True)

        run_btn = st.button("Run Simulated Annealing", type="primary", use_container_width=True)

    # ---------- SA core ----------
    def count_clashes(tt, n_slots):
        clashes = 0
        for se in STUDENTS:
            seen = set()
            for e in se:
                s = tt[e]
                if s in seen:
                    clashes += 1
                seen.add(s)
        return clashes

    def gen_neighbor(tt, n_slots):
        new = tt[:]
        e = random.randint(0, NUM_EXAMS - 1)
        cur = tt[e]
        new[e] = random.choice([s for s in range(n_slots) if s != cur])
        return new

    def run_sa(n_slots, init_t, cool, min_t, max_it, seed_val):
        random.seed(seed_val)
        curr = [random.randint(0, n_slots - 1) for _ in range(NUM_EXAMS)]
        curr_c = count_clashes(curr, n_slots)
        best = curr[:]
        best_c = curr_c

        T = init_t
        clash_log, temp_log, accept_log = [], [], []
        accepts = 0

        initial = curr[:]

        for it in range(max_it):
            if T < min_t:
                break
            nbr = gen_neighbor(curr, n_slots)
            nbr_c = count_clashes(nbr, n_slots)
            delta = nbr_c - curr_c

            accepted = False
            if delta < 0 or random.random() < math.exp(-delta / T):
                curr, curr_c = nbr, nbr_c
                accepted = True
                accepts += 1

            if curr_c < best_c:
                best, best_c = curr[:], curr_c

            clash_log.append(best_c)
            temp_log.append(T)
            accept_log.append(accepts / (it + 1))
            T *= cool

            if best_c == 0:
                break

        return initial, best, best_c, clash_log, temp_log, accept_log

    if run_btn:
        with st.spinner("Running Simulated Annealing..."):
            (
                initial_tt,
                best_tt,
                best_c,
                clash_log,
                temp_log,
                accept_log,
            ) = run_sa(
                num_slots, init_temp, cooling_rate, min_temp, max_iter, int(seed)
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final clashes", str(best_c))
        c2.metric("Iterations used", str(len(clash_log)))
        c3.metric("Starting clashes", str(clash_log[0] if clash_log else "?"))
        c4.metric("Solved?", "Yes ✅" if best_c == 0 else "No ❌")

        st.subheader("Convergence behaviour")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(clash_log, color="crimson", linewidth=1.5)
        axes[0].set_title("Best clashes over iterations")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Best clashes")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(temp_log, color="steelblue", linewidth=1.5)
        axes[1].set_title("Temperature schedule")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Temperature")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(accept_log, color="darkorange", linewidth=1.5)
        axes[2].set_title("Cumulative acceptance rate")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Acceptance rate")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.subheader("Timetables")

        def build_slot_df(tt, n_slots):
            slot_contents = {s: [] for s in range(n_slots)}
            for exam_i, slot in enumerate(tt):
                slot_contents[slot].append(EXAMS[exam_i])
            return {
                f"Slot {s+1}": ", ".join(slot_contents[s]) or "(empty)"
                for s in range(n_slots)
            }

        col_init, col_best = st.columns(2)

        if show_initial:
            with col_init:
                st.markdown("##### Initial random timetable")
                slot_df_init = build_slot_df(initial_tt, num_slots)
                df_init = pd.DataFrame(
                    list(slot_df_init.items()), columns=["Time Slot", "Exams"]
                )
                st.dataframe(df_init, use_container_width=True, hide_index=True)

        with col_best:
            st.markdown("##### Final timetable after SA")
            slot_df_best = build_slot_df(best_tt, num_slots)
            df_best = pd.DataFrame(
                list(slot_df_best.items()), columns=["Time Slot", "Exams"]
            )
            st.dataframe(df_best, use_container_width=True, hide_index=True)

        st.subheader("Heatmap of exam assignments")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        grid = np.zeros((num_slots, NUM_EXAMS))
        for exam_i, slot in enumerate(best_tt):
            grid[slot, exam_i] = 1
        im = ax2.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax2.set_xticks(range(NUM_EXAMS))
        ax2.set_xticklabels([e[:4] for e in EXAMS], rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(num_slots))
        ax2.set_yticklabels([f"Slot {s+1}" for s in range(num_slots)])
        ax2.set_title("Final timetable heatmap (orange = assigned)")
        plt.colorbar(im, ax=ax2, shrink=0.6)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        with st.expander("Clash analysis per student"):
            clash_rows = []
            for si, se in enumerate(STUDENTS):
                seen = {}
                for e in se:
                    s = best_tt[e]
                    seen[s] = seen.get(s, 0) + 1
                student_clashes = sum(v - 1 for v in seen.values() if v > 1)
                clash_rows.append(
                    {
                        "Student": si + 1,
                        "Exams": ", ".join(EXAMS[e] for e in se),
                        "Slots assigned": ", ".join(str(best_tt[e] + 1) for e in se),
                        "Clashes": student_clashes,
                    }
                )
            cdf = pd.DataFrame(clash_rows)
            st.dataframe(cdf, use_container_width=True, hide_index=True)
    else:
        st.info("Set slots and SA parameters, then click **Run Simulated Annealing**.")

with tab_notes:
    st.subheader("Things to try")
    st.markdown(
        "- Use **cooling_rate = 0.80** (very aggressive cooling) and compare with 0.995.\n"
        "- Try higher `init_temp` and see if it explores more.\n"
        "- Reduce `num_slots` and watch clashes become hard to remove.\n"
        "- Observe how acceptance rate changes as temperature drops."
    )
    st.caption("SA is basically: controlled randomness that becomes more serious over time.")
