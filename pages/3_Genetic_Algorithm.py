# pages/3_Genetic_Algorithm.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Genetic Task Allocator", layout="wide")

st.title("🧬 Genetic Task Allocator")
st.caption("Use a Genetic Algorithm to pick tasks under a time budget.")

tab_scenario, tab_play, tab_notes = st.tabs(
    ["Scenario & intuition", "Playground", "Notes"]
)

with tab_scenario:
    st.subheader("Real‑life scenario")
    st.write(
        "You are planning a project sprint with a list of possible tasks.\n"
        "Each task has **effort (hours)** and **value (impact)**, and you have a "
        "limited total time budget."
    )
    st.info(
        "This is a 0/1 knapsack‑style problem: either you **include** a task or you don't. "
        "The Genetic Algorithm will evolve a good combination for you."
    )
    st.markdown("### GA idea in one minute")
    st.markdown(
        "1. Represent a plan as a **binary chromosome** (1 = do task, 0 = skip).\n"
        "2. Initialize a **population** of random plans.\n"
        "3. Compute **fitness** as total value if within time, else 0.\n"
        "4. Repeatedly do **selection → crossover → mutation**.\n"
        "5. Keep the best plan seen so far (elitism)."
    )

# ---------- Default tasks ----------
DEFAULT_TASKS = [
    ("Write project proposal", 3.0, 9),
    ("Create system architecture", 4.0, 10),
    ("Set up Git & CI/CD", 2.0, 8),
    ("Build basic frontend", 5.0, 9),
    ("Design database schema", 3.0, 8),
    ("Implement backend API", 4.5, 10),
    ("Integrate IoT devices", 3.5, 9),
    ("Create test cases", 2.5, 7),
    ("Write documentation", 2.0, 6),
    ("Prepare demo slides", 1.5, 6),
    ("Record demo video", 2.0, 7),
    ("Performance optimization", 3.0, 8),
]

# ---------- Playground ----------
with tab_play:
    with st.sidebar:
        st.header("Sprint constraints")
        max_hours = st.slider("Max total hours", 5.0, 40.0, 18.0, 0.5)

        st.markdown("---")
        st.header("GA parameters")
        pop_size = st.slider("Population size", 10, 150, 40)
        n_generations = st.slider("Generations", 10, 300, 80)
        mutation_rate = st.slider("Mutation rate", 0.01, 0.50, 0.08, 0.01)
        crossover_rate = st.slider("Crossover rate", 0.5, 1.0, 0.85, 0.05)
        tournament_k = st.slider("Tournament size", 2, 10, 3)
        seed = st.number_input("Random seed", value=42, step=1)

        st.markdown("---")
        show_chrom = st.checkbox("Show best chromosome heatmap", value=True)
        run_btn = st.button("Run Genetic Algorithm", type="primary", use_container_width=True)

    # ---------- GA core ----------
    def fitness(chrom, hours, values, max_h):
        total_h = sum(hours[i] for i in range(len(chrom)) if chrom[i])
        total_v = sum(values[i] for i in range(len(chrom)) if chrom[i])
        return total_v if total_h <= max_h else 0

    def tournament(pop, fits, k):
        idx = random.sample(range(len(pop)), k)
        return pop[max(idx, key=lambda i: fits[i])][:]

    def crossover(p1, p2, rate):
        if random.random() > rate:
            return p1[:]
        cut = random.randint(1, len(p1) - 1)
        return p1[:cut] + p2[cut:]

    def mutate(chrom, rate):
        return [1 - g if random.random() < rate else g for g in chrom]

    def run_ga(items, max_h, pop_size, n_gen, mut_rate, cx_rate, tourn_k, seed_val):
        random.seed(seed_val)
        names = [it[0] for it in items]
        hours = [it[1] for it in items]
        values = [it[2] for it in items]
        n = len(items)

        pop = [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]
        fit_fn = lambda c: fitness(c, hours, values, max_h)

        best_val_log, avg_val_log, diversity_log = [], [], []
        best_chrom_ever, best_val_ever = None, -1

        for _ in range(n_gen):
            fits = [fit_fn(c) for c in pop]
            best_i = max(range(pop_size), key=lambda i: fits[i])
            if fits[best_i] > best_val_ever:
                best_val_ever = fits[best_i]
                best_chrom_ever = pop[best_i][:]

            # logs
            valid_fits = [f for f in fits if f > 0]
            best_val_log.append(best_val_ever)
            avg_val_log.append(np.mean(valid_fits) if valid_fits else 0)
            diversity_log.append(len(set(tuple(c) for c in pop)) / pop_size)

            # next generation
            next_pop = [pop[best_i][:]]  # elitism
            while len(next_pop) < pop_size:
                p1 = tournament(pop, fits, tourn_k)
                p2 = tournament(pop, fits, tourn_k)
                ch = crossover(p1, p2, cx_rate)
                ch = mutate(ch, mut_rate)
                next_pop.append(ch)
            pop = next_pop

        return (
            best_chrom_ever,
            best_val_ever,
            best_val_log,
            avg_val_log,
            diversity_log,
            hours,
            values,
            names,
        )

    tasks_data = DEFAULT_TASKS

    if run_btn:
        with st.spinner("Running Genetic Algorithm..."):
            (
                best_chrom,
                best_val,
                bv_log,
                av_log,
                div_log,
                hours,
                values,
                names,
            ) = run_ga(
                tasks_data,
                max_hours,
                pop_size,
                n_generations,
                mutation_rate,
                crossover_rate,
                tournament_k,
                int(seed),
            )
        st.success("✅ GA run finished for current parameters. Check the playground")

        total_h = sum(hours[i] for i in range(len(best_chrom)) if best_chrom[i])
        valid = total_h <= max_hours

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best total value", str(best_val))
        c2.metric("Total hours", f"{total_h:.1f} h")
        c3.metric("Valid plan?", "Yes ✅" if valid else "No ❌ (over limit)")
        c4.metric("Tasks selected", str(sum(best_chrom)))

        st.subheader("Convergence")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(bv_log, color="seagreen", linewidth=2)
        axes[0].plot(av_log, color="steelblue", linewidth=1.5, linestyle="--", label="Avg (valid)")
        axes[0].set_title("Best vs Avg fitness")
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Fitness (total value)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(
            range(len(bv_log)),
            [bv_log[i] - (bv_log[i - 1] if i > 0 else 0) for i in range(len(bv_log))],
            color="coral",
            alpha=0.7,
        )
        axes[1].set_title("Improvement per generation")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Δ value")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(div_log, color="purple", linewidth=2)
        axes[2].set_title("Population diversity")
        axes[2].set_xlabel("Generation")
        axes[2].set_ylabel("Unique chromosomes / pop size")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.subheader("Selected sprint plan")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            rows = [
                {
                    "Task": names[i],
                    "Hours": hours[i],
                    "Value": values[i],
                    "Selected": "✅" if best_chrom[i] else "❌",
                }
                for i in range(len(best_chrom))
            ]
            st.dataframe(rows, use_container_width=True)

        with col_right:
            chosen_names = [names[i] for i in range(len(best_chrom)) if best_chrom[i]]
            chosen_vals = [values[i] for i in range(len(best_chrom)) if best_chrom[i]]
            if chosen_names:
                fig2, ax2 = plt.subplots(figsize=(6, len(chosen_names) * 0.5 + 1))
                bars = ax2.barh(chosen_names, chosen_vals, color="seagreen", edgecolor="black")
                ax2.bar_label(bars, padding=3)
                ax2.set_xlabel("Value (impact)")
                ax2.set_title(f"Selected tasks (total value = {best_val})")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

        if show_chrom:
            st.subheader("Best chromosome view")
            fig3, ax3 = plt.subplots(figsize=(10, 1.4))
            ax3.imshow([best_chrom], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax3.set_xticks(range(len(names)))
            ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax3.set_yticks([])
            ax3.set_title("Green = selected task, red = skipped")
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close()

    else:
        st.info("Set the GA parameters and click **Run Genetic Algorithm**.")

with tab_notes:
    st.subheader("Things to experiment with")
    st.markdown(
        "- Lower **mutation rate** too much → GA may **get stuck** (low diversity).\n"
        "- Very high mutation → behaves almost like random search.\n"
        "- Small population vs large population trade‑off.\n"
        "- Try increasing `max_hours` and see which tasks join the plan."
    )
    st.caption("Remember: GA does not guarantee the absolute best, but often finds a very good plan.")
