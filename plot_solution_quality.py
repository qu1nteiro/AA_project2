import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# // ===========================================================================
# // CONFIGURATION
# // ===========================================================================
DATA_FILE = "../results/final_results.csv"
OUTPUT_DIR = "../results/"
TARGET_DENSITY_TAG = "p50,0"

# // Limit X-axis as requested
MAX_N = 25

# // VISUAL OFFSETS (Jitter)
# // This shifts points slightly left/right so they don't overlap perfectly
OFFSETS = {
    'Exhaustive (P1)': 0.0,  # Center
    'Greedy (P1)': -0.2,  # Left
    'Pure Random (P2)': -0.1,  # Slight Left
    'Rand Greedy (P2)': 0.1,  # Slight Right
    'Sim Annealing (P2)': 0.2  # Right
}

# // COLORS & MARKERS
STYLES = {
    'Exhaustive (P1)': {'c': 'black', 'm': 's', 'lw': 3, 'alpha': 0.3},  # Square, Thick, Transparent
    'Greedy (P1)': {'c': 'gray', 'm': 'x', 'lw': 1, 'alpha': 0.8},
    'Pure Random (P2)': {'c': 'green', 'm': 'o', 'lw': 1, 'alpha': 0.8},
    'Rand Greedy (P2)': {'c': 'firebrick', 'm': '^', 'lw': 1, 'alpha': 0.8},  # Triangle Up
    'Sim Annealing (P2)': {'c': 'orange', 'm': 'v', 'lw': 1, 'alpha': 0.8}  # Triangle Down
}


# // ===========================================================================
# // MAIN
# // ===========================================================================
def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    print(">> Loading and filtering data...")
    df = pd.read_csv(DATA_FILE)

    # // 1. FILTER DATA
    # // Density 50% AND N <= 25
    mask = df['Graph'].str.contains(TARGET_DENSITY_TAG) & (df['N'] <= MAX_N)
    df_plot = df[mask].copy()

    if df_plot.empty:
        print("Error: No data found for the specified criteria.")
        return

    # // 2. PLOTTING
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # // Get list of algorithms present in data
    available_algs = df_plot['Algorithm'].unique()

    # // Order of plotting: Exhaustive first (background), others later
    sort_order = ['Exhaustive (P1)', 'Greedy (P1)', 'Pure Random (P2)', 'Rand Greedy (P2)', 'Sim Annealing (P2)']
    # // Filter sort_order to include only what we actually have
    plot_sequence = [alg for alg in sort_order if alg in available_algs]

    print(">> Plotting with Jitter (Offsets)...")

    for alg in plot_sequence:
        subset = df_plot[df_plot['Algorithm'] == alg].sort_values(by='N')

        if subset.empty: continue

        # // Apply Jitter to X coordinates
        offset = OFFSETS.get(alg, 0)
        jittered_x = subset['N'] + offset

        style = STYLES.get(alg, {'c': 'blue', 'm': 'o', 'lw': 1, 'alpha': 1})

        # // Plot Lines (Low alpha)
        plt.plot(jittered_x, subset['Best_Weight'],
                 linestyle='-', linewidth=style['lw'], color=style['c'], alpha=0.4)  # Faint connecting lines

        # // Plot Markers (High alpha) - This is what matters
        plt.scatter(jittered_x, subset['Best_Weight'],
                    label=alg,
                    color=style['c'], marker=style['m'], s=60, alpha=style['alpha'], edgecolors='white')

    # // 3. STYLING THE CHART
    plt.xlabel('Graph Size (N)', fontweight='bold', fontsize=12)
    plt.ylabel('Best Weight Found (Minimization)', fontweight='bold', fontsize=12)
    plt.title(f'Solution Quality Comparison (Density 50%, N <= {MAX_N})', fontsize=14)

    # // Force Integer Ticks on X
    plt.xticks(range(4, MAX_N + 1))

    plt.legend(frameon=True, shadow=True, fancybox=True, loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()

    # // Save
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, "solution_quality_jittered.png")

    plt.savefig(out_path, dpi=300)
    print(f">> Success! Chart saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()