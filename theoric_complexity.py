import matplotlib.pyplot as plt
import numpy as np

# // Setup plot style for academic publication (clean white background)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'lines.linewidth': 2.5
})


def main():
    """
    Main execution block: Generates complexity data and renders the comparison plot.
    """

    # // -----------------------------------------------------------------------
    # // 1. DEFINE DOMAIN (Input Size)
    # // -----------------------------------------------------------------------
    # // We simulate the graph size 'n' growing from 10 up to 1000 vertices.
    n: np.ndarray = np.linspace(10, 1000, 100)

    # // To make the plot realistic, we assume 'm' (edges) scales with 'n'.
    # // Let's assume an average degree of 10 (sparse-to-medium density).
    AVG_DEGREE: int = 10
    m: np.ndarray = AVG_DEGREE * n

    # // -----------------------------------------------------------------------
    # // 2. COMPLEXITY MODELS (The Math)
    # // -----------------------------------------------------------------------

    # // A. Pure Random Strategy
    # // Complexity: O(n + m). It's just a linear scan and shuffle.
    # // Very cheap per iteration.
    y_random: np.ndarray = (n + m)

    # // B. Simulated Annealing (SA)
    # // Complexity: O(L * (n + m)).
    # // 'L' represents the cooling steps (iterations per temp * number of temp steps).
    # // It's still linear, just scaled up by a constant factor.
    L_FACTOR: int = 200
    y_sa: np.ndarray = L_FACTOR * (n + m)

    # // C. Randomized Greedy
    # // Complexity: O(n*m + n^2 log n).
    # // This is the heavy one. We pay for evaluating edges (n*m) AND sorting candidates (n log n)
    # // repeatedly for every node added to the solution (approx n times).
    y_greedy: np.ndarray = (n * m) + (n ** 2 * np.log2(n))

    # // -----------------------------------------------------------------------
    # // 3. RENDERING (Plotting)
    # // -----------------------------------------------------------------------
    plt.figure(figsize=(9, 6))

    # // Plot the curves
    plt.plot(n, y_random,
             label=r'Pure Random: $\mathcal{O}(n + m)$',
             color='green', linestyle=':')

    plt.plot(n, y_sa,
             label=r'Simulated Annealing: $\mathcal{O}(L \cdot (n + m))$',
             color='orange', linestyle='--')

    plt.plot(n, y_greedy,
             label=r'Randomized Greedy: $\mathcal{O}(nm + n^2 \log n)$',
             color='firebrick')

    # // Labels and Ticks
    plt.title('Theoretical Complexity Comparison', fontsize=14, pad=15)
    plt.xlabel('Graph Size ($n$ vertices)', fontweight='bold')
    plt.ylabel('Operational Cost (Log Scale)', fontweight='bold')

    # // CRITICAL: Use Log Scale on Y-axis.
    # // Without this, the Greedy line would dwarf the others, making them look like zero.
    plt.yscale('log')

    # // Final touches
    plt.legend(frameon=True, fancybox=True, framealpha=1, shadow=True, loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    # // Save to disk
    output_filename = 'theoretical_complexity.png'
    plt.savefig(output_filename, dpi=300)
    print(f">> Plot generated successfully: {output_filename}")

    plt.show()


# // Entry Point
if __name__ == "__main__":
    main()