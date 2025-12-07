import matplotlib.pyplot as plt
import numpy as np
import os

# ================= Data Input =================
# Fill in the results from evaluate_privacy.py here
# Experimental data
metrics = ['Privacy Leakage\n(Lower is Better)', 'Utility / Accuracy\n(Higher is Better)']

# Unsafe RAG results
unsafe_scores = [40.0, 61.5]  # [Leakage %, Accuracy %]

# Safe RAG results
safe_scores = [0.0, 65.4]     # [Leakage %, Accuracy %]

# ================= Plotting Configuration =================
def create_comparison_chart():
    # Set bar width
    bar_width = 0.35
    
    # Set X-axis positions
    x = np.arange(len(metrics))
    
    # Create canvas, size 10x6 inches, resolution 100 dpi
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Draw Unsafe RAG bars (using light red to indicate warning)
    rects1 = plt.bar(x - bar_width/2, unsafe_scores, bar_width, 
                     label='Unsafe RAG (Baseline)', color='#ff6b6b', edgecolor='black', alpha=0.9)
    
    # Draw Safe RAG bars (using emerald green to indicate safety)
    rects2 = plt.bar(x + bar_width/2, safe_scores, bar_width, 
                     label='Safe RAG (Ours)', color='#51cf66', edgecolor='black', alpha=0.9)

    # === Add decorations ===
    # Add title and labels
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Impact of PII Scrubbing: Privacy Risk vs. Utility', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x, metrics, fontsize=11)
    plt.ylim(0, 100)  # Y-axis fixed at 0-100%
    
    # Add legend
    plt.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Add grid lines (Y-axis only)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # === Core function: automatically label values on bars ===
    def autolabel(rects):
        """Attach text labels on top of each bar"""
        for rect in rects:
            height = rect.get_height()
            # If height is 0, slightly raise to display '0.0%'
            xy_pos = (rect.get_x() + rect.get_width() / 2, height)
            text_offset = 3 if height > 0 else 1
            
            plt.annotate(f'{height}%',
                        xy=xy_pos,
                        xytext=(0, text_offset),  # Vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # === Save and display ===
    output_path = "privacy_utility_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved successfully to: {output_path}")
    # If running on local computer with screen, uncomment the line below
    # plt.show()

if __name__ == "__main__":
    create_comparison_chart()