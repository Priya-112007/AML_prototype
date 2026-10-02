import matplotlib.pyplot as plt
import numpy as np
import os

def create_accuracy_comparison():
    models = ['Baseline (Classical)', 'Hybrid Ensemble (CNN + Classical)']
    accuracies = [26.21, 52.43]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['#ff9999', '#66b3ff'])
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontweight='bold')

    plt.title('Model Accuracy Comparison on IDRiD Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated results/accuracy_comparison.png")

def create_class_distribution():
    classes = ['0: No DR', '1: Mild', '2: Moderate', '3: Severe', '4: Proliferative']
    # Approximate original counts from IDRiD train set
    original_counts = [134, 20, 136, 74, 49]
    # After SMOTE, all classes are equal to the majority class (136)
    smote_counts = [136, 136, 136, 136, 136]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, original_counts, width, label='Original (Imbalanced)', color='#ffcc99')
    plt.bar(x + width/2, smote_counts, width, label='After SMOTE (Balanced)', color='#99ff99')

    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Training Set Class Distribution: Original vs SMOTE', fontsize=14, fontweight='bold')
    plt.xticks(x, classes)
    plt.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated results/class_distribution.png")

if __name__ == "__main__":
    create_accuracy_comparison()
    create_class_distribution()
