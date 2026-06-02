"""
Test script to compare different activation functions with ANN backpropagation
"""

import numpy as np
import matplotlib.pyplot as plt
from deep_learning.ann_backpropagation import (
    NeuralNetwork, 
    create_xor_dataset, 
    create_circles_dataset,
    create_linearly_separable_dataset
)


def test_activation_functions_on_dataset(dataset_name: str, X_train, y_train, X_test, y_test):
    """
    Train networks with different activation functions and compare results
    """
    print(f"\n{'='*80}")
    print(f"Testing on {dataset_name} dataset")
    print(f"{'='*80}")
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    activations = ['sigmoid', 'tanh', 'relu']
    results = {}
    
    for activation in activations:
        print(f"\nTraining with {activation.upper()} activation function...")
        
        # Create network with hidden layer of size 8
        network = NeuralNetwork(
            layer_sizes=[X_train.shape[1], 8, 1],
            activation=activation,
            learning_rate=0.5,
            momentum=0.0
        )
        
        # Train network
        network.train(X_train, y_train, epochs=2000, batch_size=len(X_train), verbose=False)
        
        # Evaluate
        train_loss, train_accuracy = network.evaluate(X_train, y_train)
        test_loss, test_accuracy = network.evaluate(X_test, y_test)
        
        results[activation] = {
            'network': network,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'loss_history': network.loss_history,
            'accuracy_history': network.accuracy_history
        }
        
        print(f"  Training Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.4f}")
        print(f"  Test Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.4f}")
    
    return results


def plot_comparison_results(dataset_name: str, results: dict):
    """
    Plot comparison of activation functions
    """
    activations = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'ANN Backpropagation: Activation Function Comparison ({dataset_name})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for activation in activations:
        ax.plot(results[activation]['loss_history'], label=activation.upper(), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    ax = axes[0, 1]
    for activation in activations:
        ax.plot(results[activation]['accuracy_history'], label=activation.upper(), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Performance Comparison (Bar chart)
    ax = axes[1, 0]
    train_accs = [results[act]['train_accuracy'] for act in activations]
    test_accs = [results[act]['test_accuracy'] for act in activations]
    
    x_pos = np.arange(len(activations))
    width = 0.35
    
    ax.bar(x_pos - width/2, train_accs, width, label='Train', alpha=0.8)
    ax.bar(x_pos + width/2, test_accs, width, label='Test', alpha=0.8)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Accuracy Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
        ax.text(i - width/2, train_acc + 0.02, f'{train_acc:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, test_acc + 0.02, f'{test_acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Test Loss Comparison
    ax = axes[1, 1]
    train_losses = [results[act]['train_loss'] for act in activations]
    test_losses = [results[act]['test_loss'] for act in activations]
    
    ax.bar(x_pos - width/2, train_losses, width, label='Train', alpha=0.8)
    ax.bar(x_pos + width/2, test_losses, width, label='Test', alpha=0.8)
    
    ax.set_ylabel('Loss')
    ax.set_title('Final Loss Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
        ax.text(i - width/2, train_loss + 0.01, f'{train_loss:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, test_loss + 0.01, f'{test_loss:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_decision_boundaries(dataset_name: str, results: dict, X_train, y_train):
    """
    Plot decision boundaries for each activation function
    """
    activations = list(results.keys())
    
    fig, axes = plt.subplots(1, len(activations), figsize=(15, 4))
    fig.suptitle(f'Decision Boundaries for {dataset_name} Dataset', fontsize=14, fontweight='bold')
    
    # Create a mesh to plot decision boundaries
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    for idx, activation in enumerate(activations):
        ax = axes[idx] if len(activations) > 1 else axes
        
        # Get predictions on mesh
        Z = results[activation]['network'].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 3), cmap='RdBu', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')
        
        # Plot training points
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), 
                           cmap='RdBu', edgecolors='black', s=50, alpha=0.8)
        
        accuracy = results[activation]['test_accuracy']
        ax.set_title(f'{activation.upper()}\n(Acc: {accuracy:.3f})')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
    
    plt.tight_layout()
    return fig


def print_detailed_results(dataset_name: str, results: dict):
    """
    Print detailed results for each activation function
    """
    print(f"\n{'-'*80}")
    print(f"DETAILED RESULTS FOR {dataset_name.upper()}")
    print(f"{'-'*80}")
    
    activations = list(results.keys())
    
    for activation in activations:
        res = results[activation]
        print(f"\n{activation.upper()} Activation Function:")
        print(f"  Training Loss: {res['train_loss']:.6f}")
        print(f"  Training Accuracy: {res['train_accuracy']:.4f}")
        print(f"  Test Loss: {res['test_loss']:.6f}")
        print(f"  Test Accuracy: {res['test_accuracy']:.4f}")
        print(f"  Final Epoch Loss: {res['loss_history'][-1]:.6f}")


def main():
    """
    Main test function - compare activation functions on multiple datasets
    """
    print("="*80)
    print("ANN with BACKPROPAGATION: Activation Function Comparison")
    print("="*80)
    
    datasets = {
        'XOR': create_xor_dataset(),
        'Linearly Separable': create_linearly_separable_dataset(n_samples=200),
        'Concentric Circles': create_circles_dataset(n_samples=200)
    }
    
    all_results = {}
    
    for dataset_name, (X_train, y_train, X_test, y_test) in datasets.items():
        # Normalize data
        X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
        X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-8)
        
        # Test and compare activation functions
        results = test_activation_functions_on_dataset(dataset_name, X_train, y_train, X_test, y_test)
        all_results[dataset_name] = results
        
        # Print detailed results
        print_detailed_results(dataset_name, results)
        
        # Plot results
        fig_comparison = plot_comparison_results(dataset_name, results)
        fig_comparison.savefig(f'/tmp/workspace/98124kiran/project1/deep_learning/comparison_{dataset_name.replace(" ", "_").lower()}.png', 
                              dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison plot: comparison_{dataset_name.replace(' ', '_').lower()}.png")
        
        # Plot decision boundaries
        fig_boundaries = plot_decision_boundaries(dataset_name, results, X_train, y_train)
        fig_boundaries.savefig(f'/tmp/workspace/98124kiran/project1/deep_learning/boundaries_{dataset_name.replace(" ", "_").lower()}.png', 
                              dpi=150, bbox_inches='tight')
        print(f"Saved decision boundaries plot: boundaries_{dataset_name.replace(' ', '_').lower()}.png")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        best_activation = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_accuracy = results[best_activation]['test_accuracy']
        print(f"  Best performing activation: {best_activation.upper()} with accuracy {best_accuracy:.4f}")
        
        for activation in results.keys():
            acc = results[activation]['test_accuracy']
            print(f"  {activation.upper()}: {acc:.4f}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS CONCLUSIONS")
    print(f"{'='*80}")
    print("""
1. SIGMOID Activation:
   - Smooth, differentiable activation function
   - Output range: (0, 1)
   - Prone to vanishing gradient problem in deep networks
   - Works well for binary classification
   - Generally good for shallow networks

2. TANH Activation:
   - Similar to sigmoid but output range: (-1, 1)
   - Zero-centered, which helps with convergence
   - Still prone to vanishing gradients in deep networks
   - Often performs better than sigmoid due to zero-centering

3. ReLU Activation:
   - Non-linear, computationally efficient
   - Output range: [0, ∞)
   - Mitigates vanishing gradient problem
   - Can suffer from "dying ReLU" problem (neurons outputting 0)
   - Preferred in deep networks

Expected Performance:
   - For XOR: Sigmoid and Tanh should perform equally well
   - For Linearly Separable: All activations should perform well
   - For Non-linear (Circles): ReLU and Tanh may perform better
    """)
    
    plt.show()


if __name__ == '__main__':
    main()
