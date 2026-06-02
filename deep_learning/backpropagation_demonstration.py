"""
Educational Script: Step-by-Step Backpropagation Algorithm
Demonstrates the mathematical steps involved in backpropagation
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_backpropagation():
    """
    Step-by-step demonstration of backpropagation algorithm
    """
    print("="*80)
    print("BACKPROPAGATION ALGORITHM: STEP-BY-STEP DEMONSTRATION")
    print("="*80)
    
    # Simple network for XOR problem: 2 -> 3 -> 1
    print("\nNetwork Architecture: 2 Input -> 3 Hidden -> 1 Output")
    print("-" * 80)
    
    # Sample input and target
    x = np.array([[1, 0]])  # Input: [1, 0]
    y = np.array([[1]])     # Target: 1 (XOR of 1 and 0 is 1)
    
    print(f"\nInput (x): {x.flatten()}")
    print(f"Target (y): {y.flatten()}")
    
    # Initialize small network manually
    np.random.seed(42)
    W1 = np.array([[0.5, -0.3, 0.2], [0.1, 0.4, -0.2]])  # (2, 3)
    b1 = np.array([[0.1, -0.1, 0.05]])                    # (1, 3)
    W2 = np.array([[0.3], [-0.5], [0.2]])                 # (3, 1)
    b2 = np.array([[0.1]])                                 # (1, 1)
    
    print("\n" + "="*80)
    print("STEP 1: FORWARD PROPAGATION")
    print("="*80)
    
    # Forward pass
    print("\n1.1 Hidden Layer:")
    z1 = np.dot(x, W1) + b1
    print(f"  z1 = x @ W1 + b1 = {z1.flatten()}")
    
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid
    print(f"  a1 = sigmoid(z1) = {a1.flatten()}")
    
    print("\n1.2 Output Layer:")
    z2 = np.dot(a1, W2) + b2
    print(f"  z2 = a1 @ W2 + b2 = {z2.flatten()}")
    
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid (output layer)
    print(f"  a2 = sigmoid(z2) = {a2.flatten()}")
    
    # Compute loss
    loss = -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))
    print(f"\n1.3 Loss:")
    print(f"  Loss = -[y * log(a2) + (1-y) * log(1-a2)] = {loss:.6f}")
    
    print("\n" + "="*80)
    print("STEP 2: BACKWARD PROPAGATION")
    print("="*80)
    
    # Backward pass
    print("\n2.1 Output Layer Gradient:")
    delta2 = (a2 - y) * a2 * (1 - a2)  # Derivative of sigmoid
    print(f"  δ2 = (a2 - y) * a2 * (1 - a2) = {delta2.flatten()}")
    
    print("\n2.2 Weight and Bias Gradients (Output Layer):")
    dW2 = np.dot(a1.T, delta2) / len(x)
    db2 = np.sum(delta2, axis=0, keepdims=True) / len(x)
    print(f"  dW2 = a1.T @ δ2 / m = \n{dW2.flatten()}")
    print(f"  db2 = sum(δ2) / m = {db2.flatten()}")
    
    print("\n2.3 Hidden Layer Gradient:")
    delta1 = np.dot(delta2, W2.T) * a1 * (1 - a1)  # Chain rule + sigmoid derivative
    print(f"  δ1 = (δ2 @ W2.T) * a1 * (1 - a1) = {delta1.flatten()}")
    
    print("\n2.4 Weight and Bias Gradients (Hidden Layer):")
    dW1 = np.dot(x.T, delta1) / len(x)
    db1 = np.sum(delta1, axis=0, keepdims=True) / len(x)
    print(f"  dW1 = x.T @ δ1 / m = \n{dW1.flatten()}")
    print(f"  db1 = sum(δ1) / m = {db1.flatten()}")
    
    print("\n" + "="*80)
    print("STEP 3: WEIGHT UPDATES")
    print("="*80)
    
    learning_rate = 0.5
    print(f"\nUsing learning rate: {learning_rate}")
    
    print("\n3.1 Update Output Layer:")
    W2_new = W2 - learning_rate * dW2
    b2_new = b2 - learning_rate * db2
    print(f"  W2_new = W2 - α * dW2")
    print(f"  W2_old = {W2.flatten()}")
    print(f"  W2_new = {W2_new.flatten()}")
    
    print("\n3.2 Update Hidden Layer:")
    W1_new = W1 - learning_rate * dW1
    b1_new = b1 - learning_rate * db1
    print(f"  W1_new = W1 - α * dW1")
    print(f"  W1_old[0] = {W1[0]}")
    print(f"  W1_new[0] = {W1_new[0]}")
    
    print("\n" + "="*80)
    print("STEP 4: VERIFICATION WITH NEW WEIGHTS")
    print("="*80)
    
    # Forward pass with new weights
    z1_new = np.dot(x, W1_new) + b1_new
    a1_new = 1 / (1 + np.exp(-z1_new))
    z2_new = np.dot(a1_new, W2_new) + b2_new
    a2_new = 1 / (1 + np.exp(-z2_new))
    loss_new = -np.mean(y * np.log(a2_new) + (1 - y) * np.log(1 - a2_new))
    
    print(f"\nAfter one weight update:")
    print(f"  Old Loss: {loss:.6f}")
    print(f"  New Loss: {loss_new:.6f}")
    print(f"  Loss Reduction: {(loss - loss_new):.6f}")
    print(f"  Old Output: {a2.flatten()[0]:.6f}")
    print(f"  New Output: {a2_new.flatten()[0]:.6f}")
    print(f"  Target: {y.flatten()[0]:.6f}")
    
    print("\n" + "="*80)
    print("KEY MATHEMATICAL CONCEPTS")
    print("="*80)
    print("""
1. CHAIN RULE IN BACKPROPAGATION
   For output layer: ∂L/∂W2 = ∂L/∂a2 * ∂a2/∂z2 * ∂z2/∂W2
   This is applied recursively through all layers

2. SIGMOID ACTIVATION AND DERIVATIVE
   σ(x) = 1 / (1 + e^(-x))
   σ'(x) = σ(x) * (1 - σ(x))

3. CROSS-ENTROPY LOSS
   L = -[y * log(a) + (1-y) * log(1-a)]
   This is differentiable and suitable for binary classification

4. GRADIENT DESCENT UPDATE
   W_new = W_old - α * ∇L/∂W
   Where α is the learning rate

5. BATCH GRADIENT COMPUTATION
   ∇L/∂W = (1/m) * Σ(∂L_i/∂W)  for all m samples
    """)
    
    print("\n" + "="*80)
    print("COMPARISON OF ACTIVATION FUNCTIONS")
    print("="*80)
    
    x_range = np.linspace(-5, 5, 100)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x_range))
    sigmoid_deriv = sigmoid * (1 - sigmoid)
    
    # Tanh
    tanh = np.tanh(x_range)
    tanh_deriv = 1 - tanh**2
    
    # ReLU
    relu = np.maximum(0, x_range)
    relu_deriv = (x_range > 0).astype(float)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Activation Functions and Their Derivatives', fontsize=14, fontweight='bold')
    
    # Sigmoid
    axes[0, 0].plot(x_range, sigmoid, 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1, 0].plot(x_range, sigmoid_deriv, 'b-', linewidth=2)
    axes[1, 0].set_title("Sigmoid' (Derivative)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Tanh
    axes[0, 1].plot(x_range, tanh, 'g-', linewidth=2)
    axes[0, 1].set_title('Tanh')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1, 1].plot(x_range, tanh_deriv, 'g-', linewidth=2)
    axes[1, 1].set_title("Tanh' (Derivative)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # ReLU
    axes[0, 2].plot(x_range, relu, 'r-', linewidth=2)
    axes[0, 2].set_title('ReLU')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1, 2].plot(x_range, relu_deriv, 'r-', linewidth=2)
    axes[1, 2].set_title("ReLU' (Derivative)")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    for ax in axes.flatten():
        ax.set_xlim(-5, 5)
    
    plt.tight_layout()
    plt.savefig('/tmp/workspace/98124kiran/project1/deep_learning/activation_functions_derivatives.png', 
                dpi=150, bbox_inches='tight')
    print("\nSaved activation functions comparison plot")
    
    print("\n" + "="*80)
    print("ALGORITHM COMPLEXITY")
    print("="*80)
    print("""
Forward Pass Complexity:
  - Each layer: O(n_in * n_out) for matrix multiplication
  - Total: O(Σ n_i * n_(i+1)) for all layers

Backward Pass Complexity:
  - Similar to forward pass: O(Σ n_i * n_(i+1))
  
Memory Complexity:
  - Storage for activations: O(batch_size * Σ n_i)
  - Storage for weights/biases: O(Σ n_i * n_(i+1))

Time Complexity per Epoch:
  - Forward + Backward: O(batch_size * Σ n_i * n_(i+1))
  - With m samples: O(m/batch_size * layer_complexity)
    """)


if __name__ == '__main__':
    demonstrate_backpropagation()
