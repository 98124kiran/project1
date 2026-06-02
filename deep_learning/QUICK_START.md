# Quick Start Guide - ANN Backpropagation

## Installation

```bash
# Install dependencies (if not already installed)
pip install numpy matplotlib
```

## Running the Examples

### 1. Compare Activation Functions on Multiple Datasets
```bash
python -m deep_learning.test_ann_comparison
```
This will:
- Train networks with Sigmoid, Tanh, and ReLU on 3 different datasets
- Generate performance comparison plots
- Generate decision boundary visualizations
- Print detailed results and analysis

**Output files:**
- `comparison_xor.png` - Performance metrics comparison
- `boundaries_xor.png` - Decision boundaries
- `comparison_linearly_separable.png` - Performance on linear data
- `boundaries_linearly_separable.png` - Linear boundaries
- `comparison_concentric_circles.png` - Performance on non-linear data
- `boundaries_concentric_circles.png` - Non-linear boundaries

### 2. Educational Backpropagation Demonstration
```bash
python -m deep_learning.backpropagation_demonstration
```
This will:
- Show step-by-step forward propagation
- Show step-by-step backward propagation
- Show weight updates with specific values
- Verify loss reduction after updates
- Generate activation function comparison plots

**Output files:**
- `activation_functions_derivatives.png` - Activation functions and derivatives

## Basic Python Usage

```python
from deep_learning import NeuralNetwork, create_xor_dataset

# Load XOR dataset
X_train, y_train, X_test, y_test = create_xor_dataset()

# Create network: 2 inputs -> 8 hidden -> 1 output
network = NeuralNetwork(
    layer_sizes=[2, 8, 1],
    activation='relu',           # sigmoid, tanh, or relu
    learning_rate=0.5,
    momentum=0.0
)

# Train the network
network.train(X_train, y_train, epochs=2000, verbose=True)

# Evaluate
loss, accuracy = network.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.6f}, Accuracy: {accuracy:.4f}")

# Make predictions
predictions = network.predict(X_test)              # 0/1 predictions
probabilities = network.predict_proba(X_test)     # Probability estimates
```

## Key Results

### Performance on Test Datasets

**XOR Dataset (Most Difficult):**
- Sigmoid: 100% accuracy, Loss: 0.1094
- Tanh: 100% accuracy, Loss: 0.0333
- ReLU: 100% accuracy, Loss: 0.0140 ✓ Best

**Linearly Separable Dataset:**
- Sigmoid: 100% accuracy, Loss: 0.0293
- Tanh: 100% accuracy, Loss: 0.0107
- ReLU: 100% accuracy, Loss: 0.0085 ✓ Best

**Concentric Circles Dataset (Non-linear):**
- Sigmoid: 95.0% accuracy, Loss: 0.3714
- Tanh: 97.5% accuracy, Loss: 0.1056
- ReLU: 100% accuracy, Loss: 0.0315 ✓ Best

## Recommendation

**Use ReLU for hidden layers** - it consistently achieves:
- Fastest convergence (lowest loss)
- Best accuracy on non-linear problems
- Computational efficiency
- Superior performance in deep networks

## Implementation Details

### Forward Pass Algorithm
1. Compute pre-activation: z = X @ W + b
2. Apply activation: a = activation(z)
3. Pass to next layer (repeat)

### Backward Pass Algorithm
1. Compute output error: δ = (a - y) * a' 
2. Backpropagate: δ_prev = (δ @ W.T) * a'_prev
3. Update weights: W -= α * (X.T @ δ)
4. Update biases: b -= α * sum(δ)

### Loss Function
Binary Cross-Entropy: L = -[y*log(a) + (1-y)*log(1-a)]

## Files Generated

```
deep_learning/
├── ann_backpropagation.py                    # Core implementation
├── test_ann_comparison.py                    # Comparison tests
├── backpropagation_demonstration.py          # Educational demo
├── README.md                                  # Full documentation
├── IMPLEMENTATION_SUMMARY.md                 # Detailed results
├── QUICK_START.md                            # This file
│
├── comparison_xor.png                        # Performance plots
├── comparison_linearly_separable.png
├── comparison_concentric_circles.png
│
├── boundaries_xor.png                        # Decision boundaries
├── boundaries_linearly_separable.png
├── boundaries_concentric_circles.png
│
└── activation_functions_derivatives.png      # Function comparison
```

## Troubleshooting

**Problem:** ModuleNotFoundError: No module named 'numpy'
```bash
pip install numpy matplotlib
```

**Problem:** Plots not showing
- Plots are automatically saved to PNG files
- Open them with an image viewer
- Or use: `plt.show()` in Python interactive mode

**Problem:** Training is slow
- Reduce epochs: `network.train(..., epochs=500)`
- Increase learning rate: `learning_rate=1.0`
- Use smaller batch size: `batch_size=4`

## Learning Progression

1. **Read** `README.md` for theory
2. **Run** `backpropagation_demonstration.py` for step-by-step understanding
3. **Run** `test_ann_comparison.py` to see performance comparison
4. **Experiment** with different architectures and hyperparameters
5. **Modify** the code to implement variants (e.g., L1/L2 regularization)

## Next Steps

After mastering this ANN implementation, explore:
- Deep neural networks (>3 layers)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Batch normalization
- Dropout regularization
- Advanced optimizers (Adam, RMSprop)

---

**Happy Learning!** 🚀
