# Artificial Neural Network with Backpropagation Algorithm

## Overview

This module implements a **Feed-Forward Artificial Neural Network (ANN)** with the **Backpropagation algorithm** from scratch using NumPy. The implementation demonstrates:

1. **Forward Propagation**: Computing outputs from inputs through multiple layers
2. **Backpropagation**: Computing gradients and updating weights
3. **Activation Functions**: Sigmoid, Tanh, and ReLU
4. **Optimization**: Gradient descent with optional momentum
5. **Loss Function**: Binary cross-entropy for binary classification

## Key Features

### Architecture
- **Fully-connected feed-forward network**
- **Configurable hidden layers**: Can create networks with arbitrary layer sizes
- **Flexible activation functions**: Support for sigmoid, tanh, and ReLU
- **Momentum-based optimization**: Optional momentum for faster convergence

### Activation Functions Implemented

#### 1. **Sigmoid**
```
σ(x) = 1 / (1 + e^(-x))
```
- **Output Range**: (0, 1)
- **Derivative**: σ(x) * (1 - σ(x))
- **Characteristics**:
  - Smooth and differentiable
  - Classic choice for binary classification
  - Prone to vanishing gradient problem in deep networks

#### 2. **Tanh (Hyperbolic Tangent)**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- **Output Range**: (-1, 1)
- **Derivative**: 1 - tanh²(x)
- **Characteristics**:
  - Zero-centered output (helps convergence)
  - Stronger gradients than sigmoid
  - Often performs better than sigmoid

#### 3. **ReLU (Rectified Linear Unit)**
```
ReLU(x) = max(0, x)
```
- **Output Range**: [0, ∞)
- **Derivative**: 1 if x > 0, else 0
- **Characteristics**:
  - Computationally efficient
  - Mitigates vanishing gradient problem
  - Preferred in deep networks
  - Can suffer from "dying ReLU" (outputs always 0)

## Implementation Details

### Class: `NeuralNetwork`

```python
network = NeuralNetwork(
    layer_sizes=[2, 8, 1],      # Input: 2, Hidden: 8, Output: 1
    activation='sigmoid',        # 'sigmoid', 'tanh', or 'relu'
    learning_rate=0.5,          # Learning rate for gradient descent
    momentum=0.0                # Momentum coefficient (0 = no momentum)
)
```

### Key Methods

#### 1. **Forward Propagation**
```python
output = network.forward_propagation(X)
```
- Computes network output for given input
- Stores activations for backpropagation

#### 2. **Backpropagation**
```python
network.backward_propagation(X, y, output)
```
- Computes gradients using chain rule
- Updates weights and biases using gradient descent

#### 3. **Training**
```python
network.train(X_train, y_train, epochs=2000, batch_size=None)
```
- Full training loop with mini-batch support
- Tracks loss and accuracy history

#### 4. **Evaluation**
```python
loss, accuracy = network.evaluate(X_test, y_test)
predictions = network.predict(X_test)
probabilities = network.predict_proba(X_test)
```

## Test Datasets

### 1. **XOR Dataset**
- **Problem**: Classic non-linearly separable problem
- **Samples**: 4 points
- **Purpose**: Tests network's ability to learn non-linear decision boundaries

### 2. **Linearly Separable Dataset**
- **Problem**: Two Gaussian clouds, linearly separable
- **Samples**: 200 points (160 train, 40 test)
- **Purpose**: Baseline for linear classification

### 3. **Concentric Circles Dataset**
- **Problem**: Non-linearly separable (concentric patterns)
- **Samples**: 200 points (160 train, 40 test)
- **Purpose**: Tests network's ability on circular decision boundaries

## Experimental Results

### Test Results on Three Datasets

#### XOR Dataset
```
SIGMOID:  Train Acc: 1.0000, Test Acc: 1.0000, Loss: 0.1094
TANH:     Train Acc: 1.0000, Test Acc: 1.0000, Loss: 0.0333
RELU:     Train Acc: 1.0000, Test Acc: 1.0000, Loss: 0.0140
```

**Analysis**: All activation functions successfully learned the XOR function. ReLU showed lowest loss (fastest convergence).

#### Linearly Separable Dataset
```
SIGMOID:  Train Acc: 1.0000, Test Acc: 1.0000, Loss: 0.0293
TANH:     Train Acc: 1.0000, Test Acc: 1.0000, Loss: 0.0107
RELU:     Train Acc: 1.0000, Test Acc: 1.0000, Loss: 0.0085
```

**Analysis**: All activations performed perfectly (as expected for linearly separable data). ReLU showed fastest convergence.

#### Concentric Circles Dataset
```
SIGMOID:  Train Acc: 0.8875, Test Acc: 0.9500, Loss: 0.3714
TANH:     Train Acc: 1.0000, Test Acc: 0.9750, Loss: 0.1055
RELU:     Train Acc: 1.0000, Test Acc: 1.0000, Loss: 0.0315
```

**Analysis**: ReLU clearly outperformed on complex non-linear data. Sigmoid showed underfitting.

## Comparison of Activation Functions

| Aspect | Sigmoid | Tanh | ReLU |
|--------|---------|------|------|
| **Output Range** | (0, 1) | (-1, 1) | [0, ∞) |
| **Gradient Range** | (0, 0.25) | (0, 1) | {0, 1} |
| **Vanishing Gradient** | High | Medium | Low |
| **Computational Cost** | Medium | Medium | Low |
| **Linear Data** | Good | Good | Excellent |
| **Non-linear Data** | Fair | Good | Excellent |
| **Deep Networks** | Poor | Fair | Excellent |
| **Convergence Speed** | Slow | Medium | Fast |

## Performance Analysis

### Key Findings

1. **Convergence Speed**
   - ReLU: Fastest convergence (steepest gradients)
   - Tanh: Medium convergence
   - Sigmoid: Slowest convergence (vanishing gradients)

2. **Complex Non-linear Problems**
   - ReLU performs best on concentric circles (100% accuracy)
   - Tanh shows good performance (97.5% accuracy)
   - Sigmoid underfits (95% accuracy)

3. **Simplicity Trade-off**
   - ReLU: Not zero-centered but faster
   - Tanh: Zero-centered but slower than ReLU
   - Sigmoid: Traditional but slowest

## Visualizations Generated

### 1. Comparison Plots
- **Training Loss**: How loss decreases across epochs
- **Training Accuracy**: How accuracy improves during training
- **Final Performance**: Bar charts comparing final accuracies and losses

### 2. Decision Boundaries
- Visual representation of decision boundaries learned by each activation function
- Shows how different activations create different decision regions

## Usage Example

```python
from deep_learning.ann_backpropagation import NeuralNetwork, create_xor_dataset

# Create and load data
X_train, y_train, X_test, y_test = create_xor_dataset()

# Create network
network = NeuralNetwork(
    layer_sizes=[2, 8, 1],
    activation='relu',
    learning_rate=0.5
)

# Train
network.train(X_train, y_train, epochs=2000)

# Evaluate
loss, accuracy = network.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict
predictions = network.predict(X_test)
probabilities = network.predict_proba(X_test)
```

## Files

- **`ann_backpropagation.py`**: Core implementation of ANN with backpropagation
- **`test_ann_comparison.py`**: Test script comparing activation functions
- **`comparison_*.png`**: Performance comparison plots
- **`boundaries_*.png`**: Decision boundary visualizations

## Conclusions

### When to Use Each Activation Function

1. **Sigmoid**
   - Use for: Output layers (binary classification)
   - Avoid for: Hidden layers in deep networks

2. **Tanh**
   - Use for: Hidden layers when you want zero-centered activations
   - Good for: Recurrent neural networks
   - Performance: Better than sigmoid, but slower than ReLU

3. **ReLU**
   - Use for: Hidden layers in most cases
   - Preferred for: Deep networks
   - Performance: Fastest convergence, best for complex non-linear problems

### Key Takeaways

1. **ReLU is the modern standard** for hidden layers due to:
   - Faster convergence
   - Better performance on complex problems
   - Lower computational cost
   - Mitigation of vanishing gradient problem

2. **Tanh is a good middle ground** offering:
   - Better performance than sigmoid
   - Faster convergence than sigmoid
   - Zero-centered outputs

3. **Sigmoid remains useful** for:
   - Output layers in binary classification
   - Probability interpretations
   - Theoretical understanding

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.
