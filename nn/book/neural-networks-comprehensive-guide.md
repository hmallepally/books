# Neural Networks: A Comprehensive Guide

## Introduction
Welcome to the world of Neural Networks! This book is your complete guide to understanding and implementing neural networks from the ground up. Whether you're a beginner or an experienced practitioner, you'll find detailed explanations of the mathematical foundations, practical implementations, and hands-on examples that will deepen your understanding of this powerful technology.

---

## Chapter 1: Foundations of Neural Networks

### 1.1 What are Neural Networks?

Neural networks are computational models inspired by biological neural networks in the human brain. They consist of interconnected nodes (neurons) that process information and learn patterns from data.

**Core Definition:**
A neural network is a collection of connected units (neurons) organized in layers that can learn to recognize patterns in data through a process called training.

**Key Characteristics:**
- **Layered Architecture**: Neurons organized in input, hidden, and output layers
- **Learning Capability**: Can learn complex patterns from data
- **Non-linear Processing**: Can model non-linear relationships
- **Parallel Processing**: Multiple neurons work simultaneously
- **Adaptive Weights**: Connection strengths change during learning

**Biological Inspiration:**
```
Biological Neuron → Artificial Neuron
Dendrites        → Input connections
Cell body        → Processing unit
Axon             → Output connection
Synapses         → Weights
```

### 1.2 The Perceptron: The First Neural Network

The perceptron, developed by Frank Rosenblatt in 1957, was the first artificial neural network.

**Perceptron Structure:**
```
Inputs: x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ
Bias: b
Output: y = f(Σᵢ wᵢxᵢ + b)
```

**Mathematical Representation:**
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
y = f(z)
```

**Learning Rule (Perceptron Learning Algorithm):**
```
wᵢ(new) = wᵢ(old) + α(y_true - y_pred)xᵢ
b(new) = b(old) + α(y_true - y_pred)
```

**Limitations:**
- Can only solve linearly separable problems
- Cannot learn XOR function
- Limited to binary classification

### 1.3 Multi-Layer Perceptrons (MLPs)

Multi-layer perceptrons extend the single perceptron by adding hidden layers, enabling the network to learn non-linear patterns.

**Architecture:**
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Mathematical Representation:**
For a network with L layers:
```
a⁽⁰⁾ = x (input)
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
```

**Forward Propagation:**
```
Layer 1: z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾, a⁽¹⁾ = f⁽¹⁾(z⁽¹⁾)
Layer 2: z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾, a⁽²⁾ = f⁽²⁾(z⁽²⁾)
...
Output: y = a⁽ᴸ⁾
```

### 1.4 The Universal Approximation Theorem

**Theorem:** A feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of ℝⁿ, under mild assumptions on the activation function.

**Implications:**
- Neural networks can approximate any continuous function
- Single hidden layer is theoretically sufficient
- Multiple layers may be more efficient in practice

---

## Chapter 2: Mathematical Foundations

### 2.1 Linear Algebra for Neural Networks

**Vectors and Matrices:**
Neural networks heavily rely on vector and matrix operations for efficient computation.

**Weight Matrix:**
```
W = [w₁₁  w₁₂  ...  w₁ₙ]
    [w₂₁  w₂₂  ...  w₂ₙ]
    [...  ...  ...  ...]
    [wₘ₁  wₘ₂  ...  wₘₙ]
```

**Matrix Multiplication:**
```
z = Wx + b
```

**Batch Processing:**
```
Z = WX + b
where X is a matrix of input samples
```

### 2.2 Calculus and Gradients

**Partial Derivatives:**
For a function f(x₁, x₂, ..., xₙ), the gradient is:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Chain Rule:**
For composite functions, the chain rule is essential:
```
∂f/∂x = Σᵢ (∂f/∂yᵢ) × (∂yᵢ/∂x)
```

**Gradient Descent:**
```
θ(new) = θ(old) - α∇J(θ)
where α is the learning rate and J is the cost function
```

### 2.3 Loss Functions

**Mean Squared Error (MSE):**
```
J(θ) = (1/n) Σᵢ(yᵢ - ŷᵢ)²
```

**Cross-Entropy Loss:**
```
J(θ) = -Σᵢ yᵢ log(ŷᵢ)
```

**Binary Cross-Entropy:**
```
J(θ) = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Categorical Cross-Entropy:**
```
J(θ) = -Σᵢ yᵢ log(ŷᵢ)
```

### 2.4 Optimization Algorithms

**Stochastic Gradient Descent (SGD):**
```
θ(t+1) = θ(t) - α∇J(θ(t))
```

**Momentum:**
```
v(t+1) = βv(t) + (1-β)∇J(θ(t))
θ(t+1) = θ(t) - αv(t+1)
```

**Adam:**
```
m(t+1) = β₁m(t) + (1-β₁)∇J(θ(t))
v(t+1) = β₂v(t) + (1-β₂)(∇J(θ(t)))²
m̂ = m(t+1)/(1-β₁ᵗ⁺¹)
v̂ = v(t+1)/(1-β₂ᵗ⁺¹)
θ(t+1) = θ(t) - αm̂/(√v̂ + ε)
```

---

## Chapter 3: Activation Functions

### 3.1 Why Activation Functions?

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

**Properties of Good Activation Functions:**
- **Non-linear**: Enables learning of non-linear patterns
- **Differentiable**: Required for gradient-based learning
- **Monotonic**: Helps with convergence
- **Bounded**: Prevents exploding gradients

### 3.2 Step Function (Heaviside)

**Mathematical Definition:**
```
f(x) = { 1 if x ≥ 0
       { 0 if x < 0
```

**Derivative:**
```
f'(x) = 0 (except at x = 0, where it's undefined)
```

**Characteristics:**
- ✅ Simple and interpretable
- ❌ Not differentiable
- ❌ Binary output only
- ❌ Vanishing gradient problem

### 3.3 Sigmoid Function

**Mathematical Definition:**
```
f(x) = 1/(1 + e⁻ˣ)
```

**Derivative:**
```
f'(x) = f(x)(1 - f(x))
```

**Characteristics:**
- ✅ Smooth and differentiable
- ✅ Output range (0, 1)
- ✅ Interpretable as probability
- ❌ Vanishing gradient for large |x|
- ❌ Not zero-centered

**Implementation:**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

### 3.4 Hyperbolic Tangent (tanh)

**Mathematical Definition:**
```
f(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)
```

**Derivative:**
```
f'(x) = 1 - f(x)²
```

**Characteristics:**
- ✅ Zero-centered output (-1, 1)
- ✅ Smooth and differentiable
- ✅ Better gradient flow than sigmoid
- ❌ Still has vanishing gradient for large |x|

**Implementation:**
```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

### 3.5 Rectified Linear Unit (ReLU)

**Mathematical Definition:**
```
f(x) = max(0, x)
```

**Derivative:**
```
f'(x) = { 1 if x > 0
        { 0 if x ≤ 0
```

**Characteristics:**
- ✅ Simple and computationally efficient
- ✅ No vanishing gradient for positive inputs
- ✅ Sparse activation (many neurons inactive)
- ❌ Dying ReLU problem (neurons can become permanently inactive)
- ❌ Not differentiable at x = 0

**Implementation:**
```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
```

### 3.6 Leaky ReLU

**Mathematical Definition:**
```
f(x) = { x if x > 0
       { αx if x ≤ 0
where α is a small positive constant (e.g., 0.01)
```

**Derivative:**
```
f'(x) = { 1 if x > 0
        { α if x ≤ 0
```

**Characteristics:**
- ✅ Addresses dying ReLU problem
- ✅ Simple and efficient
- ✅ Allows small negative gradients
- ❌ Requires tuning of α parameter

### 3.7 Exponential Linear Unit (ELU)

**Mathematical Definition:**
```
f(x) = { x if x > 0
       { α(eˣ - 1) if x ≤ 0
```

**Derivative:**
```
f'(x) = { 1 if x > 0
        { f(x) + α if x ≤ 0
```

**Characteristics:**
- ✅ Smooth for negative inputs
- ✅ Closer to zero mean outputs
- ✅ Better gradient flow
- ❌ More computationally expensive

### 3.8 Softmax Function

**Mathematical Definition:**
```
f(xᵢ) = eˣⁱ/Σⱼ eˣʲ
```

**Derivative:**
```
∂fᵢ/∂xⱼ = fᵢ(δᵢⱼ - fⱼ)
where δᵢⱼ is the Kronecker delta
```

**Characteristics:**
- ✅ Outputs sum to 1 (probability distribution)
- ✅ Used for multi-class classification
- ✅ Differentiable
- ❌ Computationally expensive for large inputs

**Implementation:**
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### 3.9 Activation Function Comparison

| Function | Range | Derivative | Pros | Cons |
|----------|-------|------------|------|------|
| Step | {0, 1} | 0 | Simple | Not differentiable |
| Sigmoid | (0, 1) | f(1-f) | Smooth, probability | Vanishing gradient |
| Tanh | (-1, 1) | 1-f² | Zero-centered | Vanishing gradient |
| ReLU | [0, ∞) | {0, 1} | Simple, efficient | Dying ReLU |
| Leaky ReLU | (-∞, ∞) | {α, 1} | No dying ReLU | Tuning required |
| ELU | (-α, ∞) | Smooth | Smooth, zero-mean | Expensive |
| Softmax | (0, 1) | Complex | Probability dist. | Expensive |

---

## Chapter 4: Neural Network Layers

### 4.1 Input Layer

The input layer receives the raw data and passes it to the first hidden layer.

**Characteristics:**
- No computation (identity function)
- Number of neurons = number of input features
- Data preprocessing often applied here

**Example:**
```
Input features: [x₁, x₂, x₃, x₄]
Input layer: 4 neurons
```

### 4.2 Hidden Layers

Hidden layers perform the main computation in neural networks.

**Fully Connected (Dense) Layer:**
```
z = Wx + b
a = f(z)
```

**Layer Types:**

**1. Dense Layer:**
- Every neuron connected to all neurons in previous layer
- Most common type
- Computationally expensive for large layers

**2. Convolutional Layer:**
- Used for spatial data (images)
- Shared weights across spatial locations
- Translation invariant

**3. Recurrent Layer:**
- Used for sequential data
- Has memory of previous inputs
- Can process variable-length sequences

**4. Pooling Layer:**
- Reduces spatial dimensions
- Common in CNNs
- Max pooling, average pooling

### 4.3 Output Layer

The output layer produces the final predictions.

**Classification:**
- **Binary**: 1 neuron with sigmoid activation
- **Multi-class**: N neurons with softmax activation

**Regression:**
- 1 or more neurons with linear activation

**Example Architectures:**
```
Binary Classification: [input] → [hidden] → [1 neuron, sigmoid]
Multi-class: [input] → [hidden] → [N neurons, softmax]
Regression: [input] → [hidden] → [1 neuron, linear]
```

### 4.4 Layer Dimensions

**Weight Matrix Dimensions:**
```
W⁽ˡ⁾: [n⁽ˡ⁾ × n⁽ˡ⁻¹⁾]
b⁽ˡ⁾: [n⁽ˡ⁾]
```

**Forward Pass:**
```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
```

**Example:**
```
Input: 4 features
Hidden layer 1: 10 neurons
Hidden layer 2: 8 neurons
Output: 3 classes

W⁽¹⁾: [10 × 4], b⁽¹⁾: [10]
W⁽²⁾: [8 × 10], b⁽²⁾: [8]
W⁽³⁾: [3 × 8], b⁽³⁾: [3]
```

---

## Chapter 5: Bias and Regularization

### 5.1 The Role of Bias

Bias terms allow neural networks to shift the activation function, making them more flexible.

**Mathematical Role:**
```
z = Wx + b
```

**Without bias:**
- Line must pass through origin
- Limited flexibility
- Cannot model functions that don't pass through origin

**With bias:**
- Line can be shifted anywhere
- Much more flexible
- Can model any linear function

**Geometric Interpretation:**
```
y = mx + b
b shifts the line up/down
m controls the slope
```

### 5.2 Bias Initialization

**Common Strategies:**

**1. Zero Initialization:**
```
b = 0
```
- Simple but may lead to symmetry breaking issues
- All neurons in a layer start with same bias

**2. Small Constant:**
```
b = 0.01
```
- Prevents dead neurons
- Common for ReLU activations

**3. Xavier/Glorot Initialization:**
```
b = 0
```
- Used with Xavier weight initialization
- Maintains variance across layers

### 5.3 Regularization Techniques

**1. L1 Regularization (Lasso):**
```
J_reg = J + λΣ|w|
```
- Encourages sparse weights
- Feature selection
- Less sensitive to outliers

**2. L2 Regularization (Ridge):**
```
J_reg = J + λΣw²
```
- Prevents overfitting
- Smooths the model
- Most commonly used

**3. Dropout:**
```
During training: randomly set some activations to 0
During inference: scale activations by dropout rate
```
- Prevents co-adaptation
- Reduces overfitting
- Acts as ensemble method

**4. Early Stopping:**
```
Stop training when validation loss increases
```
- Prevents overfitting
- Simple to implement
- No additional parameters

**5. Data Augmentation:**
```
Create additional training examples
```
- Increases effective dataset size
- Improves generalization
- Domain-specific techniques

### 5.4 Batch Normalization

**Purpose:**
- Stabilizes training
- Reduces internal covariate shift
- Allows higher learning rates

**Algorithm:**
```
μ = (1/m) Σxᵢ
σ² = (1/m) Σ(xᵢ - μ)²
x̂ = (x - μ)/√(σ² + ε)
y = γx̂ + β
```

**Benefits:**
- Faster training
- Less sensitive to initialization
- Acts as regularization

---

## Chapter 6: Backpropagation

### 6.1 The Backpropagation Algorithm

Backpropagation is the algorithm used to compute gradients in neural networks.

**Key Idea:**
- Use chain rule to compute gradients
- Propagate errors backward through the network
- Update weights using gradient descent

### 6.2 Forward Pass

**For each layer l:**
```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
```

### 6.3 Backward Pass

**Output Layer Error:**
```
δ⁽ᴸ⁾ = ∇ₐJ ⊙ f'⁽ᴸ⁾(z⁽ᴸ⁾)
```

**Hidden Layer Errors:**
```
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ f'⁽ˡ⁾(z⁽ˡ⁾)
```

**Weight Gradients:**
```
∇W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
∇b⁽ˡ⁾ = δ⁽ˡ⁾
```

**Weight Updates:**
```
W⁽ˡ⁾ = W⁽ˡ⁾ - α∇W⁽ˡ⁾
b⁽ˡ⁾ = b⁽ˡ⁾ - α∇b⁽ˡ⁾
```

### 6.4 Backpropagation Example

**Network:**
```
Input: [x₁, x₂]
Hidden: 2 neurons with sigmoid
Output: 1 neuron with sigmoid
```

**Forward Pass:**
```
z¹ = W¹x + b¹
a¹ = sigmoid(z¹)
z² = W²a¹ + b²
a² = sigmoid(z²)
```

**Backward Pass:**
```
δ² = (a² - y) × sigmoid'(z²)
δ¹ = (W²)ᵀδ² × sigmoid'(z¹)

∇W² = δ²(a¹)ᵀ
∇b² = δ²
∇W¹ = δ¹xᵀ
∇b¹ = δ¹
```

---

## Chapter 7: Training Neural Networks

### 7.1 Training Process

**1. Initialize Weights:**
- Random initialization
- Xavier/Glorot initialization
- He initialization

**2. Forward Pass:**
- Compute predictions
- Calculate loss

**3. Backward Pass:**
- Compute gradients
- Update weights

**4. Repeat:**
- Until convergence or maximum epochs

### 7.2 Learning Rate

**Importance:**
- Controls step size in gradient descent
- Too high: may not converge
- Too low: slow convergence

**Learning Rate Scheduling:**

**1. Fixed Learning Rate:**
```
α = constant
```

**2. Step Decay:**
```
α = α₀ × γ^floor(epoch/step_size)
```

**3. Exponential Decay:**
```
α = α₀ × γ^epoch
```

**4. Adaptive Learning Rate:**
- Adam, RMSprop, Adagrad
- Automatically adjust learning rate

### 7.3 Mini-batch Training

**Benefits:**
- Better gradient estimates
- Parallel processing
- Memory efficiency

**Batch Size Selection:**
- **Small batches**: More noise, better generalization
- **Large batches**: Stable gradients, faster training
- **Typical sizes**: 32, 64, 128, 256

### 7.4 Monitoring Training

**Metrics to Track:**
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

**Overfitting Detection:**
- Validation loss increases while training loss decreases
- Large gap between training and validation performance

**Underfitting Detection:**
- Both training and validation loss are high
- Model capacity too low

---

## Chapter 8: Practical Implementation

### 8.1 Neural Network Architecture Design

**Guidelines:**

**1. Start Simple:**
- Begin with a simple architecture
- Add complexity gradually
- Validate each addition

**2. Layer Sizes:**
- Input layer: match input features
- Hidden layers: typically decreasing size
- Output layer: match task requirements

**3. Number of Layers:**
- Start with 1-2 hidden layers
- Add layers if underfitting
- Use skip connections for deep networks

**4. Activation Functions:**
- Hidden layers: ReLU, Leaky ReLU, ELU
- Output layer: depends on task

### 8.2 Weight Initialization

**Xavier/Glorot Initialization:**
```
W ~ N(0, 2/(n_in + n_out))
```

**He Initialization:**
```
W ~ N(0, 2/n_in)
```

**Why Important:**
- Prevents vanishing/exploding gradients
- Ensures proper signal flow
- Faster convergence

### 8.3 Data Preprocessing

**Normalization:**
```
x_norm = (x - μ)/σ
```

**Standardization:**
```
x_std = (x - min)/(max - min)
```

**Why Important:**
- Faster convergence
- Better gradient flow
- Prevents numerical issues

### 8.4 Hyperparameter Tuning

**Key Hyperparameters:**
- Learning rate
- Number of layers
- Layer sizes
- Batch size
- Regularization strength

**Tuning Strategies:**
- Grid search
- Random search
- Bayesian optimization
- Manual tuning

---

## Chapter 9: Convolutional Neural Networks (CNNs)

### 9.1 Introduction
Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. They use convolutional layers to extract spatial features.

### 9.2 Key Concepts
- Convolution operation
- Filters/kernels
- Feature maps
- Pooling (max, average)
- Padding and stride
- Stacking convolutional layers

### 9.3 Mathematical Formulation
- Convolution: \( (I * K)(x, y) = \sum_m \sum_n I(x+m, y+n) K(m, n) \)
- Output size: \( \text{out} = \frac{\text{in} - \text{kernel} + 2 \times \text{padding}}{\text{stride}} + 1 \)

### 9.4 Practical Notes
- Used in image classification, object detection, segmentation
- Transfer learning with pretrained CNNs (e.g., VGG, ResNet)

---

## Chapter 10: Recurrent Neural Networks (RNNs)

### 10.1 Introduction
RNNs are designed for sequential data (text, time series). They maintain a hidden state to capture temporal dependencies.

### 10.2 Key Concepts
- Sequence modeling
- Hidden state propagation
- Vanishing/exploding gradients

### 10.3 Mathematical Formulation
- \( h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b) \)

### 10.4 Practical Notes
- Used in language modeling, speech recognition, time series forecasting

---

## Chapter 11: LSTMs and GRUs

### 11.1 Introduction
LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units) are advanced RNNs that address vanishing gradient problems.

### 11.2 Key Concepts
- Gates (input, forget, output)
- Cell state
- GRU simplification

### 11.3 Mathematical Formulation
- LSTM: \( f_t, i_t, o_t, c_t, h_t \) equations
- GRU: \( z_t, r_t, h_t \) equations

### 11.4 Practical Notes
- Used in translation, text generation, sequence prediction

---

## Chapter 12: Attention and Transformers

### 12.1 Introduction
Attention mechanisms allow models to focus on relevant parts of input. Transformers use self-attention for parallel sequence modeling.

### 12.2 Key Concepts
- Attention weights
- Self-attention
- Multi-head attention
- Positional encoding

### 12.3 Mathematical Formulation
- Attention: \( \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V \)
- Transformer block: LayerNorm, Feedforward, Residuals

### 12.4 Practical Notes
- Used in NLP (BERT, GPT), vision (ViT)
- State-of-the-art for many tasks

---

## Chapter 13: Generative Adversarial Networks (GANs)

### 13.1 Introduction
GANs consist of a generator and discriminator in a minimax game, used for data generation.

### 13.2 Key Concepts
- Generator, discriminator
- Adversarial loss
- Training instability

### 13.3 Mathematical Formulation
- \( \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] \)

### 13.4 Practical Notes
- Used for image synthesis, style transfer, data augmentation

---

## Chapter 14: Autoencoders

### 14.1 Introduction
Autoencoders learn to compress and reconstruct data, useful for dimensionality reduction and denoising.

### 14.2 Key Concepts
- Encoder, decoder
- Bottleneck
- Loss: reconstruction error

### 14.3 Mathematical Formulation
- \( z = f_{enc}(x) \), \( \hat{x} = f_{dec}(z) \)

### 14.4 Practical Notes
- Used for anomaly detection, pretraining, data compression

---

## Chapter 15: Transfer Learning

### 15.1 Introduction
Transfer learning leverages pretrained models for new tasks, reducing data and training time requirements.

### 15.2 Key Concepts
- Feature extraction
- Fine-tuning
- Freezing layers

### 15.3 Practical Notes
- Common in vision (ImageNet models), NLP (BERT, GPT)

---

## Chapter 16: Interpretability and Explainability

### 16.1 Introduction
Understanding model decisions is crucial for trust and debugging.

### 16.2 Key Concepts
- Feature importance
- Saliency maps
- SHAP, LIME

### 16.3 Practical Notes
- Use interpretability tools to analyze and debug models

---

## Chapter 17: Practical Tips and Best Practices

### 17.1 Model Design
- Start simple, increase complexity as needed
- Use appropriate architectures for data type

### 17.2 Training
- Monitor loss/accuracy curves
- Use callbacks (early stopping, learning rate schedules)
- Regularize and augment data

### 17.3 Debugging
- Check for data leakage
- Visualize activations and gradients

### 17.4 Deployment
- Export models (ONNX, TensorFlow SavedModel, TorchScript)
- Monitor performance in production

---

*This concludes the advanced topics section. For code and notebooks, see the `code/` and `notebooks/` directories for practical implementations of these advanced models.* 