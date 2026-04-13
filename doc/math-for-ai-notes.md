# Math for AI — Study Notes

Study notes following the math learning path from `ai-study-guide.md`.
Each level builds on the previous one, focused on what matters for AI/ML.

---

## Level 1: Arithmetic and Pre-Algebra

### Fractions, Decimals, Percentages

- Fractions represent parts of a whole: `3/4 = 0.75 = 75%`
- Converting between forms:
  - Fraction → Decimal: divide numerator by denominator
  - Decimal → Percentage: multiply by 100
  - Percentage → Decimal: divide by 100
- **AI connection**: Probabilities are expressed as decimals (0.0 to 1.0) or percentages

### Negative Numbers and Absolute Value

- Negative numbers sit left of zero on the number line
- Absolute value `|x|` = distance from zero (always positive)
  - `|−5| = 5`, `|3| = 3`
- **AI connection**: Loss values and gradients can be negative. A negative gradient means "decrease this weight."

### Exponents and Logarithms

- Exponents: `2³ = 2 × 2 × 2 = 8`
- Logarithms are the inverse: `log₂(8) = 3` (what power of 2 gives 8?)
- Key properties:
  - `log(a × b) = log(a) + log(b)`
  - `log(a/b) = log(a) − log(b)`
  - `log(aⁿ) = n × log(a)`
- Natural log `ln(x)` = log base `e` (e ≈ 2.718)
- **AI connection**: Softmax uses `e^x`, loss functions use `log`. Log-loss (cross-entropy) is the standard classification loss.

### Order of Operations

- PEMDAS: Parentheses → Exponents → Multiplication/Division → Addition/Subtraction
- Left to right within same precedence
- **AI connection**: Reading formulas in papers requires this. A misread formula = wrong implementation.

### Basic Graphing

- X-Y coordinate plane: horizontal (x-axis) and vertical (y-axis)
- A point is `(x, y)`
- **AI connection**: Training loss curves plot epoch (x) vs loss (y). If the curve goes down, training is working.

**Resource**: [Khan Academy — Pre-Algebra](https://www.khanacademy.org/math/pre-algebra)

---

## Level 2: Algebra

### Variables and Equations

- A variable represents an unknown value: `y = 3x + 2`
- Solving an equation = isolating the variable
- **AI connection**: Model parameters (weights) are variables. Training = finding the values that minimize loss.

### Functions

- A function maps input to output: `f(x) = 2x + 1`
- Composition: `f(g(x))` — apply g first, then f
- **AI connection**: A neural network is a deeply nested function: `f(g(h(x)))` where each layer is a function.

### Slopes and Lines

- Linear equation: `y = mx + b`
  - `m` = slope (rate of change)
  - `b` = y-intercept (value when x = 0)
- Slope = rise / run = `(y₂ − y₁) / (x₂ − x₁)`
- **AI connection**: Linear regression fits `y = wx + b` where `w` is the weight and `b` is the bias. This is the simplest ML model.

### Systems of Equations

- Multiple equations with multiple unknowns
- Example: `2x + y = 5` and `x − y = 1` → solve for both x and y
- **AI connection**: Training optimizes millions of parameters simultaneously under constraints.

### Summation Notation

- `Σ` (sigma) means "add up a series"
- `Σᵢ₌₁ⁿ xᵢ = x₁ + x₂ + ... + xₙ`
- **AI connection**: Loss functions sum errors over all training examples: `L = (1/n) Σ (ŷᵢ − yᵢ)²`

**Resource**: [Khan Academy — Algebra 1 & 2](https://www.khanacademy.org/math/algebra)

---

## Level 3: Linear Algebra

This is the most important math for AI — all data flows through matrices.

### Vectors

- An ordered list of numbers: `v = [3, 1, 4]`
- Represents a point or direction in space
- Dimension = number of elements
- Vector addition: `[1, 2] + [3, 4] = [4, 6]` (element-wise)
- Scalar multiplication: `2 × [1, 2] = [2, 4]`
- **AI connection**: A single data point (e.g., a word embedding) is a vector. GPT-3 uses 12,288-dimensional vectors.

### Matrices

- A 2D grid of numbers (rows × columns)
- Example (2×3 matrix):

```
| 1  2  3 |
| 4  5  6 |
```

- Matrix addition: element-wise (same shape required)
- **AI connection**: A batch of data = a matrix. Model weights at each layer = a matrix.

### Dot Product

- Multiply matching elements and sum: `[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32`
- Measures how similar two vectors are
  - Large positive → similar direction
  - Near zero → perpendicular (unrelated)
  - Large negative → opposite direction
- **AI connection**: The attention mechanism in transformers computes dot products between query and key vectors to measure relevance.

### Matrix Multiplication

- To multiply A (m×n) by B (n×p), result is (m×p)
- Each element = dot product of a row from A and a column from B
- **Not commutative**: A×B ≠ B×A in general
- **AI connection**: Every neural network layer computes `output = input × weights + bias`, which is a matrix multiply.

### Transpose

- Flip rows and columns: element at position (i,j) moves to (j,i)
- A matrix of shape (2×3) becomes (3×2)

```
| 1  2  3 |ᵀ    | 1  4 |
| 4  5  6 |  =  | 2  5 |
                 | 3  6 |
```

- **AI connection**: Reshaping data for layer compatibility. Attention formula uses Qᵀ.

### Additional Key Concepts

**Eigenvalues and Eigenvectors**:
- A vector `v` is an eigenvector of matrix `A` if `Av = λv` (multiplying by A just scales it)
- `λ` is the eigenvalue
- **AI connection**: PCA (principal component analysis) uses eigenvectors for dimensionality reduction.

**Norms** (vector length):
- L2 norm: `||v|| = √(v₁² + v₂² + ... + vₙ²)`
- L1 norm: `||v|| = |v₁| + |v₂| + ... + |vₙ|`
- **AI connection**: Regularization (L1/L2) penalizes large weights to prevent overfitting.

**Resources**:
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Khan Academy — Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- [Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/)

---

## Level 4: Calculus

You don't need all of calculus — just enough to understand how models learn.

### Derivatives (Rate of Change)

- The derivative of `f(x)` tells you how fast `f` changes when `x` changes
- Notation: `f'(x)` or `df/dx`
- Common derivatives:
  - `f(x) = x²` → `f'(x) = 2x`
  - `f(x) = x³` → `f'(x) = 3x²`
  - `f(x) = eˣ` → `f'(x) = eˣ`
  - `f(x) = ln(x)` → `f'(x) = 1/x`
- **AI connection**: "How much does the loss change when I adjust this weight?" = derivative of loss with respect to the weight.

### Chain Rule

- For composed functions: if `y = f(g(x))`, then `dy/dx = f'(g(x)) × g'(x)`
- Example: `y = (3x + 1)²` → `dy/dx = 2(3x + 1) × 3 = 6(3x + 1)`
- **AI connection**: Backpropagation = the chain rule applied through every layer. This is how neural networks learn.

### Partial Derivatives

- When a function has multiple variables, take the derivative with respect to one while treating others as constants
- `f(x, y) = x²y + 3y` → `∂f/∂x = 2xy`, `∂f/∂y = x² + 3`
- **AI connection**: Each weight gets its own partial derivative. A model with 1B parameters computes 1B partial derivatives per training step.

### Gradient

- The gradient = vector of all partial derivatives: `∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`
- Points in the direction of steepest **increase**
- To minimize loss, go in the **opposite** direction (gradient descent)
- Update rule: `w_new = w_old − learning_rate × ∇L(w)`
- **AI connection**: This is the core of training. Every optimizer (SGD, Adam) is a variation of gradient descent.

### Minima and Maxima

- A minimum is where the derivative = 0 and the function curves upward
- Local minimum: lowest point in the neighborhood
- Global minimum: lowest point overall
- **AI connection**: Training seeks the minimum of the loss function. In practice, local minima are usually good enough.

**Resources**:
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [Khan Academy — Calculus](https://www.khanacademy.org/math/calculus-1)

---

## Level 5: Probability and Statistics

### Probability Basics

- Probability = number between 0 and 1
  - 0 = impossible, 1 = certain
- `P(A) = favorable outcomes / total outcomes`
- Complement: `P(not A) = 1 − P(A)`
- **AI connection**: Every model prediction is a probability distribution over possible outputs.

### Conditional Probability

- `P(A|B)` = probability of A given that B happened
- `P(A|B) = P(A and B) / P(B)`
- Example: P(rain | cloudy) = P(rain and cloudy) / P(cloudy)
- **AI connection**: Language models compute `P(next_token | previous_tokens)` — this is literally what GPT does at each step.

### Bayes' Theorem

```
P(A|B) = P(B|A) × P(A) / P(B)
```

- Updates prior beliefs with new evidence
- **Prior**: `P(A)` — what you believed before
- **Likelihood**: `P(B|A)` — how likely is the evidence given your belief
- **Posterior**: `P(A|B)` — updated belief after seeing evidence
- **AI connection**: Foundation of probabilistic reasoning. Bayesian methods are used in hyperparameter tuning and uncertainty estimation.

### Mean, Variance, Standard Deviation

- **Mean** (average): `μ = (1/n) Σ xᵢ`
- **Variance**: `σ² = (1/n) Σ (xᵢ − μ)²` — average squared distance from mean
- **Standard deviation**: `σ = √variance` — spread in original units
- **AI connection**: Used to read training metrics. Batch normalization normalizes using mean and variance.

### Distributions

- **Uniform**: all outcomes equally likely (e.g., fair die)
- **Normal (Gaussian)**: bell curve, defined by mean (μ) and std dev (σ)
  - 68% of data within 1σ, 95% within 2σ, 99.7% within 3σ
- **Bernoulli**: binary outcome (e.g., coin flip)
- **AI connection**: Weight initialization often uses normal distribution. Data assumptions affect model choice.

### Cross-Entropy

- Measures how different two probability distributions are
- Formula: `H(p, q) = −Σ p(x) × log(q(x))`
  - `p` = true distribution, `q` = predicted distribution
- Lower cross-entropy = better predictions
- Special case — binary cross-entropy: `−[y × log(ŷ) + (1−y) × log(1−ŷ)]`
- **AI connection**: The most common loss function for classification. When you see "loss" in training logs, it's usually cross-entropy.

**Resources**:
- [StatQuest (YouTube)](https://www.youtube.com/@statquest)
- [Seeing Theory](https://seeing-theory.brown.edu/)
- [Khan Academy — Statistics & Probability](https://www.khanacademy.org/math/statistics-probability)

---

## Study Order

```text
Level 1 (1-2 weeks) → Level 2 (2-3 weeks) → Level 3 (3-4 weeks) → Level 4 (2-3 weeks) → Level 5 (2-3 weeks)
```

- Start NLP concepts (Phase 2 of ai-study-guide) while working through Levels 3-5
- Start hands-on LLM work (Phase 3) anytime — practical experience and math reinforce each other

## Quick Reference Card

| Symbol | Meaning | Example |
| ------ | ------- | ------- |
| Σ | Sum | `Σᵢ xᵢ` = sum all x values |
| ∏ | Product | `∏ᵢ xᵢ` = multiply all x values |
| ∂ | Partial derivative | `∂f/∂x` = derivative w.r.t. x |
| ∇ | Gradient | `∇f` = vector of all partials |
| \|\|v\|\| | Norm (length) | `||v|| = √(Σ vᵢ²)` |
| · | Dot product | `a · b = Σ aᵢbᵢ` |
| ᵀ | Transpose | Flip rows ↔ columns |
| P(A\|B) | Conditional probability | Probability of A given B |
| μ | Mean | Average value |
| σ | Standard deviation | Spread of values |
| e | Euler's number | ≈ 2.718 |
| log | Logarithm | Inverse of exponent |
