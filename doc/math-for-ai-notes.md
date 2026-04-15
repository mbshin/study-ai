# Math for AI — Study Notes

Study notes following the math learning path from `ai-study-guide.md`.
Each level builds on the previous one, focused on what matters for AI/ML.

## Table of Contents

- [Level 1: Arithmetic and Pre-Algebra](#level-1-arithmetic-and-pre-algebra)
  - [Fractions, Decimals, Percentages](#fractions-decimals-percentages)
  - [Negative Numbers and Absolute Value](#negative-numbers-and-absolute-value)
  - [Exponents and Logarithms](#exponents-and-logarithms)
  - [Order of Operations](#order-of-operations)
  - [Basic Graphing](#basic-graphing)
- [Level 2: Algebra](#level-2-algebra)
  - [Variables and Equations](#variables-and-equations)
  - [Functions](#functions)
  - [Slopes and Lines](#slopes-and-lines)
  - [Systems of Equations](#systems-of-equations)
  - [Summation Notation](#summation-notation)
- [Level 3: Linear Algebra](#level-3-linear-algebra)
  - [Vectors](#vectors)
  - [Matrices](#matrices)
  - [Dot Product](#dot-product)
  - [Matrix Multiplication](#matrix-multiplication)
  - [Transpose](#transpose)
  - [Additional Key Concepts](#additional-key-concepts)
- [Level 4: Calculus](#level-4-calculus)
  - [Derivatives (Rate of Change)](#derivatives-rate-of-change)
  - [Chain Rule](#chain-rule)
  - [Partial Derivatives](#partial-derivatives)
  - [Gradient](#gradient)
  - [Minima and Maxima](#minima-and-maxima)
- [Level 5: Probability and Statistics](#level-5-probability-and-statistics)
  - [Probability Basics](#probability-basics)
  - [Conditional Probability](#conditional-probability)
  - [Bayes' Theorem](#bayes-theorem)
  - [Mean, Variance, Standard Deviation](#mean-variance-standard-deviation)
  - [Distributions](#distributions)
  - [Cross-Entropy](#cross-entropy)
- [Study Order](#study-order)
- [Quick Reference Card](#quick-reference-card)

---

## Level 1: Arithmetic and Pre-Algebra

### Fractions, Decimals, Percentages

#### What Is a Fraction?

A fraction represents a part of a whole: **numerator / denominator**

```text
  3   ← numerator (how many parts you have)
 ---
  4   ← denominator (how many equal parts the whole is divided into)
```

- `3/4` means "3 out of 4 equal parts"
- The denominator can never be 0 (division by zero is undefined)
- A fraction where numerator = denominator equals 1: `5/5 = 1`

#### Types of Fractions

| Type | Definition | Example |
| ---- | ---------- | ------- |
| Proper | numerator < denominator | `3/4` (less than 1) |
| Improper | numerator ≥ denominator | `7/4` (greater than or equal to 1) |
| Mixed number | whole part + fraction | `1 3/4` (= `7/4`) |
| Unit fraction | numerator = 1 | `1/4`, `1/8` |

Converting mixed ↔ improper:

```text
1 3/4 → (1 × 4 + 3) / 4 = 7/4     mixed → improper
7/4   → 7 ÷ 4 = 1 remainder 3 → 1 3/4   improper → mixed
```

#### Equivalent Fractions and Simplifying

Multiplying or dividing both numerator and denominator by the same number gives an equivalent fraction:

```text
1/2 = 2/4 = 3/6 = 4/8 = 50/100
```

Simplify by dividing by the **greatest common divisor (GCD)**:

```text
12/18 → GCD(12, 18) = 6 → 12÷6 / 18÷6 = 2/3
```

**AI connection**: Normalizing data is the same idea — scale numerator and denominator so values become comparable.

#### Fraction Arithmetic

##### Addition and Subtraction (need common denominator)

```text
1/3 + 1/4 = 4/12 + 3/12 = 7/12
5/6 − 1/3 = 5/6 − 2/6  = 3/6 = 1/2
```

Steps: (1) find the least common denominator (LCD), (2) convert each fraction, (3) add/subtract numerators.

##### Multiplication (straight across)

```text
2/3 × 4/5 = (2×4) / (3×5) = 8/15
```

Shortcut — cross-cancel before multiplying:

```text
4/9 × 3/8 = (4×3)/(9×8) → cancel 4 and 8 (÷4), cancel 3 and 9 (÷3)
           = 1/3 × 1/2 = 1/6
```

##### Division (multiply by the reciprocal)

```text
2/3 ÷ 4/5 = 2/3 × 5/4 = 10/12 = 5/6
```

The reciprocal flips numerator and denominator: reciprocal of `4/5` is `5/4`.

#### Decimals

Decimals are fractions with denominators that are powers of 10:

```text
0.75 = 75/100 = 3/4
0.1  = 1/10
0.333... = 1/3  (repeating decimal)
```

Converting fraction → decimal: divide numerator by denominator.

```text
3/8 = 3 ÷ 8 = 0.375
1/3 = 1 ÷ 3 = 0.333...
```

##### Floating-Point Precision

Computers store decimals in binary (base 2), and many simple decimals can't be represented exactly:

```text
0.1 + 0.2 = 0.30000000000000004   (in most programming languages)
```

This isn't a bug — it's a fundamental limitation of binary floating-point. Fractions like `1/3` and `1/10` become infinite repeating sequences in binary.

**AI connection**: This is why ML frameworks use `float32` or `float16` — there's always a precision/performance tradeoff. Numerical instability in loss computation often traces back to floating-point limits.

#### Percentages

A percentage is a fraction with denominator 100:

```text
75% = 75/100 = 0.75
```

Converting between all three forms:

```text
Fraction → Decimal:    3/4 = 3 ÷ 4 = 0.75
Decimal → Percentage:  0.75 × 100 = 75%
Percentage → Decimal:  75% ÷ 100 = 0.75
Percentage → Fraction: 75% = 75/100 = 3/4
```

Common values worth memorizing:

| Fraction | Decimal | Percentage |
| -------- | ------- | ---------- |
| `1/2` | 0.5 | 50% |
| `1/3` | 0.333... | 33.3% |
| `1/4` | 0.25 | 25% |
| `1/5` | 0.2 | 20% |
| `1/8` | 0.125 | 12.5% |
| `1/10` | 0.1 | 10% |

#### Why Fractions Matter in AI/ML

##### 1. Probabilities Are Fractions

Every probability is a number between 0 and 1 — a fraction of certainty:

```text
P(cat) = 0.85  → model is 85% confident this is a cat
P(dog) = 0.10  → 10% chance it's a dog
P(bird) = 0.05 → 5% chance it's a bird
Total = 1.00   → probabilities must sum to 1 (the whole)
```

This "sum to 1" constraint is the same as "all parts must equal the whole" in fractions.

##### 2. Train/Validation/Test Splits

Datasets are divided into fractions:

```text
Training:    70% = 7/10 of data
Validation:  15% = 3/20 of data
Test:        15% = 3/20 of data
Total:      100% = 1
```

##### 3. Learning Rate and Hyperparameters

Learning rates are small fractions that control how much the model adjusts per step:

```text
lr = 0.001 = 1/1000
```

Too large (1/10) → unstable training, overshooting. Too small (1/1,000,000) → painfully slow learning.

##### 4. Metrics Are Ratios

Accuracy, precision, recall — all fractions:

```text
Accuracy  = correct predictions / total predictions
Precision = true positives / (true positives + false positives)
Recall    = true positives / (true positives + false negatives)
F1 score  = 2 × (precision × recall) / (precision + recall)
```

##### 5. Batch Size as a Fraction of Dataset

```text
Dataset: 10,000 samples
Batch size: 32
Batches per epoch: 10,000 / 32 = 312.5 → 313 batches
Each batch = 32/10,000 = 0.32% of the data
```

#### Worked Examples

**Example 1**: A model predicts 847 correct out of 1000 samples. What's the accuracy?

```text
847/1000 = 0.847 = 84.7%
```

**Example 2**: Combine two precision scores: model A gets 3/4 on dataset X, model B gets 5/6 on dataset Y. Average precision?

```text
Average = (3/4 + 5/6) / 2
        = (9/12 + 10/12) / 2
        = (19/12) / 2
        = 19/24
        ≈ 0.792 = 79.2%
```

**Example 3**: Learning rate decay — reduce lr by 1/10 every 5 epochs.

```text
Epoch 0:  lr = 0.01     = 1/100
Epoch 5:  lr = 0.001    = 1/1,000
Epoch 10: lr = 0.0001   = 1/10,000
Epoch 15: lr = 0.00001  = 1/100,000
```

Each step: multiply by `1/10` (= divide by 10).

**Example 4**: Floating-point comparison trap

```python
# This can fail due to floating-point imprecision:
if 0.1 + 0.2 == 0.3:    # False!

# Safe comparison using tolerance:
if abs((0.1 + 0.2) - 0.3) < 1e-9:    # True
```

In ML code, use `torch.allclose()` or `np.isclose()` instead of `==` for decimal comparisons.

#### Quick Summary

| Concept | Key Idea | AI Use |
| ------- | -------- | ------ |
| Fraction = part/whole | `numerator / denominator` | Probabilities, metrics, splits |
| Equivalent fractions | `1/2 = 2/4 = 50/100` | Normalization, scaling |
| Fraction → decimal | Divide top by bottom | All numerical computation |
| Percentages | Fraction with denominator 100 | Accuracy, confidence, splits |
| Sum to 1 | All parts = the whole | Probability distributions |
| Floating-point limits | `0.1 + 0.2 ≠ 0.3` | Numerical stability, tolerance checks |

**AI connection summary**: Fractions are the language of probability. Every model prediction, every metric, every data split, and every hyperparameter is a fraction. Understanding how parts relate to wholes — and how computers approximate them — is foundational to reading and debugging ML systems.

### Negative Numbers and Absolute Value

#### The Number Line

```text
  ← negative          positive →
 ───┼───┼───┼───┼───┼───┼───┼───┼───
   -4  -3  -2  -1   0   1   2   3   4
```

- Numbers to the left of 0 are negative, to the right are positive
- Zero is neither positive nor negative
- For any number `a`: exactly one of `a > 0`, `a = 0`, `a < 0` is true

#### Negative Number Arithmetic

```text
Addition:
  3 + (−5) = −2       (move 5 left from 3)
 −3 + (−2) = −5       (both negative → more negative)
 −3 + 5    =  2       (move 5 right from −3)

Subtraction (add the opposite):
  3 − 5    = 3 + (−5) = −2
  3 − (−5) = 3 + 5    =  8   (subtracting negative = adding)

Multiplication/Division (sign rules):
  positive × positive = positive     3 × 2 = 6
  negative × negative = positive    −3 × (−2) = 6
  positive × negative = negative     3 × (−2) = −6
  negative × positive = negative    −3 × 2 = −6
```

Quick rule: **same signs → positive, different signs → negative**.

#### Absolute Value

Absolute value `|x|` = distance from zero on the number line (always non-negative):

```text
|5|  = 5     (5 is 5 steps from zero)
|−5| = 5     (−5 is also 5 steps from zero)
|0|  = 0
```

Formal definition:

```text
|x| = x    if x ≥ 0
|x| = −x   if x < 0
```

Properties:

- `|x| ≥ 0` always (never negative)
- `|x| = 0` only when `x = 0`
- `|−x| = |x|` (sign doesn't matter)
- `|a × b| = |a| × |b|`
- `|a + b| ≤ |a| + |b|` (triangle inequality)

#### Why Negatives and Absolute Value Matter in AI/ML

##### 1. Gradients Are Signed

Gradients tell the model which direction to adjust:

```text
∂L/∂w = −2.5   → loss decreases when w increases → increase w
∂L/∂w =  1.8   → loss increases when w increases → decrease w
∂L/∂w =  0.0   → at a minimum (or maximum) → no change needed
```

The sign is the direction, the magnitude (absolute value) is the step size.

##### 2. Loss Functions Use Absolute Value

Mean Absolute Error (MAE / L1 Loss):

```text
MAE = (1/n) Σ |yᵢ − ŷᵢ|
```

Compared to Mean Squared Error (MSE / L2 Loss):

```text
MSE = (1/n) Σ (yᵢ − ŷᵢ)²
```

| Property | MAE (L1) | MSE (L2) |
| -------- | -------- | -------- |
| Outlier sensitivity | Robust (linear penalty) | Sensitive (squared penalty) |
| Gradient at error=0 | Undefined (not smooth) | Zero (smooth) |
| When to use | Noisy data, outliers | Clean data, precise predictions |

##### 3. Activation Functions Cross Zero

ReLU — the most common activation — relies on the sign of x:

```text
ReLU(x) = max(0, x)

 x = −3 → ReLU = 0    (negative → killed)
 x =  0 → ReLU = 0
 x =  2 → ReLU = 2    (positive → passed through)
```

##### 4. Weight Initialization Around Zero

Neural network weights are initialized with small values centered around zero:

```text
w ~ Normal(mean=0, std=0.01)
→ mix of small positive and negative values: −0.012, 0.008, −0.003, 0.015, ...
```

If all weights were positive (or all negative), the network would learn poorly — it needs both signs for expressive power.

##### 5. The Triangle Inequality in Embeddings

The triangle inequality `|a + b| ≤ |a| + |b|` generalizes to vector distances:

```text
distance(A, C) ≤ distance(A, B) + distance(B, C)
```

This property is what makes embedding spaces work — similar items are close, dissimilar items are far, and distances behave geometrically.

#### Worked Examples

**Example 1**: A model predicts ŷ = 2.3, true value y = 5.0. Compute the error.

```text
Error = y − ŷ = 5.0 − 2.3 = 2.7     (positive → model under-predicted)
|Error| = |2.7| = 2.7                 (absolute error)
Error² = 2.7² = 7.29                  (squared error)
```

**Example 2**: Gradient update step.

```text
Current weight:  w = 0.5
Gradient:        ∂L/∂w = −1.2  (negative → increase w to reduce loss)
Learning rate:   lr = 0.1

w_new = w − lr × gradient
      = 0.5 − 0.1 × (−1.2)
      = 0.5 + 0.12
      = 0.62
```

Note: subtracting a negative gradient = adding → weight increases.

**Example 3**: Comparing MAE vs MSE on an outlier.

```text
Predictions: [2.0, 3.0, 2.5, 100.0]  (last one is an outlier)
True values:  [2.1, 2.9, 2.6, 3.0]

Errors:       [−0.1, 0.1, −0.1, 97.0]
|Errors|:     [0.1, 0.1, 0.1, 97.0]
Errors²:      [0.01, 0.01, 0.01, 9409.0]

MAE = (0.1 + 0.1 + 0.1 + 97.0) / 4 = 24.3
MSE = (0.01 + 0.01 + 0.01 + 9409.0) / 4 = 2352.3
```

MSE is ~100x larger because squaring amplifies the outlier. Use MAE when outliers should not dominate.

#### Quick Summary

| Concept | Key Idea | AI Use |
| ------- | -------- | ------ |
| Negative numbers | Direction on number line | Gradient sign = update direction |
| Sign rules | Same signs → +, different → − | Weight updates, error direction |
| Absolute value | Distance from zero, always ≥ 0 | MAE/L1 loss, regularization |
| Triangle inequality | `\|a + b\| ≤ \|a\| + \|b\|` | Embedding distance properties |

**AI connection summary**: Negatives encode direction — whether a gradient says "go up" or "go down," whether an error is an over- or under-prediction. Absolute value strips direction and keeps magnitude, which is exactly what loss functions need to measure "how wrong" without caring "which direction wrong."

### Exponents and Logarithms

#### What Is an Exponent?

An exponent says "multiply the base by itself this many times":

```text
base^exponent = result

2³ = 2 × 2 × 2 = 8        "2 multiplied by itself 3 times"
5² = 5 × 5 = 25            "5 squared"
10⁴ = 10 × 10 × 10 × 10 = 10,000
```

Terminology: in `2³`, **2** is the base, **3** is the exponent (also called power or index).

#### Special Exponent Values

```text
a¹ = a              any number to the power 1 is itself
a⁰ = 1              any nonzero number to the power 0 is 1 (by convention)
0⁰ = undefined       (debated — often treated as 1 in combinatorics)
```

Why `a⁰ = 1`? Follow the pattern:

```text
2³ = 8
2² = 4     (÷2)
2¹ = 2     (÷2)
2⁰ = 1     (÷2)  ← pattern demands it
```

#### Negative and Fractional Exponents

```text
Negative exponent = reciprocal (flip to denominator):
  a⁻¹ = 1/a           2⁻¹ = 1/2 = 0.5
  a⁻ⁿ = 1/aⁿ          2⁻³ = 1/2³ = 1/8 = 0.125

Fractional exponent = root:
  a^(1/2) = √a         9^(1/2) = √9 = 3
  a^(1/3) = ³√a        8^(1/3) = ³√8 = 2
  a^(m/n) = ⁿ√(aᵐ)     4^(3/2) = √(4³) = √64 = 8
```

#### Exponent Rules

These are the "algebra of exponents" — used constantly when simplifying formulas:

| Rule | Formula | Example |
| ---- | ------- | ------- |
| Product | `aᵐ × aⁿ = aᵐ⁺ⁿ` | `2³ × 2⁴ = 2⁷ = 128` |
| Quotient | `aᵐ / aⁿ = aᵐ⁻ⁿ` | `2⁵ / 2² = 2³ = 8` |
| Power of a power | `(aᵐ)ⁿ = aᵐⁿ` | `(2³)² = 2⁶ = 64` |
| Product to a power | `(ab)ⁿ = aⁿbⁿ` | `(3×2)² = 3² × 2² = 36` |
| Quotient to a power | `(a/b)ⁿ = aⁿ/bⁿ` | `(3/2)² = 9/4` |

#### The Shape of Exponential Growth

```text
y
64|                              ·
32|                          ·
16|                      ·
 8|                  ·
 4|              ·
 2|          ·
 1|· · · ·
 0|──────────────────────────────→ x
  0  1  2  3  4  5  6
```

Key features:

- Starts slow, then explodes (opposite of logarithms)
- `2¹⁰ = 1,024` — already over a thousand
- `2²⁰ = 1,048,576` — over a million
- `2³⁰ = 1,073,741,824` — over a billion
- Always positive for positive base: `aˣ > 0` when `a > 0`

#### Euler's Number `e`

`e ≈ 2.71828...` — the most important base in mathematics and ML.

Why is `e` special? It's the only base where the exponential function is its own derivative:

```text
f(x) = eˣ  →  f'(x) = eˣ
```

No other base has this property. This makes calculus with `eˣ` clean and simple, which is why it dominates ML formulas.

Definition (one of many equivalent forms):

```text
e = lim(n→∞) (1 + 1/n)ⁿ

n=1:     (1 + 1)¹     = 2.000
n=10:    (1 + 0.1)¹⁰  = 2.594
n=100:   (1 + 0.01)¹⁰⁰ = 2.705
n=1000:  (1 + 0.001)¹⁰⁰⁰ = 2.717
n→∞:                    → 2.71828...
```

#### Why Exponents Are Everywhere in AI/ML

##### 1. Softmax — Converting Scores to Probabilities

Softmax uses `eˣ` to turn raw logits into probabilities:

```text
logits:  [2.0, 1.0, 0.1]
eˣ:      [e²·⁰, e¹·⁰, e⁰·¹] = [7.39, 2.72, 1.11]
sum:     7.39 + 2.72 + 1.11 = 11.22
softmax: [7.39/11.22, 2.72/11.22, 1.11/11.22] = [0.659, 0.242, 0.099]
```

Why `eˣ` and not just the raw numbers?

- `eˣ` is always positive (probabilities must be ≥ 0)
- `eˣ` amplifies differences (larger logits get much more probability)
- `eˣ` is smooth and differentiable (gradients work well)

##### 2. Exponential Decay in Learning Rate Schedules

```text
lr(t) = lr₀ × decay^t

lr₀ = 0.01, decay = 0.95:
  Epoch 0:  0.01 × 0.95⁰  = 0.01000
  Epoch 10: 0.01 × 0.95¹⁰ = 0.00598
  Epoch 50: 0.01 × 0.95⁵⁰ = 0.00077
  Epoch 100: 0.01 × 0.95¹⁰⁰ = 0.00006
```

The learning rate shrinks exponentially — big steps early, fine-tuning later.

##### 3. Parameter Count Grows Exponentially with Layers

A simple model with `n` binary features has `2ⁿ` possible input combinations:

```text
Features:     2ⁿ combinations
n=10:         1,024
n=20:         1,048,576
n=100:        1.27 × 10³⁰  (more than atoms in a human body)
```

This "curse of dimensionality" is why deep learning needs so much data.

##### 4. Vanishing and Exploding Gradients

When backpropagating through `L` layers, gradients get multiplied:

```text
gradient ≈ (weight)^L

If weight = 0.5 (< 1):   0.5¹⁰ = 0.00098   → vanishing (too small to learn)
If weight = 2.0 (> 1):   2.0¹⁰ = 1024       → exploding (unstable training)
If weight = 1.0 (= 1):   1.0¹⁰ = 1          → perfect (this is the goal)
```

This is why weight initialization, batch normalization, and residual connections matter — they keep the effective multiplier near 1.

##### 5. Scientific Notation (Powers of 10)

ML uses scientific notation constantly for very large and very small numbers:

```text
1e-3  = 10⁻³ = 0.001       (typical learning rate)
1e-8  = 10⁻⁸ = 0.00000001  (Adam epsilon)
1e6   = 10⁶  = 1,000,000   (1M parameters)
1e9   = 10⁹  = 1,000,000,000 (1B parameters)
1.76e11 = 176,000,000,000   (GPT-3 parameter count)
```

#### Worked Examples

**Example 1**: Simplify `(2³ × 2⁴) / 2⁵`

```text
= 2³⁺⁴ / 2⁵     product rule
= 2⁷ / 2⁵
= 2⁷⁻⁵           quotient rule
= 2² = 4
```

**Example 2**: Compute softmax temperature scaling.

Temperature `T` controls softmax sharpness by dividing logits by `T`:

```text
logits = [3.0, 1.0]

T=1.0 (normal):    e³·⁰/e¹·⁰ = 20.09/2.72 → softmax ≈ [0.88, 0.12]
T=0.5 (sharp):     e⁶·⁰/e²·⁰ = 403.4/7.39 → softmax ≈ [0.98, 0.02]
T=2.0 (smooth):    e¹·⁵/e⁰·⁵ = 4.48/1.65  → softmax ≈ [0.73, 0.27]
```

Lower temperature → more confident. Higher temperature → more uniform.

**Example 3**: How many epochs until learning rate drops below 0.001?

```text
lr₀ = 0.01, decay = 0.9 per epoch
0.01 × 0.9ᵗ < 0.001
0.9ᵗ < 0.1
t × log(0.9) < log(0.1)
t > log(0.1) / log(0.9) = −2.303 / −0.105 ≈ 21.9

Answer: after 22 epochs
```

**Example 4**: Vanishing gradient — how small is the gradient after 20 layers?

```text
If each layer multiplies gradient by 0.8:
0.8²⁰ = 0.0115

The gradient reaching the first layer is only 1.15% of the final layer's gradient.
Almost no learning signal gets through — this is vanishing gradient.
```

#### Quick Summary

| Concept | Formula | AI Use |
| ------- | ------- | ------ |
| Exponent = repeated multiplication | `aⁿ = a × a × ... × a` | Foundation for eˣ, softmax |
| Negative exponent = reciprocal | `a⁻ⁿ = 1/aⁿ` | Scientific notation (1e-3) |
| Euler's number | `e ≈ 2.718`, `d/dx eˣ = eˣ` | Softmax, loss functions |
| Exponential growth/decay | `y = a × bᵗ` | LR schedules, gradient flow |
| Powers of 2 | `2¹⁰ ≈ 1K, 2²⁰ ≈ 1M, 2³⁰ ≈ 1B` | Memory, parameter counting |

**AI connection summary**: Exponents define how neural networks compute (softmax), how they fail (vanishing/exploding gradients), and how they scale (parameter counts). The number `e` is the default base because `eˣ` is its own derivative — making gradient computation clean. Every time you see `exp()` in ML code, you're seeing exponents at work.

#### What Is a Logarithm?

A logarithm answers: **"What exponent do I need?"**

```
If    bˣ = y
Then  log_b(y) = x
```

The exponent and logarithm are inverses of each other:

```
2³ = 8      ↔  log₂(8) = 3     "2 to what power gives 8?  → 3"
10² = 100   ↔  log₁₀(100) = 2  "10 to what power gives 100? → 2"
e¹ = e      ↔  ln(e) = 1       "e to what power gives e?   → 1"
```

Inverse relationship — these always hold:
- `log_b(bˣ) = x` (log undoes exponent)
- `b^(log_b(x)) = x` (exponent undoes log)

#### Three Common Bases

| Notation | Base | Name | Where You See It |
| -------- | ---- | ---- | ---------------- |
| `log₂(x)` | 2 | Binary logarithm | Information theory, bits |
| `log(x)` or `log₁₀(x)` | 10 | Common logarithm | Decibels, pH, order-of-magnitude |
| `ln(x)` or `logₑ(x)` | e ≈ 2.718 | Natural logarithm | Calculus, ML loss functions, gradients |

In ML papers and code, `log` almost always means **natural log (ln)** unless stated otherwise.

#### Logarithm Properties

These turn multiplication into addition — the core reason logs are useful:

```
log(a × b) = log(a) + log(b)     Product rule
log(a / b) = log(a) − log(b)     Quotient rule
log(aⁿ)    = n × log(a)          Power rule
log_b(1)   = 0                    (b⁰ = 1)
log_b(b)   = 1                    (b¹ = b)
```

Change of base — convert between any two bases:

```
log_b(x) = log_a(x) / log_a(b)
```

Example: `log₂(8) = ln(8) / ln(2) = 2.079 / 0.693 = 3`

#### The Shape of log(x)

```
y
3 |                              ·····
2 |                  ·····
1 |          ·····
0 |----·------------------------------→ x
  0   1   2   3   4   5   ...
-1|·
-2|·
```

Key features:
- Domain: `x > 0` only (you cannot take log of zero or negative numbers)
- `log(1) = 0` — the curve crosses the x-axis at x=1
- Grows slowly — `log(1,000,000) = 6` when base 10
- As x → 0⁺, log(x) → −∞
- As x → ∞, log(x) → ∞ (but very slowly)

Compared to its inverse `eˣ`:
- `eˣ` grows explosively fast (exponential growth)
- `log(x)` grows painfully slowly (logarithmic growth)
- They are mirror images across the line `y = x`

#### Why Logarithms Are Everywhere in AI/ML

**1. Cross-Entropy Loss (Log Loss)**

The standard loss function for classification:

```
L = −Σ yᵢ × log(ŷᵢ)
```

Why log? Because when the model is confident and **wrong**, `log(ŷ)` produces a very large penalty:

```
Predicted ŷ   log(ŷ)     −log(ŷ) (penalty)
1.0            0.0        0.0      ← perfect, no penalty
0.9           −0.105      0.105    ← small penalty
0.5           −0.693      0.693    ← moderate penalty
0.1           −2.303      2.303    ← big penalty
0.01          −4.605      4.605    ← huge penalty
0.001         −6.908      6.908    ← extreme penalty
```

This property makes the model learn fastest when it's most wrong.

**2. Softmax and Log-Softmax**

Softmax converts raw scores (logits) into probabilities using `eˣ`:

```
softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ
```

Log-softmax = `log(softmax(z))` is used in practice because:
- Numerically more stable (avoids overflow from large `eˣ` values)
- Combines naturally with NLL loss: `log(softmax) + NLL = cross-entropy`

**3. Information Theory — Bits and Surprisal**

- **Surprisal** (self-information): `I(x) = −log₂(P(x))`
  - Common event (P=0.99) → low surprise (0.014 bits)
  - Rare event (P=0.01) → high surprise (6.64 bits)
- **Entropy**: `H = −Σ P(x) × log₂(P(x))` = average surprisal
  - Measures uncertainty in a distribution
  - Maximum when all outcomes equally likely
- **Perplexity**: `2^H` — used to evaluate language models
  - Perplexity of 10 ≈ "the model is as confused as if choosing between 10 equally likely options"

**4. Log Scale for Large Ranges**

When values span many orders of magnitude, log compresses them:

```
Dataset sizes:    100, 10,000, 1,000,000
In log₁₀ scale:  2,   4,      6
```

Used in: learning rate schedules (1e-5 to 1e-1), hyperparameter search, plotting training curves.

**5. Log Probabilities (Log-Probs)**

In LLMs, probabilities are stored and computed as log-probabilities:

```
P(token) = 0.0001     → log P = −9.21
P(token) = 0.95       → log P = −0.051
```

Why?
- Multiplying many small probabilities → underflow (too close to zero for floating point)
- Adding log-probs avoids this: `log(a × b × c) = log(a) + log(b) + log(c)`
- This is why LLM APIs return `logprobs` not raw probabilities

**6. The Derivative of log(x)**

```
d/dx [ln(x)] = 1/x
```

This means log's gradient is large when x is small and shrinks as x grows. In cross-entropy loss `−log(ŷ)`:
- When prediction ŷ is near 0 (very wrong): gradient ≈ `1/ŷ` → huge, model learns fast
- When prediction ŷ is near 1 (correct): gradient is small → model doesn't over-adjust

This automatic gradient scaling is why log-based losses train so well.

#### Worked Examples

**Example 1**: Simplify `log₂(32)`

```
32 = 2⁵  →  log₂(32) = 5
```

**Example 2**: Expand `ln(x²y/z)`

```
ln(x²y/z) = ln(x²y) − ln(z)         quotient rule
           = ln(x²) + ln(y) − ln(z)  product rule
           = 2ln(x) + ln(y) − ln(z)  power rule
```

**Example 3**: Cross-entropy loss for a single sample

True label: class 1 (y = [1, 0, 0]). Model predicts ŷ = [0.7, 0.2, 0.1].

```
L = −[1×log(0.7) + 0×log(0.2) + 0×log(0.1)]
  = −log(0.7)
  = −(−0.357)
  = 0.357
```

If model had predicted ŷ = [0.99, 0.005, 0.005]:

```
L = −log(0.99) = 0.01   ← much lower loss, model is more confident and correct
```

**Example 4**: Why log-probs avoid underflow

Computing probability of a 10-token sequence where each token has P = 0.1:

```
Direct:    0.1 × 0.1 × ... × 0.1 = 0.1¹⁰ = 1e-10  (getting dangerously small)
Log-space: log(0.1) × 10 = (−2.303) × 10 = −23.03   (perfectly fine number)
```

For a 1000-token sequence, direct multiplication would give `1e-1000` — impossible to represent in floating point. Log-space handles it easily: `−2303`.

#### Quick Summary

| Concept | Formula | AI Use |
| ------- | ------- | ------ |
| Log = inverse of exponent | `log_b(bˣ) = x` | Foundation of everything below |
| Product → sum | `log(ab) = log(a) + log(b)` | Log-probs, numerical stability |
| Cross-entropy loss | `−Σ y log(ŷ)` | Standard classification loss |
| Surprisal | `−log₂(P(x))` | Information content of events |
| Entropy | `−Σ P log(P)` | Uncertainty, perplexity |
| Derivative of ln(x) | `1/x` | Auto gradient scaling in loss |
| Log-softmax | `log(eᶻ / Σeᶻ)` | Stable probability computation |

**AI connection summary**: Logarithms are inescapable in ML — they define how models measure error (cross-entropy), compute probabilities (log-softmax), represent token likelihoods (log-probs), and quantify information (entropy/perplexity). Understanding `log` deeply is understanding how models learn.

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
