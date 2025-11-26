Manual Walkthrough: The Differentiable Perceptron (Sigmoid Neuron)

Introduction: What I learnt

This tutorial bridges the gap between abstract neural network theory and concrete implementation. By dissecting the simplest differentiable unit of Deep Learning—the Sigmoid Neuron—we will manually trace the path of "learning" from a raw error signal back to a specific weight update.

Specifically, this document covers:

The Mechanics of Learning: How Forward Propagation (prediction) and Backward Propagation (correction) mathematically interact via the Chain Rule.

Batch Processing: Transitioning from simple vector logic to Matrix Calculus ($Z = X \cdot W + b$) to handle entire datasets efficiently.

Manual Verification: Calculating gradients by hand to demystify what "Autograd" engines do behind the scenes.

Implementation: Translating these manual derivation steps directly into PyTorch code to verify our arithmetic.

1. Intuitive Analogy: "The Coffee Log" (Batch Data)

To understand how a model learns general rules (rather than just reacting to one event), we need a dataset. Imagine looking back at your purchase history for three different mornings.

The Goal: Find a set of preferences (Weights) that explains all these decisions reasonably well.

The Dataset (Matrix $X$)

We have 3 examples (rows). Each has 2 features (columns).

Scenario

Feature 1: Caffeine ($x_1$)

Feature 2: Price ($x_2$)

Decision ($y$)

Reasoning

1. Desperate Monday

1.0 (Strong)

2.0 (Expensive)

1 (Buy)

"I needed energy, ignored price."

2. Tourist Trap

0.2 (Weak)

2.0 (Expensive)

0 (Skip)

"Weak AND expensive? No way."

3. The Daily Grind

1.0 (Strong)

0.5 (Cheap)

1 (Buy)

"Strong and cheap. Easy yes."

The Initial Weights ($W$) - "Conflicted Personality"

Our model starts with the same initial guess as before.

Caffeine ($w_1 = 0.5$): "I kind of like caffeine."

Price ($w_2 = -0.5$): "I really hate paying money."

Bias ($b = 0.0$): Neutral mood.

The Conflict

In Scenario 1, the high price ($2.0$) multiplied by the hate for price ($-0.5$) creates a strong negative signal ($-1.0$).

The model currently thinks Monday's coffee is a bad deal.

Learning Goal: The model needs to realize that Caffeine is more important than Price to satisfy all 3 examples (since we bought 2/3 coffees).

2. Theoretical Foundation: Matrix Notation

To process multiple examples at once, we switch from vectors ($x$) to matrices ($X$).

2.1 The Computational Graph (Batch Version)

Linear Aggregation ($Z$):

Z = X \cdot W + b

$X$ is shape $(3, 2)$.

$W$ is shape $(2, 1)$.

$b$ is a scalar (broadcasted to shape $(3, 1)$).

Result $Z$ is shape $(3, 1)$ (One score for each coffee).

Activation ($A$): Applied element-wise.

A = \sigma(Z) = \frac{1}{1 + e^{-Z}}

Loss ($L$): Average Mean Squared Error across $N$ examples.

L = \frac{1}{2N} \sum (A - Y)^2

3. Experimental Setup (The Matrices)

We translate our "Coffee Log" into PyTorch-style tensors.

Input Matrix ($X$):

$$X = \begin{bmatrix} 
1.0 & 2.0 \\ 
0.2 & 2.0 \\ 
1.0 & 0.5 
\end{bmatrix}$$

Target Vector ($Y$):

$$Y = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$

Weights ($W$) & Bias ($b$):

$$W = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}, \quad b = 0.0$$

4. Forward Pass (Matrix Multiplication)

Step A: Linear Aggregation ($Z = X \cdot W + b$)

We perform the dot product for all 3 days simultaneously.

$$Z = \begin{bmatrix} 
1.0 & 2.0 \\ 
0.2 & 2.0 \\ 
1.0 & 0.5 
\end{bmatrix} 
\cdot 
\begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} 
+ 0.0$$

Row 1 (Monday): $(1.0 \cdot 0.5) + (2.0 \cdot -0.5) = 0.5 - 1.0 = \mathbf{-0.5}$
Row 2 (Trap): $(0.2 \cdot 0.5) + (2.0 \cdot -0.5) = 0.1 - 1.0 = \mathbf{-0.9}$
Row 3 (Daily): $(1.0 \cdot 0.5) + (0.5 \cdot -0.5) = 0.5 - 0.25 = \mathbf{0.25}$

$$Z = \begin{bmatrix} -0.5 \\ -0.9 \\ 0.25 \end{bmatrix}$$

Step B: Activation ($A = \sigma(Z)$)

We apply the sigmoid function to each score.

$\sigma(-0.5) \approx 0.3775$

$\sigma(-0.9) \approx 0.2890$

$\sigma(0.25) \approx 0.5621$

$$A = \begin{bmatrix} 0.3775 \\ 0.2890 \\ 0.5621 \end{bmatrix}$$

Step C: Loss Calculation ($L$)

Compare Predictions ($A$) vs Reality ($Y$).

Monday: $(0.3775 - 1)^2 = (-0.6225)^2 \approx 0.387$

Trap: $(0.2890 - 0)^2 = (0.2890)^2 \approx 0.083$

Daily: $(0.5621 - 1)^2 = (-0.4379)^2 \approx 0.191$

$$L_{total} = \frac{1}{2 \times 3} (0.387 + 0.083 + 0.191) = \frac{0.661}{6} \approx \mathbf{0.110}$$

5. Backward Pass (Matrix Gradients)

We need to find the "Average Direction" to move the weights to satisfy all three days.

Step A: Gradient w.r.t. Activation ($\nabla A$)

$$\nabla A = (A - Y)$$

$$\nabla A = \begin{bmatrix} 0.3775 - 1 \\ 0.2890 - 0 \\ 0.5621 - 1 \end{bmatrix} 
= \begin{bmatrix} -0.6225 \\ 0.2890 \\ -0.4379 \end{bmatrix}$$

Interpretation:

Row 1 (neg): Prediction was too low. Push UP.

Row 2 (pos): Prediction was too high. Push DOWN.

Row 3 (neg): Prediction was too low. Push UP.

Step B: Gradient w.r.t. Z ($\delta$)

Element-wise multiplication: $\delta = \nabla A \odot \sigma'(Z)$.
Recall $\sigma'(z) = a(1-a)$.

Sigmoid Derivatives:

$0.3775 \cdot (1 - 0.3775) \approx 0.235$

$0.2890 \cdot (1 - 0.2890) \approx 0.205$

$0.5621 \cdot (1 - 0.5621) \approx 0.246$

Chain Rule:

Monday: $-0.6225 \cdot 0.235 \approx \mathbf{-0.146}$

Trap: $0.2890 \cdot 0.205 \approx \mathbf{0.059}$

Daily: $-0.4379 \cdot 0.246 \approx \mathbf{-0.108}$

$$\delta = \begin{bmatrix} -0.146 \\ 0.059 \\ -0.108 \end{bmatrix}$$

Step C: Gradient w.r.t. Weights ($\nabla W$)

This is the crucial matrix operation: $X^T \cdot \delta$.
We check how each feature contributed to the error across all examples.

$$\nabla W = \begin{bmatrix} 
1.0 & 0.2 & 1.0 \\ 
2.0 & 2.0 & 0.5 
\end{bmatrix} 
\cdot 
\begin{bmatrix} -0.146 \\ 0.059 \\ -0.108 \end{bmatrix}$$

For Caffeine ($w_1$):
$(1.0 \cdot -0.146) + (0.2 \cdot 0.059) + (1.0 \cdot -0.108)$
$= -0.146 + 0.0118 - 0.108 = \mathbf{-0.242}$

For Price ($w_2$):
$(2.0 \cdot -0.146) + (2.0 \cdot 0.059) + (0.5 \cdot -0.108)$
$= -0.292 + 0.118 - 0.054 = \mathbf{-0.228}$

(Note: We divide by $N=3$ usually, but for this manual trace we keep the sum or divide at the update step).

Step D: Gradient w.r.t Bias ($\nabla b$)

Sum of $\delta$.
$-0.146 + 0.059 - 0.108 = \mathbf{-0.195}$

6. Weight Update (Learning)

$\eta = 0.1$. Let's use the sums calculated above.

Update Caffeine Weight ($w_1$)

$$w_{1\_new} = 0.5 - 0.1(-0.242) = 0.5 + 0.0242 = \mathbf{0.5242}$$

Reasoning: Even though we bought 2/3 coffees, the model realized that Caffeine was present in both "Buy" scenarios. The gradient is negative (meaning "Loss goes down if Weight goes up"), so we increase the weight.

Update Price Weight ($w_2$)

$$w_{2\_new} = -0.5 - 0.1(-0.228) = -0.5 + 0.0228 = \mathbf{-0.4772}$$

Reasoning: The model realized that a high negative weight ($-0.5$) was causing too much error on Monday (Desperate Monday). It slightly reduces the penalty for price (makes it less negative) to accommodate that purchase.

Conclusion

By processing the matrix, the model learned to prioritize Caffeine more and penalize Price slightly less to fit the aggregated behavior of the user.

7. Code Verification & Execution Trace

The following Python script implements the matrices exactly as defined in Section 3 and executes one step of learning.

import torch
import torch.nn as nn

def verify_batch_perceptron():
    print("--- 1. Setup (The Coffee Log) ---")
    # Input Matrix (3 examples, 2 features)
    # Row 1: Desperate Monday, Row 2: Tourist Trap, Row 3: Daily Grind
    X = torch.tensor([
        [1.0, 2.0],  
        [0.2, 2.0],  
        [1.0, 0.5]   
    ])
    
    # Target Vector (The Purchase Decisions)
    Y = torch.tensor([
        [1.0], 
        [0.0], 
        [1.0]
    ])
    
    # Initialize Weights and Bias (Matches Section 3)
    # w1 = 0.5 (Caffeine), w2 = -0.5 (Price)
    W = torch.tensor([[0.5], [-0.5]], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    
    print(f"Initial Weights (W):\n{W.detach().numpy().T}")
    
    print("\n--- 2. Forward Pass ---")
    # Step A: Linear Aggregation Z = XW + b (Matches Section 4A)
    Z = torch.matmul(X, W) + b
    print(f"Z (Linear Scores):\n{Z.detach().numpy().T}")
    
    # Step B: Activation A = sigmoid(Z) (Matches Section 4B)
    Z.retain_grad()
    A = torch.sigmoid(Z)
    A.retain_grad()
    print(f"A (Predictions):\n{A.detach().numpy().T}")
    
    # Step C: Loss (Mean Squared Error)
    # Note: PyTorch MSELoss divides by N=3 by default
    loss_fn = nn.MSELoss() 
    loss = loss_fn(A, Y)
    print(f"Average Loss: {loss.item():.4f}")
    
    print("\n--- 3. Backward Pass ---")
    loss.backward()
    
    print("Gradient w.r.t Weights (grad_W):")
    print(W.grad.T)
    # Manual Check: 
    # In Section 5C, we calculated sums: w1_sum = -0.242, w2_sum = -0.228
    # PyTorch divides these by N=3.
    # Expected w1 grad = -0.242 / 3 = -0.0806
    
    print("\n--- 4. Weight Update (Learning) ---")
    lr = 0.1
    with torch.no_grad():
        W_new = W - lr * W.grad
        b_new = b - lr * b.grad
        print(f"New Weights:\n{W_new.numpy().T}")
        print(f"New Bias:\n{b_new.numpy()}")

if __name__ == "__main__":
    verify_batch_perceptron()
