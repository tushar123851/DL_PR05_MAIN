<div align="center">

# 🎬 IMDB Movie Review Sentiment Analysis
## Deep Learning Practical — ANN, SimpleRNN, LSTM, Stacked LSTM & GRU

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=7B68EE&center=true&vCenter=true&width=950&lines=Deep+Learning+for+Sentiment+Analysis;ANN+%E2%86%92+SimpleRNN+%E2%86%92+LSTM+%E2%86%92+GRU;Understanding+BPTT+and+Gradient+Problems;Final+Best+Model%3A+GRU+%F0%9F%8F%86" alt="Typing SVG" />

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Evaluation-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br>

### ⭐ Complete implementation and explanation of all practical tasks ⭐

</div>

---

# 📌 Project Overview

This project performs **binary sentiment analysis** on the IMDB Movie Review dataset using deep-learning sequence models.

The purpose is not only to train a sentiment classifier, but also to understand the complete working of recurrent neural networks. The practical starts with text preprocessing and an ANN baseline, then studies a SimpleRNN manually and with Keras, investigates **Backpropagation Through Time (BPTT)** and gradient problems, and finally implements **LSTM**, **Stacked LSTM**, and **GRU**.

The final part compares every trained model using multiple evaluation metrics and investigates model behaviour on reviews of different lengths, misclassified examples, and custom unseen sentences.

---

# 🎯 Main Objectives

- Prepare raw text data for deep-learning models.
- Understand tokenization, OOV handling, padding and embeddings.
- Build a non-recurrent ANN baseline.
- Understand RNN forward propagation mathematically.
- Verify a manual RNN implementation against Keras.
- Understand one-to-one, one-to-many, many-to-one and many-to-many RNN structures.
- Train a SimpleRNN sentiment classifier.
- Understand BPTT.
- Study vanishing and exploding gradients.
- Demonstrate gradient clipping.
- Understand LSTM gates and cell state.
- Implement a manual LSTM forward pass.
- Train single and stacked LSTM models.
- Understand GRU update/reset gates.
- Train a GRU model.
- Compare RNN, LSTM and GRU gradient behaviour.
- Study the effect of hidden-unit size.
- Compare all final models.
- Analyze ROC curves and review-length performance.
- Inspect false positives and false negatives.
- Build a reusable sentiment prediction function.

---

# 🗂️ Dataset

The project uses the **IMDB Dataset of 50K Movie Reviews**.

🔗 [IMDB Dataset of 50K Movie Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The raw dataset contains:

| Feature | Description |
|---|---|
| `review` | Movie-review text |
| `sentiment` | Positive or negative sentiment |

The sentiment target is converted to:

```text
Negative → 0
Positive → 1
```

---

# 🔄 Complete Project Workflow

```text
Raw IMDB Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Review-Length Analysis
        │
        ▼
Label Encoding
        │
        ▼
Train / Test Split
        │
        ▼
Tokenizer Fit on Training Data
        │
        ▼
Text → Integer Sequences
        │
        ▼
Padding / Truncation
        │
        ▼
Embedding Layer
        │
        ├──────────────┐
        ▼              ▼
 ANN Baseline      Recurrent Models
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
         SimpleRNN    LSTM      GRU
                       │
                       ▼
                 Stacked LSTM
                       │
                       ▼
              Final Model Comparison
                       │
                       ▼
              Error & Length Analysis
                       │
                       ▼
                 Custom Inference
```

---

# 🧹 TASK 1 — Data Preparation & Text Preprocessing

## 🎯 Objective

Prepare the raw IMDB reviews for neural-network training while avoiding data leakage.

## 1.1 Load the Dataset

The raw CSV is loaded using Pandas.

```python
df = pd.read_csv("IMDB Dataset.csv")
```

Duplicate rows are removed so that repeated reviews do not unnecessarily influence the model.

## 1.2 Text Cleaning

A custom `clean_text()` function performs:

- Lowercase conversion
- HTML-tag removal
- Removal of unwanted characters
- Preservation of apostrophes
- Removal of repeated whitespace

Example pipeline:

```text
Raw Review
    ↓
Lowercase
    ↓
Remove HTML
    ↓
Remove Unwanted Characters
    ↓
Normalize Spaces
    ↓
Clean Review
```

The purpose is to reduce irrelevant variation while retaining the text needed for sentiment learning.

## 1.3 Review-Length Analysis

Review lengths are calculated before modeling.

Important statistics include:

- Mean length
- Median length
- 90th percentile
- 95th percentile

The 95th percentile is used to help select `MAXLEN`.

<div align="center">
<img src="plots/length_distribution.png" width="820">
</div>

Using a percentile-based sequence length avoids choosing a length arbitrarily while limiting excessive padding.

## 1.4 Label Encoding

```text
negative → 0
positive → 1
```

This converts the textual target into a binary target suitable for sigmoid classification.

## 1.5 Train/Test Split Before Tokenization

The dataset is split **before fitting the tokenizer**.

This is important because the vocabulary should be learned only from the training data.

```text
Clean Reviews
      ↓
Train/Test Split
      ↓
Fit Tokenizer on Training Reviews Only
```

This helps prevent **data leakage**.

## 1.6 Tokenization

The tokenizer uses:

```python
VOCAB_SIZE = 10000
oov_token = "<OOV>"
```

The tokenizer maps words to integer IDs.

Example:

```text
"this movie is amazing"
        ↓
[15, 27, 8, 341]
```

The OOV token handles words that are not part of the learned vocabulary.

## 1.7 Padding

Every review must have the same sequence length before it can be passed to the neural network.

The project uses:

```python
padding="pre"
truncating="pre"
```

Short sequences receive padding and overly long sequences are truncated.

### ✅ Task 1 Outcome

At the end of Task 1, the raw text has been transformed into fixed-length integer sequences ready for embedding and neural-network training.

---

# 🧠 TASK 2 — ANN Baseline & Manual RNN Forward Propagation

## 🎯 Objective

Create a non-recurrent baseline and understand the mathematical forward pass of a SimpleRNN.

## 2.1 ANN Baseline

Architecture:

```text
Input Sequence
      ↓
Embedding(10000, 32)
      ↓
Flatten
      ↓
Dense(64, ReLU)
      ↓
Dense(1, Sigmoid)
      ↓
Positive / Negative
```

The ANN baseline provides a reference point before recurrent architectures are introduced.

The embedding layer converts each integer word ID into a learned dense vector.

## 2.2 Why Use a Baseline?

A baseline answers an important question:

> Does recurrent sequence modeling actually improve the final sentiment-classification performance compared with a simpler architecture?

## 2.3 Manual SimpleRNN Forward Pass

A small RNN is implemented manually with NumPy.

The recurrent equation is:

\[
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)
\]

Where:

- \(x_t\) = current input
- \(h_{t-1}\) = previous hidden state
- \(W_{xh}\) = input-to-hidden weights
- \(W_{hh}\) = hidden-to-hidden recurrent weights
- \(b\) = bias
- \(h_t\) = new hidden state

The same recurrent weights are reused at every time step.

## 2.4 Hidden-State Heatmap

<div align="center">
<img src="plots/rnn_forward_heatmap.png" width="800">
</div>

The heatmap visualizes how the hidden-state values change as the sequence is processed.

## 2.5 Keras Verification

The manually calculated hidden states are compared with:

```python
tf.keras.layers.SimpleRNN(
    units=3,
    return_sequences=True
)
```

The manually generated weights are assigned to the Keras layer and the outputs are checked using `np.allclose()`.

This confirms that the manual mathematical implementation follows the same recurrent computation.

## 2.6 SimpleRNN Parameter Count

For:

```text
Input dimension = 32
Hidden units    = 64
```

the recurrent-layer parameter count is:

```text
units × (input_dim + units + 1)

64 × (32 + 64 + 1)
= 6,208
```

### ✅ Task 2 Outcome

Task 2 establishes both a practical ANN baseline and a mathematical understanding of recurrent forward propagation.

---

# 🔁 TASK 3 — Types of RNN & SimpleRNN Sentiment Analysis

## 🎯 Objective

Understand the different input/output structures of recurrent networks and train a complete SimpleRNN sentiment classifier.

## 3.1 Types of RNN

### One-to-One

```text
Input → Output
```

Example: ordinary fixed-input classification.

### One-to-Many

```text
Input → Output₁ → Output₂ → ... → Outputₜ
```

Example: generating a sequence from one input.

### Many-to-One

```text
Input₁
Input₂
Input₃ ───→ One Output
...
Inputₜ
```

Example: **sentiment analysis**.

An entire review is processed and one final sentiment is predicted.

### Many-to-Many

```text
Input Sequence → Output Sequence
```

Example: sequence labeling.

## 3.2 Why This Project Is Many-to-One

A review contains many tokens:

```text
"The movie was absolutely fantastic..."
```

but produces one label:

```text
Positive
```

Therefore, IMDB sentiment classification is a **many-to-one** sequence problem.

## 3.3 Reusable Model Builder

A reusable `build_model()` function is created so that the recurrent cell can be changed without changing the entire experimental setup.

Supported cell types:

```text
rnn
lstm
gru
```

This makes comparisons more consistent.

## 3.4 SimpleRNN Architecture

```text
Review
   ↓
Tokenizer
   ↓
Padding
   ↓
Embedding(10000, 32)
   ↓
SimpleRNN(64)
   ↓
Dense(1, Sigmoid)
   ↓
Sentiment
```

## 3.5 Training Strategy

The model uses:

- Binary cross-entropy
- Adam optimizer
- Batch size 64
- Validation split
- Early stopping
- Restoring best weights

A custom callback also records training time per epoch.

## 3.6 Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion matrix

## 3.7 Embedding Inspection

The learned embedding matrix has the form:

```text
(vocabulary size, embedding dimension)
```

For this project:

```text
(10000, 32)
```

Word vectors such as `good` and `bad` can be extracted from the embedding matrix and inspected.

### ✅ Task 3 Outcome

A complete SimpleRNN sentiment-classification pipeline is built and evaluated, providing the recurrent baseline for comparison with LSTM and GRU.

---

# 📉 TASK 4 — BPTT, Vanishing & Exploding Gradients

## 🎯 Objective

Understand how an RNN learns through time and why long sequences are challenging for a basic recurrent network.

## 4.1 Backpropagation Through Time

An RNN can be conceptually unrolled:

```text
x₁ → h₁ → h₂ → h₃ → ... → hₜ → Prediction
```

During forward propagation, information moves from earlier to later time steps.

During **Backpropagation Through Time**, gradients move backward through the unrolled recurrent computation.

```text
Loss ← hₜ ← hₜ₋₁ ← ... ← h₂ ← h₁
```

The gradient reaching an early hidden state involves repeated Jacobian multiplication.

For tanh:

\[
\frac{d}{dx}\tanh(x)=1-\tanh^2(x)
\]

Repeated multiplication is the source of the two important RNN gradient problems.

## 4.2 Vanishing Gradient

If repeated gradient factors have magnitudes below 1:

```text
0.5 × 0.5 × 0.5 × ...
```

the gradient can become extremely small.

### Effect

Earlier sequence information becomes difficult to learn.

### Practical Meaning

The model may struggle when important sentiment information appears far from the final time step.

### Main Remedy Studied

```text
LSTM / GRU
```

## 4.3 Exploding Gradient

If repeated factors have magnitudes above 1, gradients can grow rapidly.

Possible symptoms:

- Unstable training
- Large loss spikes
- Huge parameter updates
- NaN values

## 4.4 Gradient Norm Experiment

`GradientTape` is used to inspect gradient behaviour across time steps.

<div align="center">
<img src="plots/gradient_norms_all_cells.png" width="850">
</div>

The experiment is used as a practical visualization of recurrent gradient behaviour.

## 4.5 Sequence-Length Experiment

SimpleRNN is trained using:

```text
MAXLEN = 50
MAXLEN = 100
MAXLEN = 200
MAXLEN = 300
```

The purpose is to study whether increasing sequence length always improves classification.

Longer sequences contain more information, but they also make recurrent optimization more difficult and may introduce more noise.

## 4.6 Gradient Clipping

An intentionally high learning rate is used to investigate unstable training.

The experiment compares:

```text
Without clipping
vs
clipnorm = 1.0
```

Gradient clipping limits excessively large gradient norms.

### Important

Gradient clipping is primarily a remedy for **exploding gradients**.

It does not fix gradients that have already vanished.

## 4.7 Problems Summary

| Problem | Cause | Possible Symptom | Remedy |
|---|---|---|---|
| Vanishing gradient | Repeated small derivative factors | Weak learning of distant information | LSTM / GRU |
| Exploding gradient | Repeated large gradient factors | Unstable loss / huge updates | Gradient clipping |
| Short-term memory | Basic recurrent transition | Difficulty with long dependencies | Gated recurrence |
| Sequential computation | Current state depends on previous state | Training can be slower | Efficient hardware / architecture choices |

### ✅ Task 4 Outcome

Task 4 explains why a basic RNN may struggle with long-term dependencies and motivates the introduction of gated recurrent architectures.

---

# 🧠 TASK 5 — Long Short-Term Memory (LSTM)

## 🎯 Objective

Understand LSTM gates and cell state, implement the LSTM equations manually, and compare LSTM with SimpleRNN.

## 5.1 LSTM Components

LSTM maintains:

```text
Hidden State → hₜ
Cell State   → Cₜ
```

Its main components are:

- Forget gate
- Input gate
- Candidate state
- Cell-state update
- Output gate
- Hidden state

## Forget Gate

\[
f_t=\sigma(W_f[h_{t-1},x_t]+b_f)
\]

Controls how much previous memory is retained.

## Input Gate

\[
i_t=\sigma(W_i[h_{t-1},x_t]+b_i)
\]

Controls how much new information is written.

## Candidate State

\[
g_t=\tanh(W_g[h_{t-1},x_t]+b_g)
\]

Creates candidate information.

## Cell-State Update

\[
C_t=f_tC_{t-1}+i_tg_t
\]

Combines retained memory with new information.

## Output Gate

\[
o_t=\sigma(W_o[h_{t-1},x_t]+b_o)
\]

Controls how much memory is exposed.

## Hidden State

\[
h_t=o_t\tanh(C_t)
\]

## 5.2 Why LSTM Helps

The cell state provides a more direct additive memory path.

When the forget gate is near one:

\[
C_t \approx C_{t-1}
\]

information can be preserved for longer periods.

This makes LSTM better suited than SimpleRNN for learning long-range dependencies.

## 5.3 LSTM Parameter Count

For 64 hidden units and input dimension 32:

```text
4 × units × (input_dim + units + 1)

4 × 64 × (32 + 64 + 1)

= 24,832 recurrent parameters
```

The factor of four comes from the four major transformations used by the LSTM.

## 5.4 Manual LSTM

A small LSTM is implemented manually using NumPy with:

```text
Input size  = 2
Hidden size = 3
Time steps  = 4
```

At every step the implementation calculates:

```text
Forget Gate
Input Gate
Candidate State
Cell State
Output Gate
Hidden State
```

This demonstrates how memory is explicitly controlled.

## 5.5 LSTM Sentiment Model

```text
Embedding(10000, 32)
        ↓
LSTM(64)
        ↓
Dense(1, Sigmoid)
```

The same training setup used for SimpleRNN is retained to make the comparison fair.

## 5.6 SimpleRNN vs LSTM Gradient Flow

The gradient experiment is repeated with LSTM and compared with SimpleRNN.

The goal is to investigate how gating and memory affect information and gradient flow.

## 5.7 Stacked LSTM

Architecture:

```text
Embedding
    ↓
LSTM(64, return_sequences=True)
    ↓
LSTM(32)
    ↓
Dense(1, Sigmoid)
```

The first LSTM requires:

```python
return_sequences=True
```

because the second LSTM needs an entire sequence of hidden states rather than only one final vector.

### ✅ Task 5 Outcome

Task 5 demonstrates how LSTM introduces explicit memory control and provides a gated alternative to SimpleRNN.

---

# 🚀 TASK 6 — Gated Recurrent Unit (GRU)

## 🎯 Objective

Understand GRU gates, compare GRU with LSTM, train a GRU sentiment classifier and analyze model capacity.

## 6.1 GRU Architecture

GRU uses two primary gates:

```text
Update Gate
Reset Gate
```

and does not maintain a separate LSTM-style cell state.

## Update Gate

\[
z_t=\sigma(W_z[h_{t-1},x_t]+b_z)
\]

Controls the balance between old information and new candidate information.

## Reset Gate

\[
r_t=\sigma(W_r[h_{t-1},x_t]+b_r)
\]

Controls how much previous hidden information contributes to the candidate state.

## Candidate Hidden State

\[
\tilde{h}_t=
\tanh(W_h[r_t*h_{t-1},x_t]+b_h)
\]

## Final Hidden State

\[
h_t=(1-z_t)h_{t-1}+z_t\tilde{h}_t
\]

## 6.2 LSTM vs GRU

| Feature | LSTM | GRU |
|---|---|---|
| Main gates | Forget, Input, Output | Update, Reset |
| Cell state | Separate cell state | No separate cell state |
| Architecture | More complex | Simpler |
| Recurrent parameters | More | Fewer |
| Memory mechanism | Cell + hidden state | Hidden state |
| Training cost | Often higher | Often lower |

Neither architecture is guaranteed to win on every dataset; the final decision should be based on experimental results.

## 6.3 GRU Parameter Count

With Keras `reset_after=True`:

```text
3 × (units × input_dim + units² + 2 × units)

3 × (64×32 + 64×64 + 2×64)

= 18,816 recurrent parameters
```

Therefore:

```text
SimpleRNN recurrent parameters = 6,208
GRU recurrent parameters       = 18,816
LSTM recurrent parameters      = 24,832
```

## 6.4 GRU Sentiment Model

```text
Embedding(10000, 32)
        ↓
GRU(64)
        ↓
Dense(1, Sigmoid)
```

GRU is trained using the same core experimental settings as the other recurrent models.

## 6.5 Gradient Comparison

The final gradient plot compares:

```text
SimpleRNN
LSTM
GRU
```

<div align="center">
<img src="plots/gradient_norms_all_cells.png" width="850">
</div>

## 6.6 Hidden-Unit Sensitivity

Both LSTM and GRU are trained using:

```text
32 units
64 units
128 units
```

The experiment compares:

- Test accuracy
- Parameter count

This demonstrates an important deep-learning principle:

> A larger model is not automatically a better model.

Increasing hidden units increases capacity, but also increases memory usage, training cost and overfitting risk.

### ✅ Task 6 Outcome

Task 6 introduces a simpler gated recurrent architecture and experimentally compares model capacity and performance.

---

# 🏆 TASK 7 — Final Comparison, Analysis & Inference

## 🎯 Objective

Bring together every trained model and determine which architecture provides the strongest final result.

Task 7 performs:

1. Final model comparison
2. Combined ROC analysis
3. Review-length analysis
4. Misclassification analysis
5. Custom inference
6. Production recommendation

---

# 7.1 📊 Complete Final Model Comparison

The final output from this project is:

| Model | Parameters | Time / Epoch | Epochs | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ANN Baseline** | 1,548,929 | 12.60 s | 5 | 86.62% | 86.63% | 86.72% | 86.68% | 0.9379 |
| **SimpleRNN** | 326,273 | 68.88 s | 4 | 79.01% | 88.69% | 66.67% | 76.12% | 0.8787 |
| **LSTM** | 344,897 | 179.99 s | 7 | 88.68% | 87.13% | 90.86% | 88.95% | 0.9497 |
| **Stacked LSTM** | 357,281 | 1358.36 s | 6 | 88.31% | 88.58% | 88.07% | 88.32% | 0.9499 |
| **GRU 🏆** | 338,881 | 174.53 s | 6 | 89.73% | 89.69% | 89.87% | 89.78% | 0.9595 |

<div align="center">
<img src="plots/results_table.png" width="950">
</div>

## 🔎 Result Interpretation

### ANN Baseline

- Accuracy: **86.62%**
- F1: **86.68%**
- ROC-AUC: **0.9379**
- Parameters: **1,548,929**
- Time/Epoch: **12.60 s**

The ANN baseline produced strong performance and was the fastest model in the recorded final table. However, flattening the sequence resulted in a very large parameter count.

### SimpleRNN

- Accuracy: **79.01%**
- Precision: **88.69%**
- Recall: **66.67%**
- F1: **76.12%**
- ROC-AUC: **0.8787**

SimpleRNN had the weakest final accuracy and F1 among the five submitted models. Its precision remained comparatively strong, while its lower recall indicates that more actual positive reviews were missed.

### LSTM

- Accuracy: **88.68%**
- F1: **88.95%**
- ROC-AUC: **0.9497**

LSTM substantially improved the final metrics over SimpleRNN in this experiment.

### Stacked LSTM

- Accuracy: **88.31%**
- F1: **88.32%**
- ROC-AUC: **0.9499**
- Time/Epoch: **1358.36 s**

Adding another recurrent layer increased model complexity and recorded training time, but it did not produce the best final test performance.

### GRU 🏆

- Accuracy: **89.73%**
- Precision: **89.69%**
- Recall: **89.87%**
- F1: **89.78%**
- ROC-AUC: **0.9595**
- Parameters: **338,881**

GRU achieved the strongest overall final result in the supplied comparison table.

---

# 7.2 📈 Combined ROC Curve

The ROC curve evaluates model discrimination across different classification thresholds.

<div align="center">
<img src="plots/roc_curves.png" width="850">
</div>

### ROC-AUC Interpretation

```text
ROC-AUC = 1.0  → Excellent discrimination
ROC-AUC = 0.5  → Random-like discrimination
```

Final ROC-AUC values:

| Model | ROC-AUC |
|---|---:|
| ANN Baseline | 0.9379 |
| SimpleRNN | 0.8787 |
| LSTM | 0.9497 |
| Stacked LSTM | 0.9499 |
| **GRU** | **0.9595** |

GRU achieved the highest ROC-AUC in the supplied final results.

---

# 7.3 📐 Performance by Review Length

Test reviews are divided into:

```text
Short  → fewer than 100 words
Medium → 100–250 words
Long   → more than 250 words
```

The best gated model is compared with SimpleRNN.

<div align="center">
<img src="plots/accuracy_by_length.png" width="850">
</div>

### Why Perform This Analysis?

Overall accuracy alone does not show whether a model behaves differently on short and long text.

A short review may contain only a few strong sentiment words:

```text
"Amazing movie!"
```

A long review may contain:

- Multiple opinions
- Negation
- Context changes
- Mixed sentiment
- Important information separated by many words

This analysis therefore helps evaluate whether gated recurrence provides an advantage when more temporal context must be processed.

> The exact short/medium/long group accuracies should be read from the generated notebook output/plot because they are not contained in the supplied final model-comparison CSV.

---

# 7.4 ❌ Misclassification Analysis

The best gated model is used to inspect **8 incorrect predictions**:

```text
4 False Positives
4 False Negatives
```

## False Positive

```text
True sentiment      = Negative
Predicted sentiment = Positive
```

The model incorrectly believes a negative review is positive.

## False Negative

```text
True sentiment      = Positive
Predicted sentiment = Negative
```

The model incorrectly believes a positive review is negative.

For every selected error, the notebook displays:

- Error type
- True label
- Predicted label
- Positive-class probability
- Cleaned review text

### Why Can Sentiment Models Fail?

#### 1. Negation

```text
"This movie is not good."
```

The word `good` is positive, but `not` reverses the meaning.

#### 2. Mixed Sentiment

```text
"The acting was excellent but the story was terrible."
```

The same review contains both positive and negative evidence.

#### 3. Sarcasm

```text
"What a masterpiece — I loved wasting two hours."
```

The literal wording can differ from the intended sentiment.

#### 4. Long-Range Context

An early sentence may change how later words should be interpreted.

#### 5. OOV / Rare Words

Unknown vocabulary may be mapped to `<OOV>`, reducing available semantic information.

#### 6. Ambiguous Language

Human reviewers themselves may disagree about some borderline examples.

### Why Error Analysis Matters

A model with high accuracy can still fail systematically on particular language patterns. Looking at actual mistakes gives a more realistic understanding of model quality.

---

# 7.5 🔮 Custom Sentiment Prediction

A reusable function is created:

```python
predict_sentiment(text)
```

The function performs the exact preprocessing pipeline required by the trained model:

```text
New Text
   ↓
clean_text()
   ↓
Tokenizer
   ↓
Integer Sequence
   ↓
Padding to MAXLEN
   ↓
Best Model
   ↓
Positive Probability
   ↓
Threshold = 0.5
   ↓
Positive / Negative
```

Example:

```python
result = predict_sentiment(
    "This movie was absolutely wonderful."
)

print(result)
```

Output structure:

```python
{
    "label": "Positive",
    "positive_probability": 0.XX
}
```

The numerical probability depends on the trained model and input.

---

# 7.6 🧪 Six Custom Language Tests

The final model is tested on six different cases.

### 😊 1. Clearly Positive

```text
This movie was absolutely fantastic.
The acting was brilliant and I loved every moment.
```

Purpose: test straightforward positive sentiment.

### 😡 2. Clearly Negative

```text
This movie was terrible, boring and a complete waste of time.
```

Purpose: test straightforward negative sentiment.

### 🚫 3. Negation

```text
The movie was not good and I would not recommend it.
```

Purpose: determine whether the model can interpret negation.

### 🔄 4. Double Negation

```text
I cannot say that this movie was not enjoyable.
```

Purpose: test more complex linguistic structure.

### 😏 5. Sarcasm

```text
What a masterpiece, I especially loved wasting two hours of my life.
```

Purpose: test the difference between literal positive wording and negative intended meaning.

### ⚡ 6. Very Short Review

```text
Amazing movie.
```

Purpose: determine whether the model can classify sentiment with very little context.

These examples demonstrate that real-world language understanding requires more than simply recognizing positive and negative keywords.

---

# 7.7 🏭 Production Recommendation

The final model should not be selected using accuracy alone.

The comparison considers:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Parameter Count
Training Time
Gradient Behaviour
Review-Length Behaviour
Error Analysis
Custom Predictions
```

## 🏆 Final Best Model — GRU

Based on the supplied final comparison results:

| Metric | GRU Result |
|---|---:|
| **Accuracy** | **89.73%** |
| **Precision** | **89.69%** |
| **Recall** | **89.87%** |
| **F1** | **89.78%** |
| **ROC-AUC** | **0.9595** |
| **Parameters** | **338,881** |
| **Time / Epoch** | **174.53 s** |

GRU achieved the highest **test accuracy, F1 score, and ROC-AUC** in the supplied final results.

It also used fewer parameters than the LSTM and Stacked LSTM models in the final model table.

Therefore, **GRU is the preferred recurrent architecture for this experiment**.

---

# 📊 Training-Curve Comparison

<div align="center">
<img src="plots/training_curves_rnn_lstm_gru.png" width="850">
</div>

The validation curves allow comparison of how SimpleRNN, LSTM and GRU learn across epochs and help identify signs such as convergence or overfitting.

---

# 📋 Final Results at a Glance

<div align="center">

| 🏅 Category | Winner |
|---|---|
| Best Test Accuracy | **GRU — 89.73%** |
| Best F1 Score | **GRU — 89.78%** |
| Best ROC-AUC | **GRU — 0.9595** |
| Fastest Recorded Time/Epoch | **ANN — 12.60 s** |
| Best Recurrent Model | **GRU 🏆** |

</div>

---

# 📂 Project Structure

```text
DL_PR5/
│
├── DL_PR5.ipynb
├── DL_PR5.html
├── README.md
├── requirements.txt
├── model_comparison_results.csv
├── IMDB Dataset.csv
│
└── plots/
    ├── length_distribution.png
    ├── rnn_forward_heatmap.png
    ├── gradient_norms_all_cells.png
    ├── training_curves_rnn_lstm_gru.png
    ├── roc_curves.png
    ├── results_table.png
    └── accuracy_by_length.png
```

---

# ⚙️ How to Run

## 1️⃣ Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd DL_PR5
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

## 4️⃣ Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
DL_PR5.ipynb
```

Then run all cells in order.

---

# 📦 Requirements

```text
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
```

---

# 🛠️ Tools & Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep-learning framework |
| Keras | Neural-network API |
| Pandas | Dataset handling |
| NumPy | Mathematical/manual RNN and LSTM calculations |
| Matplotlib | Visualization |
| Seaborn | Heatmaps/confusion matrices |
| Scikit-learn | Splitting and evaluation metrics |
| Jupyter Notebook | Development and experimentation |
| Git/GitHub | Version control and project presentation |

---

# 🧠 What I Learned

Through this practical I learned:

### Data Preparation
- Text cleaning
- Label encoding
- Train/test splitting
- Tokenization
- OOV handling
- Padding and truncation
- Data-leakage prevention

### Deep Learning
- Embeddings
- ANN baseline modeling
- SimpleRNN
- LSTM
- Stacked LSTM
- GRU
- Sigmoid binary classification

### Recurrent-Network Mathematics
- Hidden-state calculation
- Recurrent weight sharing
- BPTT
- Vanishing gradients
- Exploding gradients
- Gradient clipping
- LSTM gates
- GRU gates

### Evaluation
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- ROC curve
- ROC-AUC

### Model Analysis
- Sequence-length sensitivity
- Hidden-unit sensitivity
- Gradient-flow analysis
- Review-length analysis
- False positives
- False negatives
- Custom inference

---

# 💡 Major Project Insights

### 1. A Simple Model Can Still Be Competitive

The ANN baseline reached **86.62% accuracy**, showing that a simpler architecture can provide a strong baseline.

### 2. Recurrent Architecture Choice Matters

SimpleRNN achieved only **79.01% accuracy** in the final results, while LSTM and GRU performed substantially better.

### 3. More Depth Did Not Automatically Mean Better Performance

The Stacked LSTM recorded **88.31% accuracy**, below the single LSTM and GRU in this experiment.

### 4. GRU Provided the Best Final Balance

GRU achieved:

```text
Accuracy : 89.73%
F1       : 89.78%
ROC-AUC  : 0.9595
```

while using fewer parameters than the LSTM and Stacked LSTM models.

---

# 🚀 Future Improvements

Possible future extensions include:

- Dropout and recurrent dropout
- Hyperparameter optimization
- Different vocabulary sizes
- Different embedding dimensions
- More advanced text cleaning
- Threshold optimization
- Class-probability calibration
- Larger sequence-length studies
- More detailed error analysis
- REST API deployment
- Interactive web interface

> This practical intentionally focuses on ANN, SimpleRNN, LSTM and GRU. Attention, Transformers and pretrained embeddings are outside the scope of this implementation.

---

# 🎥 Project Demo

Add your final project demo link here:

```text
Demo Video: <YOUR-VIDEO-LINK>
```

Suggested demo flow:

```text
1. Project Introduction
2. Dataset & Preprocessing
3. ANN Baseline
4. Manual RNN Forward Pass
5. SimpleRNN
6. BPTT + Gradient Problems
7. LSTM
8. GRU
9. Final Model Comparison
10. Live Sentiment Prediction
```

---

# 🏁 Final Conclusion

This project provides an end-to-end study of sentiment analysis using recurrent deep-learning architectures.

The practical begins with raw IMDB text and develops a complete preprocessing pipeline. It then establishes an ANN baseline before exploring SimpleRNN mathematically and experimentally.

The BPTT section demonstrates why recurrent networks can experience vanishing and exploding gradients. LSTM and GRU are then introduced as gated recurrent architectures designed to improve long-range information handling.

The final submitted comparison shows:

```text
ANN Baseline   → 86.62% Accuracy
SimpleRNN      → 79.01% Accuracy
LSTM           → 88.68% Accuracy
Stacked LSTM   → 88.31% Accuracy
GRU            → 89.73% Accuracy 🏆
```

Among the evaluated recurrent models, **GRU produced the strongest final result**, with:

```text
Test Accuracy → 89.73%
F1 Score      → 89.78%
ROC-AUC       → 0.9595
```

The project therefore demonstrates not only how to build recurrent sentiment classifiers, but also how to understand their internal mechanics, evaluate their limitations, compare architectures fairly, and analyze real prediction errors.

---

<div align="center">

# 👨‍💻 Author

## Tushar Vala

### Data Analyst • AI/ML Developer

<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=00BFFF&center=true&vCenter=true&width=900&lines=Data+%E2%86%92+Sequences+%E2%86%92+Deep+Learning;SimpleRNN+%E2%86%92+LSTM+%E2%86%92+GRU;Understand+the+Model%2C+Not+Just+the+Accuracy;Thanks+for+Visiting+the+Project+%F0%9F%9A%80" alt="Footer Animation" />

<br>

### ⭐ If this project helped you, consider giving the repository a star!

**Made with Python 🐍 + TensorFlow 🧠 + lots of experimentation 📊**

</div>
