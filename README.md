# 🧠 Building an LLM From Scratch

A hands-on, day-by-day journey through **building a Large Language Model (LLM) from scratch**, tracked through code, experiments, notes, and implementations.

This repository documents my learning while working through a Udemy course on LLMs. It starts with text processing and tokenization and progresses through embeddings, GPT training, evaluation, AI safety, and mechanistic interpretability, along with the Python and deep learning fundamentals behind them.

The goal is not simply to use existing LLM libraries, but to understand **how each component works by implementing it step by step**.

---

## 🔗 Repository

**GitHub:** https://github.com/p-arkaprava/Deep-Learning-and-Machine-Learning-Models

---

## 🎯 Objectives

* Understand how language models represent text.
* Implement tokenization and vocabulary creation from scratch.
* Understand embeddings and vector representation spaces.
* Build the mathematical foundations needed for neural networks and Transformers.
* Build, pretrain, fine-tune, and instruction-tune a GPT-style model.
* Evaluate language models with both quantitative and qualitative methods.
* Understand AI safety and mechanistic interpretability.
* Inspect and intervene on model internals (activations, hidden states, attention, MLPs).
* Maintain a daily record of progress and experiments.

---

## 🗺️ Learning Roadmap

The course is organized into 8 numbered topic folders:

| # | Topic | Focus |
|---|-------|-------|
| 01 | Text to Vectors Basics | Tokens, numeric encoding, embedding spaces |
| 02 | Core Language Model Development | Building a GPT, pretraining, fine-tuning, instruction tuning |
| 03 | Assessing Model Performance | Quantitative and qualitative evaluation of LLMs |
| 04 | Safe AI and Model Transparency | AI safety, interpretability foundations |
| 05 | Passive Model Inspection | Observational mechanistic interpretability |
| 06 | Active Model Intervention | Causal interventions on activations, hidden states, attention, MLPs |
| 07 | Python Crash Course | Python, plotting, strings, PyTorch basics |
| 08 | Neural Network Foundations | Math of deep learning, gradient descent, modeling essentials |

The overall flow:

```text
Text → Tokens → Vectors → Neural Network → Attention → Transformer → Language Model → Generated Text
```

---

## ✅ Progress Tracker

**01 - Text to Vectors Basics**
- [x] 01 - Welcome and Overview
- [x] 02 - Turning Text into Numeric Tokens
- [ ] 03 - Vector Representation Spaces

**02 - Core Language Model Development**
- [ ] 01 - Constructing a GPT Model
- [ ] 02 - Large-Scale Model Pretraining
- [ ] 03 - Adapting Pretrained Models
- [ ] 04 - Teaching Models to Follow Instructions

**03 - Assessing Model Performance**
- [ ] 01 - Numerical Metrics Assessment
- [ ] 02 - Human-Style Judgment Review

**04 - Safe AI and Model Transparency**
- [ ] 01 - Responsible AI Risks
- [ ] 02 - Understanding Model Behavior

**05 - Passive Model Inspection**
- [ ] 01 - Exploring Token Vectors I
- [ ] 02 - Probing Units and Axes
- [ ] 03 - Examining Network Depth Levels
- [ ] 04 - Exploring Token Vectors II
- [ ] 05 - Locating Circuits and Modules

**06 - Active Model Intervention**
- [ ] 01 - Ways to Alter Activations
- [ ] 02 - Rewriting Internal Representations
- [ ] 03 - Disrupting Attention Heads
- [ ] 04 - Adjusting Feed-Forward Layers

**07 - Python Crash Course**
- [ ] 01 - Notebook Environment Setup
- [ ] 02 - Variable and Value Kinds
- [ ] 03 - Accessing Elements and Ranges
- [ ] 04 - Reusable Code Blocks
- [ ] 05 - Conditions and Loops
- [ ] 06 - Plotting and Charts
- [ ] 07 - Working with Text Data
- [ ] 08 - Tensor Library Basics

**08 - Neural Network Foundations**
- [ ] 01 - Mathematical Foundations
- [ ] 02 - Optimization by Descending Gradients
- [ ] 03 - Core Modeling Principles
- [ ] 04 - Extra Material

### Status Overview

| Topic | Status |
|-------|--------|
| 01 - Text to Vectors Basics | 🟢 In Progress |
| 02 - Core Language Model Development | ⚪ Upcoming |
| 03 - Assessing Model Performance | ⚪ Upcoming |
| 04 - Safe AI and Model Transparency | ⚪ Upcoming |
| 05 - Passive Model Inspection | ⚪ Upcoming |
| 06 - Active Model Intervention | ⚪ Upcoming |
| 07 - Python Crash Course | ⚪ Upcoming |
| 08 - Neural Network Foundations | ⚪ Upcoming |

---

## 📍 Current Progress

### Day 1 — Text Processing & Basic Tokenization

The first stage focuses on converting raw text into numerical representations a machine-learning model can process.

Implemented so far:

* Splitting text into individual words
* Combining text samples and converting to lowercase
* Creating, deduplicating, and sorting a vocabulary
* Building `word2idx` and `idx2word` dictionaries
* Converting words to integer tokens and back
* Reusable encoder and decoder functions
* Visualizing tokenized integers

```text
Raw Text → Text Processing → Words → Vocabulary → word2idx → Integer Tokens → Neural Network
```

The current implementation is intentionally simple and is the foundation for more advanced tokenization and language-model components.

---

## 🗂️ Repository Structure

```text
Deep-Learning-and-Machine-Learning-Models/
│
├── README.md
├── .gitignore
│
├── 01 - Text to Vectors Basics/
│   ├── 01 - Welcome and Overview/
│   │   └── README.md
│   ├── 02 - Turning Text into Numeric Tokens/
│   │   └── README.md
│   └── 03 - Vector Representation Spaces/
│       └── README.md
│
├── 02 - Core Language Model Development/
│   └── ...
│
├── ...
│
└── 08 - Neural Network Foundations/
    └── ...
```

Each lesson folder contains:

* `README.md`: my notes and key takeaways
* notebooks, code, and experiments for that lesson, added as I go

---

## 📆 Daily Progress

This repository follows a **daily commit approach**. Each day I aim for at least one meaningful change, such as:

* Learning a new concept
* Implementing a new component
* Improving existing code
* Running an experiment
* Fixing a bug
* Adding notes or documentation

The commit history therefore acts as a chronological record of how the project developed.

```text
Day 01 - Implement basic word-level tokenization
Day 02 - Finish tokenization lesson and notes
Day 03 - Explore embedding spaces
...
```

---

## 💡 Philosophy

> **Don't just use the model. Understand how it works.**

Instead of treating an LLM as a black box, this project breaks the architecture into smaller pieces and studies each one individually.

---

## 🛠️ Technologies

* Python
* NumPy
* Matplotlib
* Jupyter Notebook / Google Colab
* PyTorch (introduced later in the course)

Additional libraries may be added as the project progresses.

---

## 🤔 Why This Repository?

1. **Learning**: to deeply understand the internals of LLMs.
2. **Documentation**: to keep a record of what I learn every day.
3. **Portfolio**: to show the progression from basic ML concepts to a working language model.

---

## 🏁 Long-Term Goal

To go from understanding basic tokenization to implementing and evaluating a complete Transformer-based language model, and to be able to explain and implement its major components rather than only calling a pre-trained model through an API.

---

## 🗒️ Progress Log

### 2026-10-07

**Topic:** Introduction to text processing and tokenization

**Done:**

* Finished the first two lessons of topic 01
* Basic text preprocessing, vocabulary generation, word/index mappings
* Encoding text to integer tokens and decoding back

**Next:** Vector representation spaces, then dataset construction and context-window based training.

---

## 📄 License

This is a personal learning and experimentation project. All course material belongs to its original instructor; this repository contains only my own notes and code. Licensing details will be added later.