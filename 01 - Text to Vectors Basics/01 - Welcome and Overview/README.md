# 01 - Welcome and Overview

> **Lesson goal:** understand what this course is building, how a language model works at the highest level, and how I will study and document it in this repository.

---

## 📌 Table of Contents

1. [What this lesson is about](#-what-this-lesson-is-about)
2. [What is a Large Language Model?](#-what-is-a-large-language-model)
3. [The big picture: from text to generated text](#-the-big-picture-from-text-to-generated-text)
4. [Course roadmap](#-course-roadmap)
5. [Core ideas to keep in mind](#-core-ideas-to-keep-in-mind)
6. [Tools and environment](#-tools-and-environment)
7. [How I study and document this course](#-how-i-study-and-document-this-course)
8. [Glossary](#-glossary)
9. [Questions I want to be able to answer by the end](#-questions-i-want-to-be-able-to-answer-by-the-end)
10. [Takeaways](#-takeaways)

---

## 🎯 What this lesson is about

This lesson is the orientation. No model is trained here. The purpose is to set expectations and build a mental map, so that every later lesson has a place to attach to.

The course follows one philosophy that I also use for this repository:

> **Don't just use the model. Understand how it works.**

Instead of calling a ready-made library and treating the result as a black box, the course breaks a language model into small components and implements each one step by step.

---

## 🤖 What is a Large Language Model?

A **Large Language Model (LLM)** is a neural network trained on a very large amount of text to do one deceptively simple job:

> **Given the tokens so far, predict the next token.**

That is all the training objective asks for. Everything an LLM appears to do, such as answering questions, summarizing, translating, writing code, or holding a conversation, emerges from getting very good at this one prediction task at scale.

Three words in the name each carry meaning:

| Word | Meaning |
|------|---------|
| **Large** | Many parameters (weights), trained on a large amount of text with large amounts of compute. |
| **Language** | The data is human language (and often code). |
| **Model** | A mathematical function with learned parameters that maps an input sequence to a probability distribution over the next token. |

Generation works by repeating the prediction:

```text
Prompt:    "The cat sat on the"
Model →    probabilities over the vocabulary   (e.g. "mat" is likely)
Pick one → "mat"
Append →   "The cat sat on the mat"
Repeat…
```

---

## 🧩 The big picture: from text to generated text

```text
Text → Tokens → Vectors → Neural Network → Attention → Transformer → Language Model → Generated Text
```

| Stage | What happens | Where it appears in the course |
|-------|--------------|--------------------------------|
| **Text → Tokens** | Raw text is split into units and each unit gets an integer ID | Topic 01, lesson 02 |
| **Tokens → Vectors** | Each ID is mapped to a vector of numbers (an embedding) | Topic 01, lesson 03 |
| **Neural Network** | Layers of weights transform vectors; learning happens by gradient descent | Topic 08 |
| **Attention** | Each token looks at other tokens to decide what matters | Topic 02 |
| **Transformer** | Attention and feed-forward layers stacked into a deep architecture | Topic 02 |
| **Language Model** | The trained network that outputs next-token probabilities | Topic 02 |
| **Generated Text** | Probabilities are sampled and decoded back into readable text | Topic 02 |

The first two stages are what this topic (01) is about: **getting language into a numerical form a network can work with**.

---

## 🗺️ Course roadmap

The repository has 8 numbered topic folders.

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

The topics fall into three broad groups:

* **Build it** (01, 02): tokenization, embeddings, a GPT model, pretraining, fine-tuning, instruction tuning.
* **Judge and understand it** (03, 04, 05, 06): evaluation, safety, and looking inside the model, first by observing and then by intervening.
* **Support material** (07, 08): Python and neural-network foundations that I can return to whenever a topic needs them.

Topics 07 and 08 are numbered last, but they are references, not prerequisites to be finished first. I can dip into them whenever a concept in an earlier topic needs backup.

---

## 💡 Core ideas to keep in mind

1. **Models only see numbers.** Text has to be converted to integers, and integers to vectors, before any learning can happen.
2. **Prediction is the engine.** Next-token prediction is the single training objective behind the behaviours we see.
3. **Context is everything.** A token's meaning depends on the tokens around it. Context windows and attention are how models use that.
4. **Scale matters, but understanding matters more.** Data, parameters, and compute drive capability, yet each component can be understood on a small example first.
5. **Build small, then scale.** Every idea is first implemented on a tiny example (a few sentences, a few dozen words) so I can see exactly what is happening.
6. **Evaluate and inspect.** A model that works is not the same as a model that is understood or safe. Later topics cover measuring, interpreting, and intervening.

---

## 🛠️ Tools and environment

| Tool | Used for |
|------|----------|
| **Python** | The main language for all code in this repository |
| **NumPy** | Arrays, random sampling, searching (`np.where`) |
| **Matplotlib** | Visualizing tokens and, later, vectors and training curves |
| **Jupyter Notebook / Google Colab** | Running experiments cell by cell and keeping notes beside code |
| **PyTorch** | Introduced later in the course for tensors and neural networks |
| **Git and GitHub** | Version history and daily progress log |

Setup is intentionally minimal for the first topic: Python plus `numpy` and `matplotlib` is enough.

```bash
pip install numpy matplotlib jupyter
```

---

## 📆 How I study and document this course

* **One lesson, one folder.** Each lesson has its own folder with a `README.md` (my notes) and code or notebooks as they appear.
* **Daily commits.** I aim for at least one meaningful commit per day: a new concept, an experiment, a bug fix, or improved notes.
* **Notes in my own words.** READMEs are written to explain things as I understand them, not copied from the course.
* **Code first, then explain.** I implement the idea, run it on a small example, and then write down what the output shows.
* **Progress tracked at the root.** The root `README.md` has the checklist and progress log.

---

## 📖 Glossary

| Term | Short definition |
|------|------------------|
| **LLM** | Large Language Model: a neural network trained to predict the next token in text. |
| **Token** | The unit of text a model reads and writes (a word, word piece, or character). |
| **Vocabulary** | The complete set of tokens a model knows. |
| **Embedding** | A vector of numbers that represents a token. |
| **Parameter / weight** | A learned number inside the network. |
| **Transformer** | The neural network architecture, built on attention, used by modern LLMs. |
| **Attention** | A mechanism that lets each token weigh the importance of other tokens. |
| **Pretraining** | Training on a large general text corpus to learn language. |
| **Fine-tuning** | Further training on a smaller, specific dataset. |
| **Instruction tuning** | Fine-tuning so the model follows human instructions. |
| **Inference** | Using a trained model to generate output. |
| **Interpretability** | Studying how and why a model produces its outputs. |

---

## ❓ Questions I want to be able to answer by the end

* How is text converted into numbers, and why is that conversion designed the way it is?
* What is an embedding, and why does geometry in vector space reflect meaning?
* How does attention let a model use context?
* What does a GPT block actually compute?
* How is a model pretrained, adapted, and made to follow instructions?
* How do we measure whether a language model is good?
* What can be learned by looking inside a model, and what happens when we change its internals?

I will revisit this list at the end of the course and check which questions I can answer from the code I wrote.

---

## ✅ Takeaways

* An LLM is a next-token predictor; everything else is built around that.
* The first job is to turn text into numbers. That is the subject of the next lesson.
* This repository exists to **learn by implementing**, with a daily record of progress.

**Next lesson:** [02 - Turning Text into Numeric Tokens](../02%20-%20Turning%20Text%20into%20Numeric%20Tokens/)

Back to the [topic overview](../README.md).
