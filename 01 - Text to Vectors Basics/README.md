# 01 - Text to Vectors Basics

> **Big idea:** a neural network cannot read words. It can only do arithmetic on numbers.
> This topic covers the first half of the bridge between human language and a language model: **turning raw text into numbers**, and then **turning those numbers into vectors that carry meaning**.

---

## 📌 Table of Contents

1. [Why this topic exists](#-why-this-topic-exists)
2. [Lessons in this topic](#-lessons-in-this-topic)
3. [The pipeline at a glance](#-the-pipeline-at-a-glance)
4. [What I have learned so far](#-what-i-have-learned-so-far)
5. [Folder structure](#-folder-structure)
6. [How to run the code](#-how-to-run-the-code)
7. [Key terms](#-key-terms)
8. [Limitations of the current implementation](#-limitations-of-the-current-implementation)
9. [What comes next](#-what-comes-next)

---

## 🎯 Why this topic exists

Every large language model, from a tiny toy model to a frontier system, starts the same way. Before any attention layer, any Transformer block, or any training loop, the input text has to be converted into numerical form.

If this step is done poorly, everything downstream suffers:

* a vocabulary that is too small means the model cannot represent many words;
* a vocabulary that is too large makes the model huge and slow;
* inconsistent text cleaning means the same word gets different numbers in different places;
* a lossy encoding means the model can never recover the original text.

So this topic is the **foundation of the whole course**. Everything that follows, including embeddings, GPT, training, evaluation, and interpretability, assumes that text can be reliably converted to numbers and back.

---

## 📚 Lessons in this topic

| # | Lesson | Folder | What it covers | Status |
|---|--------|--------|----------------|--------|
| 01 | Welcome and Overview | [`01 - Welcome and Overview`](./01%20-%20Welcome%20and%20Overview/) | Course orientation, the big picture of how an LLM works, roadmap, tools and study approach | ✅ Done |
| 02 | Turning Text into Numeric Tokens | [`02 - Turning Text into Numeric Tokens`](./02%20-%20Turning%20Text%20into%20Numeric%20Tokens/) | Splitting text, building a vocabulary, `word2idx` / `idx2word`, encoder and decoder functions, visualizing tokens, context around a token | ✅ Done |
| 03 | Vector Representation Spaces | [`03 - Vector Representation Spaces`](./03%20-%20Vector%20Representation%20Spaces/) | How integer tokens become dense vectors (embeddings) that live in a space where distance carries meaning | ⏳ Upcoming |

---

## 🔄 The pipeline at a glance

```text
Raw text
   │   "To be or not to be"
   ▼
Text processing        lowercase, split on whitespace
   │   ['to', 'be', 'or', 'not', 'to', 'be']
   ▼
Vocabulary             unique words, sorted, each assigned an index
   │   {'be': 3, 'not': 7, 'or': 9, 'to': 16, ...}
   ▼
Integer tokens         encoder(): words → integers
   │   [16, 3, 9, 7, 16, 3]
   ▼
Vectors (lesson 03)    embedding lookup: integer → vector of floats
   │   16 → [0.12, -0.80, 0.33, ...]
   ▼
Neural network / Transformer (later topics)
```

This topic covers the first three arrows in detail and prepares for the fourth.

---

## 🧠 What I have learned so far

### Lesson 01 - Welcome and Overview
* What the course is trying to build and in what order.
* How an LLM is, at its core, a **next-token predictor** trained on large amounts of text.
* The overall roadmap, from tokens to vectors to Transformers to evaluation to interpretability.

### Lesson 02 - Turning Text into Numeric Tokens
* Text must be **split** into units (here, words), and the split must be consistent.
* A **vocabulary** is the set of unique units the model knows about. Here it is built with `sorted(set(allwords))`.
* Two dictionaries make encoding reversible: **`word2idx`** (word to integer) and **`idx2word`** (integer to word).
* An **encoder** turns a sentence into a list of integers and a **decoder** turns them back. `decoder(encoder(x)) == x` (after lowercasing) is the basic correctness check.
* Plotting the token sequence shows that repeated words get repeated integers, which is the beginning of "the model can see patterns".
* Looking at the **context window** around a target token (the words immediately before and after it) previews how language models learn from neighbouring tokens.

### Lesson 03 - Vector Representation Spaces *(upcoming)*
* To be filled in as I complete the lesson.

---

## 🗂️ Folder structure

```text
01 - Text to Vectors Basics/
│
├── README.md                                   ← you are here
│
├── 01 - Welcome and Overview/
│   └── README.md
│
├── 02 - Turning Text into Numeric Tokens/
│   ├── README.md
│   ├── text2tokens.ipynb                       ← notebook (run in Jupyter / Colab)
│   └── text2tokens.py                          ← same code exported as a script
│
└── 03 - Vector Representation Spaces/
    └── README.md
```

---

## ▶️ How to run the code

### Option A: Google Colab
1. Open `text2tokens.ipynb` from the lesson 02 folder.
2. Upload it to [Google Colab](https://colab.research.google.com/) (or open it from GitHub).
3. Run all cells from the top.

### Option B: Local Jupyter

```bash
# from the repository root
pip install numpy matplotlib jupyter
jupyter notebook
# then open: 01 - Text to Vectors Basics/02 - Turning Text into Numeric Tokens/text2tokens.ipynb
```

### Option C: As a plain script

```bash
cd "01 - Text to Vectors Basics/02 - Turning Text into Numeric Tokens"
python text2tokens.py
```

> `text2tokens.py` is an automatic export of the notebook, so some lines only make sense inside a notebook (for example the bare expressions that display values). When run as a script, only the `print` calls and the matplotlib window produce visible output. The notebook is the better way to follow along.

**Requirements:** Python 3.8+, `numpy`, `matplotlib`. The regular-expression module `re` is part of the standard library.

---

## 📖 Key terms

| Term | Meaning |
|------|---------|
| **Token** | The basic unit a model reads. In this lesson, one token is one lowercase word. In real LLMs it is often a word piece or even a single byte. |
| **Tokenization** | The process of splitting text into tokens. |
| **Vocabulary (lexicon)** | The set of all unique tokens the model knows. |
| **Token ID / index** | The integer assigned to a token. |
| **Encoder** | Function mapping text to a list of token IDs. |
| **Decoder** | Function mapping token IDs back to text. |
| **Corpus** | The body of text used to build the vocabulary and train a model. |
| **Context window** | The tokens surrounding a position that the model can use as information. |
| **Embedding** | A learned vector of numbers representing a token. Covered in lesson 03. |

---

## ⚠️ Limitations of the current implementation

The word-level tokenizer built here is deliberately simple. It is a teaching tool, and it has known weaknesses:

* **Unknown words crash it.** `encoder('hello')` raises a `KeyError` because `hello` is not in the vocabulary (no `<UNK>` token yet).
* **Punctuation is not handled.** `'be,'` and `'be'` would be different tokens, and `'be,'` is not in the vocabulary.
* **Whitespace is fragile.** The encoder splits on a single space, so double spaces or a trailing space produce an empty string `''`, which also raises a `KeyError`.
* **Tiny corpus.** Three sentences give 28 words and a vocabulary of 20. Real models are trained on billions of tokens.
* **IDs are arbitrary.** The index of a word is just its alphabetical position. Nothing about the number `16` says anything about the word `to`. Giving numbers meaning is exactly what embeddings are for.

These are not mistakes to hide. They are the motivation for subword tokenizers such as Byte Pair Encoding, special tokens, and embeddings, which come later.

---

## ⏭️ What comes next

* **Lesson 03 - Vector Representation Spaces:** turn each integer ID into a vector, and see how similar words end up near each other.
* Then: build a dataset using context windows, and move toward the first neural network that predicts the next token.

Back to the [repository root](../README.md).
