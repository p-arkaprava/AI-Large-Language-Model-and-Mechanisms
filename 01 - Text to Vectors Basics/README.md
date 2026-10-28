# 01 - Text to Vectors Basics

> **Big idea:** a neural network cannot read words. It can only do arithmetic on numbers.
> This topic covers the  first half of the bridge between human language and a language model: **turning raw text into numbers**, and then **turning those numbers into vectors that carry meaning**.

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
| 02 | Turning Text into Numeric Tokens | [`02 - Turning Text into Numeric Tokens`](./02%20-%20Turning%20Text%20into%20Numeric%20Tokens/) | Two folders: `Text2Token_basics` (Part 1: toy tokenizer, encoder/decoder, context windows; Part 2: tokenizing all of *Frankenstein* with plots, Zipf/Heaps statistics and an `<\|unk\|>` token) and `Preparing_text` (cleaning and tokenizing a real book) | ✅ Done |
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

**Part 2 (intermediate): a whole novel** (`text2tokens_intermediate.ipynb`)
* Real text must be **cleaned** first: encoding artefacts (mojibake), non-ASCII characters, digits and case are removed, then the text is split on punctuation + whitespace with one regex.
* *Frankenstein* gives **73,888 words** but only **7,243 unique tokens**, and the encoder/decoder round trip works on the entire book.
* **Zipf's law:** word counts fall roughly as 1/rank. *the* appears 4,387 times, while 42% of the vocabulary appears only once. The top 100 words cover 53% of the text.
* **Heaps' law:** the vocabulary keeps growing as more text is read, so a word-level vocabulary has no natural size limit.
* Random token IDs give word salad; a **Brownian (±1) walk** over IDs gives alphabetical neighbours (*strife, strive, strong, struggle*), which shows that closeness of IDs is not closeness of meaning.
* A new sentence (*"Hi I chatgpt am going home this week"*) has 3 out-of-vocabulary words. An **`<|unk|>` special token** stops the encoder from crashing, at the cost of losing those words.

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
│   ├── Text2Token_basics/                      ← toy example
│   │   ├── README.md
│   │   ├── text2tokens.ipynb                   ← Part 1 notebook (run in Jupyter / Colab)
│   │   ├── text2tokens.py                      ← Part 1 exported as a script
│   │   ├── text2tokens_intermediate.ipynb      ← Part 2: whole novel, plots, <|unk|>
│   │   ├── text2tokens_intermediate.py         ← Part 2 exported as a script
│   │   └── Text_to_Token_Explained.pdf         ← 4-page explainer (Parts 1 and 2)
│   └── Preparing_text/                         ← real book: clean, split, vocabulary
│       ├── README.md
│       ├── preparing_text_for_tokenization.ipynb
│       ├── preparing_text_for_tokenization.py
│       └── Preparing_Text_for_Tokenization_Explained.pdf
│
└── 03 - Vector Representation Spaces/
    └── README.md
```

---

## ▶️ How to run the code

### Option A: Google Colab
1. Open `text2tokens.ipynb` or `text2tokens_intermediate.ipynb` (in `Text2Token_basics`) or `preparing_text_for_tokenization.ipynb` (in `Preparing_text`) from the lesson 02 folder.
2. Upload it to [Google Colab](https://colab.research.google.com/) (or open it from GitHub).
3. Run all cells from the top.

### Option B: Local Jupyter

```bash
# from the repository root
pip install numpy matplotlib requests jupyter
jupyter notebook
# then open the notebooks inside:
# 01 - Text to Vectors Basics/02 - Turning Text into Numeric Tokens/
```

### Option C: As a plain script

```bash
cd "01 - Text to Vectors Basics/02 - Turning Text into Numeric Tokens/Text2Token_basics"
python text2tokens.py
python text2tokens_intermediate.py     # downloads Frankenstein, needs internet
```

> `text2tokens.py` is an automatic export of the notebook, so some lines only make sense inside a notebook (for example the bare expressions that display values). When run as a script, only the `print` calls and the matplotlib window produce visible output. The notebook is the better way to follow along.

**Requirements:** Python 3.8+, `numpy`, `matplotlib`, and `requests` (the `Preparing_text` and `text2tokens_intermediate` notebooks download a book, so they need internet access). The regular-expression module `re` is part of the standard library.

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

* **Unknown words crash it.** `encoder('hello')` raises a `KeyError` because `hello` is not in the vocabulary (no `<UNK>` token yet in Part 1; Part 2 adds an `<|unk|>` fallback, which avoids the crash but maps every unknown word to the same ID).
* **Punctuation is not handled.** `'be,'` and `'be'` would be different tokens, and `'be,'` is not in the vocabulary.
* **Whitespace is fragile.** The encoder splits on a single space, so double spaces or a trailing space produce an empty string `''`, which also raises a `KeyError`.
* **Tiny corpus.** Three sentences give 28 words and a vocabulary of 20. Part 2 scales this to a full novel (73,888 words, 7,243 tokens), but real models are trained on billions of tokens.
* **The vocabulary does not scale.** On the novel, 42% of the vocabulary appears only once and the vocabulary keeps growing with more text (Zipf and Heaps laws). This is the main motivation for subword tokenizers.
* **IDs are arbitrary.** The index of a word is just its alphabetical position. Nothing about the number `16` says anything about the word `to`. Giving numbers meaning is exactly what embeddings are for.

These are not mistakes to hide. They are the motivation for subword tokenizers such as Byte Pair Encoding, special tokens, and embeddings, which come later.

---

## ⏭️ What comes next

* **Lesson 03 - Vector Representation Spaces:** turn each integer ID into a vector, and see how similar words end up near each other.
* Then: build a dataset using context windows, and move toward the first neural network that predicts the next token.

Back to the [repository root](../README.md).
