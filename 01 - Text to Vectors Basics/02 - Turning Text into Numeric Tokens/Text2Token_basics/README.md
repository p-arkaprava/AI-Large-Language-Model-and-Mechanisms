# Text2Token_basics: Turning Text into Numeric Tokens (toy example)

> **Lesson goal:** build a tokenizer from scratch. Take raw sentences, split them into words, build a vocabulary, assign every word an integer, and write an encoder and decoder that convert text to numbers and back.

📄 **Files in this folder**

| File | Description |
|------|-------------|
| [`text2tokens.ipynb`](./text2tokens.ipynb) | The notebook with all code and outputs |
| [`text2tokens.py`](./text2tokens.py) | The same code exported as a Python script |
| `README.md` | These notes |

---

## 📌 Table of Contents

1. [Why tokenization matters](#-why-tokenization-matters)
2. [Pipeline overview](#-pipeline-overview)
3. [Step-by-step walkthrough](#-step-by-step-walkthrough)
4. [The vocabulary table](#-the-vocabulary-table)
5. [Encoder and decoder functions](#-encoder-and-decoder-functions)
6. [Visualizing tokens](#-visualizing-tokens)
7. [Exploring context around a token](#-exploring-context-around-a-token)
8. [Design choices and limitations](#-design-choices-and-limitations)
9. [From this toy tokenizer to real LLM tokenizers](#-from-this-toy-tokenizer-to-real-llm-tokenizers)
10. [Common pitfalls and debugging notes](#-common-pitfalls-and-debugging-notes)
11. [Exercises to try](#-exercises-to-try)
12. [Key takeaways](#-key-takeaways)

---

## 🎯 Why tokenization matters

A neural network is a pile of arithmetic. It multiplies, adds, and applies functions to numbers. It has no concept of the string `"to be or not to be"`.

**Tokenization is the interface between human text and the model.** It defines:

* **What the model can read:** only items in the vocabulary.
* **How long sequences are:** more tokens per sentence means more compute.
* **How big the model's input and output layers are:** one entry per vocabulary item.
* **Whether text can be recovered:** a good tokenizer is reversible.

Every other component of an LLM sits on top of this step. A bug here is invisible but corrupts everything after it.

---

## 🔄 Pipeline overview

```text
Raw text (list of sentences)
        │
        ▼  join + lowercase
One long lowercase string
        │
        ▼  split on whitespace
List of words  (allwords)              28 words
        │
        ▼  set() + sorted()
Vocabulary     (vocab)                 20 unique words
        │
        ▼  enumerate()
word2idx / idx2word dictionaries
        │
        ▼  lookup
Integer tokens (text_as_int)           [0, 13, 17, 2, 6, ...]
        │
        ▼  reverse lookup
Text recovered ("decoder")
```

---

## 🪜 Step-by-step walkthrough

### Step 1. Start with raw text

```python
text = ['All that we are is the result of what we have thought',
        'To be or not to be that is the question',
        'Be yourself everyone is already taken']
```

Three short quotations act as a tiny **corpus**. Small enough to inspect by eye, large enough to contain repeated words (`be`, `is`, `that`, `to`, `we`, `the`).

### Step 2. Split text into words

```python
import re
re.split(r'\s', text[0])
# ['All', 'that', 'we', 'are', 'is', 'the', 'result', 'of', 'what', 'we', 'have', 'thought']
```

`re.split(r'\s', ...)` cuts the string at every whitespace character. This is the simplest possible tokenizer: **one word = one token**.

Joining it back confirms the split loses nothing (for this input):

```python
' '.join(re.split(r'\s', text[0]))
# 'All that we are is the result of what we have thought'
```

### Step 3. Combine all samples and lowercase

```python
allwords = re.split(r'\s', ' '.join(text).lower())
```

* `' '.join(text)` merges the three sentences into one string.
* `.lower()` makes `Be` and `be` the same token. Without it, `To`/`to` and `Be`/`be` would waste vocabulary slots as separate entries.
* The result is a list of **28 words**, including repeats.

### Step 4. Build the vocabulary

```python
vocab = sorted(set(allwords))
print(len(allwords))   # 28
print(len(vocab))      # 20
```

| Function | Role |
|----------|------|
| `set(allwords)` | Removes duplicates, leaving the unique words |
| `sorted(...)` | Gives a **deterministic order** (alphabetical) so the same text always produces the same IDs. `set` alone has no guaranteed order. |

28 words in the text but only 20 unique ones, so repeated words will share an ID.

### Step 5. Build the lookup dictionaries

```python
# word -> integer  (used to ENCODE)
word2idx = {}
for i, word in enumerate(vocab):
    word2idx[word] = i

# integer -> word  (used to DECODE)
idx2word = {}
for i, word in enumerate(vocab):
    idx2word[i] = word
```

Two dictionaries are needed because dictionary lookup only works in one direction. They are exact inverses of each other, which is what makes the conversion reversible.

### Step 6. Generate "fake quotes" (sanity check and fun)

```python
import numpy as np
randidx = np.random.randint(0, len(vocab), size=10)
' '.join([idx2word[i] for i in randidx])
# e.g. 'question we thought that or result is already have taken'
```

Picking random integers and decoding them produces random word salad. This is a first, very crude "language model": **no knowledge of grammar at all**, only random draws from the vocabulary. It shows what a trained model must improve on, since it has to choose the *next* token with good probabilities, not uniformly at random.

### Step 7. Convert the whole text to integers

```python
text_as_int = [word2idx[word] for word in allwords]
# [0, 13, 17, 2, 6, 14, 11, 8, 18, 17, 5, 15, 16, 3, 9, 7, 16, 3, 13, 6, 14, 10, 3, 19, 4, 6, 1, 12]
```

This is the moment tokenization happens. Every word is replaced by its index. Note that repeated words get repeated numbers: `we` is always `17`, `be` is always `3`, `is` is always `6`.

### Step 8. Convert back and inspect

```python
for tokeni in text_as_int:
    print(f'Token:{tokeni} : {idx2word[tokeni]}')
```

```text
Token:0 : all
Token:13 : that
Token:17 : we
Token:2 : are
Token:6 : is
...
```

Printing token and word side by side is a quick way to verify the mapping is consistent in both directions.

---

## 📋 The vocabulary table

The sorted vocabulary of the corpus, with the integer each word receives:

| ID | Word | ID | Word | ID | Word | ID | Word |
|----|------|----|------|----|------|----|------|
| 0 | all | 5 | have | 10 | question | 15 | thought |
| 1 | already | 6 | is | 11 | result | 16 | to |
| 2 | are | 7 | not | 12 | taken | 17 | we |
| 3 | be | 8 | of | 13 | that | 18 | what |
| 4 | everyone | 9 | or | 14 | the | 19 | yourself |

**Most frequent words in the corpus:** `is` (3 times), `be` (3 times), then `that`, `we`, `the`, and `to` (2 times each).

Notice that IDs are alphabetical. `all` is `0` because it sorts first, not because it is special. **The numbers carry no meaning**: `16` is not "bigger" or "more important" than `3`. This is why raw integer IDs are not fed to a network directly, and why embeddings (next lesson) exist.

---

## 🔁 Encoder and decoder functions

Wrapping the logic in functions makes it reusable for any new sentence:

```python
def encoder(text):
    words = re.split(' ', text.lower())
    return [word2idx[w] for w in words]

def decoder(indices):
    return [idx2word[i] for i in indices]
```

Usage:

```python
encoder('to be')              # [16, 3]
decoder([16, 3])              # ['to', 'be']
decoder(encoder('to be'))     # ['to', 'be']   <- round trip works
```

| Function | Input | Output | Idea |
|----------|-------|--------|------|
| `encoder` | a string | list of integers | lowercase, split, look up each word in `word2idx` |
| `decoder` | list of integers | list of words | look up each integer in `idx2word` |

**The round-trip test** `decoder(encoder(x))` returning the original words is the basic correctness check for any tokenizer. Note that the decoder returns a *list* of words, so use `' '.join(...)` to get a sentence back. Case information is lost because everything was lowercased.

---

## 📈 Visualizing tokens

```python
alltext = ' '.join(text)
tokens = encoder(alltext)

_, ax = plt.subplots(1, figsize=(12, 5))
ax.plot(tokens, 'ks', markersize=12, markerfacecolor=[.7, .7, .9])
ax.set(xlabel='Word index', yticks=range(len(vocab)))
ax.grid(linestyle='--', axis='y')

ax2 = ax.twinx()                        # invisible twin axis for labels
ax2.plot(tokens, alpha=0)
ax2.set(yticks=range(len(vocab)), yticklabels=vocab)
plt.show()
```

How to read the plot:

* **x-axis:** position of the word in the text (0 to 27).
* **y-axis (left):** the token ID. **y-axis (right):** the word, via the twin axis.
* Squares at the **same height** are the **same word** repeated, for example the three `be` tokens and the three `is` tokens.

Why bother? It makes the structure of the data visible. A sequence of text is just a sequence of integers, and repeated words appear as repeated levels.

---

## 🔍 Exploring context around a token

```python
targetWord = 'to'
targetIdx  = word2idx[targetWord]

targetLocs = np.where(np.array(allwords) == targetWord)[0]
print(f"'{targetWord}' appears at indices {targetLocs}")

for t in targetLocs:
    print(tokens[t-1:t+2])
    print(' '.join(allwords[t-1:t+2]), '\n')
```

Output:

```text
'to' appears at indices [12 16]

[15, 16, 3]
thought to be

[7, 16, 3]
not to be
```

* `np.where(... == targetWord)` finds every position where the word occurs.
* `tokens[t-1:t+2]` takes the word **before**, the word itself, and the word **after**: a context window of size 1 on each side.

**This is a preview of how language models learn.** The word `to` has the same ID (`16`) in both places, but its *neighbours* differ (`thought _ be` vs `not _ be`). Models learn the meaning and usage of a token from the company it keeps. Building training examples from these windows is the next big step.

> ⚠️ **Edge case:** if the target word is the very first token (index 0), `tokens[t-1:t+2]` becomes `tokens[-1:2]`, which is an empty slice. Boundary handling will be needed when this is turned into a proper dataset.

---

## ⚖️ Design choices and limitations

| Choice in this lesson | Reason | Limitation |
|-----------------------|--------|------------|
| Word-level tokens | Simplest and most intuitive | Huge vocabulary for real text; cannot handle new words |
| Lowercasing | Merges `To` and `to` | Loses capitalization information (names, sentence starts) |
| `sorted(set(...))` | Deterministic, unique vocabulary | IDs are arbitrary alphabetical positions |
| Split on whitespace | Very simple | Punctuation sticks to words (`'be,'`) |
| Plain dictionaries | Fast O(1) lookup, easy to read | No special tokens (padding, unknown, end-of-text) |

---

## 🚀 From this toy tokenizer to real LLM tokenizers

The ideas here are exactly the ones real systems use. What changes is the unit of splitting and the scale.

| Aspect | This lesson | Modern LLM tokenizers |
|--------|-------------|----------------------|
| Unit | Whole words | **Subwords** (e.g. Byte Pair Encoding), pieces such as `token` + `ization` |
| Unknown words | `KeyError` | Broken into smaller pieces, so almost any text can be represented |
| Vocabulary size | 20 | Tens of thousands to a few hundred thousand |
| Special tokens | None | Padding, unknown, start/end-of-text markers |
| Punctuation and spaces | Not handled | Handled as part of the token rules |
| Vocabulary built from | 3 sentences | A large corpus, using a learning algorithm |
| Core mechanism | `word2idx` and `idx2word` | Same idea: a token-to-ID table and its inverse |

Why subwords? Word-level vocabularies explode in size and fail on rare or new words. Character-level vocabularies are tiny but make sequences very long. Subwords sit in the middle and are the standard choice for modern LLMs.

---

## 🐞 Common pitfalls and debugging notes

These are real failures of the current code, verified by running it:

```python
encoder('hello world')   # KeyError: 'hello'   -> word not in vocabulary
encoder('to be, or')     # KeyError: 'be,'     -> punctuation attached to the word
encoder('to be ')        # KeyError: ''        -> trailing space makes an empty "word"
encoder('to  be')        # KeyError: ''        -> double space makes an empty "word"
```

Possible fixes, to be implemented later:

* add an `<UNK>` entry to the vocabulary and use `word2idx.get(w, word2idx['<UNK>'])`;
* separate punctuation with a regex such as `re.findall(r"\w+|[^\w\s]", text)`;
* use `text.split()` (no argument), which collapses repeated whitespace;
* reserve special tokens such as `<PAD>` and `<EOS>`.

Other things to remember:

* The notebook uses `re.split(r'\s', ...)` in one place and `re.split(' ', ...)` in the encoder. They behave differently on tabs, newlines, and multiple spaces.
* The vocabulary depends on the corpus. If the corpus changes, **all IDs can change**, and any saved tokens become invalid. Always save the vocabulary together with the data.
* `np.random.randint` gives different output every run, so the fake quotes differ each time. Set `np.random.seed(...)` for reproducibility.

---

## 🧪 Exercises to try

1. Add a fourth sentence and observe how the vocabulary size and several IDs change.
2. Add an `<UNK>` token so the encoder never crashes on unseen words.
3. Handle punctuation so that `"be,"` becomes `["be", ","]`.
4. Compute word frequencies with `collections.Counter` and re-assign IDs by frequency instead of alphabetically.
5. Write a `decode_to_string(indices)` function that returns a readable sentence.
6. Generalize the context print to window size `k` and handle the first and last positions safely.
7. Plot a histogram of token frequencies. What shape does it have?

---

## ✅ Key takeaways

* **Tokenization converts text to integers and back, and it must be reversible and consistent.**
* A **vocabulary** is the sorted set of unique tokens; `word2idx` and `idx2word` are its two lookup directions.
* **Repeated words share the same ID**, which is how patterns become visible to a model.
* **Integer IDs are labels, not quantities.** They carry no meaning on their own.
* **Context** (the neighbours of a token) is the signal language models learn from.
* A word-level tokenizer is simple but brittle. Real LLMs use subword tokenizers built on the same two-table idea.

```text
Raw Text → Text Processing → Words → Vocabulary → word2idx → Integer Tokens → Neural Network
```

**Related:** the real-book version is in [`../Preparing_text`](../Preparing_text/).

**Next lesson:** [03 - Vector Representation Spaces](../../03%20-%20Vector%20Representation%20Spaces/): turning these integers into meaningful vectors.

Back to the [topic overview](../../README.md).
