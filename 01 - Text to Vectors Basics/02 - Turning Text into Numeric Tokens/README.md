# 02 - Turning Text into Numeric Tokens

> **Lesson goal:** build a word-level tokenizer from scratch, first on three sentences, then on real books, and understand why real LLMs move beyond it.

This lesson has two folders and three notebooks. Each folder has its own detailed README.

| Folder | Notebook | What it covers | Explainer |
|--------|----------|----------------|-----------|
| [`Text2Token_basics`](./Text2Token_basics/) | [`text2tokens.ipynb`](./Text2Token_basics/text2tokens.ipynb) | **Part 1, toy example:** 3 sentences, vocabulary of 20, `word2idx` / `idx2word`, encoder and decoder, token plot, context around a token | [PDF pages 1-2](./Text2Token_basics/Text_to_Token_Explained.pdf) |
| [`Text2Token_basics`](./Text2Token_basics/) | [`text2tokens_intermediate.ipynb`](./Text2Token_basics/text2tokens_intermediate.ipynb) | **Part 2, whole novel (new):** *Frankenstein* cleaned and tokenized, 73,888 words → 7,243 tokens, random and Brownian tokens, word-length / Zipf / Heaps plots, out-of-vocabulary words and an `<\|unk\|>` token | [PDF pages 3-4](./Text2Token_basics/Text_to_Token_Explained.pdf) |
| [`Preparing_text`](./Preparing_text/) | [`preparing_text_for_tokenization.ipynb`](./Preparing_text/preparing_text_for_tokenization.ipynb) | **Real-text cleaning walkthrough:** download a book, remove unwanted characters, split, build vocabulary, encode and decode | [PDF](./Preparing_text/Preparing_Text_for_Tokenization_Explained.pdf) |

Every notebook also has a `.py` export next to it.

---

## 🔄 The pipeline

```text
Raw text ─► clean (artefacts, non-ASCII, digits, case) ─► split (punctuation + whitespace)
        ─► vocabulary  sorted(set(words)) ─► word2idx / idx2word ─► encoder / decoder
        ─► <|unk|> for words outside the vocabulary ─► (next lesson) embeddings
```

## 📊 Numbers at a glance

| | Part 1 (toy) | Part 2 (*Frankenstein*) |
|---|---|---|
| Words after cleaning | 28 | 73,888 |
| Vocabulary size | 20 | 7,243 (+1 `<\|unk\|>`) |
| Words used only once | — | 3,076 (42% of the vocabulary) |
| Text covered by the top 100 words | — | 53% |
| Unknown words handled | no (`KeyError`) | yes (`<\|unk\|>`) |

## 🧠 Main lessons

* Tokenization must be **deterministic and reversible**: the same text always gets the same IDs, and decoding recovers the words.
* **Token IDs are labels, not quantities.** They are alphabetical positions; embeddings (lesson 03) give them meaning.
* Real text needs **consistent cleaning**, and every cleaning choice changes the vocabulary.
* Word frequencies follow **Zipf's law** and the vocabulary keeps growing with more text (**Heaps' law**), so word-level vocabularies do not scale.
* An **`<|unk|>`** token prevents crashes on new words but loses information; **subword tokenizers (BPE)** are the real solution.

## ▶️ How to run

Open any notebook in Google Colab or Jupyter and run all cells. Requirements: `numpy`, `matplotlib`, `requests` (the real-book notebooks download their text, so they need internet access).

**Next lesson:** [03 - Vector Representation Spaces](../03%20-%20Vector%20Representation%20Spaces/)

Back to the [topic overview](../README.md).
