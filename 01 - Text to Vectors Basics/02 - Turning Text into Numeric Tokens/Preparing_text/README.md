# Preparing_text: Preparing Real Text for Tokenization

> **Goal:** take a real book from the internet, clean it, split it into words, build a vocabulary, and convert the words to integer tokens and back. This is the same idea as `Text2Token_basics`, but on real, messy text.

📄 **Files in this folder**

| File | Description |
|------|-------------|
| [`preparing_text_for_tokenization.ipynb`](./preparing_text_for_tokenization.ipynb) | The notebook with all code and outputs |
| `README.md` | These notes |

---

## 📌 Table of Contents

1. [Why text must be prepared](#-why-text-must-be-prepared)
2. [Pipeline overview](#-pipeline-overview)
3. [Libraries used](#-libraries-used)
4. [Step-by-step walkthrough](#-step-by-step-walkthrough)
5. [Numbers at a glance](#-numbers-at-a-glance)
6. [What the cleaning does to real words](#-what-the-cleaning-does-to-real-words)
7. [Encoder and decoder](#-encoder-and-decoder)
8. [Testing and verification](#-testing-and-verification)
9. [Observations and limitations](#-observations-and-limitations)
10. [Possible improvements](#-possible-improvements)
11. [Differences from the basics notebook](#-differences-from-the-basics-notebook)
12. [Exercises to try](#-exercises-to-try)
13. [Key takeaways](#-key-takeaways)

---

## 🎯 Why text must be prepared

Three sentences are easy to handle. A real book is not. It contains:

* **capital letters**, so `The` and `the` would become two different tokens;
* **punctuation**, so `end.` and `end` would become two different tokens;
* **numbers, symbols and special characters** such as curly quotes and long dashes;
* **line breaks and uneven spacing**;
* **headers and boilerplate** that are not part of the story.

If these are not handled first, the vocabulary fills up with near-duplicates and noise. Preparing the text is therefore the first real step of tokenization, and every choice made here directly changes the vocabulary the model will learn from.

---

## 🔄 Pipeline overview

```text
Download book (requests)              179,799 characters, one long string
        │
        ▼  replace odd characters with spaces
        ▼  remove non-ASCII characters
        ▼  remove digits
        ▼  lowercase
Cleaned text
        │
        ▼  split on punctuation and whitespace
        ▼  strip and drop empty pieces
        ▼  drop single-character words
List of words  (words)                30,699 words
        │
        ▼  sorted(set(words))
Vocabulary     (vocab)                4,589 unique tokens
        │
        ▼  enumerate()
word2idx / idx2word
        │
        ▼
encoder(words, word2idx)  →  NumPy array of token IDs
decoder(ids, idx2word)    →  readable string
```

---

## 📚 Libraries used

| Library | Used for |
|---------|----------|
| `numpy` | Storing token IDs as an integer array; random start position |
| `requests` | Downloading the book text from the internet |
| `re` | Regular expressions for replacing, removing and splitting text |
| `string` | `string.punctuation`, the list of punctuation characters |

> The notebook downloads the book from Project Gutenberg, so it needs an internet connection the first time it runs.

---

## 🪜 Step-by-step walkthrough

### Step 1. Get the book

```python
book = requests.get('https://www.gutenberg.org/files/35/35-0.txt')
text = book.text
print(type(text))   # <class 'str'>
print(len(text))    # 179799
```

The whole book arrives as **one string of 179,799 characters**. The beginning shows it also contains a header (`*** START OF THE PROJECT GUTENBERG EBOOK 35 ***`), a title and a table of contents before the story starts.

### Step 2. Replace unwanted character strings

```python
strings2replace = [
    '\r\n\r\nâ\x80\x9c',  # new paragraph
    'â\x80\x9c',          # open quote
    'â\x80\x9d',          # close quote
    '\r\n',               # new line
    'â\x80\x94',          # hyphen
    'â\x80\x99',          # single apostrophe
    'â\x80\x98',          # single quote
    '_',                  # underscore, used for stressing
]

for str2match in strings2replace:
    regexp = re.compile(r'%s' % str2match)
    text = regexp.sub(' ', text)
```

**What these odd strings are:** when a UTF-8 file is read with the wrong encoding, a curly quote such as `“` turns into the junk sequence `â\x80\x9c`. This list removes that junk. The commented line `'â\x80\x9d'.encode('latin1').decode('utf8')` in the notebook shows how to reverse the mix-up.

**What happened in this run:** the printed text already showed proper curly quotes and dashes, so the junk sequences were not present and this loop had little to match. The next step is what actually removed those characters. The loop is still a useful safeguard if the text is ever decoded incorrectly.

### Step 3. Remove non-ASCII characters, digits, and capitals

```python
text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # non-ASCII  -> space
text = re.sub('\d+', '', text)               # digits     -> removed
text = text.lower()                          # lowercase
```

| Line | Effect | Example |
|------|--------|---------|
| `[^\x00-\x7F]+` | Any character outside basic ASCII becomes a space | `Traveller’s` → `traveller s` |
| `\d+` | Deletes every run of digits | `EBOOK 35` → `ebook ` |
| `.lower()` | One spelling per word | `The` → `the` |

> ⚠️ Python printed a `SyntaxWarning: invalid escape sequence '\d'`. The pattern still works, but it should be written as a raw string: `r'\d+'`.

### Step 4. Split the text into words

```python
import string
print(string.punctuation)    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
puncts4re = f'[{string.punctuation}\s]+'

words = re.split(puncts4re, text)
words = [item.strip() for item in words if item.strip()]
words = [item for item in words if len(item) > 1]
```

* The pattern `[<punctuation>\s]+` means *"one or more punctuation marks or whitespace characters in a row"*. Splitting on it cuts the text at every gap between words.
* `item.strip()` and the `if item.strip()` check remove empty strings that splitting can leave behind.
* `len(item) > 1` **removes every single-character word** (see [Observations](#-observations-and-limitations)).

First words produced:

```text
['start', 'of', 'the', 'project', 'gutenberg', 'ebook', 'the', 'time', 'machine',
 'an', 'invention', 'by', 'wells', 'contents', 'introduction', 'ii', 'the', 'machine', ...]
```

### Step 5. Build the vocabulary

```python
vocab = sorted(set(words))

nWords = len(words)
nLex   = len(vocab)

print(f'{nWords} words')          # 30699 words
print(f' {nLex} unique tokens')   # 4589 unique tokens
```

`set` keeps one copy of each word, and `sorted` fixes a repeatable alphabetical order. `nWords` and `nLex` are saved for later use.

### Step 6. Build the lookup dictionaries

```python
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
```

Dictionary comprehensions do the same job as the loops in the basics notebook, in one line each. A sample of the mapping (every 87th entry):

| Word | ID | Word | ID | Word | ID |
|------|----|------|----|------|----|
| abandon | 0 | find | 1479 | sandals | 3393 |
| aimlessly | 87 | gold | 1740 | senses | 3480 |
| can | 522 | lamp | 2262 | sudden | 3915 |
| coat | 696 | plato | 2958 | wonderful | 4524 |

The IDs rise with the alphabet: `abandon` is `0`, `wonderful` is near the end.

---

## 📊 Numbers at a glance

| Quantity | Value |
|----------|-------|
| Characters in the raw download | 179,799 |
| Words kept after cleaning and filtering | 30,699 |
| Unique words (vocabulary size) | 4,589 |
| Average times each unique word appears | about 6.7 (30,699 ÷ 4,589) |
| Highest token ID | 4,588 |

---

## 🧹 What the cleaning does to real words

These examples are taken from the notebook's own output.

| Original text | After cleaning and splitting | After 1-letter filter |
|---------------|------------------------------|-----------------------|
| `by H. G. Wells` | `['by', 'h', 'g', 'wells']` | `['by', 'wells']` |
| `The Time Traveller’s Return` | `['the', 'time', 'traveller', 's', 'return']` | `['the', 'time', 'traveller', 'return']` |
| `I Introduction` (chapter heading) | `['i', 'introduction']` | `['introduction']` |
| `a sudden shock` | `['a', 'sudden', 'shock']` | `['sudden', 'shock']` |

The first two rows show why the filter was added: removing non-ASCII characters and punctuation leaves stray letters behind, and the filter removes them. The last two rows show its cost, since the real words `i` and `a` are removed too.

---

## 🔁 Encoder and decoder

```python
def encoder(words, encode_dict):
    idxs = np.zeros(len(words), dtype=int)       # empty integer array
    for i, w in enumerate(words):
        idxs[i] = encode_dict[w]                 # look up each word
    return idxs

def decoder(idxs, decode_dict):
    return ' '.join([decode_dict[i] for i in idxs])
```

| Function | Input | Output |
|----------|-------|--------|
| `encoder(words, word2idx)` | list of words and the encoding dictionary | NumPy integer array |
| `decoder(idxs, idx2word)` | token IDs and the decoding dictionary | one joined string |

Note that the dictionaries are **passed in as arguments** instead of being global variables, so the same functions work with any vocabulary. The encoder uses an explicit `for` loop on purpose; a list comprehension would also work.

---

## ✅ Testing and verification

```python
print(encoder(['the', 'time', 'machine'], word2idx))   # [4042 4109 2416]
print(decoder([1, 3, 10], idx2word))                   # abandoned abnormally absent
```

* `the time machine` encodes to `[4042 4109 2416]`.
* IDs `1`, `3`, `10` decode to `abandoned abnormally absent`. They are the earliest words alphabetically, which confirms that **IDs are alphabetical positions and carry no other meaning**.

### Round-trip test on a random passage

```python
startidx = np.random.choice(nWords)
idxs = np.arange(startidx, startidx + 10)
wordseq  = [words[i] for i in idxs]
tokenseq = encoder(wordseq, word2idx)
decoder(tokenseq, idx2word)
```

In the recorded run (start index 19284):

| Step | Result |
|------|--------|
| Words | `['the', 'afternoon', 'along', 'the', 'valley', 'of', 'the', 'thames', 'but', 'found']` |
| Token IDs | `[4042 72 102 4042 4322 2731 4042 4038 501 1599]` |
| Decoded | `the afternoon along the valley of the thames but found` |

The decoded text matches the original words. Notice that `the` is always `4042`: **the same word always gets the same ID**.

The notebook prints with `print(idxs), print('')`. This builds a small tuple, which is a compact trick to print two things on one line; it is harmless.

---

## ⚠️ Observations and limitations

1. **Apostrophes split words.** `traveller’s` becomes `traveller` and `s`, and `s` is then dropped. Possessives and contractions lose their ending.
2. **The 1-letter filter removes real words** such as `a` and `i`, as well as junk letters.
3. **Boilerplate stays in.** `start of the project gutenberg ebook` is part of the word list, and chapter numerals (`ii`, `iii`, `xv`) became vocabulary entries.
4. **Raw-string warnings.** `'\d+'` and `f'[...\s]+'` should be raw strings (`r'\d+'`, `rf'[...\s]+'`). Python shows a `SyntaxWarning`, and Python plans to treat this as an error in a future version.
5. **Unknown words fail.** `encoder` raises a `KeyError` for any word not in this book's vocabulary.
6. **The vocabulary belongs to this book.** A different text gives a different vocabulary and different IDs.
7. **Random start.** `np.random.choice` makes the last test differ on every run.
8. **Encoding depends on the download.** If the text is decoded incorrectly, curly quotes become sequences like `â\x80\x9c`. The notebook's replace list is the safeguard for this.

---

## 🔧 Possible improvements

* Use raw strings for all regular expressions.
* Keep apostrophes inside words with a pattern such as `[a-z]+(?:'[a-z]+)?`.
* Replace the 1-letter filter with a rule that keeps `a` and `i`.
* Cut the text at the Gutenberg start and end markers before processing.
* Add an `<UNK>` token for unseen words.
* Save `vocab` to a file so IDs stay the same between sessions.
* Set `np.random.seed(...)` for repeatable tests.

---

## 🔀 Differences from the basics notebook

| Aspect | `Text2Token_basics` | `Preparing_text` |
|--------|---------------------|------------------|
| Data | 3 hand-written sentences | A full book downloaded with `requests` |
| Cleaning | Lowercase only | Odd-character removal, non-ASCII removal, digit removal, lowercase |
| Splitting | Whitespace only | Punctuation and whitespace, with empty and 1-letter pieces removed |
| Vocabulary | 20 unique words | 4,589 unique words |
| Lookup tables | Explicit loops | Dictionary comprehensions |
| Encoder output | Python list | NumPy integer array |
| Dictionaries | Global variables | Passed as function arguments |
| Decoder output | List of words | One joined string |

---

## 🧪 Exercises to try

1. Rewrite the regular expressions as raw strings and confirm the warnings disappear.
2. Keep apostrophes so `traveller's` stays one word, and compare the vocabulary size.
3. Remove the 1-letter filter and see how many words and tokens change.
4. Cut the Gutenberg header and licence text and recount the vocabulary.
5. Use `collections.Counter` to find the 20 most common words and plot them.
6. Assign IDs by frequency instead of alphabetically.
7. Add an `<UNK>` token and encode a sentence containing a word not in the book.
8. Save the vocabulary with `json` and reload it in a new session.

---

## ✅ Key takeaways

* Real text must be **cleaned and split consistently** before it can be tokenized.
* Every cleaning choice (case, punctuation, filters) changes the vocabulary, for better or worse.
* A real book of 30,699 words needs only **4,589 unique tokens**, because words repeat.
* The same word always receives the same ID, and decoding reverses encoding exactly.
* Token IDs here are **alphabetical positions**, not meaningful numbers. Giving them meaning is the job of embeddings.

```text
Real Text → Clean → Split → Filter → Vocabulary → word2idx → Integer Tokens → Embeddings (next lesson)
```

Back to the [lesson overview](../README.md).
