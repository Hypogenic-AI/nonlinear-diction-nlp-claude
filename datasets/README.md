# Downloaded Datasets

This directory contains datasets for the Non-linear Diction research project.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: GoEmotions (Primary)

### Overview
- **Source:** [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (simplified config)
- **Size:** 54,263 samples (43,410 train / 5,426 val / 5,427 test)
- **Format:** HuggingFace Dataset
- **Task:** Multi-label emotion classification (27 emotions + neutral)
- **Labels:** admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise
- **License:** Apache 2.0

### Why Relevant
- Used by DLR-SC style vectors paper for emotion steering evaluation
- Known that **disgust** (AUC 0.51) and **surprise** (AUC 0.38) are hard to steer linearly
- 27 emotion categories provide a natural spectrum of steering difficulty
- Fine-grained emotions test the limits of linear representation

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
dataset.save_to_disk("datasets/go_emotions/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/go_emotions/data")
```

---

## Dataset 2: Yelp Polarity (Baseline)

### Overview
- **Source:** [fancyzhx/yelp_polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity)
- **Size:** 598,000 samples (560,000 train / 38,000 test)
- **Format:** HuggingFace Dataset
- **Task:** Binary sentiment classification
- **Labels:** 0 (negative), 1 (positive)
- **License:** See source

### Why Relevant
- Sentiment is the best-studied domain for steering vectors — serves as positive control
- Known to work well with linear steering (AUC ~0.99 at optimal layers)
- Baseline to compare against harder style dimensions

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("fancyzhx/yelp_polarity")
dataset.save_to_disk("datasets/yelp_polarity/data")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/yelp_polarity/data")
```

---

## Dataset 3: Tiny Shakespeare (Style Transfer)

### Overview
- **Source:** [karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
- **Size:** ~1.1MB of text (~40,000 lines)
- **Format:** Raw text file
- **Task:** Shakespearean/archaic writing style
- **License:** Public domain

### Why Relevant
- Shakespearean style was tested in the DLR-SC style vectors paper
- Required higher λ values for steering than sentiment — indicating moderate difficulty
- Good intermediate case between easy (sentiment) and hard (disgust/surprise) steering targets

### Download Instructions

```python
import requests
r = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
with open('datasets/tiny_shakespeare/shakespeare.txt', 'w') as f:
    f.write(r.text)
```

### Loading
```python
with open('datasets/tiny_shakespeare/shakespeare.txt') as f:
    text = f.read()
```

---

## Dataset 4: WritingPrompts (Diverse Styles)

### Overview
- **Source:** [euclaise/writingprompts](https://huggingface.co/datasets/euclaise/writingprompts)
- **Size:** ~300K stories from 97K prompts (full); 5,000 sample stored locally
- **Format:** HuggingFace Dataset (streaming) / local JSON sample
- **Task:** Creative writing across diverse implicit styles (horror, romance, sci-fi, humor, literary, etc.)
- **License:** Reddit user-generated content

### Why Relevant
- Contains naturally diverse writing styles without explicit labels
- Could be labeled post-hoc by genre/style for steering experiments
- Tests style categories that haven't been explored in steering literature

### Download Instructions

```python
from datasets import load_dataset
# Full dataset (streaming):
ds = load_dataset("euclaise/writingprompts", split="train", streaming=True)
# Or load specific split:
ds = load_dataset("euclaise/writingprompts", split="train[:5000]")
```

---

## Dataset 5: Blog Authorship Corpus (Novel)

### Overview
- **Source:** [barilan/blog_authorship_corpus](https://huggingface.co/datasets/barilan/blog_authorship_corpus)
- **Size:** 681,288 posts from 19,320 bloggers (~140M words)
- **Format:** HuggingFace Dataset (requires trust_remote_code=True)
- **Task:** Authorship attribution with demographic style variation
- **Labels:** Author gender (male/female), age group (13-17, 23-27, 33-47), industry, astrological sign
- **License:** Research use

### Why Relevant
- Large-scale dataset with demographic-correlated style variation
- NOT previously studied for steering vector evaluation — potential novel contribution
- Age/gender style differences may have varying linearity of representation

### Download Instructions

```python
from datasets import load_dataset
# Note: requires trust_remote_code=True due to legacy script
dataset = load_dataset("barilan/blog_authorship_corpus", trust_remote_code=True)
dataset.save_to_disk("datasets/blog_authorship/data")
```

### Notes
- Legacy HuggingFace script format — may need `trust_remote_code=True`
- If HuggingFace load fails, original data available at https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
