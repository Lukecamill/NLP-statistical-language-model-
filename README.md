# Statistical Language Modelling with N-Grams

A Natural Language Processing project implementing statistical language models using **N-gram techniques** in Python.

The project covers the complete language-modelling pipeline, including corpus preprocessing, N-gram extraction, model construction, text probability estimation, and model evaluation.

## Overview

Statistical language models estimate the probability of sequences of words based on patterns observed in a training corpus.

This project implements an N-gram based language modelling system from the ground up, providing practical experience with core NLP concepts such as:

- Text preprocessing
- Corpus preparation
- Tokenisation
- N-gram extraction
- Statistical language modelling
- Model training and evaluation
- Language model efficiency testing

Rather than relying entirely on pre-built NLP models, the project implements the main language-modelling components directly in Python.

## N-Gram Language Models

An N-gram language model estimates the probability of a word based on the preceding words.

For example, a bigram model estimates:

```text
P(w₂ | w₁)
```

while a trigram model estimates:

```text
P(w₃ | w₁, w₂)
```

These probabilities can be learned from a corpus by analysing how frequently different word sequences occur.

The project explores how statistical language models can use these patterns to model natural language.

## Project Pipeline

The implementation follows a modular NLP pipeline:

```text
Raw Corpus
     ↓
Text Preprocessing
     ↓
Corpus Splitting
     ↓
N-Gram Generation
     ↓
Language Model Construction
     ↓
Model Evaluation
```

Each stage is separated into dedicated Python modules to keep the implementation modular and easier to experiment with.

## Project Structure

```text
.
├── README.md
├── Luke_Camilleri_Process_Portfolio.pdf
│
└── Code_Folder/
    ├── main.py
    ├── preprocessing.py
    ├── split.py
    ├── ngram.py
    ├── models.py
    ├── build.py
    ├── build_models.py
    ├── eff_test.py
    └── requirements.txt
```

### Main Components

**`preprocessing.py`**  
Handles preprocessing of the raw textual data before it is used by the language models.

**`split.py`**  
Handles splitting and preparation of corpus data for the modelling pipeline.

**`ngram.py`**  
Contains the core N-gram functionality used to represent sequences of tokens.

**`models.py`**  
Contains the statistical language model implementations.

**`build.py` / `build_models.py`**  
Used to construct and train the language models from the processed corpus.

**`eff_test.py`**  
Contains functionality for testing and evaluating the implemented models.

**`main.py`**  
Main entry point for running the NLP pipeline.

## Installation

Clone the repository:

```bash
git clone https://github.com/Lukecamill/NLP-statistical-language-model-.git
cd NLP-statistical-language-model-
```

Install the required Python packages:

```bash
pip install -r Code_Folder/requirements.txt
```

## Running the Project

The main program can be executed with:

```bash
cd Code_Folder
python main.py
```

Additional scripts can be used for building and evaluating the language models.

## Technologies & Concepts

- Python
- Natural Language Processing
- Statistical Language Modelling
- N-Gram Models
- Corpus Processing
- Text Preprocessing
- Probability Estimation
- Model Evaluation

## What I Learned

This project provided practical experience with the foundations of statistical NLP, particularly how language models can learn patterns directly from text corpora.

It also provided experience in designing an NLP pipeline consisting of separate preprocessing, modelling, training and evaluation components rather than treating language modelling as a single black-box process.

## Documentation

Additional information about the development process and implementation is available in:

**[Process Portfolio](Luke_Camilleri_Process_Portfolio.pdf)**

## Author

**Luke Camilleri**

B.Sc. Artificial Intelligence  
University of Malta
