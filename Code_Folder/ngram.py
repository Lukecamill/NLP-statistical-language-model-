from collections import defaultdict

def load_preprocessed_corpus(path): # Loads the preprocessed corpus.
    
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences # Returns a list of sentences, where each sentence is a list of tokens.


def count_unigrams(sentences): 
    unigrams = defaultdict(int)
    for sent in sentences:
        for tok in sent:
            unigrams[tok] += 1
    return dict(unigrams)


def count_bigrams(sentences):
    bigrams = defaultdict(lambda: defaultdict(int))
    for sent in sentences:
        for i in range(len(sent) - 1):
            w1 = sent[i]
            w2 = sent[i+1]
            bigrams[w1][w2] += 1
    return {w1: dict(bigrams[w1]) for w1 in bigrams}


def count_trigrams(sentences):
    trigrams = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for sent in sentences:
        for i in range(len(sent) - 2):
            w1 = sent[i]
            w2 = sent[i+1]
            w3 = sent[i+2]
            trigrams[w1][w2][w3] += 1
    # convert to normal dicts
    return {
        w1: {w2: dict(trigrams[w1][w2]) for w2 in trigrams[w1]}
        for w1 in trigrams
    }
