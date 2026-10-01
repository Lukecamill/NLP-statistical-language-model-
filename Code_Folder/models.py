import math
import random


# unigram probabilities
def vanilla_unigram_prob(total_uni, unigrams, w):
    return unigrams.get(w, 0) / total_uni


def laplace_unigram_prob(total_uni, unigrams, w, V):
    return (unigrams.get(w, 0) + 1) / (total_uni + V)


# Bigram probabilities
def vanilla_bigram_prob(unigrams, bigram_totals, bigrams, w1, w2):
    if w1 not in unigrams:
        return 0.0
    return bigrams.get(w1, {}).get(w2, 0) / unigrams[w1]


def laplace_bigram_prob(unigrams, bigram_totals, bigrams, w1, w2, V):
    if w1 not in unigrams:
        return 1 / V
    return (bigrams.get(w1, {}).get(w2, 0) + 1) / (unigrams[w1] + V)


#Trigram probabilities
def vanilla_trigram_prob(trigram_totals, trigrams, w1, w2, w3):
    context_total = trigram_totals.get((w1, w2))
    if context_total is None:
        return 0.0
    return trigrams[w1][w2].get(w3, 0) / context_total


def laplace_trigram_prob(trigram_totals, trigrams, w1, w2, w3, V):
    context_total = trigram_totals.get((w1, w2))
    if context_total is None:
        return 1 / V
    return (trigrams[w1][w2].get(w3, 0) + 1) / (context_total + V)


# UNK handling
def replace_with_unk(sentences, threshold):
    freq = {}
    for sent in sentences:
        for tok in sent:
            freq[tok] = freq.get(tok, 0) + 1

    return [
        [tok if freq[tok] > threshold else "<UNK>" for tok in sent]
        for sent in sentences
    ]


def map_unknowns(tokens, vocab):
    return [w if w in vocab else "<UNK>" for w in tokens]


# interpolated probability
def interpolated_prob(model, w1, w2, w3):
    uni = model["unigrams"]
    bi = model["bigrams"]
    tri = model["trigrams"]
    total_uni = model["total_uni"]
    bigram_totals = model["bigram_totals"]
    trigram_totals = model["trigram_totals"]
    V = model["V"]

    p_uni = laplace_unigram_prob(total_uni, uni, w3, V)
    p_bi  = laplace_bigram_prob(uni, bigram_totals, bi, w2, w3, V)
    p_tri = laplace_trigram_prob(trigram_totals, tri, w1, w2, w3, V)

    return 0.6 * p_tri + 0.3 * p_bi + 0.1 * p_uni


# sentence probability calculation
def sen_probability(sentence_tokens, model, variant="unigram", model_type="vanilla"):

    vocab = model["vocab"]
    tokens = map_unknowns(sentence_tokens, vocab)

    uni  = model["unigrams"]
    bi   = model["bigrams"]
    tri  = model["trigrams"]
    total_uni = model["total_uni"]
    bigram_totals = model["bigram_totals"]
    trigram_totals = model["trigram_totals"]
    V = model["V"]

    log_prob = 0.0

    # Pre-select probability functions
    if model_type == "vanilla":
        UNI = lambda w: vanilla_unigram_prob(total_uni, uni, w)
        BI  = lambda w1, w2: vanilla_bigram_prob(uni, bigram_totals, bi, w1, w2)
        TRI = lambda w1, w2, w3: vanilla_trigram_prob(trigram_totals, tri, w1, w2, w3)
    else:
        UNI = lambda w: laplace_unigram_prob(total_uni, uni, w, V)
        BI  = lambda w1, w2: laplace_bigram_prob(uni, bigram_totals, bi, w1, w2, V)
        TRI = lambda w1, w2, w3: laplace_trigram_prob(trigram_totals, tri, w1, w2, w3, V)

    
    for i in range(len(tokens)):

        if variant == "unigram":
            p = UNI(tokens[i])

        elif variant == "bigram":
            w1 = "<s>" if i == 0 else tokens[i-1]
            p = BI(w1, tokens[i])

        elif variant == "trigram":
            if i == 0:
                p = UNI(tokens[i])
            elif i == 1:
                p = BI(tokens[i-1], tokens[i])
            else:
                p = TRI(tokens[i-2], tokens[i-1], tokens[i])

        else:  # interpolated
            if i < 2:
                continue  # p = 1
            p = interpolated_prob(model, tokens[i-2], tokens[i-1], tokens[i])

        if p == 0:
            return float("-inf")

        log_prob += math.log(p)

    return log_prob


# perplexity calculation
def perplexity(model, test_sentences, variant="unigram", model_type="vanilla"):

    total_log = 0.0
    total_tokens = 0

    for sent in test_sentences:
        logp = sen_probability(sent, model, variant, model_type)
        if math.isinf(logp):
            return float("inf")
        total_log += logp
        total_tokens += len(sent)

    avg = total_log / total_tokens
    return math.exp(-avg)


# sentence generation
def sample_from_distribution(prob_dict):
    r = random.random()
    cumulative = 0.0

    for w, p in prob_dict.items():
        cumulative += p
        if r < cumulative:
            return w

    return "</s>"


def get_next_word_distribution(model, w1, w2, variant="interpolated"):
    uni = model["unigrams"]
    vocab = list(uni.keys())
    V = model["V"]

    bi  = model["bigrams"]
    tri = model["trigrams"]
    bigram_totals = model["bigram_totals"]
    trigram_totals = model["trigram_totals"]
    total_uni = model["total_uni"]

    probs = {}

    for w3 in vocab:
        if variant == "unigram":
            p = laplace_unigram_prob(total_uni, uni, w3, V)
        elif variant == "bigram":
            p = laplace_bigram_prob(uni, bigram_totals, bi, w2, w3, V)
        elif variant == "trigram":
            p = laplace_trigram_prob(trigram_totals, tri, w1, w2, w3, V)
        else:
            p = interpolated_prob(model, w1, w2, w3)

        probs[w3] = p

    total = sum(probs.values())
    return {w: p / total for w, p in probs.items()}


def generate(model, prompt, variant="interpolated", max_len=30):

    tokens = prompt.lower().split()
    if len(tokens) == 0:
        tokens = ["<s>"]
    elif len(tokens) == 1:
        tokens = ["<s>", tokens[0]]
    else:
        tokens = ["<s>"] + tokens

    while len(tokens) < max_len:
        w1 = tokens[-2]
        w2 = tokens[-1]

        distro = get_next_word_distribution(model, w1, w2, variant)
        nxt = sample_from_distribution(distro)
        tokens.append(nxt)

        if nxt == "</s>":
            break

    return " ".join(tokens)
