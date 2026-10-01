import os
import pickle

from ngram import (
    load_preprocessed_corpus,
    count_unigrams,
    count_bigrams,
    count_trigrams
)

from models import (
    replace_with_unk,
    perplexity
)


# build models from training data
def build_models(train_path, unk_threshold=1):

    print("[INFO] Loading training data...")
    sentences = load_preprocessed_corpus(train_path)

    # Vanilla counts
    print("[INFO] Building vanilla n-gram counts...")
    uni = count_unigrams(sentences)
    bi = count_bigrams(sentences)
    tri = count_trigrams(sentences)

    # Precompute totals for speed
    total_uni = sum(uni.values())
    bigram_totals = {w1: sum(bi[w1].values()) for w1 in bi}
    trigram_totals = {
        (w1, w2): sum(tri[w1][w2].values())
        for w1 in tri
        for w2 in tri[w1]
    }

    vocab = set(uni.keys())
    V = len(vocab)

    # UNK counts
    print(f"[INFO] Replacing words with freq <= {unk_threshold} by <UNK>...")
    unk_sentences = replace_with_unk(sentences, unk_threshold)

    print("[INFO] Rebuilding counts with <UNK>...")
    uni_unk = count_unigrams(unk_sentences)
    bi_unk = count_bigrams(unk_sentences)
    tri_unk = count_trigrams(unk_sentences)

    total_uni_unk = sum(uni_unk.values())
    bigram_totals_unk = {w1: sum(bi_unk[w1].values()) for w1 in bi_unk}
    trigram_totals_unk = {
        (w1, w2): sum(tri_unk[w1][w2].values())
        for w1 in tri_unk
        for w2 in tri_unk[w1]
    }

    vocab_unk = set(uni_unk.keys())
    V_unk = len(vocab_unk)

    return {
        "vanilla": {
            "unigrams": uni,
            "bigrams": bi,
            "trigrams": tri,
            "total_uni": total_uni,
            "bigram_totals": bigram_totals,
            "trigram_totals": trigram_totals,
            "vocab": vocab,
            "V": V
        },
        "laplace": {
            "unigrams": uni,
            "bigrams": bi,
            "trigrams": tri,
            "total_uni": total_uni,
            "bigram_totals": bigram_totals,
            "trigram_totals": trigram_totals,
            "vocab": vocab,
            "V": V
        },
        "unk": {
            "unigrams": uni_unk,
            "bigrams": bi_unk,
            "trigrams": tri_unk,
            "total_uni": total_uni_unk,
            "bigram_totals": bigram_totals_unk,
            "trigram_totals": trigram_totals_unk,
            "vocab": vocab_unk,
            "V": V_unk
        }
    }


# saving and loading models
def save_models(models):
    models_dir = os.path.join(os.path.dirname(__file__), "models_folder")
    os.makedirs(models_dir, exist_ok=True)

    path = os.path.join(models_dir, "models.pkl")

    with open(path, "wb") as f:
        pickle.dump(models, f)

    print(f"[INFO] Models saved to {path}")


def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), "models_folder")
    path = os.path.join(models_dir, "models.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] No model file found at {path}. "
            "Run build_models.py first."
        )

    with open(path, "rb") as f:
        models = pickle.load(f)

    print(f"[INFO] Models loaded from {path}")
    return models


# evaluation on small models
def evaluate_small_models(train_path, test_path, unk_threshold=1):

    models = build_models(train_path, unk_threshold)
    test_sentences = load_preprocessed_corpus(test_path)

    print("[INFO] Computing full 12-model table on 5k dataset...")

    table = {
        "unigram": {},
        "bigram": {},
        "trigram": {},
        "interpolated": {}
    }

    # VANILLA
    table["unigram"]["vanilla"] = perplexity(models["vanilla"], test_sentences, "unigram", model_type="vanilla")
    table["bigram"]["vanilla"] = perplexity(models["vanilla"], test_sentences, "bigram", model_type="vanilla")
    table["trigram"]["vanilla"] = perplexity(models["vanilla"], test_sentences, "trigram", model_type="vanilla")
    table["interpolated"]["vanilla"] = perplexity(models["vanilla"], test_sentences, "interpolated", model_type="laplace")

    # LAPLACE
    table["unigram"]["laplace"] = perplexity(models["laplace"], test_sentences, "unigram", model_type="laplace")
    table["bigram"]["laplace"] = perplexity(models["laplace"], test_sentences, "bigram", model_type="laplace")
    table["trigram"]["laplace"] = perplexity(models["laplace"], test_sentences, "trigram", model_type="laplace")
    table["interpolated"]["laplace"] = perplexity(models["laplace"], test_sentences, "interpolated", model_type="laplace")

    # UNK
    table["unigram"]["unk"] = perplexity(models["unk"], test_sentences, "unigram", model_type="laplace")
    table["bigram"]["unk"] = perplexity(models["unk"], test_sentences, "bigram", model_type="laplace")
    table["trigram"]["unk"] = perplexity(models["unk"], test_sentences, "trigram", model_type="laplace")
    table["interpolated"]["unk"] = perplexity(models["unk"], test_sentences, "interpolated", model_type="laplace")

    return table


# evaluation on full models
def evaluate_full_models(models, test_path):

    test_sentences = load_preprocessed_corpus(test_path)
    test_sentences = test_sentences[:1000]  # efficiency

    table = {
        "unk_trigram": perplexity(models["unk"], test_sentences, "trigram", model_type="laplace"),
        "unk_interpolated": perplexity(models["unk"], test_sentences, "interpolated", model_type="laplace"),
        "laplace_trigram": perplexity(models["laplace"], test_sentences, "trigram", model_type="laplace"),
        "laplace_interpolated": perplexity(models["laplace"], test_sentences, "interpolated", model_type="laplace"),
    }

    return table


if __name__ == "__main__":

    print("=== REQUIRED ASSIGNMENT TABLE (5K CORPUS) ===")
    table_small = evaluate_small_models(
        train_path="corpus_folder/train_5k.txt",
        test_path="corpus_folder/test_5k.txt",
        unk_threshold=5
    )

    print("\nModel Variant      Unigram      Bigram      Trigram      Interpolated")
    for model_type in ["vanilla", "laplace", "unk"]:
        print(f"{model_type:15}", end="")
        print(f"{table_small['unigram'][model_type]:12.2f}", end="")
        print(f"{table_small['bigram'][model_type]:12.2f}", end="")
        print(f"{table_small['trigram'][model_type]:12.2f}", end="")
        print(f"{table_small['interpolated'][model_type]:16.2f}", end="")
        print()

    # train final models on full corpus
    print("\nTraining final models and saving to models_folder ===")
    full_models = build_models("corpus_folder/train_full.txt", unk_threshold=5)
    save_models(full_models)
    print("[INFO] Final models saved.\n")

    # full evaluation
    print("=== FINAL MODEL EVALUATION (FULL 100K CORPUS) ===")
    table_full = evaluate_full_models(
        full_models,
        test_path="corpus_folder/test_full.txt"
    )

    print("\nModel Variant              Perplexity")
    for model_name, value in table_full.items():
        print(f"{model_name:25s}: {value:.2f}")
