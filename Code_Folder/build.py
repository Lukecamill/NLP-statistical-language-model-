from ngram import (
    load_preprocessed_corpus,
    count_unigrams,
    count_bigrams,
    count_trigrams
)

if __name__ == "__main__":
    corpus_path = "corpus_folder/full_corpus_preprocessed.txt"

    print("[INFO] Loading corpus...")
    sentences = load_preprocessed_corpus(corpus_path)

    print("[INFO] Counting unigrams...")
    unigrams = count_unigrams(sentences)

    print("[INFO] Counting bigrams...")
    bigrams = count_bigrams(sentences)

    print("[INFO] Counting trigrams...")
    trigrams = count_trigrams(sentences)

    print("[INFO] Done! Counts ready for probability models.")
