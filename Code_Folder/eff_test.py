import time
import tracemalloc
from ngram import (
    load_preprocessed_corpus,
    count_unigrams,
    count_bigrams,
    count_trigrams
)

def measure_pipeline(corpus_path):
    print(f"\n[TEST] Measuring pipeline for: {corpus_path}")

    # Start measuring memory
    tracemalloc.start()

    # Measure corpus loading time
    t0 = time.time()
    sentences = load_preprocessed_corpus(corpus_path)
    load_time = time.time() - t0
    print(f"[TIME] Loading corpus: {load_time:.4f} seconds")

    # Unigram time
    t1 = time.time()
    unigrams = count_unigrams(sentences)
    u_time = time.time() - t1
    print(f"[TIME] Unigram counting: {u_time:.4f} seconds")

    # Bigram time
    t2 = time.time()
    bigrams = count_bigrams(sentences)
    b_time = time.time() - t2
    print(f"[TIME] Bigram counting: {b_time:.4f} seconds")

    # Trigram time
    t3 = time.time()
    trigrams = count_trigrams(sentences)
    t_time = time.time() - t3
    print(f"[TIME] Trigram counting: {t_time:.4f} seconds")

    # Memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"[MEMORY] Current: {current/1_000_000:.2f} MB | Peak: {peak/1_000_000:.2f} MB\n")

    # Stop memory tracking
    tracemalloc.stop()

if __name__ == "__main__":
    # Test on different corpus sizes
    measure_pipeline("corpus_folder/subcorpus_1000_preprocessed.txt")
    measure_pipeline("corpus_folder/subcorpus_5000_preprocessed.txt")
    measure_pipeline("corpus_folder/subcorpus_20000_preprocessed.txt")
