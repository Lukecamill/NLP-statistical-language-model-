import random

def load_sentences(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


def write_sentences(path, sentences):
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")


def train_dev_test_split(sentences, train_ratio=0.8, dev_ratio=0.1, seed=42):
    random.seed(seed)                 # fixed seed = reproducible
    shuffled = sentences[:]           # make a copy
    random.shuffle(shuffled)          # shuffle ALL sentences

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    # remaining (~10%) is test

    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    test = shuffled[n_train + n_dev:]

    return train, dev, test


if __name__ == "__main__":
    input_path = "corpus_folder/full_corpus_preprocessed.txt"


    sentences = load_sentences(input_path)
    print(f"[INFO] Loaded {len(sentences)} sentences from {input_path}")

    train, dev, test = train_dev_test_split(sentences, train_ratio=0.8, dev_ratio=0.1, seed=42)

    print(f"[INFO] Train size: {len(train)}")
    print(f"[INFO] Dev size:   {len(dev)}")
    print(f"[INFO] Test size:  {len(test)}")

    write_sentences("corpus_folder/train_full.txt", train)
    write_sentences("corpus_folder/dev_full.txt", dev)
    write_sentences("corpus_folder/test_full.txt", test)

    print("[INFO] Creating 5K train/test split...")

    # Load the preprocessed 5k corpus
    pre5k = "corpus_folder/subcorpus_5000_preprocessed.txt"
    sentences_5k = load_sentences(pre5k)

    # 80/20 split ONLY (no dev set for 5k experiment)
    train_5k, _, test_5k = train_dev_test_split(
        sentences_5k,
        train_ratio=0.8,
        dev_ratio=0.0,  # No dev set for 5k
        seed=42
    )

    # Saving them
    write_sentences("corpus_folder/train_5k.txt", train_5k)
    write_sentences("corpus_folder/test_5k.txt", test_5k)

    print("[INFO] train_5k.txt and test_5k.txt created successfully!")
