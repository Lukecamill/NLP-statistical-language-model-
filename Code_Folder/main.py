from preprocessing import preprocess_corpus
from split import load_sentences, write_sentences, train_dev_test_split
from build_models import build_models, load_models, save_models
from build_models import evaluate_small_models, evaluate_full_models
from models import generate, sen_probability



def main():

    # preprocessing small corpus
    print("[INFO] Preprocessing small corpus...")
    preprocess_corpus(
        input_path="corpus_folder/eng_news_2024_100K-sentences.txt",
        output_path="corpus_folder/subcorpus_5000_preprocessed.txt"
    )

    # preprocessing the raw corpus
    print("[INFO] Preprocessing raw corpus...")
    preprocess_corpus(
        input_path="corpus_folder/eng_news_2024_100K-sentences.txt",
        output_path="corpus_folder/processed_corpus.txt"
    )

    # splitting the corpus
    print("[INFO] Creating full 80/10/10 split...")
    sentences = load_sentences("corpus_folder/processed_corpus.txt")
    train_full, dev_full, test_full = train_dev_test_split(
        sentences,
        train_ratio=0.8,
        dev_ratio=0.1,
        seed=42
    )

    write_sentences("corpus_folder/train_full.txt", train_full)
    write_sentences("corpus_folder/dev_full.txt", dev_full)
    write_sentences("corpus_folder/test_full.txt", test_full)


    print("[INFO] Creating 5K train/test split...")

    subset_5k = sentences[:5000]
    train_5k, _, test_5k = train_dev_test_split(
        subset_5k,
        train_ratio=0.8,
        dev_ratio=0.0,
        seed=42
    )

    write_sentences("corpus_folder/train_5k.txt", train_5k)
    write_sentences("corpus_folder/test_5k.txt", test_5k)




    # Small (sub) corpus evaluation (5k)
    print("[INFO] Computing small-corpus evaluation (5k)...")
    table_small = evaluate_small_models(
        train_path="corpus_folder/train_5k.txt",
        test_path="corpus_folder/test_5k.txt",
        unk_threshold=5
    )

    print("\nSmall-corpus Results:")
    print("Model Variant      Unigram      Bigram      Trigram      Interpolated")
    for model_type in ["vanilla", "laplace", "unk"]:
        print(f"{model_type:15}", end="")
        print(f"{table_small['unigram'][model_type]:12.2f}", end="")
        print(f"{table_small['bigram'][model_type]:12.2f}", end="")
        print(f"{table_small['trigram'][model_type]:12.2f}", end="")
        print(f"{table_small['interpolated'][model_type]:16.2f}", end="")
        print()

    # Build final (full) models and save
    print("\n[INFO] Building full models...")
    models = build_models("corpus_folder/train_full.txt", unk_threshold=5)
    save_models(models)

    print("[INFO] Loading saved models...")
    models = load_models()

    print("[INFO] Generating example sentence...")
    print(generate(models["unk"], "<s>", variant="interpolated"))

    print("[INFO] Testing sentence probability...")
    sentence = ["<s>", "hello", "world", "</s>"]
    print("Probability:", sen_probability(sentence, models["unk"], variant="trigram", model_type="laplace"))

    # Full-corpus evaluation
    print("\n[INFO] Computing full-corpus evaluation...")
    table_full = evaluate_full_models(models, test_path="corpus_folder/test_full.txt")

    print("\nFull-corpus Results:")
    print("Model Variant              Perplexity")
    for model_name, value in table_full.items():
        print(f"{model_name:25s}: {value:.2f}")

if __name__ == "__main__":
    main()
