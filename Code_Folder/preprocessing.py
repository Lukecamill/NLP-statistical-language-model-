import re

def extract_subcorpus(input_path, output_path, n_sentences=5000):
    

    count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if count >= n_sentences: # exit condition when count is equal to n_sentences
                break

            line = line.strip() # removes whitespaces from sentence
            if line:  # skip empty lines
                f_out.write(line + "\n") # writethe sentence to the new file
                count += 1

    print(f"[INFO] Extracted {count} sentences → {output_path}")

def preprocess_sentence(sentence):
    
    # lowercase
    sentence = sentence.lower().strip()

    # separate punctuation (simple regex)
    sentence = re.sub(r'([.,!?;:()"])', r' \1 ', sentence)

    # collapse multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    # add sentence boundary tokens
    tokens = sentence.split()
    tokens = ["<s>"] + tokens + ["</s>"]

    return tokens

def preprocess_corpus(input_path, output_path): # does preprocessing on every single sentence in the corpus
    

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            tokens = preprocess_sentence(line)
            f_out.write(" ".join(tokens) + "\n")

    print(f"[INFO] Preprocessed corpus saved to {output_path}")


if __name__ == "__main__":
    
    raw_corpus = "corpus_folder/eng_news_2024_100K-sentences.txt"
    sub1k = "corpus_folder/subcorpus_1000.txt"
    sub5k = "corpus_folder/subcorpus_5000.txt"
    sub20k = "corpus_folder/subcorpus_20000.txt"
    whole_corpus = "corpus_folder/full_corpus.txt"
    
    # extraction of the smaller corpus
    extract_subcorpus(raw_corpus, sub1k, n_sentences=1000)
    extract_subcorpus(raw_corpus, sub5k, n_sentences=5000)
    extract_subcorpus(raw_corpus, sub20k, n_sentences=20000)
    extract_subcorpus(raw_corpus, whole_corpus, n_sentences=100000)
    
    # preprocessing of the smaller corpus
    pre1k = "corpus_folder/subcorpus_1000_preprocessed.txt"
    pre5k = "corpus_folder/subcorpus_5000_preprocessed.txt"
    pre20k = "corpus_folder/subcorpus_20000_preprocessed.txt"
    prefull = "corpus_folder/full_corpus_preprocessed.txt"
    
    preprocess_corpus(sub1k, pre1k)
    preprocess_corpus(sub5k, pre5k)
    preprocess_corpus(sub20k, pre20k)
    preprocess_corpus(whole_corpus, prefull)
