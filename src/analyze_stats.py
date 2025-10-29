"""
analyze_stats.py
Part 2 - Statistical Analysis & Visualization
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from starter_preprocess import TextPreprocessor, FrequencyAnalyzer


def load_text(author):
    author_files = {
        "austen": "data/austen_pride_prejudice.txt",
        "twain": "data/twain_tom_sawyer.txt",
        "doyle": "data/doyle_sherlock_holmes.txt"
    }
    if author not in author_files:
        raise ValueError("Author must be one of: austen, twain, doyle")
    with open(author_files[author], "r", encoding="utf-8") as f:
        return f.read()


def compute_sentence_stats(text, preprocessor):
    sentences = preprocessor.tokenize_sentences(text)
    sentence_lengths = preprocessor.get_sentence_lengths(sentences)
    mean_len = np.mean(sentence_lengths)
    std_len = np.std(sentence_lengths)
    return sentence_lengths, mean_len, std_len


def plot_sentence_length_distribution(author, lengths):
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=40, color="teal", edgecolor="black", alpha=0.7)
    plt.title(f"Sentence Length Distribution — {author.title()}")
    plt.xlabel("Sentence length (words)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{author}_sentence_length_hist.png", dpi=200)
    plt.close()


def plot_top_ngrams(author, freq_file, top_n=20, label="Word"):
    with open(freq_file, "r", encoding="utf-8") as f:
        freqs = json.load(f)
    # Pick the trigram list for visualization (3-gram)
    trigram_data = freqs.get("3-gram", freqs)
    top_items = sorted(trigram_data.items(), key=lambda x: x[1], reverse=True)[:top_n]

    labels, values = zip(*top_items)
    labels = [l.replace("||", " ") for l in labels]  # prettier keys

    plt.figure(figsize=(10, 5))
    plt.barh(labels[::-1], values[::-1], color="slateblue")
    plt.title(f"Top {top_n} {label} 3-grams — {author.title()}")
    plt.xlabel("Frequency")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{author}_{label.lower()}_top3grams.png", dpi=200)
    plt.close()


def main(author):
    pre = TextPreprocessor()

    raw_text = load_text(author)
    clean_text = pre.clean_gutenberg_text(raw_text)
    normalized = pre.normalize_text(clean_text)

    # Sentence stats
    sentence_lengths, mean_len, std_len = compute_sentence_stats(normalized, pre)
    print(f"📊 {author.title()} — Mean sentence length: {mean_len:.2f} ± {std_len:.2f}")

    plot_sentence_length_distribution(author, sentence_lengths)

    # Load frequency JSONs (generated in part 1)
    word_freq_file = f"data/freq_tables/{author}_word_3-gram.json"
    char_freq_file = f"data/freq_tables/{author}_char_3-gram.json"

    plot_top_ngrams(author, word_freq_file, label="Word")
    plot_top_ngrams(author, char_freq_file, label="Character")

    print(f"Plots saved in /outputs for {author}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics and visualize results.")
    parser.add_argument("--author", required=True, help="Author to analyze: austen | twain | doyle")
    args = parser.parse_args()
    main(args.author)