"""
analyze.py
Part 1 - Text Analysis and Frequency Extraction

Generates character- and word-level n-gram frequency tables
for the given author text using starter_preprocess.py utilities.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from starter_preprocess import TextPreprocessor, FrequencyAnalyzer


def analyze_text(author: str):
    """
    Clean, normalize, tokenize, and compute n-gram frequencies
    for the given author.
    """

    author_files = {
        "austen": "data/austen_pride_prejudice.txt",
        "twain": "data/twain_tom_sawyer.txt",
        "doyle": "data/doyle_sherlock_holmes.txt"
    }

    if author not in author_files:
        raise ValueError("Author must be one of: austen, twain, doyle")

    input_path = author_files[author]
    print(f"📖 Reading text for {author.title()}...")

    pre = TextPreprocessor()
    fa = FrequencyAnalyzer()

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = pre.clean_gutenberg_text(raw)
    normalized = pre.normalize_text(cleaned)
    print(" Cleaned and normalized text.")

    sentences = pre.tokenize_sentences(normalized)
    words = pre.tokenize_words(normalized)
    chars = pre.tokenize_chars(normalized)

    print(f" Sentences: {len(sentences)} | Words: {len(words)} | Chars: {len(chars)}")

    char_freqs_all = {}
    word_freqs_all = {}

    for n in [1, 2, 3]:
        char_freqs = fa.calculate_ngrams(chars, n)
        word_freqs = fa.calculate_ngrams(words, n)
        char_freqs_all[f"{n}-gram"] = char_freqs
        word_freqs_all[f"{n}-gram"] = word_freqs

    out_dir = "data/freq_tables"
    os.makedirs(out_dir, exist_ok=True)

    char_file = os.path.join(out_dir, f"{author}_char.json")
    word_file = os.path.join(out_dir, f"{author}_word.json")

    # Save each n-gram level separately so JSON stays valid
    for n, freqs in char_freqs_all.items():
        filename = os.path.join(out_dir, f"{author}_char_{n}.json")
        fa.save_frequencies(freqs, filename)

    for n, freqs in word_freqs_all.items():
        filename = os.path.join(out_dir, f"{author}_word_{n}.json")
        fa.save_frequencies(freqs, filename)

    print(f" Saved character frequencies → {char_file}")
    print(f" Saved word frequencies → {word_file}")
    print(" Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze text and compute n-gram frequencies.")
    parser.add_argument("--author", required=True, help="Author to analyze: austen | twain | doyle")
    args = parser.parse_args()

    analyze_text(args.author)