"""
generator.py
Part 3 — Markov Text Generator
"""

import os
import json
import random
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from starter_preprocess import TextPreprocessor, FrequencyAnalyzer



class TextGenerator:
    def __init__(self, author, level="word-1"):
        self.author = author
        self.level = level
        self.ngram_type, self.n = self._parse_level(level)
        self.freq_data = self._load_freq_data()

    def _parse_level(self, level):
        # e.g. "word-3" → ("word", 3)
        model_type, order = level.split("-")
        return model_type, int(order)

    def _load_freq_data(self):
        # Load the right frequency JSON file
        file_path = f"data/freq_tables/{self.author}_{self.ngram_type}_{self.n}-gram.json"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing frequency file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # convert back to tuple keys for n>1
        freq_data = {}
        for key, val in data.items():
            if "||" in key:
                freq_data[tuple(key.split("||"))] = val
            else:
                freq_data[key] = val
        return freq_data

    def _choose_next(self, candidates):
        # Weighted random choice
        total = sum(candidates.values())
        r = random.uniform(0, total)
        upto = 0
        for k, w in candidates.items():
            upto += w
            if upto >= r:
                return k
        return random.choice(list(candidates.keys()))

    def generate(self, length=100, seed=None):
        """Generate text using loaded n-gram model."""
        if self.ngram_type == "char":
            return self._generate_char_sequence(length, seed)
        else:
            return self._generate_word_sequence(length, seed)

    def _generate_char_sequence(self, length, seed):
        if self.n == 0:
            return ''.join(random.choices(list(self.freq_data.keys()), k=length))
        start = random.choice(list(self.freq_data.keys()))
        if isinstance(start, tuple):
            current = list(start)
        else:
            current = [start]

        output = current.copy()
        for _ in range(length):
            context = tuple(output[-(self.n - 1):]) if self.n > 1 else output[-1]
            next_candidates = {k[-1]: v for k, v in self.freq_data.items()
                               if (self.n == 1 and k == context) or
                                  (self.n > 1 and k[:-1] == context)}
            if not next_candidates:
                break
            next_token = self._choose_next(next_candidates)
            output.append(next_token)
        return ''.join(output)

    def _generate_word_sequence(self, length, seed):
        words = list(self.freq_data.keys())
        start = random.choice(words)
        if isinstance(start, tuple):
            current = list(start)
        else:
            current = [start]
        output = current.copy()

        for _ in range(length):
            context = tuple(output[-(self.n - 1):]) if self.n > 1 else output[-1]
            next_candidates = {k[-1]: v for k, v in self.freq_data.items()
                               if (self.n == 1 and k == context) or
                                  (self.n > 1 and k[:-1] == context)}
            if not next_candidates:
                break
            next_word = self._choose_next(next_candidates)
            output.append(next_word)
        return " ".join(output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using n-gram model.")
    parser.add_argument("--author", required=True, help="austen | twain | doyle")
    parser.add_argument("--level", required=True, help="char-1 | char-2 | word-3 etc.")
    parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")
    args = parser.parse_args()

    generator = TextGenerator(author=args.author, level=args.level)
    result = generator.generate(length=args.length)
    print("\n🪶 Generated Text:\n")
    print(result)