"""
shannon_gen.py
Part 4 — Unified CLI for Shannon Text Analysis & Generation
"""

import argparse
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyze import analyze_text
from src.analyze_stats import main as visualize_main
from src.generator import TextGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for Shannon-style text analysis and generation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run Part 1: generate frequency tables")
    analyze_parser.add_argument("--author", required=True, help="austen | twain | doyle")

    # visualize
    vis_parser = subparsers.add_parser("visualize", help="Run Part 2: create plots")
    vis_parser.add_argument("--author", required=True, help="austen | twain | doyle")

    # generate 
    gen_parser = subparsers.add_parser("generate", help="Run Part 3: text generation")
    gen_parser.add_argument("--author", required=True, help="austen | twain | doyle")
    gen_parser.add_argument("--level", required=True, help="char-1 | char-2 | word-3 etc.")
    gen_parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")

    args = parser.parse_args()

    # dispatch by command
    if args.command == "analyze":
        analyze_text(args.author)

    elif args.command == "visualize":
        visualize_main(args.author)

    elif args.command == "generate":
        generator = TextGenerator(author=args.author, level=args.level)
        text = generator.generate(length=args.length)
        print("\n🪶 Generated Text:")
        print(text)
        print("------------------------------------------------\n")


if __name__ == "__main__":
    main()