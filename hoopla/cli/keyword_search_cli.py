#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
from lib.keyword_search import build_command, search_command, tf_command


DEFAULT_SEARCH_LIMIT = 50


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building index...")
            build_command()
            print("Index built successfully.")

        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for r in (results):
                print(f"{r["id"]} {r["title"]}")

        case "tf":
            # It should take a document ID and a term as arguments.
            print(
                f"Term frequency for {args.doc_id} {args.term}: {tf_command(args.doc_id, args.term)}")

        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
