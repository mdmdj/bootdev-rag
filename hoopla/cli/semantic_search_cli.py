#!/usr/bin/env python3

import argparse
from lib.semantic_search import \
    verify_model, embed_text, \
    verify_embeddings, embed_query_text, \
    do_search


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify Model")

    embed_parser = subparsers.add_parser("embed_text", help="Embed Text")
    embed_parser.add_argument("text", type=str, help="Text")

    verify_embed_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify Embeddings")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="embed query")
    embed_query_parser.add_argument("query", type=str, help="query to embed")

    search_parser = subparsers.add_parser("search", help="Search")
    search_parser.add_argument("query", type=str, help="query to search")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="limit of results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            results = do_search(args.query, args.limit)
            if results is None:
                print("No results found")
                return
            for i, r in enumerate(results):
                print(
                    f"{i+1}. {r['title']} (score:{r['score']})\n\t{r['description']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
