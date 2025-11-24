#!/usr/bin/env python3

import argparse
from lib.semantic_search import \
    verify_model, embed_text, embed_chunks, \
    verify_embeddings, embed_query_text, \
    do_search, do_search_chunked, \
    chunk, semantic_chunk
from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE_LIMIT


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
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit of results")

    chunk_parser = subparsers.add_parser("chunk", help="chunk words")
    chunk_parser.add_argument("text", type=str, help="text to chunk by word")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_LIMIT, help="size of chunks in words")
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="overlap between chunks in words")

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="semantic chunk sentences")
    semantic_chunk_parser.add_argument(
        "text", type=str, help="text to chunk by sentence")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="size of chunks in sentences")
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="overlap between chunks in sentences")

    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="embed chunks")

    search_chunks_parser = subparsers.add_parser(
        "search_chunked", help="search chunks")
    search_chunks_parser.add_argument(
        "query", type=str, help="query to search")
    search_chunks_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit of results")

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

        case "chunk":
            # get character length of text
            text_length = len(args.text)
            print(f"Chunking {text_length} characters")

            chunked_text = chunk(args.text, args.chunk_size, args.overlap)
            for i, text in enumerate(chunked_text):
                print(f"{i+1}. {text}")

        case "semantic_chunk":
            text_length = len(args.text)
            print(f"Semantically chunking {text_length} characters")
            chunked_text = semantic_chunk(
                args.text, args.max_chunk_size, args.overlap)
            for i, text in enumerate(chunked_text):
                print(f"{i+1}. {text}")

        case "embed_chunks":
            embed_chunks()

        case "search_chunked":
            results = do_search_chunked(args.query, args.limit)
            if results is None:
                print("No results found")
                return
            for i, r in enumerate(results):
                print(f"\n{i+1}. {r["title"]} (score: {r["score"]:.4f})")
                print(f"   {r["description"]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
