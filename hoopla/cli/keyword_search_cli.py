#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    movieDataFile = open("./data/movies.json", "r")
    movieDataJson = json.load(movieDataFile)

    #print(movieDataJson)

    match args.command:
      case "search":
          print(f"Searching for: {args.query}")

          results = []

          for movie in movieDataJson["movies"]:
              if movie["title"].lower().find(args.query.lower()) != -1:
                  results.append(movie)

          print(f"Found {len(results)} results")

          # order results by id
          results = sorted(results, key=lambda x: x["id"])

          # slice off top 5 results
          results = results[:5]

          # print the results
          for result in results:
              print(f"{result['title']} - {result['id']}")

      case _:
          parser.print_help()


if __name__ == "__main__":
    main()