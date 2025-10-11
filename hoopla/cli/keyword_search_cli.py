#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # loading movie data
    movieDataFile = open("./data/movies.json", "r")
    movieDataJson = json.load(movieDataFile)
    movieDataFile.close()

    #print(movieDataJson)

    # loading stop words
    stopWordsFile = open("./data/stopwords.txt", "r")
    stopWords = stopWordsFile.read().splitlines()
    stopWordsFile.close()

    match args.command:
      case "search":
          query = args.query

          print(f"Searching for: {query}")

          # remove string.punctuation from query
          query = query.translate(str.maketrans('', '', string.punctuation))
          print(f"{query}")

          # lowercase query
          query = query.lower()
          print(f"{query}")

          # split on whitespace
          queryWords = query.split()
          print(f"{queryWords}")

          # remove stop words from query
          queryWords = [word for word in queryWords if word not in stopWords]
          print(f"{queryWords}")

          # filter query to stems
          stemmer = PorterStemmer()
          queryWords = [stemmer.stem(word) for word in queryWords]
          print(f"{queryWords}")

          results = []

          for movie in movieDataJson["movies"]:
              titleToMatch = movie["title"].lower().translate(str.maketrans('', '', string.punctuation))
              titleSplit = titleToMatch.split()
              # remove stop words from title
              titleSplit = [word for word in titleSplit if word not in stopWords]
              # fileter title to stems
              titleSplit = [stemmer.stem(word) for word in titleSplit]
              if any(
                  q in t
                  for q in queryWords
                  for t in titleSplit
              ):
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