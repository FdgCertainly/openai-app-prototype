import pandas as pd
import pickle
import openai

from openai.embeddings_utils import (get_embedding,distances_from_embeddings,tsne_components_from_embeddings,chart_from_components,indices_of_nearest_neighbors_from_distances)

openai.api_key = "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"


df = pd.read_csv('/Users/fran/hello/venv/sample.csv', sep=";")

# establish a cache of embeddings to avoid recomputing
# cache is a dict of tuples (text, engine) -> embedding, saved as a pickle file

# set path to embedding cache
embedding_cache_path_to_load = "example_embeddings_cache.pkl"
embedding_cache_path_to_save = "example_embeddings_cache.pkl"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path_to_load)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path_to_save, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
    string: str,
    engine: str = "text-embedding-ada-002",
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, engine) not in embedding_cache.keys():
        embedding_cache[(string, engine)] = get_embedding(string, engine)
        with open(embedding_cache_path_to_save, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, engine)]

def print_recommendations_from_strings(
    titles:list[str],
    strings: list[str],
    index_of_source_string: int,
    k_nearest_neighbors: int = 1,
    engine="text-embedding-ada-002",
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all strings
    embeddings = [embedding_from_string(string, engine=engine) for string in strings]
    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # print out source string
    query_title = titles[index_of_source_string]
    query_string = strings[index_of_source_string]
    print(f"""
    Source title:{query_title}
    Source string: {query_string}
    """)
    # print out its k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_string == strings[i]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # print out the similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        Title:{titles[i]}
        String: {strings[i]}
        Distance: {distances[i]:0.3f}"""
        )

    return indices_of_nearest_neighbors
    
article_descriptions = df["description"].tolist()
article_titles = df["title"].tolist()

tony_blair_articles = print_recommendations_from_strings(
    titles=article_titles,
    strings=article_descriptions,  # let's base similarity off of the article description
    index_of_source_string=16,  # let's look at articles similar to the first one about Tony Blair
    k_nearest_neighbors=5,  # let's look at the 5 most similar articles
)