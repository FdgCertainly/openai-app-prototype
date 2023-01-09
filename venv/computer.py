
import openai
import pandas as pd
import csv


COMPLETIONS_MODEL = "text-davinci-003"
openai.api_key = "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"

df = pd.read_csv('/Users/fran/hello/venv/sample3.csv', sep=";")
df = df.set_index(["title", "heading"])

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }


def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
}

# document_embeddings = load_embeddings("https://cdn.openai.com/API/examples/data/olympics_sections_document_embeddings.csv")
context_embeddings = compute_doc_embeddings(df)
# print(f"{context_embeddings}")
# print(context_embeddings)
# Open a file for writing
with open('file2.csv', 'w', newline='') as file:
  # Create a CSV writer object
  fieldnames = ['title', 'heading'] + [str(i) for i in range(len(list(context_embeddings.values())[0]))]
  writer = csv.DictWriter(file, fieldnames=fieldnames)
  
  # Write the header row
  writer.writeheader()
  
  # Write the rows of the list to the CSV file
  for row in context_embeddings.items():
    row_data = {'title': row[0][0], 'heading': row[0][1]}
    for i, value in enumerate(row[1]):
      row_data[str(i)] = value
    writer.writerow(row_data)