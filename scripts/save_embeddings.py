from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm

api_key = os.getenv("OPENAI_API",'')

client = OpenAI(api_key=api_key)

def fetch_gpt4_embedding(text, model="text-embedding-3-large"):
    """
    Fetch GPT-4 embedding for a single text.

    Args:
        text (str): Input text.
        model (str): GPT-4 embedding model name.

    Returns:
        list: Embedding vector.
    """
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error fetching embedding for text: {text[:30]}... - {e}")
        return None

def save_embeddings_to_csv(input_file, output_file, model="text-embedding-3-large"):
    """
    Read data from input CSV, fetch embeddings, and save them to a new CSV.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        model (str): GPT-4 embedding model name.
    """
    # Load the input data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    if "Text" not in df.columns:
        raise ValueError(f"Input file {input_file} must have a 'Text' column.")

    # Fetch embeddings for each text
    embeddings = []
    labels = []
    print("Fetching embeddings...")
    for text, label in tqdm(zip(df["Text"],df['Class']), desc=f"Processing {input_file}"):
        embedding = fetch_gpt4_embedding(text, model=model)
        embeddings.append(embedding)
        labels.append(label)

    # Add embeddings to DataFrame
    df["Embedding"] = embeddings
    df["Class"] = labels

    # Save the output DataFrame
    print(f"Saving embeddings to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}.")

def main():
    # List of files to process
    files = [
        ("../data/dev_ma.csv", "../data/open_ai_dev_ma.csv"),
        ("../data/dev_ta.csv", "../data/open_ai_dev_ta.csv"),
        ("../data/train_ma.csv", "../data/open_ai_train_ma.csv"),
        ("../data/train_ta.csv", "../data/open_ai_train_ta.csv")
    ]

    # Process each file
    for input_file, output_file in files:
        if os.path.exists(input_file):
            save_embeddings_to_csv(input_file, output_file)
        else:
            print(f"File {input_file} does not exist. Skipping.")

if __name__ == "__main__":
    main()
