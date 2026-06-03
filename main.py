import os
import time
import pandas as pd
from pypdf import PdfReader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Read PDFs

documents = []

for file in os.listdir("data"):

    if file.endswith(".pdf"):

        path = os.path.join(
            "data",
            file
        )

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        documents.append(
            {
                "name": file,
                "text": text
            }
        )

print(
    f"\nLoaded {len(documents)} PDFs"
)

# Chunking

all_chunks = []

stats = []

for doc in documents:

    tokens = tokenizer.encode(
        doc["text"]
    )

    chunks = []

    for i in range(
            0,
            len(tokens),
            512):

        chunk_tokens = tokens[
            i:i+512
        ]

        chunk_text = tokenizer.decode(
            chunk_tokens
        )

        chunks.append(
            chunk_text
        )

        all_chunks.append(
            chunk_text
        )

    stats.append({
        "Document":
        doc["name"],

        "Total Tokens":
        len(tokens),

        "Chunks":
        len(chunks),

        "Average Tokens":
        round(
            len(tokens)
            / len(chunks),
            2
        )
    })

stats_df = pd.DataFrame(stats)

print("\nTOKEN STATISTICS\n")
print(stats_df)

# Sequential Embedding

print(
    "\nSequential Embedding..."
)

start = time.time()

seq_embeddings = []

for chunk in all_chunks:

    emb = model.encode(
        chunk
    )

    seq_embeddings.append(
        emb
    )

seq_time = (
    time.time() - start
)

seq_throughput = (
    len(all_chunks)
    / seq_time
)

# Batch Embedding

print(
    "\nBatch Embedding..."
)

start = time.time()

batch_embeddings = model.encode(
    all_chunks,
    batch_size=16,
    show_progress_bar=True
)

batch_time = (
    time.time() - start
)

batch_throughput = (
    len(all_chunks)
    / batch_time
)

# ChromaDB

client = chromadb.Client()

collection = client.create_collection(
    "blockchain_docs"
)

ids = [
    str(i)
    for i in range(
        len(all_chunks)
    )
]

collection.add(
    ids=ids,
    documents=all_chunks,
    embeddings=
    batch_embeddings.tolist()
)

print(
    "\nVector DB Created"
)

# Retrieval

query = input(
    "\nAsk a Question: "
)

query_embedding = model.encode(
    query
)

start = time.time()

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=3
)

retrieval_time = (
    time.time()
    - start
)

print(
    "\nTOP 3 CHUNKS\n"
)

for i, chunk in enumerate(
        results["documents"][0],
        1):

    print(
        f"\n--- Chunk {i} ---\n"
    )

    print(chunk[:1000])

# Performance Table

performance = pd.DataFrame([
{
"Approach":
"Sequential",

"Batch Size":
1,

"Time (s)":
round(
    seq_time,
    2
),

"Throughput":
round(
    seq_throughput,
    2
),

"Retrieval Latency":
round(
    retrieval_time,
    4
)
},
{
"Approach":
"Batched",

"Batch Size":
16,

"Time (s)":
round(
    batch_time,
    2
),

"Throughput":
round(
    batch_throughput,
    2
),

"Retrieval Latency":
round(
    retrieval_time,
    4
)
}
])

print(
    "\nPERFORMANCE TABLE\n"
)

print(performance)

performance.to_csv(
    "results.csv",
    index=False
)

print(
    "\nresults.csv saved"
)