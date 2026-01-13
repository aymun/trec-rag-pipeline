# scripts/generate.py
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import json
from config import OUTPUT_DIR

# Load RAG model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# Example query
query = "What are recent trends in error analysis for document retrieval?"

inputs = tokenizer(query, return_tensors="pt")
generated = model.generate(**inputs)

print("Generated output:")
print(tokenizer.decode(generated[0], skip_special_tokens=True))
