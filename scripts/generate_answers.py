import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

def build_prompt(query, contexts):
    context_text = "\n\n".join([f"Document {i+1}: {c['text']}" for i, c in enumerate(contexts)])

    prompt = f"""
You are a helpful assistant. Answer the question using the provided documents.

Question:
{query}

Documents:
{context_text}

Answer:
"""
    return prompt.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="generator_input.jsonl")
    parser.add_argument("--output", required=True, help="final_answers.jsonl")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    print("Loading generator input...")
    with open(args.input) as f:
        data = [json.loads(line) for line in f]

    print(f"Generating answers for {len(data)} queries...")

    with open(args.output, "w") as out:
        for item in tqdm(data):
            query_id = item["query_id"]
            query = item["query"]
            contexts = item["contexts"]

            prompt = build_prompt(query, contexts)

            result = generator(
                prompt,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=0.0
            )[0]["generated_text"]

            answer = result[len(prompt):].strip()

            output = {
                "query_id": query_id,
                "answer": answer
            }

            out.write(json.dumps(output) + "\n")

    print("\nGeneration complete.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
