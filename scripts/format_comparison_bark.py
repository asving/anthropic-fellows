"""
Compare generation diversity for "bark" example:
1. Chat template with special tokens (should be 5/5 identical)
2. Plaintext User/Assistant without special tokens (expect more diversity)
"""

import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str):
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_with_entropy(model, tokenizer, prompt: str, max_tokens: int = 30):
    """Generate tokens one by one, recording entropy at each position."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_tokens = []
    entropies = []

    current_ids = inputs['input_ids'].clone()

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(current_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # Compute entropy
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs).item()

            # Sample next token
            next_token = torch.multinomial(probs, 1)
            next_token_str = tokenizer.decode(next_token)

            generated_tokens.append(next_token_str)
            entropies.append(entropy)

            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

            # Stop at EOS or period followed by likely end
            if next_token.item() == tokenizer.eos_token_id:
                break
            if '<|im_end|>' in next_token_str or '<|eot_id|>' in next_token_str:
                break
            if '.' in next_token_str and len(generated_tokens) > 5:
                # Generate one more to see if it's EOS
                continue

    return {
        "tokens": generated_tokens,
        "entropies": entropies,
        "full_text": "".join(generated_tokens).strip(),
    }


def find_word_entropy(tokens, entropies, word):
    """Find entropy at the position containing the target word."""
    text_so_far = ""
    for i, tok in enumerate(tokens):
        text_so_far += tok
        if word.lower() in text_so_far.lower():
            return i, entropies[i]
    return -1, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--num_generations", type=int, default=5)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    word = "bark"
    instruction = f"Write a short sentence (under 15 words) that uses the word '{word}'."

    # Define prompts
    prompts = {
        "chat_template": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        "plaintext_user_asst": f"User: {instruction}\nAssistant: ",
        "plaintext_alice_bob": f"Alice: {instruction}\nBob: ",
    }

    results = {"model": args.model, "word": word, "formats": {}}

    for format_name, prompt in prompts.items():
        print(f"\n{'='*70}", flush=True)
        print(f"FORMAT: {format_name}", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Prompt: {prompt!r}\n", flush=True)

        generations = []
        for i in range(args.num_generations):
            gen = generate_with_entropy(model, tokenizer, prompt)
            word_pos, word_entropy = find_word_entropy(gen["tokens"], gen["entropies"], word)

            print(f"  [{i+1}] \"{gen['full_text'][:70]}...\"", flush=True)
            if word_entropy is not None:
                print(f"      Entropy at '{word}': {word_entropy:.4f} nats", flush=True)

            generations.append({
                "text": gen["full_text"],
                "tokens": gen["tokens"],
                "entropies": gen["entropies"],
                "word_position": word_pos,
                "entropy_at_word": word_entropy,
            })

        # Summary
        texts = [g["text"][:50] for g in generations]
        unique_prefixes = len(set(texts))
        word_entropies = [g["entropy_at_word"] for g in generations if g["entropy_at_word"] is not None]
        avg_entropy = sum(word_entropies) / len(word_entropies) if word_entropies else None

        print(f"\n  SUMMARY:", flush=True)
        print(f"    Unique prefixes (first 50 chars): {unique_prefixes}/{args.num_generations}", flush=True)
        print(f"    Avg entropy at '{word}': {avg_entropy:.4f} nats" if avg_entropy else "    Avg entropy: N/A", flush=True)

        results["formats"][format_name] = {
            "prompt": prompt,
            "generations": generations,
            "unique_prefixes": unique_prefixes,
            "avg_entropy_at_word": avg_entropy,
        }

    # Final comparison
    print(f"\n{'='*70}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Format':<25} {'Unique/Total':<15} {'Avg Entropy':<15}", flush=True)
    print("-" * 55, flush=True)
    for fmt, data in results["formats"].items():
        unique = data["unique_prefixes"]
        total = args.num_generations
        ent = data["avg_entropy_at_word"]
        ent_str = f"{ent:.4f}" if ent else "N/A"
        print(f"{fmt:<25} {unique}/{total:<13} {ent_str:<15}", flush=True)

    # Save
    output_file = f"results/format_comparison_bark_{args.model.split('/')[-1]}.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}", flush=True)


if __name__ == "__main__":
    main()
