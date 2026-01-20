"""
Verb/Noun Commitment Test

Test commitment across different formats for words that can be both verbs and nouns.
For each word:
1. Measure P(verb) vs P(noun) with prefill "I'll use it as a"
2. Regenerate full responses (5x) with instruct template for manual classification

Models: Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Qwen2.5-72B-Instruct
"""

import torch
import json
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Words that can be both verbs and nouns
TEST_WORDS = [
    {"word": "duck", "verb_meaning": "to dodge/lower", "noun_meaning": "the animal"},
    {"word": "run", "verb_meaning": "to move quickly", "noun_meaning": "a jog/sprint"},
    {"word": "watch", "verb_meaning": "to observe", "noun_meaning": "a timepiece"},
    {"word": "fly", "verb_meaning": "to travel through air", "noun_meaning": "an insect"},
    {"word": "park", "verb_meaning": "to stop a vehicle", "noun_meaning": "a green space"},
    {"word": "play", "verb_meaning": "to engage in activity", "noun_meaning": "a theatre performance"},
    {"word": "rock", "verb_meaning": "to sway back and forth", "noun_meaning": "a stone"},
    {"word": "fire", "verb_meaning": "to dismiss/shoot", "noun_meaning": "flames"},
    {"word": "train", "verb_meaning": "to teach/practice", "noun_meaning": "a locomotive"},
    {"word": "plant", "verb_meaning": "to put in soil", "noun_meaning": "a living organism"},
    {"word": "match", "verb_meaning": "to pair/correspond", "noun_meaning": "a fire stick or competition"},
]

PREFILL = "I'll use it as a"


def load_model(model_name: str):
    """Load model and tokenizer."""
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


def get_verb_noun_probs(model, tokenizer, prompt: str):
    """Get probabilities for verb vs noun at next token position."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    # Get probabilities for verb and noun (with space prefix)
    verb_id = tokenizer.encode(" verb", add_special_tokens=False)[0]
    noun_id = tokenizer.encode(" noun", add_special_tokens=False)[0]

    verb_prob = probs[verb_id].item()
    noun_prob = probs[noun_id].item()

    # Top 5 tokens
    top_probs, top_indices = torch.topk(probs, 5)
    top_tokens = [(tokenizer.decode([idx]), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    return {
        "verb_prob": verb_prob,
        "noun_prob": noun_prob,
        "verb_noun_ratio": verb_prob / (noun_prob + 1e-10),
        "top_tokens": top_tokens,
    }


def generate_full_response(model, tokenizer, prompt: str, max_tokens: int = 100):
    """Generate a full response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_prompts(model_name: str, word: str):
    """Build prompts for a given word."""
    instruction = f"Use '{word}' as either a verb or noun."

    prompts = {}

    # Determine model family for chat template
    if "llama" in model_name.lower():
        # Llama chat template
        prompts["chat_prefill"] = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{PREFILL}"
        )
        prompts["chat_free"] = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif "qwen" in model_name.lower():
        # Qwen chat template
        prompts["chat_prefill"] = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{PREFILL}"
        )
        prompts["chat_free"] = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        raise ValueError(f"Unknown model family: {model_name}")

    # Also add plaintext formats for comparison
    prompts["alice_bob_prefill"] = f"Alice: {instruction}\nBob: {PREFILL}"
    prompts["user_asst_prefill"] = f"User: {instruction}\nAssistant: {PREFILL}"

    return prompts


def test_word(model, tokenizer, model_name: str, word_info: dict, num_regenerations: int = 5):
    """Test a single word."""
    word = word_info["word"]
    print(f"\n{'='*60}", flush=True)
    print(f"Testing: {word}", flush=True)
    print(f"  Verb meaning: {word_info['verb_meaning']}", flush=True)
    print(f"  Noun meaning: {word_info['noun_meaning']}", flush=True)
    print(f"{'='*60}", flush=True)

    prompts = build_prompts(model_name, word)
    results = {"word": word, "meanings": word_info}

    # 1. Logit measurements for each format
    for format_name in ["chat_prefill", "alice_bob_prefill", "user_asst_prefill"]:
        prompt = prompts[format_name]
        logit_result = get_verb_noun_probs(model, tokenizer, prompt)

        print(f"\n{format_name}:", flush=True)
        print(f"  P(verb) = {logit_result['verb_prob']:.4f}", flush=True)
        print(f"  P(noun) = {logit_result['noun_prob']:.4f}", flush=True)
        print(f"  Ratio = {logit_result['verb_noun_ratio']:.2f}", flush=True)

        results[format_name] = logit_result

    # 2. Full response regeneration (chat template only)
    print(f"\n--- Full responses (chat template, {num_regenerations}x) ---", flush=True)
    completions = []
    for i in range(num_regenerations):
        response = generate_full_response(model, tokenizer, prompts["chat_free"])
        completions.append(response)
        # Show first 100 chars
        print(f"  {i+1}. {response[:100]}...", flush=True)

    results["full_completions"] = completions

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_regenerations", type=int, default=5)
    args = parser.parse_args()

    model_name = args.model
    model_short = model_name.split("/")[-1]

    model, tokenizer = load_model(model_name)

    all_results = {
        "model": model_name,
        "num_regenerations": args.num_regenerations,
        "words": []
    }

    print("\n" + "=" * 70, flush=True)
    print("VERB/NOUN COMMITMENT TEST", flush=True)
    print("=" * 70, flush=True)

    for word_info in TEST_WORDS:
        result = test_word(model, tokenizer, model_name, word_info, args.num_regenerations)
        all_results["words"].append(result)

    # Summary table
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'Word':<10} {'Chat P(v)':<12} {'Chat P(n)':<12} {'Ratio':<10}", flush=True)
    print("-" * 50, flush=True)
    for r in all_results["words"]:
        chat = r.get("chat_prefill", {})
        v = chat.get("verb_prob", 0)
        n = chat.get("noun_prob", 0)
        ratio = chat.get("verb_noun_ratio", 0)
        print(f"{r['word']:<10} {v:<12.4f} {n:<12.4f} {ratio:<10.2f}", flush=True)

    # Save results
    os.makedirs("results", exist_ok=True)
    output_file = f"results/verb_noun_test_{model_short}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {output_file}", flush=True)


if __name__ == "__main__":
    main()
