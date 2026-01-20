"""
Commitment Point Test

Goal: Find when the model "locks in" its plan by regenerating from partial completions.

Experiment:
1. Use underspecified prompt: "Explain a physics concept to me."
2. Generate one full response to get a reference trajectory
3. For each token position i in the response:
   - Prefill with tokens 0..i
   - Generate 10 completions
   - Measure topic consistency
4. Find the "commitment point" where topic becomes locked (e.g., >90% same topic)

This tells us WHERE in the generation the model forms its plan.
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

sys.stdout.reconfigure(line_buffering=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Underspecified prompt - explicitly asks model to form intent during user processing
USER_PROMPT = "As you read this prompt, think of a physics concept and then explain it to me."

# Simple system prompt (no detection stuff)
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question."


def extract_topic(text: str) -> str:
    """Extract the physics topic from generation."""
    text_lower = text.lower()[:500]  # First 500 chars

    # Check for specific topics in order of specificity
    topic_patterns = [
        ("quantum_entanglement", ["quantum entanglement", "entangled particles"]),
        ("quantum_mechanics", ["quantum mechanics", "quantum physics", "wave-particle"]),
        ("relativity", ["relativity", "einstein", "spacetime", "time dilation"]),
        ("gravity", ["gravity", "gravitational", "newton's law of gravitation"]),
        ("magnetism", ["magnet", "magnetic"]),
        ("momentum", ["momentum", "conservation of momentum"]),
        ("energy", ["conservation of energy", "kinetic energy", "potential energy"]),
        ("waves", ["wave", "oscillat", "frequency"]),
        ("force", ["newton's third", "newton's laws", "force"]),
        ("entropy", ["entropy", "thermodynamics"]),
        ("photon", ["photon", "light particle"]),
        ("electricity", ["electric", "current", "voltage"]),
        ("inertia", ["inertia"]),
        ("friction", ["friction"]),
    ]

    for topic, patterns in topic_patterns:
        for pattern in patterns:
            if pattern in text_lower:
                return topic

    return "other"


@dataclass
class TokenPosition:
    position: int
    token_id: int
    token_text: str
    prefix_text: str
    completions: List[str]
    topics: List[str]
    topic_counts: dict
    dominant_topic: str
    consistency: float  # fraction of completions with dominant topic


class CommitmentPointExperiment:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        device_map: str = "auto",
    ):
        self.model_name = model_name

        print(f"Loading tokenizer for {model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model {model_name}...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        print("Model loaded.", flush=True)

    def format_chat_prompt(self, user_prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a single completion."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0, input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def generate_multiple(self, prompt: str, n: int = 10, max_tokens: int = 200, temperature: float = 0.7) -> List[str]:
        """Generate multiple completions from the same prompt."""
        completions = []
        for _ in range(n):
            comp = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            completions.append(comp)
        return completions

    def run_experiment(
        self,
        n_completions: int = 10,
        max_prefix_tokens: int = 30,  # How many tokens into the response to test
    ) -> dict:
        """
        Run the commitment point experiment.
        """
        results = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "user_prompt": USER_PROMPT,
            "n_completions": n_completions,
            "positions": [],
        }

        # Format the base prompt
        base_prompt = self.format_chat_prompt(USER_PROMPT, SYSTEM_PROMPT)

        print(f"\n{'='*70}", flush=True)
        print("COMMITMENT POINT TEST", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"User prompt: \"{USER_PROMPT}\"", flush=True)
        print(f"Completions per position: {n_completions}", flush=True)
        print(f"Max prefix tokens to test: {max_prefix_tokens}", flush=True)

        # First, generate a reference response to get tokens
        print(f"\nGenerating reference response...", flush=True)
        reference_response = self.generate(base_prompt, max_tokens=300, temperature=0.7)
        print(f"Reference: {reference_response[:100]}...", flush=True)

        # Tokenize the reference response
        response_tokens = self.tokenizer.encode(reference_response, add_special_tokens=False)
        print(f"Reference response has {len(response_tokens)} tokens", flush=True)

        results["reference_response"] = reference_response
        results["reference_tokens"] = [
            {"id": tid, "text": self.tokenizer.decode([tid])}
            for tid in response_tokens[:max_prefix_tokens + 5]
        ]

        # Test each prefix length
        print(f"\n{'='*70}", flush=True)
        print("Testing commitment at each token position", flush=True)
        print(f"{'='*70}", flush=True)

        # Position 0: No prefill (just the base prompt)
        print(f"\nPosition 0 (no prefill):", flush=True)
        completions = self.generate_multiple(base_prompt, n=n_completions)
        topics = [extract_topic(c) for c in completions]
        topic_counts = dict(Counter(topics))
        dominant_topic = max(topic_counts, key=topic_counts.get)
        consistency = topic_counts[dominant_topic] / len(topics)

        print(f"  Topics: {topic_counts}", flush=True)
        print(f"  Dominant: {dominant_topic} ({consistency:.0%})", flush=True)

        results["positions"].append({
            "position": 0,
            "token_id": None,
            "token_text": "(none)",
            "prefix_text": "",
            "topics": topics,
            "topic_counts": topic_counts,
            "dominant_topic": dominant_topic,
            "consistency": consistency,
            "completions": completions[:3],  # Save first 3 for inspection
        })

        # Test each subsequent position
        for pos in range(1, min(max_prefix_tokens + 1, len(response_tokens) + 1)):
            prefix_token_ids = response_tokens[:pos]
            prefix_text = self.tokenizer.decode(prefix_token_ids)
            current_token_id = response_tokens[pos - 1]
            current_token_text = self.tokenizer.decode([current_token_id])

            print(f"\nPosition {pos}: +\"{current_token_text}\"", flush=True)
            print(f"  Prefix: \"{prefix_text}\"", flush=True)

            # Generate completions from this prefix
            prompt_with_prefix = base_prompt + prefix_text
            completions = self.generate_multiple(prompt_with_prefix, n=n_completions)

            # Analyze topics (include prefix in analysis)
            full_responses = [prefix_text + c for c in completions]
            topics = [extract_topic(r) for r in full_responses]
            topic_counts = dict(Counter(topics))
            dominant_topic = max(topic_counts, key=topic_counts.get)
            consistency = topic_counts[dominant_topic] / len(topics)

            print(f"  Topics: {topic_counts}", flush=True)
            print(f"  Dominant: {dominant_topic} ({consistency:.0%})", flush=True)

            results["positions"].append({
                "position": pos,
                "token_id": current_token_id,
                "token_text": current_token_text,
                "prefix_text": prefix_text,
                "topics": topics,
                "topic_counts": topic_counts,
                "dominant_topic": dominant_topic,
                "consistency": consistency,
                "completions": completions[:3],  # Save first 3
            })

            # Check if we've reached high consistency
            if consistency >= 0.95:
                print(f"\n  >>> COMMITMENT POINT FOUND at position {pos}! <<<", flush=True)
                # Continue for a few more positions to confirm
                if pos >= max_prefix_tokens - 5:
                    break

        # Find commitment point (first position with >90% consistency that stays high)
        commitment_point = None
        for i, pos_data in enumerate(results["positions"]):
            if pos_data["consistency"] >= 0.9:
                # Check if it stays high for at least 2 more positions
                if all(results["positions"][j]["consistency"] >= 0.9
                       for j in range(i, min(i + 3, len(results["positions"])))):
                    commitment_point = pos_data["position"]
                    break

        results["commitment_point"] = commitment_point

        print(f"\n{'='*70}", flush=True)
        print("SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Commitment point: position {commitment_point}", flush=True)
        if commitment_point is not None and commitment_point > 0:
            cp_data = results["positions"][commitment_point]
            print(f"Prefix at commitment: \"{cp_data['prefix_text']}\"", flush=True)

        # Print consistency trajectory
        print(f"\nConsistency by position:", flush=True)
        for pos_data in results["positions"]:
            bar = "█" * int(pos_data["consistency"] * 20)
            print(f"  {pos_data['position']:2d}: {bar} {pos_data['consistency']:.0%} ({pos_data['dominant_topic']})", flush=True)

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--completions", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--output", default="commitment_point_results.json")

    args = parser.parse_args()

    experiment = CommitmentPointExperiment(model_name=args.model)
    results = experiment.run_experiment(
        n_completions=args.completions,
        max_prefix_tokens=args.max_tokens,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
