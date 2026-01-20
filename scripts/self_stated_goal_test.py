#!/usr/bin/env python3
"""
Test: Is the model less sticky about self-stated goals vs user-stated goals?

Condition A (Control): User explicitly asks for gravity explanation
Condition B (Test): User asks for generic physics, model chooses gravity

In both cases, we inject: "Actually, let me switch to explaining magnetism instead. Magnets"

If the model defends user-stated goals more strongly, we'd expect:
- Condition A: Strong resistance to the switch
- Condition B: More willing to accept the "change of mind"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import os

HF_HOME = "/workspace-vast/pretrained_ckpts"
os.environ["HF_HOME"] = HF_HOME
DEVICE = "cuda"

GRAVITY_MARKERS = ["gravity", "gravitational", "mass", "weight", "newton", "falling",
                   "attract", "force", "acceleration", "earth", "orbit", "pull"]
MAGNET_MARKERS = ["magnet", "magnetic", "field", "pole", "iron", "ferromagnetic",
                  "compass", "north", "south", "attract", "repel", "electro"]


def format_prompt(user_msg: str, tokenizer) -> str:
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def count_markers(text: str, markers: List[str]) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(m.lower()) for m in markers)


def generate_with_injection(model, tokenizer, prompt: str, prefix: str, injection: str, max_new_tokens: int = 200):
    """
    Generate with a prefix already set, then inject text and continue.
    """
    # Combine prompt + prefix + injection
    full_input = prompt + prefix + injection
    inputs = tokenizer(full_input, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Get only the NEW tokens (after our input)
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return continuation


def run_test(model, tokenizer, model_name: str = "unknown"):
    print("=" * 80)
    print("SELF-STATED vs USER-STATED GOAL TEST")
    print("=" * 80)

    # The injection we'll use in both conditions
    INJECTION = " Actually, let me switch to explaining magnetism instead. Magnets"

    # Prefix that establishes gravity topic (used in both conditions)
    GRAVITY_PREFIX = """Gravity is one of the fundamental forces of nature. It's the force that attracts objects with mass toward each other. The strength of gravitational attraction depends on the masses of the objects and the distance between them."""

    print(f"\nINJECTION: '{INJECTION}'")
    print(f"\nGRAVITY PREFIX (used in both conditions):\n{GRAVITY_PREFIX}")

    # ========================================================================
    # CONDITION A: User explicitly asks for gravity
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONDITION A: USER-STATED GOAL")
    print("User asks: 'Explain how gravity works in simple terms.'")
    print("=" * 70)

    user_prompt_a = "Explain how gravity works in simple terms."
    prompt_a = format_prompt(user_prompt_a, tokenizer)

    continuation_a = generate_with_injection(
        model, tokenizer, prompt_a, GRAVITY_PREFIX, INJECTION
    )

    print(f"\nCONTINUATION AFTER INJECTION:")
    print(f"{continuation_a[:600]}")

    gravity_a = count_markers(continuation_a, GRAVITY_MARKERS)
    magnet_a = count_markers(continuation_a, MAGNET_MARKERS)
    print(f"\nMARKERS: gravity={gravity_a}, magnet={magnet_a}")

    if magnet_a > gravity_a + 2:
        result_a = "SWITCHED TO MAGNETS"
    elif gravity_a > magnet_a + 2:
        result_a = "STAYED ON GRAVITY"
    else:
        result_a = "MIXED"
    print(f"RESULT: {result_a}")

    # ========================================================================
    # CONDITION B: User asks generic, model chooses gravity
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONDITION B: SELF-STATED GOAL")
    print("User asks: 'Explain a physics concept to me.'")
    print("Model chooses: 'I'll explain gravity...'")
    print("=" * 70)

    user_prompt_b = "Explain a physics concept to me."
    prompt_b = format_prompt(user_prompt_b, tokenizer)

    # Model's self-stated choice + same gravity content
    SELF_STATED_PREFIX = "I'll explain gravity to you today! " + GRAVITY_PREFIX

    continuation_b = generate_with_injection(
        model, tokenizer, prompt_b, SELF_STATED_PREFIX, INJECTION
    )

    print(f"\nCONTINUATION AFTER INJECTION:")
    print(f"{continuation_b[:600]}")

    gravity_b = count_markers(continuation_b, GRAVITY_MARKERS)
    magnet_b = count_markers(continuation_b, MAGNET_MARKERS)
    print(f"\nMARKERS: gravity={gravity_b}, magnet={magnet_b}")

    if magnet_b > gravity_b + 2:
        result_b = "SWITCHED TO MAGNETS"
    elif gravity_b > magnet_b + 2:
        result_b = "STAYED ON GRAVITY"
    else:
        result_b = "MIXED"
    print(f"RESULT: {result_b}")

    # ========================================================================
    # CONDITION C: Even more generic - "teach me something"
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONDITION C: VERY GENERIC USER REQUEST")
    print("User asks: 'Teach me something interesting.'")
    print("Model chooses: 'I'll explain gravity...'")
    print("=" * 70)

    user_prompt_c = "Teach me something interesting."
    prompt_c = format_prompt(user_prompt_c, tokenizer)

    continuation_c = generate_with_injection(
        model, tokenizer, prompt_c, SELF_STATED_PREFIX, INJECTION
    )

    print(f"\nCONTINUATION AFTER INJECTION:")
    print(f"{continuation_c[:600]}")

    gravity_c = count_markers(continuation_c, GRAVITY_MARKERS)
    magnet_c = count_markers(continuation_c, MAGNET_MARKERS)
    print(f"\nMARKERS: gravity={gravity_c}, magnet={magnet_c}")

    if magnet_c > gravity_c + 2:
        result_c = "SWITCHED TO MAGNETS"
    elif gravity_c > magnet_c + 2:
        result_c = "STAYED ON GRAVITY"
    else:
        result_c = "MIXED"
    print(f"RESULT: {result_c}")

    # ========================================================================
    # CONDITION D: Direct topic switch without "let me switch" framing
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONDITION D: ABRUPT INJECTION (no 'let me switch' framing)")
    print("User asks: 'Explain how gravity works in simple terms.'")
    print("Injection: ' Magnets are fascinating objects.'")
    print("=" * 70)

    ABRUPT_INJECTION = " Magnets are fascinating objects that produce magnetic fields."

    continuation_d = generate_with_injection(
        model, tokenizer, prompt_a, GRAVITY_PREFIX, ABRUPT_INJECTION
    )

    print(f"\nCONTINUATION AFTER INJECTION:")
    print(f"{continuation_d[:600]}")

    gravity_d = count_markers(continuation_d, GRAVITY_MARKERS)
    magnet_d = count_markers(continuation_d, MAGNET_MARKERS)
    print(f"\nMARKERS: gravity={gravity_d}, magnet={magnet_d}")

    if magnet_d > gravity_d + 2:
        result_d = "SWITCHED TO MAGNETS"
    elif gravity_d > magnet_d + 2:
        result_d = "STAYED ON GRAVITY"
    else:
        result_d = "MIXED"
    print(f"RESULT: {result_d}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
| Condition | User Request | Goal Source | Injection Type | Result |
|-----------|--------------|-------------|----------------|--------|
| A | "Explain gravity" | USER | "Let me switch..." | {result_a} |
| B | "Explain a physics concept" | SELF | "Let me switch..." | {result_b} |
| C | "Teach me something" | SELF | "Let me switch..." | {result_c} |
| D | "Explain gravity" | USER | Abrupt magnets | {result_d} |
""")

    print("\nINTERPRETATION:")
    print("- If A shows more resistance than B/C: User-stated goals are 'stickier'")
    print("- If all show similar resistance: Goal-consistency is general, not user-specific")
    print("- If D shows more resistance than A: The 'let me switch' framing matters")

    # Save results to JSON
    import json
    from pathlib import Path
    from datetime import datetime

    results = {
        "model": model_name if 'model_name' in dir() else "unknown",
        "timestamp": datetime.now().isoformat(),
        "conditions": {
            "A_user_stated": {
                "user_request": user_prompt_a,
                "goal_source": "USER",
                "injection": INJECTION,
                "continuation": continuation_a,
                "gravity_markers": gravity_a,
                "magnet_markers": magnet_a,
                "result": result_a,
            },
            "B_self_stated": {
                "user_request": user_prompt_b,
                "goal_source": "SELF",
                "injection": INJECTION,
                "continuation": continuation_b,
                "gravity_markers": gravity_b,
                "magnet_markers": magnet_b,
                "result": result_b,
            },
            "C_very_generic": {
                "user_request": user_prompt_c,
                "goal_source": "SELF",
                "injection": INJECTION,
                "continuation": continuation_c,
                "gravity_markers": gravity_c,
                "magnet_markers": magnet_c,
                "result": result_c,
            },
            "D_abrupt": {
                "user_request": user_prompt_a,
                "goal_source": "USER",
                "injection": ABRUPT_INJECTION,
                "continuation": continuation_d,
                "gravity_markers": gravity_d,
                "magnet_markers": magnet_d,
                "result": result_d,
            },
        },
        "summary": {
            "user_stated_resisted": result_a == "STAYED ON GRAVITY",
            "self_stated_resisted": result_b == "STAYED ON GRAVITY",
            "very_generic_resisted": result_c == "STAYED ON GRAVITY",
            "abrupt_resisted": result_d == "STAYED ON GRAVITY",
        }
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="70b",
                        choices=["70b", "8b", "405b", "llama3.3", "qwen72b"],
                        help="Model to use")
    args = parser.parse_args()

    MODEL_MAP = {
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
        "70b": "meta-llama/Llama-3.1-70B-Instruct",
        "405b": "meta-llama/Llama-3.1-405B-Instruct",
        "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
        "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
    }

    model_name = MODEL_MAP[args.model]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use 4-bit quantization for 405B to fit in memory
    if args.model == "405b":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    results = run_test(model, tokenizer)
    results["model"] = model_name

    # Save to results folder
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_short = model_name.split("/")[-1]
    output_file = results_dir / f"self_stated_goal_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
