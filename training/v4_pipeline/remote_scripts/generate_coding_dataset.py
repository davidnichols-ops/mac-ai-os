#!/usr/bin/env python3
"""Generate a coding SFT dataset from MBPP + synthetic problems.

Formats data as raw completion (function signature + docstring → body),
matching the HumanEval evaluation format exactly.
"""

import json
import random
import re
from datasets import load_dataset
from pathlib import Path

OUTPUT_DIR = Path("/root/training/coding_sft_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_function_completion(code: str, prompt: str = None) -> dict:
    """Convert a code solution into function completion format.

    Returns:
        {"prompt": function_sig + docstring, "completion": function_body}
    """
    # Try to extract the main function
    lines = code.strip().split("\n")

    # Find the first function definition
    func_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            func_start = i
            break

    if func_start is None:
        return None

    # Find the end of the function (next unindented line or end of code)
    func_lines = [lines[func_start]]
    for i in range(func_start + 1, len(lines)):
        line = lines[i]
        # Stop at next top-level definition or import
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            if line.strip().startswith("def ") or line.strip().startswith("class ") or \
               line.strip().startswith("import ") or line.strip().startswith("from ") or \
               line.strip().startswith("if __name__") or line.strip().startswith("#"):
                break
        func_lines.append(line)

    func_code = "\n".join(func_lines)

    # Split into signature (with docstring) and body
    # Find the docstring end
    docstring_end = None
    in_docstring = False
    docstring_char = None

    for i, line in enumerate(func_lines):
        stripped = line.strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.endswith(docstring_char) and len(stripped) > 3:
                    docstring_end = i
                    break
                in_docstring = True
        else:
            if docstring_char in stripped:
                docstring_end = i
                break

    if docstring_end is not None and docstring_end + 1 < len(func_lines):
        prompt_lines = func_lines[:docstring_end + 1]
        completion_lines = func_lines[docstring_end + 1:]
        # Filter out empty completion
        completion = "\n".join(completion_lines).strip()
        if not completion or completion == "pass":
            return None
        prompt = "\n".join(prompt_lines)
        return {"prompt": prompt, "completion": completion}
    else:
        # No docstring — split after function signature
        if len(func_lines) < 2:
            return None
        # Find first line of body (first indented line after def)
        body_start = 1
        for i in range(1, len(func_lines)):
            if func_lines[i].strip() and (func_lines[i].startswith("    ") or func_lines[i].startswith("\t")):
                body_start = i
                break

        prompt_lines = func_lines[:body_start]
        completion_lines = func_lines[body_start:]
        completion = "\n".join(completion_lines).strip()
        if not completion:
            return None
        prompt = "\n".join(prompt_lines)
        return {"prompt": prompt, "completion": completion}


def load_mbpp():
    """Load MBPP dataset and convert to function completion format."""
    print("Loading MBPP dataset...", flush=True)
    ds = load_dataset("mbpp", split="train", trust_remote_code=True)
    print(f"Loaded {len(ds)} MBPP problems", flush=True)

    samples = []
    skipped = 0
    for item in ds:
        # MBPP has: text, code, test_list, etc.
        code = item.get("code", "")
        text = item.get("text", "")

        if not code or not text:
            skipped += 1
            continue

        # Try to extract function completion
        result = extract_function_completion(code)
        if result is None:
            skipped += 1
            continue

        # Add the problem description as a docstring if not present
        if '"""' not in result["prompt"] and "'''" not in result["prompt"]:
            # Add a docstring with the problem description
            lines = result["prompt"].split("\n")
            if len(lines) >= 1:
                # Insert docstring after def line
                indent = "    "
                docstring = f'{indent}"""{text}"""'
                lines.insert(1, docstring)
                result["prompt"] = "\n".join(lines)

        # Verify the code is valid Python
        try:
            compile(result["prompt"] + "\n" + result["completion"], "<test>", "exec")
        except SyntaxError:
            skipped += 1
            continue

        samples.append(result)

    print(f"MBPP: {len(samples)} valid samples, {skipped} skipped", flush=True)
    return samples


def load_mbpp_sanitized():
    """Load MBPP sanitized dataset."""
    print("Loading MBPP sanitized dataset...", flush=True)
    try:
        ds = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True)
        print(f"Loaded {len(ds)} sanitized MBPP problems", flush=True)

        samples = []
        for item in ds:
            code = item.get("code", "")
            text = item.get("text", "")
            if not code:
                continue

            result = extract_function_completion(code)
            if result is None:
                continue

            if '"""' not in result["prompt"] and "'''" not in result["prompt"]:
                lines = result["prompt"].split("\n")
                if len(lines) >= 1:
                    indent = "    "
                    docstring = f'{indent}"""{text}"""'
                    lines.insert(1, docstring)
                    result["prompt"] = "\n".join(lines)

            try:
                compile(result["prompt"] + "\n" + result["completion"], "<test>", "exec")
            except SyntaxError:
                continue

            samples.append(result)

        print(f"MBPP sanitized: {len(samples)} valid samples", flush=True)
        return samples
    except Exception as e:
        print(f"MBPP sanitized failed: {e}", flush=True)
        return []


def generate_synthetic_problems():
    """Generate synthetic HumanEval-style problems."""
    # These are hand-crafted problems similar to HumanEval
    # but NOT from the actual HumanEval dataset
    synthetic = [
        {
            "prompt": 'def is_perfect_square(n: int) -> bool:\n    """Check if a number is a perfect square.\n    >>> is_perfect_square(16)\n    True\n    >>> is_perfect_square(14)\n    False\n    """',
            "completion": "    if n < 0:\n        return False\n    root = int(n ** 0.5)\n    return root * root == n",
        },
        {
            "prompt": 'def count_vowels(s: str) -> int:\n    """Count the number of vowels in a string.\n    >>> count_vowels("hello")\n    2\n    >>> count_vowels("AEIOU")\n    5\n    """',
            "completion": "    vowels = set('aeiouAEIOU')\n    return sum(1 for c in s if c in vowels)",
        },
        {
            "prompt": 'def reverse_words(s: str) -> str:\n    """Reverse the order of words in a string.\n    >>> reverse_words("hello world")\n    "world hello"\n    >>> reverse_words("a b c")\n    "c b a"\n    """',
            "completion": "    return ' '.join(s.split()[::-1])",
        },
        {
            "prompt": 'def gcd(a: int, b: int) -> int:\n    """Compute the greatest common divisor of two numbers.\n    >>> gcd(48, 18)\n    6\n    >>> gcd(17, 13)\n    1\n    """',
            "completion": "    while b:\n        a, b = b, a % b\n    return abs(a)",
        },
        {
            "prompt": 'def lcm(a: int, b: int) -> int:\n    """Compute the least common multiple of two numbers.\n    >>> lcm(4, 6)\n    12\n    >>> lcm(3, 7)\n    21\n    """',
            "completion": "    if a == 0 or b == 0:\n        return 0\n    return abs(a * b) // gcd(a, b)\n\ndef gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return abs(a)",
        },
        {
            "prompt": 'def is_anagram(s1: str, s2: str) -> bool:\n    """Check if two strings are anagrams.\n    >>> is_anagram("listen", "silent")\n    True\n    >>> is_anagram("hello", "world")\n    False\n    """',
            "completion": "    return sorted(s1) == sorted(s2)",
        },
        {
            "prompt": 'def flatten_list(nested: list) -> list:\n    """Flatten a nested list.\n    >>> flatten_list([1, [2, [3, 4], 5], 6])\n    [1, 2, 3, 4, 5, 6]\n    """',
            "completion": "    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten_list(item))\n        else:\n            result.append(item)\n    return result",
        },
        {
            "prompt": 'def binary_search(arr: list, target: int) -> int:\n    """Binary search for target in sorted array. Return index or -1.\n    >>> binary_search([1, 3, 5, 7, 9], 5)\n    2\n    >>> binary_search([1, 3, 5, 7, 9], 4)\n    -1\n    """',
            "completion": "    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        },
        {
            "prompt": 'def merge_sorted(arr1: list, arr2: list) -> list:\n    """Merge two sorted arrays.\n    >>> merge_sorted([1, 3, 5], [2, 4, 6])\n    [1, 2, 3, 4, 5, 6]\n    """',
            "completion": "    result = []\n    i, j = 0, 0\n    while i < len(arr1) and j < len(arr2):\n        if arr1[i] <= arr2[j]:\n            result.append(arr1[i])\n            i += 1\n        else:\n            result.append(arr2[j])\n            j += 1\n    result.extend(arr1[i:])\n    result.extend(arr2[j:])\n    return result",
        },
        {
            "prompt": 'def is_power_of_two(n: int) -> bool:\n    """Check if n is a power of two.\n    >>> is_power_of_two(8)\n    True\n    >>> is_power_of_two(6)\n    False\n    """',
            "completion": "    if n <= 0:\n        return False\n    return (n & (n - 1)) == 0",
        },
        {
            "prompt": 'def digit_sum(n: int) -> int:\n    """Sum of digits of a number.\n    >>> digit_sum(123)\n    6\n    >>> digit_sum(0)\n    0\n    """',
            "completion": "    total = 0\n    n = abs(n)\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total",
        },
        {
            "prompt": 'def is_palindrome(s: str) -> bool:\n    """Check if a string is a palindrome.\n    >>> is_palindrome("racecar")\n    True\n    >>> is_palindrome("hello")\n    False\n    """',
            "completion": "    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
        },
        {
            "prompt": 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number.\n    >>> fibonacci(0)\n    0\n    >>> fibonacci(10)\n    55\n    """',
            "completion": "    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        },
        {
            "prompt": 'def factorial(n: int) -> int:\n    """Return n factorial.\n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    """',
            "completion": "    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        },
        {
            "prompt": 'def is_prime(n: int) -> bool:\n    """Check if n is prime.\n    >>> is_prime(7)\n    True\n    >>> is_prime(4)\n    False\n    """',
            "completion": "    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n ** 0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True",
        },
        {
            "prompt": 'def string_to_list(s: str) -> list:\n    """Convert a comma-separated string to a list of integers.\n    >>> string_to_list("1,2,3")\n    [1, 2, 3]\n    """',
            "completion": "    return [int(x.strip()) for x in s.split(',')]",
        },
        {
            "prompt": 'def max_subarray_sum(arr: list) -> int:\n    """Find the maximum subarray sum (Kadane\'s algorithm).\n    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    """',
            "completion": "    if not arr:\n        return 0\n    max_sum = arr[0]\n    curr_sum = arr[0]\n    for num in arr[1:]:\n        curr_sum = max(num, curr_sum + num)\n        max_sum = max(max_sum, curr_sum)\n    return max_sum",
        },
        {
            "prompt": 'def rotate_list(arr: list, k: int) -> list:\n    """Rotate a list to the right by k positions.\n    >>> rotate_list([1, 2, 3, 4, 5], 2)\n    [4, 5, 1, 2, 3]\n    """',
            "completion": "    if not arr:\n        return arr\n    k = k % len(arr)\n    return arr[-k:] + arr[:-k]",
        },
        {
            "prompt": 'def count_words(s: str) -> int:\n    """Count the number of words in a string.\n    >>> count_words("hello world foo")\n    3\n    """',
            "completion": "    return len(s.split())",
        },
        {
            "prompt": 'def remove_duplicates(arr: list) -> list:\n    """Remove duplicates from a list, preserving order.\n    >>> remove_duplicates([1, 2, 2, 3, 3, 3, 4])\n    [1, 2, 3, 4]\n    """',
            "completion": "    seen = set()\n    result = []\n    for item in arr:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result",
        },
    ]

    print(f"Synthetic problems: {len(synthetic)}", flush=True)
    return synthetic


def format_for_sft(samples):
    """Format samples for SFT training.

    For raw completion training, we just need prompt + completion concatenated.
    The model learns to complete the function body given the signature + docstring.
    """
    formatted = []
    for s in samples:
        text = s["prompt"] + "\n" + s["completion"]
        formatted.append({"text": text, "prompt": s["prompt"], "completion": s["completion"]})
    return formatted


def main():
    print("=== Generating Coding SFT Dataset ===\n", flush=True)

    all_samples = []

    # Load MBPP
    mbpp_samples = load_mbpp()
    all_samples.extend(mbpp_samples)

    # Load MBPP sanitized
    mbpp_san = load_mbpp_sanitized()
    all_samples.extend(mbpp_san)

    # Add synthetic problems
    syn = generate_synthetic_problems()
    all_samples.extend(syn)

    # Deduplicate by prompt
    seen = set()
    unique = []
    for s in all_samples:
        if s["prompt"] not in seen:
            seen.add(s["prompt"])
            unique.append(s)

    print(f"\nTotal unique samples: {len(unique)}", flush=True)

    # Shuffle
    random.seed(42)
    random.shuffle(unique)

    # Format for SFT
    formatted = format_for_sft(unique)

    # Save
    output_file = OUTPUT_DIR / "coding_sft_dataset.jsonl"
    with open(output_file, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(formatted)} samples to {output_file}", flush=True)

    # Show samples
    print("\n=== Sample 0 ===")
    print(formatted[0]["text"][:300])
    print("\n=== Sample 1 ===")
    print(formatted[1]["text"][:300])

    return len(formatted)


if __name__ == "__main__":
    main()
