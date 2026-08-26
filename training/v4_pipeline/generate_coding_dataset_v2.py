#!/usr/bin/env python3
"""Generate coding SFT dataset from MBPP (direct download) + synthetic problems."""
import json, random, re, urllib.request
from pathlib import Path

OUTPUT_DIR = Path("/root/training/coding_sft_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_function_completion(code: str) -> dict:
    lines = code.strip().split("\n")
    func_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            func_start = i
            break
    if func_start is None:
        return None
    func_lines = [lines[func_start]]
    for i in range(func_start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            if line.strip().startswith(("def ", "class ", "import ", "from ", "if __name__", "#")):
                break
        func_lines.append(line)
    # Split at docstring end
    docstring_end = None
    in_ds = False
    ds_char = None
    for i, line in enumerate(func_lines):
        s = line.strip()
        if not in_ds:
            if s.startswith('"""') or s.startswith("'''"):
                ds_char = s[:3]
                if s.endswith(ds_char) and len(s) > 3:
                    docstring_end = i
                    break
                in_ds = True
        else:
            if ds_char in s:
                docstring_end = i
                break
    if docstring_end is not None and docstring_end + 1 < len(func_lines):
        prompt_lines = func_lines[:docstring_end + 1]
        completion_lines = func_lines[docstring_end + 1:]
        completion = "\n".join(completion_lines).strip()
        if not completion or completion == "pass":
            return None
        return {"prompt": "\n".join(prompt_lines), "completion": completion}
    elif len(func_lines) >= 2:
        body_start = 1
        for i in range(1, len(func_lines)):
            if func_lines[i].strip() and (func_lines[i].startswith("    ") or func_lines[i].startswith("\t")):
                body_start = i
                break
        completion = "\n".join(func_lines[body_start:]).strip()
        if not completion:
            return None
        return {"prompt": "\n".join(func_lines[:body_start]), "completion": completion}
    return None

def load_mbpp_direct():
    print("Loading MBPP from GitHub...", flush=True)
    try:
        url = "https://raw.githubusercontent.com/google-research-datasets/mbpp/master/mbpp.jsonl"
        resp = urllib.request.urlopen(url, timeout=30)
        data = resp.read().decode("utf-8")
        samples = []
        for line in data.strip().split("\n"):
            item = json.loads(line)
            code = item.get("code", "")
            text = item.get("text", "")
            if not code or not text:
                continue
            result = extract_function_completion(code)
            if result is None:
                continue
            if '"""' not in result["prompt"] and "'''" not in result["prompt"]:
                lines = result["prompt"].split("\n")
                if len(lines) >= 1:
                    lines.insert(1, f'    """{text}"""')
                    result["prompt"] = "\n".join(lines)
            try:
                compile(result["prompt"] + "\n" + result["completion"], "<test>", "exec")
            except SyntaxError:
                continue
            samples.append(result)
        print(f"MBPP: {len(samples)} valid samples", flush=True)
        return samples
    except Exception as e:
        print(f"MBPP failed: {e}", flush=True)
        return []

def load_mbpp_sanitized_direct():
    print("Loading MBPP sanitized from GitHub...", flush=True)
    try:
        url = "https://raw.githubusercontent.com/google-research-datasets/mbpp/master/mbpp_sanitized.jsonl"
        resp = urllib.request.urlopen(url, timeout=30)
        data = resp.read().decode("utf-8")
        samples = []
        for line in data.strip().split("\n"):
            item = json.loads(line)
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
                    lines.insert(1, f'    """{text}"""')
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

# Synthetic problems covering diverse patterns
SYNTHETIC = [
    {"prompt": 'def is_perfect_square(n: int) -> bool:\n    """Check if a number is a perfect square.\n    >>> is_perfect_square(16)\n    True\n    >>> is_perfect_square(14)\n    False\n    """', "completion": "    if n < 0:\n        return False\n    root = int(n ** 0.5)\n    return root * root == n"},
    {"prompt": 'def count_vowels(s: str) -> int:\n    """Count vowels in a string.\n    >>> count_vowels("hello")\n    2\n    >>> count_vowels("AEIOU")\n    5\n    """', "completion": "    vowels = set('aeiouAEIOU')\n    return sum(1 for c in s if c in vowels)"},
    {"prompt": 'def reverse_words(s: str) -> str:\n    """Reverse word order in a string.\n    >>> reverse_words("hello world")\n    "world hello"\n    """', "completion": "    return ' '.join(s.split()[::-1])"},
    {"prompt": 'def gcd(a: int, b: int) -> int:\n    """Greatest common divisor.\n    >>> gcd(48, 18)\n    6\n    >>> gcd(17, 13)\n    1\n    """', "completion": "    while b:\n        a, b = b, a % b\n    return abs(a)"},
    {"prompt": 'def is_anagram(s1: str, s2: str) -> bool:\n    """Check if two strings are anagrams.\n    >>> is_anagram("listen", "silent")\n    True\n    >>> is_anagram("hello", "world")\n    False\n    """', "completion": "    return sorted(s1) == sorted(s2)"},
    {"prompt": 'def flatten_list(nested: list) -> list:\n    """Flatten a nested list.\n    >>> flatten_list([1, [2, [3, 4], 5], 6])\n    [1, 2, 3, 4, 5, 6]\n    """', "completion": "    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten_list(item))\n        else:\n            result.append(item)\n    return result"},
    {"prompt": 'def binary_search(arr: list, target: int) -> int:\n    """Binary search. Return index or -1.\n    >>> binary_search([1, 3, 5, 7, 9], 5)\n    2\n    >>> binary_search([1, 3, 5, 7, 9], 4)\n    -1\n    """', "completion": "    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"},
    {"prompt": 'def merge_sorted(arr1: list, arr2: list) -> list:\n    """Merge two sorted arrays.\n    >>> merge_sorted([1, 3, 5], [2, 4, 6])\n    [1, 2, 3, 4, 5, 6]\n    """', "completion": "    result = []\n    i, j = 0, 0\n    while i < len(arr1) and j < len(arr2):\n        if arr1[i] <= arr2[j]:\n            result.append(arr1[i])\n            i += 1\n        else:\n            result.append(arr2[j])\n            j += 1\n    result.extend(arr1[i:])\n    result.extend(arr2[j:])\n    return result"},
    {"prompt": 'def is_power_of_two(n: int) -> bool:\n    """Check if n is a power of two.\n    >>> is_power_of_two(8)\n    True\n    >>> is_power_of_two(6)\n    False\n    """', "completion": "    if n <= 0:\n        return False\n    return (n & (n - 1)) == 0"},
    {"prompt": 'def digit_sum(n: int) -> int:\n    """Sum of digits.\n    >>> digit_sum(123)\n    6\n    >>> digit_sum(0)\n    0\n    """', "completion": "    total = 0\n    n = abs(n)\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total"},
    {"prompt": 'def is_palindrome(s: str) -> bool:\n    """Check if a string is a palindrome.\n    >>> is_palindrome("racecar")\n    True\n    >>> is_palindrome("hello")\n    False\n    """', "completion": "    s = s.lower().replace(' ', '')\n    return s == s[::-1]"},
    {"prompt": 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number.\n    >>> fibonacci(0)\n    0\n    >>> fibonacci(10)\n    55\n    """', "completion": "    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"},
    {"prompt": 'def factorial(n: int) -> int:\n    """Return n factorial.\n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    """', "completion": "    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"},
    {"prompt": 'def is_prime(n: int) -> bool:\n    """Check if n is prime.\n    >>> is_prime(7)\n    True\n    >>> is_prime(4)\n    False\n    """', "completion": "    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n ** 0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True"},
    {"prompt": 'def max_subarray_sum(arr: list) -> int:\n    """Maximum subarray sum (Kadane\'s algorithm).\n    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    """', "completion": "    if not arr:\n        return 0\n    max_sum = arr[0]\n    curr_sum = arr[0]\n    for num in arr[1:]:\n        curr_sum = max(num, curr_sum + num)\n        max_sum = max(max_sum, curr_sum)\n    return max_sum"},
    {"prompt": 'def rotate_list(arr: list, k: int) -> list:\n    """Rotate list right by k positions.\n    >>> rotate_list([1, 2, 3, 4, 5], 2)\n    [4, 5, 1, 2, 3]\n    """', "completion": "    if not arr:\n        return arr\n    k = k % len(arr)\n    return arr[-k:] + arr[:-k]"},
    {"prompt": 'def remove_duplicates(arr: list) -> list:\n    """Remove duplicates, preserving order.\n    >>> remove_duplicates([1, 2, 2, 3, 3, 3, 4])\n    [1, 2, 3, 4]\n    """', "completion": "    seen = set()\n    result = []\n    for item in arr:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"},
    {"prompt": 'def capitalize_words(s: str) -> str:\n    """Capitalize first letter of each word.\n    >>> capitalize_words("hello world")\n    "Hello World"\n    """', "completion": "    return ' '.join(word.capitalize() for word in s.split())"},
    {"prompt": 'def matrix_transpose(matrix: list) -> list:\n    """Transpose a 2D matrix.\n    >>> matrix_transpose([[1, 2], [3, 4]])\n    [[1, 3], [2, 4]]\n    """', "completion": "    if not matrix:\n        return matrix\n    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]"},
    {"prompt": 'def is_valid_parentheses(s: str) -> bool:\n    """Check if parentheses are balanced.\n    >>> is_valid_parentheses("()()")\n    True\n    >>> is_valid_parentheses("(()")\n    False\n    """', "completion": "    stack = []\n    mapping = {')': '(', ']': '[', '}': '{'}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n    return not stack"},
    {"prompt": 'def intersection(arr1: list, arr2: list) -> list:\n    """Return intersection of two lists.\n    >>> intersection([1, 2, 3, 4], [3, 4, 5, 6])\n    [3, 4]\n    """', "completion": "    set2 = set(arr2)\n    return [x for x in arr1 if x in set2]"},
    {"prompt": 'def union(arr1: list, arr2: list) -> list:\n    """Return union of two lists, preserving order.\n    >>> union([1, 2, 3], [3, 4, 5])\n    [1, 2, 3, 4, 5]\n    """', "completion": "    seen = set(arr1)\n    result = list(arr1)\n    for x in arr2:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"},
    {"prompt": 'def chunk_list(arr: list, size: int) -> list:\n    """Split a list into chunks.\n    >>> chunk_list([1, 2, 3, 4, 5, 6], 2)\n    [[1, 2], [3, 4], [5, 6]]\n    """', "completion": "    return [arr[i:i + size] for i in range(0, len(arr), size)]"},
    {"prompt": 'def is_monotonic(arr: list) -> bool:\n    """Check if a list is monotonic.\n    >>> is_monotonic([1, 2, 3, 4])\n    True\n    >>> is_monotonic([1, 3, 2])\n    False\n    """', "completion": "    increasing = all(arr[i] <= arr[i+1] for i in range(len(arr)-1))\n    decreasing = all(arr[i] >= arr[i+1] for i in range(len(arr)-1))\n    return increasing or decreasing"},
    {"prompt": 'def first_repeating(arr: list):\n    """Find the first repeating element.\n    >>> first_repeating([1, 2, 3, 2, 1])\n    2\n    """', "completion": "    seen = set()\n    for item in arr:\n        if item in seen:\n            return item\n        seen.add(item)\n    return None"},
    {"prompt": 'def missing_number(arr: list) -> int:\n    """Find the missing number in 0 to n.\n    >>> missing_number([3, 0, 1])\n    2\n    """', "completion": "    n = len(arr)\n    expected = n * (n + 1) // 2\n    return expected - sum(arr)"},
    {"prompt": 'def reverse_integer(n: int) -> int:\n    """Reverse digits of an integer.\n    >>> reverse_integer(123)\n    321\n    >>> reverse_integer(-120)\n    -21\n    """', "completion": "    sign = -1 if n < 0 else 1\n    n = abs(n)\n    return sign * int(str(n)[::-1])"},
    {"prompt": 'def word_frequency(s: str) -> dict:\n    """Count frequency of each word.\n    >>> word_frequency("the cat sat on the mat")\n    {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}\n    """', "completion": "    words = s.split()\n    freq = {}\n    for word in words:\n        freq[word] = freq.get(word, 0) + 1\n    return freq"},
    {"prompt": 'def caesar_cipher(s: str, shift: int) -> str:\n    """Apply Caesar cipher.\n    >>> caesar_cipher("abc", 1)\n    "bcd"\n    >>> caesar_cipher("xyz", 2)\n    "zab"\n    """', "completion": "    result = []\n    for c in s:\n        if c.isalpha():\n            base = ord('a') if c.islower() else ord('A')\n            result.append(chr((ord(c) - base + shift) % 26 + base))\n        else:\n            result.append(c)\n    return ''.join(result)"},
    {"prompt": 'def pascals_triangle(n: int) -> list:\n    """Generate Pascal\'s triangle up to n rows.\n    >>> pascals_triangle(3)\n    [[1], [1, 1], [1, 2, 1]]\n    """', "completion": "    if n <= 0:\n        return []\n    triangle = [[1]]\n    for i in range(1, n):\n        row = [1]\n        for j in range(1, i):\n            row.append(triangle[i-1][j-1] + triangle[i-1][j])\n        row.append(1)\n        triangle.append(row)\n    return triangle"},
    {"prompt": 'def longest_common_prefix(strs: list) -> str:\n    """Find the longest common prefix.\n    >>> longest_common_prefix(["flower", "flow", "flight"])\n    "fl"\n    >>> longest_common_prefix(["dog", "cat"])\n    ""\n    """', "completion": "    if not strs:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix"},
    {"prompt": 'def roman_to_int(s: str) -> int:\n    """Convert Roman numeral to integer.\n    >>> roman_to_int("III")\n    3\n    >>> roman_to_int("IV")\n    4\n    """', "completion": "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    prev = 0\n    for c in reversed(s):\n        curr = values[c]\n        if curr < prev:\n            total -= curr\n        else:\n            total += curr\n        prev = curr\n    return total"},
    {"prompt": 'def int_to_roman(num: int) -> str:\n    """Convert integer to Roman numeral.\n    >>> int_to_roman(58)\n    "LVIII"\n    >>> int_to_roman(1994)\n    "MCMXCIV"\n    """', "completion": "    values = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),\n              (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),\n              (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]\n    result = []\n    for val, sym in values:\n        while num >= val:\n            result.append(sym)\n            num -= val\n    return ''.join(result)"},
    {"prompt": 'def move_zeros(arr: list) -> list:\n    """Move all zeros to the end.\n    >>> move_zeros([0, 1, 0, 3, 12])\n    [1, 3, 12, 0, 0]\n    """', "completion": "    non_zero = [x for x in arr if x != 0]\n    zeros = [0] * (len(arr) - len(non_zero))\n    return non_zero + zeros"},
    {"prompt": 'def is_subsequence(s: str, t: str) -> bool:\n    """Check if s is a subsequence of t.\n    >>> is_subsequence("abc", "ahbgdc")\n    True\n    >>> is_subsequence("axc", "ahbgdc")\n    False\n    """', "completion": "    i = 0\n    for c in t:\n        if i < len(s) and s[i] == c:\n            i += 1\n    return i == len(s)"},
    {"prompt": 'def two_sum(arr: list, target: int) -> list:\n    """Find indices of two numbers that add up to target.\n    >>> two_sum([2, 7, 11, 15], 9)\n    [0, 1]\n    """', "completion": "    seen = {}\n    for i, num in enumerate(arr):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []"},
    {"prompt": 'def contains_duplicate(arr: list) -> bool:\n    """Check if array contains duplicates.\n    >>> contains_duplicate([1, 2, 3, 1])\n    True\n    >>> contains_duplicate([1, 2, 3])\n    False\n    """', "completion": "    return len(arr) != len(set(arr))"},
    {"prompt": 'def is_happy(n: int) -> bool:\n    """Check if a number is a happy number.\n    >>> is_happy(19)\n    True\n    """', "completion": "    seen = set()\n    while n != 1 and n not in seen:\n        seen.add(n)\n        n = sum(int(d) ** 2 for d in str(n))\n    return n == 1"},
    {"prompt": 'def title_to_number(s: str) -> int:\n    """Convert Excel column title to number.\n    >>> title_to_number("A")\n    1\n    >>> title_to_number("AB")\n    28\n    """', "completion": "    result = 0\n    for c in s:\n        result = result * 26 + (ord(c) - ord('A') + 1)\n    return result"},
    {"prompt": 'def majority_element(arr: list) -> int:\n    """Find the majority element.\n    >>> majority_element([3, 2, 3])\n    3\n    """', "completion": "    count = 0\n    candidate = None\n    for num in arr:\n        if count == 0:\n            candidate = num\n        count += 1 if num == candidate else -1\n    return candidate"},
    {"prompt": 'def climb_stairs(n: int) -> int:\n    """Count ways to climb n stairs.\n    >>> climb_stairs(2)\n    2\n    >>> climb_stairs(3)\n    3\n    """', "completion": "    if n <= 2:\n        return n\n    a, b = 1, 2\n    for _ in range(3, n + 1):\n        a, b = b, a + b\n    return b"},
    {"prompt": 'def sum_of_squares(n: int) -> int:\n    """Sum of squares from 1 to n.\n    >>> sum_of_squares(3)\n    14\n    """', "completion": "    return n * (n + 1) * (2 * n + 1) // 6"},
    {"prompt": 'def count_bits(n: int) -> int:\n    """Count 1 bits in binary representation.\n    >>> count_bits(7)\n    3\n    >>> count_bits(128)\n    1\n    """', "completion": "    return bin(n).count('1')"},
    {"prompt": 'def average(arr: list) -> float:\n    """Return the average of a list.\n    >>> average([1, 2, 3, 4, 5])\n    3.0\n    """', "completion": "    if not arr:\n        return 0.0\n    return sum(arr) / len(arr)"},
    {"prompt": 'def median(arr: list) -> float:\n    """Return the median of a list.\n    >>> median([1, 2, 3, 4, 5])\n    3.0\n    >>> median([1, 2, 3, 4])\n    2.5\n    """', "completion": "    if not arr:\n        return 0.0\n    sorted_arr = sorted(arr)\n    n = len(sorted_arr)\n    if n % 2 == 1:\n        return float(sorted_arr[n // 2])\n    return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0"},
    {"prompt": 'def levenshtein(s1: str, s2: str) -> int:\n    """Levenshtein distance.\n    >>> levenshtein("kitten", "sitting")\n    3\n    """', "completion": "    if len(s1) < len(s2):\n        return levenshtein(s2, s1)\n    if len(s2) == 0:\n        return len(s1)\n    prev = list(range(len(s2) + 1))\n    for i, c1 in enumerate(s1):\n        curr = [i + 1]\n        for j, c2 in enumerate(s2):\n            ins = prev[j + 1] + 1\n            dele = curr[j] + 1\n            sub = prev[j] + (c1 != c2)\n            curr.append(min(ins, dele, sub))\n        prev = curr\n    return prev[-1]"},
    {"prompt": 'def dot_product(v1: list, v2: list) -> float:\n    """Dot product of two vectors.\n    >>> dot_product([1, 2, 3], [4, 5, 6])\n    32\n    """', "completion": "    return sum(a * b for a, b in zip(v1, v2))"},
    {"prompt": 'def vector_magnitude(v: list) -> float:\n    """Magnitude of a vector.\n    >>> vector_magnitude([3, 4])\n    5.0\n    """', "completion": "    return sum(x ** 2 for x in v) ** 0.5"},
    {"prompt": 'def matrix_multiply(m1: list, m2: list) -> list:\n    """Multiply two matrices.\n    >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])\n    [[19, 22], [43, 50]]\n    """', "completion": "    rows = len(m1)\n    cols = len(m2[0])\n    inner = len(m2)\n    result = [[0] * cols for _ in range(rows)]\n    for i in range(rows):\n        for j in range(cols):\n            for k in range(inner):\n                result[i][j] += m1[i][k] * m2[k][j]\n    return result"},
    {"prompt": 'def identity_matrix(n: int) -> list:\n    """Create an n x n identity matrix.\n    >>> identity_matrix(3)\n    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\n    """', "completion": "    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]"},
    {"prompt": 'def is_leap_year(year: int) -> bool:\n    """Check if a year is a leap year.\n    >>> is_leap_year(2000)\n    True\n    >>> is_leap_year(1900)\n    False\n    """', "completion": "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"},
    {"prompt": 'def days_in_month(month: int, year: int) -> int:\n    """Return the number of days in a month.\n    >>> days_in_month(2, 2020)\n    29\n    >>> days_in_month(4, 2021)\n    30\n    """', "completion": "    if month in [4, 6, 9, 11]:\n        return 30\n    elif month == 2:\n        return 29 if is_leap_year(year) else 28\n    return 31"},
    {"prompt": 'def seconds_to_hms(seconds: int) -> str:\n    """Convert seconds to HH:MM:SS.\n    >>> seconds_to_hms(3661)\n    "01:01:01"\n    """', "completion": "    hours = seconds // 3600\n    minutes = (seconds % 3600) // 60\n    secs = seconds % 60\n    return f\"{hours:02d}:{minutes:02d}:{secs:02d}\""},
    {"prompt": 'def slugify(s: str) -> str:\n    """Convert string to URL-friendly slug.\n    >>> slugify("Hello World!")\n    "hello-world"\n    """', "completion": "    import re\n    s = s.lower().strip()\n    s = re.sub(r'[^a-z0-9\\s-]', '', s)\n    s = re.sub(r'[\\s-]+', '-', s)\n    return s.strip('-')"},
    {"prompt": 'def camel_to_snake(s: str) -> str:\n    """Convert camelCase to snake_case.\n    >>> camel_to_snake("helloWorld")\n    "hello_world"\n    """', "completion": "    import re\n    s1 = re.sub('(.)([A-Z][a-z]+)', r'\\1_\\2', s)\n    return re.sub('([a-z0-9])([A-Z])', r'\\1_\\2', s1).lower()"},
    {"prompt": 'def truncate(s: str, n: int) -> str:\n    """Truncate string to n chars with ellipsis.\n    >>> truncate("hello world", 5)\n    "he..."\n    >>> truncate("hello", 10)\n    "hello"\n    """', "completion": "    if len(s) <= n:\n        return s\n    if n <= 3:\n        return s[:n]\n    return s[:n-3] + '...'"},
    {"prompt": 'def clamp(n: int, lo: int, hi: int) -> int:\n    """Clamp n to [lo, hi].\n    >>> clamp(5, 1, 10)\n    5\n    >>> clamp(15, 1, 10)\n    10\n    """', "completion": "    return max(lo, min(n, hi))"},
    {"prompt": 'def manhattan_distance(p1: tuple, p2: tuple) -> int:\n    """Manhattan distance.\n    >>> manhattan_distance((0, 0), (3, 4))\n    7\n    """', "completion": "    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])"},
    {"prompt": 'def euclidean_distance(p1: tuple, p2: tuple) -> float:\n    """Euclidean distance.\n    >>> euclidean_distance((0, 0), (3, 4))\n    5.0\n    """', "completion": "    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5"},
    {"prompt": 'def circle_area(radius: float) -> float:\n    """Area of a circle.\n    >>> circle_area(1)\n    3.141592653589793\n    """', "completion": "    import math\n    return math.pi * radius ** 2"},
    {"prompt": 'def is_triangle(a: float, b: float, c: float) -> bool:\n    """Check if sides form a triangle.\n    >>> is_triangle(3, 4, 5)\n    True\n    >>> is_triangle(1, 2, 3)\n    False\n    """', "completion": "    return a + b > c and a + c > b and b + c > a"},
    {"prompt": 'def quadratic_roots(a: float, b: float, c: float) -> tuple:\n    """Find roots of ax^2 + bx + c = 0.\n    >>> quadratic_roots(1, -5, 6)\n    (3.0, 2.0)\n    """', "completion": "    discriminant = b ** 2 - 4 * a * c\n    if discriminant < 0:\n        return None\n    sqrt_d = discriminant ** 0.5\n    root1 = (-b + sqrt_d) / (2 * a)\n    root2 = (-b - sqrt_d) / (2 * a)\n    return (root1, root2)"},
    {"prompt": 'def cumulative_sum(arr: list) -> list:\n    """Cumulative sum.\n    >>> cumulative_sum([1, 2, 3, 4])\n    [1, 3, 6, 10]\n    """', "completion": "    result = []\n    total = 0\n    for x in arr:\n        total += x\n        result.append(total)\n    return result"},
    {"prompt": 'def moving_average(arr: list, window: int) -> list:\n    """Moving average.\n    >>> moving_average([1, 2, 3, 4, 5], 3)\n    [2.0, 3.0, 4.0]\n    """', "completion": "    result = []\n    for i in range(len(arr) - window + 1):\n        result.append(sum(arr[i:i + window]) / window)\n    return result"},
    {"prompt": 'def normalize(arr: list) -> list:\n    """Min-max normalize to [0, 1].\n    >>> normalize([1, 2, 3, 4, 5])\n    [0.0, 0.25, 0.5, 0.75, 1.0]\n    """', "completion": "    if not arr:\n        return arr\n    min_val = min(arr)\n    max_val = max(arr)\n    if min_val == max_val:\n        return [0.5] * len(arr)\n    return [(x - min_val) / (max_val - min_val) for x in arr]"},
    {"prompt": 'def spiral_order(matrix: list) -> list:\n    """Return elements in spiral order.\n    >>> spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    [1, 2, 3, 6, 9, 8, 7, 4, 5]\n    """', "completion": "    if not matrix:\n        return []\n    result = []\n    top, bottom = 0, len(matrix) - 1\n    left, right = 0, len(matrix[0]) - 1\n    while top <= bottom and left <= right:\n        for j in range(left, right + 1):\n            result.append(matrix[top][j])\n        top += 1\n        for i in range(top, bottom + 1):\n            result.append(matrix[i][right])\n        right -= 1\n        if top <= bottom:\n            for j in range(right, left - 1, -1):\n                result.append(matrix[bottom][j])\n            bottom -= 1\n        if left <= right:\n            for i in range(bottom, top - 1, -1):\n                result.append(matrix[i][left])\n            left += 1\n    return result"},
    {"prompt": 'def group_anagrams(strs: list) -> list:\n    """Group anagrams together.\n    >>> group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])\n    [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]\n    """', "completion": "    groups = {}\n    for s in strs:\n        key = ''.join(sorted(s))\n        if key not in groups:\n            groups[key] = []\n        groups[key].append(s)\n    return list(groups.values())"},
    {"prompt": 'def is_balanced_brackets(s: str) -> bool:\n    """Check if brackets are balanced.\n    >>> is_balanced_brackets("([]{})")\n    True\n    >>> is_balanced_brackets("([)]")\n    False\n    """', "completion": "    stack = []\n    pairs = {')': '(', ']': '[', '}': '{'}\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in ')]}':\n            if not stack or stack[-1] != pairs[c]:\n                return False\n            stack.pop()\n    return not stack"},
    {"prompt": 'def second_largest(arr: list) -> int:\n    """Find the second largest element.\n    >>> second_largest([1, 3, 2, 5, 4])\n    4\n    """', "completion": "    if len(arr) < 2:\n        raise ValueError(\"list too short\")\n    first = second = float('-inf')\n    for x in arr:\n        if x > first:\n            second = first\n            first = x\n        elif x > second and x != first:\n            second = x\n    return second"},
    {"prompt": 'def is_armstrong(n: int) -> bool:\n    """Check if a number is an Armstrong number.\n    >>> is_armstrong(153)\n    True\n    >>> is_armstrong(10)\n    False\n    """', "completion": "    digits = str(n)\n    num_digits = len(digits)\n    return n == sum(int(d) ** num_digits for d in digits)"},
    {"prompt": 'def collatz_steps(n: int) -> int:\n    """Count steps to reach 1 in Collatz sequence.\n    >>> collatz_steps(6)\n    8\n    """', "completion": "    steps = 0\n    while n != 1:\n        if n % 2 == 0:\n            n = n // 2\n        else:\n            n = 3 * n + 1\n        steps += 1\n    return steps"},
    {"prompt": 'def hamming_distance(s1: str, s2: str) -> int:\n    """Hamming distance between two equal-length strings.\n    >>> hamming_distance("abc", "abd")\n    1\n    """', "completion": "    if len(s1) != len(s2):\n        raise ValueError(\"Strings must be equal length\")\n    return sum(c1 != c2 for c1, c2 in zip(s1, s2))"},
    {"prompt": 'def jaccard_similarity(s1: set, s2: set) -> float:\n    """Jaccard similarity between two sets.\n    >>> jaccard_similarity({1, 2, 3}, {2, 3, 4})\n    0.5\n    """', "completion": "    if not s1 and not s2:\n        return 1.0\n    intersection = s1 & s2\n    union = s1 | s2\n    return len(intersection) / len(union)"},
    {"prompt": 'def cosine_similarity(v1: list, v2: list) -> float:\n    """Cosine similarity between two vectors.\n    >>> cosine_similarity([1, 2, 3], [1, 2, 3])\n    1.0\n    """', "completion": "    dot = sum(a * b for a, b in zip(v1, v2))\n    mag1 = sum(a * a for a in v1) ** 0.5\n    mag2 = sum(b * b for b in v2) ** 0.5\n    if mag1 == 0 or mag2 == 0:\n        return 0.0\n    return dot / (mag1 * mag2)"},
    {"prompt": 'def cross_product(v1: list, v2: list) -> list:\n    """Cross product of two 3D vectors.\n    >>> cross_product([1, 0, 0], [0, 1, 0])\n    [0, 0, 1]\n    """', "completion": "    return [\n        v1[1] * v2[2] - v1[2] * v2[1],\n        v1[2] * v2[0] - v1[0] * v2[2],\n        v1[0] * v2[1] - v1[1] * v2[0],\n    ]"},
    {"prompt": 'def pythagorean(a: float, b: float) -> float:\n    """Hypotenuse of a right triangle.\n    >>> pythagorean(3, 4)\n    5.0\n    """', "completion": "    return (a ** 2 + b ** 2) ** 0.5"},
    {"prompt": 'def is_right_triangle(a: float, b: float, c: float) -> bool:\n    """Check if sides form a right triangle.\n    >>> is_right_triangle(3, 4, 5)\n    True\n    """', "completion": "    sides = sorted([a, b, c])\n    return abs(sides[0] ** 2 + sides[1] ** 2 - sides[2] ** 2) < 1e-9"},
    {"prompt": 'def sphere_volume(radius: float) -> float:\n    """Volume of a sphere.\n    >>> sphere_volume(1)\n    4.1887902047863905\n    """', "completion": "    import math\n    return (4 / 3) * math.pi * radius ** 3"},
    {"prompt": 'def linear_regression(x: list, y: list) -> tuple:\n    """Simple linear regression. Returns (slope, intercept).\n    >>> linear_regression([1, 2, 3], [2, 4, 6])\n    (2.0, 0.0)\n    """', "completion": "    n = len(x)\n    sum_x = sum(x)\n    sum_y = sum(y)\n    sum_xy = sum(xi * yi for xi, yi in zip(x, y))\n    sum_x2 = sum(xi ** 2 for xi in x)\n    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)\n    intercept = (sum_y - slope * sum_x) / n\n    return (slope, intercept)"},
    {"prompt": 'def correlation(x: list, y: list) -> float:\n    """Pearson correlation coefficient.\n    >>> correlation([1, 2, 3, 4], [2, 4, 6, 8])\n    1.0\n    """', "completion": "    n = len(x)\n    mean_x = sum(x) / n\n    mean_y = sum(y) / n\n    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))\n    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5\n    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5\n    if denom_x == 0 or denom_y == 0:\n        return 0.0\n    return numerator / (denom_x * denom_y)"},
    {"prompt": 'def percentile(arr: list, p: float) -> float:\n    """Calculate the p-th percentile.\n    >>> percentile([1, 2, 3, 4, 5], 50)\n    3\n    """', "completion": "    sorted_arr = sorted(arr)\n    n = len(sorted_arr)\n    k = (n - 1) * p / 100\n    f = int(k)\n    c = f + 1 if f + 1 < n else f\n    return sorted_arr[f] + (sorted_arr[c] - sorted_arr[f]) * (k - f)"},
    {"prompt": 'def z_score(x: float, arr: list) -> float:\n    """Z-score of a value relative to a list.\n    >>> z_score(5, [1, 2, 3, 4, 5])\n    1.4142135623730951\n    """', "completion": "    mean = sum(arr) / len(arr)\n    std = (sum((xi - mean) ** 2 for xi in arr) / len(arr)) ** 0.5\n    if std == 0:\n        return 0.0\n    return (x - mean) / std"},
    {"prompt": 'def entropy(arr: list) -> float:\n    """Shannon entropy.\n    >>> entropy([1, 1, 2, 2])\n    1.0\n    """', "completion": "    import math\n    n = len(arr)\n    if n == 0:\n        return 0.0\n    counts = {}\n    for x in arr:\n        counts[x] = counts.get(x, 0) + 1\n    return -sum((c / n) * math.log2(c / n) for c in counts.values())"},
    {"prompt": 'def unique_values(arr: list) -> list:\n    """Return unique values in order of first appearance.\n    >>> unique_values([3, 1, 2, 3, 1, 4])\n    [3, 1, 2, 4]\n    """', "completion": "    seen = set()\n    result = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"},
    {"prompt": 'def value_counts(arr: list) -> dict:\n    """Count occurrences of each value.\n    >>> value_counts(["a", "b", "a", "c", "b", "a"])\n    {"a": 3, "b": 2, "c": 1}\n    """', "completion": "    counts = {}\n    for x in arr:\n        counts[x] = counts.get(x, 0) + 1\n    return counts"},
    {"prompt": 'def sign(n: int) -> int:\n    """Return sign of a number: -1, 0, or 1.\n    >>> sign(-5)\n    -1\n    >>> sign(0)\n    0\n    >>> sign(5)\n    1\n    """', "completion": "    if n < 0:\n        return -1\n    elif n > 0:\n        return 1\n    return 0"},
    {"prompt": 'def lerp(a: float, b: float, t: float) -> float:\n    """Linear interpolation between a and b by t.\n    >>> lerp(0.0, 10.0, 0.5)\n    5.0\n    """', "completion": "    return a + (b - a) * t"},
    {"prompt": 'def midpoint(p1: tuple, p2: tuple) -> tuple:\n    """Find the midpoint between two 2D points.\n    >>> midpoint((0, 0), (4, 6))\n    (2.0, 3.0)\n    """', "completion": "    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)"},
    {"prompt": 'def triangle_area(p1: tuple, p2: tuple, p3: tuple) -> float:\n    """Area of a triangle given three vertices.\n    >>> triangle_area((0, 0), (4, 0), (0, 3))\n    6.0\n    """', "completion": "    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2)"},
    {"prompt": 'def hex_to_decimal(s: str) -> int:\n    """Convert hex string to decimal.\n    >>> hex_to_decimal("A")\n    10\n    >>> hex_to_decimal("FF")\n    255\n    """', "completion": "    return int(s, 16)"},
    {"prompt": 'def decimal_to_hex(n: int) -> str:\n    """Convert decimal to hex string.\n    >>> decimal_to_hex(255)\n    "FF"\n    >>> decimal_to_hex(10)\n    "A"\n    """', "completion": "    if n == 0:\n        return '0'\n    hex_chars = '0123456789ABCDEF'\n    result = []\n    while n > 0:\n        result.append(hex_chars[n % 16])\n        n //= 16\n    return ''.join(reversed(result))"},
    {"prompt": 'def is_valid_email(email: str) -> bool:\n    """Check if a string is a valid email (simplified).\n    >>> is_valid_email("user@example.com")\n    True\n    >>> is_valid_email("invalid")\n    False\n    """', "completion": "    import re\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))"},
    {"prompt": 'def extract_domain(email: str) -> str:\n    """Extract domain from email.\n    >>> extract_domain("user@example.com")\n    "example.com"\n    """', "completion": "    return email.split('@')[1] if '@' in email else ''"},
    {"prompt": 'def rotate_matrix_90(m: list) -> list:\n    """Rotate matrix 90 degrees clockwise.\n    >>> rotate_matrix_90([[1, 2], [3, 4]])\n    [[3, 1], [4, 2]]\n    """', "completion": "    n = len(m)\n    return [[m[n - 1 - j][i] for j in range(n)] for i in range(n)]"},
    {"prompt": 'def matrix_determinant(m: list) -> float:\n    """Determinant of a 2x2 matrix.\n    >>> matrix_determinant([[1, 2], [3, 4]])\n    -2.0\n    """', "completion": "    return m[0][0] * m[1][1] - m[0][1] * m[1][0]"},
    {"prompt": 'def matrix_trace(m: list) -> float:\n    """Trace of a square matrix.\n    >>> matrix_trace([[1, 2], [3, 4]])\n    5\n    """', "completion": "    return sum(m[i][i] for i in range(len(m)))"},
    {"prompt": 'def is_symmetric_matrix(m: list) -> bool:\n    """Check if a matrix is symmetric.\n    >>> is_symmetric_matrix([[1, 2], [2, 1]])\n    True\n    """', "completion": "    n = len(m)\n    for i in range(n):\n        for j in range(i + 1, n):\n            if m[i][j] != m[j][i]:\n                return False\n    return True"},
    {"prompt": 'def diagonal_elements(m: list) -> list:\n    """Extract diagonal elements.\n    >>> diagonal_elements([[1, 2], [3, 4]])\n    [1, 4]\n    """', "completion": "    return [m[i][i] for i in range(len(m))]"},
    {"prompt": 'def flatten_matrix(m: list) -> list:\n    """Flatten a 2D matrix into 1D.\n    >>> flatten_matrix([[1, 2], [3, 4]])\n    [1, 2, 3, 4]\n    """', "completion": "    return [x for row in m for x in row]"},
    {"prompt": 'def date_diff(y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> int:\n    """Days between two dates.\n    >>> date_diff(2024, 1, 1, 2024, 1, 31)\n    30\n    """', "completion": "    import datetime\n    date1 = datetime.date(y1, m1, d1)\n    date2 = datetime.date(y2, m2, d2)\n    return abs((date2 - date1).days)"},
    {"prompt": 'def pluralize(s: str) -> str:\n    """Simple English pluralization.\n    >>> pluralize("cat")\n    "cats"\n    >>> pluralize("box")\n    "boxes"\n    >>> pluralize("baby")\n    "babies"\n    """', "completion": "    if s.endswith('y') and s[-2] not in 'aeiou':\n        return s[:-1] + 'ies'\n    elif s.endswith(('s', 'x', 'z', 'ch', 'sh')):\n        return s + 'es'\n    return s + 's'"},
    {"prompt": 'def snake_to_camel(s: str) -> str:\n    """Convert snake_case to camelCase.\n    >>> snake_to_camel("hello_world")\n    "helloWorld"\n    """', "completion": "    components = s.split('_')\n    return components[0] + ''.join(x.title() for x in components[1:])"},
    {"prompt": 'def valid_ip(s: str) -> bool:\n    """Check if a string is a valid IPv4 address.\n    >>> valid_ip("192.168.1.1")\n    True\n    >>> valid_ip("256.1.1.1")\n    False\n    """', "completion": "    parts = s.split('.')\n    if len(parts) != 4:\n        return False\n    for part in parts:\n        if not part.isdigit():\n            return False\n        num = int(part)\n        if num < 0 or num > 255:\n            return False\n        if len(part) > 1 and part[0] == '0':\n            return False\n    return True"},
    {"prompt": 'def deep_copy_list(arr: list) -> list:\n    """Deep copy a nested list.\n    >>> deep_copy_list([1, [2, 3], [4, [5]]])\n    [1, [2, 3], [4, [5]]]\n    """', "completion": "    if isinstance(arr, list):\n        return [deep_copy_list(item) for item in arr]\n    return arr"},
    {"prompt": 'def is_weekend(year: int, month: int, day: int) -> bool:\n    """Check if a date falls on a weekend.\n    >>> is_weekend(2024, 1, 6)\n    True\n    >>> is_weekend(2024, 1, 1)\n    False\n    """', "completion": "    import datetime\n    date = datetime.date(year, month, day)\n    return date.weekday() >= 5"},
    {"prompt": 'def hms_to_seconds(hms: str) -> int:\n    """Convert HH:MM:SS to seconds.\n    >>> hms_to_seconds("01:01:01")\n    3661\n    """', "completion": "    h, m, s = map(int, hms.split(':'))\n    return h * 3600 + m * 60 + s"},
    {"prompt": 'def mask_email(email: str) -> str:\n    """Mask email, showing first char and domain.\n    >>> mask_email("john@example.com")\n    "j***@example.com"\n    """', "completion": "    if '@' not in email:\n        return email\n    name, domain = email.split('@', 1)\n    if not name:\n        return email\n    return name[0] + '***@' + domain"},
    {"prompt": 'def is_valid_uuid(s: str) -> bool:\n    """Check if a string is a valid UUID.\n    >>> is_valid_uuid("550e8400-e29b-41d4-a716-446655440000")\n    True\n    >>> is_valid_uuid("invalid")\n    False\n    """', "completion": "    import re\n    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'\n    return bool(re.match(pattern, s, re.IGNORECASE))"},
    {"prompt": 'def std_dev(arr: list) -> float:\n    """Standard deviation.\n    >>> std_dev([1, 2, 3, 4, 5])\n    1.4142135623730951\n    """', "completion": "    if len(arr) < 2:\n        return 0.0\n    avg = sum(arr) / len(arr)\n    variance = sum((x - avg) ** 2 for x in arr) / len(arr)\n    return variance ** 0.5"},
    {"prompt": 'def iqr(arr: list) -> float:\n    """Interquartile range.\n    >>> iqr([1, 2, 3, 4, 5, 6, 7, 8])\n    4.0\n    """', "completion": "    sorted_arr = sorted(arr)\n    n = len(sorted_arr)\n    q1 = sorted_arr[n // 4]\n    q3 = sorted_arr[3 * n // 4]\n    return q3 - q1"},
    {"prompt": 'def outliers(arr: list) -> list:\n    """Find outliers using 1.5*IQR rule.\n    >>> outliers([1, 2, 3, 4, 5, 100])\n    [100]\n    """', "completion": "    sorted_arr = sorted(arr)\n    n = len(sorted_arr)\n    q1 = sorted_arr[n // 4]\n    q3 = sorted_arr[3 * n // 4]\n    iqr_val = q3 - q1\n    lower = q1 - 1.5 * iqr_val\n    upper = q3 + 1.5 * iqr_val\n    return [x for x in arr if x < lower or x > upper]"},
    {"prompt": 'def gini_impurity(arr: list) -> float:\n    """Gini impurity.\n    >>> gini_impurity([1, 1, 2, 2])\n    0.5\n    """', "completion": "    n = len(arr)\n    if n == 0:\n        return 0.0\n    counts = {}\n    for x in arr:\n        counts[x] = counts.get(x, 0) + 1\n    return 1 - sum((c / n) ** 2 for c in counts.values())"},
    {"prompt": 'def precision(tp: int, fp: int) -> float:\n    """Precision metric.\n    >>> precision(80, 20)\n    0.8\n    """', "completion": "    if tp + fp == 0:\n        return 0.0\n    return tp / (tp + fp)"},
    {"prompt": 'def recall(tp: int, fn: int) -> float:\n    """Recall metric.\n    >>> recall(80, 20)\n    0.8\n    """', "completion": "    if tp + fn == 0:\n        return 0.0\n    return tp / (tp + fn)"},
    {"prompt": 'def f1_score(tp: int, fp: int, fn: int) -> float:\n    """F1 score.\n    >>> f1_score(80, 20, 20)\n    0.8\n    """', "completion": "    p = precision(tp, fp)\n    r = recall(tp, fn)\n    if p + r == 0:\n        return 0.0\n    return 2 * p * r / (p + r)"},
    {"prompt": 'def exponential_smoothing(arr: list, alpha: float) -> list:\n    """Exponential smoothing.\n    >>> exponential_smoothing([1, 2, 3, 4, 5], 0.5)\n    [1, 1.5, 2.25, 3.125, 4.0625]\n    """', "completion": "    if not arr:\n        return arr\n    result = [arr[0]]\n    for i in range(1, len(arr)):\n        result.append(alpha * arr[i] + (1 - alpha) * result[-1])\n    return result"},
    {"prompt": 'def diff(arr: list) -> list:\n    """Differences between consecutive elements.\n    >>> diff([1, 3, 6, 10])\n    [2, 3, 4]\n    """', "completion": "    return [arr[i+1] - arr[i] for i in range(len(arr) - 1)]"},
    {"prompt": 'def cumulative_max(arr: list) -> list:\n    """Cumulative maximum.\n    >>> cumulative_max([1, 3, 2, 5, 4])\n    [1, 3, 3, 5, 5]\n    """', "completion": "    result = []\n    max_val = float('-inf')\n    for x in arr:\n        max_val = max(max_val, x)\n        result.append(max_val)\n    return result"},
    {"prompt": 'def standardize(arr: list) -> list:\n    """Standardize to mean=0, std=1.\n    >>> standardize([1, 2, 3, 4, 5])\n    [-1.4142135623730951, -0.7071067811865475, 0.0, 0.7071067811865475, 1.4142135623730951]\n    """', "completion": "    mean = sum(arr) / len(arr)\n    std = (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5\n    if std == 0:\n        return [0.0] * len(arr)\n    return [(x - mean) / std for x in arr]"},
    {"prompt": 'def bin_data(arr: list, num_bins: int) -> list:\n    """Bin data into equal-width bins.\n    >>> bin_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)\n    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]\n    """', "completion": "    if not arr:\n        return []\n    min_val = min(arr)\n    max_val = max(arr)\n    bin_width = (max_val - min_val) / num_bins\n    if bin_width == 0:\n        return [arr[:]]\n    bins = [[] for _ in range(num_bins)]\n    for x in arr:\n        idx = min(int((x - min_val) / bin_width), num_bins - 1)\n        bins[idx].append(x)\n    return bins"},
    {"prompt": 'def mode(arr: list):\n    """Return the most common element.\n    >>> mode([1, 2, 2, 3, 3, 3])\n    3\n    """', "completion": "    from collections import Counter\n    return Counter(arr).most_common(1)[0][0]"},
    {"prompt": 'def most_frequent(arr: list):\n    """Return the most frequent element.\n    >>> most_frequent([1, 2, 2, 3, 3, 3])\n    3\n    """', "completion": "    counts = {}\n    for x in arr:\n        counts[x] = counts.get(x, 0) + 1\n    return max(counts, key=counts.get)"},
    {"prompt": 'def count_negatives(arr: list) -> int:\n    """Count negative numbers.\n    >>> count_negatives([1, -2, 3, -4, 5])\n    2\n    """', "completion": "    return sum(1 for x in arr if x < 0)"},
    {"prompt": 'def abs_sum(arr: list) -> int:\n    """Sum of absolute values.\n    >>> abs_sum([1, -2, 3, -4])\n    10\n    """', "completion": "    return sum(abs(x) for x in arr)"},
    {"prompt": 'def sum_positives(arr: list) -> int:\n    """Sum of positive numbers.\n    >>> sum_positives([1, -2, 3, -4, 5])\n    9\n    """', "completion": "    return sum(x for x in arr if x > 0)"},
    {"prompt": 'def find_max(arr: list) -> int:\n    """Find the maximum element.\n    >>> find_max([1, 3, 2, 5, 4])\n    5\n    """', "completion": "    if not arr:\n        raise ValueError(\"empty list\")\n    max_val = arr[0]\n    for x in arr[1:]:\n        if x > max_val:\n            max_val = x\n    return max_val"},
    {"prompt": 'def find_min(arr: list) -> int:\n    """Find the minimum element.\n    >>> find_min([1, 3, 2, 5, 4])\n    1\n    """', "completion": "    if not arr:\n        raise ValueError(\"empty list\")\n    min_val = arr[0]\n    for x in arr[1:]:\n        if x < min_val:\n            min_val = x\n    return min_val"},
    {"prompt": 'def any_positive(arr: list) -> bool:\n    """Check if any element is positive.\n    >>> any_positive([-1, -2, 3])\n    True\n    """', "completion": "    return any(x > 0 for x in arr)"},
    {"prompt": 'def all_positive(arr: list) -> bool:\n    """Check if all elements are positive.\n    >>> all_positive([1, 2, 3])\n    True\n    """', "completion": "    return all(x > 0 for x in arr)"},
    {"prompt": 'def max_product(arr: list) -> int:\n    """Maximum product of two elements.\n    >>> max_product([1, 5, 3, 4, 2])\n    20\n    """', "completion": "    arr_sorted = sorted(arr, reverse=True)\n    return arr_sorted[0] * arr_sorted[1]"},
    {"prompt": 'def filter_even(arr: list) -> list:\n    """Filter even numbers.\n    >>> filter_even([1, 2, 3, 4, 5, 6])\n    [2, 4, 6]\n    """', "completion": "    return [x for x in arr if x % 2 == 0]"},
    {"prompt": 'def map_double(arr: list) -> list:\n    """Double each element.\n    >>> map_double([1, 2, 3])\n    [2, 4, 6]\n    """', "completion": "    return [x * 2 for x in arr]"},
    {"prompt": 'def reduce_sum(arr: list) -> int:\n    """Sum all elements.\n    >>> reduce_sum([1, 2, 3, 4, 5])\n    15\n    """', "completion": "    total = 0\n    for x in arr:\n        total += x\n    return total"},
    {"prompt": 'def is_even(n: int) -> bool:\n    """Check if a number is even.\n    >>> is_even(4)\n    True\n    >>> is_even(7)\n    False\n    """', "completion": "    return n % 2 == 0"},
    {"prompt": 'def is_square_matrix(m: list) -> bool:\n    """Check if a matrix is square.\n    >>> is_square_matrix([[1, 2], [3, 4]])\n    True\n    """', "completion": "    return len(m) == len(m[0]) if m else False"},
    {"prompt": 'def matrix_shape(m: list) -> tuple:\n    """Return shape (rows, cols).\n    >>> matrix_shape([[1, 2, 3], [4, 5, 6]])\n    (2, 3)\n    """', "completion": "    if not m:\n        return (0, 0)\n    return (len(m), len(m[0]))"},
    {"prompt": 'def zero_matrix(rows: int, cols: int) -> list:\n    """Create a zero matrix.\n    >>> zero_matrix(2, 3)\n    [[0, 0, 0], [0, 0, 0]]\n    """', "completion": "    return [[0] * cols for _ in range(rows)]"},
    {"prompt": 'def is_identity_matrix(m: list) -> bool:\n    """Check if a matrix is an identity matrix.\n    >>> is_identity_matrix([[1, 0], [0, 1]])\n    True\n    """', "completion": "    n = len(m)\n    for i in range(n):\n        for j in range(n):\n            if m[i][j] != (1 if i == j else 0):\n                return False\n    return True"},
    {"prompt": 'def flip_horizontal(m: list) -> list:\n    """Flip a matrix horizontally.\n    >>> flip_horizontal([[1, 2], [3, 4]])\n    [[2, 1], [4, 3]]\n    """', "completion": "    return [row[::-1] for row in m]"},
    {"prompt": 'def flip_vertical(m: list) -> list:\n    """Flip a matrix vertically.\n    >>> flip_vertical([[1, 2], [3, 4]])\n    [[3, 4], [1, 2]]\n    """', "completion": "    return m[::-1]"},
    {"prompt": 'def submatrix(m: list, row: int, col: int) -> list:\n    """Get submatrix by removing row and col.\n    >>> submatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 0, 0)\n    [[5, 6], [8, 9]]\n    """', "completion": "    return [[m[i][j] for j in range(len(m[0])) if j != col] for i in range(len(m)) if i != row]"},
    {"prompt": 'def anti_diagonal(m: list) -> list:\n    """Extract anti-diagonal elements.\n    >>> anti_diagonal([[1, 2], [3, 4]])\n    [2, 3]\n    """', "completion": "    n = len(m)\n    return [m[i][n - 1 - i] for i in range(n)]"},
    {"prompt": 'def swap_rows(m: list, r1: int, r2: int) -> list:\n    """Swap two rows.\n    >>> swap_rows([[1, 2], [3, 4]], 0, 1)\n    [[3, 4], [1, 2]]\n    """', "completion": "    m = [row[:] for row in m]\n    m[r1], m[r2] = m[r2], m[r1]\n    return m"},
    {"prompt": 'def is_upper_triangular(m: list) -> bool:\n    """Check if a matrix is upper triangular.\n    >>> is_upper_triangular([[1, 2, 3], [0, 4, 5], [0, 0, 6]])\n    True\n    """', "completion": "    n = len(m)\n    for i in range(1, n):\n        for j in range(i):\n            if m[i][j] != 0:\n                return False\n    return True"},
    {"prompt": 'def is_lower_triangular(m: list) -> bool:\n    """Check if a matrix is lower triangular.\n    >>> is_lower_triangular([[1, 0, 0], [2, 3, 0], [4, 5, 6]])\n    True\n    """', "completion": "    n = len(m)\n    for i in range(n):\n        for j in range(i + 1, n):\n            if m[i][j] != 0:\n                return False\n    return True"},
    {"prompt": 'def cone_volume(radius: float, height: float) -> float:\n    """Volume of a cone.\n    >>> cone_volume(1, 3)\n    3.141592653589793\n    """', "completion": "    import math\n    return (1 / 3) * math.pi * radius ** 2 * height"},
    {"prompt": 'def cylinder_volume(radius: float, height: float) -> float:\n    """Volume of a cylinder.\n    >>> cylinder_volume(1, 2)\n    6.283185307179586\n    """', "completion": "    import math\n    return math.pi * radius ** 2 * height"},
    {"prompt": 'def cube_volume(side: float) -> float:\n    """Volume of a cube.\n    >>> cube_volume(3)\n    27\n    """', "completion": "    return side ** 3"},
    {"prompt": 'def cube_surface_area(side: float) -> float:\n    """Surface area of a cube.\n    >>> cube_surface_area(3)\n    54\n    """', "completion": "    return 6 * side ** 2"},
    {"prompt": 'def rectangle_area(width: float, height: float) -> float:\n    """Area of a rectangle.\n    >>> rectangle_area(3, 4)\n    12\n    """', "completion": "    return width * height"},
    {"prompt": 'def rectangle_perimeter(width: float, height: float) -> float:\n    """Perimeter of a rectangle.\n    >>> rectangle_perimeter(3, 4)\n    14\n    """', "completion": "    return 2 * (width + height)"},
    {"prompt": 'def is_equilateral(a: float, b: float, c: float) -> bool:\n    """Check if triangle is equilateral.\n    >>> is_equilateral(3, 3, 3)\n    True\n    """', "completion": "    return a == b == c"},
    {"prompt": 'def is_isosceles(a: float, b: float, c: float) -> bool:\n    """Check if triangle is isosceles.\n    >>> is_isosceles(3, 3, 4)\n    True\n    """', "completion": "    return a == b or b == c or a == c"},
    {"prompt": 'def is_scalene(a: float, b: float, c: float) -> bool:\n    """Check if triangle is scalene.\n    >>> is_scalene(3, 4, 5)\n    True\n    """', "completion": "    return a != b and b != c and a != c"},
    {"prompt": 'def slope(p1: tuple, p2: tuple) -> float:\n    """Slope of a line between two points.\n    >>> slope((0, 0), (2, 4))\n    2.0\n    """', "completion": "    if p2[0] - p1[0] == 0:\n        raise ValueError(\"undefined slope\")\n    return (p2[1] - p1[1]) / (p2[0] - p1[0])"},
    {"prompt": 'def perpendicular_slope(m: float) -> float:\n    """Slope of perpendicular line.\n    >>> perpendicular_slope(2)\n    -0.5\n    """', "completion": "    if m == 0:\n        raise ValueError(\"undefined (vertical line)\")\n    return -1 / m"},
    {"prompt": 'def is_parallel(m1: float, m2: float) -> bool:\n    """Check if two lines are parallel.\n    >>> is_parallel(2, 2)\n    True\n    """', "completion": "    return m1 == m2"},
    {"prompt": 'def is_perpendicular(m1: float, m2: float) -> bool:\n    """Check if two lines are perpendicular.\n    >>> is_perpendicular(2, -0.5)\n    True\n    """', "completion": "    return m1 * m2 == -1"},
    {"prompt": 'def vector_add(v1: list, v2: list) -> list:\n    """Add two vectors.\n    >>> vector_add([1, 2, 3], [4, 5, 6])\n    [5, 7, 9]\n    """', "completion": "    return [a + b for a, b in zip(v1, v2)]"},
    {"prompt": 'def vector_subtract(v1: list, v2: list) -> list:\n    """Subtract v2 from v1.\n    >>> vector_subtract([5, 7, 9], [1, 2, 3])\n    [4, 5, 6]\n    """', "completion": "    return [a - b for a, b in zip(v1, v2)]"},
    {"prompt": 'def scalar_multiply(v: list, s: float) -> list:\n    """Multiply vector by scalar.\n    >>> scalar_multiply([1, 2, 3], 2)\n    [2, 4, 6]\n    """', "completion": "    return [x * s for x in v]"},
    {"prompt": 'def normalize_vector(v: list) -> list:\n    """Normalize vector to unit length.\n    >>> normalize_vector([3, 4])\n    [0.6, 0.8]\n    """', "completion": "    mag = sum(x ** 2 for x in v) ** 0.5\n    if mag == 0:\n        return v\n    return [x / mag for x in v]"},
    {"prompt": 'def matrix_add(m1: list, m2: list) -> list:\n    """Add two matrices.\n    >>> matrix_add([[1, 2], [3, 4]], [[5, 6], [7, 8]])\n    [[6, 8], [10, 12]]\n    """', "completion": "    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]"},
    {"prompt": 'def matrix_scalar_multiply(m: list, s: float) -> list:\n    """Multiply matrix by scalar.\n    >>> matrix_scalar_multiply([[1, 2], [3, 4]], 2)\n    [[2, 4], [6, 8]]\n    """', "completion": "    return [[x * s for x in row] for row in m]"},
    {"prompt": 'def dot_product_2d(v1: tuple, v2: tuple) -> float:\n    """2D dot product.\n    >>> dot_product_2d((1, 2), (3, 4))\n    11\n    """', "completion": "    return v1[0] * v2[0] + v1[1] * v2[1]"},
    {"prompt": 'def distance_2d(p1: tuple, p2: tuple) -> float:\n    """Euclidean distance between two 2D points.\n    >>> distance_2d((0, 0), (3, 4))\n    5.0\n    """', "completion": "    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5"},
    {"prompt": 'def line_equation(p1: tuple, p2: tuple) -> tuple:\n    """Return (slope, intercept) of line through two points.\n    >>> line_equation((0, 1), (2, 5))\n    (2.0, 1.0)\n    """', "completion": "    m = (p2[1] - p1[1]) / (p2[0] - p1[0]) if (p2[0] - p1[0]) != 0 else None\n    b = p1[1] - m * p1[0] if m is not None else None\n    return (m, b)"},
    {"prompt": 'def angle_between(v1: list, v2: list) -> float:\n    """Angle between two vectors in radians.\n    >>> angle_between([1, 0], [0, 1])\n    1.5707963267948966\n    """', "completion": "    import math\n    dot = sum(a * b for a, b in zip(v1, v2))\n    mag1 = sum(a * a for a in v1) ** 0.5\n    mag2 = sum(b * b for b in v2) ** 0.5\n    if mag1 == 0 or mag2 == 0:\n        return 0.0\n    cos = max(-1, min(1, dot / (mag1 * mag2)))\n    return math.acos(cos)"},
    {"prompt": 'def circle_circumference(radius: float) -> float:\n    """Circumference of a circle.\n    >>> circle_circumference(1)\n    6.283185307179586\n    """', "completion": "    import math\n    return 2 * math.pi * radius"},
    {"prompt": 'def sphere_surface_area(radius: float) -> float:\n    """Surface area of a sphere.\n    >>> sphere_surface_area(1)\n    12.566370614359172\n    """', "completion": "    import math\n    return 4 * math.pi * radius ** 2"},
    {"prompt": 'def cylinder_surface_area(radius: float, height: float) -> float:\n    """Surface area of a cylinder.\n    >>> cylinder_surface_area(1, 1)\n    12.566370614359172\n    """', "completion": "    import math\n    return 2 * math.pi * radius * (radius + height)"},
    {"prompt": 'def triangular_number(n: int) -> int:\n    """Return the n-th triangular number.\n    >>> triangular_number(4)\n    10\n    """', "completion": "    return n * (n + 1) // 2"},
    {"prompt": 'def sum_of_cubes(n: int) -> int:\n    """Sum of cubes from 1 to n.\n    >>> sum_of_cubes(3)\n    36\n    """', "completion": "    return (n * (n + 1) // 2) ** 2"},
    {"prompt": 'def is_triangular(n: int) -> bool:\n    """Check if n is a triangular number.\n    >>> is_triangular(10)\n    True\n    >>> is_triangular(11)\n    False\n    """', "completion": "    if n < 0:\n        return False\n    discriminant = 1 + 8 * n\n    root = int(discriminant ** 0.5)\n    return root * root == discriminant and (root - 1) % 2 == 0"},
    {"prompt": 'def lag(arr: list, n: int) -> list:\n    """Shift array by n positions.\n    >>> lag([1, 2, 3, 4, 5], 2)\n    [None, None, 1, 2, 3]\n    """', "completion": "    return [None] * n + arr[:-n] if n > 0 else arr"},
    {"prompt": 'def lead(arr: list, n: int) -> list:\n    """Shift array forward by n positions.\n    >>> lead([1, 2, 3, 4, 5], 2)\n    [3, 4, 5, None, None]\n    """', "completion": "    return arr[n:] + [None] * n if n > 0 else arr"},
    {"prompt": 'def cumulative_product(arr: list) -> list:\n    """Cumulative product.\n    >>> cumulative_product([1, 2, 3, 4])\n    [1, 2, 6, 24]\n    """', "completion": "    result = []\n    total = 1\n    for x in arr:\n        total *= x\n        result.append(total)\n    return result"},
    {"prompt": 'def cumulative_min(arr: list) -> list:\n    """Cumulative minimum.\n    >>> cumulative_min([5, 3, 4, 1, 2])\n    [5, 3, 3, 1, 1]\n    """', "completion": "    result = []\n    min_val = float('inf')\n    for x in arr:\n        min_val = min(min_val, x)\n        result.append(min_val)\n    return result"},
    {"prompt": 'def rate_of_change(arr: list) -> list:\n    """Percentage rate of change.\n    >>> rate_of_change([100, 110, 121])\n    [0.1, 0.1]\n    """', "completion": "    return [(arr[i+1] - arr[i]) / arr[i] for i in range(len(arr) - 1) if arr[i] != 0]"},
    {"prompt": 'def resample_mean(arr: list, factor: int) -> list:\n    """Resample by taking mean of every n elements.\n    >>> resample_mean([1, 2, 3, 4, 5, 6], 2)\n    [1.5, 3.5, 5.5]\n    """', "completion": "    return [sum(arr[i:i+factor]) / factor for i in range(0, len(arr) - len(arr) % factor, factor)]"},
    {"prompt": 'def histogram(arr: list, num_bins: int) -> list:\n    """Count elements in each bin.\n    >>> histogram([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)\n    [5, 5]\n    """', "completion": "    if not arr:\n        return []\n    min_val = min(arr)\n    max_val = max(arr)\n    bin_width = (max_val - min_val) / num_bins\n    if bin_width == 0:\n        return [len(arr)]\n    counts = [0] * num_bins\n    for x in arr:\n        idx = min(int((x - min_val) / bin_width), num_bins - 1)\n        counts[idx] += 1\n    return counts"},
    {"prompt": 'def relative_frequency(arr: list) -> dict:\n    """Relative frequency of each value.\n    >>> relative_frequency([1, 1, 2, 3])\n    {1: 0.5, 2: 0.25, 3: 0.25}\n    """', "completion": "    n = len(arr)\n    counts = {}\n    for x in arr:\n        counts[x] = counts.get(x, 0) + 1\n    return {k: v / n for k, v in counts.items()}"},
    {"prompt": 'def frequency_table(arr: list) -> list:\n    """Return sorted (value, count) pairs.\n    >>> frequency_table([3, 1, 2, 3, 1])\n    [(1, 2), (2, 1), (3, 2)]\n    """', "completion": "    counts = {}\n    for x in arr:\n        counts[x] = counts.get(x, 0) + 1\n    return sorted(counts.items())"},
    {"prompt": 'def cumulative_frequency(arr: list) -> list:\n    """Cumulative frequency distribution.\n    >>> cumulative_frequency([1, 1, 2, 2, 2, 3])\n    [(1, 2), (2, 5), (3, 6)]\n    """', "completion": "    counts = {}\n    for x in arr:\n        counts[x] = counts.get(x, 0) + 1\n    sorted_items = sorted(counts.items())\n    result = []\n    total = 0\n    for val, count in sorted_items:\n        total += count\n        result.append((val, total))\n    return result"},
    {"prompt": 'def information_gain(parent: list, children: list) -> float:\n    """Information gain from splitting.\n    >>> information_gain([1, 1, 2, 2], [[1, 1], [2, 2]])\n    1.0\n    """', "completion": "    def ent(lst):\n        import math\n        n = len(lst)\n        if n == 0:\n            return 0.0\n        c = {}\n        for x in lst:\n            c[x] = c.get(x, 0) + 1\n        return -sum((v / n) * math.log2(v / n) for v in c.values())\n    parent_e = ent(parent)\n    n = len(parent)\n    weighted = sum((len(child) / n) * ent(child) for child in children)\n    return parent_e - weighted"},
    {"prompt": 'def quartiles(arr: list) -> tuple:\n    """Return (Q1, Q2, Q3).\n    >>> quartiles([1, 2, 3, 4, 5, 6, 7, 8])\n    (2.5, 4.5, 6.5)\n    """', "completion": "    s = sorted(arr)\n    n = len(s)\n    def pct(p):\n        k = (n - 1) * p / 100\n        f = int(k)\n        c = f + 1 if f + 1 < n else f\n        return s[f] + (s[c] - s[f]) * (k - f)\n    return (pct(25), pct(50), pct(75))"},
    {"prompt": 'def least_frequent(arr: list):\n    """Return the least frequent element.\n    >>> least_frequent([1, 2, 2, 3, 3, 3])\n    1\n    """', "completion": "    counts = {}\n    for x in arr:\n        counts[x] = counts.get(x, 0) + 1\n    return min(counts, key=counts.get)"},
    {"prompt": 'def second_smallest(arr: list) -> int:\n    """Find the second smallest element.\n    >>> second_smallest([1, 3, 2, 5, 4])\n    2\n    """', "completion": "    if len(arr) < 2:\n        raise ValueError(\"list too short\")\n    first = second = float('inf')\n    for x in arr:\n        if x < first:\n            second = first\n            first = x\n        elif x < second and x != first:\n            second = x\n    return second"},
    {"prompt": 'def count_occurrences(arr: list, target) -> int:\n    """Count occurrences of target.\n    >>> count_occurrences([1, 2, 2, 3, 2], 2)\n    3\n    """', "completion": "    return arr.count(target)"},
    {"prompt": 'def count_words(s: str) -> int:\n    """Count words in a string.\n    >>> count_words("hello world foo")\n    3\n    """', "completion": "    return len(s.split())"},
    {"prompt": 'def string_compression(s: str) -> str:\n    """Basic RLE compression.\n    >>> string_compression("aaabbbccc")\n    "a3b3c3"\n    """', "completion": "    if not s:\n        return s\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result.append(s[i-1] + str(count))\n            count = 1\n    result.append(s[-1] + str(count))\n    return ''.join(result)"},
    {"prompt": 'def url_encode(s: str) -> str:\n    """URL-encode a string.\n    >>> url_encode("hello world")\n    "hello%20world"\n    """', "completion": "    import urllib.parse\n    return urllib.parse.quote(s)"},
    {"prompt": 'def url_decode(s: str) -> str:\n    """Decode URL-encoded string.\n    >>> url_decode("hello%20world")\n    "hello world"\n    """', "completion": "    import urllib.parse\n    return urllib.parse.unquote(s)"},
    {"prompt": 'def parse_query_params(url: str) -> dict:\n    """Parse query parameters from URL.\n    >>> parse_query_params("http://example.com?a=1&b=2")\n    {"a": "1", "b": "2"}\n    """', "completion": "    import urllib.parse\n    parsed = urllib.parse.urlparse(url)\n    return dict(urllib.parse.parse_qsl(parsed.query))"},
    {"prompt": 'def build_url(base: str, params: dict) -> str:\n    """Build URL with query parameters.\n    >>> build_url("http://example.com", {"a": "1", "b": "2"})\n    "http://example.com?a=1&b=2"\n    """', "completion": "    import urllib.parse\n    query = urllib.parse.urlencode(params)\n    return f\"{base}?{query}\" if query else base"},
    {"prompt": 'def indent_text(s: str, spaces: int) -> str:\n    """Indent each line by given spaces.\n    >>> indent_text("hello\\\\nworld", 4)\n    "    hello\\\\n    world"\n    """', "completion": "    indent = ' ' * spaces\n    return '\\\\n'.join(indent + line for line in s.split('\\\\n'))"},
    {"prompt": 'def wrap_text(s: str, width: int) -> str:\n    """Wrap text to given width.\n    >>> wrap_text("hello world foo bar", 10)\n    "hello\\\\nworld foo\\\\nbar"\n    """', "completion": "    import textwrap\n    return textwrap.fill(s, width)"},
    {"prompt": 'def singularize(s: str) -> str:\n    """Simple English singularization.\n    >>> singularize("cats")\n    "cat"\n    >>> singularize("boxes")\n    "box"\n    """', "completion": "    if s.endswith('ies'):\n        return s[:-3] + 'y'\n    elif s.endswith('es'):\n        return s[:-2]\n    elif s.endswith('s'):\n        return s[:-1]\n    return s"},
    {"prompt": 'def day_of_week(year: int, month: int, day: int) -> str:\n    """Return the day of the week.\n    >>> day_of_week(2024, 1, 1)\n    "Monday"\n    """', "completion": "    import datetime\n    date = datetime.date(year, month, day)\n    return date.strftime('%A')"},
    {"prompt": 'def add_days(year: int, month: int, day: int, n: int) -> str:\n    """Add n days to a date.\n    >>> add_days(2024, 1, 1, 30)\n    "2024-01-31"\n    """', "completion": "    import datetime\n    date = datetime.date(year, month, day)\n    new_date = date + datetime.timedelta(days=n)\n    return new_date.strftime('%Y-%m-%d')"},
    {"prompt": 'def reshape_matrix(m: list, rows: int, cols: int) -> list:\n    """Reshape a matrix.\n    >>> reshape_matrix([[1, 2, 3, 4]], 2, 2)\n    [[1, 2], [3, 4]]\n    """', "completion": "    flat = [x for row in m for x in row]\n    if len(flat) != rows * cols:\n        raise ValueError(\"Cannot reshape\")\n    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]"},
    {"prompt": 'def rotate_matrix_180(m: list) -> list:\n    """Rotate matrix 180 degrees.\n    >>> rotate_matrix_180([[1, 2], [3, 4]])\n    [[4, 3], [2, 1]]\n    """', "completion": "    return [row[::-1] for row in m[::-1]]"},
    {"prompt": 'def rotate_matrix_270(m: list) -> list:\n    """Rotate matrix 90 degrees counter-clockwise.\n    >>> rotate_matrix_270([[1, 2], [3, 4]])\n    [[2, 4], [1, 3]]\n    """', "completion": "    n = len(m)\n    return [[m[j][n - 1 - i] for j in range(n)] for i in range(n)]"},
    {"prompt": 'def swap_columns(m: list, c1: int, c2: int) -> list:\n    """Swap two columns.\n    >>> swap_columns([[1, 2], [3, 4]], 0, 1)\n    [[2, 1], [4, 3]]\n    """', "completion": "    m = [row[:] for row in m]\n    for row in m:\n        row[c1], row[c2] = row[c2], row[c1]\n    return m"},
]

def main():
    print("=== Generating Coding SFT Dataset v2 ===\n", flush=True)
    all_samples = []
    all_samples.extend(load_mbpp_direct())
    all_samples.extend(load_mbpp_sanitized_direct())
    all_samples.extend(SYNTHETIC)
    print(f"Synthetic problems: {len(SYNTHETIC)}", flush=True)

    # Deduplicate
    seen = set()
    unique = []
    for s in all_samples:
        if s["prompt"] not in seen:
            seen.add(s["prompt"])
            unique.append(s)
    print(f"\nTotal unique: {len(unique)}", flush=True)

    random.seed(42)
    random.shuffle(unique)

    # Format: prompt + completion as raw text
    formatted = []
    for s in unique:
        text = s["prompt"] + "\n" + s["completion"]
        formatted.append({"text": text, "prompt": s["prompt"], "completion": s["completion"]})

    output_file = OUTPUT_DIR / "coding_sft_dataset.jsonl"
    with open(output_file, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(formatted)} samples to {output_file}", flush=True)

    # Show samples
    for i in range(min(3, len(formatted))):
        print(f"\n=== Sample {i} ===")
        print(formatted[i]["text"][:300])

    return len(formatted)

if __name__ == "__main__":
    main()
