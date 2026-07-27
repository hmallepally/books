```python
def all_longest_strings(self, input_array: list[str]) -> list[str]:
    # Pass 1: Find the maximum length
    max_length = 0
    for s in input_array:
        if len(s) > max_length:
            max_length = len(s)

    # Pass 2: Collect strings matching the max length
    result = []
    for s in input_array:
        if len(s) == max_length:
            result.append(s)

    return result
# Time: O(N), Space: O(N) for output
```