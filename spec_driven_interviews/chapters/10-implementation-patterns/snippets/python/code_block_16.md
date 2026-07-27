```python
def transform_words(self, words: list[str]) -> list[str]:
    if not words:
        return []
    result = [""] * len(words)

    for i in range(len(words)):
        if len(words[i]) % 2 != 0:
            result[i] = words[i].upper()
        else:
            result[i] = words[i][::-1]

    return result
# Time: O(N * K) where K is average word length, Space: O(N * K) for output
```