```python
def find_substring(self, s: str, words: list[str]) -> list[int]:
    res = []
    if not s or not words: return res
    word_len = len(words[0])
    total_len = word_len * len(words)
    
    from collections import Counter
    counts = Counter(words)
    
    for i in range(len(s) - total_len + 1):
        seen = {}
        j = 0
        while j < len(words):
            w = s[i + j * word_len : i + (j + 1) * word_len]
            if w in counts:
                seen[w] = seen.get(w, 0) + 1
                if seen[w] > counts[w]: break
            else:
                break
            j += 1
        if j == len(words): res.append(i)
    return res
# Time Complexity: O(N * M * L) | Space Complexity: O(M)
```