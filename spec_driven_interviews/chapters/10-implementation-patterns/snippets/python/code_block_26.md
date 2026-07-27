```python
def common_character_count(self, s1: str, s2: str) -> int:
    count1 = [0] * 26
    count2 = [0] * 26

    for c in s1: count1[ord(c) - ord('a')] += 1
    for c in s2: count2[ord(c) - ord('a')] += 1

    common = 0
    for i in range(26):
        common += min(count1[i], count2[i])

    return common
# Time: O(N + M), Space: O(1) — fixed 26-element lists
```