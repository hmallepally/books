```python
def total_fruit(self, fruits: list[int]) -> int:
    from collections import defaultdict
    count = defaultdict(int)
    left = max_val = 0
    for right in range(len(fruits)):
        count[fruits[right]] += 1
        while len(count) > 2:
            count[fruits[left]] -= 1
            if count[fruits[left]] == 0:
                del count[fruits[left]]
            left += 1
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(1)
```