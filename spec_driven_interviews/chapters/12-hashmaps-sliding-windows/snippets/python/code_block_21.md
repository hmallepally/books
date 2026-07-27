```python
def least_interval(self, tasks: list[str], n: int) -> int:
    count = [0] * 26
    max_val = max_count = 0
    for c in tasks:
        idx = ord(c) - ord('A')
        count[idx] += 1
        if count[idx] == max_val:
            max_count += 1
        elif count[idx] > max_val:
            max_val = count[idx]
            max_count = 1
            
    empty_slots = (max_val - 1) * (n - (max_count - 1))
    available_tasks = len(tasks) - max_val * max_count
    idles = max(0, empty_slots - available_tasks)
    return len(tasks) + idles
# Time Complexity: O(N) | Space Complexity: O(1)
```