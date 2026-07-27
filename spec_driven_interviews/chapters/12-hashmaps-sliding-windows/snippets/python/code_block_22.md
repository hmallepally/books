```python
def insert(self, intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
    res = []
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < new_interval[0]:
        res.append(intervals[i]) # Before
        i += 1
    while i < n and intervals[i][0] <= new_interval[1]: # Merge
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    res.append(new_interval)
    while i < n:
        res.append(intervals[i]) # After
        i += 1
    return res
# Time Complexity: O(N) | Space Complexity: O(N)
```