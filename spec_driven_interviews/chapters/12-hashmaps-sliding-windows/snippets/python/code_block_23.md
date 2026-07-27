```python
def top_k_frequent(self, nums: list[int], k: int) -> list[int]:
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
# Time Complexity: O(N log K) | Space Complexity: O(N)
```