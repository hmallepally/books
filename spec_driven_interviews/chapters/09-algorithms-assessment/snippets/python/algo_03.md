```python
from collections import defaultdict

def subarray_sum(nums: list[int], k: int) -> int:
    pref_counts = defaultdict(int)
    pref_counts[0] = 1
    current_sum = 0
    count = 0
    
    for num in nums:
        current_sum += num
        if current_sum - k in pref_counts:
            count += pref_counts[current_sum - k]
        pref_counts[current_sum] += 1
        
    return count
```
