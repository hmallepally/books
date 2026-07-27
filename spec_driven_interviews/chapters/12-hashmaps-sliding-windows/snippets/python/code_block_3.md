```python
from collections import defaultdict
hash_map = defaultdict(int)
hash_map[0] = 1 # Base case for subarrays starting at index 0
total_sum = count = 0
for num in nums:
    total_sum += num
    if (total_sum - k) in hash_map:
        count += hash_map[total_sum - k]
    hash_map[total_sum] += 1
```