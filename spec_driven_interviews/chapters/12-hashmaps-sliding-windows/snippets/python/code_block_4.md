```python
from collections import defaultdict
hash_map = defaultdict(list)
for s in strs:
    count = [0] * 26
    for c in s: count[ord(c) - ord('a')] += 1
    key = str(count)
    hash_map[key].append(s)
```