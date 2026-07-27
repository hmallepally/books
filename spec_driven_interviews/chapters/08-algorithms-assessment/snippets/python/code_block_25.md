```python
# dict: Key -> Value mapping
map = {}
map["apple"] = 3                           # Insert/update — O(1)
map["apple"]                               # Lookup — O(1), raises KeyError if missing
map.get("banana", 0)                       # Lookup with fallback — O(1)
"apple" in map                             # Key existence check — O(1)
del map["apple"]                           # Remove by key — O(1)
map.keys()                                 # All keys (for iteration)
map.values()                               # All values
map.items()                                # All key-value pairs

# Frequency counting pattern (extremely common)
from collections import Counter
freq = Counter(text)                       # One-liner frequency count

# set: Unique element storage
seen = set()
seen.add(42)               # Add element — O(1)
42 in seen                 # Membership check — O(1)
seen.discard(42)           # Remove element — O(1), no error if missing
```
