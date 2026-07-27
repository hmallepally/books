```python
from sortedcontainers import SortedDict, SortedList

# SortedDict — sorted key-value mapping (pip install sortedcontainers)
sd = SortedDict()
sd[10] = "ten"
sd[30] = "thirty"
sd[20] = "twenty"

sd.peekitem(0)             # Smallest key-value pair — O(log N)
sd.peekitem(-1)            # Largest key-value pair — O(log N)
sd.bisect_left(25)         # Index where 25 would be inserted

# SortedList — sorted unique elements with O(log N) operations
sl = SortedList([30, 10, 20])
sl[0]                      # 10 (smallest)
sl[-1]                     # 30 (largest)
sl.bisect_left(25)         # Index for floor/ceiling calculations
# Floor: sl[sl.bisect_right(25) - 1]  -> 20
# Ceiling: sl[sl.bisect_left(25)]     -> 30
```
