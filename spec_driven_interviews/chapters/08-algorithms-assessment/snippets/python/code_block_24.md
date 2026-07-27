```python
lst = []
lst.append(42)             # Append to end — O(1) amortized
lst.insert(0, 99)          # Insert at index 0 — O(N) shift
lst[0]                     # Random access — O(1)
lst[1] = 50                # Replace at index — O(1)
del lst[0]                 # Remove at index — O(N) shift
len(lst)                   # Current element count
42 in lst                  # Linear search — O(N)
not lst                    # Check if empty (Pythonic)
lst.sort()                 # Sort in-place — O(N log N)
```
