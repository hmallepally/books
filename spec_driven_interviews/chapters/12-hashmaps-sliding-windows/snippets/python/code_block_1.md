```python
left = max_len = 0 # <1>
for right in range(len(arr)): # <2>
    # Ingest arr[right] into window state
    while False: # window state violates invariant <3>
        # Remove arr[left] from window state
        left += 1 # <4>
    max_len = max(max_len, right - left + 1) # <5>
```