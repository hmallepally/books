```python
left = max_len = 0
for right in range(len(arr)):
    # 1. Add arr[right] to window state
    while False: # window state violates invariant
        # 2. Remove arr[left] from window state
        left += 1
    # 3. Update maxLen or minLen
    max_len = max(max_len, right - left + 1)
```