```python
k, total_sum, max_val = 3, 0, 0
for i in range(len(arr)):
    total_sum += arr[i] # Add current element
    if i >= k - 1:
        max_val = max(max_val, total_sum) # Update result
        total_sum -= arr[i - (k - 1)]     # Remove leftmost element for next iteration
```