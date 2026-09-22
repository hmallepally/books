```python
k, total_sum, max_val = 3, 0, 0 # <1>
for i in range(len(arr)): # <2>
    total_sum += arr[i] # <3>
    if i >= k - 1: # <4>
        max_val = max(max_val, total_sum)
        total_sum -= arr[i - (k - 1)] # <5>
```