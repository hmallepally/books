```python
# Retains elements satisfying a condition, overwrites list in-place
write = 0 # <1>
for read in range(len(arr)): # <2>
    if keep_condition(arr[read]): # <3>
        arr[write] = arr[read] # <4>
        write += 1 # <5>
# Result is arr[0..write-1], return write as the new length
```