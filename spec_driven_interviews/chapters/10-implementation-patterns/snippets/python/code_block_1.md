```python
# Retains elements satisfying a condition, overwrites list in-place
write = 0
for read in range(len(arr)):
    if keep_condition(arr[read]):
        arr[write] = arr[read]
        write += 1
# Result is arr[0..write-1], return write as the new length
```