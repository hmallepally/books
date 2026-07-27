```python
def sort_by_height(self, a: list[int]) -> list[int]:
    # Step 1: Extract all non-tree heights
    heights = [h for h in a if h != -1]

    # Step 2: Sort the extracted heights
    heights.sort()

    # Step 3: Reinsert sorted heights at non-tree positions
    index = 0
    for i in range(len(a)):
        if a[i] != -1:
            a[i] = heights[index]
            index += 1

    return a
# Time: O(N log N) for sorting, Space: O(N) for extracted list
```