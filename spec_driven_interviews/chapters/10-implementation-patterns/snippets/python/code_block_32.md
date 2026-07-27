```python
def matrix_elements_sum(self, matrix: list[list[int]]) -> int:
    rows = len(matrix)
    cols = len(matrix[0])
    total = 0

    for c in range(cols):
        for r in range(rows):
            if matrix[r][c] == 0:
                break # All rooms below are haunted — skip rest of column
            total += matrix[r][c]

    return total
# Time: O(rows * cols), Space: O(1)
```