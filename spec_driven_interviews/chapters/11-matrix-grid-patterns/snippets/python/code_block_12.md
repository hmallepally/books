```python
def is_toeplitz_matrix(self, matrix: list[list[int]]) -> bool:
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][j] != matrix[i-1][j-1]:
                return False
    return True
```