```python
def matrix_reshape(self, mat: list[list[int]], r: int, c: int) -> list[list[int]]:
    m, n = len(mat), len(mat[0])
    if m * n != r * c: return mat # Invalid shape
    
    res = [[0] * c for _ in range(r)]
    for i in range(m * n):
        res[i // c][i % c] = mat[i // n][i % n]
    return res
```