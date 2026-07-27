```python
def transpose(self, matrix: list[list[int]]) -> list[list[int]]:
    r = len(matrix)
    c = len(matrix[0])
    ans = [[0] * r for _ in range(c)]
    for i in range(r):
        for j in range(c):
            ans[j][i] = matrix[i][j]
    return ans
```