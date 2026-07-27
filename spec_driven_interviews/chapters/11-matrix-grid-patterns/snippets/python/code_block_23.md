```python
def find_rotation(self, mat: list[list[int]], target: list[list[int]]) -> bool:
    for k in range(4):
        if mat == target: return True
        self.rotate(mat)
    return False

def rotate(self, mat: list[list[int]]) -> None:
    n = len(mat)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    for i in range(n):
        for j in range(n // 2):
            mat[i][j], mat[i][n-1-j] = mat[i][n-1-j], mat[i][j]
```