```python
def find_diagonal_order(self, mat: list[list[int]]) -> list[int]:
    m, n = len(mat), len(mat[0])
    res = [0] * (m * n)
    r, c = 0, 0
    for i in range(m * n):
        res[i] = mat[r][c]
        if (r + c) % 2 == 0: # Moving Up-Right
            if c == n - 1: r += 1
            elif r == 0: c += 1
            else: r -= 1; c += 1
        else: # Moving Down-Left
            if r == m - 1: c += 1
            elif c == 0: r += 1
            else: r += 1; c -= 1
    return res
```