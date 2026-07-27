```python
def generate_matrix(self, n: int) -> list[list[int]]:
    mat = [[0] * n for _ in range(n)]
    t, b, l, r = 0, n - 1, 0, n - 1
    val = 1
    while t <= b and l <= r:
        for j in range(l, r + 1):
            mat[t][j] = val
            val += 1
        t += 1
        for i in range(t, b + 1):
            mat[i][r] = val
            val += 1
        r -= 1
        if t <= b:
            for j in range(r, l - 1, -1):
                mat[b][j] = val
                val += 1
            b -= 1
        if l <= r:
            for i in range(b, t - 1, -1):
                mat[i][l] = val
                val += 1
            l += 1
    return mat
```