```python
def spiral_order(self, matrix: list[list[int]]) -> list[int]:
    res = []
    t, b, l, r = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while t <= b and l <= r:
        for j in range(l, r + 1): res.append(matrix[t][j]) # Top
        t += 1
        for i in range(t, b + 1): res.append(matrix[i][r]) # Right
        r -= 1
        if t <= b:
            for j in range(r, l - 1, -1): res.append(matrix[b][j]) # Bottom
            b -= 1
        if l <= r:
            for i in range(b, t - 1, -1): res.append(matrix[i][l]) # Left
            l += 1
    return res
```