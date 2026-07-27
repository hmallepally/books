```python
def box_blur(self, image: list[list[int]]) -> list[list[int]]:
    m, n = len(image), len(image[0])
    res = [[0] * (n - 2) for _ in range(m - 2)]
    
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            s = sum(image[i + di][j + dj] for di in range(-1, 2) for dj in range(-1, 2))
            res[i-1][j-1] = s // 9
    return res
```