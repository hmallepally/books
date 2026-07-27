```python
def max_sum(self, mat: list[list[int]], k: int) -> int:
    m, n = len(mat), len(mat[0])
    pre = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pre[i][j] = mat[i-1][j-1] + pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1]
            
    max_val = float('-inf')
    for i in range(k, m + 1):
        for j in range(k, n + 1):
            s = pre[i][j] - pre[i-k][j] - pre[i][j-k] + pre[i-k][j-k]
            max_val = max(max_val, s)
    return max_val
```