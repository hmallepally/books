```python
def solve(self, board: list[list[str]]) -> None:
    m, n = len(board), len(board[0])
    for i in range(m):
        self._dfs_s(board, i, 0)
        self._dfs_s(board, i, n-1)
    for j in range(n):
        self._dfs_s(board, 0, j)
        self._dfs_s(board, m-1, j)
        
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O': board[i][j] = 'X'
            elif board[i][j] == '#': board[i][j] = 'O'

def _dfs_s(self, b: list[list[str]], r: int, c: int) -> None:
    if r < 0 or r >= len(b) or c < 0 or c >= len(b[0]) or b[r][c] != 'O': return
    b[r][c] = '#'
    self._dfs_s(b, r+1, c); self._dfs_s(b, r-1, c)
    self._dfs_s(b, r, c+1); self._dfs_s(b, r, c-1)
```