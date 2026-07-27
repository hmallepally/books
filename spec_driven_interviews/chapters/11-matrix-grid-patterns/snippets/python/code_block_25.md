```python
def update_board(self, board: list[list[str]], click: list[int]) -> list[list[str]]:
    r, c = click[0], click[1]
    if board[r][c] == 'M':
        board[r][c] = 'X'
        return board
    self._dfs(board, r, c)
    return board

def _dfs(self, b: list[list[str]], r: int, c: int) -> None:
    if r < 0 or c < 0 or r >= len(b) or c >= len(b[0]) or b[r][c] != 'E': return
    mines = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            nr, nc = r + i, c + j
            if 0 <= nr < len(b) and 0 <= nc < len(b[0]) and b[nr][nc] == 'M':
                mines += 1
                
    if mines > 0:
        b[r][c] = str(mines)
    else:
        b[r][c] = 'B'
        for i in range(-1, 2):
            for j in range(-1, 2):
                self._dfs(b, r+i, c+j)
```