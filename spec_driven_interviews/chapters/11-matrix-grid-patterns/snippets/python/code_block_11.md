```python
def game_of_life(self, board: list[list[int]]) -> None:
    m, n = len(board), len(board[0])
    for r in range(m):
        for c in range(n):
            live = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0: continue
                    nr, nc = r + i, c + j
                    if 0 <= nr < m and 0 <= nc < n and abs(board[nr][nc]) == 1: live += 1
            if board[r][c] == 1 and (live < 2 or live > 3): board[r][c] = -1
            if board[r][c] == 0 and live == 3: board[r][c] = 2
            
    for r in range(m):
        for c in range(n):
            if board[r][c] > 0: board[r][c] = 1
            else: board[r][c] = 0
```