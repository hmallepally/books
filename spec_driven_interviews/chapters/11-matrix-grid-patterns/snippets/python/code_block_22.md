```python
def exist(self, board: list[list[str]], word: str) -> bool:
    for i in range(len(board)):
        for j in range(len(board[0])):
            if self._dfs(board, i, j, word, 0): return True
    return False

def _dfs(self, b: list[list[str]], r: int, c: int, word: str, idx: int) -> bool:
    if idx == len(word): return True
    if r < 0 or c < 0 or r >= len(b) or c >= len(b[0]) or b[r][c] != word[idx]: return False
    
    temp = b[r][c]
    b[r][c] = '#'
    found = (self._dfs(b, r+1, c, word, idx+1) or self._dfs(b, r-1, c, word, idx+1) or
             self._dfs(b, r, c+1, word, idx+1) or self._dfs(b, r, c-1, word, idx+1))
    b[r][c] = temp
    return found
```