```csharp
public char[][] UpdateBoard(char[][] board, int[] click) {
  int r = click[0], c = click[1];
  if (board[r][c] == 'M') {
    board[r][c] = 'X';
    return board;
  }
  Dfs(board, r, c);
  return board;
}
private void Dfs(char[][] b, int r, int c) {
  if (r < 0 || c < 0 || r >= b.Length || c >= b[0].Length || b[r][c] != 'E') return;
  int mines = 0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int nr = r + i, nc = c + j;
      if (nr >= 0 && nr < b.Length && nc >= 0 && nc < b[0].Length && b[nr][nc] == 'M') mines++;
    }
  }
  if (mines > 0) {
    b[r][c] = (char)(mines + '0');
  } else {
    b[r][c] = 'B';
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) Dfs(b, r+i, c+j);
    }
  }
}
```