```csharp
public void Solve(char[][] board) {
  int m = board.Length, n = board[0].Length;
  for (int i = 0; i < m; i++) { Dfs(board, i, 0); Dfs(board, i, n-1); }
  for (int j = 0; j < n; j++) { Dfs(board, 0, j); Dfs(board, m-1, j); }
  
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (board[i][j] == 'O') board[i][j] = 'X';
      else if (board[i][j] == '#') board[i][j] = 'O';
    }
  }
}
private void Dfs(char[][] b, int r, int c) {
  if (r<0 || r>=b.Length || c<0 || c>=b[0].Length || b[r][c] != 'O') return;
  b[r][c] = '#';
  Dfs(b, r+1, c); Dfs(b, r-1, c); Dfs(b, r, c+1); Dfs(b, r, c-1);
}
```