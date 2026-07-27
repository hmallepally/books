```java
public void solve(char[][] board) {
  int m = board.length, n = board[0].length;
  for (int i = 0; i < m; i++) { dfs(board, i, 0); dfs(board, i, n-1); }
  for (int j = 0; j < n; j++) { dfs(board, 0, j); dfs(board, m-1, j); }
  
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (board[i][j] == 'O') board[i][j] = 'X';
      else if (board[i][j] == '#') board[i][j] = 'O';
    }
  }
}
private void dfs(char[][] b, int r, int c) {
  if (r<0 || r>=b.length || c<0 || c>=b[0].length || b[r][c] != 'O') return;
  b[r][c] = '#';
  dfs(b, r+1, c); dfs(b, r-1, c); dfs(b, r, c+1); dfs(b, r, c-1);
}
```
