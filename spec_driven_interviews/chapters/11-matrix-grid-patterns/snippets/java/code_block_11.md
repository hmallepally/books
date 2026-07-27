```java
public void gameOfLife(int[][] board) {
  int m = board.length, n = board[0].length;
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < n; c++) {
      int live = 0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          if (i == 0 && j == 0) continue;
          int nr = r + i, nc = c + j;
          if (nr >= 0 && nr < m && nc >= 0 && nc < n && Math.abs(board[nr][nc]) == 1) live++;
        }
      }
      if (board[r][c] == 1 && (live < 2 || live > 3)) board[r][c] = -1;
      if (board[r][c] == 0 && live == 3) board[r][c] = 2;
    }
  }
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < n; c++) {
      if (board[r][c] > 0) board[r][c] = 1;
      else board[r][c] = 0;
    }
  }
}
```
