```java
public boolean exist(char[][] board, String word) {
  for (int i = 0; i < board.length; i++) {
    for (int j = 0; j < board[0].length; j++) {
      if (dfs(board, i, j, word, 0)) return true;
    }
  }
  return false;
}
private boolean dfs(char[][] b, int r, int c, String word, int idx) {
  if (idx == word.length()) return true;
  if (r < 0 || c < 0 || r >= b.length || c >= b[0].length || b[r][c] != word.charAt(idx)) return false;
  char temp = b[r][c];
  b[r][c] = '#';
  boolean found = dfs(b, r+1, c, word, idx+1) || dfs(b, r-1, c, word, idx+1) ||
                  dfs(b, r, c+1, word, idx+1) || dfs(b, r, c-1, word, idx+1);
  b[r][c] = temp;
  return found;
}
```
