```csharp
public bool Exist(char[][] board, string word) {
  for (int i = 0; i < board.Length; i++) {
    for (int j = 0; j < board[0].Length; j++) {
      if (Dfs(board, i, j, word, 0)) return true;
    }
  }
  return false;
}
private bool Dfs(char[][] b, int r, int c, string word, int idx) {
  if (idx == word.Length) return true;
  if (r < 0 || c < 0 || r >= b.Length || c >= b[0].Length || b[r][c] != word[idx]) return false;
  char temp = b[r][c];
  b[r][c] = '#';
  bool found = Dfs(b, r+1, c, word, idx+1) || Dfs(b, r-1, c, word, idx+1) ||
               Dfs(b, r, c+1, word, idx+1) || Dfs(b, r, c-1, word, idx+1);
  b[r][c] = temp;
  return found;
}
```