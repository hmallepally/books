```csharp
public boolean Exist(char[][] board, string word) {
    int m = board.Length;
    int n = board[0].Length;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (Dfs(board, word, i, j, 0)) return true;
        }
    }
    return false;
}

private bool Dfs(char[][] board, string word, int r, int c, int index) {
    if (index == word.Length) return true;
    if (r < 0 || r >= board.Length || c < 0 || c >= board[0].Length || board[r][c] != word[index]) {
        return false;
    }
    char temp = board[r][c];
    board[r][c] = '#';
    bool found = Dfs(board, word, r + 1, c, index + 1)
              || Dfs(board, word, r - 1, c, index + 1)
              || Dfs(board, word, r, c + 1, index + 1)
              || Dfs(board, word, r, c - 1, index + 1);
    board[r][c] = temp;
    return found;
}
```