```csharp
public bool IsValidSudoku(char[][] board) {
  HashSet<string> seen = new HashSet<string>();
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      char number = board[i][j];
      if (number != '.') {
        int boxIdx = (i / 3) * 3 + j / 3;
        if (!seen.Add(number + " in row " + i) ||
            !seen.Add(number + " in col " + j) ||
            !seen.Add(number + " in box " + boxIdx))
          return false;
      }
    }
  }
  return true;
}
```