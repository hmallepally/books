```java
public boolean isValidSudoku(char[][] board) {
  Set<Integer> seen = new HashSet<>();
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      char number = board[i][j];
      if (number != '.') {
        int boxIdx = (i / 3) * 3 + j / 3;
        int rowKey = number * 100 + i;
        int colKey = number * 100 + j + 27;
        int boxKey = number * 100 + boxIdx + 54;
        if (!seen.add(rowKey) ||
            !seen.add(colKey) ||
            !seen.add(boxKey))
          return false;
      }
    }
  }
  return true;
}
```
