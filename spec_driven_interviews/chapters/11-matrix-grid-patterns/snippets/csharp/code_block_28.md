```csharp
public string Convert(string s, int numRows) {
  if (numRows == 1) return s;
  StringBuilder[] rows = new StringBuilder[Math.Min(numRows, s.Length)];
  for (int i = 0; i < rows.Length; i++) rows[i] = new StringBuilder();
  
  int curRow = 0;
  bool goingDown = false;
  foreach (char c in s.ToCharArray()) {
    rows[curRow].Append(c);
    if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
    curRow += goingDown ? 1 : -1;
  }
  
  StringBuilder ret = new StringBuilder();
  foreach (StringBuilder row in rows) ret.Append(row);
  return ret.ToString();
}
```