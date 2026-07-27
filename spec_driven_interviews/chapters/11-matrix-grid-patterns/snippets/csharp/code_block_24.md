```csharp
public bool Solution(string cell1, string cell2) {
  int sum1 = (cell1[0] - 'A') + (cell1[1] - '1');
  int sum2 = (cell2[0] - 'A') + (cell2[1] - '1');
  return (sum1 % 2) == (sum2 % 2);
}
```