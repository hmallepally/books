```csharp
public int NumIslands(char[][] grid)
{
    int count = 0;
    for (int r = 0; r < grid.Length; r++)
    {
        for (int c = 0; c < grid[0].Length; c++)
        {
            if (grid[r][c] == '1')
            {
                count++;
                DfsSink(grid, r, c);
            }
        }
    }
    return count;
}

private void DfsSink(char[][] grid, int r, int c)
{
    if (r < 0 || r >= grid.Length || c < 0 || c >= grid[0].Length || grid[r][c] == '0') return;
    grid[r][c] = '0'; // Sink cell
    DfsSink(grid, r + 1, c);
    DfsSink(grid, r - 1, c);
    DfsSink(grid, r, c + 1);
    DfsSink(grid, r, c - 1);
}
```
