```csharp
public int OrangesRotting(int[][] grid)
{
    int rows = grid.Length, cols = grid[0].Length;
    var queue = new Queue<int[]>();
    int freshCount = 0;

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (grid[r][c] == 2) queue.Enqueue(new int[] { r, c }); // Push ALL sources
            else if (grid[r][c] == 1) freshCount++;
        }
    }
    if (freshCount == 0) return 0;
    
    int minutes = 0;
    int[][] DIRS = new int[][] {
        new int[] { 1, 0 }, new int[] { -1, 0 },
        new int[] { 0, 1 }, new int[] { 0, -1 }
    };

    while (queue.Count > 0 && freshCount > 0)
    {
        int size = queue.Count;
        minutes++;
        for (int i = 0; i < size; i++)
        {
            int[] curr = queue.Dequeue();
            foreach (int[] d in DIRS)
            {
                int nr = curr[0] + d[0], nc = curr[1] + d[1];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1)
                {
                    grid[nr][nc] = 2; // Mutate grid as visited
                    freshCount--;
                    queue.Enqueue(new int[] { nr, nc });
                }
            }
        }
    }
    return freshCount == 0 ? minutes : -1;
}
```
