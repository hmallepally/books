```csharp
public int ShortestPath(char[][] grid, int startR, int startC)
{
    int rows = grid.Length, cols = grid[0].Length;
    var queue = new Queue<int[]>();
    bool[][] visited = new bool[rows][];
    for (int i = 0; i < rows; i++) visited[i] = new bool[cols];

    queue.Enqueue(new int[] { startR, startC });
    visited[startR][startC] = true; // Mark visited ON PUSH
    int steps = 0;
    int[][] DIRS = new int[][] {
        new int[] { 1, 0 }, new int[] { -1, 0 },
        new int[] { 0, 1 }, new int[] { 0, -1 }
    };

    while (queue.Count > 0)
    {
        int size = queue.Count;
        for (int i = 0; i < size; i++)
        {
            int[] curr = queue.Dequeue();
            if (grid[curr[0]][curr[1]] == 'E') return steps;

            foreach (int[] d in DIRS)
            {
                int nr = curr[0] + d[0], nc = curr[1] + d[1];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols
                    && !visited[nr][nc] && grid[nr][nc] != 'X')
                {
                    visited[nr][nc] = true; // MARK ON PUSH!
                    queue.Enqueue(new int[] { nr, nc });
                }
            }
        }
        steps++;
    }
    return -1;
}
```
