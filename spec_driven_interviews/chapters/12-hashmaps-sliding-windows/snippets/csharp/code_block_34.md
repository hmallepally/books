```csharp
public int NumDistinctIslands(int[][] grid) {
    HashSet<string> set = new HashSet<string>();
    for (int i = 0; i < grid.Length; i++) {
        for (int j = 0; j < grid[0].Length; j++) {
            if (grid[i][j] == 1) {
                StringBuilder sb = new StringBuilder();
                Dfs(grid, i, j, "S", sb); // Start with 'S'
                set.Add(sb.ToString());
            }
        }
    }
    return set.Count;
}
private void Dfs(int[][] grid, int r, int c, string dir, StringBuilder sb) {
    if (r < 0 || c < 0 || r >= grid.Length || c >= grid[0].Length || grid[r][c] == 0) return;
    grid[r][c] = 0; // mark visited
    sb.Append(dir);
    Dfs(grid, r + 1, c, "D", sb);
    Dfs(grid, r - 1, c, "U", sb);
    Dfs(grid, r, c + 1, "R", sb);
    Dfs(grid, r, c - 1, "L", sb);
    sb.Append("B"); // Backtrack to distinguish paths
}
// Time Complexity: O(R * C) | Space Complexity: O(R * C)
```