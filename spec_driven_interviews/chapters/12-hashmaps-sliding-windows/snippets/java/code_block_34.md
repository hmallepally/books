```java
public int numDistinctIslands(int[][] grid) {
    Set<String> set = new HashSet<>();
    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
            if (grid[i][j] == 1) {
                StringBuilder sb = new StringBuilder();
                dfs(grid, i, j, "S", sb); // Start with 'S'
                set.add(sb.toString());
            }
        }
    }
    return set.size();
}
private void dfs(int[][] grid, int r, int c, String dir, StringBuilder sb) {
    if (r < 0 || c < 0 || r >= grid.length || c >= grid[0].length || grid[r][c] == 0) return;
    grid[r][c] = 0; // mark visited
    sb.append(dir);
    dfs(grid, r + 1, c, "D", sb);
    dfs(grid, r - 1, c, "U", sb);
    dfs(grid, r, c + 1, "R", sb);
    dfs(grid, r, c - 1, "L", sb);
    sb.append("B"); // Backtrack to distinguish paths
}
// Time Complexity: O(R * C) | Space Complexity: O(R * C)
```
