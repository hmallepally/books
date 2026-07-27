```java
public int minPathSum(int[][] grid) {
    int rows = grid.length, cols = grid[0].length;
    int[][] dp = new int[rows][cols];

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (r == 0 && c == 0) dp[r][c] = grid[r][c];
            else if (r == 0) dp[r][c] = dp[r][c - 1] + grid[r][c];
            else if (c == 0) dp[r][c] = dp[r - 1][c] + grid[r][c];
            else dp[r][c] = Math.min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c];
        }
    }
    return dp[rows - 1][cols - 1];
}
```
