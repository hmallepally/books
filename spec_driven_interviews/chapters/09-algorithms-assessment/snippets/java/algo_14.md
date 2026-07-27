```java
public int orangesRotting(int[][] grid) {
    int rows = grid.length, cols = grid[0].length;
    var queue = new ArrayDeque<int[]>();
    int freshCount = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == 2) queue.offer(new int[]{r, c}); // Push ALL sources
            else if (grid[r][c] == 1) freshCount++;
        }
    }
    if (freshCount == 0) return 0;
    int minutes = 0;
    int[][] DIRS = {{1,0},{-1,0},{0,1},{0,-1}};

    while (!queue.isEmpty() && freshCount > 0) {
        int size = queue.size();
        minutes++;
        for (int i = 0; i < size; i++) {
            int[] curr = queue.poll();
            for (int[] d : DIRS) {
                int nr = curr[0] + d[0], nc = curr[1] + d[1];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                    grid[nr][nc] = 2; // Mutate grid as visited
                    freshCount--;
                    queue.offer(new int[]{nr, nc});
                }
            }
        }
    }
    return freshCount == 0 ? minutes : -1;
}
```
