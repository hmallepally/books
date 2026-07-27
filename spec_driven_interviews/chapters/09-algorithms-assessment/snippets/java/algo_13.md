```java
public int shortestPath(char[][] grid, int startR, int startC) {
    int rows = grid.length, cols = grid[0].length;
    var queue = new ArrayDeque<int[]>();
    boolean[][] visited = new boolean[rows][cols];

    queue.offer(new int[]{startR, startC});
    visited[startR][startC] = true; // Mark visited ON PUSH
    int steps = 0;
    int[][] DIRS = {{1,0},{-1,0},{0,1},{0,-1}};

    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            int[] curr = queue.poll();
            if (grid[curr[0]][curr[1]] == 'E') return steps;

            for (int[] d : DIRS) {
                int nr = curr[0] + d[0], nc = curr[1] + d[1];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols 
                    && !visited[nr][nc] && grid[nr][nc] != 'X') {
                    visited[nr][nc] = true; // MARK ON PUSH!
                    queue.offer(new int[]{nr, nc});
                }
            }
        }
        steps++;
    }
    return -1;
}
```
