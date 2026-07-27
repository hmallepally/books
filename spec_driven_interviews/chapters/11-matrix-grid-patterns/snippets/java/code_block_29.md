```java
public int robotSim(int[] commands, int[][] obstacles) {
  int[] dx = {0, 1, 0, -1}, dy = {1, 0, -1, 0};
  Set<String> obs = new HashSet<>();
  for (int[] o : obstacles) obs.add(o[0] + "," + o[1]);
  
  int x = 0, y = 0, dir = 0, maxDist = 0;
  for (int cmd : commands) {
    if (cmd == -2) dir = (dir + 3) % 4;
    else if (cmd == -1) dir = (dir + 1) % 4;
    else {
      for (int k = 0; k < cmd; k++) {
        int nx = x + dx[dir], ny = y + dy[dir];
        if (obs.contains(nx + "," + ny)) break;
        x = nx; y = ny;
        maxDist = Math.max(maxDist, x*x + y*y);
      }
    }
  }
  return maxDist;
}
```
