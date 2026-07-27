```java
public int[][] boxBlur(int[][] image) {
  int m = image.length, n = image[0].length;
  int[][] res = new int[m-2][n-2];
  for (int i = 1; i < m - 1; i++) {
    for (int j = 1; j < n - 1; j++) {
      int sum = 0;
      for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
          sum += image[i + di][j + dj];
        }
      }
      res[i-1][j-1] = sum / 9;
    }
  }
  return res;
}
```
