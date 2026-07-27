```java
public int kthSmallest(int[][] matrix, int k) {
    int n = matrix.length;
    int left = matrix[0][0], right = matrix[n-1][n-1];
    while (left < right) {
        int mid = left + (right - left) / 2;
        int count = countLessEqual(matrix, mid);
        if (count < k) left = mid + 1;
        else right = mid;
    }
    return left;
}
private int countLessEqual(int[][] matrix, int target) {
    int n = matrix.length, i = n - 1, j = 0, count = 0;
    while (i >= 0 && j < n) {
        if (matrix[i][j] <= target) { count += i + 1; j++; }
        else { i--; }
    }
    return count;
}
// Time Complexity: O(N log(Max - Min))
// Space Complexity: O(1)
```
