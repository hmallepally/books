```java
public double findMedianSortedArrays(int[] A, int[] B) {
    if (A.length > B.length) return findMedianSortedArrays(B, A); // ensure A is smaller
    int m = A.length, n = B.length;
    int left = 0, right = m;
    
    while (left <= right) {
        int i = (left + right) / 2; // partition A
        int j = (m + n + 1) / 2 - i; // partition B
        
        int maxLeftA = (i == 0) ? Integer.MIN_VALUE : A[i - 1];
        int minRightA = (i == m) ? Integer.MAX_VALUE : A[i];
        int maxLeftB = (j == 0) ? Integer.MIN_VALUE : B[j - 1];
        int minRightB = (j == n) ? Integer.MAX_VALUE : B[j];
        
        if (maxLeftA <= minRightB && maxLeftB <= minRightA) {
            // Correct partition found
            if ((m + n) % 2 == 0) {
                return (Math.max(maxLeftA, maxLeftB) + Math.min(minRightA, minRightB)) / 2.0;
            } else {
                return Math.max(maxLeftA, maxLeftB);
            }
        } else if (maxLeftA > minRightB) {
            right = i - 1; // move partition left in A
        } else {
            left = i + 1; // move partition right in A
        }
    }
    return 0.0;
}
// Time Complexity: O(log(min(M, N)))
// Space Complexity: O(1)
```
