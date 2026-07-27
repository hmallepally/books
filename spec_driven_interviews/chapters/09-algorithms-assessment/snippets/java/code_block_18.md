```java
public double findMedianSortedArrays(int[] A, int[] B) {
    if (A.length > B.length) {
        return findMedianSortedArrays(B, A); // Ensure A is the shorter array
    }
    int m = A.length;
    int n = B.length;
    int left = 0, right = m;
    while (left <= right) {
        int i = left + (right - left) / 2;
        int j = (m + n + 1) / 2 - i;
        
        int aLeft = (i == 0) ? Integer.MIN_VALUE : A[i - 1];
        int aRight = (i == m) ? Integer.MAX_VALUE : A[i];
        int bLeft = (j == 0) ? Integer.MIN_VALUE : B[j - 1];
        int bRight = (j == n) ? Integer.MAX_VALUE : B[j];
        
        if (aLeft <= bRight && bLeft <= aRight) {
            if ((m + n) % 2 == 1) {
                return Math.max(aLeft, bLeft);
            }
            return (Math.max(aLeft, bLeft) + Math.min(aRight, bRight)) / 2.0;
        } else if (aLeft > bRight) {
            right = i - 1;
        } else {
            left = i + 1;
        }
    }
    return 0.0;
}
```