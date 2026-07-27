```csharp
public double FindMedianSortedArrays(int[] A, int[] B) {
    if (A.Length > B.Length) return FindMedianSortedArrays(B, A); // ensure A is smaller
    int m = A.Length, n = B.Length;
    int left = 0, right = m;
    
    while (left <= right) {
        int i = (left + right) / 2; // partition A
        int j = (m + n + 1) / 2 - i; // partition B
        
        int maxLeftA = (i == 0) ? int.MinValue : A[i - 1];
        int minRightA = (i == m) ? int.MaxValue : A[i];
        int maxLeftB = (j == 0) ? int.MinValue : B[j - 1];
        int minRightB = (j == n) ? int.MaxValue : B[j];
        
        if (maxLeftA <= minRightB && maxLeftB <= minRightA) {
            // Correct partition found
            if ((m + n) % 2 == 0) {
                return (Math.Max(maxLeftA, maxLeftB) + Math.Min(minRightA, minRightB)) / 2.0;
            } else {
                return Math.Max(maxLeftA, maxLeftB);
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