```csharp
public double FindMedianSortedArrays(int[] A, int[] B) {
    if (A.Length > B.Length) {
        return FindMedianSortedArrays(B, A);
    }
    int m = A.Length;
    int n = B.Length;
    int left = 0, right = m;
    while (left <= right) {
        int i = left + (right - left) / 2;
        int j = (m + n + 1) / 2 - i;
        
        int aLeft = (i == 0) ? int.MinValue : A[i - 1];
        int aRight = (i == m) ? int.MaxValue : A[i];
        int bLeft = (j == 0) ? int.MinValue : B[j - 1];
        int bRight = (j == n) ? int.MaxValue : B[j];
        
        if (aLeft <= bRight && bLeft <= aRight) {
            if ((m + n) % 2 == 1) {
                return Math.Max(aLeft, bLeft);
            }
            return (Math.Max(aLeft, bLeft) + Math.Min(aRight, bRight)) / 2.0;
        } else if (aLeft > bRight) {
            right = i - 1;
        } else {
            left = i + 1;
        }
    }
    return 0.0;
}
```