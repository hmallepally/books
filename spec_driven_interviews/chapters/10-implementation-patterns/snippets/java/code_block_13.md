```java
public int[] neighborSum(int[] a) {
    if (a == null) return new int[0];
    int n = a.length;
    int[] b = new int[n];

    for (int i = 0; i < n; i++) {
        int leftVal  = (i > 0) ? a[i - 1] : 0;
        int rightVal = (i < n - 1) ? a[i + 1] : 0;
        b[i] = leftVal + a[i] + rightVal;
    }

    return b;
}
// Time: O(N), Space: O(N) for output array
```
