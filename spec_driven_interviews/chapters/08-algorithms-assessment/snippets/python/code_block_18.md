```python
def findMedianSortedArrays(A: list[int], B: list[int]) -> float:
    if len(A) > len(B):
        A, B = B, A
    m, n = len(A), len(B)
    left, right = 0, m
    while left <= right:
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i
        
        aLeft = float('-inf') if i == 0 else A[i - 1]
        aRight = float('inf') if i == m else A[i]
        bLeft = float('-inf') if j == 0 else B[j - 1]
        bRight = float('inf') if j == n else B[j]
        
        if aLeft <= bRight and bLeft <= aRight:
            if (m + n) % 2 == 1:
                return float(max(aLeft, bLeft))
            return (max(aLeft, bLeft) + min(aRight, bRight)) / 2.0
        elif aLeft > bRight:
            right = i - 1
        else:
            left = i + 1
    return 0.0
```