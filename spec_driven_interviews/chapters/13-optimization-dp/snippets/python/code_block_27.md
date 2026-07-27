```python
def kth_smallest(self, matrix: list[list[int]], k: int) -> int:
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    while left < right:
        mid = left + (right - left) // 2
        count = self._count_less_equal(matrix, mid)
        if count < k: left = mid + 1
        else: right = mid
    return left

def _count_less_equal(self, matrix: list[list[int]], target: int) -> int:
    n, i, j, count = len(matrix), len(matrix) - 1, 0, 0
    while i >= 0 and j < n:
        if matrix[i][j] <= target:
            count += i + 1
            j += 1
        else:
            i -= 1
    return count
# Time Complexity: O(N log(Max - Min))
# Space Complexity: O(1)
```