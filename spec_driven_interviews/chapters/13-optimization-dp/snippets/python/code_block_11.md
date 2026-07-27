```python
def find_median_sorted_arrays(self, A: list[int], B: list[int]) -> float:
    if len(A) > len(B): return self.find_median_sorted_arrays(B, A) # ensure A is smaller
    m, n = len(A), len(B)
    left, right = 0, m
    
    while left <= right:
        i = (left + right) // 2 # partition A
        j = (m + n + 1) // 2 - i # partition B
        
        max_left_a = float('-inf') if i == 0 else A[i - 1]
        min_right_a = float('inf') if i == m else A[i]
        max_left_b = float('-inf') if j == 0 else B[j - 1]
        min_right_b = float('inf') if j == n else B[j]
        
        if max_left_a <= min_right_b and max_left_b <= min_right_a:
            # Correct partition found
            if (m + n) % 2 == 0:
                return (max(max_left_a, max_left_b) + min(min_right_a, min_right_b)) / 2.0
            else:
                return max(max_left_a, max_left_b)
        elif max_left_a > min_right_b:
            right = i - 1 # move partition left in A
        else:
            left = i + 1 # move partition right in A
            
    return 0.0
# Time Complexity: O(log(min(M, N)))
# Space Complexity: O(1)
```