```python
def maximal_rectangle(self, matrix: list[list[str]]) -> int:
    if not matrix or not matrix[0]: return 0
    cols = len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for row in matrix:
        # Update histogram heights
        for c in range(cols):
            heights[c] = heights[c] + 1 if row[c] == '1' else 0
        max_area = max(max_area, self._max_histogram(heights))
        
    return max_area

def _max_histogram(self, heights: list[int]) -> int:
    stack = []
    max_val = 0
    n = len(heights)
    
    for i in range(n + 1):
        h = 0 if i == n else heights[i]
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_val = max(max_val, height * width)
        stack.append(i)
        
    return max_val
# Time Complexity: O(R * C)
# Space Complexity: O(C)
```