```python
class StockSpanner:
    def __init__(self):
        # Array holds [price, span]
        self.stack = []
        
    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1] # Accumulate previous spans
        self.stack.append([price, span])
        return span
# Time Complexity: Amortized O(1) per next() call
# Space Complexity: O(N)
```