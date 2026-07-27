```java
public class StockSpanner {
    // Array holds {price, span}
    private Deque<int[]> stack = new ArrayDeque<>(); 
    
    public int next(int price) {
        int span = 1;
        while (!stack.isEmpty() && stack.peek()[0] <= price) {
            span += stack.pop()[1]; // Accumulate previous spans
        }
        stack.push(new int[]{price, span});
        return span;
    }
}
// Time Complexity: Amortized O(1) per next() call
// Space Complexity: O(N)
```
