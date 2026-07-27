```csharp
public class StockSpanner {
    // Stack holds {price, span}
    private Stack<int[]> stack = new Stack<int[]>(); 
    
    public int Next(int price) {
        int span = 1;
        while (stack.Count > 0 && stack.Peek()[0] <= price) {
            span += stack.Pop()[1]; // Accumulate previous spans
        }
        stack.Push(new int[]{price, span});
        return span;
    }
}
// Time Complexity: Amortized O(1) per next() call
// Space Complexity: O(N)
```