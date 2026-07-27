```csharp
public int[] DailyTemperatures(int[] temperatures) {
    int n = temperatures.Length;
    int[] res = new int[n];
    Stack<int> stack = new Stack<int>();
    
    for (int i = 0; i < n; i++) {
        // While current temp is greater than temp at stack top
        while (stack.Count > 0 && temperatures[i] > temperatures[stack.Peek()]) {
            int prevIndex = stack.Pop();
            res[prevIndex] = i - prevIndex;
        }
        stack.Push(i);
    }
    return res;
}
// Time Complexity: O(N)
// Space Complexity: O(N)
```