```csharp
public int[] NextGreaterElement(int[] nums) {
    int n = nums.Length;
    int[] result = new int[n];
    Array.Fill(result, -1);
    Stack<int> stack = new Stack<int>(); // stores indices
    for (int i = 0; i < n; i++) {
        // Maintain strictly decreasing stack
        while (stack.Count > 0 && nums[i] > nums[stack.Peek()]) {
            int prevIndex = stack.Pop();
            result[prevIndex] = nums[i]; // Found next greater!
        }
        stack.Push(i);
    }
    return result;
}
```