```csharp
public int[] NextGreaterElement(int[] nums) {
    int n = nums.Length;
    int[] result = new int[n];
    Array.Fill(result, -1); // <1>
    Stack<int> stack = new Stack<int>(); // <2>
    for (int i = 0; i < n; i++) {
        // Maintain strictly decreasing stack
        while (stack.Count > 0 && nums[i] > nums[stack.Peek()]) {
            int prevIndex = stack.Pop();
            result[prevIndex] = nums[i]; // <3> Found next greater element
        }
        stack.Push(i); // <4>
    }
    return result;
}
```