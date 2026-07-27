```csharp
public int[] DailyTemperatures(int[] temps)
{
    var ans = new int[temps.Length];
    var stack = new Stack<int>(); // Stores INDICES

    for (int i = 0; i < temps.Length; i++)
    {
        while (stack.Count > 0 && temps[stack.Peek()] < temps[i])
        {
            int prevIdx = stack.Pop();
            ans[prevIdx] = i - prevIdx;
        }
        stack.Push(i);
    }
    return ans;
}
```
