```csharp
public int[] MaxSlidingWindow(int[] nums, int k)
{
    var deque = new LinkedList<int>();
    var res = new int[nums.Length - k + 1];
    int idx = 0;

    for (int i = 0; i < nums.Length; i++)
    {
        while (deque.Count > 0 && deque.First.Value < i - k + 1) deque.RemoveFirst(); // Expire
        while (deque.Count > 0 && nums[deque.Last.Value] < nums[i]) deque.RemoveLast(); // Kill weaker
        deque.AddLast(i);
        if (i >= k - 1) res[idx++] = nums[deque.First.Value];
    }
    return res;
}
```
