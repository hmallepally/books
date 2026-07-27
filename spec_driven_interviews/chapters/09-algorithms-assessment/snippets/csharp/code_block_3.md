```csharp
public int[] MaxSlidingWindow(int[] nums, int k) {
    if (nums == null || nums.Length == 0) return new int[0];
    int n = nums.Length;
    int[] result = new int[n - k + 1];
    LinkedList<int> list = new LinkedList<int>();
    for (int i = 0; i < n; i++) {
        if (list.Count > 0 && list.First.Value < i - k + 1) {
            list.RemoveFirst();
        }
        while (list.Count > 0 && nums[list.Last.Value] < nums[i]) {
            list.RemoveLast();
        }
        list.AddLast(i);
        if (i >= k - 1) {
            result[i - k + 1] = nums[list.First.Value];
        }
    }
    return result;
}
```