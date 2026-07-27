```csharp
public int[] MaxSlidingWindow(int[] nums, int k) {
    if (nums == null || k <= 0) return new int[0];
    int n = nums.Length;
    int[] res = new int[n - k + 1];
    int resIndex = 0;
    LinkedList<int> q = new LinkedList<int>();
    
    for (int i = 0; i < n; i++) {
        // Remove indices outside the current window
        if (q.Count > 0 && q.First.Value < i - k + 1) {
            q.RemoveFirst();
        }
        // Remove smaller elements (maintain decreasing order)
        while (q.Count > 0 && nums[q.Last.Value] < nums[i]) {
            q.RemoveLast();
        }
        q.AddLast(i);
        
        // Record max for the window
        if (i >= k - 1) {
            res[resIndex++] = nums[q.First.Value];
        }
    }
    return res;
}
// Time Complexity: O(N) since each element is pushed/popped at most once
// Space Complexity: O(K) for the deque
```