```csharp
public int Rob(int[] nums) {
    if (nums == null || nums.Length == 0) return 0;
    int prev1 = 0; // max so far excluding current
    int prev2 = 0; // max so far including current (-2)
    
    foreach (int num in nums) {
        int temp = Math.Max(prev1, prev2 + num); // rob or don't rob
        prev2 = prev1;
        prev1 = temp;
    }
    return prev1;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```