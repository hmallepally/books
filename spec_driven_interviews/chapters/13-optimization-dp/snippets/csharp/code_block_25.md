```csharp
public int LengthOfLIS(int[] nums) {
    int[] tails = new int[nums.Length];
    int size = 0;
    foreach (int x in nums) {
        int left = 0, right = size;
        while (left != right) {
            int mid = left + (right - left) / 2;
            if (tails[mid] < x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        tails[left] = x;
        if (left == size) size++; // Found a larger element, expand LIS
    }
    return size;
}
// Time Complexity: O(N log N)
// Space Complexity: O(N)
```