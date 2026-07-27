```csharp
public int LongestOnes(int[] nums, int k) {
    int left = 0;
    for (int right = 0; right < nums.Length; right++) {
        if (nums[right] == 0) k--;
        if (k < 0) { // Over budget
            if (nums[left++] == 0) k++;
        }
    }
    return nums.Length - left; // Trick to return max valid length seen
}
// Time Complexity: O(N) | Space Complexity: O(1)
```