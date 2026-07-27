```java
public int maxProduct(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    int maxVal = nums[0], minVal = nums[0], result = nums[0];
    
    for (int i = 1; i < nums.length; i++) {
        // If current is negative, max and min will swap roles
        if (nums[i] < 0) {
            int temp = maxVal; 
            maxVal = minVal; 
            minVal = temp;
        }
        maxVal = Math.max(nums[i], maxVal * nums[i]);
        minVal = Math.min(nums[i], minVal * nums[i]);
        result = Math.max(result, maxVal);
    }
    return result;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```
