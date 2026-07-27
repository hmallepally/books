```java
// Standard Binary Search
int binarySearch(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// Binary Search on Answer Space (Leftmost valid)
int binarySearchAnswerSpace(int min, int max) {
    int left = min, right = max;
    int best = -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (isValid(mid)) {
            best = mid;
            right = mid - 1; // Try to find a smaller valid answer
        } else {
            left = mid + 1;
        }
    }
    return best;
}
```
