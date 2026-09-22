```java
public int[] nextGreaterElement(int[] nums) {
    int n = nums.length;
    int[] result = new int[n];
    Arrays.fill(result, -1); // <1>
    Deque<Integer> stack = new ArrayDeque<>(); // <2>
    for (int i = 0; i < n; i++) {
        // Maintain strictly decreasing stack
        while (!stack.isEmpty() && nums[i] > nums[stack.peek()]) {
            int prevIndex = stack.pop();
            result[prevIndex] = nums[i]; // <3> Found next greater element
        }
        stack.push(i); // <4>
    }
    return result;
}
```
