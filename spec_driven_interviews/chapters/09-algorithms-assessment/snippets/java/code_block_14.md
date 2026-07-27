```java
public double[] medianSlidingWindow(int[] nums, int k) {
    int n = nums.length;
    double[] result = new double[n - k + 1];
    // Max heap for lower half
    PriorityQueue<Integer> left = new PriorityQueue<>(Collections.reverseOrder());
    // Min heap for upper half
    PriorityQueue<Integer> right = new PriorityQueue<>();

    for (int i = 0; i < n; i++) {
        if (left.isEmpty() || nums[i] <= left.peek()) {
            left.add(nums[i]);
        } else {
            right.add(nums[i]);
        }
        balance(left, right);

        if (i >= k - 1) {
            if (left.size() == right.size()) {
                result[i - k + 1] = ((double)left.peek() + right.peek()) / 2.0;
            } else {
                result[i - k + 1] = left.peek();
            }
            
            // Remove element sliding out of window
            int elementToRemove = nums[i - k + 1];
            if (elementToRemove <= left.peek()) {
                left.remove(elementToRemove);
            } else {
                right.remove(elementToRemove);
            }
            balance(left, right);
        }
    }
    return result;
}

private void balance(PriorityQueue<Integer> left, PriorityQueue<Integer> right) {
    if (left.size() > right.size() + 1) {
        right.add(left.poll());
    } else if (left.size() < right.size()) {
        left.add(right.poll());
    }
}
```