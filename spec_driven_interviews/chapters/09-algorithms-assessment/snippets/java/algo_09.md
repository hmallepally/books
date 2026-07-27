```java
public int[] dailyTemperatures(int[] temps) {
    var ans = new int[temps.length];
    Deque<Integer> stack = new ArrayDeque<>(); // Stores INDICES

    for (int i = 0; i < temps.length; i++) {
        while (!stack.isEmpty() && temps[stack.peek()] < temps[i]) {
            int prevIdx = stack.pop();
            ans[prevIdx] = i - prevIdx;
        }
        stack.push(i);
    }
    return ans;
}
```
