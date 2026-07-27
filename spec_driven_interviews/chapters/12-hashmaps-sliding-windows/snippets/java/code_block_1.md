```java
int left = 0, maxLen = 0;
for (int right = 0; right < arr.length; right++) {
    // 1. Add arr[right] to window state
    while (/* window state violates invariant */) {
        // 2. Remove arr[left] from window state
        left++;
    }
    // 3. Update maxLen or minLen
    maxLen = Math.max(maxLen, right - left + 1);
}
```
