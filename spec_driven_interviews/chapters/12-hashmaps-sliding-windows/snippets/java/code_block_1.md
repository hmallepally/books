```java
int left = 0, maxLen = 0; // <1>
for (int right = 0; right < arr.length; right++) { // <2>
    // Ingest arr[right] into window state
    while (/* window state violates invariant */) { // <3>
        // Remove arr[left] from window state
        left++; // <4>
    }
    maxLen = Math.max(maxLen, right - left + 1); // <5>
}
```
