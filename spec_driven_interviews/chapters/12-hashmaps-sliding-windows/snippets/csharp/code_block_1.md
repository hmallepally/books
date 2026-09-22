```csharp
int left = 0, maxLen = 0; // <1>
for (int right = 0; right < arr.Length; right++) { // <2>
    // Ingest arr[right] into window state
    while (false /* window state violates invariant */) { // <3>
        // Remove arr[left] from window state
        left++; // <4>
    }
    maxLen = Math.Max(maxLen, right - left + 1); // <5>
}
```