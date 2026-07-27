```csharp
// Retains elements satisfying a condition, overwrites array in-place
int write = 0;
for (int read = 0; read < arr.Length; read++) {
    if (KeepCondition(arr[read])) {
        arr[write] = arr[read];
        write++;
    }
}
// Result is arr[0..write-1], return write as the new length
```