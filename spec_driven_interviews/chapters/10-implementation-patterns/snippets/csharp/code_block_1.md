```csharp
// Retains elements satisfying a condition, overwrites array in-place
int write = 0; // <1>
for (int read = 0; read < arr.Length; read++) { // <2>
    if (KeepCondition(arr[read])) { // <3>
        arr[write] = arr[read]; // <4>
        write++; // <5>
    }
}
// Result is arr[0..write-1], return write as the new length
```