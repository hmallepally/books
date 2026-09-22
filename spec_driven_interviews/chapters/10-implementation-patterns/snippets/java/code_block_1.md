```java
// Retains elements satisfying a condition, overwrites array in-place
int write = 0; // <1>
for (int read = 0; read < arr.length; read++) { // <2>
    if (keepCondition(arr[read])) { // <3>
        arr[write] = arr[read]; // <4>
        write++; // <5>
    }
}
// Result is arr[0..write-1], return write as the new length
```
