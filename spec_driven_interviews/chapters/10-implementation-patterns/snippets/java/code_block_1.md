```java
// Retains elements satisfying a condition, overwrites array in-place
int write = 0;
for (int read = 0; read < arr.length; read++) {
    if (keepCondition(arr[read])) {
        arr[write] = arr[read];
        write++;
    }
}
// Result is arr[0..write-1], return write as the new length
```
