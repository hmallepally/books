```java
public char findTheDifference(String s, String t) {
    char result = 0;
    for (char c : s.toCharArray()) result ^= c;
    for (char c : t.toCharArray()) result ^= c;
    return result; // Only the unpaired character survives
}
// Time: O(N), Space: O(1)
```
