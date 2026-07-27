```java
public int firstUniqueChar(String s) {
    int[] counts = new int[256];
    for (int i = 0; i < s.length(); i++) {
        counts[s.charAt(i)]++;
    }
    for (int i = 0; i < s.length(); i++) {
        if (counts[s.charAt(i)] == 1) return i;
    }
    return -1;
}
```
