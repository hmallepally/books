```java
public String[] transformWords(String[] words) {
    if (words == null) return new String[0];
    String[] result = new String[words.length];

    for (int i = 0; i < words.length; i++) {
        if (words[i].length() % 2 != 0) {
            result[i] = words[i].toUpperCase();
        } else {
            result[i] = new StringBuilder(words[i]).reverse().toString();
        }
    }

    return result;
}
// Time: O(N * K) where K is average word length, Space: O(N * K) for output
```
