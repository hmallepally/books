```java
public String[] allLongestStrings(String[] inputArray) {
    // Pass 1: Find the maximum length
    int maxLength = 0;
    for (String s : inputArray) {
        if (s.length() > maxLength) {
            maxLength = s.length();
        }
    }

    // Pass 2: Collect strings matching the max length
    List<String> result = new ArrayList<>();
    for (String s : inputArray) {
        if (s.length() == maxLength) {
            result.add(s);
        }
    }

    return result.toArray(new String[0]);
}
// Time: O(N), Space: O(N) for output
```
