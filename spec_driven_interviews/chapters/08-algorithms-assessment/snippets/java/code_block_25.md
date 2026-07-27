```java
// HashMap: Key -> Value mapping
Map<String, Integer> map = new HashMap<>();
map.put("apple", 3);                      // Insert/update — O(1)
map.get("apple");                          // Lookup — O(1), returns null if missing
map.getOrDefault("banana", 0);             // Lookup with fallback — O(1)
map.containsKey("apple");                  // Key existence check — O(1)
map.remove("apple");                       // Remove by key — O(1)
map.keySet();                              // All keys (for iteration)
map.values();                              // All values
map.entrySet();                            // All key-value pairs

// Frequency counting pattern (extremely common)
for (char c : text.toCharArray()) {
    map.merge(c, 1, Integer::sum);         // Increment count elegantly
}

// HashSet: Unique element storage
Set<Integer> seen = new HashSet<>();
seen.add(42);              // Add element — O(1)
seen.contains(42);         // Membership check — O(1)
seen.remove(42);           // Remove element — O(1)
```
