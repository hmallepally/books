```csharp
// Dictionary: Key -> Value mapping
var map = new Dictionary<string, int>();
map["apple"] = 3;                          // Insert/update — O(1)
map["apple"];                               // Lookup — O(1), throws if missing
map.GetValueOrDefault("banana", 0);         // Lookup with fallback — O(1)
map.ContainsKey("apple");                   // Key existence check — O(1)
map.Remove("apple");                        // Remove by key — O(1)
map.Keys;                                   // All keys (for iteration)
map.Values;                                 // All values

// Frequency counting pattern (extremely common)
foreach (char c in text) {
    map[c] = map.GetValueOrDefault(c, 0) + 1;
}

// HashSet: Unique element storage
var seen = new HashSet<int>();
seen.Add(42);              // Add element — O(1)
seen.Contains(42);         // Membership check — O(1)
seen.Remove(42);           // Remove element — O(1)
```
