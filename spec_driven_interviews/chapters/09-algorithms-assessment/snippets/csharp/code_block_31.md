```csharp
var map = new SortedDictionary<int, string>();
map[10] = "ten";
map[30] = "thirty";
map[20] = "twenty";

map.Keys.First();          // Smallest key (10)
map.Keys.Last();           // Largest key (30)
// C# SortedDictionary lacks floor/ceiling — use SortedSet for that

// SortedSet — sorted unique elements with range queries
var set = new SortedSet<int>();
set.Add(30); set.Add(10); set.Add(20);
set.Min;                   // 10
set.Max;                   // 30
set.GetViewBetween(10, 30); // Elements in range [10, 30]
// For floor/ceiling, use LINQ: set.Where(x => x <= 25).Last()
```
