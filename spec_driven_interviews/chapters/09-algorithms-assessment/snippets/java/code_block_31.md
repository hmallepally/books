```java
TreeMap<Integer, String> map = new TreeMap<>();
map.put(10, "ten");
map.put(30, "thirty");
map.put(20, "twenty");

map.firstKey();            // Smallest key (10) — O(log N)
map.lastKey();             // Largest key (30) — O(log N)
map.floorKey(25);          // Largest key <= 25 -> returns 20
map.ceilingKey(25);        // Smallest key >= 25 -> returns 30
map.lowerKey(20);          // Largest key < 20 -> returns 10
map.higherKey(20);         // Smallest key > 20 -> returns 30
map.subMap(10, 30);        // Keys in range [10, 30)

// TreeSet — sorted unique elements
TreeSet<Integer> set = new TreeSet<>();
set.add(30); set.add(10); set.add(20);
set.first();               // 10
set.floor(25);             // 20 (largest <= 25)
set.ceiling(25);           // 30 (smallest >= 25)
```
