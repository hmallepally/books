# 🚀 **Java Collection Implementations - Complete Guide**

## **Overview**

This document provides a comprehensive guide to four essential Java collection implementations that every developer should understand for interviews and real-world applications:

1. **Custom ArrayList Implementation** 📚
2. **Custom HashMap Implementation** 🗺️
3. **Filtering Iterator with Predicates** 🔍
4. **LRU Cache Implementation** 💾

---

## **1. 📚 Custom ArrayList Implementation**

### **Key Concepts**

- **Dynamic Resizing**: Automatically grows when capacity is exceeded
- **Array Backing**: Uses `Object[]` for storage with type safety via generics
- **Bounds Checking**: Validates all index operations
- **Iterator Support**: Implements both `Iterator` and `ListIterator`
- **Fail-Fast Behavior**: Detects concurrent modifications

### **Core Implementation Details**

```java
public class SimpleArrayList<E> implements List<E> {
    private Object[] elements;  // Backing array
    private int size;           // Current number of elements
    
    // Dynamic resizing: doubles capacity when full
    private void ensureCapacity(int minCapacity) {
        if (minCapacity > elements.length) {
            int newCapacity = Math.max(elements.length * 2, minCapacity);
            elements = Arrays.copyOf(elements, newCapacity);
        }
    }
}
```

### **Key Methods Implementation**

- **`add(E element)`**: O(1) amortized, O(n) worst case (resizing)
- **`add(int index, E element)`**: O(n) due to array shifting
- **`remove(int index)`**: O(n) due to array shifting
- **`get(int index)`**: O(1) - direct array access
- **`set(int index, E element)`**: O(1) - direct array access

### **Interview Tips**

1. **Always mention amortized complexity** for add operations
2. **Explain the resizing strategy** (1.5x or 2x growth)
3. **Discuss trade-offs**: Fast access vs. slow insertion/deletion
4. **Mention `System.arraycopy()`** for efficient shifting
5. **Explain fail-fast iterators** and `ConcurrentModificationException`

---

## **2. 🗺️ Custom HashMap Implementation**

### **Key Concepts**

- **Hash Table**: Array of linked lists (buckets)
- **Collision Handling**: Separate chaining with linked lists
- **Load Factor**: Triggers resizing when 75% full
- **Hash Function**: `Objects.hashCode()` with bit manipulation
- **Power of 2 Capacity**: Efficient modulo with `&` operator

### **Core Implementation Details**

```java
public class CustomHashMap<K, V> implements Map<K, V> {
    private Node<K, V>[] table;     // Hash table array
    private int size;                // Number of key-value pairs
    private float loadFactor;        // Resize threshold
    private int threshold;           // Current resize threshold
    
    // Hash function with bit manipulation
    private int hash(Object key) {
        int h;
        return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
    }
}
```

### **Collision Handling Strategy**

```java
// Separate chaining with linked lists
if (p.hash == hash && ((k = p.key) == key || (key != null && key.equals(k)))) {
    e = p;  // Key exists, update value
} else {
    // Handle collision - add to linked list
    for (int binCount = 0; ; ++binCount) {
        if ((e = p.next) == null) {
            p.next = newNode(hash, key, value, null);
            break;
        }
        if (e.hash == hash && ((k = e.key) == key || (key != null && key.equals(k)))) {
            break;
        }
        p = e;
    }
}
```

### **Key Methods Implementation**

- **`put(K key, V value)`**: O(1) average, O(n) worst case (collision chain)
- **`get(K key)`**: O(1) average, O(n) worst case (collision chain)
- **`remove(K key)`**: O(1) average, O(n) worst case (collision chain)
- **`containsKey(K key)`**: O(1) average, O(n) worst case

### **Interview Tips**

1. **Explain the hash function** and why we use `^ (h >>> 16)`
2. **Discuss collision handling** strategies (separate chaining vs. open addressing)
3. **Mention load factor** and resizing strategy
4. **Explain why capacity is power of 2** (efficient modulo)
5. **Discuss hash table vs. tree** (Java 8+ optimization)

---

## **3. 🔍 Filtering Iterator with Predicates**

### **Key Concepts**

- **Functional Programming**: Uses `Predicate<T>` for filtering
- **Lazy Evaluation**: Filters are applied only when iterating
- **Iterator Pattern**: Implements standard Java `Iterator` interface
- **Chaining**: Can combine multiple filters
- **Generic Design**: Works with any data type

### **Core Implementation Details**

```java
public class FilteredIterator<T> implements Iterator<T> {
    private final Iterator<T> sourceIterator;
    private final Predicate<T> predicate;
    private T nextElement;
    private boolean hasNextElement;
    
    // Find next element matching predicate
    private void findNext() {
        hasNextElement = false;
        while (sourceIterator.hasNext()) {
            T element = sourceIterator.next();
            if (predicate.test(element)) {
                nextElement = element;
                hasNextElement = true;
                break;
            }
        }
    }
}
```

### **Advanced Features**

```java
// Multiple predicates with AND/OR logic
public class AdvancedFilteredIterator<T> implements Iterator<T> {
    private final List<Predicate<T>> predicates;
    private final boolean requireAll; // true = AND, false = OR
    
    private boolean matchesPredicates(T element) {
        if (requireAll) {
            return predicates.stream().allMatch(p -> p.test(element));
        } else {
            return predicates.stream().anyMatch(p -> p.test(element));
        }
    }
}
```

### **Usage Examples**

```java
// Basic filtering
FilteredIterator<Integer> evenIterator = new FilteredIterator<>(
    numbers.iterator(), n -> n % 2 == 0);

// Chained filters
FilteredIterator<Person> youngEngineers = new FilteredIterator<>(
    new FilteredIterator<>(people.iterator(), p -> "Engineer".equals(p.getProfession())),
    p -> p.getAge() < 30
);

// Utility filters
FilteredIterator<String> longWords = new FilteredIterator<>(
    words.iterator(), Filters.longerThan(5));
```

### **Interview Tips**

1. **Explain lazy evaluation** benefits (memory efficiency)
2. **Discuss functional programming** concepts
3. **Show how to chain filters** for complex queries
4. **Mention performance implications** of multiple predicates
5. **Demonstrate custom predicate creation**

---

## **4. 💾 LRU Cache Implementation**

### **Key Concepts**

- **Least Recently Used**: Evicts oldest accessed elements
- **Doubly Linked List**: Maintains access order
- **HashMap**: O(1) key-value lookups
- **Thread Safety**: Uses `ReentrantReadWriteLock`
- **Statistics Tracking**: Hit/miss ratios and performance metrics

### **Core Implementation Details**

```java
public class LRUCache<K, V> {
    private final Map<K, CacheNode<K, V>> cache;  // Fast lookups
    private final int capacity;                    // Maximum size
    private CacheNode<K, V> head, tail;           // Doubly linked list
    
    // Cache node for linked list
    private static class CacheNode<K, V> {
        K key;
        V value;
        CacheNode<K, V> prev, next;
    }
}
```

### **LRU Algorithm**

```java
// Move accessed element to front (most recently used)
private void moveToFront(CacheNode<K, V> node) {
    removeFromList(node);
    addToFront(node);
}

// Evict least recently used (tail of list)
private void evict() {
    CacheNode<K, V> nodeToEvict = tail.prev;
    if (nodeToEvict != null && nodeToEvict != head) {
        cache.remove(nodeToEvict.key);
        removeFromList(nodeToEvict);
    }
}
```

### **Thread Safety Implementation**

```java
public V get(K key) {
    lock.readLock().lock();
    try {
        CacheNode<K, V> node = cache.get(key);
        if (node != null) {
            hitCount++;
            moveToFront(node);
            return node.value;
        } else {
            missCount++;
            return null;
        }
    } finally {
        lock.readLock().unlock();
    }
}
```

### **Key Methods Implementation**

- **`put(K key, V value)`**: O(1) - hash map + linked list operations
- **`get(K key)`**: O(1) - hash map lookup + linked list move
- **`remove(K key)`**: O(1) - hash map + linked list operations
- **`evict()`**: O(1) - remove from tail of linked list

### **Interview Tips**

1. **Explain the LRU algorithm** and why it's useful
2. **Discuss the data structure choice** (HashMap + Doubly Linked List)
3. **Mention thread safety** considerations
4. **Explain eviction policies** and alternatives (LFU, FIFO)
5. **Show how to implement statistics** and monitoring

---

## **🎯 Interview Questions & Answers**

### **ArrayList Implementation**

**Q: How would you implement a dynamic array?**
A: Use an array with automatic resizing. When full, create a new larger array (typically 1.5x or 2x size), copy existing elements using `System.arraycopy()`, and update the reference.

**Q: What's the time complexity of ArrayList operations?**
A: `get/set`: O(1), `add/remove`: O(1) amortized (O(n) worst case due to resizing), `add/remove at index`: O(n) due to shifting.

**Q: How do you handle concurrent modifications?**
A: Use `modCount` to detect changes during iteration and throw `ConcurrentModificationException` for fail-fast behavior.

### **HashMap Implementation**

**Q: How do you handle hash collisions?**
A: Use separate chaining with linked lists. When multiple keys hash to the same bucket, store them as a linked list. For Java 8+, convert to red-black tree if chain length exceeds 8.

**Q: What's the load factor and why is it important?**
A: Load factor (default 0.75) determines when to resize. When size/capacity exceeds load factor, resize to maintain O(1) performance by reducing collision chains.

**Q: Why use power of 2 for capacity?**
A: Enables efficient modulo operation using bitwise AND: `hash % capacity` becomes `hash & (capacity - 1)`, which is much faster.

### **Iterator Implementation**

**Q: How do you implement a custom iterator?**
A: Implement `Iterator<T>` interface with `hasNext()`, `next()`, and optionally `remove()`. Use internal state to track current position and handle concurrent modifications.

**Q: What are the benefits of lazy evaluation?**
A: Memory efficiency - filters aren't applied until needed, allows infinite sequences, and enables pipeline processing of large datasets.

### **Cache Implementation**

**Q: How do you implement an LRU cache?**
A: Use HashMap for O(1) lookups and doubly linked list to maintain access order. Move accessed elements to front, evict from tail when capacity exceeded.

**Q: How do you make a cache thread-safe?**
A: Use `ReentrantReadWriteLock` for concurrent reads and exclusive writes, or `ConcurrentHashMap` with atomic operations for better performance.

---

## **🚀 Performance Characteristics**

| **Operation** | **ArrayList** | **HashMap** | **LRU Cache** | **Filtered Iterator** |
|---------------|---------------|-------------|----------------|----------------------|
| **Access** | O(1) | O(1) avg | O(1) | O(1) per element |
| **Insertion** | O(1) amortized | O(1) avg | O(1) | N/A |
| **Deletion** | O(n) | O(1) avg | O(1) | N/A |
| **Search** | O(n) | O(1) avg | O(1) | O(n) |
| **Memory** | O(n) | O(n) | O(n) | O(1) |

---

## **🔧 Testing & Validation**

### **Compilation**
```bash
javac code/*.java
```

### **Running Tests**
```bash
# Individual implementations
java -cp code LRUCache
java -cp code FilteringIterator

# Comprehensive test
java -cp code CollectionTestRunner
```

### **Expected Output**
- All implementations should compile without errors
- Tests should demonstrate core functionality
- Performance metrics should be reasonable
- No memory leaks or crashes

---

## **📚 Further Reading & Resources**

1. **Java Collections Framework**: Official documentation
2. **Design Patterns**: Gang of Four patterns
3. **Concurrent Programming**: Java concurrency utilities
4. **Performance Tuning**: JVM optimization techniques
5. **System Design**: Cache design patterns

---

## **🎉 Conclusion**

These four implementations demonstrate essential Java programming concepts:

- **Data Structures**: Arrays, linked lists, hash tables
- **Algorithms**: Hashing, caching, iteration
- **Design Patterns**: Iterator, Factory, Strategy
- **Performance Optimization**: Memory management, complexity analysis
- **Thread Safety**: Concurrency control, synchronization

Mastering these implementations will significantly improve your Java skills and interview performance. Practice implementing them from scratch and experiment with different approaches and optimizations.

**Happy Coding! 🚀**

