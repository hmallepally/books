import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * LRU (Least Recently Used) Cache Implementation
 * Demonstrates cache design patterns and eviction strategies
 */
public class LRUCache<K, V> {
    
    public static void main(String[] args) {
        System.out.println("=== LRU Cache Demo ===\n");
        
        // Create an LRU cache with capacity 3
        LRUCache<String, String> cache = new LRUCache<>(3);
        
        System.out.println("Adding elements to cache:");
        cache.put("A", "Apple");
        cache.put("B", "Banana");
        cache.put("C", "Cherry");
        System.out.println("Cache after adding A, B, C: " + cache);
        
        // Access an element (makes it most recently used)
        System.out.println("\nAccessing element 'A': " + cache.get("A"));
        System.out.println("Cache after accessing A: " + cache);
        
        // Add a new element (should evict least recently used)
        System.out.println("\nAdding element 'D':");
        cache.put("D", "Date");
        System.out.println("Cache after adding D: " + cache);
        
        // Access another element
        System.out.println("\nAccessing element 'B': " + cache.get("B"));
        System.out.println("Cache after accessing B: " + cache);
        
        // Add another element
        System.out.println("\nAdding element 'E':");
        cache.put("E", "Elderberry");
        System.out.println("Cache after adding E: " + cache);
        
        // Demonstrate cache statistics
        System.out.println("\n=== Cache Statistics ===");
        System.out.println("Cache size: " + cache.size());
        System.out.println("Cache capacity: " + cache.getCapacity());
        System.out.println("Hit count: " + cache.getHitCount());
        System.out.println("Miss count: " + cache.getMissCount());
        System.out.println("Hit ratio: " + String.format("%.2f%%", cache.getHitRatio() * 100));
        
        // Demonstrate cache operations
        System.out.println("\n=== Cache Operations ===");
        System.out.println("Contains 'A': " + cache.containsKey("A"));
        System.out.println("Contains 'B': " + cache.containsKey("B"));
        System.out.println("Contains 'C': " + cache.containsKey("C"));
        
        // Remove an element
        System.out.println("\nRemoving element 'B':");
        cache.remove("B");
        System.out.println("Cache after removing B: " + cache);
        
        // Clear cache
        System.out.println("\nClearing cache:");
        cache.clear();
        System.out.println("Cache after clearing: " + cache);
        
        // Performance test
        System.out.println("\n=== Performance Test ===");
        LRUCache<Integer, String> perfCache = new LRUCache<>(1000);
        
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < 10000; i++) {
            perfCache.put(i, "Value" + i);
            if (i % 1000 == 0) {
                perfCache.get(i / 2); // Access some elements
            }
        }
        long endTime = System.currentTimeMillis();
        
        System.out.println("Performance test completed in " + (endTime - startTime) + "ms");
        System.out.println("Final cache size: " + perfCache.size());
        System.out.println("Hit ratio: " + String.format("%.2f%%", perfCache.getHitRatio() * 100));
        
        // Demonstrate different eviction policies
        System.out.println("\n=== Eviction Policy Demo ===");
        LRUCache<String, Integer> policyCache = new LRUCache<>(3, EvictionPolicy.LRU);
        
        policyCache.put("X", 1);
        policyCache.put("Y", 2);
        policyCache.put("Z", 3);
        System.out.println("Initial cache: " + policyCache);
        
        // Access X to make it most recently used
        policyCache.get("X");
        System.out.println("After accessing X: " + policyCache);
        
        // Add new element - should evict Y (least recently used)
        policyCache.put("W", 4);
        System.out.println("After adding W: " + policyCache);
    }
    
    // Cache node for doubly linked list
    private static class CacheNode<K, V> {
        K key;
        V value;
        CacheNode<K, V> prev;
        CacheNode<K, V> next;
        
        CacheNode(K key, V value) {
            this.key = key;
            this.value = value;
        }
        
        @Override
        public String toString() {
            return key + "=" + value;
        }
    }
    
    // Eviction policies
    public enum EvictionPolicy {
        LRU,    // Least Recently Used
        LFU,    // Least Frequently Used
        FIFO    // First In, First Out
    }
    
    private final int capacity;
    private final Map<K, CacheNode<K, V>> cache;
    private final EvictionPolicy evictionPolicy;
    
    // Doubly linked list for LRU ordering
    private CacheNode<K, V> head;
    private CacheNode<K, V> tail;
    
    // Statistics
    private int hitCount;
    private int missCount;
    
    // Thread safety
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    
    // Constructors
    public LRUCache(int capacity) {
        this(capacity, EvictionPolicy.LRU);
    }
    
    public LRUCache(int capacity, EvictionPolicy evictionPolicy) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("Capacity must be positive");
        }
        
        this.capacity = capacity;
        this.evictionPolicy = evictionPolicy;
        this.cache = new HashMap<>(capacity);
        this.hitCount = 0;
        this.missCount = 0;
        
        // Initialize doubly linked list
        this.head = new CacheNode<>(null, null);
        this.tail = new CacheNode<>(null, null);
        head.next = tail;
        tail.prev = head;
    }
    
    /**
     * Put a key-value pair into the cache
     */
    public V put(K key, V value) {
        if (key == null) {
            throw new IllegalArgumentException("Key cannot be null");
        }
        
        lock.writeLock().lock();
        try {
            CacheNode<K, V> node = cache.get(key);
            
            if (node != null) {
                // Key already exists, update value and move to front
                V oldValue = node.value;
                node.value = value;
                moveToFront(node);
                return oldValue;
            } else {
                // New key, create new node
                node = new CacheNode<>(key, value);
                cache.put(key, node);
                addToFront(node);
                
                // Check if we need to evict
                if (cache.size() > capacity) {
                    evict();
                }
                
                return null;
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Get a value from the cache by key
     */
    public V get(K key) {
        if (key == null) {
            return null;
        }
        
        lock.readLock().lock();
        try {
            CacheNode<K, V> node = cache.get(key);
            
            if (node != null) {
                // Cache hit
                hitCount++;
                moveToFront(node);
                return node.value;
            } else {
                // Cache miss
                missCount++;
                return null;
            }
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Remove a key-value pair from the cache
     */
    public V remove(K key) {
        if (key == null) {
            return null;
        }
        
        lock.writeLock().lock();
        try {
            CacheNode<K, V> node = cache.remove(key);
            
            if (node != null) {
                removeFromList(node);
                return node.value;
            }
            
            return null;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Check if the cache contains a key
     */
    public boolean containsKey(K key) {
        lock.readLock().lock();
        try {
            return cache.containsKey(key);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get the current size of the cache
     */
    public int size() {
        lock.readLock().lock();
        try {
            return cache.size();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Check if the cache is empty
     */
    public boolean isEmpty() {
        lock.readLock().lock();
        try {
            return cache.isEmpty();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Clear all entries from the cache
     */
    public void clear() {
        lock.writeLock().lock();
        try {
            cache.clear();
            head.next = tail;
            tail.prev = head;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Get the capacity of the cache
     */
    public int getCapacity() {
        return capacity;
    }
    
    /**
     * Get the hit count
     */
    public int getHitCount() {
        return hitCount;
    }
    
    /**
     * Get the miss count
     */
    public int getMissCount() {
        return missCount;
    }
    
    /**
     * Get the hit ratio (hits / (hits + misses))
     */
    public double getHitRatio() {
        int total = hitCount + missCount;
        return total == 0 ? 0.0 : (double) hitCount / total;
    }
    
    /**
     * Get all keys in the cache
     */
    public Set<K> keySet() {
        lock.readLock().lock();
        try {
            return new HashSet<>(cache.keySet());
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get all values in the cache
     */
    public Collection<V> values() {
        lock.readLock().lock();
        try {
            List<V> values = new ArrayList<>();
            for (CacheNode<K, V> node : cache.values()) {
                values.add(node.value);
            }
            return values;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get all entries in the cache
     */
    public Set<Map.Entry<K, V>> entrySet() {
        lock.readLock().lock();
        try {
            Set<Map.Entry<K, V>> entries = new HashSet<>();
            for (CacheNode<K, V> node : cache.values()) {
                entries.add(new AbstractMap.SimpleEntry<>(node.key, node.value));
            }
            return entries;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    // Private helper methods for doubly linked list management
    
    /**
     * Add a node to the front of the list (most recently used)
     */
    private void addToFront(CacheNode<K, V> node) {
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
    }
    
    /**
     * Remove a node from the list
     */
    private void removeFromList(CacheNode<K, V> node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    /**
     * Move a node to the front of the list (most recently used)
     */
    private void moveToFront(CacheNode<K, V> node) {
        removeFromList(node);
        addToFront(node);
    }
    
    /**
     * Evict an entry based on the eviction policy
     */
    private void evict() {
        CacheNode<K, V> nodeToEvict = null;
        
        switch (evictionPolicy) {
            case LRU:
                // Remove the least recently used (tail)
                nodeToEvict = tail.prev;
                break;
                
            case FIFO:
                // Remove the first in (tail)
                nodeToEvict = tail.prev;
                break;
                
            case LFU:
                // For LFU, we'd need frequency tracking
                // For simplicity, we'll use LRU as fallback
                nodeToEvict = tail.prev;
                break;
        }
        
        if (nodeToEvict != null && nodeToEvict != head) {
            cache.remove(nodeToEvict.key);
            removeFromList(nodeToEvict);
        }
    }
    
    @Override
    public String toString() {
        lock.readLock().lock();
        try {
            if (cache.isEmpty()) {
                return "{}";
            }
            
            StringBuilder sb = new StringBuilder();
            sb.append('{');
            
            // Traverse the list in order (most recently used first)
            CacheNode<K, V> current = head.next;
            while (current != tail) {
                sb.append(current.toString());
                if (current.next != tail) {
                    sb.append(", ");
                }
                current = current.next;
            }
            
            sb.append('}');
            return sb.toString();
        } finally {
            lock.readLock().unlock();
        }
    }
}

/**
 * Thread-safe LRU Cache with additional features
 */
class ThreadSafeLRUCache<K, V> extends LRUCache<K, V> {
    
    private final ReentrantReadWriteLock.WriteLock writeLock;
    private final ReentrantReadWriteLock.ReadLock readLock;
    
    public ThreadSafeLRUCache(int capacity) {
        super(capacity);
        this.writeLock = new ReentrantReadWriteLock().writeLock();
        this.readLock = new ReentrantReadWriteLock().readLock();
    }
    
    /**
     * Put with timeout
     */
    public V putWithTimeout(K key, V value, long timeoutMs) {
        writeLock.lock();
        try {
            V result = put(key, value);
            
            // Schedule removal after timeout
            Timer timer = new Timer();
            timer.schedule(new TimerTask() {
                @Override
                public void run() {
                    remove(key);
                }
            }, timeoutMs);
            
            return result;
        } finally {
            writeLock.unlock();
        }
    }
    
    /**
     * Get with default value if key doesn't exist
     */
    public V getOrDefault(K key, V defaultValue) {
        readLock.lock();
        try {
            V value = get(key);
            return value != null ? value : defaultValue;
        } finally {
            readLock.unlock();
        }
    }
    
    /**
     * Atomic put-if-absent operation
     */
    public V putIfAbsent(K key, V value) {
        writeLock.lock();
        try {
            if (!containsKey(key)) {
                return put(key, value);
            }
            return get(key);
        } finally {
            writeLock.unlock();
        }
    }
}

/**
 * LRU Cache with statistics and monitoring
 */
class MonitoredLRUCache<K, V> extends LRUCache<K, V> {
    
    private final long creationTime;
    private long lastAccessTime;
    private long totalAccessTime;
    private int evictionCount;
    
    public MonitoredLRUCache(int capacity) {
        super(capacity);
        this.creationTime = System.currentTimeMillis();
        this.lastAccessTime = creationTime;
        this.totalAccessTime = 0;
        this.evictionCount = 0;
    }
    
    @Override
    public V get(K key) {
        long startTime = System.currentTimeMillis();
        V result = super.get(key);
        long endTime = System.currentTimeMillis();
        
        lastAccessTime = endTime;
        totalAccessTime += (endTime - startTime);
        
        return result;
    }
    
    @Override
    public V put(K key, V value) {
        long startTime = System.currentTimeMillis();
        V result = super.put(key, value);
        long endTime = System.currentTimeMillis();
        
        lastAccessTime = endTime;
        totalAccessTime += (endTime - startTime);
        
        return result;
    }
    
    /**
     * Get cache statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("size", size());
        stats.put("capacity", getCapacity());
        stats.put("hitCount", getHitCount());
        stats.put("missCount", getMissCount());
        stats.put("hitRatio", getHitRatio());
        stats.put("evictionCount", evictionCount);
        stats.put("creationTime", creationTime);
        stats.put("lastAccessTime", lastAccessTime);
        stats.put("totalAccessTime", totalAccessTime);
        stats.put("averageAccessTime", size() > 0 ? (double) totalAccessTime / (getHitCount() + getMissCount()) : 0.0);
        
        return stats;
    }
    
    /**
     * Print detailed statistics
     */
    public void printStatistics() {
        Map<String, Object> stats = getStatistics();
        System.out.println("\n=== Cache Statistics ===");
        stats.forEach((key, value) -> {
            if (key.equals("creationTime") || key.equals("lastAccessTime")) {
                System.out.printf("%s: %s%n", key, new Date((Long) value));
            } else if (key.equals("hitRatio") || key.equals("averageAccessTime")) {
                System.out.printf("%s: %.4f%n", key, (Double) value);
            } else {
                System.out.printf("%s: %s%n", key, value);
            }
        });
    }
}
