import java.util.*;

/**
 * Custom HashMap Implementation
 * Demonstrates hash table concepts with collision handling
 */
public class CustomHashMap<K, V> implements Map<K, V> {
    private static final int DEFAULT_CAPACITY = 16;
    private static final float DEFAULT_LOAD_FACTOR = 0.75f;
    private static final int MAX_CAPACITY = 1 << 30;
    
    private Node<K, V>[] table;
    private int size;
    private int threshold;
    private final float loadFactor;
    private int modCount;
    
    @SuppressWarnings("unchecked")
    public CustomHashMap() {
        this(DEFAULT_CAPACITY, DEFAULT_LOAD_FACTOR);
    }
    
    @SuppressWarnings("unchecked")
    public CustomHashMap(int initialCapacity, float loadFactor) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("Illegal initial capacity");
        }
        if (loadFactor <= 0 || Float.isNaN(loadFactor)) {
            throw new IllegalArgumentException("Illegal load factor");
        }
        
        this.loadFactor = loadFactor;
        this.threshold = tableSizeFor(initialCapacity);
    }
    
    // Core Map methods
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public boolean isEmpty() {
        return size == 0;
    }
    
    @Override
    public boolean containsKey(Object key) {
        return getNode(hash(key), key) != null;
    }
    
    @Override
    public boolean containsValue(Object value) {
        Node<K, V>[] tab;
        V v;
        if ((tab = table) != null && size > 0) {
            for (Node<K, V> e : tab) {
                for (; e != null; e = e.next) {
                    if ((v = e.value) == value || (value != null && value.equals(v))) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    @Override
    public V get(Object key) {
        Node<K, V> e;
        return (e = getNode(hash(key), key)) == null ? null : e.value;
    }
    
    @Override
    public V put(K key, V value) {
        return putVal(hash(key), key, value, false, true);
    }
    
    @Override
    public V remove(Object key) {
        Node<K, V> e;
        return (e = removeNode(hash(key), key, null, false, true)) == null ? null : e.value;
    }
    
    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
        putMapEntries(m, true);
    }
    
    @Override
    public void clear() {
        Node<K, V>[] tab;
        modCount++;
        if ((tab = table) != null && size > 0) {
            size = 0;
            for (int i = 0; i < tab.length; ++i) {
                tab[i] = null;
            }
        }
    }
    
    @Override
    public Set<K> keySet() {
        return new KeySet();
    }
    
    @Override
    public Collection<V> values() {
        return new Values();
    }
    
    @Override
    public Set<Map.Entry<K, V>> entrySet() {
        return new EntrySet();
    }
    
    // Core implementation methods
    private V putVal(int hash, K key, V value, boolean onlyIfAbsent, boolean evict) {
        Node<K, V>[] tab;
        Node<K, V> p;
        int n, i;
        
        if ((tab = table) == null || (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        
        if ((p = tab[i = (n - 1) & hash]) == null) {
            tab[i] = newNode(hash, key, value, null);
        } else {
            Node<K, V> e;
            K k;
            if (p.hash == hash && ((k = p.key) == key || (key != null && key.equals(k)))) {
                e = p;
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
            
            if (e != null) {
                V oldValue = e.value;
                if (!onlyIfAbsent || oldValue == null) {
                    e.value = value;
                }
                return oldValue;
            }
        }
        
        if (++size > threshold) {
            resize();
        }
        
        return null;
    }
    
    private Node<K, V> getNode(int hash, Object key) {
        Node<K, V>[] tab;
        Node<K, V> first, e;
        int n;
        K k;
        
        if ((tab = table) != null && (n = tab.length) > 0 && (first = tab[(n - 1) & hash]) != null) {
            if (first.hash == hash && ((k = first.key) == key || (key != null && key.equals(k)))) {
                return first;
            }
            if ((e = first.next) != null) {
                do {
                    if (e.hash == hash && ((k = e.key) == key || (key != null && key.equals(k)))) {
                        return e;
                    }
                } while ((e = e.next) != null);
            }
        }
        return null;
    }
    
    private Node<K, V> removeNode(int hash, Object key, Object value, boolean matchValue, boolean movable) {
        Node<K, V>[] tab;
        Node<K, V> p;
        int n, index;
        
        if ((tab = table) != null && (n = tab.length) > 0 && (p = tab[index = (n - 1) & hash]) != null) {
            Node<K, V> node = null, e;
            K k;
            V v;
            
            if (p.hash == hash && ((k = p.key) == key || (key != null && key.equals(k)))) {
                node = p;
            } else if ((e = p.next) != null) {
                do {
                    if (e.hash == hash && ((k = e.key) == key || (key != null && key.equals(k)))) {
                        node = e;
                        break;
                    }
                    p = e;
                } while ((e = e.next) != null);
            }
            
            if (node != null && (!matchValue || (v = node.value) == value || (value != null && value.equals(v)))) {
                if (node == p) {
                    tab[index] = node.next;
                } else {
                    p.next = node.next;
                }
                modCount++;
                size--;
                return node;
            }
        }
        return null;
    }
    
    @SuppressWarnings("unchecked")
    private Node<K, V>[] resize() {
        Node<K, V>[] oldTab = table;
        int oldCap = (oldTab == null) ? 0 : oldTab.length;
        int oldThr = threshold;
        int newCap, newThr = 0;
        
        if (oldCap > 0) {
            if (oldCap >= MAX_CAPACITY) {
                threshold = Integer.MAX_VALUE;
                return oldTab;
            } else if ((newCap = oldCap << 1) < MAX_CAPACITY && oldCap >= DEFAULT_CAPACITY) {
                newThr = oldThr << 1;
            }
        } else if (oldThr > 0) {
            newCap = oldThr;
        } else {
            newCap = DEFAULT_CAPACITY;
            newThr = (int) (DEFAULT_LOAD_FACTOR * DEFAULT_CAPACITY);
        }
        
        if (newThr == 0) {
            float ft = (float) newCap * loadFactor;
            newThr = (newCap < MAX_CAPACITY && ft < (float) MAX_CAPACITY ? (int) ft : Integer.MAX_VALUE);
        }
        
        threshold = newThr;
        @SuppressWarnings({"rawtypes", "unchecked"})
        Node<K, V>[] newTab = (Node<K, V>[]) new Node[newCap];
        table = newTab;
        
        if (oldTab != null) {
            for (int j = 0; j < oldCap; ++j) {
                Node<K, V> e;
                if ((e = oldTab[j]) != null) {
                    oldTab[j] = null;
                    if (e.next == null) {
                        newTab[e.hash & (newCap - 1)] = e;
                    } else {
                        // Handle collision chains during resize
                        Node<K, V> loHead = null, loTail = null;
                        Node<K, V> hiHead = null, hiTail = null;
                        Node<K, V> next;
                        do {
                            next = e.next;
                            if ((e.hash & oldCap) == 0) {
                                if (loTail == null) {
                                    loHead = e;
                                } else {
                                    loTail.next = e;
                                }
                                loTail = e;
                            } else {
                                if (hiTail == null) {
                                    hiHead = e;
                                } else {
                                    hiTail.next = e;
                                }
                                hiTail = e;
                            }
                        } while ((e = next) != null);
                        
                        if (loTail != null) {
                            loTail.next = null;
                            newTab[j] = loHead;
                        }
                        if (hiTail != null) {
                            hiTail.next = null;
                            newTab[j + oldCap] = hiHead;
                        }
                    }
                }
            }
        }
        
        return newTab;
    }
    
    private void putMapEntries(Map<? extends K, ? extends V> m, boolean evict) {
        int s = m.size();
        if (s > 0) {
            if (table == null) {
                float ft = ((float) s / loadFactor) + 1.0F;
                int t = ((ft < (float) MAX_CAPACITY) ? (int) ft : MAX_CAPACITY);
                if (t > threshold) {
                    threshold = tableSizeFor(t);
                }
            } else if (s > threshold) {
                resize();
            }
            
            for (Map.Entry<? extends K, ? extends V> e : m.entrySet()) {
                K key = e.getKey();
                V value = e.getValue();
                putVal(hash(key), key, value, false, evict);
            }
        }
    }
    
    // Utility methods
    private int hash(Object key) {
        int h;
        return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
    }
    
    private static int tableSizeFor(int cap) {
        int n = cap - 1;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;
        return (n < 0) ? 1 : (n >= MAX_CAPACITY) ? MAX_CAPACITY : n + 1;
    }
    
    private Node<K, V> newNode(int hash, K key, V value, Node<K, V> next) {
        return new Node<>(hash, key, value, next);
    }
    
    // Node class for linked list collision handling
    static class Node<K, V> implements Map.Entry<K, V> {
        final int hash;
        final K key;
        V value;
        Node<K, V> next;
        
        Node(int hash, K key, V value, Node<K, V> next) {
            this.hash = hash;
            this.key = key;
            this.value = value;
            this.next = next;
        }
        
        @Override
        public K getKey() {
            return key;
        }
        
        @Override
        public V getValue() {
            return value;
        }
        
        @Override
        public V setValue(V newValue) {
            V oldValue = value;
            value = newValue;
            return oldValue;
        }
        
        @Override
        public boolean equals(Object o) {
            if (o == this) return true;
            if (o instanceof Map.Entry) {
                Map.Entry<?, ?> e = (Map.Entry<?, ?>) o;
                return Objects.equals(key, e.getKey()) && Objects.equals(value, e.getValue());
            }
            return false;
        }
        
        @Override
        public int hashCode() {
            return Objects.hashCode(key) ^ Objects.hashCode(value);
        }
        
        @Override
        public String toString() {
            return key + "=" + value;
        }
    }
    
    // Iterator implementations
    private class KeySet extends AbstractSet<K> {
        @Override
        public Iterator<K> iterator() {
            return new KeyIterator();
        }
        
        @Override
        public int size() {
            return size;
        }
        
        @Override
        public boolean contains(Object o) {
            return containsKey(o);
        }
        
        @Override
        public boolean remove(Object o) {
            return CustomHashMap.this.removeNode(hash(o), o, null, false, true) != null;
        }
        
        @Override
        public void clear() {
            CustomHashMap.this.clear();
        }
    }
    
    private class Values extends AbstractCollection<V> {
        @Override
        public Iterator<V> iterator() {
            return new ValueIterator();
        }
        
        @Override
        public int size() {
            return size;
        }
        
        @Override
        public boolean contains(Object o) {
            return containsValue(o);
        }
        
        @Override
        public void clear() {
            CustomHashMap.this.clear();
        }
    }
    
    private class EntrySet extends AbstractSet<Map.Entry<K, V>> {
        @Override
        public Iterator<Map.Entry<K, V>> iterator() {
            return new EntryIterator();
        }
        
        @Override
        public boolean contains(Object o) {
            if (!(o instanceof Map.Entry)) {
                return false;
            }
            Map.Entry<?, ?> e = (Map.Entry<?, ?>) o;
            Object key = e.getKey();
            Node<K, V> candidate = getNode(hash(key), key);
            return candidate != null && candidate.equals(e);
        }
        
        @Override
        public boolean remove(Object o) {
            if (o instanceof Map.Entry) {
                Map.Entry<?, ?> e = (Map.Entry<?, ?>) o;
                Object key = e.getKey();
                Object value = e.getValue();
                return removeNode(hash(key), key, value, true, true) != null;
            }
            return false;
        }
        
        @Override
        public int size() {
            return size;
        }
        
        @Override
        public void clear() {
            CustomHashMap.this.clear();
        }
    }
    
    // Abstract iterator class
    abstract class HashIterator {
        Node<K, V> next;
        Node<K, V> current;
        int expectedModCount;
        int index;
        
        HashIterator() {
            expectedModCount = modCount;
            Node<K, V>[] t = table;
            current = next = null;
            index = 0;
            if (t != null && size > 0) {
                do {} while (index < t.length && (next = t[index++]) == null);
            }
        }
        
        public final boolean hasNext() {
            return next != null;
        }
        
        final Node<K, V> nextNode() {
            Node<K, V>[] t;
            Node<K, V> e = next;
            if (modCount != expectedModCount) {
                throw new ConcurrentModificationException();
            }
            if (e == null) {
                throw new NoSuchElementException();
            }
            
            if ((next = (current = e).next) == null && (t = table) != null) {
                do {} while (index < t.length && (next = t[index++]) == null);
            }
            return e;
        }
        
        public final void remove() {
            Node<K, V> p = current;
            if (p == null) {
                throw new IllegalStateException();
            }
            if (modCount != expectedModCount) {
                throw new ConcurrentModificationException();
            }
            current = null;
            K key = p.key;
            removeNode(hash(key), key, null, false, false);
            expectedModCount = modCount;
        }
    }
    
    // Concrete iterator implementations
    private class KeyIterator extends HashIterator implements Iterator<K> {
        @Override
        public K next() {
            return nextNode().key;
        }
    }
    
    private class ValueIterator extends HashIterator implements Iterator<V> {
        @Override
        public V next() {
            return nextNode().value;
        }
    }
    
    private class EntryIterator extends HashIterator implements Iterator<Map.Entry<K, V>> {
        @Override
        public Map.Entry<K, V> next() {
            return nextNode();
        }
    }
    
    @Override
    public String toString() {
        if (size == 0) {
            return "{}";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        for (Map.Entry<K, V> e : entrySet()) {
            K key = e.getKey();
            V value = e.getValue();
            sb.append(key == this ? "(this Map)" : key);
            sb.append('=');
            sb.append(value == this ? "(this Map)" : value);
            sb.append(", ");
        }
        
        // Remove last ", " and add closing brace
        if (sb.length() > 2) {
            sb.setLength(sb.length() - 2);
        }
        sb.append('}');
        return sb.toString();
    }
}
