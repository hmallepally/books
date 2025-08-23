import java.util.*;

/**
 * Simplified ArrayList Implementation
 * Demonstrates core ArrayList concepts without complex edge cases
 */
public class SimpleArrayList<E> implements List<E> {
    private static final int DEFAULT_CAPACITY = 10;
    private Object[] elements;
    private int size;
    
    // Constructors
    public SimpleArrayList() {
        this(DEFAULT_CAPACITY);
    }
    
    public SimpleArrayList(int initialCapacity) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("Illegal Capacity: " + initialCapacity);
        }
        elements = new Object[initialCapacity];
        size = 0;
    }
    
    // Core methods
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public boolean isEmpty() {
        return size == 0;
    }
    
    @Override
    public boolean contains(Object o) {
        return indexOf(o) >= 0;
    }
    
    @Override
    public Iterator<E> iterator() {
        return new SimpleIterator();
    }
    
    @Override
    public Object[] toArray() {
        return Arrays.copyOf(elements, size);
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public <T> T[] toArray(T[] a) {
        if (a.length < size) {
            return (T[]) Arrays.copyOf(elements, size, a.getClass());
        }
        System.arraycopy(elements, 0, a, 0, size);
        if (a.length > size) {
            a[size] = null;
        }
        return a;
    }
    
    @Override
    public boolean add(E e) {
        ensureCapacity(size + 1);
        elements[size++] = e;
        return true;
    }
    
    @Override
    public boolean remove(Object o) {
        int index = indexOf(o);
        if (index >= 0) {
            remove(index);
            return true;
        }
        return false;
    }
    
    @Override
    public boolean addAll(Collection<? extends E> c) {
        Object[] a = c.toArray();
        int numNew = a.length;
        ensureCapacity(size + numNew);
        System.arraycopy(a, 0, elements, size, numNew);
        size += numNew;
        return numNew != 0;
    }
    
    @Override
    public boolean addAll(int index, Collection<? extends E> c) {
        rangeCheckForAdd(index);
        
        Object[] a = c.toArray();
        int numNew = a.length;
        ensureCapacity(size + numNew);
        
        int numMoved = size - index;
        if (numMoved > 0) {
            System.arraycopy(elements, index, elements, index + numNew, numMoved);
        }
        
        System.arraycopy(a, 0, elements, index, numNew);
        size += numNew;
        return numNew != 0;
    }
    
    @Override
    public boolean removeAll(Collection<?> c) {
        Objects.requireNonNull(c);
        boolean modified = false;
        Iterator<?> it = iterator();
        while (it.hasNext()) {
            if (c.contains(it.next())) {
                it.remove();
                modified = true;
            }
        }
        return modified;
    }
    
    @Override
    public boolean retainAll(Collection<?> c) {
        Objects.requireNonNull(c);
        boolean modified = false;
        Iterator<E> it = iterator();
        while (it.hasNext()) {
            if (!c.contains(it.next())) {
                it.remove();
                modified = true;
            }
        }
        return modified;
    }
    
    @Override
    public void clear() {
        for (int i = 0; i < size; i++) {
            elements[i] = null;
        }
        size = 0;
    }
    
    @Override
    public E get(int index) {
        rangeCheck(index);
        return elementData(index);
    }
    
    @Override
    public E set(int index, E element) {
        rangeCheck(index);
        E oldValue = elementData(index);
        elements[index] = element;
        return oldValue;
    }
    
    @Override
    public void add(int index, E element) {
        rangeCheckForAdd(index);
        ensureCapacity(size + 1);
        System.arraycopy(elements, index, elements, index + 1, size - index);
        elements[index] = element;
        size++;
    }
    
    @Override
    public E remove(int index) {
        rangeCheck(index);
        E oldValue = elementData(index);
        
        int numMoved = size - index - 1;
        if (numMoved > 0) {
            System.arraycopy(elements, index + 1, elements, index, numMoved);
        }
        elements[--size] = null; // Help GC
        
        return oldValue;
    }
    
    @Override
    public int indexOf(Object o) {
        if (o == null) {
            for (int i = 0; i < size; i++) {
                if (elements[i] == null) {
                    return i;
                }
            }
        } else {
            for (int i = 0; i < size; i++) {
                if (o.equals(elements[i])) {
                    return i;
                }
            }
        }
        return -1;
    }
    
    @Override
    public int lastIndexOf(Object o) {
        if (o == null) {
            for (int i = size - 1; i >= 0; i--) {
                if (elements[i] == null) {
                    return i;
                }
            }
        } else {
            for (int i = size - 1; i >= 0; i--) {
                if (o.equals(elements[i])) {
                    return i;
                }
            }
        }
        return -1;
    }
    
    @Override
    public ListIterator<E> listIterator() {
        return new SimpleListIterator(0);
    }
    
    @Override
    public ListIterator<E> listIterator(int index) {
        rangeCheckForAdd(index);
        return new SimpleListIterator(index);
    }
    
    @Override
    public List<E> subList(int fromIndex, int toIndex) {
        throw new UnsupportedOperationException("SubList not implemented in this simplified version");
    }
    
    @Override
    public boolean containsAll(Collection<?> c) {
        for (Object e : c) {
            if (!contains(e)) {
                return false;
            }
        }
        return true;
    }
    
    // Helper methods
    private void ensureCapacity(int minCapacity) {
        if (minCapacity > elements.length) {
            int newCapacity = Math.max(elements.length * 2, minCapacity);
            elements = Arrays.copyOf(elements, newCapacity);
        }
    }
    
    @SuppressWarnings("unchecked")
    E elementData(int index) {
        return (E) elements[index];
    }
    
    private void rangeCheck(int index) {
        if (index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
    }
    
    private void rangeCheckForAdd(int index) {
        if (index > size || index < 0) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
    }
    
    // Simple Iterator implementation
    private class SimpleIterator implements Iterator<E> {
        int cursor = 0;
        int lastRet = -1;
        
        @Override
        public boolean hasNext() {
            return cursor < size;
        }
        
        @Override
        public E next() {
            if (cursor >= size) {
                throw new NoSuchElementException();
            }
            lastRet = cursor;
            return elementData(cursor++);
        }
        
        @Override
        public void remove() {
            if (lastRet < 0) {
                throw new IllegalStateException();
            }
            SimpleArrayList.this.remove(lastRet);
            cursor = lastRet;
            lastRet = -1;
        }
    }
    
    // Simple ListIterator implementation
    private class SimpleListIterator extends SimpleIterator implements ListIterator<E> {
        SimpleListIterator(int index) {
            cursor = index;
        }
        
        @Override
        public boolean hasPrevious() {
            return cursor > 0;
        }
        
        @Override
        public E previous() {
            if (cursor <= 0) {
                throw new NoSuchElementException();
            }
            cursor--;
            lastRet = cursor;
            return elementData(cursor);
        }
        
        @Override
        public int nextIndex() {
            return cursor;
        }
        
        @Override
        public int previousIndex() {
            return cursor - 1;
        }
        
        @Override
        public void set(E e) {
            if (lastRet < 0) {
                throw new IllegalStateException();
            }
            SimpleArrayList.this.set(lastRet, e);
        }
        
        @Override
        public void add(E e) {
            SimpleArrayList.this.add(cursor, e);
            cursor++;
            lastRet = -1;
        }
    }
    
    @Override
    public String toString() {
        if (size == 0) {
            return "[]";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < size; i++) {
            sb.append(elements[i]);
            if (i < size - 1) {
                sb.append(", ");
            }
        }
        sb.append(']');
        return sb.toString();
    }
}
