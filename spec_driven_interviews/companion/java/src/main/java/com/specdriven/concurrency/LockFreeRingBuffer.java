package com.specdriven.concurrency;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/**
 * High-Throughput Lock-Free Single-Producer Single-Consumer (SPSC) Ring Buffer.
 * Eliminates mutex contention using atomic VarHandle memory fences.
 * Corresponds to Chapter 08.
 */
public class LockFreeRingBuffer<T> {
    private final Object[] buffer;
    private final int capacity;
    private final int mask;

    // Cache-line padding to prevent false sharing
    private long p1, p2, p3, p4, p5, p6, p7;
    private volatile long head = 0;
    private long p8, p9, p10, p11, p12, p13, p14;
    private volatile long tail = 0;

    private static final VarHandle HEAD;
    private static final VarHandle TAIL;

    static {
        try {
            MethodHandles.Lookup l = MethodHandles.lookup();
            HEAD = l.findVarHandle(LockFreeRingBuffer.class, "head", long.class);
            TAIL = l.findVarHandle(LockFreeRingBuffer.class, "tail", long.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    public LockFreeRingBuffer(int capacityPowerOfTwo) {
        if (Integer.bitCount(capacityPowerOfTwo) != 1) {
            throw new IllegalArgumentException("Capacity must be a power of 2");
        }
        this.capacity = capacityPowerOfTwo;
        this.mask = capacityPowerOfTwo - 1;
        this.buffer = new Object[capacityPowerOfTwo];
    }

    public boolean offer(T item) {
        if (item == null) throw new NullPointerException("Null items forbidden");
        long currentTail = (long) TAIL.getOpaque(this);
        long currentHead = (long) HEAD.getAcquire(this);

        if (currentTail - currentHead >= capacity) {
            return false; // Buffer Full
        }

        buffer[(int) (currentTail & mask)] = item;
        TAIL.setRelease(this, currentTail + 1);
        return true;
    }

    @SuppressWarnings("unchecked")
    public T poll() {
        long currentHead = (long) HEAD.getOpaque(this);
        long currentTail = (long) TAIL.getAcquire(this);

        if (currentHead >= currentTail) {
            return null; // Buffer Empty
        }

        int index = (int) (currentHead & mask);
        T item = (T) buffer[index];
        buffer[index] = null; // Prevent memory leak

        HEAD.setRelease(this, currentHead + 1);
        return item;
    }
}
