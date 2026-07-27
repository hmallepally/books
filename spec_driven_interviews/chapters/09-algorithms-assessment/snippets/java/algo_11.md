```java
public int shipWithinDays(int[] weights, int days) {
    int lo = 0, hi = 0;
    for (int w : weights) { lo = Math.max(lo, w); hi += w; }

    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (canShip(weights, days, mid)) hi = mid; // Try smaller capacity
        else lo = mid + 1;                         // Must increase capacity
    }
    return lo;
}

private boolean canShip(int[] weights, int days, int capacity) {
    int dayCount = 1, currentLoad = 0;
    for (int w : weights) {
        if (currentLoad + w > capacity) {
            dayCount++;
            currentLoad = 0;
        }
        currentLoad += w;
    }
    return dayCount <= days;
}
```
