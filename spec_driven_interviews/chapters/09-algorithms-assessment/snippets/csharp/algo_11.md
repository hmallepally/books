```csharp
public int ShipWithinDays(int[] weights, int days)
{
    int lo = 0, hi = 0;
    foreach (int w in weights)
    {
        lo = Math.Max(lo, w);
        hi += w;
    }

    while (lo < hi)
    {
        int mid = lo + (hi - lo) / 2;
        if (CanShip(weights, days, mid)) hi = mid; // Try smaller capacity
        else lo = mid + 1;                         // Must increase capacity
    }
    return lo;
}

private bool CanShip(int[] weights, int days, int capacity)
{
    int dayCount = 1, currentLoad = 0;
    foreach (int w in weights)
    {
        if (currentLoad + w > capacity)
        {
            dayCount++;
            currentLoad = 0;
        }
        currentLoad += w;
    }
    return dayCount <= days;
}
```
