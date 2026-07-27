```csharp
public class UnionFind
{
    private int[] parent;
    private int[] rank;

    public UnionFind(int n)
    {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    public int Find(int i)
    {
        if (parent[i] == i) return i;
        return parent[i] = Find(parent[i]); // Path compression
    }

    public bool Union(int i, int j)
    {
        int rootI = Find(i), rootJ = Find(j);
        if (rootI != rootJ)
        {
            if (rank[rootI] < rank[rootJ]) parent[rootI] = rootJ;
            else if (rank[rootI] > rank[rootJ]) parent[rootJ] = rootI;
            else { parent[rootJ] = rootI; rank[rootI]++; }
            return true;
        }
        return false; // Already connected!
    }
}
```
