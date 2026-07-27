```csharp
public class UnionFind {
    private int[] _parent;
    private int[] _rank;
    public int ComponentCount { get; private set; }

    public UnionFind(int n) {
        _parent = new int[n];
        _rank = new int[n];
        ComponentCount = n;
        for (int i = 0; i < n; i++) _parent[i] = i;
    }

    public int Find(int x) {
        if (_parent[x] != x) {
            _parent[x] = Find(_parent[x]);  // Path compression
        }
        return _parent[x];
    }

    public bool Union(int x, int y) {
        int rootX = Find(x), rootY = Find(y);
        if (rootX == rootY) return false;
        if (_rank[rootX] < _rank[rootY]) (rootX, rootY) = (rootY, rootX);
        _parent[rootY] = rootX;
        if (_rank[rootX] == _rank[rootY]) _rank[rootX]++;
        ComponentCount--;
        return true;
    }
}
```
