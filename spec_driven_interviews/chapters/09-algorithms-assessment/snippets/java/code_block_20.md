```java
public class UnionFind {
    private int[] parent;
    private int[] rank;
    private int componentCount;

    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        componentCount = n;
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }

    public boolean union(int x, int y) {
        int rootX = find(x), rootY = find(y);
        if (rootX == rootY) return false;
        if (rank[rootX] < rank[rootY]) { int tmp = rootX; rootX = rootY; rootY = tmp; }
        parent[rootY] = rootX;
        if (rank[rootX] == rank[rootY]) rank[rootX]++;
        componentCount--;
        return true;
    }

    public int getComponentCount() { return componentCount; }
}
```
