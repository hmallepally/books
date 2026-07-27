```csharp
public int NetworkDelayTime(int[][] times, int n, int k)
{
    var adj = new Dictionary<int, List<int[]>>();
    foreach (int[] t in times)
    {
        if (!adj.ContainsKey(t[0])) adj[t[0]] = new List<int[]>();
        adj[t[0]].Add(new int[] { t[1], t[2] });
    }

    var pq = new PriorityQueue<int, int>(); // [node, dist] ordered by dist
    pq.Enqueue(k, 0);
    var dist = new Dictionary<int, int>();

    while (pq.Count > 0)
    {
        pq.TryDequeue(out int node, out int d);
        
        if (dist.ContainsKey(node)) continue;
        dist[node] = d;

        if (adj.ContainsKey(node))
        {
            foreach (int[] edge in adj[node])
            {
                if (!dist.ContainsKey(edge[0]))
                {
                    pq.Enqueue(edge[0], d + edge[1]);
                }
            }
        }
    }
    return dist.Count == n ? dist.Values.Max() : -1;
}
```
