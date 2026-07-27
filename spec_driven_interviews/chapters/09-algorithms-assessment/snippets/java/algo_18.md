```java
public int networkDelayTime(int[][] times, int n, int k) {
    Map<Integer, List<int[]>> adj = new HashMap<>();
    for (int[] t : times) {
        adj.computeIfAbsent(t[0], x -> new ArrayList<>()).add(new int[]{t[1], t[2]});
    }

    var pq = new PriorityQueue<int[]>((a, b) -> a[1] - b[1]); // [node, dist]
    pq.offer(new int[]{k, 0});
    var dist = new HashMap<Integer, Integer>();

    while (!pq.isEmpty()) {
        int[] curr = pq.poll();
        int node = curr[0], d = curr[1];
        if (dist.containsKey(node)) continue;
        dist.put(node, d);

        if (adj.containsKey(node)) {
            for (int[] edge : adj.get(node)) {
                if (!dist.containsKey(edge[0])) {
                    pq.offer(new int[]{edge[0], d + edge[1]});
                }
            }
        }
    }
    return dist.size() == n ? dist.values().stream().max(Integer::compare).get() : -1;
}
```
