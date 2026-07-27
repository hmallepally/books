import os
import re

# FIX 1
f1_path = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\15-mock-assessment-sets\base.md'
with open(f1_path, 'r', encoding='utf-8') as f:
    content1 = f.read()

# Replace Set 8 Q4
set8_replacement = r"* **Q4 (Hard): Minimum Spanning Tree**\n  * *Specification:* Given a weighted undirected graph, find the MST weight using Kruskal's algorithm with Union-Find.\n  * *Sample Test Case:* Input: `edges -> weight`\n  * *Constraints:* V \le 10^4, E \le 5 \times 10^4.\n  * *Hint:* [PAT-17] Disjoint Set Union + greedy edge sorting."
content1 = re.sub(
    r"## Set 8: Timed Mock Assessment 8(.*?)(\* \*\*Q4 \(Hard\): Union Find Network\*\*\\n  \* \*Specification:\* Find the redundant connection in a graph that should be a tree\.\\n  \* \*Sample Test Case:\* Input: `\[\[1,2\],\[1,3\],\[2,3\]\] -> \[2,3\]`\\n  \* \*Constraints:\* Complexity bounds requiring optimal solution\.\\n  \* \*Hint:\* \[PAT-17\] Disjoint Set Union)",
    lambda m: "## Set 8: Timed Mock Assessment 8" + m.group(1) + set8_replacement.replace('\\n', '\\\\n').replace('\\', '\\\\'),
    content1,
    flags=re.DOTALL
)

# Replace Set 14 Q4
set14_replacement = r"* **Q4 (Hard): Course Schedule III**\n  * *Specification:* Given N courses with (duration, deadline), maximize courses completed.\n  * *Sample Test Case:* Input: `courses -> max`\n  * *Constraints:* N \le 10^4.\n  * *Hint:* [PAT-25] Priority Queue / Greedy with heap."
content1 = re.sub(
    r"## Set 14: Timed Mock Assessment 14(.*?)(\* \*\*Q4 \(Hard\): Topological Sort Complex\*\*\\n  \* \*Specification:\* Find the longest path in a Directed Acyclic Graph representing tasks\.\\n  \* \*Sample Test Case:\* Input: `tasks -> 10 days`\\n  \* \*Constraints:\* Complexity bounds requiring optimal solution\.\\n  \* \*Hint:\* \[PAT-16\] Topo Sort / DP)",
    lambda m: "## Set 14: Timed Mock Assessment 14" + m.group(1) + set14_replacement.replace('\\n', '\\\\n').replace('\\', '\\\\'),
    content1,
    flags=re.DOTALL
)

# Replace Set 20 Q4
set20_replacement = r"* **Q4 (Hard): Alien Dictionary**\n  * *Specification:* Given sorted alien words, derive character ordering.\n  * *Sample Test Case:* Input: `words -> ordering`\n  * *Constraints:* words \le 300, word length \le 100.\n  * *Hint:* Topological Sort on character graph."
content1 = re.sub(
    r"## Set 20: Timed Mock Assessment 20(.*?)(\* \*\*Q4 \(Hard\): Dijkstra Shortest\*\*\\n  \* \*Specification:\* Find network delay time for a signal to reach all nodes\.\\n  \* \*Sample Test Case:\* Input: `nodes=4, edges -> 2`\\n  \* \*Constraints:\* Complexity bounds requiring optimal solution\.\\n  \* \*Hint:\* \[PAT-18\] Dijkstra Priority Queue)",
    lambda m: "## Set 20: Timed Mock Assessment 20" + m.group(1) + set20_replacement.replace('\\n', '\\\\n').replace('\\', '\\\\'),
    content1,
    flags=re.DOTALL
)

with open(f1_path, 'w', encoding='utf-8') as f:
    f.write(content1)

# FIX 2
f2_path = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\24-references\base.md'
with open(f2_path, 'r', encoding='utf-8') as f:
    content2 = f.read()

new_refs = """
Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.03762

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33. https://arxiv.org/abs/2005.11401

Elhemaly, M., Gallagher, N., Tang, B., et al. (2022). Amazon DynamoDB: A Scalable, Predictably Performant, and Fully Managed NoSQL Database Service. *Proceedings of USENIX ATC '22*.

Forsgren, N., Humble, J., & Kim, G. (2018). *Accelerate: The Science of Lean Software and DevOps*. IT Revolution.

Burns, B., Beda, J., Hightower, K., & Evenson, L. (2022). *Kubernetes: Up and Running* (3rd ed.). O'Reilly.
"""
with open(f2_path, 'w', encoding='utf-8') as f:
    f.write(content2 + new_refs)

# FIX 3
f3_path = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\23-appendix\base.md'
with open(f3_path, 'r', encoding='utf-8') as f:
    content3 = f.read()

table_to_replace = """| Operation | Time (ns) | Time (Human Scale) |
|---|---|---|
| **L1 Cache reference** | 0.5 ns | 0.5 sec |
| **Branch mispredict** | 5 ns | 5 sec |
| **L2 Cache reference** | 7 ns | 7 sec |
| **Main Memory reference (RAM)** | 100 ns | 1.6 min |
| **Compress 1K bytes with Zippy** | 3,000 ns | 50 min |
| **Send 2K bytes over 1 Gbps network** | 20,000 ns | 5.5 hours |
| **Read 1MB sequentially from SSD** | 1,000,000 ns | 11.5 days |
| **Round trip within same datacenter** | 500,000 ns | 5.7 days |
| **Read 1MB sequentially from Disk** | 20,000,000 ns | 7.5 months |
| **Send packet CA to Netherlands to CA** | 150,000,000 ns | 4.7 years |"""

new_table = """| Operation | Time | Time (Human Scale) |
|---|---|---|
| **L1 Cache reference** | 1 ns | 1 sec |
| **Branch mispredict** | 5 ns | 5 sec |
| **L2 Cache reference** | 4 ns | 4 sec |
| **Main Memory reference (DDR5)** | 50 ns | 50 sec |
| **Compress 1K bytes with Zippy** | 3,000 ns | 50 min |
| **Send 2K bytes over 1 Gbps network** | 20,000 ns | 5.5 hours |
| **NVMe SSD random read** | 10-20 μs | ~3-6 hours |
| **NVMe SSD sequential 1MB read** | 100-200 μs | ~1-2 days |
| **Round trip within same datacenter** | 250-500 μs | ~3-6 days |
| **HDD seek** | 2-5 ms | ~1-2 months |
| **Read 1MB sequentially from Disk** | 20,000,000 ns | 7.5 months |
| **Send packet CA to Netherlands to CA** | 150,000,000 ns | 4.7 years |

These numbers reflect 2024 NVMe Gen4/5 SSDs and DDR5 RAM. Original latency numbers by Jeff Dean (2012) have been updated. Cloud VM performance may vary based on instance type and IO throttling."""

content3 = content3.replace(table_to_replace, new_table)
with open(f3_path, 'w', encoding='utf-8') as f:
    f.write(content3)

# FIX 4
f4_path = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\16-system-architecture\base.md'
with open(f4_path, 'r', encoding='utf-8') as f:
    content4 = f.read()

modern_infra_section = """
## Modern Infrastructure Patterns (2024+)

Modern system design interviews increasingly expect familiarity with container orchestration and cloud-native patterns:

**Kubernetes Pod Autoscaling:** Horizontal Pod Autoscaler (HPA) scales replicas based on CPU/memory or custom metrics. For AuraPay's payment gateway, HPA with target CPU utilization of 70% ensures elastic scaling during Black Friday traffic spikes.

**Sidecar Proxy Pattern (Envoy/Istio):** Instead of application-level circuit breakers (like Resilience4j), modern architectures delegate traffic management to sidecar proxies. Each microservice pod gets an Envoy sidecar that handles circuit breaking, retry budgets, and mutual TLS — without any application code changes.

**Observability with eBPF:** Extended Berkeley Packet Filter enables kernel-level observability without code instrumentation. Tools like Cilium and Pixie capture request latencies, error rates, and network flows at the kernel level, providing distributed tracing with zero application overhead.

**Serverless Trade-offs:** Lambda/Cloud Functions eliminate infrastructure management but introduce cold start latency (100ms-2s), vendor lock-in, and debugging complexity. Use for event-driven workloads (image processing, webhook handling), not for latency-critical paths.

> ⭐ **STAR Moment: Bounded Context Isolation**"""

content4 = content4.replace("> ⭐ **STAR Moment: Bounded Context Isolation**", modern_infra_section)
with open(f4_path, 'w', encoding='utf-8') as f:
    f.write(content4)

print("Done")
