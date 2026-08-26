# Designing for Performance and Concurrency

> *"High throughput is achieved not by making code run faster, but by eliminating waiting, contention, and coordination."*


## Concurrency in System Design Interviews

When interviewing for a senior staff or engineering manager role, you will inevitably face questions about system bottlenecks. Typical candidates suggest "adding a cache" or "using multi-threading." 

An interviewer wants to hear about the trade-offs of thread management and database contention. You must be able to detail the trade-offs between **Virtual Threads** (Loom) and **Reactive Programming**, explain why a database connection pool that is too large actually degrades system throughput, and articulate exactly when to use **Optimistic** vs. **Pessimistic Concurrency Control** in high-stakes financial operations.

In this chapter, we explore how AuraPay designs its ledger persistence layers to sustain high-volume transaction throughput without risking data drift or transaction races.


## Concurrency Models: Virtual Threads vs. Reactive

In Java 21+, the JVM introduces **Virtual Threads** (Project Loom). In the past, scaling web applications to handle thousands of concurrent connections required reactive frameworks (e.g., Spring WebFlux, Project Reactor). 

### The Thread-per-Request Model
Historically, web servers mapped one platform thread to one HTTP request. Since platform threads map 1-to-1 with operating system threads, they are expensive. Memory footprints (typically 1MB per thread stack) and operating system context-switching overhead capped JVM throughput at a few thousand concurrent threads.

### The Reactive Approach
Reactive programming solved this by decoupling processing execution from threads. Event loops processed chunks of data asynchronously via non-blocking callbacks. 

*   **Advantage:** Extreme scalability with very low resource utilization.
*   **Disadvantage:** Increased code complexity ("callback hell"), difficult stack traces, and complete incompatibility with standard Java threading tools like `ThreadLocal`.

### The Virtual Thread Revolution
Virtual threads are lightweight threads managed by the JVM rather than the OS. They are mounted onto a small carrier pool of platform threads. When a virtual thread blocks on I/O (e.g., executing a SQL query), the JVM unmounts the virtual thread, parking it, and assigns the carrier thread to another task.

![Thread Lifecycle and Context Switching States](visuals/thread_lifecycle.jpg){width=85%}

*   **Impact:** You can run millions of virtual threads concurrently while writing standard, synchronous, block-on-write code that is easy to read, debug, and trace.

![Virtual Threads vs Platform Threads](visuals/virtual_threads.png){width=85%}


## Application-Level Concurrency Primitives

Before leaning on database locks or distributed lock managers, distributed systems rely heavily on in-memory synchronization. In system design and coding interviews, demonstrating mastery over these primitives proves your ability to write thread-safe, high-performance execution pipelines without introducing deadlocks.

### Mutex / Synchronized
A **Mutex** (Mutual Exclusion) provides exclusive access to a critical section of code, ensuring that only one thread can execute it at a given moment. In Java, the native `synchronized` keyword provides intrinsic locking based on the object's monitor. While straightforward, it lacks flexibility. High-throughput platforms typically leverage `ReentrantLock`, which offers advanced semantics like lock timeouts, fairness policies, and interruptibility. Use a mutex when you need to execute complex state mutations across multiple variables atomically, but be wary of lock contention bottlenecking your application.

### Semaphore
A **Semaphore** acts as a bounded counting lock that controls access to a limited pool of shared resources. Instead of a binary lock, a semaphore initializes with a set number of permits. Threads invoke the `acquire()` method to claim a permit and `release()` when the resource is freed. If all permits are exhausted, subsequent threads block or fail fast. Semaphores are the standard mechanism for building bounded connection pools, bulkhead rate limiters, and throttling bursts of traffic in upstream API clients.

### Atomic Variables
When simply incrementing a metric or flipping a single state flag, standard locking incurs unnecessary context-switching overhead. **Atomic Variables** (such as `AtomicInteger`, `AtomicLong`, and `AtomicReference`) utilize low-level **Compare-And-Swap (CAS)** operations provided directly by modern CPU architectures. The CPU checks if the memory value matches the expected state; if it does, the update succeeds, otherwise it spins and retries. This pattern is foundational for lock-free accumulators, sequence generators, and high-performance metrics aggregation.

### Concurrent Collections
Wrapping a standard `HashMap` or `ArrayList` with a mutex creates immediate contention, severely degrading system throughput. Modern runtimes provide highly optimized **Concurrent Collections** designed for specific access patterns:

*   `ConcurrentHashMap` relies on fine-grained bucket-level locks or CAS operations, allowing many threads to read and write simultaneously without blocking the entire data structure.
*   `CopyOnWriteArrayList` copies the underlying array on every modification. It is heavily used in read-dominant structures, such as caching routing tables or managing event listeners.
*   `BlockingQueue` variants are essential for thread-safe producer-consumer queues, handling backpressure between asynchronous job workers.

### async/await & Non-Blocking I/O
While threads map execution to operating system resources, modern languages use cooperative multitasking to scale concurrency independently of OS threads. C#'s **async/await** and Python's **asyncio** allow developers to write sequential-looking code that does not block the underlying thread during I/O delays. Java takes a different approach: rather than async/await syntax, Java 21+ uses **Virtual Threads** (Project Loom) to achieve the same goal — blocking calls in virtual threads are automatically non-blocking at the OS level, preserving sequential code style. (Java's `CompletableFuture` provides similar capability but requires callback chaining via `.thenApply()` and `.thenCompose()`, losing the sequential readability.) When an I/O call yields, the execution returns control to an event loop or scheduler, allowing a single physical thread to manage thousands of simultaneous network requests.


## Database Locking: Optimistic vs. Pessimistic

When two concurrent transactions attempt to debit the same ledger account, we must prevent double-debiting and race conditions. This requires strict concurrency control.

### Pessimistic Concurrency Control (PCC)
Pessimistic locking assumes that a conflict is highly likely. It blocks concurrent transactions by locking the records at the database level:

```sql
SELECT * FROM accounts WHERE id = ? FOR UPDATE;
```

*   **Pros:** Guaranteed safety; concurrent transactions wait in line until the lock is released.
*   **Cons:** High lock contention, database thread starvation, and high risk of deadlocks under load.

**When to use:** When transaction frequency on a single account (e.g., a corporate merchant account) is extremely high, and you cannot afford transaction retries.

### Optimistic Concurrency Control (OCC)
Optimistic locking assumes conflicts are rare. It allows concurrent threads to read and edit records without blocking. When saving the entity, the engine verifies that the record has not been modified by checking a `version` field (`WHERE id = ? AND version = ?`).

![Optimistic vs Pessimistic Concurrency Control](visuals/occ_vs_pcc.png){width=70%}

- **Pros:** High throughput; no database locks are held while executing business logic.
- **Cons:** If a conflict occurs, one of the transactions fails, forcing the application to catch the exception and retry the entire workflow.

**When to use:** In low-to-medium contention systems where write conflicts are rare, maximizing parallel performance.


## Concurrency Control & Locking Matrix

When designing financial ledgers, selecting the right locking paradigm is critical. The following matrix contrasts the three primary concurrency control options:

| Criteria | Optimistic Locking (OCC) | Pessimistic Locking (PCC) | Distributed Locking (e.g., Redis Redlock) |
|---|---|---|---|
| **Mechanism** | Application version check (`WHERE version = ?`) | Database row-level locks (`SELECT FOR UPDATE`) | In-memory distributed key lease |
| **Complexity** | Low (handled natively by ORM/SQL) | Medium (requires managing database locks) | High (requires distributed lock manager infrastructure) |
| **Contention Cost** | Low (no blocking, fails fast) | High (blocking threads waiting for lock) | Medium (spins or rejects requests) |
| **Lock Duration** | Nanoseconds (during DB UPDATE commit) | Milliseconds (entire DB transaction block) | Leased duration (typically 5–30 seconds) |
| **Starvation Risk** | High for hot accounts (constant retries) | Low (threads queue in order) | Medium (depends on retry/backoff settings) |
| **Scale Limits** | Scales with DB capacity | Hard limit based on DB connection pool size | Scales horizontally with distributed key store |
| **Deadlock Risk** | Zero | High (requires deterministic lexicographical ordering of resources) | Medium (depends on lock lease expiration / release logic) |

![Database Deadlock Cycle — Circular Wait Conditions](visuals/deadlock_diagram.jpg){width=85%}


## Caching Patterns & Consistency Architectural Overview

In high-throughput platforms, caching offloads read traffic from primary databases. However, introducing a cache creates the classic problem of **cache invalidation**.

### Caching Architectures Summary

1. **Cache-Aside (Recommended for Ledgers):** The application queries the cache first. On a *cache hit*, data is returned immediately. On a *cache miss*, it reads from the database, populates the cache, and returns.
2. **Write-Through:** Synchronously writes to both cache and database.
3. **Write-Behind (Write-Back):** Asynchronously flushes cached writes to disk. **WARNING:** Never use Write-Behind for financial ledgers due to crash-induced data loss risks.

### Cache Invalidation & Race Conditions

When updating the database, the application must invalidate the cache key.

- **Correct Pattern:** Always **delete** the cache key after writing to the database (inside a post-commit transaction hook) rather than updating it, forcing the next read operation to perform a fresh Cache-Aside query from the source database.

> [!TIP]
> **Dedicated Caching Deep-Dive:**
> For an in-depth algorithmic treatment of LRU Cache implementation ($\mathcal{O}(1)$ get/put via Doubly-Linked List + HashMap) and distributed Redis sliding-window caching mechanisms, refer to **Chapter 13 (Optimization & Dynamic Programming)** and **Chapter 17 (Resiliency & Integration Systems)**.


## CPU Cache Locality (L1/L2/L3) in HFT Matching Loops

In ultra-low-latency matching engines (like ZenithTrade), garbage collection pauses and CPU cache misses are the primary bottlenecks. To write code that runs in the microsecond range, you must design for **cache locality**:

- **The Problem with Linked Lists:** A standard `LinkedList` contains nodes linked by memory references. These references can be scattered randomly across heap memory. When the CPU traverses a linked list to match orders, it incurs constant **L1/L2/L3 cache misses**, forcing the CPU to fetch data from physical RAM, which is up to 200 times slower than L1 cache.
- **The Array/Contiguous Layout:** To minimize cache misses, the matching loop must use contiguous memory structures. By storing orders in flat array layouts or utilizing pre-allocated object pools, the CPU can load adjacent elements into L1/L2 cache pre-emptively, accelerating execution speed.


## Connection Pool Sizing: The HikariCP Formula

A common design flaw is over-allocating database connection pool sizes. If you have 500 thread workers, candidates often set the connection pool size to 500.

**The Pitfall:** A database engine is limited by physical resources (CPU cores, disk write I/O speed, memory). When hundreds of threads attempt to execute database operations concurrently, the database server spends more time performing CPU context switches than executing queries.

HikariCP (the industry-standard connection pool manager) uses a formula derived from PostgreSQL benchmark testing to size database pools:

```
Pool Size = (Core Count * 2) + Effective Spindle Count
```

For example, a database server with 8 CPU cores and an SSD array (spindle count of 1) should have a pool size of:

```
(8 * 2) + 1 = 17 Connections
```

Setting the pool size to 17 will yield *higher* overall throughput than setting it to 100, due to the minimization of CPU context switching and disk spindle thrashing.

**Important Context:** This formula was derived empirically by the PostgreSQL community for spinning disk (HDD) workloads where 'Effective Spindle Count' represents physical disk heads. For modern NVMe SSDs and cloud-managed databases (e.g., Aurora, Cloud SQL), this formula is a starting point, not a universal law. Cloud databases often recommend pool sizes of 2-5× CPU cores. Always benchmark with your specific database engine and storage backend.

![HikariCP Connection Pool Sizing](visuals/hikaricp_formula.png){width=85%}




> ⭐ **STAR Moment: The Cache Invalidation Design**
> 
> When discussing performance during an interview, never say *"We will add a cache."* Say: *"We will implement a Cache-Aside pattern using Redis. To prevent stale reads in our double-entry ledger, we will use a transactional write-through strategy, invalidating cache keys atomically inside the database commit boundary to ensure absolute consistency."* This shows you understand caching boundaries in financial transaction systems.
