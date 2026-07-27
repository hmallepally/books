## Memory Architecture: Thread Stack, Managed Heap, Metaspace, and GC Lifecycle

In enterprise Java systems (such as financial ledgers and trade matching engines), performance optimization requires a precise understanding of how the Java Virtual Machine (JVM) manages memory. High object allocation rates lead to frequent Garbage Collection (GC) pauses, cache misses, and latency spikes.

### The JVM Memory Regions

The JVM divides memory into distinct regions, broadly categorized into thread-private memory (Stack) and shared memory (Heap and Metaspace).

![JVM Memory Architecture Layout](visuals/jvm_memory_layout.png){width=85%}

#### 1. The Thread Stack (Stack Memory)
- **Scope:** Thread-private. Every thread (platform thread or virtual thread) has its own dedicated execution stack.
- **Contents:** Primitive local variables (e.g., `int`, `double`), method parameters, and **object references** (pointers pointing to objects on the Heap).
- **Behavior:** Operates strictly on a Last-In, First-Out (LIFO) stack frame structure. When a method is called, a new stack frame is pushed; when the method returns, the frame is popped.
- **Garbage Collection:** Stack memory is never garbage collected. Allocation and deallocation are instantaneous as stack frames move.

#### 2. The JVM Heap (Heap Memory)
- **Scope:** Shared across all threads in the JVM process.
- **Contents:** All object instances (e.g., `new LedgerAccount()`), arrays, and instance fields of objects.
- **Garbage Collection:** Managed entirely by the automatic Garbage Collector.

#### 3. Metaspace (Native Memory)
- **Scope:** Shared across all threads. Introduced in Java 8 (replacing legacy `PermGen`).
- **Contents:** Class metadata, method bytecodes, the runtime constant pool, and static variables.
- **Memory Source:** Metaspace is allocated out of native OS memory (off-heap RAM), meaning its size is not limited by `-Xmx` (max heap size), though it can be bounded via `-XX:MaxMetaspaceSize`.

---

### Object Storage: What Goes Where?

A common interview question asks candidates to trace where specific variables reside in memory. The following rules govern object placement:

| Variable / Element Type | Memory Location | Explanation |
|---|---|---|
| **Local Primitive** (`int x = 5` inside a method) | **Thread Stack Frame** | Stored directly on the stack frame of the executing thread. |
| **Local Reference Pointer** (`Account acc = new Account()`) | **Thread Stack Frame** | The reference variable `acc` (a 64-bit pointer) lives on the Stack; the actual `Account` object instance lives on the Heap. |
| **Instance Primitive Field** (`private int age` inside `User` class) | **JVM Heap** | Primitive fields declared inside an object instance are stored *inside* the object layout on the Heap. |
| **Instance Reference Field** (`private String name` inside `User`) | **JVM Heap** | The reference field pointer AND the underlying `String` object live on the Heap. |
| **Static Variable** (`public static final int MAX_LIMIT = 100`) | **Metaspace / Class Metadata** | Associated with the class definition in Metaspace. |

---

### The Heap Generations & Object Promotion Lifecycle

To optimize Garbage Collection efficiency, the HotSpot JVM divides the Heap into two main generations based on the **Weak Generational Hypothesis**: *most objects die young (shortly after allocation).*

![JVM Heap Generation Promotion Lifecycle](visuals/jvm_generations.png){width=85%}

#### 1. The Young Generation
The Young Generation is dedicated to newly allocated objects and is divided into three spaces:
- **Eden Space:** The initial landing pad where 99% of new objects are instantiated.
- **Survivor Spaces ($S_0$ / $S_1$ or "From" / "To"):** Two equal-sized spaces used during Minor GC to age surviving objects.

#### 2. The Tenured (Old) Generation
Stores long-lived objects that have survived multiple Minor GC cycles (e.g., Spring singletons, connection pools, long-term domain caches).

---

### The Object Promotion Walkthrough (Step-by-Step)

1. **Instantiation:** When code executes `new Transaction()`, the object is allocated in the **Eden Space**.
2. **Minor GC Triggered:** When Eden fills up, a **Minor GC** occurs. The JVM stops application threads briefly (Stop-The-World pause).
3. **Survivor Move ($S_0$):** Live objects in Eden are copied to $S_0$ (Survivor 0). Dead objects in Eden are abandoned. Eden is wiped clean. The surviving object receives an age counter of `1`.
4. **Survivor Ping-Pong ($S_0 \rightarrow S_1$):** On the next Minor GC, live objects in Eden AND $S_0$ are copied to $S_1$ (Survivor 1). $S_0$ is cleared. The age counter increments to `2`. The roles of $S_0$ and $S_1$ swap.
5. **Promotion to Tenured (Old Gen):** When an object's age counter reaches the **Tenuring Threshold** (default `-XX:MaxTenuringThreshold=15` in HotSpot), the object is promoted to the **Tenured (Old) Generation**.
6. **Pretenure Bypass:** Exceptionally large objects (e.g., massive byte arrays exceeding `-XX:PretenureSizeThreshold`) bypass the Young Generation entirely and are allocated directly in the Old Generation to prevent expensive copying across Survivor spaces.

---

### Impact on High-Performance Systems

- **Minor GC vs. Major/Full GC:** Minor GCs clear the Young Gen in sub-milliseconds. Major/Full GCs inspect the Old Gen and Metaspace, causing longer STW pauses that degrade real-time throughput.
- **Zero-Allocation Programming:** In high-frequency matching engines (ZenithTrade), developers pre-allocate reusable object pools to achieve zero allocations in the hot path, preventing Eden from filling up and completely eliminating Minor GC pauses.
