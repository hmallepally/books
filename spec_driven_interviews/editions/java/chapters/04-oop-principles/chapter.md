# Principles of Object-Oriented Design & Domain-Driven Craftsmanship

> *"Do not expose your state to the world. Encapsulate your data, expose your contracts, and let polymorphism handle the variance."*

## The Foundations: Connecting OOP Principles to Domain-Driven Design (DDD)

In enterprise software engineering and senior-level technical interviews, Object-Oriented Programming (OOP) is not merely about syntax or class hierarchies. Its primary purpose is to model real-world business domains, enforce critical invariants, and protect data integrity under high concurrency.

When designing large-scale enterprise systems, core OOP principles map directly to **Domain-Driven Design (DDD)** tactical patterns. Understanding this bridge prevents code from degenerating into unmaintainable scripts:

![Figure 4.1: The OOP to DDD Architectural Bridge](visuals/oop_to_ddd_bridge.png){width=90%}

### Core DDD Definitions Every Candidate Must Master:

1. **Entities:** Objects defined by a unique, enduring identity that persists across state changes (e.g., a `LedgerAccount` identified by a unique `accountId`). Two entities with identical balances are distinct if their IDs differ.
2. **Value Objects:** Immutable objects defined entirely by their attribute values, possessing no conceptual identity (e.g., `Money`, `Currency`, or `Address`). If two `Money` objects both represent `\$100 USD`, they are completely interchangeable.
3. **Aggregates & Aggregate Roots:** A cluster of associated domain objects (Entities and Value Objects) treated as a single unit for data changes. The **Aggregate Root** is the sole gateway through which external code interacts with internal objects, guaranteeing that all domain invariants remain valid across operations.
4. **Domain Services:** Operations or business transformations that do not naturally belong to a single Entity or Value Object (e.g., cross-account fund routing engines).

## The Anemic Domain Model Anti-Pattern

Despite understanding basic OOP syntax, many enterprise applications fall into a common architectural trap: treating domain classes as passive data holders—simple bags of private fields with auto-generated getters and setters. Martin Fowler termed this the **Anemic Domain Model** anti-pattern.

When domain models are anemic, business logic escapes into external, stateless service classes (e.g., `LedgerService`). The service pulls raw data out of the domain object, validates it externally, mutates the fields via setters, and pushes the modified object back to storage.

![Figure 4.2: Anemic vs Rich Domain Model Architecture](visuals/anemic_vs_rich_architecture.png){width=90%}

The following code illustrates this fragile, anemic design:

```java
// Anemic Account Model (Fragile Data Holder)
public class Account {
    private String id;
    private BigDecimal balance;
    private String currency;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public BigDecimal getBalance() { return balance; }
    public void setBalance(BigDecimal balance) { this.balance = balance; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
}

// Stateless Service containing business invariants (Anti-pattern)
public class LedgerService {
    public void transfer(Account from, Account to, BigDecimal amount) {
        if (from.getBalance().compareTo(amount) < 0) {
            throw new IllegalArgumentException("Insufficient funds");
        }
        if (!from.getCurrency().equals(to.getCurrency())) {
            throw new IllegalArgumentException("Currency mismatch");
        }
        from.setBalance(from.getBalance().subtract(amount));
        to.setBalance(to.getBalance().add(amount));
    }
}
```


### Why the Anemic Model Fails in Production

1. **Loss of Encapsulation & Invariant Leakage:** Any component in the application can directly modify account state (e.g., `account.setBalance(new BigDecimal("-1000.00"))`), bypassing validation checks entirely and creating invalid data.
2. **Scatter-Shot Business Logic:** Validation rules become duplicated across multiple service layers (`BillingService`, `PayoutService`, `TransferService`). When a business rule changes, developers must hunt through every service to update logic, risking logic drift and bugs.
3. **Concurrency Vulnerability (TOCTOU):** Separating state checks from state mutation in external services creates **Time-of-Check to Time-of-Use (TOCTOU)** race conditions in multi-threaded environments, leading to negative balances and ledger corruption.

#### Chronological Breakdown of a TOCTOU Race Condition

```text
Initial Database State: Account A Balance = $100.00 (Overdraft Limit = $0.00)

Thread 1 (Withdraw $80.00)                 Thread 2 (Withdraw $70.00)
─────────────────────────────────────     ─────────────────────────────────────

1. Read balance from DB ($100.00)
2. Check: $100.00 >= $80.00 (PASSES)
                                          3. Read balance from DB ($100.00)
                                          4. Check: $100.00 >= $70.00 (PASSES)
5. Compute new balance = $20.00
6. Write DB balance = $20.00
                                          7. Compute new balance = $30.00
                                          8. Write DB balance = $30.00 (FATAL CORRUPTION!)
───────────────────────────────────────────────────────────────────────────────
Result: $150.00 withdrawn from account, but final database balance shows $30.00!
```

In a senior coding or architecture interview, presenting an anemic model signals a lack of software craftsmanship. Candidates must demonstrate how to refactor anemic structures into **rich domain models**.

## Refactoring Walkthrough: Building Rich Aggregate Boundaries

To refactor an anemic domain model into a secure, self-validating rich aggregate, adhere to three core refactoring rules:

### Rule 1: Protect Domain Invariants in the Constructor (Fail-Fast Instantiation)
An object must never exist in an invalid state. Validate all pre-conditions inside the constructor or static factory method. If invalid arguments are passed (e.g., null currency, negative initial balance), fail-fast immediately by throwing an explicit domain exception.

### Rule 2: Eliminate Setters and Restrict Direct State Access
Remove all public setter methods. Mark internal fields as `private` (and `final` where applicable). The only way external code can modify state is by invoking explicit, intent-revealing business methods (`debit()`, `credit()`, `freeze()`).

### Rule 3: Encapsulate Operations & Concurrency Protections Inside the Aggregate
Move validation checks and mutation logic directly into the entity. The aggregate root must protect its own state boundaries and manage its internal synchronization.

## Rich Abstraction & Encapsulation in Practice

In AuraPay, our `LedgerAccount` domain model is a rich aggregate root. It encapsulates its own `debit()`, `credit()`, and `transferTo()` methods, ensuring that no transfer occurs without validating currencies, enforcing overdraft limits, and acquiring locks safely.

The following code demonstrates rich encapsulation:

```java
package com.aurapay.domain;

import java.math.BigDecimal;
import java.util.Objects;

/**

 * Demonstrates a rich domain model encapsulating transfer logic and enforcing 
 * cross-entity invariants.
 */
public class LedgerAccount {
    private final String accountId;
    private final String currency;
    private BigDecimal balance;
    private final BigDecimal overdraftLimit;

    public LedgerAccount(String accountId, String currency, BigDecimal initialBalance, BigDecimal overdraftLimit) {
        this.accountId = Objects.requireNonNull(accountId);
        this.currency = Objects.requireNonNull(currency);
        this.balance = Objects.requireNonNull(initialBalance);
        this.overdraftLimit = Objects.requireNonNull(overdraftLimit);
    }

    public synchronized BigDecimal getBalance() { return balance; }
    public String getCurrency() { return currency; }

    public synchronized void debit(BigDecimal amount) {
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Debit amount must be positive");
        }
        BigDecimal newBalance = this.balance.subtract(amount);
        if (newBalance.add(this.overdraftLimit).compareTo(BigDecimal.ZERO) < 0) {
            throw new InsufficientFundsException("Overdraft limit exceeded");
        }
        this.balance = newBalance;
    }

    public synchronized void credit(BigDecimal amount) {
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Credit amount must be positive");
        }
        this.balance = this.balance.add(amount);
    }

    /**

     * Executes a thread-safe transfer to a target account, enforcing business invariants.
     * Prevents mismatched currencies (pre-condition) and double-debiting.
     */
    public void transferTo(LedgerAccount target, BigDecimal amount) {
        Objects.requireNonNull(target, "Destination account cannot be null");
        Objects.requireNonNull(amount, "Transfer amount cannot be null");

        // PRE-CONDITION ENFORCEMENT: Currency matching
        if (!this.currency.equals(target.getCurrency())) {
            throw new CurrencyMismatchException(
                String.format("Cannot transfer between mismatched currencies: %s and %s", 
                this.currency, target.getCurrency())
            );
        }

        // PRE-CONDITION ENFORCEMENT: Self-transfer check
        if (this.accountId.equals(target.accountId)) {
            throw new IllegalArgumentException("Cannot transfer to the same account");
        }

        // To prevent deadlocks, lock accounts in a stable global order
        LedgerAccount firstLock = this.accountId.compareTo(target.accountId) < 0 ? this : target;
        LedgerAccount secondLock = firstLock == this ? target : this;

        synchronized (firstLock) {
            synchronized (secondLock) {
                // Execute atomic debit-credit sequence
                this.debit(amount);
                target.credit(amount);
            }
        }
    }
}
```


### Granular Code Dissection & Design Annotations

Let us examine the architectural decisions embedded in the `LedgerAccount` implementation:

- **`<1>` Fail-Fast Constructor Invariants (`Objects.requireNonNull`):**  
  An entity must never enter memory in a partially constructed or illegal state. By validating all constructor parameters immediately, we eliminate the need for defensive null checks throughout downstream business methods.

- **`<2>` Granular Synchronization on Mutators (`synchronized void debit` / `credit`):**  
  State mutation is protected at the aggregate boundary. Notice that validation (`newBalance.add(overdraftLimit) >= 0`) and state assignment (`this.balance = newBalance`) occur within the same synchronized monitor, rendering TOCTOU race conditions impossible on a single account.

- **`<3>` Domain-Specific Custom Exceptions (`InsufficientFundsException`):**  
  Instead of throwing generic `RuntimeException` or `IllegalStateException`, the domain emits explicit business exceptions. This allows the API Gateway and Web layer to map business domain errors directly to standard HTTP status codes (`422 Unprocessable Entity` or `409 Conflict`) without brittle string parsing.

- **`<4>` Deterministic Global Lock Ordering (`compareTo`):**  
  When moving funds between two accounts, locking both instances simultaneously introduces circular wait risks. By sorting accounts by their immutable `accountId`, we establish a strict total order $\prec$, guaranteeing deadlock-free multi-entity operations.

### Deadlock Prevention via Global Lock Ordering

Notice the synchronization logic inside `transferTo()`. In high-concurrency payment engines, locking two entities simultaneously (e.g., Account A transferring to B while Account B is transferring to A) creates a classic circular-wait deadlock.

#### The 4 Coffman Deadlock Conditions & Mathematical Proof

Formalized by Edward G. Coffman Jr. in 1971, a deadlock can occur if and only if all four of the following conditions hold simultaneously:

1. **Mutual Exclusion:** At least one resource must be held in a non-shareable mode (exclusive lock).
2. **Hold and Wait:** A thread currently holding at least one resource is waiting to acquire additional resources held by other threads.
3. **No Preemption:** Resources cannot be forcibly confiscated from a thread holding them until the thread voluntarily releases them.
4. **Circular Wait:** A closed chain of threads $\{T_1, T_2, \dots, T_n\}$ exists such that $T_1$ waits for a resource held by $T_2$, $T_2$ waits for $T_3$, and $T_n$ waits for $T_1$.

```text
Circular Wait Deadlock:
[Thread 1 (Holds Lock A)] ──────(Requests Lock B)─────► [Thread 2 (Holds Lock B)]
          ▲                                                          │
          └─────────────────────(Requests Lock A)────────────────────┘
```

**Mathematical Proof of Deterministic Lock Ordering:**  
Let the universe of lockable resources be denoted by $R = \{r_1, r_2, \dots, r_m\}$. Establish a strict, global total ordering relation $\prec$ over $R$ such that for any two distinct resources $r_j, r_k$, either $r_j \prec r_k$ or $r_k \prec r_j$.

Define the protocol: *A thread requesting multiple resources must acquire them in strictly increasing order according to $\prec$.*

Assume, for contradiction, that a deadlock occurs. Under Coffman's fourth condition, there must exist a circular chain of threads:
$$T_1 \to T_2 \to T_3 \to \dots \to T_n \to T_1$$
where $T_i$ holds resource $A_i$ and waits for resource $B_i$ held by $T_{i+1}$ (with $T_n$ waiting for $B_n = A_1$ held by $T_1$).

By the protocol, each thread $T_i$ holding $A_i$ can only request $B_i$ if:
$$A_i \prec B_i$$
Since $T_{i+1}$ holds $B_i$ and later requests $B_{i+1}$, it must be that $B_i \prec B_{i+1}$. Transitivity of the total order $\prec$ implies:
$$A_1 \prec B_1 \le A_2 \prec B_2 \le \dots \le A_n \prec B_n = A_1$$
This yields the strict inequality:
$$A_1 \prec A_1$$
Because $\prec$ is an irreflexive partial order, no element can precede itself ($A_1 \not\prec A_1$). This contradiction proves that a cyclic dependency graph cannot form. Condition 4 (**Circular Wait**) is mathematically impossible, eliminating deadlocks entirely.


## Domain Events: Decoupling Aggregates in Event-Driven Architectures

In sophisticated enterprise systems, state mutation within an Aggregate Root often has ripple effects across other bounded contexts. For example, when an account balance dips below a minimum threshold, the Notification Service must dispatch an SMS alert, the Risk Engine must update fraud scores, and the Analytics Warehouse must ingest the ledger transition.

A common design flaw is injecting external services directly into the aggregate:

```java
// ANTI-PATTERN: Leaking infrastructure and external dependencies into Domain Model
public class LedgerAccount {
    @Autowired private NotificationClient notificationClient; // FATAL COUPLING!
    @Autowired private KafkaTemplate kafkaTemplate;           // FATAL COUPLING!
    
    public void debit(BigDecimal amount) {
        // ... state mutation ...
        notificationClient.sendSms(...); // If network fails, debit rolls back!
    }
}
```

This violates the Single Responsibility Principle and couples the pure domain model to volatile network infrastructure. If the notification service experiences latency or network timeouts, the core financial debit transaction fails.

### The Domain Event Accumulator Pattern

The DDD solution is **Domain Events**. An aggregate root mutates its state and records an immutable event payload in an internal event collection. The domain model remains pure, with zero network dependencies:

```java
public abstract class AbstractAggregateRoot {
    private final List<DomainEvent> domainEvents = new ArrayList<>();

    protected void registerEvent(DomainEvent event) {
        this.domainEvents.add(Objects.requireNonNull(event));
    }

    public List<DomainEvent> pollEvents() {
        List<DomainEvent> snapshot = Collections.unmodifiableList(new ArrayList<>(this.domainEvents));
        this.domainEvents.clear();
        return snapshot;
    }
}
```

When `LedgerAccount` executes a debit:

```java
public void debit(BigDecimal amount) {
    // 1. Verify business invariant
    BigDecimal newBalance = this.balance.subtract(amount);
    if (newBalance.add(this.overdraftLimit).compareTo(BigDecimal.ZERO) < 0) {
        throw new InsufficientFundsException("Overdraft limit exceeded");
    }
    // 2. Mutate internal state
    this.balance = newBalance;
    
    // 3. Register Domain Event
    registerEvent(new AccountDebitedEvent(this.accountId, amount, this.balance, Instant.now()));
}
```

The application service layer (or repository) persists the aggregate and flushes the events atomically into the database Outbox table within the same transaction. This guarantees zero lost events without coupling the domain to message brokers.


## The 3 Golden Rules of DDD Aggregate Boundaries

When interviewing for Staff or Principal roles, interviewers probe your understanding of aggregate boundary design. In an e-commerce or financial system, novice candidates often make the mistake of creating giant aggregates (e.g., an `Order` aggregate that contains all `Customer` details, all `Inventory` rows, and all `Payment` records).

Giant aggregates cause catastrophic concurrency contention: every time an order is placed, the entire customer record and inventory catalog are locked, throttling system throughput.

Adhere to the **3 Golden Invariant Rules of Aggregate Design** (Vernon, 2013):

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE 3 GOLDEN RULES OF AGGREGATE BOUNDARIES                      │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│ Rule                           │ Architectural Mandate & Enforcement Mechanism         │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 1. Model True Invariants       │ An aggregate encapsulates only those fields that must │
│                                │ remain consistently valid in real time. If data can   │
│                                │ be eventually consistent, it belongs in another aggregate.│
│ 2. Design Small Aggregates     │ Small aggregates maximize throughput and eliminate    │
│                                │ multi-row database lock contention.                   │
│ 3. Reference by Identity Only  │ Aggregates never hold direct object references to     │
│                                │ other aggregates; they reference them solely by ID.   │
└────────────────────────────────┴───────────────────────────────────────────────────────┘
```

> **The Single-Transaction Rule:** *A single database transaction should modify exactly ONE aggregate instance.* If a business workflow spans multiple aggregates (e.g., deducting inventory from `Product` and charging `LedgerAccount`), use asynchronous Domain Events and a Saga Orchestrator to achieve eventual consistency rather than distributed two-phase locking.


## Virtual Method Table (VTable) Dynamic Dispatch Mechanics

How does the runtime resolve polymorphic method calls (such as `route.settle()`) without conditional branches?

In compiled and managed runtimes (JVM HotSpot, .NET CLR, C++), every class defining or overriding virtual methods contains an internal pointer to a **Virtual Method Table (VTable)**.

- The VTable is a contiguous array of function pointers stored in process memory.
- When `route.settle()` is invoked:
  1. The CPU loads the object's VTable reference at memory offset 0 (`*vptr`).
  2. It performs an indexed array lookup at a fixed method offset (e.g., `vtable[3]`).
  3. It executes an indirect jump instruction (`CALL [vtable + offset]`) to the concrete method implementation.

```text
Object Memory Layout & VTable Dispatch:
[Route Instance in Heap]
┌─────────────────────────┐
│ *vptr (Offset 0)        │───────► [VTable for VisaSettlementRoute]
├─────────────────────────┤         ┌─────────────────────────────────┐
│ accountId (Offset 8)    │         │ Index 0: hashCode()             │
├─────────────────────────┤         │ Index 1: equals()               │
│ networkId (Offset 16)   │         │ Index 2: toString()             │
└─────────────────────────┘         │ Index 3: settle() ──────────────┼──► Machine Code
                                    └─────────────────────────────────┘
```

### JIT Call-Site Optimization: Monomorphic vs. Bimorphic vs. Megamorphic

Modern JIT compilers (HotSpot C2, CLR RyuJIT) monitor polymorphic call sites during execution and optimize them dynamically:

1. **Monomorphic Call Site (1 Receiver Type):**  
   If the JIT observes that 99.9% of calls through `SettlementRoute` always pass `VisaSettlementRoute`, it completely **inlines the target method code directly into the caller**. VTable lookup is eliminated; dispatch cost drops to **$0\text{ ns}$**.

2. **Bimorphic Call Site (2 Receiver Types):**  
   If two types alternate (e.g., `Visa` and `Mastercard`), the JIT generates an inline conditional branch:
   ```c
   if (obj.class == VisaSettlementRoute.class) {
       // inlined Visa logic
   } else if (obj.class == MastercardSettlementRoute.class) {
       // inlined Mastercard logic
   }
   ```

3. **Megamorphic Call Site ($\ge 3$ Receiver Types):**  
   When three or more distinct classes pass through the same call site, the JIT gives up on inlining and falls back to a full indirect VTable lookup (`CALL [vtable + offset]`). This incurs a $2\text{--}4\text{ ns}$ penalty and can cause CPU branch target buffer (BTB) cache misses in ultra-low-latency loops.


## Composition over Inheritance

A frequent OOP mistake in technical interviews is abusing inheritance to support distinct feature variations. For example, when building a settlement routing engine for different payment networks (ACH, FedWire, Visa), a candidate might create a base `SettlementService` class and subclass it: `AchSettlementService`, `FedWireSettlementService`, etc.

This introduces tight coupling and brittle hierarchies:

- **Fragile Base Class Problem:** Modifying an internal helper method in the parent class can silently break invariants in child classes.
- **Class Explosion:** If routes need to support different fee models (Flat Fee, Tiered Fee) and encryption standards, inheritance requires combinatorial subclasses (`AchFlatFeeEncryptedService`, `AchTieredFeeEncryptedService`), yielding an unmaintainable codebase.

The golden rule of enterprise OOP design is to **favor composition over inheritance**. Instead of subclassing, compose the routing engine by injecting a collection of independent strategy routes:

![Figure 4.3: Composition over Inheritance](visuals/composition_vs_inheritance.png){width=85%}


## Polymorphism over Conditional Branching

A common indicator of junior-level code is using long `if-else` or `switch` blocks that inspect object types or enum flags to determine execution logic:

```java
// Anti-pattern: Inspecting properties to determine routing
if (tx.getAmount().compareTo(LIMIT) > 0) {
    fedWireRoute.process(tx);
} else {
    achRoute.process(tx);
}
```


### Why `switch` on Type Violates the Open/Closed Principle (OCP)

1. **High Regression Risk:** Adding a new payment network requires modifying the central router file. Any developer editing this file risks introducing regressions across unrelated networks.
2. **Scatter-Shot Code Changes:** Every time a new network is added, developers must find every `switch (networkType)` block in the codebase (`RoutingService`, `FeeCalculationService`, `ValidationService`, `ReconciliationService`). Inevitably, one switch statement is forgotten, causing runtime `UnhandledCaseException` crashes in production.

Polymorphism resolves this cleanly. By defining a generic `SettlementRoute` interface, the routing engine delegates network-specific validation, fee calculation, and wire dispatch to the individual route classes:

```java
package com.aurapay.settlement;

import com.aurapay.domain.TransactionRecord;
import java.math.BigDecimal;

/**

 * Interface defining the polymorphic contract for payment settlement networks.
 */
public interface SettlementRoute {
    boolean supports(TransactionRecord transaction);
    void process(TransactionRecord transaction);
    BigDecimal calculateFees(TransactionRecord transaction);
}

/**

 * Concrete implementation for the ACH network (low cost, delayed).
 */
public class AchRoute implements SettlementRoute {
    private static final BigDecimal ACH_FLAT_FEE = new BigDecimal("0.50");

    @Override
    public boolean supports(TransactionRecord transaction) {
        // ACH supports amounts up to $100,000
        return transaction.amount().compareTo(new BigDecimal("100000.00")) <= 0;
    }

    @Override
    public void process(TransactionRecord transaction) {
        System.out.println("Routing transaction " + transaction.transactionId() + " via ACH network.");
    }

    @Override
    public BigDecimal calculateFees(TransactionRecord transaction) {
        return ACH_FLAT_FEE;
    }
}

/**

 * Concrete implementation for the FedWire network (instant, high cost).
 */
public class FedWireRoute implements SettlementRoute {
    private static final BigDecimal WIRE_FLAT_FEE = new BigDecimal("15.00");

    @Override
    public boolean supports(TransactionRecord transaction) {
        // FedWire is used for high-value transactions above $10,000
        return transaction.amount().compareTo(new BigDecimal("10000.00")) > 0;
    }

    @Override
    public void process(TransactionRecord transaction) {
        System.out.println("Routing transaction " + transaction.transactionId() + " via FedWire network.");
    }

    @Override
    public BigDecimal calculateFees(TransactionRecord transaction) {
        return WIRE_FLAT_FEE;
    }
}
```


The main transaction processor can then execute settlements via a clean, extensible polymorphic loop:

```java
public class SettlementProcessor {
    private final List<SettlementRoute> routes;

    public SettlementProcessor(List<SettlementRoute> routes) {
        this.routes = routes;
    }

    public void execute(TransactionRecord transaction) {
        SettlementRoute activeRoute = routes.stream()
            .filter(route -> route.supports(transaction))
            .findFirst()
            .orElseThrow(() -> new NoRouteFoundException("No supported route found"));
            
        activeRoute.process(transaction);
    }
}
```



## When NOT to Use Rich Models: The CQRS Command-Query Duality

A senior engineer understands that no pattern is universally optimal. While Rich Domain Models are essential for **write-heavy transactional command paths** where complex business invariants must be protected, they are an anti-pattern for **read-heavy query paths**.

If an application needs to render a dashboard displaying an account's recent 50 transactions with merchant names and fee totals:

- **The Rich Model Trap:** Instantiating 50 full `Transaction` aggregate roots and a `LedgerAccount` aggregate into memory incurs massive object allocation overhead, triggers lazy-loading N+1 query cascades, and wastes CPU cycles hydrating business logic that will never be executed.
- **The CQRS Solution:** Use **Command Query Responsibility Segregation (CQRS)**:
  - **Command Path (Writes):** Use the Rich Domain Model (`LedgerAccount`) with strict aggregate boundaries, synchronous invariants, and transactional locking.
  - **Query Path (Reads):** Bypass the domain model entirely. Query the database directly into flat, read-only DTO projections using lightweight SQL joins or specialized read views (e.g. Elasticsearch or Redis Read Models).

```text
Command Query Responsibility Segregation (CQRS) Flow:
[Client Write Request] ──► [LedgerService] ──► [Rich Domain Aggregate] ──► [Postgres Write DB]
                                                                                   │ (CDC / Outbox)
                                                                                   ▼
[Client Read Request]  ──◄ [Read Controller] ◄── [Flat Read DTO] ◄──────── [Read Projection View]
```

---

> ⭐ **STAR Moment: The Encapsulation & Aggregate Test**
> 
> During object-oriented design interviews, evaluate your domain classes with this test: *Can a client developer instantiate this object or invoke a method that leaves the system in an invalid state?* If setters allow negative balances, unvalidated currencies, or race conditions, encapsulation has failed. Emphasize in your interview: *"I encapsulate state inside Rich Aggregate Roots with fail-fast constructors and intent-revealing methods, ensuring domain invariants are protected natively without relying on external services."*
