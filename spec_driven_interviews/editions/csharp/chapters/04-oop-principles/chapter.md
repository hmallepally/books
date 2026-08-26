# Principles of Object-Oriented Design & Domain-Driven Craftsmanship

> *"Do not expose your state to the world. Encapsulate your data, expose your contracts, and let polymorphism handle the variance."*

## The Foundations: Connecting OOP Principles to Domain-Driven Design (DDD)

In enterprise software engineering and senior-level technical interviews, Object-Oriented Programming (OOP) is not merely about syntax or class hierarchies. Its primary purpose is to model real-world business domains, enforce critical invariants, and protect data integrity under high concurrency.

When designing large-scale enterprise systems, core OOP principles map directly to **Domain-Driven Design (DDD)** tactical patterns. Understanding this bridge prevents code from degenerating into unmaintainable scripts:

![The OOP to DDD Architectural Bridge](visuals/oop_to_ddd_bridge.png){width=90%}

### Core DDD Definitions Every Candidate Must Master:

1. **Entities:** Objects defined by a unique, enduring identity that persists across state changes (e.g., a `LedgerAccount` identified by a unique `accountId`). Two entities with identical balances are distinct if their IDs differ.
2. **Value Objects:** Immutable objects defined entirely by their attribute values, possessing no conceptual identity (e.g., `Money`, `Currency`, or `Address`). If two `Money` objects both represent `$100 USD`, they are completely interchangeable.
3. **Aggregates & Aggregate Roots:** A cluster of associated domain objects (Entities and Value Objects) treated as a single unit for data changes. The **Aggregate Root** is the sole gateway through which external code interacts with internal objects, guaranteeing that all domain invariants remain valid across operations.
4. **Domain Services:** Operations or business transformations that do not naturally belong to a single Entity or Value Object (e.g., cross-account fund routing engines).

## The Anemic Domain Model Anti-Pattern

Despite understanding basic OOP syntax, many enterprise applications fall into a common architectural trap: treating domain classes as passive data holders—simple bags of private fields with auto-generated getters and setters. Martin Fowler termed this the **Anemic Domain Model** anti-pattern.

When domain models are anemic, business logic escapes into external, stateless service classes (e.g., `LedgerService`). The service pulls raw data out of the domain object, validates it externally, mutates the fields via setters, and pushes the modified object back to storage.

![Anemic vs Rich Domain Model Architecture](visuals/anemic_vs_rich_architecture.png){width=90%}

The following code illustrates this fragile, anemic design:

```csharp
// Anemic Account Model (Fragile Data Holder)
public class Account 
{
    public string Id { get; set; }
    public decimal Balance { get; set; }
    public string Currency { get; set; }
}

// Stateless Service containing business invariants (Anti-pattern)
public class LedgerService 
{
    public void Transfer(Account from, Account to, decimal amount) 
    {
        if (from.Balance < amount) 
        {
            throw new ArgumentException("Insufficient funds");
        }
        if (from.Currency != to.Currency) 
        {
            throw new ArgumentException("Currency mismatch");
        }
        from.Balance -= amount;
        to.Balance += amount;
    }
}
```


### Why the Anemic Model Fails in Production

1. **Loss of Encapsulation & Invariant Leakage:** Any component in the application can directly modify account state (e.g., `account.setBalance(new BigDecimal("-1000.00"))`), bypassing validation checks entirely and creating invalid data.
2. **Scatter-Shot Business Logic:** Validation rules become duplicated across multiple service layers (`BillingService`, `PayoutService`, `TransferService`). When a business rule changes, developers must hunt through every service to update logic, risking logic drift and bugs.
3. **Concurrency Vulnerability (TOCTOU):** Separating state checks from state mutation in external services creates **Time-of-Check to Time-of-Use (TOCTOU)** race conditions in multi-threaded environments, leading to negative balances and ledger corruption.

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

```csharp
using System;

namespace AuraPay.Domain
{
    /// <summary>
    /// Demonstrates a rich domain model encapsulating transfer logic and enforcing 
    /// cross-entity invariants.
    /// </summary>
    public class LedgerAccount
    {
        private readonly object _lock = new object();
        public string AccountId { get; }
        public string Currency { get; }
        private decimal _balance;
        public decimal OverdraftLimit { get; }

        public decimal Balance
        {
            get
            {
                lock (_lock)
                {
                    return _balance;
                }
            }
        }

        public LedgerAccount(string accountId, string currency, decimal initialBalance, decimal overdraftLimit)
        {
            AccountId = accountId ?? throw new ArgumentNullException(nameof(accountId));
            Currency = currency ?? throw new ArgumentNullException(nameof(currency));
            _balance = initialBalance;
            OverdraftLimit = overdraftLimit;
        }

        public void Debit(decimal amount)
        {
            if (amount <= 0) throw new ArgumentException("Debit amount must be positive", nameof(amount));
            lock (_lock)
            {
                decimal newBalance = _balance - amount;
                if (newBalance + OverdraftLimit < 0)
                {
                    throw new InvalidOperationException("Overdraft limit exceeded");
                }
                _balance = newBalance;
            }
        }

        public void Credit(decimal amount)
        {
            if (amount <= 0) throw new ArgumentException("Credit amount must be positive", nameof(amount));
            lock (_lock)
            {
                _balance += amount;
            }
        }

        /// <summary>
        /// Executes a thread-safe transfer to a target account, enforcing business invariants.
        /// Prevents mismatched currencies and double-debiting.
        /// </summary>
        public void TransferTo(LedgerAccount target, decimal amount)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));

            // PRE-CONDITION ENFORCEMENT: Currency matching
            if (Currency != target.Currency)
            {
                throw new InvalidOperationException($"Cannot transfer between mismatched currencies: {Currency} and {target.Currency}");
            }

            // PRE-CONDITION ENFORCEMENT: Self-transfer check
            if (AccountId == target.AccountId)
            {
                throw new ArgumentException("Cannot transfer to the same account");
            }

            // To prevent deadlocks, lock accounts in a stable global order
            var firstLock = string.Compare(AccountId, target.AccountId, StringComparison.Ordinal) < 0 ? this : target;
            var secondLock = firstLock == this ? target : this;

            lock (firstLock._lock)
            {
                lock (secondLock._lock)
                {
                    // Execute atomic debit-credit sequence
                    this.Debit(amount);
                    target.Credit(amount);
                }
            }
        }
    }
}
```


### Deadlock Prevention via Global Lock Ordering

Notice the synchronization logic inside `transferTo()`. In high-concurrency payment engines, locking two entities simultaneously (e.g., Account A transferring to B while Account B is transferring to A) creates a classic circular-wait deadlock.

The aggregate enforces two strict invariants before locking:

1. **Self-Transfer Precondition:** The method immediately rejects transfers where `this.accountId.equals(target.accountId)` (throwing an `InvalidTransferException`), preventing redundant reentrant lock acquisitions.
2. **Deterministic Lock Ordering:** To eliminate circular wait deadlocks, the method compares the two account identifiers and acquires intrinsic/explicit locks in a deterministic **lexicographical ordering** (e.g., locking the account with the smaller UUID/string ID first, regardless of transfer direction). This guarantees that concurrent transfers between the same two accounts always acquire locks in identical sequence.

## Composition over Inheritance

A frequent OOP mistake in technical interviews is abusing inheritance to support distinct feature variations. For example, when building a settlement routing engine for different payment networks (ACH, FedWire, Visa), a candidate might create a base `SettlementService` class and subclass it: `AchSettlementService`, `FedWireSettlementService`, etc.

This introduces tight coupling and brittle hierarchies. Modifying parent behavior or adding multi-network routing rules risks breaking child implementations. The golden rule of enterprise OOP design is to **favor composition over inheritance**.

Instead of subclassing, compose the routing engine by injecting a collection of independent strategy routes. The core engine is decoupled from network-specific settlement details:

![Composition over Inheritance](visuals/composition_vs_inheritance.png){width=85%}

## Polymorphism over Conditional Branching

A common indicator of junior-level code is using long `if-else` or `switch` blocks that inspect object types or enum flags to determine execution logic:

```csharp
// Anti-pattern: Inspecting properties to determine routing
if (tx.Amount > LIMIT) {
    fedWireRoute.Process(tx);
} else {
    achRoute.Process(tx);
}
```


This violates the **Open/Closed Principle (OCP)**. Adding a new payment network requires modifying existing routing blocks, increasing regression risks.

Polymorphism resolves this cleanly. By defining a generic `SettlementRoute` interface, the routing engine iterates through available route implementations, asking each route if it supports the transaction, and executing settlement dynamically:

```csharp
using System;

namespace AuraPay.Settlement
{
    /// <summary>
    /// Interface defining the polymorphic contract for payment settlement networks.
    /// </summary>
    public interface ISettlementRoute
    {
        bool Supports(TransactionRecord transaction);
        void Process(TransactionRecord transaction);
        decimal CalculateFees(TransactionRecord transaction);
    }

    /// <summary>
    /// Concrete implementation for the ACH network (low cost, delayed).
    /// </summary>
    public class AchRoute : ISettlementRoute
    {
        private static readonly decimal AchFlatFee = 0.50m;

        public bool Supports(TransactionRecord transaction)
        {
            return transaction.Amount <= 100000.00m;
        }

        public void Process(TransactionRecord transaction)
        {
            Console.WriteLine($"Routing transaction {transaction.TransactionId} via ACH network.");
        }

        public decimal CalculateFees(TransactionRecord transaction)
        {
            return AchFlatFee;
        }
    }

    /// <summary>
    /// Concrete implementation for the FedWire network (instant, high cost).
    /// </summary>
    public class FedWireRoute : ISettlementRoute
    {
        private static readonly decimal WireFlatFee = 15.00m;

        public bool Supports(TransactionRecord transaction)
        {
            return transaction.Amount > 10000.00m;
        }

        public void Process(TransactionRecord transaction)
        {
            Console.WriteLine($"Routing transaction {transaction.TransactionId} via FedWire network.");
        }

        public decimal CalculateFees(TransactionRecord transaction)
        {
            return WireFlatFee;
        }
    }
}
```


The main transaction processor can then execute settlements via a clean, extensible polymorphic loop:

```csharp
public class SettlementProcessor 
{
    private readonly List<ISettlementRoute> _routes;

    public SettlementProcessor(List<ISettlementRoute> routes) 
    {
        _routes = routes;
    }

    public void Execute(TransactionRecord transaction) 
    {
        var activeRoute = _routes
            .FirstOrDefault(route => route.Supports(transaction))
            ?? throw new NoRouteFoundException("No supported route found");
            
        activeRoute.Process(transaction);
    }
}
```



> ⭐ **STAR Moment: The Encapsulation & Aggregate Test**
> 
> During object-oriented design interviews, evaluate your domain classes with this test: *Can a client developer instantiate this object or invoke a method that leaves the system in an invalid state?* If setters allow negative balances, unvalidated currencies, or race conditions, encapsulation has failed. Emphasize in your interview: *"I encapsulate state inside Rich Aggregate Roots with fail-fast constructors and intent-revealing methods, ensuring domain invariants are protected natively without relying on external services."*
