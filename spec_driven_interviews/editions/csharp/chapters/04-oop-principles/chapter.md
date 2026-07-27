# Principles of Object-Oriented Design

> *"Do not expose your state to the world. Encapsulate your data, expose your contracts, and let polymorphism handle the variance."*


## The Anemic Domain Model Anti-Pattern

In many enterprise applications, domain classes are treated as passive data holders—simple collections of fields with auto-generated getters and setters. This is the **Anemic Domain Model** anti-pattern. 

When your domain models are anemic, the business logic shifts into stateless service classes (e.g., `LedgerService`). The service pulls the state out of the domain model, performs validation, modifies the fields, and pushes the data back to the database. The danger of this design is that the domain object itself has no control over its state. Any developer can instantiate a ledger account, set the balance to a negative value without checks, and persist it, violating the core safety boundaries of the system.

![God Object Violation Detector — Single Responsibility Principle](visuals/oop_violation_detector.jpg){width=85%}

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

1. **Lack of Encapsulation:** Any part of the application can modify the account balance directly: `account.setBalance(new BigDecimal("-1000.00"))`, bypassing the business checks entirely.
2. **Scatter-Shot Validation:** Validation logic is duplicated across multiple services (e.g., `BillingService`, `PayoutService`, `TransferService`). If a validation rule changes, you must locate and modify every instance across the codebase, risking logic drift.
3. **Concurrency Vulnerability:** In high-concurrency systems, separating state from checks leads to **Time-of-Check to Time-of-Use (TOCTOU)** race conditions, resulting in balance corruption.

In a senior coding or architecture interview, presenting an anemic model is a missed opportunity. To demonstrate true software craftsmanship, you must show how to design **rich domain models** that encapsulate state and enforce invariants.

![Anemic vs Rich Domain Model Comparison](visuals/anemic_vs_rich.png){width=85%}


## Refactoring Walkthrough: From Anemic to Rich

To refactor a fragile anemic domain into a secure, self-validating rich domain model, follow these three rules:

### Protect Domain Invariants in the Constructor
Ensure that an object can never be created in an invalid state. Validate all inputs during instantiation. If a pre-condition is violated, fail-fast immediately by throwing an exception.

### Remove Setters and Restrict State Access
Eliminate all public setter methods. Fields should be `private` and, where possible, `final`. The only way to modify state is through explicit, domain-specific methods that protect the object's invariants.

### Move Operations Inside the Aggregate Boundary
Instead of letting external service classes manipulate fields, encapsulate the business behavior inside the entity itself. The entity must protect its own state.


## Abstraction & Encapsulation

Encapsulation is not merely the practice of making fields `private` and exposing public getters and setters. True encapsulation means that an object protects its own state, ensuring that its internal data can never enter an invalid state.

In AuraPay, our `LedgerAccount` domain model is rich. It contains its own `debit`, `credit`, and `transferTo` methods, making it impossible to perform a transfer without validating currencies, checking overdraft limits, and preventing concurrency deadlocks.

The following code illustrates this rich encapsulation:

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


### Deadlock Prevention via Global Ordering
Notice the synchronization logic inside the `transferTo` method. In a high-concurrency payment engine, locking two entities simultaneously (e.g., account $A$ transferring to $B$, while $B$ is transferring to $A$) can lead to a circular wait deadlock. 

To prevent this, the method compares the account identifiers (`this.accountId` and `target.accountId`) and locks them in a consistent, alphabetical global order. This is a classic concurrency pattern that demonstrates your readiness to design banking-grade production code.


## OOP Principles vs. DDD Concepts

Object-Oriented Design and Domain-Driven Design (DDD) are deeply interconnected. When designing enterprise systems, OOD principles map directly to DDD tactical design patterns:

| OOP Principle | DDD Tactical Pattern | Architectural Mapping |
|---|---|---|
| **Encapsulation** | Aggregate Root | The aggregate root acts as a consistency boundary, encapsulating internal entities and protecting invariants from external modification. |
| **Immutability** | Value Object | Objects without distinct identity (like `Money`) are designed as immutable value objects, preventing side effects during sharing. |
| **Polymorphism** | Domain Strategy | Swapping of algorithm strategies (like different fee calculations) is modeled as polymorphic strategy interfaces. |
| **Abstraction** | Repository / Service | Shielding the domain from infrastructure adapters (database, message queues) using clean interface abstractions. |


## Composition over Inheritance

A common mistake in object-oriented design is abusing inheritance. For example, if you are asked to support different settlement networks (ACH, FedWire, Visa), a naive developer might create a base `SettlementService` class and subclass it: `AchSettlementService`, `FedWireSettlementService`, etc.

This creates tight coupling. If you need to change how fees are calculated, or add a new network channel, you risk breaking parent behaviors. The first rule of enterprise OOP design is to **favor composition over inheritance**.

Instead of sub-classing, we compose our routing engine by injecting a collection of independent strategy routes. The core engine is decoupled from the network-specific details.

![Composition over Inheritance](visuals/composition_vs_inheritance.png){width=85%}


## Polymorphism over Conditional Logic

One of the easiest ways to spot a junior candidate's code is looking for large `if-else` or `switch` blocks that inspect the type of an object to determine behavior. For example:

```csharp
// Anti-pattern: Inspecting properties to determine routing
if (tx.Amount > LIMIT) {
    fedWireRoute.Process(tx);
} else {
    achRoute.Process(tx);
}
```


This violates the Open/Closed Principle. Every time you support a new payment network, you must modify this routing block.

Polymorphism allows you to clean this up. By defining a generic `SettlementRoute` interface, the routing engine can iterate through all available routes, asking each route if it supports the transaction, and executing the process dynamically.

The following code defines this polymorphic settlement design:

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


By utilizing this interface, the main transaction processor can execute settlements using a clean polymorphic loop, completely decoupled from specific network implementations:

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



> ⭐ **STAR Moment: The Encapsulation Test**
> 
> When designing class structures in a technical interview, ask yourself: *Can this class enter an invalid state?* If a client developer can instantiate your object and set its properties to values that violate business rules, your encapsulation has failed. Build your validation boundaries directly into the constructors and state-transition methods of your domain objects.
