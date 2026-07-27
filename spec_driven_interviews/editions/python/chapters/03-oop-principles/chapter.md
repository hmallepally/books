# Principles of Object-Oriented Design

> *"Do not expose your state to the world. Encapsulate your data, expose your contracts, and let polymorphism handle the variance."*


## The Anemic Domain Model Anti-Pattern

In many enterprise applications, domain classes are treated as passive data holders—simple collections of fields with auto-generated getters and setters. This is the **Anemic Domain Model** anti-pattern. 

When your domain models are anemic, the business logic shifts into stateless service classes (e.g., `LedgerService`). The service pulls the state out of the domain model, performs validation, modifies the fields, and pushes the data back to the database. The danger of this design is that the domain object itself has no control over its state. Any developer can instantiate a ledger account, set the balance to a negative value without checks, and persist it, violating the core safety boundaries of the system.

The following code illustrates this fragile, anemic design:

```python
# Anemic Account Model (Fragile Data Holder)
class Account:
    def __init__(self, id: str, balance: float, currency: str):
        self.id = id
        self.balance = balance
        self.currency = currency

# Stateless Service containing business invariants (Anti-pattern)
class LedgerService:
    def transfer(self, from_acc: Account, to_acc: Account, amount: float) -> None:
        if from_acc.balance < amount:
            raise ValueError("Insufficient funds")
        if from_acc.currency != to_acc.currency:
            raise ValueError("Currency mismatch")
        from_acc.balance -= amount
        to_acc.balance += amount
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

```python
from decimal import Decimal
import threading

class LedgerAccount:
    """
    Demonstrates a rich domain model encapsulating transfer logic and enforcing 
    cross-entity invariants.
    """
    def __init__(self, account_id: str, currency: str, initial_balance: Decimal, overdraft_limit: Decimal):
        self.account_id = account_id
        self.currency = currency
        self._balance = initial_balance
        self.overdraft_limit = overdraft_limit
        self._lock = threading.Lock()

    @property
    def balance(self) -> Decimal:
        with self._lock:
            return self._balance

    def debit(self, amount: Decimal):
        if amount <= 0:
            raise ValueError("Debit amount must be positive")
        with self._lock:
            new_balance = self._balance - amount
            if new_balance + self.overdraft_limit < 0:
                raise ValueError("Overdraft limit exceeded")
            self._balance = new_balance

    def credit(self, amount: Decimal):
        if amount <= 0:
            raise ValueError("Credit amount must be positive")
        with self._lock:
            self._balance += amount

    def transfer_to(self, target: 'LedgerAccount', amount: Decimal):
        """
        Executes a thread-safe transfer to a target account, enforcing business invariants.
        Prevents mismatched currencies and double-debiting.
        """
        if not target or amount is None:
            raise ValueError("Target and amount cannot be null")
        
        # PRE-CONDITION ENFORCEMENT: Currency matching
        if self.currency != target.currency:
            raise ValueError(f"Cannot transfer between mismatched currencies: {self.currency} and {target.currency}")

        # PRE-CONDITION ENFORCEMENT: Self-transfer check
        if self.account_id == target.account_id:
            raise ValueError("Cannot transfer to the same account")

        # To prevent deadlocks, lock accounts in a stable global order
        locks = [self, target]
        locks.sort(key=lambda acc: acc.account_id)

        with locks[0]._lock:
            with locks[1]._lock:
                # Execute atomic debit-credit sequence
                self.debit(amount)
                target.credit(amount)
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

```python
# Anti-pattern: Inspecting properties to determine routing
if tx.amount > LIMIT:
    fed_wire_route.process(tx)
else:
    ach_route.process(tx)
```


This violates the Open/Closed Principle. Every time you support a new payment network, you must modify this routing block.

Polymorphism allows you to clean this up. By defining a generic `SettlementRoute` interface, the routing engine can iterate through all available routes, asking each route if it supports the transaction, and executing the process dynamically.

The following code defines this polymorphic settlement design:

```python
from abc import ABC, abstractmethod
from decimal import Decimal
from uuid import UUID

class SettlementRoute(ABC):
    """
    Interface/Abstract Base Class defining the polymorphic contract for payment settlement networks.
    """
    @abstractmethod
    def supports(self, transaction) -> bool:
        pass

    @abstractmethod
    def process(self, transaction):
        pass

    @abstractmethod
    def calculate_fees(self, transaction) -> Decimal:
        pass

class AchRoute(SettlementRoute):
    """
    Concrete implementation for the ACH network (low cost, delayed).
    """
    ACH_FLAT_FEE = Decimal("0.50")

    def supports(self, transaction) -> bool:
        return transaction.amount <= Decimal("100000.00")

    def process(self, transaction):
        print(f"Routing transaction {transaction.transaction_id} via ACH network.")

    def calculate_fees(self, transaction) -> Decimal:
        return self.ACH_FLAT_FEE

class FedWireRoute(SettlementRoute):
    """
    Concrete implementation for the FedWire network (instant, high cost).
    """
    WIRE_FLAT_FEE = Decimal("15.00")

    def supports(self, transaction) -> bool:
        return transaction.amount > Decimal("10000.00")

    def process(self, transaction):
        print(f"Routing transaction {transaction.transaction_id} via FedWire network.")

    def calculate_fees(self, transaction) -> Decimal:
        return self.WIRE_FLAT_FEE
```


By utilizing this interface, the main transaction processor can execute settlements using a clean polymorphic loop, completely decoupled from specific network implementations:

```python
class SettlementProcessor:
    def __init__(self, routes: list[SettlementRoute]):
        self._routes = routes

    def execute(self, transaction: TransactionRecord) -> None:
        active_route = next(
            (route for route in self._routes if route.supports(transaction)), 
            None
        )
        if not active_route:
            raise NoRouteFoundException("No supported route found")
            
        active_route.process(transaction)
```



> ⭐ **STAR Moment: The Encapsulation Test**
> 
> When designing class structures in a technical interview, ask yourself: *Can this class enter an invalid state?* If a client developer can instantiate your object and set its properties to values that violate business rules, your encapsulation has failed. Build your validation boundaries directly into the constructors and state-transition methods of your domain objects.
