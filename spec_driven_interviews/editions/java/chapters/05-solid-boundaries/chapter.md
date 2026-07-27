# SOLID Principles: Enforcing Boundaries

> *"Software architecture is the art of drawing lines between components. SOLID is the rulebook for placing those lines."*


## SOLID in the Senior Interview

In senior and lead engineering interviews, you are almost guaranteed to be asked about the SOLID principles. Too many candidates respond by simply reciting the acronym: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion. 

If you stop there, you fail to show architectural maturity. An interviewer wants to know *why* these principles matter at scale. They want to see how applying these principles prevents structural rot, allows multiple teams to work in parallel without code collisions, and ensures that a change in database technology does not break the core transaction engine.

In this chapter, we will implement the core processing pipeline of AuraPay using a design that strictly conforms to all five SOLID principles.

![The Five SOLID Principles — Quick Reference](visuals/solid_summary.png){width=70%}

## The SOLID Transaction Pipeline

To illustrate SOLID, we will examine the `TransactionProcessor` in AuraPay. This component is responsible for retrieving ledger accounts, calculating fees, updating account balances, persisting the changes to storage, and notifying external systems.

Here is the decoupled, SOLID-compliant transaction execution flow:

```java
package com.aurapay.processing;

import com.aurapay.domain.LedgerAccount;
import com.aurapay.domain.TransactionRecord;
import java.math.BigDecimal;
import java.util.Objects;
import java.util.UUID;

/**
 * Abstraction for database operations (Dependency Inversion Principle).
 */
public interface LedgerRepository {
    LedgerAccount findById(UUID accountId);
    void save(LedgerAccount account);
}

/**
 * Abstraction for fee calculations (Open/Closed Principle).
 */
public interface FeeCalculator {
    BigDecimal calculate(TransactionRecord transaction);
}

/**
 * Interface Segregation Principle: Focused notification dispatch interface.
 */
public interface TransactionNotificationSender {
    void sendNotification(TransactionRecord transaction, String status);
}

/**
 * Core transaction processor showing SOLID compliance.
 */
public class TransactionProcessor {
    private final LedgerRepository repository;
    private final FeeCalculator feeCalculator;
    private final TransactionNotificationSender notificationSender;

    public TransactionProcessor(
        LedgerRepository repository,
        FeeCalculator feeCalculator,
        TransactionNotificationSender notificationSender
    ) {
        this.repository = Objects.requireNonNull(repository);
        this.feeCalculator = Objects.requireNonNull(feeCalculator);
        this.notificationSender = Objects.requireNonNull(notificationSender);
    }

    /**
     * Processes a transaction. Decoupled from repository, fee, and notification details.
     */
    public void process(TransactionRecord transaction) {
        Objects.requireNonNull(transaction, "Transaction cannot be null");

        // 1. Retrieve accounts from abstraction (DIP)
        LedgerAccount source = repository.findById(transaction.sourceAccountId());
        LedgerAccount destination = repository.findById(transaction.destinationAccountId());

        if (source == null || destination == null) {
            throw new IllegalArgumentException("Source or destination account not found");
        }

        // 2. Calculate fee dynamically (OCP)
        BigDecimal fee = feeCalculator.calculate(transaction);
        BigDecimal totalDebit = transaction.amount().add(fee);

        // 3. Coordinate state transitions on rich domain objects (SRP / LSP)
        // Overdraft check is executed internally within source.debit()
        source.debit(totalDebit);
        destination.credit(transaction.amount());

        // 4. Persist updated states (DIP)
        repository.save(source);
        repository.save(destination);

        // 5. Notify via segregated interface (ISP)
        notificationSender.sendNotification(transaction, "SUCCESS");
    }
}
```


Let us break down how this single class enforces all five design boundaries.


## Single Responsibility Principle (SRP)

The Single Responsibility Principle is often summarized as "a class should do only one thing." A more precise architectural definition is: **"a module should have one, and only one, reason to change."**

In our transaction pipeline, the `TransactionProcessor` has one responsibility: coordinating the business workflow of a transaction. It does not contain database queries, does not know how to format SMS or Email notifications, and does not hardcode fee calculation percentages.

- If the database schema changes, only `LedgerRepository` implementations change.
- If we switch from email notifications to SMS notifications, only `TransactionNotificationSender` implementations change.
- The `TransactionProcessor` remains untouched.


## Open/Closed Principle (OCP)

The Open/Closed Principle states that **software entities should be open for extension, but closed for modification.**

In AuraPay, we must support multiple fee models (e.g., flat fees for retail clients, percentage-based fees for merchants, waived fees for corporate accounts). 
Instead of adding nested `if-else` blocks inside the transaction processor, we inject the `FeeCalculator` interface. If we need to add a new fee model, we simply write a new class implementing `FeeCalculator` and pass it to the processor. The core processor is closed to modifications, yet the fee behavior is infinitely extendable.


## Liskov Substitution Principle (LSP)

The Liskov Substitution Principle states that **subtypes must be substitutable for their base types without altering the correctness of the program.**

In financial systems, this is highly relevant when modeling different account types. For example, a `SavingsAccount` might not allow overdrafts, while a `CheckingAccount` allows up to a certain limit.
If a developer creates a subclass `BlockedAccount` that throws an `UnsupportedOperationException` whenever `debit()` is called, they violate LSP. The `TransactionProcessor` assumes that any `LedgerAccount` returned by the repository can be debited and credited.
LSP ensures that subclass behaviors remain consistent with the contracts defined on their parent classes, preventing runtime crashes.


## Interface Segregation Principle (ISP)

The Interface Segregation Principle states that **clients should not be forced to depend on interfaces they do not use.**

In a large enterprise system, you might have a broad `NotificationService` that handles email, Slack channels, internal logging, and mobile push alerts. 
If the `TransactionProcessor` injected a giant `NotificationService` interface containing twenty unrelated methods, it would be coupled to changes in mobile app push logic. 
Instead, we define a small, segregated interface: `TransactionNotificationSender`, containing only the single `sendNotification` method. The processor only knows about what it needs to execute its task.


## Dependency Inversion Principle (DIP)

The Dependency Inversion Principle states that **high-level modules should not import anything from low-level modules. Both should depend on abstractions.**

This is the most critical principle for decoupling business logic from infrastructure.

- **Low-level modules:** Database APIs, file system writers, network channels, and concrete frameworks.
- **High-level modules:** The core business rules of your application (like transaction routing and double-entry validation).

In our implementation, the `TransactionProcessor` does not import a concrete SQL database connector or Hibernate manager. It depends entirely on the `LedgerRepository` interface. The business logic is at the top of the dependency tree, and database adapters are plugged in at the bottom. This allows you to run unit tests using a mock repository in memory, completely decoupled from a database connection.

![SOLID Dependency Inversion Principle — Before and After](visuals/solid_dip.png){width=85%}


## SOLID Violation Detector & Remedies

In technical interviews, you must be able to spot structural violations in a code sample and offer clear architectural remedies:

**SRP Violations**

- *Red Flags:* Large source files (500+ lines). Imports both database drivers and UI libraries. Multiple developers editing the same class for unrelated features.
- *Remedy:* **Decomposition** — Split into separate, focused classes coordinated by an Orchestrator or Application Service.

**OCP Violations**

- *Red Flags:* Switch statements or `if-else` blocks inspecting enums/types. Modifying existing service classes to add support for new payment partners.
- *Remedy:* **Abstraction** — Define an interface and implement the Strategy pattern. Inject a collection of these strategies.

**LSP Violations**

- *Red Flags:* Subclass methods returning dummy values or throwing `UnsupportedOperationException`. Typecasting (`instanceof`) inside helper methods.
- *Remedy:* **Hierarchy Flattening** — Replace inheritance with composition, or split the interface into smaller, specialized interfaces.

**ISP Violations**

- *Red Flags:* Concrete classes implementing interfaces with empty or dummy methods. Small client classes coupled to changes in unused interface methods.
- *Remedy:* **Segregation** — Split the fat interface into multiple single-method or small role-based interfaces.

**DIP Violations**

- *Red Flags:* Use of the `new` keyword to instantiate databases/gateways directly inside services. Direct imports of low-level infrastructure modules.
- *Remedy:* **Dependency Injection** — Program to interfaces. Pass dependencies via Constructor Injection, managed by an IoC container.


## Framework Integration: SOLID in Enterprise Containers

Modern web frameworks are designed explicitly around SOLID principles:

### Dependency Injection (IoC) Containers
Frameworks like Spring Boot (Java), ASP.NET Core (C#), and FastAPI/Dependency Injector (Python) serve as Dependency Inversion engines. By registering interfaces and their concrete implementations in the container, the framework automates constructor injection. High-level business modules declare their dependencies as constructor interfaces, completely decoupled from concrete instantiation.

### Aspect-Oriented Programming (AOP)
To adhere to OCP, frameworks use AOP to apply cross-cutting concerns (such as transactions, security, and logging) to service boundaries dynamically using **Proxy decorators**. For instance, adding `@Transactional` in Spring Boot or `[Transaction]` in ASP.NET Core wraps the service class in a proxy container, injecting commit and rollback logic without modifying the service's source code.


### When SOLID Hurts: The Trade-off Analysis

SOLID principles are design heuristics, not commandments. Over-application creates its own category of architectural failures:

**Interface Segregation Overdose:** Splitting every interface into single-method contracts creates an explosion of types. A microservice with 47 single-method interfaces has replaced coupling with cognitive overload. The team spends more time navigating the interface graph than building features.

**Dependency Inversion Overhead:** In small microservices (< 500 lines), injecting every dependency through constructor parameters adds boilerplate without benefit. If a service has exactly one implementation of each dependency and will never be swapped, direct instantiation is simpler and more honest.

**Open-Closed Paralysis:** Designing every class for extension before you have a second use case is speculative generality. YAGNI (You Ain't Gonna Need It) often trumps OCP in early-stage systems. Add extension points when you have evidence of variation, not before.

**Liskov Substitution in Practice:** The classic Rectangle/Square violation is taught in every textbook, but the real-world impact is subtler. When your service contract promises idempotent retries but a subclass implementation has side effects on retry, you've violated LSP in a way that causes production incidents, not just type errors.

> The senior engineer's skill is knowing WHEN to apply SOLID and when the cure is worse than the disease.


> ⭐ **STAR Moment: The Mockability Test**
> 
> The ultimate test of a SOLID design is **mockability**. In a technical interview, explain that a correctly decoupled class can be unit-tested in isolation by mocking all of its interface dependencies. If you cannot test a method without spinning up a real database, an active web server, or a third-party messaging channel, your design violates the Dependency Inversion Principle.
