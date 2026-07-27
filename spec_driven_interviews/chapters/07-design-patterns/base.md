# Design Patterns in Enterprise Frameworks

> *"Design patterns are not templates to copy; they are vocabulary to describe architectural relationships."*


## Pattern Abuse in Interviews

Many software professionals prepare for design pattern questions by memorizing standard descriptions: "Singleton is a class with one instance," or "Factory creates objects." 

During a senior engineering interview, this is insufficient. A senior candidate must show how patterns solve real architectural problems, such as auditing transaction status, wrapping legacy systems, or handling dynamic business rules. You must also show that you know how these patterns are integrated into the frameworks you use daily (like Spring, Hibernate, or ASP.NET Core).

In this chapter, we will examine how AuraPay utilizes design patterns, focusing on the **Observer Pattern** to audit payment settlement events for financial compliance.


## Creational Patterns

Creational patterns abstract the instantiation process, decoupling your application from how objects are created and composed.

### The Builder Pattern
When constructing complex domain objects like AuraPay's `TransactionRecord`, constructors with ten parameters lead to unreadable code. The **Builder Pattern** solves this, allowing you to build objects step-by-step while maintaining immutability:

{{ inject('code_block_2.md') }}

### The Factory Pattern
When the core ledger processor needs to route a payment, it uses a **Factory Pattern** to dynamically instantiate the correct `SettlementRoute` processor based on the transaction metadata (such as routing cards via Visa vs. executing ACH).

### The Singleton Pattern (Creational Deep-Dive)
The Singleton pattern guarantees that a class has only one instance and provides a global point of access to it. In multi-threaded enterprise engines (such as a shared connection pool managed by HikariCP), writing a thread-safe Singleton requires **Double-Checked Locking**:

{{ inject('code_block_4.md') }}

> **Warning for Senior Candidates:** In cloud-native systems, classical Singletons are often considered an anti-pattern:
> 1. **Testing Complexity:** They introduce global mutable state, making parallel unit tests prone to side effects.
> 2. **Scalability limits:** A Singleton is only single per JVM instance. If your service scales out to ten microservice containers, you have ten connection pool instances, not one.
> 3. **IoC Managed Singletons:** Modern systems delegate singleton lifecycle management to Dependency Injection (IoC) containers rather than hardcoding static `getInstance()` methods.


## Structural Patterns

Structural patterns explain how to assemble objects and classes into larger structures while keeping these structures flexible and efficient.

### The Adapter Pattern
In banking-grade environments, you must frequently integrate with legacy core systems (e.g., COBOL-based mainframes or SOAP APIs). 
The **Adapter Pattern** wraps the legacy API with a clean interface that complies with your domain. For example, a `LegacySoapAdapter` implements the modern `LedgerRepository` interface, converting domain calls into SOAP requests under the hood.

### The Decorator Pattern
If you need to add auditing, metrics, or retry behaviors to transaction execution, do not pollute the core processing code. Use a **Decorator Pattern** to wrap the transaction processor, adding the cross-cutting concerns dynamically:

{{ inject('code_block_3.md') }}


## Behavioral Patterns

Behavioral patterns identify common communication patterns between objects and realize these patterns.

### The Strategy Pattern
AuraPay utilizes the **Strategy Pattern** to swap fee calculations dynamically. A `FlatFeeStrategy`, `TieredFeeStrategy`, and `MerchantDiscountRateStrategy` all implement `FeeCalculator`, allowing the routing engine to choose the strategy at runtime based on client profiles.

### The Observer Pattern (Injected)
When a transaction succeeds, external systems—such as the ledger audit index, fraud detection, and SMS notification dispatchers—must be notified. Hardcoding these calls inside the core transaction loop creates tight coupling.

We solve this using the **Observer Pattern**. The `TransactionEventPublisher` manages a list of observers and notifies them of transaction success or failure.

Here is the implementation:

{{ inject('code_block_1.md') }}

![Observer Pattern Class Diagram](visuals/observer_pattern.png){width=90%}

### The State Pattern (Behavioral Deep-Dive)
In payment platforms, transactions transition through a strict sequence of states: `CREATED` $\to$ `PENDING` $\to$ `SETTLED` or `FAILED` $\to$ `REFUNDED`.

Instead of writing a massive, hard-to-maintain switch block inside the transaction manager:

- We apply the **State Pattern**.
- We define a `TransactionState` interface representing the allowed operations (e.g., `approve()`, `fail()`, `refund()`).
- Each state is implemented as a concrete class (e.g., `PendingState`, `SettledState`).
- The transition logic is encapsulated inside each state class, preventing invalid state jumps (e.g., you cannot refund a `CREATED` transaction, only a `SETTLED` one), enforcing business invariants at runtime.


## Enterprise Integration & Data Access Patterns

In production-grade enterprise systems, designing clean persistence boundaries is as critical as GoF object coordination:

### Repository and Unit of Work Patterns

- **The Repository Pattern:** Mediates between the domain and data mapping layers using a collection-like interface for accessing domain objects (e.g., `LedgerRepository`). The business layer remains completely ignorant of whether data is stored in Postgres, MongoDB, or an in-memory map.
- **The Unit of Work Pattern:** Tracks all database-modifying operations (inserts, updates, deletes) during a single transaction context. Instead of each repository committing changes independently, the Unit of Work coordinates the commit boundary (e.g., Spring's `@Transactional` boundary or Entity Framework's `DbContext.SaveChanges()`). This guarantees that multiple repository updates succeed or fail together, protecting transactional boundaries.

### Data Transfer Object (DTO) Pattern
Exposing raw database entities directly over public REST/gRPC endpoints is a major security and design vulnerability. Doing so leaks internal database schemas, primary IDs, and sensitive columns (like password hashes).

- **The Solution:** Use **DTOs** (Data Transfer Objects) to define explicit data contracts for request inputs and response outputs. 
- **Mapping:** Utilize mapper libraries to map entities to DTOs before serialization, decoupling internal database schemas from external API consumers.

### Active Record vs. Data Mapper
When designing data access layers, select the persistence mapping style suited for the workload complexity:

- **Active Record (e.g., Ruby on Rails, Django ORM):** An approach where the entity class holds both the data attributes and the database access methods (e.g., `user.save()`, `user.delete()`). Very simple and fast to implement for CRUD applications. However, it violates SRP by coupling the domain model to database connection engines.
- **Data Mapper (e.g., Hibernate, JPA, Entity Framework):** An approach that completely separates data representation (the entity class) from database operations (the mapper/repository layer). The domain object remains database-ignorant, simplifying business unit testing and maintaining clean domain boundaries.


## Framework Integration: Patterns in the Wild

In senior interviews, you must connect patterns to the frameworks you use. Here is how modern enterprise engines implement them natively:

| Pattern | Framework Application | How It Works |
|---|---|---|
| **Factory** | Spring Bean Container | Spring's `BeanFactory` instantiates beans dynamically using reflection and dependency injection maps. |
| **Proxy** | Hibernate Lazy Loading | Hibernate generates proxy wrappers for entity relationships, loading child records from the database only when getter methods are invoked (Lazy Initialization). |
| **Observer** | Spring Application Events | Publishing events via `ApplicationEventPublisher` and consuming them using `@EventListener` decouples services asynchronously. |
| **Adapter** | Spring MVC Handlers | `HandlerAdapter` maps incoming HTTP requests to controller methods, shielding the servlet container from concrete execution signatures. |
| **Template Method** | Spring `JdbcTemplate` | `JdbcTemplate` defines the skeleton of database execution (opening connection, statement preparation, cleanup) while letting subclasses map rows to domain objects. |

> **Why is it called \"Spring\"?** Rod Johnson created the Spring Framework in 2003 as a reaction to the overwhelming complexity of **J2EE** (Java 2 Enterprise Edition). He chose the name *Spring* to represent a **fresh start** \u2014 a new season after the long, cold \"winter\" of J2EE's XML-heavy, boilerplate-ridden configuration. Spring made enterprise Java feel light and productive again, and the name perfectly captures that rebirth.

> **Why is it called \"Hibernate\"?** Gavin King created the ORM framework in 2001 and chose the name because Java objects *\"hibernate\"* (go dormant) inside the database and wake up when the application needs them. Just as animals hibernate through winter and emerge in spring, your domain objects are serialized into database rows and later rehydrated into live Java objects. The bear logo reinforces the metaphor.


> ⭐ **STAR Moment: The Framework Pattern Test**
> 
> During system design interviews, explain design patterns in terms of the framework concepts the interviewer already knows. Instead of drawing a generic observer diagram, say: *"We will implement this like a Spring ApplicationEventPublisher or a Kafka Event Broker, decoupling the transactional write thread from the audit and search indexing consumers."* This shows you understand patterns in modern, production-grade architectures.
