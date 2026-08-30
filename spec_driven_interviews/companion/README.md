# Spec-Driven Coding Interviews — Official Companion Repository

[![GitHub Repository](https://img.shields.io/badge/GitHub-hmallepally%2Fspec--driven--interviews-blue?logo=github)](https://github.com/hmallepally/spec-driven-interviews)
[![Java 21+](https://img.shields.io/badge/Java-21%2B-orange?logo=openjdk)](https://openjdk.org/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://www.python.org/)
[![.NET 8 / C# 12](https://img.shields.io/badge/.NET-8.0%20%2F%20C%23%2012-purple?logo=dotnet)](https://dotnet.microsoft.com/)
[![Docker Compose](https://img.shields.io/badge/Docker-Playground-blue?logo=docker)](docker-compose.yml)

This repository is the official companion codebase for the book **Spec-Driven Coding Interviews: The Advanced Reference Manual for Software Professionals** by Harinath Mallepally.

---

## Repository Architecture & Chapter Code Map

| Book Chapter & Topic | Java Implementation | Python Implementation | C# (.NET 8) Implementation | Distributed Lab |
| :--- | :--- | :--- | :--- | :--- |
| **Ch 01–03: Invariants & Ledgers** | java/craftsmanship/ledger | python/craftsmanship/ledger | csharp/Craftsmanship/Ledger | PostgreSQL Double-Entry DB |
| **Ch 04–05: Rich Domain & SOLID** | java/craftsmanship/domain | python/craftsmanship/domain | csharp/Craftsmanship/Domain | Clean Architecture Suite |
| **Ch 06: Functional Streams & Monads** | java/craftsmanship/streams | python/craftsmanship/streams | csharp/Craftsmanship/Streams | Parallel Stream Benchmarks |
| **Ch 07–08: Concurrency & Lock-Free** | java/concurrency/lockfree | python/concurrency/asyncio | csharp/Concurrency/Channels | Virtual Threads & CAS Ring Buffer |
| **Ch 09–13: 25 Canonical Patterns** | java/algorithms/patterns | python/algorithms/patterns | csharp/Algorithms/Patterns | 80 Automated Unit Test Suites |
| **Ch 14–15: Mock Assessment Sets** | java/algorithms/mocksets | python/algorithms/mocksets | csharp/Algorithms/MockSets | 20 Timed Assessment Runners |
| **Ch 16–17: 14 System Design Blueprints**| java/systemdesign/solutions| python/systemdesign/solutions| csharp/SystemDesign/Solutions | Docker Compose Topologies |
| **Ch 18: Outbox, Sagas & Jitter** | java/resiliency/outbox | python/resiliency/outbox | csharp/Resiliency/Outbox | Debezium CDC + PostgreSQL |
| **Ch 19: DB Isolation & Tokenization** | java/database/isolation | python/database/isolation | csharp/Database/Isolation | Postgres MVCC & AES-256 Vault |
| **Ch 21: Testcontainers & CI/CD** | java/testing/testcontainers| python/testing/testcontainers| csharp/Testing/Testcontainers | Pact Contract Testing Suite |
| **Ch 22: Kafka (KRaft) & RabbitMQ** | java/messaging/kafka | python/messaging/kafka | csharp/Messaging/Kafka | Kafka KRaft Multi-Broker Cluster |
| **Ch 23: AI/ML, RAG & Semantic Cache** | java/aiml/rag | python/aiml/rag | csharp/AIML/RAG | Qdrant / Milvus Vector Search Lab |

---

## Quickstart Guide

### 1. Launch the Local Distributed Infrastructure Playground
Spin up a complete distributed test environment containing Kafka (KRaft mode), PostgreSQL with pgvector, Redis Cluster, RabbitMQ, and Qdrant:

`ash
docker compose up -d
`

### 2. Run Multi-Language Algorithmic Test Suites

#### Java 21+ (Maven/Gradle)
`ash
cd java
mvn clean test
`

#### Python 3.12+ (pytest / uv)
`ash
cd python
pip install -r requirements.txt
pytest -v
`

#### C# 12 / .NET 8
`ash
cd csharp
dotnet test
`

---

## License & Citation
Copyright (C) 2026 Harinath Mallepally. All rights reserved.  
Open-source reference code licensed under the Apache 2.0 License.
