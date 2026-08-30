# Master System Design Solutions & Architectural Blueprints

> *"Senior system design is not about guessing technology names; it is the discipline of decomposing complex domain requirements into resilient, mathematically bounded distributed architectures."*

In the preceding chapter, we established the foundational principles of system design: Domain-Driven Design (DDD) bounded contexts, monolithic vs. microservices trade-offs, consistent hash sharding, CQRS, CAP theorem trade-offs, rate limiting, and caching architectures.

This chapter provides **14 Master End-to-End System Design Solutions**. Each solution represents a complete, production-grade architectural blueprint designed to answer real-world senior and staff engineering interview prompts across financial infrastructure, security, social platforms, geospatial dispatch, AI/ML, cloud storage, search engines, real-time messaging, task scheduling, collaborative editors, time-series observability, notification systems, and booking inventory management.


## The 7-Part Architecture Blueprint

To ensure complete clarity and zero ambiguity, every system design solution in this chapter follows a standardized **7-Part Architecture Blueprint**:

1. **Problem Statement & SLAs:** Precise functional requirements and quantitative non-functional SLAs (QPS, latency $p99$, availability, consistency).
2. **Capacity Estimation & Hardware Math:** First-principles mathematical derivations for network ingress bandwidth, memory footprints, and daily/annual disk storage.
3. **Visual Architecture Blueprint:** High-resolution structural diagrams illustrating Gateways, Load Balancers, Worker Pools, In-Memory Caches, Message Brokers, and Databases.
4. **Architectural Workflow & Mechanics:** Deep technical walkthrough of subsystem interactions, fault isolation boundaries, concurrency controls, and state pipelines.
5. **API Contracts & Interface Specs:** Production-grade REST JSON DTOs or gRPC Protobuf definitions.
6. **Database Schema & Data Model:** Relational PostgreSQL DDL, NoSQL Document Schema, or Spatial H3 Index structures.
7. **Execution Sequence & Staff Verbalization:** Step-by-step write/read paths, failure compensation, and high-scoring 45-minute verbalization scripts.


## Master System Design Solutions Catalog

| Solution | System Design Case Study | Primary Architectural Patterns | Generated Diagram Asset |
| :--- | :--- | :--- | :--- |
| **Solution 1** | **AuraPay:** Distributed Global Payment Gateway & Ledger | Idempotency Keys, Double-Entry SQL DDL, Transactional Outbox, Saga Orchestration | `visuals/arch_payment_gateway.png` |
| **Solution 2** | **ZenithTrade:** High-Frequency Order Matching Exchange | In-Memory OrderBook, Raft Consensus, Write-Ahead Log (WAL), CQRS Read Projections | `visuals/arch_matching_engine.png` |
| **Solution 3** | **ChiramTrust:** Distributed Rate Limiter & Fraud Pipeline | Redis Sliding Window Lua script, eBPF Kernel probes, Real-Time ML scoring, SOAR dynamic blocking | `visuals/arch_rate_limiter_fraud.png` |
| **Solution 4** | **Consumer Social:** Real-Time Social Feed & Video Streaming | Hybrid Push/Pull Timeline (Celebrity vs Regular), Redis Sorted Sets, S3 HLS/DASH Transcoding | `visuals/arch_social_video_platform.png` |
| **Solution 5** | **Geospatial:** Real-Time Ride-Sharing Dispatch (Uber/Lyft) | Uber H3 Hexagonal Spatial Indexing, QuadTrees, 3.3M QPS WebSocket ingest, Dynamic Surge Engine | `visuals/arch_rideshare_geospatial.png` |
| **Solution 6** | **AI Infrastructure:** Distributed Vector Search & RAG Engine | HNSW Vector Indexing (Milvus), Sparse BM25 (Elasticsearch), Reciprocal Rank Fusion, LLM Context Assembly | `visuals/arch_vector_rag_system.png` |
| **Solution 7** | **Cloud Storage:** Distributed File Sync Engine (Google Drive) | Rabin Fingerprint Chunking (4MB), Content-Addressable Block Store (S3), Vector Clock Sync | `visuals/arch_drive_sync_storage.png` |
| **Solution 8** | **Search Engine:** Distributed Web Crawler & Search Indexer | URL Frontier (Politeness Queue), SimHash Deduplication, Inverted Index Posting Lists, PageRank Graph | `visuals/arch_web_crawler_search.png` |
| **Solution 9** | **Real-Time Chat:** Distributed Messaging & Presence (Slack/Discord) | WebSocket Gateway, Redis Bitmaps Presence, Cassandra Sequence ID Store, Double Ratchet E2EE | `visuals/arch_chat_messaging_presence.png` |
| **Solution 10** | **Task Scheduler:** Distributed Workflow & Job Engine (Temporal) | Hierarchical Timing Wheel, Task Dependency DAG, Distributed Locks (etcd), Dead Letter Queue | `visuals/arch_task_scheduler_workflow.png` |
| **Solution 11** | **Collaborative Editor:** Real-Time CRDT & Whiteboard (Figma/Docs) | CRDT State Vector Sync, Operational Transformation (OT), Cursor Pub/Sub Stream, Snapshot Engine | `visuals/arch_collaborative_crdt_editor.png` |
| **Solution 12** | **Observability:** Distributed Time-Series Metrics TSDB (Prometheus) | Gorilla Delta-of-Delta Compression, Ring Buffer Chunk Store, Downsampling Aggregator, Alerting Rules | `visuals/arch_metrics_timeseries_observability.png` |
| **Solution 13** | **Notifications:** Multi-Channel Notification & Alerting Platform | Priority Queue Routing, Bloom Filter Deduplication, Channel Adapters (Email/SMS/Push/In-App), Rate Limiting | `visuals/arch_notification_platform.png` |
| **Solution 14** | **Booking Engine:** Distributed Hotel & Flight Inventory System | Redis Redlock Distributed Locks, Reservation Saga, Overbooking Prevention, Calendar Row-Level Locks | `visuals/arch_booking_inventory.png` |


## Master System Design Solutions

### Solution 1: AuraPay — Global Distributed Payment Gateway & Ledger

#### Problem Statement & SLAs
Design a global payment gateway and double-entry ledger capable of processing credit card and bank transactions across international merchants.

- **Target QPS:** 50,000 requests/sec peak.
- **Latency SLA:** $p99 < 150\text{ms}$ end-to-end API response.
- **Consistency SLA:** Strict financial consistency ($0$ double-spending, $0$ lost ledger entries).

#### Capacity Estimation & Hardware Math
- **Traffic Ingress:** $50,000 \text{ QPS} \times 1 \text{ KB payload} = 50 \text{ MB/sec} = 400 \text{ Mbps}$ network ingress.
- **Transaction Volume:** $50,000 \text{ tx/sec} \times 86,400 \text{ sec/day} = 4.32 \text{ billion tx/day}$.
  - Daily storage: $4.32 \times 10^9 \times 500 \text{ bytes} \approx 2.16 \text{ TB/day}$.
  - Annual storage: $\approx 788 \text{ TB/year}$.

#### Visual Architecture Blueprint
![AuraPay Payment Gateway & Ledger Architecture](visuals/arch_payment_gateway.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Edge Ingress & Fast Idempotency (API Gateway):**
   - Intercepts incoming payment requests bearing an `Idempotency-Key` header.
   - Queries a distributed fast store (Redis) to verify request state. If the key exists and is `COMPLETED`, the cached response is served immediately. If `PENDING`, concurrent duplicate calls are rejected.
2. **Synchronous Payment Processing:**
   - Forwards brand-new requests to the **Payment Processing Service**, which initiates an authorization call via the **Bank Adapter Service** (translating REST/JSON to legacy ISO 8583 / FIX protocols).
3. **Transactional Outbox Pattern (Dual-Write Prevention):**
   - The Payment Processing Service writes the updated payment entity and emits an outbox event into a single local relational database (**PostgreSQL**) within an atomic `BEGIN ... COMMIT` boundary.
   - An asynchronous relay worker (or CDC pipeline like Debezium) tails the outbox table and reliably publishes messages (`PaymentCreated`, `PaymentAuthorized`) to **Apache Kafka**.
4. **Decoupled Asynchronous Settlement & Double-Entry Ledger:**
   - The **Saga Orchestrator** consumes events from Kafka and coordinates the multi-step transaction.
   - Dispatches strict double-entry accounting commands to the **Ledger Service** (recording balanced immutable `DEBIT` and `CREDIT` rows with high-precision `NUMERIC(18, 4)` types).
5. **Compensating Actions:**
   - If downstream ledger validation or settlement fails, the Saga Orchestrator executes compensating transactions, marks the Redis idempotency key as `FAILED`, and issues webhook failure alerts.

#### API Contracts & Interface Specs
```json
// POST /v1/payments (HTTP REST / Idempotency Protected)
Header: Idempotency-Key: "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
{
  "account_id": "acc_usr_99812",
  "merchant_id": "mch_stripe_001",
  "amount": 14999,
  "currency": "USD",
  "payment_method_token": "tok_visa_4412"
}
```

#### Database Schema & Data Model (PostgreSQL DDL)
```sql
CREATE TABLE ledger_entries (
    entry_id UUID PRIMARY KEY,
    transaction_id UUID NOT NULL,
    account_id UUID NOT NULL,
    entry_type VARCHAR(10) CHECK (entry_type IN ('DEBIT', 'CREDIT')),
    amount NUMERIC(18, 4) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_ledger_account ON ledger_entries(account_id, created_at);
CREATE INDEX idx_ledger_transaction ON ledger_entries(transaction_id);
```

#### Step-by-Step Execution Sequence
1. **Ingest & Idempotency Check:** API Gateway intercepts request, checks Redis for `Idempotency-Key`. If present and `COMPLETED`, returns cached response. If new, sets `PENDING`.
2. **Payment Processing:** Payment Service authorizes funds via external Bank Adapter.
3. **Transactional Outbox:** Payment Service writes transaction record AND an Outbox event into PostgreSQL in a single local ACID transaction.
4. **Asynchronous Ledger Event:** Outbox Worker relays `PaymentAuthorized` event to Kafka topic (`payments.settlement`).
5. **Saga Orchestration:** Saga Orchestrator consumes event, executes double-entry debit/credit commits in Ledger DB, and updates status to `COMPLETED` in Redis.
6. **Failure Compensation:** If bank authorization fails or ledger constraint is violated, Saga Orchestrator publishes a `PaymentFailed` event, reverses any provisional ledger entries, updates the idempotency key to `FAILED`, and triggers a webhook notification to the merchant.

#### Staff-Level Interview Verbalization
> *"In designing AuraPay, we enforce two critical invariants: API idempotency via Redis atomic locks, and financial double-entry balance preservation via the Transactional Outbox pattern. By decoupling bank network authorization from ledger settlement using Kafka, we guarantee that database write latencies never block the client response path."*


### Solution 2: ZenithTrade — High-Frequency Order Matching Exchange

#### Problem Statement & SLAs
Design a high-frequency cryptocurrency and equity order matching exchange.

- **Target Throughput:** 100,000 orders/sec peak per partition.
- **Latency SLA:** Sub-millisecond matching latency ($p99 < 1\text{ms}$).
- **Availability:** $99.999\%$ uptime with sub-second active-passive failover.

> **Why Single-AZ Raft?** Cross-AZ Raft round-trips add 1–5ms network latency, violating the sub-millisecond SLA. The matching engine Raft cluster is co-located within a single Availability Zone using kernel bypass (DPDK) and NVMe direct I/O. Cross-region disaster recovery uses asynchronous WAL shipping rather than synchronous Raft.

#### Capacity Estimation & Hardware Math
- **Order Payload:** 200 bytes per order.
- **Network Bandwidth:** $100,000 \text{ QPS} \times 200 \text{ B} = 20 \text{ MB/sec} = 160 \text{ Mbps}$.
- **In-Memory OrderBook Memory:** $10,000,000 \text{ active open orders} \times 128 \text{ B/order} \approx 1.28 \text{ GB RAM}$ per instrument. Fits comfortably in RAM.

#### Visual Architecture Blueprint
![ZenithTrade High-Frequency Order Matching Architecture](visuals/arch_matching_engine.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Deterministic Order Partitioning:**
   - A **Consistent Hash Ring Router** inspects the incoming `instrument_id` (e.g., `BTC-USD`, `ETH-USD`) and routes the order to the designated shard/partition, preventing cross-symbol lock contention.
2. **Single-AZ In-Memory Matching Engine (Raft Group):**
   - To strictly enforce sub-millisecond execution ($p99 < 1\text{ms}$), Raft consensus groups are co-located within a single Availability Zone. This bypasses cross-AZ network round-trips ($1\text{--}5\text{ms}$).
   - Employs **DPDK (Data Plane Development Kit)** for kernel-bypass networking and direct NVMe I/O.
   - The **Leader Node** updates in-memory limit order books (LOB) and commits sequential operations to an append-only Write-Ahead Log (WAL).
   - Hot standby **Follower Nodes** replicate the WAL for immediate active-passive failover.
3. **CQRS & Downstream Projections:**
   - Trade executions bypass disk bottlenecks on the read path via CQRS projections.
   - Matched trades stream through **Apache Kafka** out to **Redis** (for real-time order-book dashboards and ticker feeds) and **Elasticsearch** (for historical trade analytics, regulatory compliance, and user trade history).
4. **Disaster Recovery (DR):**
   - Asynchronous WAL shipping replicates state across geographically distinct regions without blocking the critical matching path.

#### API Contracts & Interface Specs (gRPC Protobuf)
```protobuf
syntax = "proto3";
package zenithtrade;

message PlaceOrderRequest {
    string idempotency_key = 1;
    string account_id = 2;
    string instrument_id = 3; // e.g., "BTC-USD"
    enum Side { BUY = 0; SELL = 1; }
    Side side = 4;
    int64 price_in_cents = 5;
    int64 quantity_in_satoshis = 6;
}
```

#### Database & In-Memory Data Structure
- **In-Memory OrderBook:** `TreeMap<Long, DoublyLinkedList<Order>>`
  - Bids: Sorted descending by price.
  - Asks: Sorted ascending by price.
  - Match lookup: $O(1)$ at tree head; Insert/Cancel: $O(\log P)$ where $P$ is distinct price levels.
- **Replication Log:** Write-Ahead Log (WAL) streamed via Raft consensus group.

#### Step-by-Step Execution Sequence
1. **Instrument Routing:** Consistent Hash Ring routes incoming order by `instrument_id` to designated partition Raft Leader node.
2. **WAL Append:** Leader appends order to sequential Write-Ahead Log (WAL) on NVMe SSD and replicates to Raft Followers.
3. **In-Memory Matching:** Engine matches order against opposing tree head based on Price-Time priority.
4. **CQRS Projection:** Engine emits `TradeExecuted` event to Kafka. Read workers update Elasticsearch (search) and Redis (order book display).

#### Staff-Level Interview Verbalization
> *"ZenithTrade decouples in-memory order matching from disk and network bottlenecks. We partition matching by Instrument ID using consistent hashing. Each matching engine node runs as a single-threaded in-memory Raft Leader with append-only WAL logging, achieving sub-millisecond execution without lock contention."*


### Solution 3: ChiramTrust — Distributed Rate Limiter & Real-Time Fraud Pipeline

#### Problem Statement & SLAs
Design an enterprise-grade rate limiter and real-time security fraud detection pipeline.

- **Target Throughput:** 500,000 requests/sec across 50 microservices.
- **Latency SLA:** Rate limiting evaluation $p99 < 2\text{ms}$. Fraud scoring delay $p99 < 50\text{ms}$.

#### Capacity Estimation & Hardware Math
- **Rate Limit Keys:** 100 million active users.
- **Redis Memory:** $100 \times 10^6 \text{ keys} \times 64 \text{ bytes} \approx 6.4 \text{ GB RAM}$. Redis Cluster easily handles state.

#### Visual Architecture Blueprint
![ChiramTrust Distributed Rate Limiter & Fraud Pipeline](visuals/arch_rate_limiter_fraud.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Low-Latency Edge Rate Limiting:**
   - Built directly into the **API Gateway (NGINX / Envoy)**.
   - Leverages a **Redis Cluster** running an atomic **Sliding Window Counter** implemented via Lua scripts (`ZREMRANGEBYSCORE`, `ZCARD`, `ZADD`, `EXPIRE`) to eliminate distributed race conditions while enforcing sliding-window rate limits.
2. **Kernel-Level Observability via eBPF:**
   - Embeds **eBPF (Extended Berkeley Packet Filter)** hooks directly inside the OS kernel to capture low-overhead network events (`SYN`, `ACK`, TCP/IP payloads) with zero user-space context-switching cost.
   - A local **Telemetry Agent** gathers and streams telemetry over gRPC into Kafka.
3. **Asynchronous ML Fraud Inference Pipeline:**
   - High-throughput Kafka topics (`API_GATEWAY_EVENTS`, `NETWORK_TELEMETRY`) feed stream workers that extract dynamic behavioral features (e.g., velocity spikes, geo-hopping, credential stuffing).
   - Deep learning fraud models score transactions in real time.
4. **Closed-Loop SOAR Feedback:**
   - High-risk fraud scores trigger the **Security Orchestration (SOAR)** platform to dynamically inject updated IP blocklists directly back into the API Gateway's edge filters.
   - Historical logs land in a **Data Lake (S3 / HDFS)** for continuous offline model retraining.

#### Redis Lua Script (Sliding Window Counter)
```lua
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local clearBefore = now - window

redis.call('ZREMRANGEBYSCORE', key, 0, clearBefore)
local currentRequests = redis.call('ZCARD', key)
if currentRequests < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, math.ceil(window / 1000))
    return 1
else
    return 0
end
```

#### Sliding Window Counter Approximation Formula

To achieve sub-millisecond edge rate limiting without storing individual request timestamps in Sorted Sets, the **Sliding Window Counter** approximates rolling volume using two fixed-window counters:

$$\text{Estimated Count} = M_{\text{current}} + M_{\text{previous}} \times \left(1 - \frac{t - t_{\text{start}}}{W}\right)$$

- $M_{\text{current}}$: Request count in current 60-second window.
- $M_{\text{previous}}$: Request count in previous 60-second window.
- $t - t_{\text{start}}$: Elapsed time within current window (in seconds).
- $W$: Window duration (60 seconds).
- **Accuracy Bound:** Maximum error is strictly bounded below $0.05\%$ under steady traffic, consuming only 16 bytes of RAM per client key (`INCRBY` / `GET`).

> **DDoS Fallback:** Under volumetric attack, `ZREMRANGEBYSCORE` complexity rises to $O(\log N + M)$ where $M$ is evicted elements. If $M$ spikes, fall back to a fixed-window counter (`INCR key; EXPIRE key window`) to protect the single-threaded Redis event loop.

#### API Contracts & Interface Specs
```json
// GET /v1/rate-limit/check
Header: X-Client-IP: "203.0.113.42"
Header: X-Service-ID: "payment-svc"
Response (200 OK):
{
  "allowed": true,
  "remaining": 847,
  "limit": 1000,
  "window_seconds": 60,
  "retry_after_ms": null
}
// Response (429 Too Many Requests):
{
  "allowed": false,
  "remaining": 0,
  "limit": 1000,
  "window_seconds": 60,
  "retry_after_ms": 12400
}
```

#### Database Schema & Data Model (PostgreSQL DDL)
```sql
CREATE TABLE rate_limit_policies (
    policy_id UUID PRIMARY KEY,
    service_id VARCHAR(64) NOT NULL,
    endpoint_pattern VARCHAR(255) NOT NULL,
    max_requests INT NOT NULL,
    window_seconds INT NOT NULL,
    burst_multiplier DECIMAL(3,1) DEFAULT 1.5
);
CREATE TABLE fraud_events (
    event_id UUID PRIMARY KEY,
    client_ip INET NOT NULL,
    fraud_score DECIMAL(5,4) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    action_taken VARCHAR(20) CHECK (action_taken IN
        ('ALLOWED', 'THROTTLED', 'BLOCKED')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_fraud_ip ON fraud_events(client_ip, created_at);
CREATE INDEX idx_policies_service ON rate_limit_policies(service_id);
```

#### Step-by-Step Execution Sequence
1. **Gateway Evaluation:** Envoy API Gateway intercepts request, executes Lua script in Redis cluster. If `0`, returns `429 Too Many Requests`.
2. **eBPF Telemetry Hook:** Linux kernel eBPF probe captures TCP connection metadata without user-space context switching overhead.
3. **Streaming Scoring:** Kernel telemetry streams to Kafka (`telemetry.events`). Real-time ML worker scores fraud probability.
4. **SOAR Enforcement:** If fraud score $> 0.90$, automated Security Orchestration (SOAR) pushes IP to Redis block list, dynamically dropping subsequent requests at the gateway.

#### Staff-Level Interview Verbalization
> *"Our design pairs atomic Lua scripts in Redis for sliding-window rate enforcement with eBPF kernel probes for zero-overhead telemetry gathering. This guarantees sub-2ms throttling overhead while feeding an asynchronous ML pipeline that dynamically blocks malicious IPs."*


### Solution 4: Consumer Scale — Real-Time Social Feed & Video Streaming Platform

#### Problem Statement & SLAs
Design a consumer social timeline (Twitter/X) and adaptive video streaming platform (YouTube).

- **Users:** 300 million daily active users (DAU).
- **Latency SLA:** Timeline generation $p99 < 200\text{ms}$. Video start-to-play $< 1.5\text{s}$.

#### Capacity Estimation & Hardware Math
- **Write QPS (Posts):** $5,000 \text{ posts/sec}$.
- **Read QPS (Timeline):** $300,000 \text{ requests/sec}$ ($60:1$ read/write ratio).
- **Video Storage:** $50,000 \text{ hours uploaded/day} \times 10 \text{ GB/hour} = 500 \text{ TB/day}$.

#### Visual Architecture Blueprint
![Consumer Social Feed & Video Streaming Architecture](visuals/arch_social_video_platform.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Hybrid Fan-Out Feed Strategy:**
   - **Regular Users (<10k followers) — Fan-out-on-write (Push):** Posts are pushed asynchronously into every follower's timeline stored as Redis Sorted Sets (`ZADD timeline:{follower_id} {timestamp} {post_id}`), guaranteeing ultra-fast $O(1)$ feed reads.
   - **Celebrity / High-Follower Accounts (>10k followers) — Fan-out-on-read (Pull):** High-follower posts are saved in a scalable NoSQL store (Cassandra / DynamoDB) and merged into the user's timeline dynamically at read time, avoiding fan-out write amplification.
2. **Adaptive Bitrate Video Processing Pipeline:**
   - Clients upload large video payloads directly to raw S3 buckets using **S3 Presigned URLs**, keeping video data plane traffic off the API Gateway.
   - S3 upload notifications enqueue encoding jobs into **AWS SQS**.
   - Auto-scaling FFmpeg worker clusters transcode video into multi-bitrate HLS and DASH segments (1080p, 720p, 480p, 360p manifests `.m3u8` and `.ts` chunk segments) stored in public S3 buckets.
3. **Global CDN Edge Distribution:**
   - Video manifests and chunk files are aggressively cached across edge PoPs, ensuring start-to-play times under $1.5\text{s}$.

> **Why Hybrid Push/Pull?** Pure push fan-out for celebrity accounts (10M+ followers) would require writing 10M Redis entries per post — a 30-second blocking storm. Pure pull adds latency for regular users. The hybrid model caps fan-out cost at the celebrity threshold while keeping regular timeline reads at $O(1)$ Redis `ZRANGEBYSCORE`.

#### API Contracts & Interface Specs
```json
// POST /v1/posts (Create Post)
Header: Authorization: Bearer <token>
{
  "author_id": "usr_291a8f",
  "content_text": "Exploring system design patterns",
  "media_urls": ["s3://bucket/vid_chunk_001.mp4"],
  "visibility": "PUBLIC"
}
// GET /v1/timeline?user_id=usr_42&cursor=ts_172800&limit=20
Response (200 OK):
{
  "posts": [
    {"post_id": "p_8812", "author_id": "usr_291a8f",
     "content_text": "...", "created_at": "2026-08-25T10:00:00Z"}
  ],
  "next_cursor": "ts_172780"
}
```

#### Database Schema & Data Model (PostgreSQL + Redis)
```sql
CREATE TABLE posts (
    post_id UUID PRIMARY KEY,
    author_id UUID NOT NULL,
    content_text TEXT,
    media_manifest_url VARCHAR(512),
    visibility VARCHAR(10) DEFAULT 'PUBLIC',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_posts_author ON posts(author_id, created_at DESC);
-- Redis Timeline: ZADD timeline:{follower_id} {timestamp} {post_id}
-- Redis Celebrity Outbox: ZADD outbox:{celebrity_id} {ts} {post_id}
```

#### Staff-Level Interview Verbalization
> *"To solve the celebrity fan-out bottleneck, we implement a Hybrid Push/Pull timeline architecture. Regular posts fan out asynchronously into Redis Sorted Sets, while high-follower accounts are merged on read. Video content uses S3 presigned uploads and multi-bitrate HLS transcoding distributed via global CDN edges."*


### Solution 5: Consumer Scale — Real-Time Ride-Sharing Geospatial Dispatch System

#### Problem Statement & SLAs
Design a real-time ride-sharing dispatch system (Uber/Lyft).

- **Scale:** 10 million active drivers streaming GPS locations every 3 seconds.
- **Latency SLA:** Driver-rider matching $< 3\text{ seconds}$. Location update ingest $< 100\text{ms}$.

#### Capacity Estimation & Hardware Math
- **Location Ingest QPS:** $10,000,000 \text{ drivers} / 3 \text{ seconds} \approx 3.33 \text{ million QPS}$.
- **Network Ingress:** $3.33 \times 10^6 \times 64 \text{ bytes} \approx 213 \text{ MB/sec} = 1.7 \text{ Gbps}$.

#### Visual Architecture Blueprint
![Ride-Sharing Geospatial Dispatch System](visuals/arch_rideshare_geospatial.png){width=95%}

#### Architectural Workflow & Mechanics
1. **High-Throughput Telemetry Ingest:**
   - Mobile driver applications stream continuous GPS coordinates (Lat/Lon, Driver ID, Status) over persistent **Secure WebSockets (WSS)** into a scalable Kafka ingest pipeline handling 3.33M location QPS.
2. **Hierarchical Geospatial Indexing:**
   - Converts spatial coordinates into **Uber H3 hexagonal grid cells** (resolutions 8–10) and **Geohashes**.
   - Current driver locations and cell membership counters are cached in **Redis Hashes and Sorted Sets** (`GEOADD active_drivers:{h3_cell} lon lat driver_id`) for sub-millisecond neighborhood radius queries.
3. **Dynamic Surge Pricing Engine:**
   - Ingests demand signals (rider open-app searches) and supply signals (available drivers per H3 hexagon) in real time.
   - Computes localized surge multipliers ($1.0\times\text{--}3.5\times$) to balance marketplace equilibrium:
     $$\text{Surge Multiplier} = \min\left(3.5, \max\left(1.0, \frac{\text{Unmatched Rider Requests}}{\text{Available Drivers in H3 Cell}}\right)\right)$$

4. **Search & Dispatch Matching Engine:**
   - Employs recursive spatial partitioning (Quadtrees / $k$-NN search) to identify the optimal top-ranked active drivers near the pickup point within a 3km radius.
   - Dispatches trip offers to drivers over WebSockets; completed trip records persist into a relational **PostgreSQL Trip History DB**.

> **Why H3 over Geohash or S2?** Geohash rectangles create edge discontinuities where neighbors share no prefix. S2 cells are complex to implement. Uber H3 hexagons provide uniform distance to all 6 contiguous neighbors ($122\text{ meters}$ edge length at resolution 9) and smooth spatial aggregation without edge artifacts — critical for accurate surge pricing across cell boundaries.

#### Haversine Great-Circle Distance Metric

To compute the spherical surface distance between rider coordinates $(\phi_1, \lambda_1)$ and driver coordinates $(\phi_2, \lambda_2)$ with earth radius $R \approx 6,371\text{ km}$:

$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$

Where $\Delta \phi = \phi_2 - \phi_1$ (latitude difference in radians) and $\Delta \lambda = \lambda_2 - \lambda_1$ (longitude difference in radians). Redis Geo internally computes this spherical distance via geohash integer bit-interleaving in $\mathcal{O}(1)$ time.

#### API Contracts & Interface Specs
```json
// POST /v1/trips/request (Rider requests a trip)
Header: Authorization: Bearer <token>
{
  "rider_id": "rdr_55812",
  "pickup": {"lat": 37.7749, "lon": -122.4194},
  "dropoff": {"lat": 37.3382, "lon": -121.8863},
  "ride_type": "POOL"
}
// Response (201 Created):
{
  "trip_id": "trip_a91f2",
  "surge_multiplier": 1.4,
  "estimated_fare_cents": 3250,
  "matched_driver_id": "drv_77201",
  "eta_seconds": 180
}
```

#### Database Schema & Data Model (PostgreSQL + Redis Geo)
```sql
CREATE TABLE trips (
    trip_id UUID PRIMARY KEY,
    rider_id UUID NOT NULL,
    driver_id UUID,
    pickup_h3_index BIGINT NOT NULL,
    dropoff_h3_index BIGINT NOT NULL,
    status VARCHAR(20) CHECK (status IN
        ('REQUESTED','MATCHED','IN_PROGRESS','COMPLETED','CANCELLED')),
    surge_multiplier DECIMAL(3,1) DEFAULT 1.0,
    fare_cents INT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_trips_driver ON trips(driver_id, created_at DESC);
CREATE INDEX idx_trips_rider ON trips(rider_id, created_at DESC);
CREATE INDEX idx_trips_h3 ON trips(pickup_h3_index);
-- Redis Geo: GEOADD active_drivers:{h3_cell} lon lat driver_id
```

#### Step-by-Step Execution Sequence
1. **GPS Telemetry Ingest:** Driver app streams `(driver_id, lat, lon, status)` via WebSocket to WSS Load Balancers.
2. **H3 Cell Mapping:** Location worker calculates H3 hexagon cell key and updates Redis Geo index with 10-second TTL.
3. **Trip Request & Surge Calculation:** Rider requests trip. Surge Pricing Engine calculates demand/supply ratio in H3 cell:
   $$\text{Surge Multiplier} = \min\left(3.5, \max\left(1.0, \frac{\text{Unmatched Rider Requests}}{\text{Available Drivers in H3 Cell}}\right)\right)$$

4. **KNN Dispatch Match:** Dispatch Engine queries QuadTree / Redis Geo for nearest available drivers within 3km radius, sending dispatch offer to optimal driver via WebSocket.

#### Staff-Level Interview Verbalization
> *"Our ride-sharing dispatch system uses Uber H3 hexagonal spatial indexing in Redis to partition 3.3 million QPS of GPS telemetry. We compute real-time surge multipliers per H3 cell based on demand-supply ratios and execute $k$-nearest neighbor driver matching via QuadTrees."*


### Solution 6: Modern AI/ML — Distributed Vector Search & RAG Knowledge Engine

#### Problem Statement & SLAs
Design an enterprise Retrieval-Augmented Generation (RAG) knowledge search system over millions of unstructured documents.

- **Document Scale:** 100 million document chunks.
- **Latency SLA:** Hybrid vector search $p99 < 50\text{ms}$. LLM generation $p99 < 2\text{s}$.

#### Capacity Estimation & Hardware Math
- **Embedding Dimensions:** 768-dimensional float32 vectors ($768 \times 4 \text{ bytes} = 3,072 \text{ bytes/vector}$).
- **Vector RAM Index:** $100,000,000 \text{ vectors} \times 3,072 \text{ bytes} \approx 307.2 \text{ GB RAM}$. Fits across a 4-node HNSW Milvus/Qdrant cluster.

#### Visual Architecture Blueprint
![Distributed Vector Search & RAG Architecture](visuals/arch_vector_rag_system.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Document Ingestion & Chunking Pipeline:**
   - Ingestion workers scrape documents (PDFs, HTML, CMS databases), strip boilerplate, and segment text into overlapping semantic chunks (e.g., 512 tokens with 64-token overlap).
   - Chunks are passed to embedding model worker clusters (e.g., BGE, Cohere, text-embedding-3) running on GPU inference clusters.
2. **Dual Index Storage Architecture:**
   - **Dense Vectors:** High-dimensional embeddings are indexed using **HNSW (Hierarchical Navigable Small World)** graphs in vector databases (Milvus / Qdrant).
   - **Sparse Lexical Keywords:** Raw text chunks are tokenized and stored in **BM25 / Elasticsearch / OpenSearch** indexes.
   - Chunk metadata and lineage are maintained in PostgreSQL.
3. **Hybrid Retrieval & Reciprocal Rank Fusion (RRF):**
   - User queries execute simultaneous dense ANN vector similarity search ($\mathcal{O}(\log N)$) and sparse BM25 keyword matching.
   - Results are unified and reranked using Reciprocal Rank Fusion:
     $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$

4. **Context Assembly & LLM Generation:**
   - The top reranked chunks are filtered, formatted into prompt context windows, and sent to LLMs (GPT-4, Claude, Llama 3) to generate grounded, hallucination-free answers.

#### API Contracts & Interface Specs
```json
// POST /v1/search (Hybrid RAG Query)
Header: Authorization: Bearer <token>
{
  "query": "How does circuit breaker pattern prevent cascade failures?",
  "top_k": 5,
  "rerank": true,
  "generate_answer": true
}
// Response (200 OK):
{
  "chunks": [
    {"chunk_id": "chk_9a12", "score": 0.934,
     "text": "The circuit breaker transitions between CLOSED..."}
  ],
  "generated_answer": "Circuit breakers prevent cascade...",
  "model": "llama-3-70b",
  "latency_ms": 1420
}
```

#### Database Schema & Data Model (PostgreSQL + Milvus)
```sql
CREATE TABLE document_chunks (
    chunk_id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    chunk_index INT NOT NULL,
    content_text TEXT NOT NULL,
    token_count INT NOT NULL,
    embedding_model VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_chunks_doc ON document_chunks(document_id, chunk_index);
-- Milvus Collection: 768d float32 HNSW index on chunk_id
-- Elasticsearch Index: BM25 full-text on content_text
```

#### Step-by-Step Execution Sequence
1. **Ingestion & Chunking:** Document Processing Service splits documents into 512-token overlapping chunks.
2. **Embedding Generation:** Embedding Worker cluster (BGE / Cohere model) generates 768d vectors and indexes into Milvus (HNSW) and PostgreSQL (metadata).
3. **Query Embedding & Hybrid Retrieval:** User query is embedded into a vector. Parallel queries execute against Milvus (dense vector) and Elasticsearch (sparse BM25).
4. **Context Window Assembly:** RRF Ranker merges top 5 chunks, passes context prompt to LLM (GPT-4 / Claude / Llama 3) for response generation.

#### Staff-Level Interview Verbalization
> *"Our RAG architecture combines dense HNSW vector search with sparse BM25 keyword search via Reciprocal Rank Fusion. This hybrid retrieval approach captures both semantic intent and exact code/identifier tokens, populating LLM context windows in under 50ms."*


### Solution 7: Cloud Storage — Distributed File Storage & Sync Engine (Google Drive / Dropbox)

#### Problem Statement & SLAs
Design a distributed file storage and sync platform capable of handling multi-gigabyte files across millions of devices.

- **Scale:** 500 million registered users, 100 million active files synced per day.
- **Latency SLA:** File metadata sync $< 200\text{ms}$. Delta upload latency proportional to modified byte count only.
- **Consistency SLA:** Strict block immutability and file version ordering (Vector Clocks).

#### Capacity Estimation & Hardware Math
- **Average File Size:** 2 MB average. Daily storage ingest: $100 \times 10^6 \times 2 \text{ MB} = 200 \text{ TB/day}$.
- **Block Size (Chunking):** 4 MB fixed/variable chunk size via Rabin Fingerprinting.
- **Metadata Storage:** 1 billion files $\times 1 \text{ KB metadata} = 1 \text{ TB}$ metadata DB index in CockroachDB/PostgreSQL.

#### Visual Architecture Blueprint
![Distributed File Storage & Sync Engine Architecture](visuals/arch_drive_sync_storage.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Client-Side File Watching & Chunking:**
   - A background OS file watcher monitors local directory changes.
   - Modified files are partitioned into dynamic chunk boundaries using **Rabin Fingerprinting** content-defined chunking (CDC):
     $$H(b_1, \dots, b_k) = \left(\sum_{i=1}^k b_i \cdot p^{k-i}\right) \pmod M$$

   - When rolling hash $H \equiv 0 \pmod D$ (where $D = 4\text{ MB} = 2^{22}$), a chunk boundary is declared. Inserting a byte at the start of a 10GB file shifts only the first chunk boundary; all remaining chunks retain identical hashes, eliminating 99.9% of re-upload bandwidth.
2. **Content-Addressable Storage (CAS) & Deduplication:**
   - Each chunk generates a cryptographic hash (SHA-256).
   - The client performs a metadata lookup against the server. If the hash exists, upload is bypassed and the server simply increments `reference_count` in the `file_blocks` table.
3. **Chunk Upload & Block Store:**
   - New, unique chunks are streamed directly to **Content Addressable Block Storage (AWS S3)** via presigned URLs.
4. **Metadata & Conflict Resolution:**
   - File trees, paths, chunk lists, and permissions are stored in a distributed relational database (PostgreSQL / CockroachDB).
   - Background sync workers notify connected client devices over persistent WebSocket connections to pull changed chunk manifests.

#### API Contracts & Interface Specs
```json
// POST /v1/files/upload_chunk
Header: Authorization: Bearer <token>
{
  "file_id": "file_88192a3",
  "chunk_index": 4,
  "chunk_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "byte_size": 4194304
}
```

#### Database Schema & Data Model (SQL DDL)
```sql
CREATE TABLE file_blocks (
    block_hash VARCHAR(64) PRIMARY KEY, -- SHA-256
    storage_url VARCHAR(255) NOT NULL,
    byte_size INT NOT NULL,
    reference_count INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE file_manifests (
    file_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    vector_clock JSONB NOT NULL DEFAULT '{}', -- e.g. {"client_A": 3, "client_B": 1}
    block_hashes JSONB NOT NULL -- Ordered list of block_hash UUIDs
);
CREATE INDEX idx_manifests_user ON file_manifests(user_id);
```

#### Step-by-Step Execution Sequence
1. **Rabin Chunking & Hash Lookup:** Client File Watcher detects change, splits file into 4MB chunks using Rabin Fingerprinting, and calculates SHA-256 checksum per chunk.
2. **Deduplication Check:** Client queries Deduplication Service with block hashes. If hash exists in `file_blocks`, upload is skipped ($O(1)$ cross-user deduplication).
3. **Block Store Upload:** New unique chunks are uploaded directly to S3 block storage using presigned URLs.
4. **Manifest Commit & Sync Event:** Client updates file manifest in CockroachDB. Async Sync Workers notify other user devices via WebSockets.

#### Staff-Level Interview Verbalization
> *"Our file sync architecture uses content-addressable storage with Rabin Fingerprint chunking to achieve global cross-user deduplication. By separating immutable 4MB block uploads to S3 from lightweight metadata manifest commits in CockroachDB, we minimize bandwidth consumption and guarantee seamless delta sync."*


### Solution 8: Search Engine — Distributed Web Crawler & Search Indexer (Google Search)

#### Problem Statement & SLAs
Design a distributed web crawler and search indexer capable of crawling billions of web pages and updating an inverted search index.

- **Scale:** 10 billion web pages crawled per month ($\approx 3,850 \text{ pages/sec}$).
- **Latency SLA:** Search query execution $p99 < 100\text{ms}$ over a 50-billion document corpus.
- **Politeness SLA:** Enforce robots.txt and strict per-host rate limits (no more than 1 request/sec per domain).

#### Capacity Estimation & Hardware Math
- **Page Size:** 100 KB average HTML page.
- **Storage Ingest:** $3,850 \text{ pages/sec} \times 100 \text{ KB} = 385 \text{ MB/sec} = 3.08 \text{ Gbps}$ ingress.
- **Monthly Storage:** $10 \times 10^9 \text{ pages} \times 100 \text{ KB} = 1 \text{ PB/month}$.

#### Visual Architecture Blueprint
![Distributed Web Crawler & Inverted Search Indexer Architecture](visuals/arch_web_crawler_search.png){width=95%}

#### Architectural Workflow & Mechanics
1. **URL Frontier & DNS Resolution:**
   - The URL Frontier manages crawling priority queues while enforcing domain politeness rules (rate limits per host, `robots.txt` compliance).
   - Uses an in-memory DNS caching layer to eliminate redundant DNS round trips.
2. **HTML Parsing & Near-Duplicate Filtering (SimHash Algorithm):**
   - Fetched documents are parsed to extract outgoing links (fed back to the frontier) and clean textual content.
   - Computes a **64-bit SimHash fingerprint** per document:
     $$V[i] = \sum_{w \in \text{Doc}} \text{weight}(w) \times \begin{cases} +1 & \text{if } \text{hash}(w)_i = 1 \\ -1 & \text{if } \text{hash}(w)_i = 0 \end{cases}$$

   - Final SimHash bit $i = 1$ if $V[i] > 0$, else $0$. Two documents are near-duplicates if their **Hamming Distance $\le 3$ bits** (calculated via bitwise XOR and `popcount`), pruning $>90\%$ of duplicate web pages.
3. **Inverted Index Construction:**
   - Tokenizes text into inverted posting lists mapping terms to occurrences and token offsets:
     $$\text{"algorithm"} \rightarrow [(\text{Doc1}, [14, 88]), (\text{Doc8}, [3]), (\text{Doc104}, [201])]$$

4. **PageRank & Graph Scoring:**
   - Hyperlink structures are written to a distributed graph database. Distributed graph algorithms compute global PageRank scores, which are joined with inverted indexes during query execution.

#### API Contracts & Interface Specs
```json
// GET /v1/search?q=distributed+consensus&limit=10
// (cursor-based: use search_after for deep pagination)
// GET /v1/search?q=distributed+consensus&limit=10&search_after=d_10482
Response (200 OK):
{
  "results": [
    {"doc_id": "d_10482", "title": "Raft Consensus Explained",
     "url": "https://example.com/raft",
     "snippet": "Raft achieves consensus via leader election...",
     "pagerank_score": 0.00147}
  ],
  "total_results": 248100,
  "next_cursor": "d_10483",
  "query_latency_ms": 42
}
```

#### Database Schema & Data Model (Bigtable + Inverted Index)
```sql
-- Crawled Pages Metadata (PostgreSQL / Bigtable)
CREATE TABLE crawled_pages (
    doc_id UUID PRIMARY KEY,
    url VARCHAR(2048) UNIQUE NOT NULL,
    simhash BIGINT NOT NULL,
    pagerank_score DOUBLE PRECISION DEFAULT 0.0,
    last_crawled_at TIMESTAMPTZ,
    content_hash VARCHAR(64) NOT NULL
);
CREATE INDEX idx_pages_simhash ON crawled_pages(simhash);
-- Inverted Index stored in columnar format (Bigtable/HDFS):
-- Key: term_id -> Value: compressed PostingList[(doc_id, positions)]
```

#### Step-by-Step Execution Sequence
1. **Frontier Enqueue:** URL Frontier maintains host-based queues to enforce politeness delays ($1\text{s}$ per host) and priority rankings.
2. **Fetch & Parse:** HTML Fetcher queries local DNS Cache, downloads page, extracts hyperlinks, and runs SimHash deduplication.
3. **Inverted Index Construction:** Index Builder tokenizes text, strips stopwords, builds posting lists, and writes compressed inverted index segments to Bigtable/HDFS.
4. **PageRank Computation:** Web Graph Engine runs iterative distributed PageRank over hyperlink adjacency graph to compute authority scores for query ranking.

#### Staff-Level Interview Verbalization
> *"Our web crawler isolates domain politeness via a multi-queue URL Frontier while preventing infinite loops using SimHash document fingerprints. Inverted index posting lists are compressed using delta-encoding and combined with PageRank scores in distributed memory to serve search queries under 100ms."*


### Solution 9: Real-Time Chat — Distributed Messaging & Presence Platform (WhatsApp / Slack / Discord)

#### Problem Statement & SLAs
Design a real-time messaging and user presence platform supporting 1-on-1 and group chats.

- **Scale:** 500 million daily active users (DAU), 50 billion messages/day ($\approx 580,000 \text{ msg/sec}$ average, $\approx 1.5 \text{ million msg/sec}$ peak).
- **Latency SLA:** End-to-end message delivery $p99 < 100\text{ms}$.
- **Presence SLA:** Online/Offline state propagation $< 2\text{ seconds}$.

#### Capacity Estimation & Hardware Math
- **Message Bandwidth:** $580,000 \text{ msg/sec} \times 500 \text{ bytes} = 290 \text{ MB/sec} = 2.32 \text{ Gbps}$.
- **Storage:** $50 \times 10^9 \text{ msgs/day} \times 500 \text{ bytes} = 25 \text{ TB/day}$ in Cassandra/ScyllaDB.

#### Visual Architecture Blueprint
![Real-Time Messaging & Presence Platform Architecture](visuals/arch_chat_messaging_presence.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Stateful Connection Management:**
   - Edge **WebSocket Gateway Clusters** maintain millions of long-lived, persistent TLS connections from web and mobile clients.
2. **User Presence Engine:**
   - Uses **Redis Bitmaps** and Redis Hashes to maintain real-time online/offline/last-seen heartbeats efficiently with minimal memory overhead.
3. **Message Persistence & Channel Ordering:**
   - Ingested messages are assigned monotonically increasing sequence IDs/timestamps and written to a distributed wide-column store (**Cassandra / ScyllaDB**), partitioned by `channel_id`.
4. **Group Chat Fan-Out & Push Notifications:**
   - A Group Fan-Out Engine routes messages to active WebSocket sessions for online channel members.
   - Offline recipients are queued via Kafka/RabbitMQ to dispatch push notifications via Apple APNs and Google FCM.
   - End-to-End Encryption (E2EE) keys are verified via a separate Key Server.

#### API Contracts & Interface Specs
```json
// POST /v1/messages/send
Header: Authorization: Bearer <token>
{
  "channel_id": "ch_grp_42a1",
  "sender_id": "usr_88102",
  "encrypted_content": "<base64-encoded-E2EE-payload>",
  "client_message_id": "cm_f81d4fae"
}
// GET /v1/messages?channel_id=ch_grp_42a1&before=msg_ts_1724601600&limit=50
Response (200 OK):
{
  "messages": [
    {"message_id": "msg_7e2a", "sender_id": "usr_88102",
     "encrypted_content": "...", "sent_at": "2026-08-25T10:00:00Z"}
  ],
  "has_more": true
}
```

#### Database Schema & Data Model (Cassandra CQL)
```sql
CREATE TABLE messages (
    channel_id UUID,
    bucket_id INT, -- Partition by channel + month
    message_id TIMEUUID, -- Guarantees monotonic time ordering
    sender_id UUID,
    encrypted_content BLOB,
    PRIMARY KEY ((channel_id, bucket_id), message_id)
) WITH CLUSTERING ORDER BY (message_id ASC);
```

#### Step-by-Step Execution Sequence
1. **WebSocket Connect & Presence:** Client establishes persistent WSS connection. Presence Service sets user online bit in Redis Bitmaps and broadcasts heartbeats.
2. **Message Ingest & E2EE Key Lookup:** Client encrypts payload using Signal Double Ratchet algorithm, sends message via WebSocket to Gateway.
3. **Cassandra Commit & Fan-Out:** Message Service commits payload to Cassandra partition `(channel_id, bucket_id)` ordered by `TIMEUUID`. Group Chat Fan-Out Engine pushes payload to active WebSocket sessions of channel members.
4. **Push Fallback:** For offline members, Gateway pushes notification to APNs / FCM.

#### Staff-Level Interview Verbalization
> *"We partition chat history in ScyllaDB/Cassandra using channel IDs and TIMEUUID clustering keys to guarantee absolute message ordering without locking. User presence is tracked via Redis Bitmaps with 30-second heartbeat TTLs, and offline devices receive alerts through asynchronous push worker queues."*


### Solution 10: Task Scheduler — Distributed Workflow & Job Scheduler Engine (Temporal / Airflow)

#### Problem Statement & SLAs
Design a distributed task scheduler and workflow orchestration engine capable of executing delayed, recurring, and dependent DAG jobs.

- **Scale:** 100 million scheduled tasks/day ($\approx 10,000 \text{ executions/sec}$ peak).
- **Execution SLA:** Task execution delay $< 500\text{ms}$ from scheduled target time.
- **Reliability SLA:** At-least-once execution guarantee with automatic retry exponential backoff.

#### Capacity Estimation & Hardware Math
- **Task Payload:** 2 KB task context payload.
- **Storage:** $100 \times 10^6 \text{ tasks/day} \times 2 \text{ KB} = 200 \text{ GB/day}$ state log.

#### Visual Architecture Blueprint
![Distributed Task Scheduler & Workflow Engine Architecture](visuals/arch_task_scheduler_workflow.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Hierarchical Timing Wheel (Delayed Scheduling Engine):**
   - Implements multi-level in-memory ring buffers (millisecond, second, minute, hour, day wheels) in Go/Redis.
   - Achieves **$O(1)$ insertion and expiration** complexity for delayed tasks, avoiding the $O(\log N)$ overhead of min-heap priority queues.
2. **DAG Workflow Orchestrator:**
   - Evaluates workflow execution graphs (Directed Acyclic Graphs), managing task dependencies, preconditions, and retry policies.
   - Coordinates cluster state and leader elections via distributed lock managers (**etcd / Apache ZooKeeper**).
3. **Worker Pool & Priority Dispatch:**
   - Ready tasks enter prioritized pending queues. Distributed worker nodes pull tasks, stream heartbeats, and report execution state.
   - Unrecoverable task failures are routed to a Dead Letter Queue (DLQ) for manual inspection and replay.

#### API Contracts & Interface Specs (gRPC Protobuf)
```protobuf
syntax = "proto3";
package scheduler;

message SubmitWorkflowRequest {
    string idempotency_key = 1;
    string workflow_name = 2;
    repeated TaskDefinition tasks = 3;
    map<string, string> input_params = 4;
}
message TaskDefinition {
    string task_id = 1;
    string task_type = 2;
    int64 delay_seconds = 3;
    repeated string depends_on = 4; // Task IDs
    int32 max_retries = 5;
}
```

#### Database Schema & Data Model (PostgreSQL)
```sql
CREATE TABLE workflows (
    workflow_id UUID PRIMARY KEY,
    workflow_name VARCHAR(128) NOT NULL,
    status VARCHAR(20) CHECK (status IN
        ('PENDING','RUNNING','COMPLETED','FAILED')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);
CREATE TABLE tasks (
    task_id UUID PRIMARY KEY,
    workflow_id UUID REFERENCES workflows(workflow_id),
    task_type VARCHAR(64) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',
    scheduled_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    payload JSONB
);
CREATE INDEX idx_tasks_scheduled ON tasks(scheduled_at)
    WHERE status = 'PENDING';
```

#### Step-by-Step Execution Sequence
1. **Workflow Submission:** User submits task DAG via REST/gRPC. Workflow Orchestrator validates graph topology and writes task states to PostgreSQL/etcd.
2. **Delayed Queue Ingest:** Scheduler places delayed tasks into a Hierarchical Timing Wheel in Redis sorted by execution timestamp.
3. **Dispatcher Lock & Claim:** When timer fires, Task Dispatcher acquires an etcd distributed lock on the task ID (`SETNX task_id_lock`) and pushes work item to Worker Pool queue.
4. **Worker Execution & DLQ Retry:** Worker processes job and heartbeats status. If task fails after $N$ retries, orchestrator moves job to Dead Letter Queue (DLQ) for manual inspection.

#### Staff-Level Interview Verbalization
> *"Our task scheduler utilizes Hierarchical Timing Wheels to achieve $O(1)$ delayed job scheduling at scale. We enforce idempotency and prevent duplicate execution across distributed workers using etcd locks, routing persistently failing jobs to Dead Letter Queues."*


### Solution 11: Collaborative Editor — Real-Time CRDT & Whiteboard Engine (Figma / Google Docs / Notion)

#### Problem Statement & SLAs
Design a real-time collaborative document editor and interactive whiteboard allowing concurrent editing by hundreds of users per document.

- **Scale:** 50,000 active concurrent editing sessions.
- **Latency SLA:** Local keypress edit rendering $0\text{ms}$ (instant optimistic UI). Remote peer sync $p99 < 50\text{ms}$.
- **Consistency SLA:** Strong Eventual Consistency (SEC) — all connected clients converge to identical document states.

#### Capacity Estimation & Hardware Math
- **Real-Time Cursor Ingest:** $50,000 \text{ active sessions} \times 10 \text{ updates/sec} = 500,000 \text{ messages/sec}$.
- **Bandwidth:** $500,000 \text{ msgs/sec} \times 64 \text{ bytes} = 32 \text{ MB/sec} = 256 \text{ Mbps}$.

#### Visual Architecture Blueprint
![Real-Time Collaborative Document Editor Architecture](visuals/arch_collaborative_crdt_editor.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Conflict Resolution Strategy (CRDT vs. OT):**
   - **CRDT (Conflict-Free Replicated Data Types):** Leverages state-based (LWW-Element-Set) and operation-based (RGA / Yjs / Automerge) algorithms. Every character and canvas shape is given an immutable unique identifier (`client_id`, `lamport_clock`). Operations are commutative, associative, and idempotent, enabling peer-to-peer convergence without a single central authority.
   - **OT (Operational Transformation):** Used for centralized linear document editing where operations are transformed against concurrent edits ($op_1 \circ op_2'$).
2. **State Sync & Vector Clocks:**
   - **Vector Clock & State Sync Managers** coordinate operation streams to guarantee causal consistency across multi-client sessions.
3. **Ephemeral Awareness & Persistence:**
   - Transient cursor positions, live selections, and presence indicators are broadcast through low-latency **Redis Pub/Sub**.
   - Document operations and deltas persist to an immutable distributed log store, while periodic full snapshots are stored in S3.

#### API Contracts & Interface Specs
```json
// WebSocket: wss://collab.example.com/doc/{doc_id}
// Client -> Server (CRDT Operation Delta):
{
  "type": "INSERT",
  "op_id": {"client_id": "user_A", "lamport": 42},
  "parent_id": {"client_id": "user_A", "lamport": 41},
  "value": "X"
}
// Server -> Client (Peer Sync Broadcast):
{
  "type": "SYNC_DELTA",
  "origin_client": "user_B",
  "operations": [/* array of CRDT ops */],
  "server_seq": 10482
}
```

#### Database Schema & Data Model (PostgreSQL + S3)
```sql
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    title VARCHAR(512) NOT NULL,
    current_snapshot_url VARCHAR(512),
    op_count BIGINT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE document_operations (
    doc_id UUID NOT NULL,
    server_seq BIGINT NOT NULL,
    client_id VARCHAR(64) NOT NULL,
    lamport_clock BIGINT NOT NULL,
    op_type VARCHAR(10) CHECK (op_type IN ('INSERT','DELETE','FORMAT')),
    op_payload JSONB NOT NULL,
    PRIMARY KEY (doc_id, server_seq)
);
-- S3: s3://snapshots/{doc_id}/snapshot_{op_count}.bin.zst
CREATE INDEX idx_documents_owner ON documents(owner_id);
```

#### Step-by-Step Execution Sequence
1. **Optimistic Local Edit:** User types text or moves a shape. Client immediately updates local DOM/Canvas and appends CRDT operation `Insert(id: (user_A, 42), val: 'X', parent: (user_A, 41))`.
2. **WebSocket Sync Broadcast:** Client streams CRDT operation delta over WebSocket Gateway to document session room in Redis Pub/Sub.
3. **CRDT State Merge:** Peer clients receive delta and merge operation into local CRDT tree structure. Because operations are commutative, peer documents converge identically regardless of network latency jitter.
4. **Snapshot Storage:** Background Snapshot Worker periodically collapses CRDT operation logs into compressed document snapshots in S3 every 1,000 operations.

#### Staff-Level Interview Verbalization
> *"To achieve sub-50ms peer collaboration without server locks, we utilize Operation-based CRDTs (Conflict-Free Replicated Data Types). Each document operation is tagged with Lamport timestamps and unique client IDs, guaranteeing strong eventual consistency across all devices even during temporary offline disconnections."*


### Solution 12: Observability — Distributed Time-Series Metrics Platform (Prometheus / Datadog / Grafana)

#### Problem Statement & SLAs
Design a distributed time-series database (TSDB) and observability platform for ingesting system metrics, generating alerts, and serving dashboards.

- **Scale:** 10 million active time-series metrics ingested every 10 seconds ($\approx 1 \text{ million metric data points/sec}$).
- **Query SLA:** PromQL dashboard query execution $p99 < 200\text{ms}$.
- **Retention:** Raw metrics stored for 14 days; downsampled 5-minute rollups stored for 1 year.

#### Capacity Estimation & Hardware Math
- **Uncompressed Metric Point:** 16 bytes (8B timestamp + 8B float value).
- **Gorilla Delta-of-Delta Compression:** Compresses 16 bytes down to average **1.37 bytes** per sample ($11.6 \times$ compression ratio).
- **Daily Storage Ingress:** $1 \times 10^6 \text{ samples/sec} \times 86,400 \text{ sec/day} \times 1.37 \text{ bytes} \approx 118.3 \text{ GB/day}$. Highly compact.

#### Visual Architecture Blueprint
![Distributed Time-Series Metrics & Observability Platform Architecture](visuals/arch_metrics_timeseries_observability.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Metrics Collection (Push/Pull):**
   - Metric collectors (Prometheus pushgateway, node_exporter, OpenTelemetry agents) aggregate counters, gauges, and histograms from infrastructure and application nodes.
2. **Gorilla TSDB Compression (Facebook VLDB 2015):**
   - **Timestamps:** Compressed using **Delta-of-Delta encoding** ($D = (t_i - t_{i-1}) - (t_{i-1} - t_{i-2})$). If $D = 0$, only a single bit `0` is stored. If $-63 \le D \le 64$, store `10` + 7 bits.
   - **Floating-Point Values:** Compressed via **XOR Encoding** ($X = V_i \oplus V_{i-1}$):
     - If $X = 0$ (identical value): Store single bit `0`.
     - If $X \ne 0$: Store bit `1`. If the leading and trailing zero counts match the previous sample, store `0` + meaningful bits. Otherwise store `1` + (5 bits leading count) + (6 bits length) + meaningful bits.
     - This reduces 64-bit IEEE 754 floats to an average of **$1.37\text{ bytes/sample}$** ($11.6 \times$ memory compression).
3. **Tiered Storage & Rollup Aggregation:**
   - Recent hot data is buffered in memory ring buffers before being flushed to immutable WAL blocks.
   - Downsampling workers aggregate historical data into broader intervals (5m, 1h, 1d).
   - Inverted label indexes map metric names and label sets to chunk IDs for rapid PromQL range queries.

#### API Contracts & Interface Specs
```json
// POST /api/v1/query_range (PromQL Query)
{
  "query": "rate(http_requests_total{status='500'}[5m])",
  "start": "2026-08-25T10:00:00Z",
  "end": "2026-08-25T11:00:00Z",
  "step": "15s"
}
// Response (200 OK):
{
  "status": "success",
  "data": {
    "resultType": "matrix",
    "result": [
      {"metric": {"instance": "web-01", "status": "500"},
       "values": [[1724601600, "0.42"], [1724601615, "0.38"]]}
    ]
  }
}
```

#### Database Schema & Data Model (TSDB + Inverted Index)
```sql
-- Time-Series Metadata (PostgreSQL / embedded index)
CREATE TABLE metric_series (
    series_id BIGINT PRIMARY KEY,
    metric_name VARCHAR(128) NOT NULL,
    labels JSONB NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL,
    last_seen TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_series_labels ON metric_series
    USING GIN (labels);
-- Gorilla-compressed chunks stored as immutable 2-hour blocks
-- on local NVMe SSD, flushed to S3 after 14-day retention
```

#### Step-by-Step Execution Sequence
1. **Push/Pull Metrics Collection:** Metrics Exporter (Pushgateway / Prometheus Agent) pulls metrics from service `/metrics` endpoints every 10 seconds.
2. **In-Memory Ring Buffer Chunk Store:** Ingestion Agent appends metric sample into 2-hour in-memory ring buffer chunk, applying Gorilla compression.
3. **WAL Flush & Head Chunk Commit:** Once 2-hour chunk is full, TSDB flushes chunk to disk as an immutable block and updates label index.
4. **Alert Rules Engine & Downsampling:** Alertmanager evaluates PromQL alert rules (`CPU > 85% for 5m`) against hot ring buffers. Downsampling Aggregator collapses 14-day raw blocks into 5-minute min/max/avg rollups for long-term S3 storage.

#### Staff-Level Interview Verbalization
> *"Our time-series metrics architecture achieves an 11.6x memory reduction using Gorilla delta-of-delta timestamp and XOR float compression. We split metrics into 2-hour in-memory head chunks for sub-200ms PromQL dashboard queries while asynchronously downsampling historical data for long-term S3 retention."*


### Solution 13: Notifications — Distributed Multi-Channel Notification & Alerting Platform

#### Problem Statement & SLAs
Design a multi-channel notification platform supporting Email, SMS, Push (APNs/FCM), and In-App WebSocket delivery with deduplication and user preference management.

- **Scale:** 1 billion notifications/day ($\approx 12,000 \text{ notifications/sec}$ sustained, $50,000/\text{sec}$ peak).
- **Delivery SLA:** Push/In-App delivery $p99 < 500\text{ms}$. Email delivery $p99 < 30\text{s}$. SMS delivery $p99 < 5\text{s}$.
- **Deduplication SLA:** Zero duplicate notifications to the same user for the same event within a 24-hour window.

#### Capacity Estimation & Hardware Math
- **Notification Payload:** Average 1 KB per notification (template ID + user context + channel metadata).
- **Daily Storage:** $1 \times 10^9 \text{ notifications/day} \times 1 \text{ KB} = 1 \text{ TB/day}$ delivery log.
- **Redis Bloom Filter (Dedup Math):**
  - Optimal bit array size $m$:
    $$m = -\frac{n \ln p}{(\ln 2)^2} = -\frac{10^9 \cdot \ln(0.001)}{(0.6931)^2} \approx 14.37 \text{ billion bits} \approx 1.79 \text{ GB RAM}$$

  - Optimal number of hash functions $k$:
    $$k = \frac{m}{n} \ln 2 = \frac{14.37 \times 10^9}{10^9} \times 0.6931 \approx 10 \text{ hash functions}$$

#### Visual Architecture Blueprint
![Distributed Multi-Channel Notification & Alerting Platform](visuals/arch_notification_platform.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Ingress & Edge Deduplication:**
   - API Gateway validates incoming alert payloads and evaluates deduplication keys against a **Redis Bloom Filter** (holding 1 billion items in $\approx 1.79\text{ GB RAM}$ at $0.1\%$ false-positive rate with $k = 10$).
2. **Priority Message Routing:**
   - Directs messages into dedicated message queues (Kafka / RabbitMQ) categorized by urgency (`HIGH`, `MEDIUM`, `LOW`) and channel (`Email`, `SMS`, `Push`, `In-App`).
3. **Template Rendering & User Preferences:**
   - Ingestion consumers fetch user contact preferences and locale-specific templates from PostgreSQL.
4. **Third-Party Adapters & Delivery Tracking:**
   - Dispatches rendered payloads through downstream provider adapters (AWS SES for Email, Twilio for SMS, APNs/FCM for Mobile Push, WebSockets for In-App).
   - Delivery statuses and audit trails land in a NoSQL / Elasticsearch delivery store.

#### API Contracts & Interface Specs
```json
// POST /v1/notifications/send (Single Notification)
Header: X-Idempotency-Key: "evt_payment_confirmed_usr42"
{
  "user_id": "usr_88102",
  "template_id": "tmpl_payment_success",
  "priority": "HIGH",
  "channels": ["PUSH", "EMAIL"],
  "context": {"amount": "$129.99", "order_id": "ord_7712"}
}
// POST /v1/notifications/batch (Batch Send)
{
  "template_id": "tmpl_weekly_digest",
  "segment_query": "active_users_last_7d",
  "priority": "LOW",
  "channels": ["EMAIL"],
  "scheduled_at": "2026-08-26T09:00:00Z"
}
// Response (202 Accepted):
{
  "notification_id": "ntf_a91f2",
  "status": "QUEUED",
  "estimated_delivery_ms": 450
}
```

#### Database Schema & Data Model (PostgreSQL + Redis)
```sql
CREATE TABLE notification_templates (
    template_id VARCHAR(64) PRIMARY KEY,
    channel VARCHAR(10) NOT NULL,
    subject_template TEXT,
    body_template TEXT NOT NULL,
    version INT DEFAULT 1
);
CREATE TABLE notification_log (
    notification_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    template_id VARCHAR(64) NOT NULL,
    channel VARCHAR(10) NOT NULL,
    status VARCHAR(20) CHECK (status IN
        ('QUEUED','SENT','DELIVERED','FAILED','BOUNCED')),
    priority VARCHAR(10) DEFAULT 'NORMAL',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMPTZ
);
CREATE INDEX idx_notif_user ON notification_log(user_id, created_at DESC);
CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY,
    email_enabled BOOLEAN DEFAULT true,
    sms_enabled BOOLEAN DEFAULT true,
    push_enabled BOOLEAN DEFAULT true,
    quiet_hours_start TIME,
    quiet_hours_end TIME
);
-- Redis Bloom: BF.ADD dedup:{date} {idempotency_key}
```

#### Step-by-Step Execution Sequence
1. **Ingestion & Deduplication:** API Gateway receives notification request. Deduplication Engine checks Redis Bloom Filter (`BF.EXISTS dedup:{date} {idempotency_key}`). If duplicate, returns `200 OK` without re-sending.
2. **User Preference Lookup & Channel Routing:** Preference Service checks user's enabled channels and quiet hours. Priority Queue Router routes HIGH-priority notifications to express Kafka partitions, LOW-priority to batch partitions.
3. **Template Rendering:** Template Rendering Service hydrates the notification body using Mustache/Jinja templates with user context variables.
4. **Channel Adapter Dispatch:** Channel-specific adapters deliver the notification: Email via AWS SES, SMS via Twilio, Push via APNs/FCM, In-App via WebSocket Gateway. Each adapter reports delivery status back to the notification log.

#### Staff-Level Interview Verbalization
> *"Our notification platform achieves zero-duplicate delivery using Redis Bloom Filters with idempotency keys, routing notifications through priority-partitioned Kafka topics. We decouple channel adapters (Email/SMS/Push/In-App) behind a unified template rendering service, enabling independent scaling per channel while respecting user quiet hours and preference opt-outs."*


### Solution 14: Booking Engine — Distributed Hotel & Flight Inventory Reservation System (Airbnb / Booking.com)

#### Problem Statement & SLAs
Design a distributed inventory reservation system for hotels and flights that prevents double-booking under concurrent access from millions of users.

- **Scale:** 50,000 concurrent booking sessions, 5,000 reservations/minute peak.
- **Consistency SLA:** Strong consistency — a room or seat sold to one customer is never simultaneously sold to another.
- **Latency SLA:** Availability check $p99 < 100\text{ms}$. Reservation confirmation $p99 < 2\text{s}$ (end-to-end including payment).

#### Capacity Estimation & Hardware Math
- **Inventory Units:** 10 million hotel rooms + 500,000 flights $\times$ 365 days = $\approx 3.8 \text{ billion calendar-day slots}$.
- **Calendar Slot Size:** 64 bytes per slot (room\_id, date, status, reservation\_id, price).
- **Hot Partition Storage:** Active 90-day window: $3.8 \times 10^9 \times (90/365) \times 64 \text{ B} \approx 60 \text{ GB}$. Fits in PostgreSQL with aggressive indexing.

> **Why PostgreSQL `FOR UPDATE SKIP LOCKED` over Distributed Locks?** Distributed locks (e.g., Redis Redlock) are vulnerable to clock drift and GC pause expiry — a lock can expire while the holder is still processing, allowing a second client to acquire it. For strict inventory correctness, the database itself must be the single source of truth. PostgreSQL `FOR UPDATE SKIP LOCKED` provides ACID mutual exclusion without cross-service coordination failures. An optional Redis fast-reject layer (`SETNX lock:room:{id} EX 10`) can reduce contention by short-circuiting requests for already-locked slots, but correctness is never delegated to Redis.

#### Visual Architecture Blueprint
![Distributed Hotel & Flight Booking Inventory System](visuals/arch_booking_inventory.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Search vs. Reservation Flow Separation:**
   - Search queries are handled via cached read replicas and search services to prevent heavy search traffic from impacting inventory transaction locks.
2. **Distributed Reservation Lock & Concurrency Control:**
   - When a user initiates a booking, the **Reservation Saga Orchestrator** manages row-level locking via PostgreSQL `FOR UPDATE SKIP LOCKED` or fast-reject Redis locks (`SETNX lock:room:{id} EX 10`).
3. **Transactional Inventory Allocation:**
   - The **Inventory Availability Service** updates inventory records in **PostgreSQL** using atomic row-level locking and conditional decrement constraints (`CHECK (booked_units <= total_units)`).
4. **Saga Orchestration & Payment Settlement:**
   - The Saga Orchestrator directs the user through payment processing outside database transaction locks.
   - If payment succeeds, inventory is marked permanently booked (`CONFIRMED`), and confirmation events stream to Kafka.
   - If payment times out or fails, the Saga triggers compensating inventory restoration (Tx3 decrement `booked_units`), notifying waitlisted users via Kafka.
5. **Third-Party Partner GDS Integration:**
   - Integrates with hotel partner APIs and airline Global Distribution Systems (GDS like Amadeus/Sabre) via dedicated partner adapter gateways.

#### API Contracts & Interface Specs
```json
// GET /v1/availability?property_id=htl_42&check_in=2026-09-01&check_out=2026-09-05&guests=2
Response (200 OK):
{
  "property_id": "htl_42",
  "available_rooms": [
    {"room_type": "DELUXE_KING", "units_available": 3,
     "price_per_night_cents": 25000, "cancellation_policy": "FREE_48H"}
  ]
}
// POST /v1/reservations (Create Reservation)
Header: X-Idempotency-Key: "res_usr42_htl42_20260901"
{
  "user_id": "usr_42",
  "property_id": "htl_42",
  "room_type": "DELUXE_KING",
  "check_in": "2026-09-01",
  "check_out": "2026-09-05",
  "payment_method_id": "pm_visa_8812"
}
// Response (201 Created):
{
  "reservation_id": "rsv_f81d4",
  "status": "CONFIRMED",
  "total_cents": 100000,
  "cancellation_deadline": "2026-08-30T00:00:00Z"
}
```

#### Database Schema & Data Model (PostgreSQL)
```sql
CREATE TABLE properties (
    property_id UUID PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    location_lat DOUBLE PRECISION,
    location_lon DOUBLE PRECISION,
    total_rooms INT NOT NULL
);
CREATE TABLE inventory_calendar (
    property_id UUID NOT NULL REFERENCES properties(property_id),
    room_type VARCHAR(32) NOT NULL,
    calendar_date DATE NOT NULL,
    total_units INT NOT NULL,
    booked_units INT DEFAULT 0 CHECK (booked_units <= total_units),
    price_per_night_cents INT NOT NULL,
    PRIMARY KEY (property_id, room_type, calendar_date)
);
CREATE TABLE reservations (
    reservation_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    property_id UUID NOT NULL REFERENCES properties(property_id),
    room_type VARCHAR(32) NOT NULL,
    check_in DATE NOT NULL,
    check_out DATE NOT NULL,
    status VARCHAR(20) CHECK (status IN
        ('PENDING','CONFIRMED','CANCELLED','COMPLETED','FAILED')),
    total_cents INT NOT NULL,
    idempotency_key VARCHAR(128) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_reservations_user ON reservations(user_id, created_at DESC);
CREATE INDEX idx_reservations_property ON reservations(property_id, check_in);
```

#### Reservation Saga Orchestration
The booking process is a true distributed Saga — each step is a committed local transaction, and failures trigger explicit compensating transactions. Database row locks are **never** held across external network calls.

1. **Tx1 — Lock, Increment & Reserve:** Open a PostgreSQL transaction. Execute `SELECT 1 FROM inventory_calendar WHERE property_id = ? AND room_type = ? AND calendar_date BETWEEN ? AND ? AND booked_units < total_units FOR UPDATE SKIP LOCKED` on the inventory calendar rows for each night. If any row is already locked or fully booked (`booked_units >= total_units`), return "unavailable" immediately. Otherwise, increment `booked_units = booked_units + 1`, insert the reservation in `PENDING` status, and **COMMIT**. The inventory is now reserved and the DB connection is released.
2. **External Call — Payment:** Call the Payment Gateway to authorize and capture the charge. No database locks are held during this network call.
3. **Tx2 — Confirm (on payment success):** Update the reservation status from `PENDING` to `CONFIRMED`. Commit.
4. **Tx3 — Compensate (on payment failure):** Execute a compensating transaction: `UPDATE inventory_calendar SET booked_units = booked_units - 1` for each reserved date, and update the reservation status to `FAILED`. Commit.

> **Why not hold the DB transaction open during payment?** Holding `FOR UPDATE` row locks while waiting for an external HTTP response (which can take 2-30 seconds) blocks all concurrent bookings for those calendar slots and exhausts the database connection pool under load. A true Saga releases locks immediately after the local state change, keeping lock hold times under 10ms.

#### Step-by-Step Execution Sequence
1. **Availability Query:** Search Service queries `inventory_calendar` with date range filter and returns available room types with pricing. Read replicas serve this read-heavy path.
2. **Reservation Request:** User submits booking. Saga Orchestrator opens Tx1: acquires `FOR UPDATE SKIP LOCKED` row locks, increments `booked_units`, inserts reservation as `PENDING`, and commits — releasing the DB connection immediately.
3. **Payment Authorization:** Payment Gateway charges the card. No database resources are held during this step.
4. **Confirmation or Compensation:** On payment success, Tx2 sets reservation to `CONFIRMED`. On payment failure or timeout, Tx3 decrements `booked_units` back to restore inventory and marks the reservation `FAILED`. A background reaper job also cleans up `PENDING` reservations older than 5 minutes as a safety net.

#### Staff-Level Interview Verbalization
> *"Our booking system prevents double-booking using PostgreSQL FOR UPDATE SKIP LOCKED as the single source of truth. The Saga has three committed transactions: Tx1 increments booked_units and creates a PENDING reservation — then immediately commits and releases all row locks. The payment call happens outside any database transaction. On success, Tx2 flips the status to CONFIRMED. On failure, Tx3 runs a compensating transaction to decrement booked_units back. A background reaper catches orphaned PENDING reservations as a safety net. This design keeps lock hold times under 10ms while handling thousands of concurrent bookings."*
