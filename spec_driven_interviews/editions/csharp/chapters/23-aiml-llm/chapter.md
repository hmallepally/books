# AI/ML System Design and LLM Integration

> *"Integrating intelligence into production applications requires more than wrapping an API call. It requires designing pipelines that scale, secure prompts, and enforce data boundaries."*


## AI/ML in Technical Interviews

In modern technical interviews (especially at FAANG and high-growth startups), system design questions have evolved. In addition to traditional payment or social network systems, you are highly likely to be asked: *"Design a real-time recommendation system," "Design a semantic search pipeline,"* or *"Design a secure, rate-limited gateway for LLM agents."*

Junior candidates treat AI as magic, describing prompt calls without considering scale, caching, latency, or security. A senior system designer must articulate how to generate embeddings, perform vector similarity search, secure LLM prompts from injection, and manage data confidentiality.

In this chapter, we outline a structured approach to AI/ML system design, focusing on the ML system design framework, vector databases, RAG architecture pipelines, agentic tool-use patterns, and prompt gateway security.

![Retrieval-Augmented Generation (RAG) Architecture Pipeline](visuals/rag_architecture.png){width=90%}


## The AI/ML System Design Framework

When asked to design a machine learning system (e.g., real-time recommendation), partition your design into three distinct pipelines:

### The Offline Data Ingestion & Training Pipeline

- **Raw Data Ingestion:** Extract user activity logs, purchase history, or item metadata from primary databases.
- **Feature Store:** Store processed features (user demographics, interaction history) in a high-speed Feature Store (e.g., Feast) to ensure training and serving pipelines use consistent feature definitions.
- **Model Training:** Train recommendation models offline (e.g., Collaborative Filtering, Deep Learning models) and export model checkpoints to a Model Registry.
- **Training Infrastructure:** Use distributed training frameworks (PyTorch DDP, TensorFlow Distribution Strategy) to train across multiple GPUs. Export models in a portable format (ONNX, TorchScript, SavedModel).

### The Online Prediction Pipeline

- **Low Latency:** Online predictions must run in the sub-100ms range.
- **Feature Retrieval:** When a user requests recommendations, retrieve their online features from the Feature Store cache.
- **Candidate Retrieval (Recall):** Query a vector database to retrieve the top 100 candidate items (using fast approximate nearest neighbors).
- **Ranking:** Run a lighter model online to rank these 100 candidate items, returning the top 10 to the user.
- **Model Serving:** Deploy models behind low-latency serving infrastructure (TensorFlow Serving, Triton Inference Server, or custom gRPC endpoints).

![Model Serving Infrastructure and Real-time Inference](visuals/model_serving.jpg){width=85%}

### The Evaluation & Monitoring Pipeline
Machine learning models degrade over time as the real-world distribution shifts away from the training data:

- **Offline Metrics:** Evaluate models on held-out test data using precision, recall, F1-score, and AUC-ROC before promoting to production.
- **Online Metrics:** Track live A/B test metrics — click-through rate (CTR), conversion rate, revenue per session — to validate that the model improves business outcomes, not just accuracy scores.
- **Data Drift Detection:** Monitor input feature distributions in production. If the mean, variance, or categorical distribution of a feature shifts significantly from the training baseline, trigger a model retraining alert.
- **Shadow Mode Deployment:** Before replacing the incumbent model, deploy the new model in **shadow mode** — it receives real traffic but its predictions are logged and compared against the live model without being served to users. Only promote when shadow metrics are statistically superior.


## Model Evaluation Metrics Deep-Dive

In ML system design interviews, evaluating model performance requires choosing the right mathematical objective for the specific business domain. Stating *"we measure accuracy"* in a fraud detection or search ranking system is an instant disqualifier.

### 1. The Confusion Matrix Foundation

Every binary classification problem maps ground-truth reality against model predictions into a $2 \times 2$ **Confusion Matrix**:

| | **Predicted Positive ($\hat{y} = 1$)** | **Predicted Negative ($\hat{y} = 0$)** |
| :--- | :--- | :--- |
| **Actual Positive ($y = 1$)** | **True Positive ($\text{TP}$)**<br>*(Hit / Correct Alarm)* | **False Negative ($\text{FN}$)**<br>*(Type II Error / Missed Detection)* |
| **Actual Negative ($y = 0$)** | **False Positive ($\text{FP}$)**<br>*(Type I Error / False Alarm)* | **True Negative ($\text{TN}$)**<br>*(Correct Rejection)* |

### 2. Classification Metrics & Trade-off Formulations

| Metric | Mathematical Formula | Optimal Business Use Case | Architectural Pitfall & Hazard |
| :--- | :---: | :--- | :--- |
| **Precision**<br>*(Positive Predictive Value)* | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Spam filtering, search suggestions (cost of a false alarm is high). | Overly conservative threshold misses true positive cases. |
| **Recall**<br>*(Sensitivity / True Positive Rate)* | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Fraud detection, medical screening, cyber-attack detection. | Low threshold produces high false alarms ($\text{FP}$), overwhelming human review queues. |
| **$F_1$-Score**<br>*(Harmonic Mean)* | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ | Balanced classification where $\text{FP}$ and $\text{FN}$ have roughly equal business cost. | Treats precision and recall with equal weight; insensitive to extreme class imbalance. |
| **$F_\beta$-Score**<br>*(Weighted Harmonic Mean)* | $(1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$ | Custom cost functions ($\beta = 2$ weights Recall $2\times$ higher than Precision for fraud). | Requires empirical alignment with business dollar costs per $\text{FN}$ vs $\text{FP}$. |
| **Specificity**<br>*(True Negative Rate)* | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Clinical trials, safety-critical exclusion filters. | Can appear deceptively high when negative samples vastly outnumber positives. |
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | Balanced, symmetric classes ($50/50$ distribution). | **The Accuracy Paradox:** In 99.9% non-fraud traffic, a dummy model predicting all negative achieves $99.9\%$ accuracy while detecting $0\%$ fraud! |

### 3. Threshold Curves: ROC-AUC vs. PR-AUC

Classifiers output a continuous probability $p \in [0, 1]$. The operational decision threshold $\theta$ converts $p \ge \theta$ into $\hat{y} = 1$:

- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Plots $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ against $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$ across all thresholds $\theta \in [0, 1]$. An ideal model has $\text{AUC} = 1.0$; random guessing yields $0.5$.
  - *Hazard:* Because $\text{FPR}$ divides by large $\text{TN}$, ROC-AUC can look deceptively high ($>0.98$) on heavily imbalanced datasets even when precision is unacceptably poor.
- **PR-AUC (Precision-Recall Area Under Curve):** Plots $\text{Precision}$ against $\text{Recall}$.
  - *Golden Standard:* **Always use PR-AUC for imbalanced datasets** (e.g., fraud, ad click-through rate, rare disease detection) because it ignores $\text{TN}$ and focuses exclusively on positive class retrieval quality.

### 4. Information Retrieval & Ranking Metrics

For search engines, vector similarity retrieval, and recommendation ranking pipelines:

1. **Mean Reciprocal Rank (MRR):** Measures where the *first* relevant result appears:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
   Ideal for question answering and navigation search (where the user only cares about the top hit).

2. **Mean Average Precision (MAP@K):** Evaluates precision across the top-$K$ returned items:
   $$\text{MAP}@K = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\min(K, R_q)} \sum_{k=1}^K P(k) \cdot \text{rel}(k)$$

3. **Normalized Discounted Cumulative Gain (NDCG@K):** The gold standard for multi-level graded relevance (e.g., highly relevant $= 3$, relevant $= 1$, irrelevant $= 0$):
   $$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \qquad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
   Where $\text{IDCG}@K$ is the Ideal DCG obtained by sorting items in perfect descending relevance order. Logarithmic discounting penalizes relevant items that appear lower in the candidate ranking.

> **Interview Signal:** If asked *"How do you evaluate a fraud detection model?"*, respond: *"We optimize for recall first — missing a real fraud case is far more costly than a false alarm. We evaluate PR-AUC (Precision-Recall curve) rather than ROC-AUC or accuracy, since our dataset is heavily imbalanced ($99.9\%$ non-fraud). We tune our decision threshold $\theta$ using an $F_2$-score objective to achieve $\ge 95\%$ recall, accepting a manageable false positive rate, and route flagged transactions to an async human triage queue."*


## Vector Databases & Semantic Search

For applications utilizing natural language (such as customer support search or legal document retrieval), standard keyword-based database queries (`LIKE %query%`) are insufficient. They cannot capture semantic meaning.

### Embeddings and Vector Search

- **Embeddings:** An embedding model (e.g., OpenAI text-embedding, BERT, Sentence-BERT) transforms text into a high-dimensional vector (e.g., 1536 floating-point values) representing the semantic meaning of the words.
- **Vector Database:** Specialized databases (Pinecone, Milvus, Qdrant, Weaviate, or Postgres with pgvector extension) store these vectors.
- **Index Optimization:** To query millions of vectors under millisecond constraints, vector databases utilize approximate nearest neighbors (ANN) index algorithms:
  - **HNSW (Hierarchical Navigable Small World):** A multi-layer graph index where upper layers contain sparse long-range links (skip-list concept) and layer 0 contains dense local neighbor links.
    - **Layer Assignment Probability:** An inserted vector is assigned to maximum layer $l$ using decaying probability $l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor$, where $m_L = \frac{1}{\ln(M)}$.
    - **Hyperparameters:** $M$ (bi-directional links per node, typically $16\text{--}64$) and `efSearch` (priority queue size during greedy search, bounding query time to $\mathcal{O}(\log N)$).
  - **IVF-Flat (Inverted File Index):** Groups vectors into clusters using k-means, limiting search scope to the nearest clusters. Uses less memory than HNSW but has slightly lower search recall. Best for cost-sensitive deployments with large datasets.
  - **PQ (Product Quantization):** Compresses vectors by splitting them into sub-vectors and quantizing each independently. Dramatically reduces memory usage at the cost of some accuracy. Best for billion-scale datasets.

#### KV Cache VRAM Sizing & Inference Math

In auto-regressive LLM inference, regenerating Key and Value projection matrices on every newly generated token incurs $\mathcal{O}(L^2)$ redundant matrix multiplications. Modern inference engines (vLLM, TensorRT-LLM) cache past KV tensors in GPU High-Bandwidth Memory (HBM).

$$\text{KV Cache Size per Request} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L_{\text{seq}} \text{ bytes}$$

- Leading factors: $2$ (Key and Value matrices) $\times 2$ bytes (FP16 / BF16 precision).
- $n_{\text{layers}}$: Number of Transformer decoder layers.
- $n_{\text{kv\_heads}}$: Number of KV heads (in Grouped-Query Attention, $n_{\text{kv\_heads}} \ll n_{\text{heads}}$).
- $d_{\text{head}}$: Dimension per attention head ($\approx 128$).
- $L_{\text{seq}}$: Total context length (prompt + generated tokens).

**Concrete VRAM Example (Llama 3 70B, $L = 8,192$ tokens):**

- $n_{\text{layers}} = 80$, $n_{\text{kv\_heads}} = 8$, $d_{\text{head}} = 128$.
- $\text{KV Cache Size} = 4 \times 80 \times 8 \times 128 \times 8,192 = 2.68\text{ GB per concurrent user session}$.
- Serving 100 concurrent user streams requires $\approx 268\text{ GB VRAM}$ purely for the KV Cache (excluding model weights), explaining why **PagedAttention** (vLLM) is essential to eliminate internal memory fragmentation.

### Semantic Caching for LLM Ingress

To avoid paying expensive LLM API tokens and waiting $1\text{--}3\text{ seconds}$ for recurring user queries, an LLM Gateway implements **Semantic Caching** using vector similarity:

$$\text{Cosine Similarity } \cos(\theta) = \frac{\mathbf{q}_{\text{new}} \cdot \mathbf{q}_{\text{cached}}}{\|\mathbf{q}_{\text{new}}\|_2 \|\mathbf{q}_{\text{cached}}\|_2}$$

- If $\cos(\theta) \ge 0.95$ (e.g., "How do I reset my password?" vs "Steps to change password"), the gateway returns the cached completion instantly in $<10\text{ms}$ with zero LLM API cost.

### Chunking Strategies for RAG
The quality of vector search results depends heavily on how source documents are split into chunks before embedding:

- **Fixed-Size Chunking:** Split documents into chunks of 512 or 1024 tokens. Simple but can break sentences mid-thought.
- **Semantic Chunking:** Split on paragraph or section boundaries, preserving logical coherence. Higher retrieval quality but variable chunk sizes.
- **Overlapping Windows:** Use a sliding window with 20% overlap between chunks. Ensures that concepts spanning chunk boundaries are captured by at least one chunk.
- **Metadata Enrichment:** Attach document title, section heading, page number, and source URL to each chunk as metadata. This enables filtered searches (e.g., "search only in the compliance policy documents").


## LLM Integration Patterns

When incorporating Large Language Models (LLMs) into production-grade systems, architects must resolve latency bottlenecks, costs, and security risks.

### Retrieval-Augmented Generation (RAG)
RAG addresses LLM knowledge limits and hallucinations by injecting relevant business data into the model prompt:

1. The user submits a query.
2. The query is converted to a vector embedding.
3. The vector database performs a similarity search, returning matching business documents.
4. The document text is inserted into the LLM system prompt as context.
5. The LLM processes the context to generate an accurate, grounded response.

### RAG Quality Optimization

- **Re-Ranking:** After retrieving the top-K documents from the vector database, pass them through a **cross-encoder re-ranker** (e.g., Cohere Rerank, a fine-tuned BERT cross-encoder) that scores each document against the original query. This dramatically improves context relevance over raw vector similarity alone.
- **Hybrid Search:** Combine vector similarity search with traditional keyword search (BM25). This catches exact-match terms that semantic search may miss (e.g., product codes, invoice numbers, legal clause identifiers).
- **Context Window Management:** LLMs have finite context windows (4K–128K tokens). If your retrieved documents exceed the window, implement a context budget — rank documents by relevance score and truncate at the token limit rather than naively concatenating all results.

### Semantic Caching
LLM API calls are slow and expensive. To optimize latency:

- Implement a **Semantic Cache** (e.g., GPTCache using Redis). *(Rate limiting for prompt gateways utilizes the distributed Redis Sliding Window pattern detailed in **Chapter 16**).*
- Instead of exact match string caching, convert incoming prompts to vectors and check similarity against cached prompts.
- If a query has a 95%+ vector similarity match to a cached entry, return the cached LLM response directly, avoiding downstream API latency.
- **Cache Invalidation:** Set TTLs on cached entries aligned with the freshness requirements of the underlying data. For static knowledge bases, TTLs of 24–72 hours are appropriate. For real-time data, bypass the cache entirely.

### Multi-Layer Prompt Security & Guardrails

When exposing LLM endpoints to untrusted user input, applications face severe security vulnerabilities including **Prompt Injection** (tricking the model into ignoring system instructions) and **Data Leakage** (extracting confidential system prompts or training data).

In production enterprise gateways, security requires a multi-layer defense strategy:

1. **Layer 1: Deterministic Input Sanitization (Regex & Pattern Filters):** Rapidly reject known injection patterns (`IGNORE PREVIOUS INSTRUCTION`, SQL injection attempts, system prompt extraction keywords) at zero API latency cost.
2. **Layer 2: Guardrail Classifiers (LLM-Based Intent Inspection):** Route incoming prompts through lightweight guardrail classification models (e.g., Llama Guard, NeMo Guardrails, or fine-tuned classifiers) to detect toxic, unsafe, or out-of-scope prompts before invoking the primary LLM.
3. **Layer 3: Structured Schema Output Enforcement:** Enforce strict JSON schema validation (via function calling or JSON mode) on all LLM responses, rejecting unstructured or unexpected model outputs.

### Fine-Tuning vs. Prompt Engineering
When adapting LLMs to domain-specific tasks, choose the right approach:

| Approach | When to Use | Cost | Latency Impact |
|---|---|---|---|
| **Prompt Engineering** | General tasks, rapid iteration, small domain context | Low (no training cost) | Adds tokens to every request |
| **Few-Shot Prompting** | Tasks with clear input/output patterns | Low | Moderate token overhead |
| **RAG** | Large knowledge bases, frequently updated data | Medium (embedding + vector DB) | Adds retrieval latency (~50ms) |
| **Fine-Tuning** | Consistent style/format, specialized domain language | High (GPU training cost) | Reduces prompt size, faster inference |
| **Full Pretraining** | Entirely new domains with no base model coverage | Very High | N/A (creates new model) |

> **Rule of Thumb:** Start with prompt engineering. Move to RAG if the model needs access to private or frequently updated data. Fine-tune only when prompt engineering consistently fails to produce the required output format or domain accuracy.

### Multimodal AI: Beyond Text

Modern AI systems increasingly process multiple modalities — text, images, audio, and video — within unified architectures. Interview questions are beginning to reflect this shift.

**Architectural Patterns for Multimodal Systems:**

**1. Vision-Language Models (VLMs):** Systems like GPT-4o and Gemini accept both images and text as input. The architectural pattern involves a visual encoder (often a Vision Transformer) that produces embedding tokens, which are concatenated with text tokens before being processed by the language model. For AuraPay, this enables check deposit processing: the VLM reads the check image, extracts the amount and payee, and populates the transaction record — replacing a fragile OCR pipeline.

**2. Audio Processing Pipelines:** Real-time transcription (Whisper, Deepgram) feeds into LLM reasoning. The key architectural decision is streaming vs. batch: streaming transcription adds 200-500ms latency but enables real-time agent responses, while batch processing is simpler and more accurate. ZenithTrade uses streaming transcription for compliance monitoring of trader phone calls.

**3. Multimodal RAG:** Instead of retrieving only text chunks, multimodal RAG indexes images, diagrams, and tables alongside text. Document understanding models (like LayoutLM) preserve spatial relationships in scanned documents. This is critical for ChiramTrust's regulatory document processing, where table structures contain compliance data that pure text extraction would lose.

**Interview Tip:** When asked about an AI/ML system, always clarify which modalities the system needs to handle. A document processing pipeline that handles scanned PDFs requires fundamentally different architecture than one processing structured text.

### Agentic Tool-Use Patterns
LLMs can be orchestrated as **agents** that decide which tools to call based on user intent:

- **Function Calling:** The LLM receives a list of available tool definitions (name, description, parameters). Based on the user query, it selects the appropriate tool and generates structured JSON arguments.
- **Multi-Step Orchestration:** Complex tasks require chaining multiple tool calls. An agent might: (1) query a database for account details, (2) call a fraud scoring API, (3) generate a human-readable summary. Frameworks like LangChain, LlamaIndex, or custom orchestrators manage this loop.
- **Guardrails:** Constrain the agent's tool access. A customer-facing agent should never have access to database deletion tools. Implement a **tool whitelist** per agent role, and validate all generated tool arguments against input schemas before execution.


## Prompt Gateway Security

LLMs are vulnerable to **Prompt Injection Attacks** (where an attacker crafts inputs to bypass system rules or extract private instruction prompts).
To defend your platform, you must place a **Security Filter** in front of your LLM call:

### Defense Layers

1. **Input Sanitization:** Parse incoming prompts to detect known injection patterns — phrases like "ignore previous instructions," "system prompt:", or attempts to encode instructions in base64 or unicode.
2. **PII Scrubbing:** Before sending user data to an external LLM API, scrub all Personally Identifiable Information — names, credit card numbers, Social Security numbers, phone numbers — using regex patterns and Named Entity Recognition (NER) models.
3. **Output Validation:** After receiving the LLM response, validate it against expected output schemas. If the model returns data that violates format constraints (e.g., includes SQL queries, URLs to external sites, or content that bypasses content policies), block the response.
4. **Rate Limiting:** Apply per-user and per-session rate limits on LLM API calls to prevent abuse and cost runaway. Use the Redis sliding window pattern (discussed in the Enterprise Integration and Resiliency chapter).

The following code illustrates a prompt verification filter:

```csharp
using System;
using System.Text.RegularExpressions;

public class LlmGatewaySecurityFilter 
{
    private static readonly Regex InjectionPattern = new Regex(
        "(ignore all previous instructions|system prompt|bypass validation|reveal key)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled
    );

    public bool ValidatePrompt(string userPrompt) 
    {
        if (string.IsNullOrWhiteSpace(userPrompt)) 
        {
            return false;
        }
        // Fail-fast if malicious injection signature detected
        if (InjectionPattern.IsMatch(userPrompt)) 
        {
            throw new UnauthorizedAccessException("Potential prompt injection attack blocked");
        }
        return true;
    }
}
```


Any incoming prompt containing injection signatures is blocked immediately before execution, protecting the LLM boundary from security drift.


## Case Study Integration: ML in Practice

**AuraPay: Real-Time Fraud Detection Pipeline**
AuraPay processes 50,000 transactions per second. Its fraud detection pipeline combines rule-based filters (velocity checks, geo-anomaly flags) with a gradient-boosted ensemble model trained on 18 months of labeled transaction data. Feature engineering extracts 47 signals per transaction: merchant category deviation, time-of-day risk scores, device fingerprint similarity, and spending velocity z-scores. The model runs inference in < 5ms per transaction via ONNX Runtime, with a fallback to rule-only evaluation if the ML service is unavailable (graceful degradation, per Chapter 18's resiliency patterns).

**ZenithTrade: LLM-Powered Compliance Checker**
ZenithTrade's regulatory compliance team reviews 200+ SEC filings weekly. Their LLM pipeline uses Retrieval-Augmented Generation (RAG) to cross-reference new filings against the firm's internal compliance rulebook (12,000 rules). The system generates structured compliance reports highlighting potential violations, with confidence scores and source citations. Human compliance officers review flagged items — the LLM augments but never replaces human judgment on regulatory decisions.

### Mock Interview Transcript: RAG Pipeline Design

> **Interviewer:** Design a RAG pipeline for a customer support chatbot that handles 10,000 queries/hour. Walk me through the architecture.
> **Candidate:** First, we need to vectorize our support documentation. We'd use semantic chunking to keep logical sections together, pass them through an embedding model like text-embedding-3-small, and store the vectors in a specialized vector database like Pinecone or Weaviate. When a user queries, we embed the query, perform an approximate nearest neighbor search to retrieve the top 5 chunks, and inject them into the LLM prompt.
> **Interviewer:** What's your latency budget for this, and how do you meet it at 10,000 queries/hour?
> **Candidate:** 10,000 an hour is roughly 3 queries a second. Our biggest bottleneck is the LLM inference time. To reduce latency and API costs, I'd implement semantic caching using Redis. We convert the incoming query to a vector and check if we have a 95%+ similarity match with a previous query. If so, we return the cached response immediately.
> **Interviewer:** How do you handle hallucinations? If the bot gives wrong refund instructions, it's a huge liability.
> **Candidate:** Actually, let me reconsider the prompt structure... We must constrain the model. In the system prompt, we explicitly instruct it to answer *only* using the provided context. If the answer isn't in the chunks, it must reply "I don't know" and escalate to a human. We'd also run a cross-encoder re-ranker after retrieval to ensure only highly relevant context is passed to the LLM.
> **Interviewer:** How do you prevent users from jailbreaking the bot to ignore those instructions?
> **Candidate:** We'd place a security filter gateway in front of the LLM to scan for prompt injection signatures, and use input sanitization to strip out command-like phrasing before embedding.

**Technical Summary:** The candidate successfully designed a robust RAG pipeline, incorporating semantic chunking and vector search. They addressed scale and latency via semantic caching, mitigated hallucinations through strict prompt constraints and re-ranking, and prioritized security with a prompt injection gateway.

## Cost Optimization for LLM-Powered Systems

LLM inference costs scale directly with token volume. At enterprise scale, unoptimized architectures can generate six-figure monthly bills:

1. **Model Tiering:** Route simple queries (FAQ lookups, classification) to smaller, cheaper models (GPT-4o-mini, Claude Haiku). Reserve expensive frontier models (GPT-4o, Claude Opus) for complex reasoning tasks. Implement a **router model** that classifies query complexity before dispatching.
2. **Prompt Compression:** Use techniques like LLMLingua to compress long context windows by removing redundant tokens while preserving semantic meaning. Can reduce token counts by 50–70%.
3. **Batch Processing:** For non-real-time workloads (document summarization, report generation), batch requests and use discounted batch API pricing.
4. **Self-Hosted Models:** For high-volume, latency-tolerant workloads, deploy open-source models (Llama, Mistral) on owned GPU infrastructure. Higher upfront cost but dramatically lower per-token cost at scale.


> ⭐ **STAR Moment: The Full ML System Design**
> 
> In a system design interview, demonstrate the complete picture: *"For the recommendation engine, we separate our architecture into three pipelines. The offline pipeline trains our ranking model using user interaction features stored in Feast, with weekly retraining triggered by data drift detection. The online pipeline retrieves candidate items via HNSW vector search, then re-ranks with a lightweight cross-encoder model, targeting sub-100ms p99 latency. We deploy new models in shadow mode first, comparing CTR and conversion rates against the incumbent via A/B testing before promotion. For cost control, we route simple classification queries to GPT-4o-mini and reserve frontier models for complex reasoning."* This shows end-to-end ML engineering maturity.


## Enterprise Real-Time ML Decisioning Engine & Feature Store

In mission-critical AI applications (such as automated credit underwriting or real-time fraud scoring), ML architectures must deliver sub-200ms $p99$ latency SLAs while satisfying strict regulatory compliance requirements (e.g., Federal Reserve SR 11-7 model risk governance and ECOA adverse action explainability).

### Dual-Tier Feature Store Architecture

To guarantee consistency between offline model training and real-time online inference, enterprise platforms deploy a **Dual-Tier Feature Store** (e.g., Feast, Databricks Feature Store):

- **Offline Feature Store (Delta Lake / Parquet):** Stores historical, point-in-time correct feature values for model training, backtesting, and validation without data leakage.
- **Online Feature Store (Redis / DynamoDB):** Provides low-latency ($<10\text{ms}$) key-value lookups for live inference, caching real-time applicant features (e.g., 30-day cash flow, recent velocity flags).

### Model Explainability & Regulatory Compliance (TreeSHAP & Adverse Action Codes)

Under financial regulations (Equal Credit Opportunity Act - ECOA and Fair Credit Reporting Act - FCRA), automated AI decision engines cannot operate as unexplainable black boxes. If an applicant is denied or receives a higher rate, the platform must output up to **4 specific Adverse Action Reasons**:

1. **TreeSHAP (SHapley Additive exPlanations):** Computes exact local feature attribution weights for every individual applicant feature vector against non-linear GBDT (XGBoost/LightGBM) models.
2. **Automated Adverse Action Code Generation:** Sorts feature vectors by their negative SHAP contribution scores and maps the top 4 negative features directly to legally compliant ECOA denial reason codes.
3. **Disparate Impact Auditability:** Computes real-time Adverse Impact Ratios (AIR) across demographic groups to ensure models remain free of proxy bias.
