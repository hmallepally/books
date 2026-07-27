# Visual Prompts Registry — Spec-Driven Quality Engineering

This file contains all image generation prompts used for visuals in this book.
Each entry includes the target chapter, filename, and the exact prompt used.

---

## Book Cover

**File:** `cover.png`
**Prompt:** Professional book cover for "Spec-Driven Quality Engineering" by Harinath Mallepally. Subtitle: "From Test Executor to Quality Partner". Clean, modern corporate design with deep emerald green and silver accent colors. Abstract geometric pattern suggesting quality gates, testing pipelines, and verification layers. Professional typography. No photos of people. Book dimensions 6x9 inches portrait orientation. Premium business book aesthetic. Matching style with a navy blue companion book.

---

## Chapter 01: The Evolution of Quality Engineering

### QE Role Evolution
**File:** `chapters/01-evolution-quality-engineering/visuals/qe_evolution.png`
**Prompt:** Professional infographic showing the evolution of quality engineering roles. Horizontal timeline/arrow from left to right with 4 stages: 1) "QA Tester" (gray) - icon of checklist, 2) "Quality Engineer" (light green) - icon of automation gear, 3) "Quality Advocate" (medium green) - icon of shield with checkmark, 4) "Quality Partner" (emerald/gold) - icon of handshake with specification document. Each stage shows key skills below. Arrow flows left to right showing progression. Clean white background, corporate style. Title: "The Quality Engineering Evolution".

### The Sidekick vs Partner Model
**File:** `chapters/01-evolution-quality-engineering/visuals/sidekick_vs_partner.png`
**Prompt:** Professional side-by-side comparison infographic. Left side "The Sidekick Model" (red/gray, negative): QE sits outside the team, receives tickets after development, only executes test cases, no domain knowledge, reports bugs reactively. Right side "The Partner Model" (green/gold, positive): QE embedded in POD 1:1 with developer, reviews specifications, has deep domain expertise, validates against spec proactively. Large X on left, large checkmark on right. Title: "Sidekick vs Partner". White background, clean corporate style.

### SDSD-POD Model
**File:** `chapters/01-evolution-quality-engineering/visuals/sdsd_pod_model.png`
**Prompt:** Clean professional infographic diagram showing the SDSD-POD organizational model. Show a 6-person pod structure: 2 "Product Specialist" roles (navy blue), 2 "Development Expert" roles (teal), and 2 "Junior" roles (light gray). Draw clear 1:1 pairing lines between Product Specialist and Development Expert. Highlight that the Quality Partner IS the Product Specialist who validates. Center label: "SDSD-POD". White background, corporate clean style, minimal geometric design. No photos of people.

---

## Chapter 03: The Quality Partner Maturity Model

### Maturity Pyramid
**File:** `chapters/03-quality-partner-maturity/visuals/maturity_pyramid.png`
**Prompt:** Clean professional pyramid infographic showing 5 maturity levels for Quality Engineering. Bottom to top: Level 1 "Test Executor" (gray), Level 2 "Test Designer" (light green), Level 3 "Quality Advocate" (medium green), Level 4 "Quality Partner" (emerald, highlighted with gold border), Level 5 "Quality Architect" (dark green/gold). Each level shows 2-3 key competencies. Arrow on the right side pointing up labeled "Growth Path". Title: "Quality Partner Maturity Model". White background, modern corporate style.

---

## Chapter 04: Quality Concepts & Test Strategy

### Test Pyramid
**File:** `chapters/04-quality-concepts-strategy/visuals/test_pyramid.png`
**Prompt:** Clean professional diagram showing the testing pyramid. Triangle divided into 3 horizontal sections. Bottom (largest, green): "Unit Tests" - Fast, Cheap, Many. Middle (medium, blue): "Integration Tests" - API, Contract, Database. Top (smallest, orange): "E2E Tests" - Slow, Expensive, Few. Anti-pattern shown to the right: inverted "Ice Cream Cone" with too many E2E tests (red X). Title: "The Test Pyramid". White background, clean corporate style.

### Risk-Based Testing Matrix
**File:** `chapters/04-quality-concepts-strategy/visuals/risk_matrix.png`
**Prompt:** Clean professional 2x2 matrix for risk-based testing prioritization. X-axis: "Probability of Failure" (Low to High). Y-axis: "Business Impact" (Low to High). Quadrants: Top-Right (red): "Critical - Test Extensively", Top-Left (orange): "Important - Test Thoroughly", Bottom-Right (yellow): "Moderate - Test Adequately", Bottom-Left (green): "Low - Smoke Test Only". Title: "Risk-Based Testing Priority Matrix". White background, corporate style.

---

## Chapter 05: Advanced Manual Testing Techniques

### Boundary Value Analysis
**File:** `chapters/05-manual-testing-techniques/visuals/boundary_analysis.png`
**Prompt:** Clean professional diagram showing boundary value analysis. Number line showing a valid range [18, 65] for age field. Test points marked: 17 (just below, red X), 18 (boundary, green check), 19 (just above, green check), 64 (just below upper, green check), 65 (upper boundary, green check), 66 (just above upper, red X). Labels explain: "Below minimum", "At minimum boundary", "Just inside", "Just inside upper", "At maximum boundary", "Above maximum". Title: "Boundary Value Analysis". White background, clean technical style.

### State Transition Diagram
**File:** `chapters/05-manual-testing-techniques/visuals/state_transition.png`
**Prompt:** Clean professional state transition diagram for a loan application. States shown as rounded rectangles: "Draft" → "Submitted" → "Under Review" → (decision diamond) → "Approved" → "Funded" OR → "Rejected" → "Appeal". Transitions labeled with actions: "Submit", "Assign Reviewer", "Approve/Reject", "Fund", "Appeal". Invalid transitions shown with red dashed lines (e.g., Draft directly to Funded). Title: "Loan Application State Transitions". White background, professional diagram style with emerald green and gray colors.

---

## Chapter 07: API Testing & Contract Validation

### API Testing Layers
**File:** `chapters/07-api-testing-contracts/visuals/api_testing_layers.png`
**Prompt:** Clean professional layered diagram showing API testing layers. Concentric rectangles from outer to inner: 1) "Contract Testing" (outermost, blue) - validates API schema/contract, 2) "Functional Testing" (green) - validates business logic, 3) "Security Testing" (red) - validates auth, injection, OWASP, 4) "Performance Testing" (orange) - validates latency, throughput. Center: "API Endpoint". Arrows pointing inward. Title: "API Testing Layers". White background, corporate style.

### Contract Testing Flow
**File:** `chapters/07-api-testing-contracts/visuals/contract_testing_flow.png`
**Prompt:** Clean professional flowchart showing consumer-driven contract testing with Pact. Left side: "Consumer" (frontend/mobile app) generates "Pact Contract" (JSON). Center: contract file is published to "Pact Broker". Right side: "Provider" (API service) verifies against the contract. Arrows show flow: Consumer → Pact Broker → Provider. Green checkmark if contract matches, red X if mismatch. Title: "Consumer-Driven Contract Testing". White background, clean technical style.

---

## Chapter 08: Web Test Automation Frameworks

### Framework Comparison
**File:** `chapters/08-web-automation-frameworks/visuals/framework_comparison.png`
**Prompt:** Clean professional comparison infographic for 3 web testing frameworks in card format. Card 1: "Selenium" (orange) - Architecture: WebDriver protocol, Languages: Java/Python/C#/JS, Speed: Medium, Reliability: Requires explicit waits. Card 2: "Cypress" (green) - Architecture: In-browser, Languages: JS only, Speed: Fast, Reliability: Auto-waiting. Card 3: "Playwright" (purple) - Architecture: CDP/WebSocket, Languages: JS/Python/Java/C#, Speed: Fast, Reliability: Auto-waiting + codegen. Title: "Web Automation Framework Comparison". White background, modern card design.

### Page Object Model
**File:** `chapters/08-web-automation-frameworks/visuals/page_object_model.png`
**Prompt:** Clean professional architectural diagram showing the Page Object Model design pattern. Three layers: Top: "Test Classes" (green) - contains test logic, assertions. Middle: "Page Objects" (blue) - LoginPage, DashboardPage, CheckoutPage - contains locators and actions. Bottom: "WebDriver / Browser" (gray) - actual browser interaction. Arrows showing: Tests call Page Objects, Page Objects interact with Browser. Title: "Page Object Model Architecture". White background, clean layered diagram style.

---

## Chapter 09: Performance & Load Testing

### Load Test Stages
**File:** `chapters/09-performance-load-testing/visuals/load_test_stages.png`
**Prompt:** Clean professional line chart showing a load test execution profile. X-axis: Time (minutes). Y-axis: Concurrent Users. Line shows: Ramp-up phase (0 to 100 users over 2 min), Steady state (hold at 100 for 5 min), Spike (jump to 500 for 2 min), Cool-down (ramp down to 0). Overlay metrics: Response time (p95) shown as a second line, Error rate as a third line. Key threshold line at 250ms. Title: "Load Test Execution Profile". White background, professional chart style with green/orange/red colors.

### Performance Metrics
**File:** `chapters/09-performance-load-testing/visuals/percentile_distribution.png`
**Prompt:** Clean professional histogram/distribution chart showing response time distribution. X-axis: Response Time (ms). Y-axis: Frequency. Bell curve with tail. Vertical lines marking: p50 (median, green, 45ms), p95 (orange, 180ms), p99 (red, 450ms), mean (dashed blue, 85ms). Shaded area beyond p99 labeled "Tail Latency". Title: "Response Time Distribution — Why Percentiles Matter More Than Averages". White background, clean data visualization style.

---

## Chapter 11: CI/CD Integration & Shift-Left

### Shift-Left Diagram
**File:** `chapters/11-cicd-shift-left/visuals/shift_left.png`
**Prompt:** Clean professional diagram showing the Shift-Left testing concept. Timeline from left (Design) to right (Production). Traditional approach (top, red): testing concentrated at the end, defects expensive. Shift-Left approach (bottom, green): testing distributed throughout — requirements review, unit tests, integration tests, contract tests during development. Cost of defect curve showing exponential increase from left to right. Title: "Shift-Left: Test Earlier, Fail Cheaper". White background, corporate style.

### CI/CD Pipeline
**File:** `chapters/11-cicd-shift-left/visuals/cicd_pipeline.png`
**Prompt:** Clean professional horizontal pipeline diagram showing CI/CD stages with quality gates. Stages left to right: 1) "Commit" (gray) → 2) "Build" (blue) → 3) "Unit Tests" (green, gate: >90% coverage) → 4) "Integration Tests" (teal, gate: all pass) → 5) "Contract Tests" (purple) → 6) "Performance Tests" (orange, gate: p95 < 250ms) → 7) "Deploy to Staging" (blue) → 8) "E2E Tests" → 9) "Deploy to Production" (gold). Red stop signs at each quality gate. Title: "CI/CD Pipeline with Quality Gates". White background, modern pipeline design.

---

## Chapter 13: AI-Augmented Quality Engineering

### AI-QE Workflow
**File:** `chapters/13-ai-augmented-quality/visuals/ai_qe_workflow.png`
**Prompt:** Clean professional flowchart showing the AI-augmented quality engineering workflow. Steps: 1) "Quality Partner writes specification" → 2) "AI generates comprehensive test suite" → 3) "QP reviews and refines tests" → 4) "AI identifies coverage gaps" → 5) "Automated execution in CI/CD" → 6) "AI analyzes failures, suggests root cause" → 7) "QP validates and reports". Circular feedback loop from step 6 back to step 3. Emerald green and gold color scheme. White background, modern corporate style. Title: "AI-Augmented Quality Engineering Workflow".
