# API Literacy for Product Specialists

> *"APIs are not just technical integrations; they are the digital products that define your company's boundaries."*

## The Communication Chasm

Sarah, a talented Product Owner at FinLend, was leading the integration of a new third-party identity verification service. She gathered the business requirements meticulously: the system needed to verify a user's driver's license, run a background check, and update the loan application status to 'Verified'. 

She wrote the user stories and handed them to the development team. A week later, during sprint review, the lead engineer demonstrated the feature. 

"It works," the engineer said, "but we had to build a polling mechanism that queries their server every 5 seconds because they don't support webhooks. It's going to spike our infrastructure costs, and if they rate-limit us, the whole application pipeline will block. Also, they return a 200 OK even when the background check fails, they just put 'status: failed' in the payload, so I had to write custom error handling. Oh, and their authentication token expires every ten minutes without a refresh endpoint, meaning we have to store user credentials in a highly secure vault and re-authenticate constantly, which our InfoSec team is going to hate."

Sarah was stunned. She hadn't asked the vendor about webhooks, rate limits, pagination, or HTTP status standards. She had treated the integration as a black box---a business capability rather than a technical contract. Because she lacked API literacy, she couldn't assess the technical viability or the architectural impact of the vendor she had chosen. The team had to spend two more sprints refactoring the system to handle the vendor's poorly designed API, delaying the product launch by a month and costing the company hundreds of thousands of dollars in lost revenue and engineering time.

In the modern enterprise, APIs (Application Programming Interfaces) are the nervous system of the business. A Product Specialist cannot afford to treat them as black boxes. You must be able to read an API contract, test it, and specify its behavior with the same fluency that you read a business process diagram. The days of 'just let the devs figure out the API' are over. If you own the product, you own the API integration.

## Case Studies: Real-World Contexts

To ground our understanding of APIs, we will rely on three primary case studies throughout this chapter:

1. **MedClaim Pro (Healthcare):** A complex B2B platform that handles medical claims processing. MedClaim Pro interacts with hospital EHR (Electronic Health Record) systems, insurance providers, and government compliance databases. Their APIs must be highly secure (HIPAA compliant), capable of handling massive XML/JSON payloads, and exceptionally reliable.
2. **FinLend (FinTech):** A consumer and SMB lending platform. FinLend aggregates data from credit bureaus, bank accounts (via Open Banking APIs), and identity verification services to make real-time underwriting decisions. Their APIs must be lightning-fast, idempotent, and capable of gracefully handling third-party outages.
3. **ShipStream (E-Commerce):** A global logistics and fulfillment engine. ShipStream connects storefronts (like Shopify) with warehouses, shipping carriers (FedEx, UPS), and last-mile delivery contractors. Their APIs heavily rely on asynchronous event-driven architectures and webhooks to track physical goods moving around the world.

## REST API Fundamentals

Representational State Transfer (REST) is the architectural style that governs most web APIs today. It relies on standard web protocols (HTTP) to manage data. Understanding REST is the absolute baseline for any Product Specialist.

### Resources and URIs

In REST, everything is a **Resource** (e.g., a Loan, a Claim, an Order). Resources are identified by URIs (Uniform Resource Identifiers). The structure of these URIs should be intuitive, hierarchical, and noun-based.

- Good: `/api/v1/loans/` (Represents the collection of loans)
- Good: `/api/v1/loans/98765` (Represents a specific loan)
- Good: `/api/v1/loans/98765/documents` (Represents the documents attached to a specific loan)
- Bad: `/api/v1/getLoanById?id=98765` (REST is noun-based, not verb-based. Do not use verbs in URIs.)
- Bad: `/api/v1/createDocumentForLoan` (Use standard HTTP methods instead of verbs in the path)

### HTTP Methods

The action you want to perform on a resource is defined by the HTTP Method (also called the verb).

- **GET:** Retrieve a resource. (Read). A GET request should never modify data. It is strictly for reading.
- **POST:** Create a new resource. (Create). It can also be used for actions that don't neatly fit into CRUD, though this is less RESTful.
- **PUT:** Replace a resource entirely. (Update). If you send a PUT request, you are sending the *entire* state of the object to replace what is currently there.
- **PATCH:** Partially update a resource. (Modify). If you only want to change the 'status' of an order, you send a PATCH request with just the status field.
- **DELETE:** Remove a resource. (Delete).

### The Concept of Idempotency

An API endpoint is **idempotent** if making multiple identical requests has the same effect as making a single request. This is critical for network reliability. If a client sends a request, and the network drops before the response is received, the client doesn't know if the request succeeded. If the endpoint is idempotent, the client can safely just send it again.

- **GET, PUT, PATCH, DELETE** must be idempotent. If you DELETE a claim, doing it again shouldn't crash the system; it should just confirm it's gone (or return a 404). If you PUT the exact same data twice, the end state is the same as if you did it once.
- **POST** is generally *not* idempotent. If you POST a payment twice, the customer gets charged twice. 

To solve the POST idempotency problem, robust APIs implement Idempotency Keys. The client generates a unique ID (the key) and includes it in the header of the POST request. The server remembers this key. If the client retries the exact same request with the exact same key, the server says, "Ah, I already processed this," and returns the cached successful response instead of charging the credit card a second time.

> ### For the Candidate
> In an interview, if you are designing a financial integration (like FinLend's loan funding), proactively mention idempotency keys. "To prevent double-funding if a network timeout occurs, I would specify that the POST `/disbursements` endpoint must require a unique `Idempotency-Key` in the header." This shows senior-level architectural thinking and demonstrates you understand failure modes in distributed systems.

## HTTP Status Codes: A Comprehensive Reference

As discussed in Chapter 04, status codes are how the API communicates the result of a request. Product Specialists must specify these in their acceptance criteria. A 200 OK for everything is a massive anti-pattern.

![HTTP Status Codes](visuals/http_status_codes.png){width=85%}

### 2xx (Success)

- **200 OK:** The request succeeded. Used for successful GET, PUT, or PATCH requests.
- **201 Created:** A POST request successfully created a new resource. The response should ideally include a `Location` header pointing to the new resource.
- **204 No Content:** The request succeeded, but there is no payload to return. This is very common for successful DELETE requests.

### 4xx (Client Errors - The requester messed up)

- **400 Bad Request:** The server cannot process the request due to client error (e.g., malformed request syntax, invalid JSON formatting).
- **401 Unauthorized:** Missing or invalid authentication token. "Who are you?" The client must authenticate itself to get the requested response.
- **403 Forbidden:** The client is authenticated, but does not have access rights to the content. "I know who you are, but you can't do that." For example, a standard user trying to access admin endpoints.
- **404 Not Found:** The server cannot find the requested resource. The URI doesn't exist.
- **409 Conflict:** The request conflicts with the current state of the server. Example: Trying to delete a user who still has active loans, or trying to update a record that has been modified by someone else since you last fetched it.
- **422 Unprocessable Entity:** The request was well-formed (valid JSON), but was unable to be followed due to semantic errors. Example: `age` must be an integer > 18, but the user sent `12`. This is the most common status code for business logic validation failures.
- **429 Too Many Requests:** The user has sent too many requests in a given amount of time (Rate limiting). The response should include a `Retry-After` header.

### 5xx (Server Errors - The API provider messed up)

- **500 Internal Server Error:** A generic error message, given when an unexpected condition was encountered and no more specific message is suitable. It means the backend code crashed.
- **502 Bad Gateway:** The server, while acting as a gateway or proxy, received an invalid response from the upstream server.
- **503 Service Unavailable:** The server is not ready to handle the request. Common causes are a server that is down for maintenance or that is overloaded.

### Status Code When-to-Use Table

| Scenario | HTTP Method | Expected Status Code | Notes |
| :--- | :--- | :--- | :--- |
| Successfully fetching a list of active MedClaim claims. | GET | 200 OK | Payload contains JSON array of claims. |
| Successfully creating a new ShipStream shipment. | POST | 201 Created | Response includes the new tracking number. |
| Deleting a canceled loan application in FinLend. | DELETE | 204 No Content | No body needed in the response. |
| Sending XML instead of JSON to an API expecting JSON. | POST | 400 Bad Request | The server couldn't parse the body. |
| Attempting to view MedClaim records without logging in. | GET | 401 Unauthorized | Token missing or expired. |
| A nurse trying to delete a hospital from MedClaim. | DELETE | 403 Forbidden | Nurse is authenticated, but lacks admin permissions. |
| Searching for a ShipStream tracking number that does not exist. | GET | 404 Not Found | Resource missing. |
| Trying to mark a ShipStream package as 'In Transit' when it is already 'Delivered'. | PATCH | 409 Conflict | State transition is invalid. |
| FinLend applicant submits income of -$5000. | POST | 422 Unprocessable Entity | Valid JSON, but fails business rules. |
| Hitting the FinLend API 10,000 times a second. | GET | 429 Too Many Requests | Rate limit enforced. |
| The FinLend database server catches on fire. | GET | 500 Internal Server Error | Unhandled server crash. |

## Reading and Writing OpenAPI / Swagger

OpenAPI (formerly Swagger) is the industry standard for defining REST APIs. It is a machine-readable document (usually written in YAML or JSON) that describes the entire API contract: endpoints, request formats, response formats, authentication methods, and validation rules.

As a Product Specialist, you should advocate for **Contract-First Development**. This means you (often collaborating with a Tech Lead) write the OpenAPI specification *before* any code is written. This spec becomes the single source of truth. The frontend team can build UI components against mock servers generated from the spec, while the backend team implements the actual logic.

### Annotated Example: FinLend Loan Application API

Let us look at an extensive snippet of an OpenAPI specification for creating a loan application in FinLend. 

```yaml
openapi: 3.0.3
info:
  title: FinLend Origination API
  description: Core API for submitting and managing loan applications.
  version: 2.1.0
servers:
  - url: https://api.finlend.com/v2
    description: Production Server
  - url: https://sandbox.finlend.com/v2
    description: Sandbox Environment
paths:
  /applications:
    post:
      summary: Create a new loan application
      description: Submits a new applicant payload for immediate decisioning.
      operationId: createApplication
      security:
        - bearerAuth: []
      parameters:
        - in: header
          name: Idempotency-Key
          schema:
            type: string
            format: uuid
          required: true
          description: Unique key to prevent duplicate applications.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ApplicationRequest'
      responses:
        '201':
          description: Application successfully created and decisioned.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ApplicationResponse'
        '400':
          description: Malformed JSON syntax.
        '401':
          description: Invalid API key.
        '409':
          description: Duplicate Idempotency-Key detected.
        '422':
          description: Validation error (e.g., amount outside limits, invalid SSN).
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  schemas:
    ApplicationRequest:
      type: object
      required:
        - applicantId
        - requestedAmount
        - termMonths
      properties:
        applicantId:
          type: string
          format: uuid
          description: The UUID of the pre-registered user.
        requestedAmount:
          type: number
          minimum: 500
          maximum: 50000
          description: Loan amount in USD.
        termMonths:
          type: integer
          enum: [12, 24, 36, 48, 60]
          description: Repayment term in months.
        purpose:
          type: string
          maxLength: 255
          description: User-provided reason for the loan.
    ApplicationResponse:
      type: object
      properties:
        applicationId:
          type: string
          format: uuid
        status:
          type: string
          enum: [APPROVED, DECLINED, MANUAL_REVIEW]
        approvedAmount:
          type: number
        apr:
          type: number
          format: float
    ErrorResponse:
      type: object
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: array
          items:
            type: string
```

By reading this specification, a Product Specialist instantly comprehends the boundaries of the system:
1. The endpoint expects a POST request to `/applications`.
2. It requires `applicantId`, `requestedAmount`, and `termMonths`.
3. The amount is strictly constrained between $500 and $50,000.
4. The term must be exactly 12, 24, 36, 48, or 60 months.
5. It enforces an `Idempotency-Key` header (solving the duplicate submission problem).
6. It uses Bearer Token (JWT) security.
7. It defines specific responses for 201, 400, 401, 409, and 422.

This YAML file IS the product requirements document for the API. There is no ambiguity.

## Postman Deep Dive for Product Specialists

Postman is an essential tool for API exploration and validation. You do not need to be an automation engineer to use it, but you must know how to manually interact with APIs. Product Specialists use Postman to verify vendor APIs during discovery, and to validate their own team's APIs during sprint reviews.

### Core Features You Must Know

1. **Collections:** Group related API requests together. You might have a "FinLend Underwriting" collection that contains folders for "Applicant Creation", "Credit Pull", and "Decisioning". Collections can be shared with the team.
2. **Environments and Variables:** Never hardcode URLs or tokens. Use variables like `{{baseUrl}}` so you can easily switch between Development, Staging, and Production environments. If a vendor gives you a sandbox API key, store it in the Postman Environment as `{{apiKey}}`.
3. **Pre-request Scripts:** Scripts that run before a request is sent. For example, if an API requires a timestamp signature for security, you can write a short Javascript snippet in the pre-request script to calculate the signature and inject it into the header automatically.
4. **Mock Servers:** Postman allows you to generate a mock server from an OpenAPI spec. Before developers write backend code, they can hit the Postman Mock Server to get realistic responses based on the spec.
5. **Test Assertions:** Postman allows you to write simple JavaScript to validate responses. As a Product Specialist, you can write assertions to verify your acceptance criteria:

```javascript
// Check that the status code is what we expect
pm.test("Status code is 201 Created", function () {
    pm.response.to.have.status(201);
});

// Check that the response contains an application ID
pm.test("Response has applicationId", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('applicationId');
});

// Validate business logic constraints
pm.test("Credit limit is within bounds", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.approvedAmount).to.be.below(50001);
});

// Validate response time
pm.test("Response time is less than 500ms", function () {
    pm.expect(pm.response.responseTime).to.be.below(500);
});
```

These assertions turn manual API poking into automated validation. When you run a collection via **Newman** (Postman's command-line companion), it executes all these tests in seconds.

## Integration Patterns: Webhooks, Polling, and Event-Driven Architectures

When systems talk to each other, they need a way to communicate updates asynchronously. If MedClaim Pro submits a 500-page medical chart to a machine-learning service for ICD-10 code extraction, it might take 10 minutes to process. The API cannot simply keep the HTTP connection open for 10 minutes (it will timeout). It needs an asynchronous integration pattern.

### Polling

System A repeatedly asks System B, "Are you done yet? Are you done yet?"

- *Example:* FinLend asks the ID verification service every 10 seconds if the background check is complete.
- *Mechanism:* System A makes a POST request to start the job. System B returns a `202 Accepted` with a `jobId`. System A then makes a GET request to `/jobs/{jobId}` every 10 seconds.
- *Pros:* Easy to implement. Works behind firewalls (System A is initiating all outbound traffic).
- *Cons:* Extremely inefficient. Wastes bandwidth and compute resources. You will hit rate limits quickly. If thousands of clients are polling, the server will collapse under the load.

### Webhooks (Event-Driven)

System A tells System B, "Here is a URL. POST a message to this URL when you are done."

- *Example:* ShipStream receives a massive order file. Two hours later, when the warehouse finishes packing, ShipStream POSTs a status update to `shopify.com/api/webhooks/fulfillment`.
- *Mechanism:* System A registers a callback URL with System B. When the event occurs, System B initiates an HTTP POST to System A's URL.
- *Pros:* Highly efficient. Real-time. No wasted polling traffic.
- *Cons:* Requires your system (System A) to expose a public endpoint to receive the webhook. Requires complex error handling---what if System A's server is down when System B sends the webhook? System B must implement a retry strategy (e.g., exponential backoff) to ensure the message is eventually delivered.

### Event Streaming (Pub/Sub)

For high-throughput internal microservices, systems use message brokers like Apache Kafka or AWS EventBridge.

- *Example:* When FinLend approves a loan, the `DecisionService` publishes an `ApplicationApproved` event to a Kafka topic. The `NotificationService` (to send emails), the `LedgerService` (to prep funds), and the `AnalyticsService` (to update dashboards) are all 'subscribed' to this topic. They receive the event simultaneously and process it independently.
- *Pros:* Decoupled architecture. Highly scalable.
- *Cons:* Complex infrastructure. Harder to trace a single transaction end-to-end.

> ### For the Interviewer
> Ask candidates: "We are integrating with a third-party shipping provider to get tracking updates. How would you design the data flow?" Look for candidates who contrast polling vs. webhooks, discussing the trade-offs in server load, real-time necessity, and the implications of exponential backoff retry strategies.

## FinLend Worked Examples: The Anatomy of an API Interaction

Let us trace a comprehensive API interaction within the FinLend ecosystem to cement these concepts. The goal is to fund an approved loan. 

The client application (the FinLend mobile app) needs to instruct the backend to disburse $10,000. 

**Step 1: The Request (Client to Server)**

The mobile app sends an HTTP POST request. Notice the headers and the body payload.

```http
POST /api/v2/disbursements HTTP/1.1
Host: api.finlend.com
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6Ik...
Content-Type: application/json
Idempotency-Key: 8f7e6d5c-4b3a-2190-1234-56789abcdef0
Accept: application/json

{
  "applicationId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "amount": 10000.00,
  "destinationBank": {
    "routingNumber": "122000661",
    "accountNumberMasked": "******7890"
  }
}
```

**Step 2: The Business Logic Evaluation**

The FinLend server receives the request. It performs a sequence of checks:

1. **Authentication:** Is the JWT token valid? Yes.
2. **Authorization:** Does this user own `applicationId` a1b2c3d4? Yes.
3. **Idempotency Check:** Have we seen `8f7e6d5c...` before? No. Proceed.
4. **Validation:** Is the amount within limits? Yes.
5. **State Machine:** Is the application in a state that allows disbursement? (e.g., it must be 'APPROVED' and not already 'FUNDED'). Let's assume yes.
6. **Execution:** The server initiates the ACH transfer via a third-party payment rail API.
7. **Database Update:** The server updates the loan status to 'FUNDED'.

**Step 3: The Response (Server to Client)**

The server sends back the result.

```http
HTTP/1.1 201 Created
Date: Wed, 21 Oct 2026 07:28:00 GMT
Content-Type: application/json
Location: /api/v2/disbursements/d9e8f7g6-h5i4-j3k2-l1m0-n9o8p7q6r5s4

{
  "disbursementId": "d9e8f7g6-h5i4-j3k2-l1m0-n9o8p7q6r5s4",
  "status": "PROCESSING_ACH",
  "estimatedArrival": "2026-10-23T00:00:00Z",
  "amountDisbursed": 10000.00
}
```

If the client's internet connection drops *during* Step 2, the mobile app won't receive the 201 Created response. The app will show an error and the user might click 'Fund Loan' again. The app sends the exact same request with the exact same `Idempotency-Key`. In step 3, the server sees the key, says "I already processed this!", ignores the execution phase, and simply re-sends the exact same 201 Created response, preventing the user from receiving $20,000 by accident.

## Defining API Acceptance Criteria

When writing user stories or specifications for APIs, be explicit. Do not leave the contract up to the developer's imagination. You must specify the endpoint, the payload schema, the expected success response, the expected error responses, and the performance requirements.

**Poor AC (What traditional BAs write):**

- Ensure the API saves the claim.
- Return an error if it fails.

**Spec-Driven AC (SDSD-POD standard):**

- **Endpoint:** `POST /api/v2/claims`
- **Request Payload:** Must validate against `ClaimSchema_v2.json`.
- **Success:** Return `201 Created` with the generated `claimId`.
- **Failure (Syntax):** If required fields are missing, return `400 Bad Request` with an array of validation errors.
- **Failure (Business Logic):** If the Provider ID is inactive, return `422 Unprocessable Entity` with error code `ERR_PROVIDER_INACTIVE`.
- **Idempotency:** Must implement `Idempotency-Key` header logic.
- **Latency SLA:** 95th percentile response time must be < 400ms.

## Reading Karate Tests: API Specifications as Living Documentation

Karate is an API testing framework that uses human-readable Gherkin syntax. As a Product Specialist, you may not write Karate tests yourself, but understanding them is a superpower --- they ARE your living API specification.

### Why Product Specialists Should Care About Karate

- Karate tests are readable by non-developers (Given/When/Then).
- They serve as executable API documentation. They never go out of date, unlike a wiki page.
- They validate YOUR specifications automatically in the CI/CD pipeline.
- In the SDSD-POD, reviewing Karate tests is how you validate the Development Expert's implementation matches your spec.

### Reading a Karate Test (Annotated)

Here is a simple annotated example of a Karate test validating the FinLend disbursement API we designed above:

```gherkin
Feature: Loan Disbursement API
  # This feature validates the FinLend funding endpoints

  Background:
    * url 'https://api.finlend.example.com'
    * def authHeader = call read('classpath:helpers/get-auth-token.feature')
    * header Authorization = 'Bearer ' + authHeader.token

  Scenario: Successfully disburse a valid, approved loan
    Given path '/api/v2/disbursements'
    And header Idempotency-Key = java.util.UUID.randomUUID()
    And request
    """
    {
      "applicationId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "amount": 10000.00,
      "destinationBank": {
        "routingNumber": "122000661",
        "accountNumberMasked": "******7890"
      }
    }
    """
    When method post
    Then status 201                                 # ← Your spec says: return 201 Created
    And match response.disbursementId == '#uuid'    # ← Auto-generated ID exists
    And match response.status == 'PROCESSING_ACH'   # ← Initial state per your state machine

  Scenario: Reject disbursement if loan is not in APPROVED state
    Given path '/api/v2/disbursements'
    And header Idempotency-Key = java.util.UUID.randomUUID()
    And request { "applicationId": "rejected-loan-id", "amount": 10000.00 }
    When method post
    Then status 409                                 # ← State conflict!
    And match response.code == 'ERR_INVALID_STATE'
    And match response.message == 'Cannot disburse a loan that is not APPROVED'
```

When reviewing such tests, the Product Specialist should verify:

- Does the URL match the API contract you specified?
- Does the request body match the schema you defined?
- Does the status code match your specification?
- Does the error message match the business rule you wrote?

### The Specification-to-Test Mapping

Here is how your business specifications map to Karate validations:

| Your Specification Says | Karate Test Validates |
| :--- | :--- |
| "Return 201 Created on success" | `Then status 201` |
| "ApplicationId is auto-generated UUID" | `And match response.applicationId == '#uuid'` |
| "Initial status is PENDING_REVIEW" | `And match response.status == 'PENDING_REVIEW'` |
| "Reject if credit score < 620" | `Then status 422` + error message match |
| "Required fields: name, amount, term" | Scenario Outline testing each missing field |

### What to Look for in a Karate Test Review

As a Product Specialist reviewing your Development Expert's Karate tests:

1. **Coverage**: Is every API endpoint tested? Are there tests for GET, POST, PUT, DELETE?
2. **Happy path**: Does the success scenario match your spec exactly?
3. **Edge cases**: Are boundary conditions tested (min/max values, empty strings, nulls, exceptionally long strings)?
4. **Error responses**: Do error messages match your specification language? Are they user-friendly?
5. **State transitions**: Are invalid state transitions tested (e.g., approving an already-rejected application)?
6. **Security**: Are there tests ensuring that an unauthenticated user gets a 401, and a user trying to access someone else's data gets a 403?

> ⭐ **STAR Moment --- Specification Validation**
> When you can read a Karate test and say 'this doesn't match my spec --- line 42 should return 409 Conflict, not 400 Bad Request, because it is a state violation, not a syntax error,' you've crossed from BSA to Product Specialist.

## Mock Interview Dialogue: Evaluating API Competence

Let us observe how these concepts play out in a rigorous interview scenario for a Senior Product Manager role at ShipStream.

**Interviewer (Director of Product, ShipStream):** "We are building a new integration with a regional last-mile delivery courier. They need to receive our shipping manifests, and we need to know when a package is delivered. How would you approach defining this API integration?"

**Candidate:** "First, I'd want to determine the system boundaries and the data flow. Are we pushing the manifest to them, or are they pulling it from us? Given it's a manifest for fulfillment, we should probably push it to them via a `POST /manifests` endpoint on their system when our warehouse finishes packing."

**Interviewer:** "Good. And what about the delivery updates? We need those as close to real-time as possible so we can text the customer."

**Candidate:** "For real-time updates, polling is usually the wrong answer. If we have to hit their `GET /packages/{id}/status` endpoint every five minutes for a hundred thousand packages, we'll hammer both our infrastructure and theirs, and most of those calls will just return 'still in transit'. I would require them to support Webhooks. We would expose an endpoint on our side, maybe `POST /api/webhooks/courier-updates`, and they would push a JSON payload to us the second the delivery driver scans the package at the doorstep."

**Interviewer:** "Excellent. Now, let's say the driver scans the package, their system sends the webhook to our server, but our server happens to be down for a 30-second maintenance window. We return a 503 Service Unavailable. What happens to that delivery update?"

**Candidate:** "If they just drop the webhook, we lose the data, the customer never gets the text, and our database says the package is still in transit forever. This is why the webhook integration contract *must* specify a retry policy. I would explicitly require in the acceptance criteria that the vendor must use exponential backoff---retrying after 5 seconds, then 10, then 20, up to a certain threshold---if they receive anything other than a 2xx success code from our webhook endpoint."

**Interviewer:** "Perfect. Let's switch gears. In our core REST API, a user is trying to update their billing address using a `PUT` request, but they leave the 'Zip Code' field completely blank. What HTTP status code should our API return?"

**Candidate:** "The JSON syntax is valid, so it's not a 400 Bad Request. But semantically, a blank Zip Code violates our business rules for a valid address. So, we should return a 422 Unprocessable Entity, along with a payload specifying exactly which field failed validation so the frontend can highlight the Zip Code box in red."

**Interviewer:** "What if they tried to update an order that has already shipped?"

**Candidate:** "That's a state violation. The data is valid, but the current state of the resource prohibits the action. I would return a 409 Conflict."

> ### For the Candidate
> Notice how the candidate doesn't just name-drop 'Webhooks' or 'REST'. They explain *why* (efficiency, avoiding polling), anticipate failure modes (retry policies for 503s), and map specific business scenarios to exact HTTP status codes (422 vs 409). This demonstrates deep API literacy and a product ownership mindset.

## Conclusion

API literacy bridges the gap between business intent and technical execution. By understanding HTTP methods, status codes, integration patterns, and OpenAPI specifications, you transition from a scribe who passes messages to a Product Specialist who architects solutions. 

You no longer have to blindly trust that an integration 'works' just because a developer says it does. You can read the Swagger spec. You can fire up Postman and test the endpoints yourself. You can review the Karate tests to ensure every edge case is covered.

In the SDSD-POD model, your Development Expert relies on you to define the API contract precisely. When you provide a flawless OpenAPI spec, complete with rigid schemas and exact status codes, they can use AI to generate the boilerplate routing and validation code in seconds. This allows them to focus their human intelligence on the complex domain logic inside the endpoints, dramatically accelerating the delivery of robust, enterprise-grade software.
