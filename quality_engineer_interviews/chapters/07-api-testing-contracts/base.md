# API Testing & Contract Validation

> *"UIs change daily. APIs are promises. Break a UI, and a user is annoyed. Break an API contract, and a business stops functioning."*

## The API Testing Imperative

In modern distributed architectures---like the MedPortal microservices or the CartFlow orchestration engine---the graphical user interface is merely a thin facade over a complex web of APIs. Testing exclusively through the UI is slow, brittle, and provides feedback too late in the cycle. The Quality Partner understands that true quality assurance begins at the API layer.

The API (Application Programming Interface) represents the central nervous system of any modern platform. When MedPortal processes a patient's protected health information (PHI), or when TradeForge executes a sub-millisecond stock transaction, the core logic does not reside in the browser or the mobile app. It resides in the backend services communicating via HTTP and WebSocket protocols.

This chapter transitions you from executing manual Postman requests to engineering robust, automated API validation suites and embracing the future of consumer-driven contract testing. 

### Dual Intent: Today and Tomorrow

**TODAY:** You need to master the mechanics of REST and GraphQL APIs, demonstrate proficiency in tools like Postman and REST-assured, and understand how to integrate these tests into CI/CD pipelines to pass technical interviews. 

**TOMORROW:** In the SDSD-POD model, you won't just be testing APIs after they are built. You will engage in contract-first validation, defining the API specifications alongside the Development Expert and ensuring the AI-generated implementation perfectly adheres to those predefined contracts. You will be thinking in terms of API invariants, latency budgets, and security boundaries.

## REST API Testing Fundamentals

Before diving into tools and frameworks, you must speak the language of HTTP fluently. An API tester who does not understand the nuances of the HTTP protocol will inevitably write superficial tests.

### The Anatomy of a Request

Every HTTP request consists of several critical components that a Quality Partner must systematically validate:

*   **HTTP Methods (Verbs):** The action being requested.
    *   `GET`: Retrieve data. Must be idempotent (calling it multiple times produces the same result) and safe (does not modify state). Example: Fetching a patient's allergy list in MedPortal.
    *   `POST`: Create new resources. Not idempotent. Example: Submitting a new checkout cart in CartFlow.
    *   `PUT`: Replace an entire resource. Must be idempotent. If a resource exists, it is overwritten. If it does not, it is created. Example: Updating a complete user profile in MedPortal.
    *   `PATCH`: Partially update a resource. Example: Changing only the status of an order from PENDING to SHIPPED in CartFlow.
    *   `DELETE`: Remove a resource. Example: Canceling an open order in TradeForge.

*   **Headers:** Metadata accompanying the request or response.
    *   `Content-Type`: Tells the server what format the body is in (e.g., `application/json`, `application/xml`).
    *   `Accept`: Tells the server what format the client expects in return.
    *   `Authorization`: Contains credentials to authenticate the client.

*   **Query Parameters vs. Path Variables:**
    *   *Path Variable:* Identifies a specific resource (e.g., `/users/123`).
    *   *Query Parameter:* Sorts, filters, or modifies the request (e.g., `/users?role=admin&status=active`).

### Authentication & Authorization Mechanisms

Security is paramount, especially in domains like healthcare and finance. Testing an API without validating its security perimeter is malpractice.

*   **API Keys:** Simple strings passed in headers or query parameters. Vulnerable to interception if not used over HTTPS.
*   **Bearer Tokens (JWT - JSON Web Tokens):** Tokens that encode user identity and claims. Testing must verify:
    *   Token expiration (Does the API reject an expired token?)
    *   Signature validation (If you tamper with the payload, does the server reject it?)
    *   Claim validation (Does the token actually belong to a user with the right permissions?)

*   **OAuth 2.0 Flows:** The industry standard for delegated access. Testing involves simulating the full handshake (authorization code, client credentials) to obtain access tokens.

### HTTP Status Codes: The Full Reference

You must know these without hesitation during an interview. An answer like "200 means good, 400 means bad" is an immediate red flag.

![HTTP Status Codes](visuals/http_status_codes.png){width=85%}

*   **2xx (Success):** 
    *   `200 OK`: Standard success.
    *   `201 Created`: Crucial for POST requests. Indicates a resource was successfully created.
    *   `204 No Content`: Often used for DELETE requests where no body needs to be returned.

*   **3xx (Redirection):** 
    *   `301 Moved Permanently`: The resource has a new URI.
    *   `302 Found`: Temporary redirection.

*   **4xx (Client Error):** The client messed up.
    *   `400 Bad Request`: Validation failure. The payload was malformed.
    *   `401 Unauthorized`: Missing or invalid authentication token.
    *   `403 Forbidden`: Valid token, but insufficient permissions to perform the action.
    *   `404 Not Found`: The resource does not exist.
    *   `409 Conflict`: Business logic conflict, such as trying to register an email that already exists.
    *   `429 Too Many Requests`: Rate limiting has been triggered (critical for TradeForge testing).

*   **5xx (Server Error):** The server messed up.
    *   `500 Internal Server Error`: An unhandled exception occurred on the server.
    *   `502 Bad Gateway`: The API gateway received an invalid response from the upstream service.
    *   `503 Service Unavailable`: The server is down for maintenance or overloaded.
    *   `504 Gateway Timeout`: The upstream service took too long to respond.

> **For the Interviewer:** Ask candidates to explain the difference between a 401 and a 403 status code. A strong Quality Partner will immediately explain that 401 means "I don't know who you are," while 403 means "I know exactly who you are, but you aren't allowed to do this."

> **For the Candidate:** When asked how you test an endpoint, do not just list the happy path. Start by explaining how you validate the HTTP methods (e.g., "I verify that sending a POST to a read-only GET endpoint returns a 405 Method Not Allowed"). This demonstrates a deep understanding of REST principles.

## Postman Mastery: The CartFlow Walkthrough

Postman is ubiquitous in the industry. However, interviewers are looking for *advanced* Postman usage, not just the ability to click the "Send" button on a pre-configured request.

To demonstrate this, we will walk through a complete test collection setup for the **CartFlow E-commerce platform**.

### Collections and Environments

A Quality Partner organizes tests logically. In Postman, this means creating a Collection for "CartFlow Checkout Flow" and structuring it into folders: `Authentication`, `Inventory`, `Cart Management`, and `Payment`.

Instead of hardcoding URLs, you use **Environments**. You create a `QA Environment` and a `Staging Environment`, defining a variable like `{{baseUrl}}`. 
A request URL looks like: `{{baseUrl}}/api/v1/carts`. This allows you to run the exact same tests across different infrastructure tiers seamlessly.

### Variables and Scope

Understanding scope is critical for robust test execution:

*   **Global:** Available across all workspaces (use sparingly).
*   **Collection:** Available to all requests within the specific collection.
*   **Environment:** Bound to the currently selected environment (e.g., database credentials, base URLs).
*   **Data:** Values injected from an external CSV or JSON file during a collection run.
*   **Local:** Temporary variables created during a script execution that disappear after the request finishes.

### Pre-request Scripts

Pre-request scripts execute JavaScript *before* the request is sent. This is essential for dynamic data generation.

In CartFlow, when adding an item to the cart, we need a unique request ID to ensure idempotency.

```javascript
// Pre-request script to generate a dynamic idempotency key
const uuid = require('uuid');
pm.variables.set("idempotencyKey", uuid.v4());

// Generate dynamic test data
const randomSku = "SKU-" + Math.floor(Math.random() * 10000);
pm.environment.set("testSku", randomSku);
```

### Test Assertions

Postman uses the `pm.test` and `pm.expect` syntax (based on the Chai assertion library) to validate responses. You must go beyond simple status code checks.

```javascript
// Validating the CartFlow Add to Cart response
pm.test("Status code is 201 Created", function () {
    pm.response.to.have.status(201);
});

pm.test("Response time is acceptable (under 200ms)", function () {
    pm.expect(pm.response.responseTime).to.be.below(200);
});

pm.test("Cart schema is valid", function () {
    const jsonData = pm.response.json();
    
    // Assert presence and type
    pm.expect(jsonData.cartId).to.be.a('string');
    pm.expect(jsonData.totalItems).to.be.a('number');
    pm.expect(jsonData.items).to.be.an('array');
    
    // Assert business logic
    pm.expect(jsonData.totalItems).to.be.above(0);
    pm.expect(jsonData.status).to.eql("ACTIVE");
});

// Chaining requests by saving data
pm.test("Save cartId for next request", function () {
    const jsonData = pm.response.json();
    pm.environment.set("currentCartId", jsonData.cartId);
});
```

### Newman CLI

The Quality Partner doesn't run tests manually; they use **Newman**, the command-line runner for Postman. Newman allows you to execute collections within a CI/CD pipeline (e.g., GitHub Actions, Jenkins).

```bash
# Running a collection in Newman with a specific environment and generating an HTML report
newman run CartFlow.postman_collection.json \
  -e QA.postman_environment.json \
  -r cli,htmlextra \
  --reporter-htmlextra-export ./results/report.html
```

## Programmatic API Testing: REST-assured

While Postman is excellent for exploration and rapid test creation, many engineering teams prefer having API tests sit directly alongside the application code in the same repository. For Java-based teams, **REST-assured** is the undisputed industry standard.

REST-assured utilizes a fluid, BDD-style `Given/When/Then` syntax that makes tests highly readable.

### Worked Example: TradeForge Order Placement

In the high-stakes environment of the TradeForge trading engine, we need rigorous validation of order placement.

```java
import io.restassured.RestAssured;
import io.restassured.builder.RequestSpecBuilder;
import io.restassured.builder.ResponseSpecBuilder;
import io.restassured.specification.RequestSpecification;
import io.restassured.specification.ResponseSpecification;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static io.restassured.RestAssured.*;
import static org.hamcrest.Matchers.*;

public class TradeForgeOrderApiTest {

    private RequestSpecification requestSpec;
    private ResponseSpecification responseSpec;

    @BeforeClass
    public void setupSpecifications() {
        // Build reusable request specifications
        requestSpec = new RequestSpecBuilder()
            .setBaseUri("https://api.tradeforge.qa")
            .addHeader("Authorization", "Bearer " + getValidToken())
            .addHeader("Content-Type", "application/json")
            .build();

        // Build reusable response specifications
        responseSpec = new ResponseSpecBuilder()
            .expectResponseTime(lessThan(100L)) // TradeForge requires sub-100ms latency
            .expectHeader("Content-Type", containsString("application/json"))
            .build();
    }

    @Test
    public void testPlaceMarketOrder() {
        String requestBody = "{\n" +
            "  \"symbol\": \"BTC/USD\",\n" +
            "  \"side\": \"BUY\",\n" +
            "  \"type\": \"MARKET\",\n" +
            "  \"quantity\": 1.5\n" +
            "}";

        given()
            .spec(requestSpec)
            .body(requestBody)
        .when()
            .post("/v1/orders")
        .then()
            .spec(responseSpec)
            .statusCode(201)
            .body("orderId", notNullValue())
            .body("status", equalTo("PENDING_MATCH"))
            .body("executionFee", greaterThanOrEqualTo(0.0f));
    }
}
```

### Data-Driven Testing with REST-assured

REST-assured integrates perfectly with testing frameworks like TestNG to enable data-driven testing using DataProviders. This allows you to test multiple edge cases without duplicating code.

```java
@DataProvider(name = "invalidOrderData")
public Object[][] createInvalidOrderData() {
    return new Object[][] {
        { "INVALID_SYMBOL", "BUY", "MARKET", 1.5, 400, "Symbol not recognized" },
        { "BTC/USD", "INVALID_SIDE", "MARKET", 1.5, 400, "Invalid order side" },
        { "BTC/USD", "BUY", "MARKET", -1.0, 422, "Quantity must be greater than zero" }
    };
}

@Test(dataProvider = "invalidOrderData")
public void testInvalidOrderValidation(String symbol, String side, String type, 
                                     double quantity, int expectedStatusCode, 
                                     String expectedError) {
    // Construct payload...
    // Execute REST-assured call...
    // Assert status code and error message...
}
```

## Karate: The Unified API Testing Framework

Karate (created by Peter Thomas) is a revolutionary open-source framework that combines API test automation, mocking, performance testing, and UI automation into a single tool. 

It uses a Gherkin-like syntax (`Given/When/Then`) but crucially **does NOT require step definitions**. The Gherkin *is* the test. This makes it dramatically faster to write and maintain than traditional BDD frameworks like Cucumber coupled with REST-assured.

### Why Karate Stands Out

*   **No step definitions needed:** Eliminates the boilerplate code that plagues traditional BDD.
*   **Built-in JSON/XML assertion engine:** Incredibly powerful fuzzy matching and schema validation.
*   **Built-in parallel execution:** Runs tests concurrently without complex configuration.
*   **Built-in mock server:** Service virtualization out of the box.
*   **Performance testing:** Native integration with Gatling.
*   **Data-driven testing:** Embedded expressions and direct JSON data loading.
*   **Native support:** Full support for GraphQL, WebSocket, gRPC, and SOAP.

### Karate Syntax Deep Dive: MedPortal Example

Let's look at how Karate handles a complex workflow in the MedPortal application.

```gherkin
Feature: MedPortal Patient Records API Workflow

  Background:
    * url 'https://api.medportal.qa.internal'
    # Call another feature file to authenticate and extract the token
    * def auth = call read('classpath:auth/login.feature')
    * header Authorization = 'Bearer ' + auth.token
    * configure connectTimeout = 5000
    * configure readTimeout = 10000

  Scenario: Create a new patient record and verify data integrity
    Given path '/api/v1/patients'
    # Karate allows direct inline JSON without escaping quotes
    And request
    """
    {
      "firstName": "Jane",
      "lastName": "Doe",
      "dateOfBirth": "1990-05-15",
      "insuranceId": "INS-2024-78901",
      "email": "jane.doe@example.com"
    }
    """
    When method post
    Then status 201
    # Fuzzy matching assertions
    And match response.id == '#notnull'
    And match response.firstName == 'Jane'
    And match response.createdAt == '#regex \\d{4}-\\d{2}-\\d{2}T.*'
    
    # Save the ID for the next step in the same scenario
    * def patientId = response.id

    # Retrieve the patient we just created
    Given path '/api/v1/patients', patientId
    When method get
    Then status 200
    
    # Complex schema validation using fuzzy matchers
    And match response ==
    """
    {
      id: '#(patientId)',
      firstName: '#string',
      lastName: '#string',
      dateOfBirth: '#regex \\d{4}-\\d{2}-\\d{2}',
      insuranceId: '#string',
      email: '#string',
      active: '#boolean',
      appointments: '#[] #object',
      _links: '#object'
    }
    """
```

### Karate's Killer Features for Quality Partners

#### 1. Fuzzy Matching (Schema Validation)

Karate's assertion engine handles deeply nested payloads elegantly. Key fuzzy markers include:

*   `#null`: Validates the field is explicitly null.
*   `#notnull`: Validates the field is present and not null.
*   `#string`, `#number`, `#boolean`: Asserts data types effortlessly.
*   `#array`, `#object`: Asserts JSON structure types.
*   `#uuid`: Validates standard UUID formats.
*   `#regex`: Allows powerful regular expression matching directly within the JSON.
*   `#ignore`: Ignores a specific field during a strict match.
*   `#present`, `#notpresent`: Checks for the existence of a key regardless of value.
*   `#[] #object`: Validates an array where every element is an object.

#### 2. Mock Server (Service Virtualization)

When downstream services are unavailable, Karate lets you spin up a mock server using the exact same syntax, allowing teams to start testing before the real API is ready.

```gherkin
Feature: Mock Insurance Verification Service

  # This acts as a routing condition
  Scenario: pathMatches('/api/insurance/verify') && methodIs('post')
    # Extract data from the incoming request
    * def requestedId = request.insuranceId
    
    # Define the mock response dynamically
    * def response = { verified: true, coverage: 'FULL', copay: 25.00, id: '#(requestedId)' }
    * def responseStatus = 200
```

#### 3. Performance Testing with Gatling

Karate integrates beautifully with Gatling. You don't need to rewrite your API tests in Scala; you simply reuse your existing Karate `.feature` files and inject them into a Gatling simulation.

```scala
import com.intuit.karate.gatling.PreDef._
import io.gatling.core.Predef._

class PatientApiSimulation extends Simulation {
  val protocol = karateProtocol()

  val createPatient = scenario("Create Patient Load")
    .exec(karateFeature("classpath:patients/create.feature"))

  setUp(
    createPatient.inject(rampUsers(100).during(30))
  ).protocols(protocol)
}
```

### Karate vs REST-assured vs Postman

| Feature | Karate | REST-assured | Postman |
|---|---|---|---|
| Language | Gherkin (no code) | Java | JavaScript |
| Step definitions | NOT needed | N/A | N/A |
| Learning curve | Low | Medium | Low |
| JSON assertions | Built-in fuzzy | Hamcrest/JsonPath | chai-like (Chai JS) |
| Parallel execution | Built-in | TestNG/JUnit | Newman (needs wrappers) |
| Mock server | Built-in | WireMock (separate) | Mock server |
| Performance test | Gatling integration | JMeter (separate) | Not built-in |
| GraphQL | Native | Manual | Manual |

> ⭐ **STAR Moment --- The Quality Partner Advantage**
> In the SDSD-POD model, a Quality Partner who masters Karate can single-handedly build: API tests, contract validations, mock services for development, AND performance baselines --- all in one framework, all in readable Gherkin. This is the ultimate multiplier effect.

## Contract Testing Deep Dive: Pact

Traditional end-to-end API testing is fragile. It requires spinning up multiple microservices in a dedicated staging environment. If the MedPortal UI team expects a `patientName` field, but the Backend team renames it to `fullName`, the integration breaks, and the CI pipeline grinds to a halt.

**Consumer-Driven Contract Testing (CDCT)** solves this problem permanently. Using a tool like **Pact**:

1.  **The Consumer (e.g., MedPortal Web Frontend) defines a "Contract".** This contract dictates exactly what the frontend expects the backend to return.
2.  **The contract is published to a Pact Broker.** This acts as a central repository for all service contracts.
3.  **The Provider (e.g., Patient Records API Backend) runs automated tests against the contract.** During the backend's CI pipeline, it pulls the contract from the broker and verifies that its API responses fulfill the consumer's expectations.
4.  **The Result:** If the Backend team attempts to rename `patientName` to `fullName`, their build fails *locally or in their own CI* before deployment, preventing the integration break.

### The Pact Workflow in Practice

**Step 1: The Consumer Test (Java/JUnit)**
The frontend team writes a test defining the expected behavior.

```java
@ExtendWith(PactVerificationInvocationContextProvider.class)
public class MedPortalConsumerTest {

    @Pact(consumer = "MedPortal-Frontend", provider = "PatientRecords-API")
    public RequestResponsePact createPact(PactDslWithProvider builder) {
        return builder
            .given("A patient with ID 123 exists")
            .uponReceiving("A request for patient details")
            .path("/api/v1/patients/123")
            .method("GET")
            .willRespondWith()
            .status(200)
            .body(new PactDslJsonBody()
                .stringType("patientName", "John Doe")
                .date("dateOfBirth", "yyyy-MM-dd", Date.valueOf("1980-01-01"))
            )
            .toPact();
    }

    @Test
    @PactTestFor(pactMethod = "createPact")
    public void testPatientRetrieval(MockServer mockServer) {
        // The consumer tests its own code against the mock server generated by Pact
        // ... assertions ...
    }
}
```

**Step 2: The Provider Verification**
The backend team runs a test that replays the interactions defined in the pact file against their actual running API.

```java
@Provider("PatientRecords-API")
@PactBroker(host = "pact-broker.internal.medportal.com")
public class PatientRecordsProviderTest {

    @TestTemplate
    @ExtendWith(PactVerificationInvocationContextProvider.class)
    void pactVerificationTestTemplate(PactVerificationContext context) {
        context.verifyInteraction();
    }

    @State("A patient with ID 123 exists")
    public void setupPatientState() {
        // Setup database state for the provider test
        database.insertPatient(123, "John Doe", "1980-01-01");
    }
}
```

> **For the Candidate:** When asked about brittle integration tests in microservices, pivoting the conversation to Consumer-Driven Contract Testing using Pact instantly elevates you from a standard test executor to a Quality Architect.

## OpenAPI/Swagger Validation

The OpenAPI Specification (OAS) defines a standard, language-agnostic interface to RESTful APIs. 

A traditional QE reads the Swagger documentation to figure out what tests to write. A **Quality Partner** automates tests *against* the Swagger documentation. 

Using libraries like `swagger-request-validator` (in Java) or `openapi-response-validator` (in Node.js), you can intercept API traffic during your automated tests and automatically assert that every request and response perfectly matches the defined OpenAPI schema. 

If the spec says `totalAmount` is a `number`, but the API starts returning a `string` like `"100.00"`, the schema validator will automatically fail the test, even if you forgot to write a specific assertion for the `totalAmount` data type.

## GraphQL Testing

Unlike REST, which uses different URLs for different resources, GraphQL exposes a single endpoint (typically `/graphql`). Data retrieval is entirely dependent on the query structure sent by the client.

Testing GraphQL requires a different mindset:

*   **Queries:** Requesting specific deeply nested data structures. You must validate that the API returns *exactly* what was requested, no more, no less.
*   **Mutations:** Modifying data and ensuring the correct subset of updated fields are returned.
*   **Subscriptions:** Testing real-time WebSocket updates.
*   **Security & Validation:** GraphQL is highly susceptible to Denial of Service (DoS) attacks via deeply nested queries (e.g., requesting an author, their books, the author of those books, their books, ad infinitum). Testing must ensure the server enforces query depth limits and query complexity analysis.

## API Security Testing Basics (OWASP API Top 10)

You cannot be a Quality Partner without incorporating security into your API testing strategy. The OWASP API Security Top 10 highlights the most critical vulnerabilities. Here are key areas you must test, applied to our case studies:

*   **API1: Broken Object Level Authorization (BOLA/IDOR):** 
    *   *Test:* Can Patient A (ID: 100) authenticate, but then manipulate the URL to `GET /api/records/200` to view Patient B's records?

*   **API2: Broken Authentication:** 
    *   *Test:* Does the API accept expired JWT tokens? Does it lack brute-force protection on the login endpoint?

*   **API3: Broken Object Property Level Authorization (Mass Assignment):** 
    *   *Test:* In CartFlow, when a standard user updates their profile using `PUT /api/users/me`, what happens if they include `{"role": "admin"}` in the JSON body? Does the API blindly apply it?

*   **API4: Unrestricted Resource Consumption:** 
    *   *Test:* In TradeForge, can a user submit 10,000 orders per second, circumventing rate limits and crashing the matching engine?

*   **API5: Broken Function Level Authorization:** 
    *   *Test:* Can a regular user access an endpoint like `DELETE /api/admin/users/123` just by guessing the URL?

*   **API6: Unrestricted Access to Sensitive Business Flows:** 
    *   *Test:* Can an attacker automate the purchase of all inventory for a highly anticipated product in CartFlow using a bot script?

## Interview Scenarios & Mock Questions

> **For the Interviewer:** Stop asking candidates to recite HTTP status codes. Instead, ask them to design an API testing strategy for a new microservice. Look for discussions on mocking, schema validation, contract testing, and CI integration.

### Mock Questions

**1. "We are breaking our monolith into microservices. How should we approach API testing?"**
> *Ideal Answer:* "I would implement a layered approach. First, component-level API tests using Karate or REST-assured to verify business logic in isolation, aggressively mocking external dependencies. Second, I would introduce Consumer-Driven Contract Testing with Pact to ensure the microservices can communicate without relying on brittle, full-environment end-to-end tests. Finally, I would integrate lightweight smoke tests into the deployment pipelines to validate the OpenAPI schemas upon deployment to staging."

**2. "When would you choose Karate over REST-assured?"**
> *Ideal Answer:* "I would choose Karate if the team values rapid test creation and cross-functional readability without maintaining complex Java frameworks and step definitions. Its built-in fuzzy matching is unparalleled for complex JSON payloads. It's also ideal when we want to unify functional testing, mock servers, and Gatling performance tests under one tool. I would prefer REST-assured if the team is already deeply entrenched in Java and prefers programmatic test construction and complex custom assertions over DSLs."

**3. "How do you test the security of a REST API?"**
> *Ideal Answer:* "I structure my security tests around the OWASP API Top 10. For instance, I write tests to specifically check for BOLA (Broken Object Level Authorization) by attempting to access resource IDs belonging to other users. I test for Mass Assignment by injecting unauthorized fields like `isAdmin: true` into POST/PUT payloads. I also ensure robust negative testing for authentication---verifying that missing, malformed, or expired tokens result in a strict 401 response."

**4. "How do you handle API test data that changes constantly?"**
> *Ideal Answer:* "Hardcoding test data is an anti-pattern. I use pre-request scripts (in Postman) or dynamic payload generation (in Karate/REST-assured) to generate unique data like UUIDs or timestamps for every request. For data that requires a specific state, I use the API itself to create the necessary prerequisite data in a `BeforeSuite` or `Background` step, and then use that dynamically generated data for the core test, ensuring test isolation and stability."
