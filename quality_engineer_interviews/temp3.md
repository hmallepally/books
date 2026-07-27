
\bigskip

# Security Testing Fundamentals

In modern software development, security is no longer solely the responsibility of a dedicated InfoSec team. With the rise of "Shift-Left" security, Quality Engineers are expected to identify basic vulnerabilities early in the software development lifecycle. Understanding security fundamentals is essential for regulatory compliance, protecting user data, and passing senior-level QE interviews.

## The OWASP Top 10 for Quality Engineers

The Open Worldwide Application Security Project (OWASP) Top 10 is the standard awareness document for web application security. Here is an overview of how QEs should approach testing the most critical risks:

### 1. Broken Access Control
**The Risk:** Users can perform actions outside of their intended permissions.
**How to Test:** 
*   Create a Role-Based Access Control (RBAC) matrix.
*   Attempt horizontal privilege escalation: Try to access another user's private data by manipulating IDs in the URL (e.g., changing `/user/123/profile` to `/user/124/profile`).
*   Attempt vertical privilege escalation: Log in as a standard user and attempt to access admin-only endpoints or UI elements.

### 2. Cryptographic Failures
**The Risk:** Sensitive data (passwords, health records, credit cards) is exposed due to lack of encryption.
**What to Check:**
*   Ensure all data in transit uses HTTPS/TLS. Reject HTTP connections.
*   Verify that passwords are never stored in plain text (they should be salted and hashed).
*   Check that sensitive data is masked in the UI and not exposed in API response payloads unless strictly necessary.

### 3. Injection (SQL, XSS, Command)
**The Risk:** Untrusted data is sent to an interpreter as part of a command or query.
**Safe Testing Examples:**
*   **SQLi:** Enter `' OR 1=1 --` into login or search fields. A secure application will treat this as a literal string (using parameterized queries) rather than executing it as code.
*   **Command Injection:** If an application pings an IP address, enter `127.0.0.1 & dir` (Windows) or `127.0.0.1; ls` (Linux) to see if underlying system commands are executed.

### 4. Insecure Design
**The Risk:** Flaws in the architecture or business logic that cannot be fixed by simple coding changes.
**How to Test:**
*   Review security requirements during sprint planning and user story refinement.
*   Ask questions like: "What happens if a user bypasses the UI and calls this API directly?" or "Is there a rate limit on this password reset endpoint?"

### 5. Security Misconfiguration
**The Risk:** Insecure default settings, open cloud storage, or verbose error messages.
**Checklist Approach:**
*   Ensure custom error pages are used. Stack traces or database errors should *never* be visible to the end user.
*   Verify that default accounts and passwords have been disabled or changed.
*   Ensure directory listing is disabled on the web server.

### 6-10. Brief Overview of Remaining Risks
*   **6. Vulnerable and Outdated Components:** Check dependency scanners (like Dependabot) to ensure third-party libraries don't have known CVEs.
*   **7. Identification and Authentication Failures:** Test for weak password policies, lack of MFA, and session fixation vulnerabilities.
*   **8. Software and Data Integrity Failures:** Verify that CI/CD pipelines use signed commits and secure artifact repositories.
*   **9. Security Logging and Monitoring Failures:** Ensure that critical events (failed logins, high-value transactions) are logged accurately for auditing.
*   **10. Server-Side Request Forgery (SSRF):** If the application fetches resources from external URLs, test if you can force it to access internal, protected servers (e.g., `http://localhost/admin`).

## SQL Injection (SQLi) Testing Walkthrough

SQL Injection occurs when malicious SQL statements are inserted into entry fields for execution.

**How to Test (Safely):**
1. Identify input fields that interact with the database (Login, Search, Filters).
2. Input a single quote `'`. If the application throws a database syntax error, it is likely vulnerable.
3. Input a tautology: `admin' OR '1'='1`. If this bypasses authentication, the system is highly vulnerable.

**The Fix (What developers should do):**
The application must use Parameterized Queries (Prepared Statements) or an ORM.
*   *Vulnerable:* `SELECT * FROM users WHERE username = '` + userInput + `'`
*   *Secure:* `SELECT * FROM users WHERE username = ?`

## Cross-Site Scripting (XSS) Testing Walkthrough

XSS occurs when an application includes untrusted data in a web page without proper validation or escaping.

**Types of XSS to Test:**
1.  **Reflected XSS:** The malicious script comes from the current HTTP request.
    *   *Test:* Append `<script>alert('XSS')</script>` to a search URL parameter. If the alert box pops up on the resulting page, it's vulnerable.
2.  **Stored XSS:** The malicious script is saved on the server and served to users later.
    *   *Test:* Enter `<script>alert('XSS')</script>` into a comment field or profile bio. Save it. If the alert pops up every time you (or anyone else) visits that page, it's vulnerable.
3.  **DOM-based XSS:** The vulnerability exists in client-side code rather than server-side code.

**The Fix:** The application must encode/sanitize output (e.g., converting `<` to `&lt;`).

## Authentication & Authorization Testing Checklist

*   [ ] Are passwords required to meet complexity requirements (length, special characters)?
*   [ ] Does the system implement account lockout after X failed attempts?
*   [ ] Are session tokens (cookies, JWTs) invalidated immediately upon logout?
*   [ ] Do session tokens expire after a reasonable period of inactivity?
*   [ ] Is the "Secure" flag set on session cookies (ensuring they are only sent over HTTPS)?
*   [ ] Is the "HttpOnly" flag set on session cookies (preventing access via client-side JavaScript)?
*   [ ] Can a user access admin APIs by simply changing their role ID in a JWT payload?

## Common Security Testing Tools

While manual testing is important, QEs should be familiar with automated security tools:
*   **OWASP ZAP (Zed Attack Proxy):** A free, open-source penetration testing tool for finding vulnerabilities in web applications. Great for intercepting traffic and automated scanning.
*   **Burp Suite:** The industry standard for web application security testing. The proxy feature allows QEs to manipulate requests before they hit the server.
*   **SonarQube:** A static application security testing (SAST) tool that analyzes source code for vulnerabilities and code smells during the CI/CD pipeline.

## Interview Question: "How do you approach security testing in your current role?"

**Model STAR Answer:**

*   **Situation:** "In my current role, security wasn't initially integrated into our agile testing process. We were relying on annual penetration tests, which meant vulnerabilities were found very late."
*   **Task:** "I took the initiative to implement a 'shift-left' security approach to catch basic vulnerabilities during the normal QA cycle."
*   **Action:** "I started by training the QA team on the OWASP Top 10. We incorporated security checks into our standard test case templates—for example, adding XSS and SQL injection payloads to all input validation tests, and creating RBAC matrices for authorization testing. I also integrated a tool like OWASP ZAP into our CI/CD pipeline to run baseline dynamic scans automatically."
*   **Result:** "As a result, we caught and fixed over 15 medium-to-high severity vulnerabilities before they reached the external pen testers, saving the company significant remediation time and money."
