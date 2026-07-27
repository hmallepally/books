
\bigskip

# Test Case Writing Workshop: Interview Exercises

In many Quality Engineering interviews, you will be asked to demonstrate your practical testing mindset by writing test cases for a common application feature on the spot.

Below are 5 complete "Write test cases for X" exercises with model answers. For each feature, tests are categorized to show a structured, comprehensive approach.

## 1. Login Page

**Scenario:** Write test cases for a standard login page containing a Username/Email field, a Password field, a "Remember Me" checkbox, a "Show Password" toggle, and a Login button.

**Positive Test Cases:**
*   Verify successful login with valid email and correct password.
*   Verify successful login with valid username (if supported) and correct password.
*   Verify the "Remember Me" checkbox functionality keeps the user logged in after closing and reopening the browser.
*   Verify that clicking the "Show Password" toggle displays the password in plain text, and toggling it again masks it.

**Negative Test Cases:**
*   Verify login fails with a valid email but incorrect password.
*   Verify login fails with an unregistered email address.
*   Verify login fails when both email and password fields are left empty.
*   Verify login fails when only the password field is empty.
*   Verify login fails when only the email field is empty.
*   Verify that trailing/leading spaces in the email field are handled correctly (either trimmed successfully or rejected gracefully).
*   Verify login fails if the email format is invalid (e.g., `user@.com`).

**Security Test Cases:**
*   Verify that multiple failed login attempts trigger an account lockout or CAPTCHA (e.g., after 5 failed attempts).
*   Verify that the application is not vulnerable to basic SQL injection in the login fields (e.g., entering `' OR 1=1 --`).
*   Verify that password data is masked by default (`type="password"`).
*   Verify that session times out after a period of inactivity (Session Timeout).
*   Verify that clicking the "Back" button after logging out does not allow access to authenticated pages.

**Edge Cases & Other:**
*   Verify SSO (Single Sign-On) integration works correctly, if applicable (e.g., "Login with Google").
*   Verify the behavior when a user tries to log in with an account that has been disabled or banned.

## 2. Shopping Cart

**Scenario:** Write test cases for an e-commerce shopping cart where users can add items, update quantities, apply discount codes, and proceed to checkout.

**Positive Test Cases:**
*   Verify a user can add a single item to the empty cart.
*   Verify a user can add multiple different items to the cart.
*   Verify a user can increase the quantity of an item already in the cart.
*   Verify a user can decrease the quantity of an item in the cart.
*   Verify the cart total calculates correctly based on items and quantities.
*   Verify a user can successfully apply a valid coupon code and the discount is reflected in the total.
*   Verify a user can remove an item completely from the cart.
*   Verify a user can empty the entire cart.

**Negative Test Cases:**
*   Verify a user cannot update the quantity to a negative number or zero (zero should ideally remove the item).
*   Verify a user cannot add more items than the current available inventory (Out-of-stock handling).
*   Verify a user cannot apply an expired or invalid coupon code.
*   Verify a user cannot apply multiple mutually exclusive coupon codes.

**Edge Cases & Performance:**
*   Verify the maximum quantity limit for a single item (e.g., trying to add 9,999 units).
*   Verify price recalculation happens in real-time when quantities change.
*   Verify cart persistence: items remain in the cart if the user closes the browser and returns later (if logged in or via cookies).
*   Verify behavior when an item in the cart becomes out of stock before the user checks out.
*   Verify cart behavior when multiple tabs are open and cart state is modified in one tab.

## 3. File Upload

**Scenario:** Write test cases for a profile picture file upload feature.

**Positive Test Cases:**
*   Verify successful upload of a valid file type (e.g., .jpg, .png) that is within the acceptable size limit.
*   Verify that the newly uploaded image is displayed correctly on the profile page.
*   Verify the progress bar updates accurately during a large file upload.

**Negative Test Cases:**
*   Verify upload fails when attempting to upload an unsupported file type (e.g., .exe, .sh, .pdf).
*   Verify upload fails when attempting to upload a file that exceeds the maximum size limit (e.g., > 5MB).
*   Verify upload fails gracefully when a zero-byte (empty) file is selected.
*   Verify the system handles files with special characters in the filename (e.g., `my_pic!@#.jpg`).
*   Verify upload fails if the user attempts to submit the form without selecting a file.

**Security & Edge Cases:**
*   Verify that the system detects and blocks a virus-infected file (if antivirus scanning is integrated).
*   Verify that a file with a spoofed extension (e.g., a `.exe` file renamed to `.jpg`) is rejected.
*   Verify behavior when the network connection is interrupted during the upload process.
*   Verify concurrent uploads (if the UI allows multiple files, or if the user clicks the upload button rapidly multiple times).

## 4. Search Functionality

**Scenario:** Write test cases for a global search bar on an e-commerce website.

**Positive Test Cases:**
*   Verify search returns accurate results for an exact product name match.
*   Verify search returns relevant results for a partial match or substring.
*   Verify search handles variations in casing (case-insensitive search).
*   Verify search returns results when searching by product category or keywords.
*   Verify sorting options work correctly on the search results page (e.g., Sort by Price: Low to High).
*   Verify filters work correctly on the search results page (e.g., Filter by Brand).
*   Verify pagination works correctly when there are many search results.

**Negative Test Cases:**
*   Verify the system displays an appropriate "No results found" message when searching for a non-existent item.
*   Verify the behavior when an empty search query is submitted (should either do nothing or return a prompt).
*   Verify search handles special characters gracefully (e.g., searching for `%`, `*`, or `?`).

**Security & Performance:**
*   Verify the search input is not vulnerable to SQL Injection (e.g., `' OR '1'='1`).
*   Verify the search input is not vulnerable to Cross-Site Scripting (XSS) (e.g., entering `<script>alert(1)</script>` and ensuring it is sanitized and not executed).
*   Verify search performance with a very large dataset (results should load within acceptable SLA, e.g., < 2 seconds).
*   Verify search performance when querying a very long string (e.g., 500+ characters).

## 5. Payment Processing

**Scenario:** Write test cases for a credit card payment gateway on a checkout page.

**Positive Test Cases:**
*   Verify successful payment processing using a valid credit card.
*   Verify successful payment processing using different valid card types (Visa, MasterCard, Amex).
*   Verify the successful completion of a 3D Secure authentication flow (if applicable).
*   Verify the user receives a confirmation email/receipt after a successful payment.
*   Verify that a refund flow (full refund) processes correctly from the admin dashboard.
*   Verify that a partial refund processes correctly.
*   Verify that the correct currency conversion is applied if the user is purchasing in a foreign currency.

**Negative Test Cases:**
*   Verify payment fails when an expired credit card is used.
*   Verify payment fails when a card with insufficient funds is used.
*   Verify payment fails when an invalid CVV/CVC is entered.
*   Verify payment fails when an invalid credit card format (e.g., letters instead of numbers) is entered.
*   Verify payment fails if mandatory fields (e.g., Billing Address) are left blank.

**Edge Cases & Security:**
*   Verify duplicate payment prevention: If the user double-clicks the "Pay Now" button rapidly, only one transaction should be processed.
*   Verify timeout handling: What happens if the payment gateway API takes too long to respond? (Should fail gracefully and not charge the user).
*   Verify that sensitive card data (like the full PAN) is masked on the UI and never stored in plain text in the database (PCI compliance check).
