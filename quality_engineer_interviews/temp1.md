
\bigskip

# SQL for Quality Engineers

Quality Engineers (QEs) frequently interact with databases to perform thorough testing. Understanding SQL is critical for several reasons:

* **Data Validation:** Ensuring that the application correctly stores, updates, and retrieves data according to business rules.
* **Test Data Setup:** Creating specific data states required to test various edge cases or complex business logic.
* **Defect Investigation:** When a bug occurs, querying the database can help isolate whether the issue is in the UI, the API, or the data layer itself.

Below are 15 progressively harder SQL problems specifically framed for testing scenarios.

## Data Validation Queries

### 1. Find duplicate records in a customer table

**Problem:** You are testing a registration flow and want to ensure the system is not creating duplicate customer records based on email addresses.
**Schema Context:** `customers (id, first_name, last_name, email, created_at)`
**SQL Solution:**
```sql
SELECT email, COUNT(*)
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;
```
**Explanation:** The `GROUP BY` clause groups the records by email address. The `HAVING` clause filters the results to only show groups that have more than one record, which indicates duplicates.

### 2. Find NULL values in required fields

**Problem:** A bug was reported where some users lack a last name. You need to identify all affected records in the database.
**Schema Context:** `users (user_id, first_name, last_name, email)`
**SQL Solution:**
```sql
SELECT user_id, first_name, email
FROM users
WHERE last_name IS NULL;
```
**Explanation:** The `WHERE last_name IS NULL` condition specifically checks for the absence of a value (NULL) in the `last_name` column.

### 3. Validate referential integrity: orders without matching customers

**Problem:** You suspect that when a customer is deleted, their orders are not being removed (an orphaned record issue). You need to find any orders that reference a non-existent customer.
**Schema Context:** `orders (order_id, customer_id, total_amount)`, `customers (customer_id, name)`
**SQL Solution:**
```sql
SELECT o.order_id, o.customer_id
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;
```
**Explanation:** A `LEFT JOIN` returns all records from the `orders` table and the matched records from `customers`. If an order's `customer_id` doesn't exist in the `customers` table, the `c.customer_id` will be NULL.

### 4. Check data type consistency: find non-numeric values in a numeric column

**Problem:** A legacy `zip_code` column is stored as a string (VARCHAR), but it should only contain numeric characters. Find all records with invalid zip codes.
**Schema Context:** `addresses (address_id, street, city, zip_code)`
**SQL Solution:**
*(Note: Syntax varies by SQL dialect; this uses T-SQL/SQL Server style)*
```sql
SELECT address_id, zip_code
FROM addresses
WHERE TRY_CAST(zip_code AS INT) IS NULL 
  AND zip_code IS NOT NULL;
```
**Explanation:** `TRY_CAST` attempts to convert the string to an integer. If it fails (because it contains letters or symbols), it returns NULL, highlighting the invalid data.

### 5. Validate date ranges: find records with end_date before start_date

**Problem:** Testing a subscription service, you need to ensure no subscriptions were created with an end date that occurs before the start date.
**Schema Context:** `subscriptions (sub_id, user_id, start_date, end_date)`
**SQL Solution:**
```sql
SELECT sub_id, start_date, end_date
FROM subscriptions
WHERE end_date < start_date;
```
**Explanation:** A simple comparison operator (`<`) is used in the `WHERE` clause to find illogical date combinations.

## Test Data & Investigation Queries

### 6. Compare record counts between staging and production tables

**Problem:** After a database migration, you need a quick sanity check to ensure the row counts match between the old (production backup) and new (staging) tables.
**Schema Context:** `prod.transactions`, `staging.transactions`
**SQL Solution:**
```sql
SELECT 'Production' AS Environment, COUNT(*) AS TotalRecords FROM prod.transactions
UNION ALL
SELECT 'Staging' AS Environment, COUNT(*) AS TotalRecords FROM staging.transactions;
```
**Explanation:** `UNION ALL` combines the results of the two aggregate queries into a single result set for easy comparison.

### 7. Find records that changed between two database snapshots

**Problem:** You ran a test suite and want to see exactly which product prices were modified during the run.
**Schema Context:** `products_before_test (product_id, price)`, `products_after_test (product_id, price)`
**SQL Solution:**
```sql
SELECT product_id, price FROM products_after_test
EXCEPT
SELECT product_id, price FROM products_before_test;
```
**Explanation:** The `EXCEPT` operator returns all distinct rows from the first query that are not present in the second query's results.

### 8. Generate test data: INSERT with random values

**Problem:** You need to create 5 test users quickly with random active statuses for an automated test.
**Schema Context:** `test_users (username, is_active, created_date)`
**SQL Solution:**
*(Syntax for PostgreSQL)*
```sql
INSERT INTO test_users (username, is_active, created_date)
SELECT 
    'user_' || generate_series(1, 5),
    (random() > 0.5),
    CURRENT_DATE;
```
**Explanation:** `generate_series` creates 5 rows. `random() > 0.5` generates a boolean (true/false) randomly, allowing for quick mass data generation.

### 9. Find the most recent record per customer

**Problem:** To test the "last login" feature, you need to retrieve only the most recent login event for every user.
**Schema Context:** `login_history (login_id, user_id, login_timestamp, ip_address)`
**SQL Solution:**
```sql
WITH RankedLogins AS (
    SELECT user_id, login_timestamp, ip_address,
           ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY login_timestamp DESC) as rn
    FROM login_history
)
SELECT user_id, login_timestamp, ip_address
FROM RankedLogins
WHERE rn = 1;
```
**Explanation:** The `ROW_NUMBER()` window function assigns a sequential integer to each row within a partition (grouped by `user_id`), ordered by timestamp descending. Filtering for `rn = 1` gives the latest record.

### 10. Track defect trends: count bugs by severity per sprint

**Problem:** You are building a quality metrics dashboard and need to count the number of defects grouped by their severity, for a specific sprint.
**Schema Context:** `defects (defect_id, sprint_id, severity, status)`
**SQL Solution:**
```sql
SELECT 
    sprint_id,
    COUNT(CASE WHEN severity = 'Critical' THEN 1 END) AS Critical_Bugs,
    COUNT(CASE WHEN severity = 'High' THEN 1 END) AS High_Bugs,
    COUNT(CASE WHEN severity = 'Medium' THEN 1 END) AS Medium_Bugs,
    COUNT(CASE WHEN severity = 'Low' THEN 1 END) AS Low_Bugs
FROM defects
WHERE sprint_id = 42
GROUP BY sprint_id;
```
**Explanation:** This uses conditional aggregation. The `CASE` statement inside the `COUNT` function only tallies rows that match the specific severity.

## Advanced Testing Queries

### 11. Data migration validation: compare checksums across source and target

**Problem:** You need a highly reliable way to verify that a large table was copied perfectly, without comparing millions of individual rows.
**Schema Context:** `source_table`, `target_table`
**SQL Solution:**
*(Syntax varies heavily; example using SQL Server `CHECKSUM_AGG`)*
```sql
SELECT 'Source' AS db, CHECKSUM_AGG(BINARY_CHECKSUM(*)) AS CheckSumValue 
FROM source_table
UNION ALL
SELECT 'Target' AS db, CHECKSUM_AGG(BINARY_CHECKSUM(*)) AS CheckSumValue 
FROM target_table;
```
**Explanation:** `BINARY_CHECKSUM(*)` generates a hash for each row, and `CHECKSUM_AGG` aggregates them into a single value for the entire table. If the values match, the tables are identical.

### 12. Find orphaned records after a cascade delete

**Problem:** A bug was reported where deleting a parent 'Project' failed to delete associated 'Tasks' because cascade delete wasn't configured properly. Find all such tasks.
**Schema Context:** `projects (project_id, name)`, `tasks (task_id, project_id, name)`
**SQL Solution:**
```sql
SELECT t.task_id, t.name, t.project_id
FROM tasks t
WHERE NOT EXISTS (
    SELECT 1 
    FROM projects p 
    WHERE p.project_id = t.project_id
);
```
**Explanation:** The `NOT EXISTS` subquery efficiently checks if there is any matching `project_id` in the `projects` table for the given task. If not, the task is orphaned.

### 13. Verify pagination: ensure no gaps in sequential IDs

**Problem:** You are testing an API that requires sequential transaction IDs. You need to find if there are any gaps in the sequence.
**Schema Context:** `transactions (transaction_id, amount)`
**SQL Solution:**
```sql
WITH Sequenced AS (
    SELECT transaction_id, 
           LAG(transaction_id) OVER (ORDER BY transaction_id) as prev_id
    FROM transactions
)
SELECT prev_id + 1 AS missing_start, transaction_id - 1 AS missing_end
FROM Sequenced
WHERE transaction_id - prev_id > 1;
```
**Explanation:** The `LAG` window function looks at the previous row's `transaction_id`. If the difference between the current ID and the previous ID is greater than 1, a gap exists.

### 14. Calculate test execution trends: pass rate over time

**Problem:** You need to calculate the daily pass rate (percentage of passed tests) for an automated test suite over the last 7 days.
**Schema Context:** `test_runs (run_id, execution_date, status)`
**SQL Solution:**
```sql
SELECT 
    execution_date,
    COUNT(run_id) AS total_runs,
    SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) AS passed_runs,
    (SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) * 100.0 / COUNT(run_id)) AS pass_rate_percentage
FROM test_runs
WHERE execution_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY execution_date
ORDER BY execution_date DESC;
```
**Explanation:** This query combines grouping by date with conditional sums to calculate the numerator (passed tests) and denominator (total tests) to derive a percentage.

### 15. Complex JOIN: validate that API response data matches database state

**Problem:** A user's "Total Balance" in the API is the sum of their "Checking" and "Savings" account balances, minus any "Pending Fees". You need to write a query to calculate this exact value from the database to validate the API response.
**Schema Context:** `users (user_id)`, `accounts (account_id, user_id, account_type, balance)`, `fees (fee_id, user_id, amount, status)`
**SQL Solution:**
```sql
SELECT 
    u.user_id,
    COALESCE(SUM(CASE WHEN a.account_type IN ('Checking', 'Savings') THEN a.balance ELSE 0 END), 0) 
    - COALESCE((SELECT SUM(amount) FROM fees f WHERE f.user_id = u.user_id AND f.status = 'Pending'), 0) AS calculated_total_balance
FROM users u
LEFT JOIN accounts a ON u.user_id = a.user_id
GROUP BY u.user_id;
```
**Explanation:** This involves joining `users` and `accounts`, conditionally summing balances based on account type, and using a correlated subquery (or another join) to subtract the pending fees. `COALESCE` handles potential NULL values if a user has no accounts or fees.
