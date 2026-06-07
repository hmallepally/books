```java
// SECURE IMPLEMENTATION
private static final Set<String> ALLOWED_SORT_COLUMNS = Set.of("amount", "created_at", "status");
@GetMapping("/admin/transactions/report")
public List<Transaction> getReport(
    @RequestParam String sortBy, 
    @RequestAttribute("adminOrgId") String adminOrgId
) {
    // THREAT MITIGATION: SQLi Whitelist Validation
    if (!ALLOWED_SORT_COLUMNS.contains(sortBy)) {
        throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid sort parameter.");
    }
    // INVARIANT: Tenant Isolation via Spring Data JPA (No raw SQL)
    // The ORM automatically escapes all inputs, preventing SQLi.
    Sort sort = Sort.by(Sort.Direction.DESC, sortBy);
    return transactionRepository.findByOrgId(adminOrgId, sort);
}
```