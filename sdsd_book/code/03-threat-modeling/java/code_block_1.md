```java
// WARNING: VULNERABLE CODE
@GetMapping("/admin/transactions/report")
public List<Map<String, Object>> getReport(@RequestParam String sortBy) {
    // The AI uses string concatenation for dynamic column sorting
    String query = "SELECT * FROM transactions ORDER BY " + sortBy + " DESC";
    return jdbcTemplate.queryForList(query);
}
```