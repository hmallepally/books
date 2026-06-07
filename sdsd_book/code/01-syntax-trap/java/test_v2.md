```java
    private String generateCacheKey(HttpServletRequest request, String userId) throws Exception {
        /** Generates a tenant-isolated cache key. */
        String queryString = request.getQueryString();
        MessageDigest md = MessageDigest.getInstance("MD5");
        // ... hash generation logic ...
        // The cache key is now rigidly bound to the tenant ID
        return "cache:tenant:" + userId + ":transactions:" + hashString;
    }
```