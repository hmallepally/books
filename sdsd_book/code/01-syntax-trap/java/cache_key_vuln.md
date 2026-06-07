```java
    private String generateCacheKey(HttpServletRequest request) throws Exception {
        String queryString = request.getQueryString();
        MessageDigest md = MessageDigest.getInstance("MD5");
        // ... hash generation ...
        return "cache:transactions:" + hashString;
    }
```