```java
// src/main/java/com/aetherfi/controllers/TransactionController.java (AI-Generated)
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.security.MessageDigest;
import java.util.List;
import javax.servlet.http.HttpServletRequest;
@RestController
@RequestMapping("/api/v1")
public class TransactionController {
    @Autowired
    private RedisTemplate<String, String> redisTemplate;
    @Autowired
    private TransactionRepository transactionRepository;
    @Autowired
    private ObjectMapper objectMapper;
    /**
     * Generates a unique cache key based on the HTTP query parameters.
     * Hashes the query string to prevent excessively long Redis keys.
     */
    private String generateCacheKey(HttpServletRequest request) throws Exception {
        String queryString = request.getQueryString();
        if (queryString == null || queryString.isEmpty()) {
            return "cache:transactions:default";
        }
        // AI Logic: MD5 hash is fast and deterministic for caching
        MessageDigest md = MessageDigest.getInstance("MD5");
        byte[] hashBytes = md.digest(queryString.getBytes());
        StringBuilder sb = new StringBuilder();
        for (byte b : hashBytes) {
            sb.append(String.format("%02x", b));
        }
        return "cache:transactions:" + sb.toString();
    }
    @GetMapping("/transactions")
    public List<TransactionResponse> getUserTransactions(
            HttpServletRequest request,
            @RequestParam(defaultValue = "50") int limit,
            @RequestParam(defaultValue = "0") int offset,
            @RequestAttribute("currentUser") User currentUser) throws Exception {
        // 1. Generate the cache key based on the request URL
        String cacheKey = generateCacheKey(request);
        // 2. Check the Redis cache for existing data
        String cachedData = redisTemplate.opsForValue().get(cacheKey);
        if (cachedData != null) {
            return objectMapper.readValue(cachedData, 
                objectMapper.getTypeFactory().constructCollectionType(List.class, TransactionResponse.class));
        }
        // 3. Cache Miss: Fetch from the database
        // Note: currentUser.getId() is correctly passed to the DB layer
        List<TransactionResponse> transactions = transactionRepository.fetchTransactions(
            currentUser.getId(), limit, offset);
        // 4. Store the result in Cache (TTL 5 minutes)
        redisTemplate.opsForValue().set(cacheKey, objectMapper.writeValueAsString(transactions), 
            java.time.Duration.ofMinutes(5));
        return transactions;
    }
}
```