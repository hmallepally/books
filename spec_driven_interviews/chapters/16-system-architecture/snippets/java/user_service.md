```java
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    private final UserRepository dbRepository;
    private final RedisTemplate<String, User> redisTemplate;

    public UserService(UserRepository dbRepository, RedisTemplate<String, User> redisTemplate) {
        this.dbRepository = dbRepository;
        this.redisTemplate = redisTemplate;
    }

    public User getUser(String userId) {
        String cacheKey = "user:" + userId;
        User user = redisTemplate.opsForValue().get(cacheKey);
        
        if (user == null) {
            // Cache miss: read from DB
            user = dbRepository.findById(userId).orElseThrow();
            // Populate cache
            redisTemplate.opsForValue().set(cacheKey, user);
        }
        return user;
    }
}
```