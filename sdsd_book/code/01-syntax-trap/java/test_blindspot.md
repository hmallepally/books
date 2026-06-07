```java
// src/test/java/com/aetherfi/controllers/TransactionControllerTest.java (AI-Generated)
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;
import org.springframework.test.web.servlet.MockMvc;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
@SpringBootTest
@AutoConfigureMockMvc
public class TransactionControllerTest {
    @Autowired
    private MockMvc mockMvc;
    @MockBean
    private RedisTemplate<String, String> redisTemplate;
    @MockBean
    private ValueOperations<String, String> valueOperations;
    @Test
    public void testTransactionCaching() throws Exception {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get(anyString())).thenReturn(null).thenReturn("[{...}]");
        // First request: Cache Miss
        mockMvc.perform(get("/api/v1/transactions?limit=10")
                .header("Authorization", "Bearer valid-token"))
                .andExpect(status().isOk());
        verify(valueOperations, times(1)).get(anyString());
        verify(valueOperations, times(1)).set(anyString(), anyString(), any());
        // Second request: Cache Hit
        mockMvc.perform(get("/api/v1/transactions?limit=10")
                .header("Authorization", "Bearer valid-token"))
                .andExpect(status().isOk());
        verify(valueOperations, times(2)).get(anyString());
        // Verify set was not called again (cache hit)
        verify(valueOperations, times(1)).set(anyString(), anyString(), any());
    }
}
```