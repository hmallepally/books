```java
import java.util.regex.Pattern;

public class LlmGatewaySecurityFilter {
    // Basic prompt injection defensive pattern match
    private static final Pattern INJECTION_PATTERN = Pattern.compile(
        "(ignore all previous instructions|system prompt|bypass validation|reveal key)",
        Pattern.CASE_INSENSITIVE
    );

    public boolean validatePrompt(String userPrompt) {
        if (userPrompt == null || userPrompt.trim().isEmpty()) {
            return false;
        }
        // Fail-fast if malicious injection signature detected
        if (INJECTION_PATTERN.matcher(userPrompt).find()) {
            throw new SecurityException("Potential prompt injection attack blocked");
        }
        return true;
    }
}
```