```python
import re

class LlmGatewaySecurityFilter:
    _INJECTION_PATTERN = re.compile(
        r"(ignore all previous instructions|system prompt|bypass validation|reveal key)",
        re.IGNORECASE
    )

    def validate_prompt(self, user_prompt: str) -> bool:
        if not user_prompt or not user_prompt.strip():
            return False
        # Fail-fast if malicious injection signature detected
        if self._INJECTION_PATTERN.search(user_prompt):
            raise PermissionError("Potential prompt injection attack blocked")
        return True
```