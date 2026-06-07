```java
// Pipeline integration for OpenAI models
ChatCompletionRequest request = ChatCompletionRequest.builder()
    .model("gpt-4o")
    .messages(List.of(
        new ChatMessage(ChatMessageRole.SYSTEM.value(),
            "You are a Zero-Trust Security Architect.\n"
            + "\n"
            + "ABSOLUTE INVARIANTS (NEVER VIOLATE):\n"
            + "1. tenant_id isolation on ALL queries\n"
            + "2. BigDecimal for ALL financial math\n"
            + "3. bcrypt for ALL password hashing\n"
            + "\n"
            + "BLAST RADIUS: You may ONLY modify files in src/transfer/\n"
            + "You are FORBIDDEN from touching src/auth/ or src/crypto/\n"
            + "\n"
            + "STATE MACHINE: DRAFT -> PENDING -> SETTLED | FAILED\n"
            + "No other transitions are permitted."),
        new ChatMessage(ChatMessageRole.USER.value(),
            sdsdPromptWithCodeContext)
    ))
    .build();

ChatCompletionResult result = openAiService.createChatCompletion(request);
```