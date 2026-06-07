```java
// Gemini structured output enforcement
GenerativeModel model = new GenerativeModel("gemini-2.5-pro");

Schema responseSchema = Schema.newBuilder()
    .setType(Type.OBJECT)
    .putProperties("implementation", Schema.newBuilder()
        .setType(Type.STRING).build())
    .putProperties("invariants_verified", Schema.newBuilder()
        .setType(Type.ARRAY)
        .setItems(Schema.newBuilder().setType(Type.STRING).build())
        .build())
    .putProperties("blast_radius_files_modified", Schema.newBuilder()
        .setType(Type.ARRAY)
        .setItems(Schema.newBuilder().setType(Type.STRING).build())
        .build())
    .addAllRequired(List.of("implementation",
        "invariants_verified", "blast_radius_files_modified"))
    .build();

GenerationConfig config = GenerationConfig.newBuilder()
    .setResponseMimeType("application/json")
    .setResponseSchema(responseSchema)
    .build();

GenerateContentResponse response = model.generateContent(
    sdsdPrompt, config);
```