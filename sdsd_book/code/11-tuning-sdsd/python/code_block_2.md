```python
# Gemini structured output enforcement
import google.generativeai as genai

model = genai.GenerativeModel("gemini-2.5-pro")
response = model.generate_content(
    sdsd_prompt,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "implementation": {"type": "string"},
                "invariants_verified": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "blast_radius_files_modified": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["implementation", "invariants_verified",
                         "blast_radius_files_modified"]
        }
    )
)
```