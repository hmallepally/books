import re

path = "c:/Users/hari/Documents/DBA/735/Week4/Walmart_Failures_Case_Study.md"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace all variants of the old title with the new one
# Handle both smart apostrophe and regular apostrophe
old_title = "Walmart\u2019s Failures in Entering Developed Markets: An Organizational Design and Leadership Analysis"
new_title = "Walmart\u2019s Failures in Entering Developed Markets: Case Study"

count = content.count(old_title)
content = content.replace(old_title, new_title)

print(f"Replaced {count} occurrences of old title")

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(content)
print("Saved!")
