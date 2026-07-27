import os
import re

chapters = [
    '00-prologue', '01-invariant-first', '02-problem-decomposition', '03-case-studies', 
    '04-oop-principles', '05-solid-boundaries', '06-functional-streams', '07-design-patterns', 
    '08-concurrency-performance', '09-algorithms-assessment', '11-matrix-grid-patterns', 
    '13-optimization-dp', '14-mastering-decomposition', '16-system-architecture', 
    '17-resiliency', '18-database-compliance', '19-behavioral-leadership', '20-testing-cicd', 
    '21-message-brokers', '22-aiml-llm', '23-appendix'
]

reverted = 0

for chapter in chapters:
    filepath = f"chapters/{chapter}/base.md"
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    orig_content = content
    
    # Apply precise safe regex
    content = content.replace('> * **STAR Moment:', '> ⭐ **STAR Moment:')
    content = content.replace('> WARNING: **', '> ⚠️ **')
    content = content.replace('$\\mu$', 'μ')
    content = content.replace('$\\square$', '∎')
    
    # For leq, geq, neq, we only replace if it's the exact token bounded by spaces or punctuation,
    # or just literally replacing the exact string `$\leq$` -> `≤`
    # The prompt says: "$\leq$ back to ≤ in running text (NOT in LaTeX math blocks)"
    # LaTeX math blocks usually look like `$$...$$` or `$ O(N \leq M) $`.
    # A standalone replacement by the previous subagent would literally be `$\leq$` surrounded by spaces.
    
    # We can use regex with negative lookbehind/lookahead to avoid replacing if it's part of a larger math expression.
    # If the previous subagent replaced `≤` with `$\leq$`, it's literally just the word.
    # Wait, the instruction says:
    # "2. $\leq$ back to ≤ in running text (NOT in LaTeX math blocks)"
    # "3. $\geq$ back to ≥ in running text"
    # "4. $\neq$ back to ≠ in running text"
    
    # Let's just replace them if they are alone, e.g. ` $\leq$ ` -> ` ≤ `
    content = re.sub(r'(?<!\$)\$\\leq\$(?!\$)', '≤', content)
    content = re.sub(r'(?<!\$)\$\\geq\$(?!\$)', '≥', content)
    content = re.sub(r'(?<!\$)\$\\neq\$(?!\$)', '≠', content)
    
    if content != orig_content:
        reverted += 1
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

print(f"Modified {reverted} files.")
