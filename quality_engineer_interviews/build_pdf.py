import os
import re
import subprocess
import shutil

def combine_chapters(base_dir):
    """
    Combines all base.md files in the sorted directory structure.
    """
    combined_content = ""
    chapters_path = os.path.join(base_dir, 'chapters')
    if not os.path.exists(chapters_path):
        raise FileNotFoundError(f"No chapters found in: {chapters_path}")

    folders = sorted([f for f in os.listdir(chapters_path) if os.path.isdir(os.path.join(chapters_path, f))])
    
    print(f"Combining {len(folders)} chapters...")
    for folder in folders:
        chapter_file = os.path.join(chapters_path, folder, 'base.md')
        if not os.path.exists(chapter_file):
            continue
        
        # Insert Part divider if we hit the boundary
        if folder == '00-prologue':
            combined_content += "\n\n\\part{The Quality Paradigm - Today and Tomorrow}\n"
        elif folder == '04-quality-concepts-strategy':
            combined_content += "\n\n\\part{Manual Mastery \\& Domain Expertise}\n"
        elif folder == '08-web-automation-frameworks':
            combined_content += "\n\n\\part{Test Automation \\& Performance}\n"
        elif folder == '13-ai-augmented-quality':
            combined_content += "\n\n\\part{The Future-State Quality Partner}\n"
        elif folder == '16-mock-interview-sets':
            combined_content += "\n\n\\part{Interview Mastery \\& Reference}\n"

        print(f"  Adding {folder}...")
        with open(chapter_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Strip LaTeX \newpage commands
        content = re.sub(r'\\newpage\r?\n?', '', content)
        # Strip stray \b lines (LaTeX accent command that crashes XeLaTeX)
        content = re.sub(r'\n\\b\n', '\n\n', content)
        
        # Replace --- horizontal rules with LaTeX \bigskip to avoid YAML parsing issues
        content = re.sub(r'\n---\n', lambda m: '\n\n\\bigskip\n\n', content)
        
        # Strip non-printable characters (except newline, tab, carriage return)
        content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\uFFFF]', '', content)
        
        # Sanitize Unicode characters that break XeLaTeX
        content = content.replace('\u00D7', '$\\times$')   # multiplication sign
        content = content.replace('\u2014', '---')          # em-dash
        content = content.replace('\u2013', '--')           # en-dash
        content = content.replace('\u2022', '-')            # bullet
        content = content.replace('\u201C', '"')            # left double quote
        content = content.replace('\u201D', '"')            # right double quote
        content = content.replace('\u2018', "'")            # left single quote
        content = content.replace('\u2019', "'")            # right single quote
        content = content.replace('\u2026', '...')           # ellipsis
        content = content.replace('\u00A0', ' ')            # non-breaking space
        content = content.replace('\u2192', '->')            # right arrow
        content = content.replace('\u2190', '<-')            # left arrow
        content = content.replace('\u2191', '^')             # up arrow
        content = content.replace('\u2193', 'v')             # down arrow
        content = content.replace('\u2194', '<->')           # left-right arrow
        content = content.replace('\u2B50', '[*]')           # star emoji
        content = content.replace('\u2705', '[v]')           # check mark
        content = content.replace('\u274C', '[x]')           # cross mark
        content = content.replace('\u26A0', '[!]')           # warning sign
        content = content.replace('\uFE0F', '')              # variation selector
        # Strip any remaining non-ASCII non-Latin characters that could break LaTeX
        content = re.sub(r'[\U00010000-\U0010FFFF]', '', content)  # strip supplementary planes
        # Replace triple-hyphens (em-dash markers) with en-dash to prevent XeLaTeX accent errors
        content = content.replace('---', '--')
        
        # Convert relative image paths to absolute or correct relative paths for building
        pattern = r'!\[(.*?)\]\((visuals/[^)]+)\)'
        def image_path_replacer(match):
            alt_text = match.group(1)
            img_rel_path = match.group(2)
            # Make path relative to book root
            resolved_img_path = os.path.join('chapters', folder, img_rel_path).replace('\\', '/')
            return f"![{alt_text}]({resolved_img_path})"
            
        content = re.sub(pattern, image_path_replacer, content)
        combined_content += f"\n\n{content}"
        
    return combined_content

def build_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    build_dir = os.path.join(base_dir, '_build')
    os.makedirs(build_dir, exist_ok=True)
    
    # 1. Combine chapters
    try:
        combined_md = combine_chapters(base_dir)
    except Exception as e:
        print(f"Error combining chapters: {e}")
        return

    manuscript_path = os.path.join(build_dir, 'manuscript.md')
    with open(manuscript_path, 'w', encoding='utf-8') as f:
        f.write(combined_md)
    print(f"Combined manuscript written to: {manuscript_path}")

    # 2. Run Pandoc to generate PDF
    pdf_output_name = "Spec_Driven_Quality_Engineering.pdf"
    pdf_output_path = os.path.join(base_dir, pdf_output_name)
    metadata_path = os.path.join(base_dir, 'metadata.yaml')
    
    print(f"Compiling PDF via Pandoc + XeLaTeX to: {pdf_output_name}...")
    try:
        # Build command
        cmd = [
            'pandoc',
            metadata_path,
            manuscript_path,
            '-o', pdf_output_path,
            '--pdf-engine=xelatex',
            '--toc',
            '--toc-depth=2',
            '--number-sections',
            '--variable', 'geometry:margin=1in'
        ]
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        print("PDF generated successfully!")
        
        # 3. Stamp page numbers if stamp_pages.py is available
        stamp_script = os.path.join(base_dir, 'stamp_pages.py')
        if os.path.exists(stamp_script):
            print("Stamping page numbers...")
            subprocess.run(['python', stamp_script, pdf_output_path, '6'], check=True)
            print("Page numbers stamped successfully.")
            
    except subprocess.CalledProcessError as e:
        print(f"Pandoc compilation failed: {e}")
        print(f"Error output:\n{e.stderr}")
    except Exception as e:
        print(f"Error building PDF: {e}")

if __name__ == '__main__':
    build_pdf()
