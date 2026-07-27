import os
import re
import argparse
import subprocess
import shutil

def combine_chapters(edition_dir):
    """
    Combines all chapter.md files in the sorted directory structure of the given edition.
    """
    combined_content = ""
    chapters_path = os.path.join(edition_dir, 'chapters')
    if not os.path.exists(chapters_path):
        raise FileNotFoundError(f"No chapters found for edition in: {chapters_path}")

    folders = sorted([f for f in os.listdir(chapters_path) if os.path.isdir(os.path.join(chapters_path, f))])
    
    print(f"Combining {len(folders)} chapters...")
    for folder in folders:
        chapter_file = os.path.join(chapters_path, folder, 'chapter.md')
        if not os.path.exists(chapter_file):
            continue
        
        # Insert Part divider if we hit the boundary
        if folder == '00-prologue':
            combined_content += "\n\n\\part{The Spec-Driven Paradigm for Technical Interviews}\n"
        elif folder == '03-oop-principles':
            combined_content += "\n\n\\part{Code Design and Craftsmanship}\n"
        elif folder == '07-concurrency-performance':
            combined_content += "\n\n\\part{Code Performance and Data Structures}\n"
        elif folder == '14-system-architecture':
            combined_content += "\n\n\\part{System Design \\& Architecture at Scale}\n"

        print(f"  Adding {folder}...")
        with open(chapter_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Strip LaTeX \newpage commands
        content = re.sub(r'\\newpage\r?\n?', '', content)
        
        # Convert relative image paths to absolute or correct relative paths for building
        pattern = r'!\[(.*?)\]\((visuals/[^)]+)\)'
        def image_path_replacer(match):
            alt_text = match.group(1)
            img_rel_path = match.group(2)
            # Make path relative to book root
            resolved_img_path = os.path.join('editions', os.path.basename(edition_dir), 'chapters', folder, img_rel_path).replace('\\', '/')
            return f"![{alt_text}]({resolved_img_path})"
            
        content = re.sub(pattern, image_path_replacer, content)
        combined_content += f"\n\n{content}"
        
    return combined_content

def build_pdf(edition):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edition_dir = os.path.join(base_dir, 'editions', edition)
    
    if not os.path.exists(edition_dir):
        print(f"Edition '{edition}' not compiled. Run: python build_edition.py --lang {edition} first.")
        return

    build_dir = os.path.join(base_dir, '_build')
    os.makedirs(build_dir, exist_ok=True)
    
    # 1. Combine chapters
    try:
        combined_md = combine_chapters(edition_dir)
    except Exception as e:
        print(f"Error combining chapters: {e}")
        return

    manuscript_path = os.path.join(build_dir, f'manuscript_{edition}.md')
    with open(manuscript_path, 'w', encoding='utf-8') as f:
        f.write(combined_md)
    print(f"Combined manuscript written to: {manuscript_path}")

    # 2. Run Pandoc to generate PDF
    pdf_output_name = f"Spec_Driven_Coding_Interviews_{edition.capitalize()}_Edition.pdf"
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
    parser = argparse.ArgumentParser(description="PDF builder for Spec-Driven Coding Interviews")
    parser.add_argument('--edition', default='java', choices=['java', 'python', 'csharp'], help="Language edition to build (default: java)")
    args = parser.parse_args()
    build_pdf(args.edition)
