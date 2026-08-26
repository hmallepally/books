import os
import re
import argparse
import shutil

REPLACEMENT_MAPS = {
    'java': {
        'the test framework': 'JUnit 5',
        'the package manager': 'Maven',
        'the web framework': 'Spring Boot 3.x',
        'the ORM': 'Hibernate',
        'the CLI tool': 'sdsd-cli',
        'the programming language': 'Java 21'
    },
    'python': {
        'the test framework': 'pytest',
        'the package manager': 'pip',
        'the web framework': 'FastAPI',
        'the ORM': 'SQLAlchemy',
        'the CLI tool': 'sdsd-cli',
        'the programming language': 'Python 3.11'
    },
    'csharp': {
        'the test framework': 'xUnit',
        'the package manager': 'NuGet',
        'the web framework': 'ASP.NET Core',
        'the ORM': 'Entity Framework Core',
        'the CLI tool': 'sdsd-cli',
        'the programming language': 'C# .NET 8'
    }
}

def inject_snippets(base_content, snippets_dir, lang):
    """
    Find and replace all {{ inject('filename') }} instances with snippet file contents.
    """
    pattern = r"\{\{\s*inject\(['\"](.+?)['\"]\)\s*\}\}"
    
    def replacer(match):
        snippet_file = match.group(1)
        snippet_path = os.path.join(snippets_dir, lang, snippet_file)
        if not os.path.exists(snippet_path):
            raise FileNotFoundError(f"Snippet not found: {snippet_path}")
        with open(snippet_path, 'r', encoding='utf-8') as sf:
            content = sf.read()
        if not content.endswith('\n'):
            content += '\n'
        return content

    return re.sub(pattern, replacer, base_content)

def apply_replacements(content, lang):
    """
    Apply language-specific text replacements.
    """
    replacement_map = REPLACEMENT_MAPS.get(lang, {})
    for placeholder, value in replacement_map.items():
        # Match case-insensitive placeholder but replace exactly
        content = re.sub(re.escape(placeholder), value, content, flags=re.IGNORECASE)
    return content

def build_edition(lang):
    print(f"Building edition: {lang}...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chapters_dir = os.path.join(base_dir, 'chapters')
    editions_dir = os.path.join(base_dir, 'editions', lang, 'chapters')

    if not os.path.exists(chapters_dir):
        print("Chapters directory not found. Please scaffold chapters first.")
        return

    # Clean and recreate editions directory
    if os.path.exists(os.path.join(base_dir, 'editions', lang)):
        shutil.rmtree(os.path.join(base_dir, 'editions', lang), ignore_errors=True)
    os.makedirs(editions_dir, exist_ok=True)

    chapter_folders = sorted([f for f in os.listdir(chapters_dir) if os.path.isdir(os.path.join(chapters_dir, f))])

    for folder in chapter_folders:
        chapter_path = os.path.join(chapters_dir, folder)
        base_md_path = os.path.join(chapter_path, 'base.md')
        
        if not os.path.exists(base_md_path):
            continue

        print(f"  Processing {folder}...")
        with open(base_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Inject language-specific code snippets
        snippets_dir = os.path.join(chapter_path, 'snippets')
        if os.path.exists(snippets_dir):
            try:
                content = inject_snippets(content, snippets_dir, lang)
            except FileNotFoundError as e:
                print(f"  Warning: {e}")

        # 2. Apply text replacements
        content = apply_replacements(content, lang)

        # 3. Write resolved chapter file to editions/
        dest_folder = os.path.join(editions_dir, folder)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Copy visuals if they exist in chapter directory
        chapter_visuals_src = os.path.join(chapter_path, 'visuals')
        chapter_visuals_dst = os.path.join(dest_folder, 'visuals')
        if os.path.exists(chapter_visuals_src):
            shutil.copytree(chapter_visuals_src, chapter_visuals_dst, dirs_exist_ok=True)

        with open(os.path.join(dest_folder, 'chapter.md'), 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"Edition '{lang}' compiled successfully to: editions/{lang}/chapters/\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-language technical book builder")
    parser.add_argument('--lang', required=True, choices=['java', 'python', 'csharp'], help="Target programming language")
    args = parser.parse_args()
    build_edition(args.lang)
