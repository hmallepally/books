import os
import re
import shutil

BASE_DIR = r"c:\Users\hari\Documents\DBA\books\spec_driven_interviews"
CHAPTERS_DIR = os.path.join(BASE_DIR, "chapters")

def renumber_chapter_folders():
    """
    Renames chapter folders so 16b -> 17, 17 -> 18, etc.
    """
    folder_map = [
        ("16b-system-design-solutions", "17-system-design-solutions"),
        ("17-resiliency", "18-resiliency"),
        ("18-database-compliance", "19-database-compliance"),
        ("19-behavioral-leadership", "20-behavioral-leadership"),
        ("20-testing-cicd", "21-testing-cicd"),
        ("21-message-brokers", "22-message-brokers"),
        ("22-aiml-llm", "23-aiml-llm"),
        ("23-appendix", "24-appendix"),
        ("24-references", "25-references"),
    ]
    # Execute in reverse order to prevent collision
    for old_name, new_name in reversed(folder_map):
        old_path = os.path.join(CHAPTERS_DIR, old_name)
        new_path = os.path.join(CHAPTERS_DIR, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Renamed folder: {old_name} -> {new_name}")

def clean_chapter_markdown(filepath):
    """
    Cleans markdown formatting:
    1. Removes horizontal lines '---'
    2. Strips leading numbers from heading titles (e.g., '#### 1. Title' -> '#### Title')
    """
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    modified = False

    for line in lines:
        stripped = line.strip()
        # 1. Remove standalone horizontal rules '---' or '***'
        if stripped in ['---', '***', '--------------------------------------------------']:
            modified = True
            continue

        # 2. Fix double numbering on headings (e.g. '#### 1. Title' or '### 2. Title')
        heading_match = re.match(r'^(#{1,6}\s+)\d+\.\s+(.*)$', line)
        if heading_match:
            hashes = heading_match.group(1)
            title = heading_match.group(2)
            cleaned_line = f"{hashes}{title}\n"
            cleaned_lines.append(cleaned_line)
            modified = True
            continue

        cleaned_lines.append(line)

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        print(f"Cleaned formatting in: {filepath}")

def main():
    print("=== STEP 1: Renumbering Chapter Folders ===")
    renumber_chapter_folders()

    print("\n=== STEP 2: Cleaning Book-Wide Formatting & Headings ===")
    for root, dirs, files in os.walk(CHAPTERS_DIR):
        for file in files:
            if file == "base.md":
                clean_chapter_markdown(os.path.join(root, file))

if __name__ == "__main__":
    main()
