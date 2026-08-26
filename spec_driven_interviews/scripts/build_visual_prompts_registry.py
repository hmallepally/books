import os
import glob
import re
import json

def build_prompts_registry():
    chapter_files = sorted(glob.glob('chapters/*/base.md'))
    visuals = []
    seen_ids = set()

    for ch_path in chapter_files:
        ch_folder = os.path.basename(os.path.dirname(ch_path))
        match_num = re.match(r'^(\d+)', ch_folder)
        ch_num = int(match_num.group(1)) if match_num else 0

        with open(ch_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_section = f"{ch_num}.0"
        for line in lines:
            line_str = line.strip()
            sec_match = re.match(r'^(#{2,4})\s+([\d\.]+)?\s*(.*)', line_str)
            if sec_match:
                sec_num = sec_match.group(2)
                sec_title = sec_match.group(3)
                if sec_num:
                    current_section = sec_num
                else:
                    current_section = f"{ch_num}.x"

            img_match = re.search(r'!\[([^\]]*)\]\(visuals/([^\)]+)\)', line_str)
            if img_match:
                caption = img_match.group(1).strip()
                filename_full = img_match.group(2).strip()
                filename = filename_full.split('{')[0].strip()
                img_id = os.path.splitext(filename)[0]

                if img_id in seen_ids:
                    continue
                seen_ids.add(img_id)

                ext = os.path.splitext(filename)[1].replace('.', '')
                
                # Detailed prompt descriptions based on caption and context
                prompt_text = (
                    f"A clean, high-resolution, professional technical software architecture diagram "
                    f"for '{caption if caption else img_id}'. Minimalist corporate blueprint style on a "
                    f"light grid background with vibrant teal and deep navy accent colors. Crisp typography, "
                    f"clear directional arrows, and structured modular components."
                )

                visuals.append({
                    "id": img_id,
                    "chapter": ch_num,
                    "section": current_section,
                    "title": caption if caption else img_id.replace('_', ' ').title(),
                    "filename": filename,
                    "format": ext,
                    "aspect_ratio": "16:9",
                    "prompt": prompt_text
                })

    registry = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Spec-Driven Coding Interviews - Complete Visual Assets & Prompt Registry",
        "description": "Comprehensive registry of all visual diagrams, prompt specifications, and rendering parameters across all 25 chapters.",
        "total_visuals": len(visuals),
        "visuals": visuals
    }

    out_path = os.path.join('visuals', 'PROMPTS.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)

    print(f"Successfully generated visual prompt registry at '{out_path}' with {len(visuals)} entries.")

if __name__ == '__main__':
    build_prompts_registry()
