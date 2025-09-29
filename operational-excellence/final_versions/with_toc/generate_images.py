#!/usr/bin/env python3
"""
Script to generate PNG images from Mermaid diagrams
Requires: mermaid-cli (npm install -g @mermaid-js/mermaid-cli)
"""

import subprocess
import os
import sys

def generate_image(mmd_file, output_file):
    """Generate PNG image from Mermaid file using mermaid-cli"""
    try:
        # Use mermaid-cli to generate PNG
        cmd = [
            'mmdc',  # mermaid-cli command
            '-i', mmd_file,  # input file
            '-o', output_file,  # output file
            '-w', '1200',  # width
            '-H', '900',   # height
            '-b', 'white', # background color
            '-s', '2'      # scale factor for better quality
        ]
        
        print(f"Generating {output_file} from {mmd_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully generated {output_file}")
            return True
        else:
            print(f"❌ Error generating {output_file}: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ mermaid-cli not found. Please install it with: npm install -g @mermaid-js/mermaid-cli")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function to generate all missing images"""
    
    # Define the mappings of mermaid files to output images
    images_to_generate = [
        ('ai_enhanced_kpi_system.mmd', 'images/ai_enhanced_kpi_system.png'),
        ('human_ai_collaboration_diagram.mmd', 'images/human_ai_collaboration_diagram.png'),
        ('ai_ethics_framework.mmd', 'images/ai_ethics_framework.png')
    ]
    
    print("🎨 Generating missing images from Mermaid diagrams...")
    print("=" * 60)
    
    success_count = 0
    total_count = len(images_to_generate)
    
    for mmd_file, output_file in images_to_generate:
        if os.path.exists(mmd_file):
            if generate_image(mmd_file, output_file):
                success_count += 1
        else:
            print(f"❌ Mermaid file not found: {mmd_file}")
    
    print("=" * 60)
    print(f"📊 Results: {success_count}/{total_count} images generated successfully")
    
    if success_count == total_count:
        print("🎉 All images generated successfully!")
        print("\n📝 Next steps:")
        print("1. Verify the generated images in the 'images/' directory")
        print("2. Test PDF generation to ensure images load correctly")
        print("3. Check print quality of the final PDF")
    else:
        print("⚠️  Some images failed to generate. Please check the errors above.")
        print("\n💡 Alternative approach:")
        print("1. Use Mermaid Live Editor: https://mermaid.live/")
        print("2. Copy each .mmd file content")
        print("3. Paste into the editor and export as PNG")
        print("4. Save with the correct filename in the 'images/' directory")

if __name__ == "__main__":
    main()


