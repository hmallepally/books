#!/usr/bin/env python3
"""
Generate Mermaid diagrams as images for the HTML book
"""

import os
import subprocess
import sys

def generate_mermaid_image(mermaid_file, output_file):
    """Generate an image from a Mermaid file using mermaid-cli"""
    try:
        # Use mermaid-cli to generate the image
        cmd = [
            'mmdc',
            '-i', mermaid_file,
            '-o', output_file,
            '-w', '1200',
            '-H', '800',
            '-b', 'white',
            '-t', 'neutral'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Generated: {output_file}")
            return True
        else:
            print(f"❌ Error generating {output_file}: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ mermaid-cli (mmdc) not found. Please install it:")
        print("   npm install -g @mermaid-js/mermaid-cli")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Generate all Mermaid diagrams as images"""
    print("🎨 Generating Mermaid Diagrams as Images")
    print("=" * 50)
    
    # Ensure diagrams directory exists
    diagrams_dir = "diagrams"
    images_dir = "images"
    
    if not os.path.exists(diagrams_dir):
        print(f"❌ Diagrams directory not found: {diagrams_dir}")
        return
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"📁 Created images directory: {images_dir}")
    
    # List of diagrams to generate
    diagrams = [
        {
            'mermaid_file': f"{diagrams_dir}/six_sigma_dmaic_process.md",
            'output_file': f"{images_dir}/six_sigma_dmaic_process.png"
        },
        {
            'mermaid_file': f"{diagrams_dir}/kaizen_improvement_cycle.md",
            'output_file': f"{images_dir}/kaizen_improvement_cycle.png"
        }
    ]
    
    success_count = 0
    
    for diagram in diagrams:
        mermaid_file = diagram['mermaid_file']
        output_file = diagram['output_file']
        
        if os.path.exists(mermaid_file):
            print(f"🔄 Processing: {mermaid_file}")
            if generate_mermaid_image(mermaid_file, output_file):
                success_count += 1
        else:
            print(f"❌ Mermaid file not found: {mermaid_file}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {success_count}/{len(diagrams)} diagrams generated successfully")
    
    if success_count == len(diagrams):
        print("🎉 All diagrams generated successfully!")
        print("\n📋 Next steps:")
        print("1. Update HTML to use the generated images")
        print("2. Test the images in the browser")
    else:
        print("⚠️  Some diagrams failed to generate")
        print("💡 Make sure mermaid-cli is installed: npm install -g @mermaid-js/mermaid-cli")

if __name__ == "__main__":
    main()
