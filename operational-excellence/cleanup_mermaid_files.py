#!/usr/bin/env python3
"""
Clean up temporary Mermaid files and keep only the final PNG images
"""

import os

def main():
    """Clean up temporary files"""
    print("🧹 Cleaning up temporary Mermaid files")
    print("=" * 40)
    
    # Files to remove
    files_to_remove = [
        "images/six_sigma_dmaic_process-1.png",
        "images/kaizen_improvement_cycle-1.png",
        "images/six_sigma_dmaic_process.svg",
        "images/kaizen_improvement_cycle.svg",
        "diagrams/six_sigma_dmaic_process.md",
        "diagrams/kaizen_improvement_cycle.md"
    ]
    
    removed_count = 0
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
        else:
            print(f"ℹ️  Not found: {file_path}")
    
    print(f"\n📊 Cleanup complete: {removed_count} files removed")
    
    # List remaining files
    print("\n📁 Remaining image files:")
    images_dir = "images"
    if os.path.exists(images_dir):
        for file in sorted(os.listdir(images_dir)):
            if file.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                print(f"  📄 {file}")

if __name__ == "__main__":
    main()
