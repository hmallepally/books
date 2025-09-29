#!/usr/bin/env python3
"""
Create colorful 32x32 PNG images to replace emojis in the Living Strategy book.
These images will be larger and more visible for better readability.
"""

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Please install Pillow: pip install Pillow")

import os

def create_large_colorful_images():
    """Create colorful 32x32 PNG images"""
    if not PIL_AVAILABLE:
        print("Cannot create images without PIL. Please install Pillow.")
        return
    
    os.makedirs('images', exist_ok=True)
    
    # Create each colorful image
    images = {
        'rocket.png': create_large_colorful_rocket(),
        'lightbulb.png': create_large_colorful_lightbulb(),
        'tools.png': create_large_colorful_tools(),
        'book.png': create_large_colorful_book(),
        'chart.png': create_large_colorful_chart()
    }
    
    # Save images
    for filename, img in images.items():
        img.save(f'images/{filename}')
        print(f"Created 32x32 colorful images/{filename}")
    
    print("\nAll 32x32 colorful images created successfully!")

def create_large_colorful_rocket():
    """Create a large colorful rocket icon"""
    img = Image.new('RGBA', (32, 32), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Rocket body (blue gradient effect) - larger
    draw.rectangle([12, 4, 20, 24], fill=(30, 144, 255))  # Dodger blue
    
    # Rocket nose (red) - larger
    draw.polygon([(16, 0), (8, 6), (24, 6)], fill=(220, 20, 60))  # Crimson
    
    # Rocket fins (dark blue) - larger
    draw.polygon([(12, 24), (6, 32), (12, 28)], fill=(25, 25, 112))  # Midnight blue
    draw.polygon([(20, 24), (26, 32), (20, 28)], fill=(25, 25, 112))  # Midnight blue
    
    # Rocket flame (orange/yellow) - larger
    draw.line([(14, 28), (14, 32)], fill=(255, 165, 0), width=4)  # Orange
    draw.line([(18, 28), (18, 32)], fill=(255, 215, 0), width=4)  # Gold
    
    # Add some highlights - larger
    draw.line([(14, 6), (14, 22)], fill=(135, 206, 250), width=2)  # Light sky blue
    draw.line([(18, 6), (18, 22)], fill=(135, 206, 250), width=2)  # Light sky blue
    
    # Add window (white circle)
    draw.ellipse([(14, 8), (18, 12)], fill=(255, 255, 255))  # White
    
    return img

def create_large_colorful_lightbulb():
    """Create a large colorful lightbulb icon"""
    img = Image.new('RGBA', (32, 32), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Lightbulb body (bright yellow) - larger
    draw.ellipse([(8, 4), (24, 20)], fill=(255, 255, 0))  # Yellow
    
    # Lightbulb base (dark gray) - larger
    draw.rectangle([(10, 20), (22, 24)], fill=(105, 105, 105))  # Dim gray
    
    # Lightbulb screw base (darker gray) - larger
    draw.rectangle([(12, 24), (20, 28)], fill=(64, 64, 64))  # Dark gray
    
    # Light rays (bright yellow) - larger
    draw.line([(4, 8), (8, 8)], fill=(255, 255, 0), width=4)  # Yellow
    draw.line([(24, 8), (28, 8)], fill=(255, 255, 0), width=4)  # Yellow
    draw.line([(4, 12), (8, 12)], fill=(255, 255, 0), width=4)  # Yellow
    draw.line([(24, 12), (28, 12)], fill=(255, 255, 0), width=4)  # Yellow
    draw.line([(4, 16), (8, 16)], fill=(255, 255, 0), width=4)  # Yellow
    draw.line([(24, 16), (28, 16)], fill=(255, 255, 0), width=4)  # Yellow
    
    # Add filament (orange) - larger
    draw.line([(14, 10), (18, 10)], fill=(255, 140, 0), width=2)  # Dark orange
    draw.line([(14, 14), (18, 14)], fill=(255, 140, 0), width=2)  # Dark orange
    
    return img

def create_large_colorful_tools():
    """Create a large colorful tools icon"""
    img = Image.new('RGBA', (32, 32), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Hammer head (brown) - larger
    draw.rectangle([(4, 4), (12, 8)], fill=(139, 69, 19))  # Saddle brown
    
    # Hammer handle (dark brown) - larger
    draw.line([(8, 8), (8, 24)], fill=(101, 67, 33), width=6)  # Dark brown
    
    # Wrench (silver/gray) - larger
    draw.line([(20, 4), (28, 4)], fill=(192, 192, 192), width=6)  # Silver
    draw.line([(28, 4), (28, 12)], fill=(192, 192, 192), width=6)  # Silver
    draw.line([(20, 12), (28, 12)], fill=(192, 192, 192), width=6)  # Silver
    
    # Add some highlights - larger
    draw.line([(6, 6), (10, 6)], fill=(160, 82, 45), width=2)  # Sienna
    
    # Add grip texture to handle
    draw.line([(6, 16), (10, 16)], fill=(139, 69, 19), width=2)  # Saddle brown
    draw.line([(6, 20), (10, 20)], fill=(139, 69, 19), width=2)  # Saddle brown
    
    return img

def create_large_colorful_book():
    """Create a large colorful book icon"""
    img = Image.new('RGBA', (32, 32), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Book cover (green) - larger
    draw.rectangle([(6, 4), (26, 28)], fill=(34, 139, 34))  # Forest green
    
    # Book pages (white) - larger
    draw.rectangle([(8, 6), (24, 26)], fill=(255, 255, 255))  # White
    
    # Book spine (dark green) - larger
    draw.line([(6, 4), (6, 28)], fill=(0, 100, 0), width=4)  # Dark green
    
    # Book text lines (dark green) - larger
    draw.line([(10, 10), (22, 10)], fill=(0, 100, 0), width=2)  # Dark green
    draw.line([(10, 14), (22, 14)], fill=(0, 100, 0), width=2)  # Dark green
    draw.line([(10, 18), (22, 18)], fill=(0, 100, 0), width=2)  # Dark green
    draw.line([(10, 22), (22, 22)], fill=(0, 100, 0), width=2)  # Dark green
    
    # Add bookmark (red) - larger
    draw.line([(26, 4), (26, 16)], fill=(220, 20, 60), width=4)  # Crimson
    
    # Add title highlight
    draw.line([(10, 8), (22, 8)], fill=(0, 100, 0), width=3)  # Dark green
    
    return img

def create_large_colorful_chart():
    """Create a large colorful chart icon"""
    img = Image.new('RGBA', (32, 32), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Chart axes (dark gray) - larger
    draw.line([(4, 28), (28, 28)], fill=(64, 64, 64), width=4)  # Dark gray
    draw.line([(4, 4), (4, 28)], fill=(64, 64, 64), width=4)  # Dark gray
    
    # Chart line (green ascending) - larger
    draw.line([(6, 24), (12, 18), (18, 12), (24, 6)], fill=(0, 255, 0), width=6)  # Lime green
    
    # Data points (different colors) - larger
    draw.ellipse([(4, 22), (8, 26)], fill=(255, 0, 0))  # Red
    draw.ellipse([(10, 16), (14, 20)], fill=(255, 165, 0))  # Orange
    draw.ellipse([(16, 10), (20, 14)], fill=(255, 255, 0))  # Yellow
    draw.ellipse([(22, 4), (26, 8)], fill=(0, 255, 0))  # Green
    
    # Add grid lines (light gray) - larger
    draw.line([(10, 4), (10, 28)], fill=(200, 200, 200), width=2)  # Light gray
    draw.line([(16, 4), (16, 28)], fill=(200, 200, 200), width=2)  # Light gray
    draw.line([(22, 4), (22, 28)], fill=(200, 200, 200), width=2)  # Light gray
    draw.line([(4, 10), (28, 10)], fill=(200, 200, 200), width=2)  # Light gray
    draw.line([(4, 16), (28, 16)], fill=(200, 200, 200), width=2)  # Light gray
    draw.line([(4, 22), (28, 22)], fill=(200, 200, 200), width=2)  # Light gray
    
    return img

if __name__ == "__main__":
    create_large_colorful_images()

