#!/usr/bin/env python3
"""
Create conceptual images for each chapter's "What Changed" section.
These will be larger, more detailed images that illustrate the key transformation in each chapter.
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Please install Pillow: pip install Pillow")

import os

def create_chapter_images():
    """Create conceptual images for chapter transformations"""
    if not PIL_AVAILABLE:
        print("Cannot create images without PIL. Please install Pillow.")
        return
    
    os.makedirs('images', exist_ok=True)
    
    # Create conceptual images for first 10 chapters
    chapter_images = {
        'chapter1_what_changed.png': create_linear_to_adaptive_image(),
        'chapter2_what_changed.png': create_founder_to_distributed_image(),
        'chapter3_what_changed.png': create_speed_to_velocity_image(),
        'chapter4_what_changed.png': create_silos_to_transparency_image(),
        'chapter5_what_changed.png': create_annual_to_realtime_image(),
        'chapter6_what_changed.png': create_campaigns_to_connection_image(),
        'chapter7_what_changed.png': create_ipo_to_autonomy_image(),
        'chapter8_what_changed.png': create_financial_to_strategic_image(),
        'chapter9_what_changed.png': create_slogans_to_compass_image(),
        'chapter10_what_changed.png': create_static_to_living_image()
    }
    
    # Save images
    for filename, img in chapter_images.items():
        img.save(f'images/{filename}')
        print(f"Created conceptual image images/{filename}")
    
    print("\nAll chapter conceptual images created successfully!")

def create_linear_to_adaptive_image():
    """Chapter 1: From Linear Plans to Perpetual Flux"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 1: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Linear Plans to Perpetual Flux", fill=(100, 100, 100), font_size=14)
    
    # Left side - Linear Plan
    draw.text((50, 80), "OLD: Linear Planning", fill=(200, 0, 0), font_size=12)
    draw.line([(50, 100), (150, 100)], fill=(200, 0, 0), width=3)  # Straight line
    draw.text((50, 110), "Year 1", fill=(0, 0, 0), font_size=10)
    draw.text((100, 110), "Year 2", fill=(0, 0, 0), font_size=10)
    draw.text((150, 110), "Year 3", fill=(0, 0, 0), font_size=10)
    
    # Arrow
    draw.polygon([(200, 90), (220, 100), (200, 110)], fill=(0, 0, 0))
    
    # Right side - Adaptive Planning
    draw.text((250, 80), "NEW: Adaptive Planning", fill=(0, 150, 0), font_size=12)
    # Curved, branching path
    draw.line([(250, 100), (280, 90), (310, 100), (340, 85), (370, 100)], fill=(0, 150, 0), width=3)
    draw.line([(310, 100), (320, 120)], fill=(0, 150, 0), width=2)  # Branch
    draw.line([(340, 85), (350, 105)], fill=(0, 150, 0), width=2)  # Branch
    
    # Feedback loops
    draw.ellipse([(280, 70), (300, 90)], outline=(0, 150, 0), width=2)
    draw.text((285, 75), "FB", fill=(0, 150, 0), font_size=8)
    
    # Time compression indicator
    draw.text((50, 150), "Market Cycles:", fill=(0, 0, 0), font_size=12)
    draw.text((50, 170), "Years → Months", fill=(255, 0, 0), font_size=14)
    
    return img

def create_founder_to_distributed_image():
    """Chapter 2: From Founder-Focused Decisions to Distributed Complexity"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 2: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Founder-Focused to Distributed Decisions", fill=(100, 100, 100), font_size=14)
    
    # Left side - Founder-focused
    draw.text((50, 80), "OLD: Founder-Centric", fill=(200, 0, 0), font_size=12)
    # Central founder
    draw.ellipse([(120, 100), (140, 120)], fill=(200, 0, 0))
    draw.text((125, 105), "F", fill=(255, 255, 255), font_size=12)
    
    # Lines to team members
    draw.line([(130, 120), (100, 150)], fill=(200, 0, 0), width=2)
    draw.line([(130, 120), (160, 150)], fill=(200, 0, 0), width=2)
    draw.line([(130, 120), (130, 180)], fill=(200, 0, 0), width=2)
    
    # Team members
    draw.ellipse([(90, 150), (110, 170)], fill=(150, 150, 150))
    draw.ellipse([(150, 150), (170, 170)], fill=(150, 150, 150))
    draw.ellipse([(120, 180), (140, 200)], fill=(150, 150, 150))
    
    # Arrow
    draw.polygon([(200, 130), (220, 140), (200, 150)], fill=(0, 0, 0))
    
    # Right side - Distributed
    draw.text((250, 80), "NEW: Distributed Network", fill=(0, 150, 0), font_size=12)
    
    # Network of decision makers
    positions = [(280, 100), (320, 100), (300, 130), (280, 160), (320, 160)]
    for pos in positions:
        draw.ellipse([(pos[0]-10, pos[1]-10), (pos[0]+10, pos[1]+10)], fill=(0, 150, 0))
    
    # Connect all nodes
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            draw.line([positions[i], positions[j]], fill=(0, 150, 0), width=1)
    
    return img

def create_speed_to_velocity_image():
    """Chapter 3: From Speed-at-Any-Cost to Sustainable Acceleration"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 3: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Speed-at-Any-Cost to Sustainable Velocity", fill=(100, 100, 100), font_size=14)
    
    # Left side - Chaotic Speed
    draw.text((50, 80), "OLD: Breakneck Speed", fill=(200, 0, 0), font_size=12)
    # Chaotic, broken arrows
    draw.line([(50, 100), (80, 90)], fill=(200, 0, 0), width=3)
    draw.line([(80, 90), (110, 110)], fill=(200, 0, 0), width=3)
    draw.line([(110, 110), (140, 95)], fill=(200, 0, 0), width=3)
    # Broken parts
    draw.line([(80, 90), (85, 95)], fill=(200, 0, 0), width=2)
    draw.line([(110, 110), (115, 105)], fill=(200, 0, 0), width=2)
    
    # Warning signs
    draw.polygon([(60, 120), (70, 120), (65, 130)], fill=(255, 0, 0))
    draw.polygon([(100, 120), (110, 120), (105, 130)], fill=(255, 0, 0))
    
    # Arrow
    draw.polygon([(200, 100), (220, 110), (200, 120)], fill=(0, 0, 0))
    
    # Right side - Sustainable Velocity
    draw.text((250, 80), "NEW: Sustainable Velocity", fill=(0, 150, 0), font_size=12)
    # Smooth, rhythmic pattern
    draw.line([(250, 100), (280, 100), (310, 100), (340, 100), (370, 100)], fill=(0, 150, 0), width=3)
    # Rhythm indicators
    draw.ellipse([(265, 95), (275, 105)], fill=(0, 150, 0))
    draw.ellipse([(295, 95), (305, 105)], fill=(0, 150, 0))
    draw.ellipse([(325, 95), (335, 105)], fill=(0, 150, 0))
    draw.ellipse([(355, 95), (365, 105)], fill=(0, 150, 0))
    
    # Guardrails
    draw.line([(250, 120), (370, 120)], fill=(0, 0, 255), width=2)  # Bottom guardrail
    draw.line([(250, 80), (370, 80)], fill=(0, 0, 255), width=2)   # Top guardrail
    
    return img

def create_silos_to_transparency_image():
    """Chapter 4: From Information Silos to Strategic Visibility"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 4: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Information Silos to Strategic Visibility", fill=(100, 100, 100), font_size=14)
    
    # Left side - Silos
    draw.text((50, 80), "OLD: Information Silos", fill=(200, 0, 0), font_size=12)
    
    # Three separate silos with barriers
    draw.rectangle([(50, 100), (90, 180)], fill=(200, 200, 200), outline=(200, 0, 0), width=2)
    draw.text((60, 140), "Silo 1", fill=(0, 0, 0), font_size=10)
    
    draw.rectangle([(110, 100), (150, 180)], fill=(200, 200, 200), outline=(200, 0, 0), width=2)
    draw.text((120, 140), "Silo 2", fill=(0, 0, 0), font_size=10)
    
    draw.rectangle([(170, 100), (210, 180)], fill=(200, 200, 200), outline=(200, 0, 0), width=2)
    draw.text((180, 140), "Silo 3", fill=(0, 0, 0), font_size=10)
    
    # Barriers between silos
    draw.line([(90, 100), (90, 180)], fill=(200, 0, 0), width=3)
    draw.line([(150, 100), (150, 180)], fill=(200, 0, 0), width=3)
    
    # Arrow
    draw.polygon([(230, 130), (250, 140), (230, 150)], fill=(0, 0, 0))
    
    # Right side - Transparent Network
    draw.text((270, 80), "NEW: Transparent Network", fill=(0, 150, 0), font_size=12)
    
    # Connected, transparent boxes
    draw.rectangle([(270, 100), (310, 180)], fill=(200, 255, 200), outline=(0, 150, 0), width=2)
    draw.text((280, 140), "Data", fill=(0, 0, 0), font_size=10)
    
    draw.rectangle([(330, 100), (370, 180)], fill=(200, 255, 200), outline=(0, 150, 0), width=2)
    draw.text((340, 140), "Decisions", fill=(0, 0, 0), font_size=10)
    
    # Connection lines
    draw.line([(310, 140), (330, 140)], fill=(0, 150, 0), width=3)
    
    # Visibility indicators (light rays)
    draw.line([(290, 90), (290, 100)], fill=(255, 255, 0), width=2)
    draw.line([(350, 90), (350, 100)], fill=(255, 255, 0), width=2)
    
    return img

def create_annual_to_realtime_image():
    """Chapter 5: From Annual Reports to Real-Time Learning"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 5: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Annual Reports to Real-Time Learning", fill=(100, 100, 100), font_size=14)
    
    # Left side - Annual Reports
    draw.text((50, 80), "OLD: Annual Reports", fill=(200, 0, 0), font_size=12)
    
    # Calendar showing annual cycle
    draw.rectangle([(50, 100), (150, 180)], fill=(200, 200, 200), outline=(200, 0, 0), width=2)
    draw.text((70, 110), "2023 Report", fill=(0, 0, 0), font_size=10)
    draw.text((70, 130), "2024 Report", fill=(0, 0, 0), font_size=10)
    draw.text((70, 150), "2025 Report", fill=(0, 0, 0), font_size=10)
    
    # Clock showing slow cycle
    draw.ellipse([(160, 120), (180, 140)], fill=(200, 0, 0))
    draw.line([(170, 130), (170, 125)], fill=(255, 255, 255), width=2)  # Hour hand
    draw.line([(170, 130), (175, 130)], fill=(255, 255, 255), width=1)  # Minute hand
    
    # Arrow
    draw.polygon([(200, 130), (220, 140), (200, 150)], fill=(0, 0, 0))
    
    # Right side - Real-Time Learning
    draw.text((250, 80), "NEW: Real-Time Learning", fill=(0, 150, 0), font_size=12)
    
    # Continuous data stream
    draw.line([(250, 120), (280, 110), (310, 120), (340, 115), (370, 120)], fill=(0, 150, 0), width=3)
    
    # Real-time indicators
    draw.ellipse([(260, 100), (270, 110)], fill=(0, 255, 0))
    draw.ellipse([(290, 100), (300, 110)], fill=(0, 255, 0))
    draw.ellipse([(320, 100), (330, 110)], fill=(0, 255, 0))
    draw.ellipse([(350, 100), (360, 110)], fill=(0, 255, 0))
    
    # Learning loops
    draw.ellipse([(280, 130), (300, 150)], outline=(0, 150, 0), width=2)
    draw.text((285, 135), "Learn", fill=(0, 150, 0), font_size=8)
    
    return img

def create_campaigns_to_connection_image():
    """Chapter 6: From Campaigns to Continuous Connection"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 6: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Campaigns to Continuous Connection", fill=(100, 100, 100), font_size=14)
    
    # Left side - Campaigns
    draw.text((50, 80), "OLD: Campaign-Based", fill=(200, 0, 0), font_size=12)
    
    # Sporadic campaign spikes
    draw.line([(50, 150), (60, 120)], fill=(200, 0, 0), width=3)  # Campaign spike
    draw.line([(60, 120), (70, 150)], fill=(200, 0, 0), width=3)
    
    draw.line([(80, 150), (90, 110)], fill=(200, 0, 0), width=3)  # Campaign spike
    draw.line([(90, 110), (100, 150)], fill=(200, 0, 0), width=3)
    
    draw.line([(110, 150), (120, 130)], fill=(200, 0, 0), width=3)  # Campaign spike
    draw.line([(120, 130), (130, 150)], fill=(200, 0, 0), width=3)
    
    # Baseline
    draw.line([(50, 150), (130, 150)], fill=(100, 100, 100), width=2)
    
    # Arrow
    draw.polygon([(160, 130), (180, 140), (160, 150)], fill=(0, 0, 0))
    
    # Right side - Continuous Connection
    draw.text((200, 80), "NEW: Continuous Connection", fill=(0, 150, 0), font_size=12)
    
    # Steady engagement line
    draw.line([(200, 130), (380, 130)], fill=(0, 150, 0), width=4)
    
    # Connection points
    draw.ellipse([(220, 125), (230, 135)], fill=(0, 150, 0))
    draw.ellipse([(250, 125), (260, 135)], fill=(0, 150, 0))
    draw.ellipse([(280, 125), (290, 135)], fill=(0, 150, 0))
    draw.ellipse([(310, 125), (320, 135)], fill=(0, 150, 0))
    draw.ellipse([(340, 125), (350, 135)], fill=(0, 150, 0))
    draw.ellipse([(370, 125), (380, 135)], fill=(0, 150, 0))
    
    # Heart symbol for connection
    draw.text((300, 100), "♥", fill=(255, 0, 0), font_size=20)
    
    return img

def create_ipo_to_autonomy_image():
    """Chapter 7: From IPO Pressure to Strategic Autonomy"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 7: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From IPO Pressure to Strategic Autonomy", fill=(100, 100, 100), font_size=14)
    
    # Left side - IPO Pressure
    draw.text((50, 80), "OLD: IPO Pressure", fill=(200, 0, 0), font_size=12)
    
    # Company under pressure
    draw.rectangle([(70, 100), (130, 160)], fill=(200, 200, 200), outline=(200, 0, 0), width=2)
    draw.text((85, 125), "Company", fill=(0, 0, 0), font_size=10)
    
    # Pressure arrows from all sides
    draw.polygon([(50, 120), (70, 115), (70, 125)], fill=(200, 0, 0))  # Left pressure
    draw.polygon([(130, 115), (150, 120), (130, 125)], fill=(200, 0, 0))  # Right pressure
    draw.polygon([(100, 80), (105, 100), (95, 100)], fill=(200, 0, 0))  # Top pressure
    draw.polygon([(100, 160), (95, 180), (105, 180)], fill=(200, 0, 0))  # Bottom pressure
    
    # Dollar signs for financial pressure
    draw.text((40, 120), "$", fill=(200, 0, 0), font_size=16)
    draw.text((150, 120), "$", fill=(200, 0, 0), font_size=16)
    
    # Arrow
    draw.polygon([(180, 130), (200, 140), (180, 150)], fill=(0, 0, 0))
    
    # Right side - Strategic Autonomy
    draw.text((220, 80), "NEW: Strategic Autonomy", fill=(0, 150, 0), font_size=12)
    
    # Independent company
    draw.rectangle([(240, 100), (300, 160)], fill=(200, 255, 200), outline=(0, 150, 0), width=2)
    draw.text((255, 125), "Company", fill=(0, 0, 0), font_size=10)
    
    # Shield around company
    draw.ellipse([(230, 90), (310, 170)], outline=(0, 150, 0), width=3)
    
    # Independence indicators
    draw.text((320, 120), "✓", fill=(0, 150, 0), font_size=16)
    draw.text((320, 140), "✓", fill=(0, 150, 0), font_size=16)
    
    return img

def create_financial_to_strategic_image():
    """Chapter 8: From Financial Updates to Strategic Alignment"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 8: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Financial Updates to Strategic Alignment", fill=(100, 100, 100), font_size=14)
    
    # Left side - Financial Updates
    draw.text((50, 80), "OLD: Financial Updates", fill=(200, 0, 0), font_size=12)
    
    # One-way communication
    draw.rectangle([(70, 100), (130, 160)], fill=(200, 200, 200), outline=(200, 0, 0), width=2)
    draw.text((85, 125), "Company", fill=(0, 0, 0), font_size=10)
    
    # Arrow pointing to investors
    draw.line([(130, 130), (180, 130)], fill=(200, 0, 0), width=3)
    draw.polygon([(180, 125), (190, 130), (180, 135)], fill=(200, 0, 0))
    
    # Investors
    draw.ellipse([(190, 120), (210, 140)], fill=(150, 150, 150))
    draw.text((195, 125), "I", fill=(255, 255, 255), font_size=10)
    
    # Dollar signs
    draw.text((200, 100), "$", fill=(200, 0, 0), font_size=20)
    draw.text((200, 150), "$", fill=(200, 0, 0), font_size=20)
    
    # Arrow
    draw.polygon([(230, 130), (250, 140), (230, 150)], fill=(0, 0, 0))
    
    # Right side - Strategic Alignment
    draw.text((270, 80), "NEW: Strategic Alignment", fill=(0, 150, 0), font_size=12)
    
    # Two-way communication
    draw.rectangle([(290, 100), (350, 160)], fill=(200, 255, 200), outline=(0, 150, 0), width=2)
    draw.text((305, 125), "Company", fill=(0, 0, 0), font_size=10)
    
    # Bidirectional arrows
    draw.line([(350, 130), (380, 130)], fill=(0, 150, 0), width=3)
    draw.line([(380, 130), (350, 130)], fill=(0, 150, 0), width=3)
    
    # Strategic partners
    draw.ellipse([(380, 120), (400, 140)], fill=(150, 255, 150))
    draw.text((385, 125), "S", fill=(0, 0, 0), font_size=10)
    
    # Alignment indicators
    draw.text((320, 80), "↕", fill=(0, 150, 0), font_size=16)
    draw.text((320, 170), "↕", fill=(0, 150, 0), font_size=16)
    
    return img

def create_slogans_to_compass_image():
    """Chapter 9: From Slogans to Strategic Guidance"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 9: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Slogans to Strategic Guidance", fill=(100, 100, 100), font_size=14)
    
    # Left side - Slogans
    draw.text((50, 80), "OLD: Static Slogans", fill=(200, 0, 0), font_size=12)
    
    # Words on wall
    draw.rectangle([(50, 100), (150, 180)], fill=(200, 200, 200), outline=(200, 0, 0), width=2)
    draw.text((60, 110), "Innovation", fill=(0, 0, 0), font_size=10)
    draw.text((60, 130), "Excellence", fill=(0, 0, 0), font_size=10)
    draw.text((60, 150), "Quality", fill=(0, 0, 0), font_size=10)
    
    # Static indicator
    draw.text((160, 140), "📌", fill=(200, 0, 0), font_size=20)
    
    # Arrow
    draw.polygon([(180, 130), (200, 140), (180, 150)], fill=(0, 0, 0))
    
    # Right side - Strategic Compass
    draw.text((220, 80), "NEW: Strategic Compass", fill=(0, 150, 0), font_size=12)
    
    # Compass
    draw.ellipse([(250, 100), (350, 200)], outline=(0, 150, 0), width=3)
    
    # Compass directions
    draw.text((320, 110), "N", fill=(0, 150, 0), font_size=16)  # North
    draw.text((320, 190), "S", fill=(0, 150, 0), font_size=16)  # South
    draw.text((250, 150), "W", fill=(0, 150, 0), font_size=16)  # West
    draw.text((350, 150), "E", fill=(0, 150, 0), font_size=16)  # East
    
    # Compass needle
    draw.line([(300, 150), (300, 120)], fill=(255, 0, 0), width=3)  # North needle
    draw.polygon([(300, 120), (295, 125), (305, 125)], fill=(255, 0, 0))
    
    # Decision paths
    draw.line([(300, 150), (280, 130)], fill=(0, 150, 0), width=2)
    draw.line([(300, 150), (320, 130)], fill=(0, 150, 0), width=2)
    
    return img

def create_static_to_living_image():
    """Chapter 10: From Static Plans to Adaptive Frameworks"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((20, 20), "Chapter 10: What Changed", fill=(0, 0, 0), font_size=16)
    draw.text((20, 40), "From Static Plans to Adaptive Frameworks", fill=(100, 100, 100), font_size=14)
    
    # Left side - Static Plan
    draw.text((50, 80), "OLD: Static Plan", fill=(200, 0, 0), font_size=12)
    
    # Rigid framework
    draw.rectangle([(50, 100), (150, 180)], fill=(200, 200, 200), outline=(200, 0, 0), width=2)
    draw.line([(50, 120), (150, 120)], fill=(200, 0, 0), width=2)  # Horizontal line
    draw.line([(50, 140), (150, 140)], fill=(200, 0, 0), width=2)  # Horizontal line
    draw.line([(50, 160), (150, 160)], fill=(200, 0, 0), width=2)  # Horizontal line
    draw.line([(100, 100), (100, 180)], fill=(200, 0, 0), width=2)  # Vertical line
    
    # Static indicator
    draw.text((160, 140), "🔒", fill=(200, 0, 0), font_size=20)
    
    # Arrow
    draw.polygon([(180, 130), (200, 140), (180, 150)], fill=(0, 0, 0))
    
    # Right side - Adaptive Framework
    draw.text((220, 80), "NEW: Adaptive Framework", fill=(0, 150, 0), font_size=12)
    
    # Flexible, organic framework
    draw.ellipse([(250, 100), (350, 200)], outline=(0, 150, 0), width=3)
    
    # Adaptive nodes
    draw.ellipse([(280, 120), (290, 130)], fill=(0, 150, 0))
    draw.ellipse([(310, 120), (320, 130)], fill=(0, 150, 0))
    draw.ellipse([(280, 150), (290, 160)], fill=(0, 150, 0))
    draw.ellipse([(310, 150), (320, 160)], fill=(0, 150, 0))
    draw.ellipse([(300, 170), (310, 180)], fill=(0, 150, 0))
    
    # Flexible connections
    draw.line([(285, 125), (315, 125)], fill=(0, 150, 0), width=2)
    draw.line([(285, 155), (315, 155)], fill=(0, 150, 0), width=2)
    draw.line([(285, 125), (285, 155)], fill=(0, 150, 0), width=2)
    draw.line([(315, 125), (315, 155)], fill=(0, 150, 0), width=2)
    draw.line([(305, 155), (305, 175)], fill=(0, 150, 0), width=2)
    
    # Growth indicators
    draw.text((360, 140), "🌱", fill=(0, 150, 0), font_size=20)
    
    return img

if __name__ == "__main__":
    create_chapter_images()

