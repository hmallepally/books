#!/usr/bin/env python3
"""
Comprehensive recovery script to rebuild the working HTML file
"""

import re
import os
from pathlib import Path

def apply_table_conversions():
    """Apply all table conversions from today's work"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print("🔄 Applying table conversions...")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define conversion patterns
    conversions = [
        # Chapter 7: Total Quality Management
        {
            'pattern': r'<h4 id="tqm-principles">TQM Principles</h4>.*?<p><div class="tech-label">Key Principles:</div>.*?</p>',
            'replacement': '''<h4 id="tqm-principles">TQM Principles</h4>
            <table class="blue-table">
                <thead>
                    <tr>
                        <th>Principle</th>
                        <th>Description</th>
                        <th>AI Enhancement</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Customer Focus</strong></td>
                        <td>Understanding and meeting customer needs</td>
                        <td>AI-powered customer analytics and predictive satisfaction modeling</td>
                    </tr>
                    <tr>
                        <td><strong>Continuous Improvement</strong></td>
                        <td>Ongoing process optimization</td>
                        <td>Machine learning-driven process optimization and automated improvement suggestions</td>
                    </tr>
                    <tr>
                        <td><strong>Employee Involvement</strong></td>
                        <td>Engaging all employees in quality efforts</td>
                        <td>AI-assisted training, skill assessment, and collaborative decision-making platforms</td>
                    </tr>
                    <tr>
                        <td><strong>Process Approach</strong></td>
                        <td>Managing activities as processes</td>
                        <td>AI process mining, workflow optimization, and intelligent automation</td>
                    </tr>
                    <tr>
                        <td><strong>Systematic Approach</strong></td>
                        <td>Understanding system interdependencies</td>
                        <td>AI system modeling, predictive analytics, and holistic performance monitoring</td>
                    </tr>
                </tbody>
            </table>''',
            'flags': re.DOTALL
        },
        
        # Chapter 8: Customer Satisfaction
        {
            'pattern': r'<h4 id="ai-customer-analytics">AI-Powered Customer Analytics</h4>.*?<p><div class="tech-label">Key Components:</div>.*?</p>',
            'replacement': '''<h4 id="ai-customer-analytics">AI-Powered Customer Analytics</h4>
            <table class="blue-table">
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Function</th>
                        <th>Business Impact</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Sentiment Analysis</strong></td>
                        <td>Real-time emotion detection from customer feedback</td>
                        <td>Immediate issue identification and proactive response</td>
                    </tr>
                    <tr>
                        <td><strong>Predictive Modeling</strong></td>
                        <td>Forecasting customer behavior and satisfaction</td>
                        <td>Preventive actions to maintain customer loyalty</td>
                    </tr>
                    <tr>
                        <td><strong>Personalization Engine</strong></td>
                        <td>Tailored experiences based on customer data</td>
                        <td>Increased engagement and satisfaction scores</td>
                    </tr>
                    <tr>
                        <td><strong>Churn Prediction</strong></td>
                        <td>Early warning system for customer departure</td>
                        <td>Reduced customer loss and retention improvement</td>
                    </tr>
                </tbody>
            </table>''',
            'flags': re.DOTALL
        },
        
        # Chapter 9: Software Development
        {
            'pattern': r'<h4 id="ai-sdlc-integration">AI Integration in SDLC</h4>.*?<p><div class="tech-label">Implementation Areas:</div>.*?</p>',
            'replacement': '''<h4 id="ai-sdlc-integration">AI Integration in SDLC</h4>
            <table class="green-table">
                <thead>
                    <tr>
                        <th>SDLC Phase</th>
                        <th>AI Application</th>
                        <th>Expected Outcome</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Planning & Analysis</strong></td>
                        <td>AI requirement analysis, feasibility assessment</td>
                        <td>More accurate project estimation and scope definition</td>
                    </tr>
                    <tr>
                        <td><strong>Design</strong></td>
                        <td>AI-assisted architecture design, code generation</td>
                        <td>Faster design cycles and improved code quality</td>
                    </tr>
                    <tr>
                        <td><strong>Development</strong></td>
                        <td>Automated testing, code review, debugging</td>
                        <td>Reduced development time and fewer bugs</td>
                    </tr>
                    <tr>
                        <td><strong>Testing</strong></td>
                        <td>Intelligent test case generation, automated QA</td>
                        <td>Comprehensive coverage and faster testing cycles</td>
                    </tr>
                    <tr>
                        <td><strong>Deployment</strong></td>
                        <td>AI-driven deployment strategies, monitoring</td>
                        <td>Smoother deployments and proactive issue detection</td>
                    </tr>
                    <tr>
                        <td><strong>Maintenance</strong></td>
                        <td>Predictive maintenance, automated updates</td>
                        <td>Reduced downtime and improved system reliability</td>
                    </tr>
                </tbody>
            </table>''',
            'flags': re.DOTALL
        }
    ]
    
    # Apply conversions
    for i, conversion in enumerate(conversions):
        print(f"  Applying conversion {i+1}/{len(conversions)}...")
        content = re.sub(conversion['pattern'], conversion['replacement'], content, flags=conversion.get('flags', 0))
    
    # Save the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Table conversions applied successfully")
    return True

def update_styling():
    """Update styling to match today's work"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    print("🎨 Updating styling...")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update font size to 10pt and add table styling
    style_updates = '''
        /* Updated styling for today's work */
        body {
            font-family: 'Times New Roman', serif;
            font-size: 10pt; /* Changed from 12pt to 10pt */
            line-height: 1.6;
            color: #2c3e50;
            background: white;
        }
        
        /* Table styling */
        .blue-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #f8f9fa;
            border: 2px solid #3498db;
        }
        
        .blue-table th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        
        .blue-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #bdc3c7;
        }
        
        .blue-table tr:nth-child(even) {
            background: #ecf0f1;
        }
        
        .green-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #f8fff8;
            border: 2px solid #27ae60;
        }
        
        .green-table th {
            background: #27ae60;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        
        .green-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #95a5a6;
        }
        
        .green-table tr:nth-child(even) {
            background: #d5f4e6;
        }
        
        /* Page setup for KDP */
        @page {
            size: 8.5in 11in; /* US Letter size for KDP */
            margin: 0.75in;
        }
    '''
    
    # Replace the existing body styling
    content = re.sub(r'body\s*\{[^}]*\}', 'body {\n            font-family: \'Times New Roman\', serif;\n            font-size: 10pt;\n            line-height: 1.6;\n            color: #2c3e50;\n            background: white;\n        }', content)
    
    # Add table styling after the existing styles
    content = re.sub(r'(</style>)', style_updates + r'\n    \1', content)
    
    # Save the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Styling updated successfully")
    return True

def add_chapter_markers():
    """Add chapter end markers"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    print("📍 Adding chapter end markers...")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all chapter titles
    chapter_pattern = r'<h1 class="chapter-title">(Chapter \d+:.*?)</h1>'
    chapters = re.findall(chapter_pattern, content)
    
    print(f"Found {len(chapters)} chapters")
    
    # Add markers at the end of each chapter
    for i, chapter in enumerate(chapters):
        print(f"  Adding marker for {chapter}...")
        
        # Find the next chapter or end of content
        if i < len(chapters) - 1:
            next_chapter = chapters[i + 1]
            # Find content between current and next chapter
            pattern = rf'<h1 class="chapter-title">{re.escape(chapter)}</h1>(.*?)<h1 class="chapter-title">{re.escape(next_chapter)}</h1>'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                chapter_content = match.group(1)
                # Add marker at the end of chapter content
                marker = '''
                <div class="chapter-end-marker">
                    <div class="marker-line"></div>
                    <div class="marker-dots">●●●</div>
                </div>
                '''
                updated_content = chapter_content + marker
                content = content.replace(match.group(0), f'<h1 class="chapter-title">{chapter}</h1>{updated_content}<h1 class="chapter-title">{next_chapter}</h1>')
    
    # Add marker for Introduction
    intro_pattern = r'(<h1 class="chapter-title">Introduction: The AI Transformation Journey</h1>.*?)(<h1 class="chapter-title">Chapter 1:)'
    intro_match = re.search(intro_pattern, content, re.DOTALL)
    if intro_match:
        marker = '''
        <div class="chapter-end-marker">
            <div class="marker-line"></div>
            <div class="marker-dots">●●●</div>
        </div>
        '''
        content = content.replace(intro_match.group(1), intro_match.group(1) + marker)
    
    # Add CSS for chapter markers
    marker_css = '''
        .chapter-end-marker {
            text-align: center;
            margin: 40px 0;
            padding: 20px 0;
        }
        
        .marker-line {
            height: 2px;
            background: linear-gradient(to right, transparent, #3498db, transparent);
            margin: 0 20%;
        }
        
        .marker-dots {
            color: #3498db;
            font-size: 16pt;
            margin-top: 10px;
            letter-spacing: 10px;
        }
    '''
    
    content = re.sub(r'(</style>)', marker_css + r'\n    \1', content)
    
    # Save the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Chapter markers added successfully")
    return True

def main():
    """Main recovery function"""
    
    print("🚀 Starting comprehensive recovery process...")
    print("=" * 60)
    
    # Step 1: Apply table conversions
    if not apply_table_conversions():
        print("❌ Failed to apply table conversions")
        return False
    
    # Step 2: Update styling
    if not update_styling():
        print("❌ Failed to update styling")
        return False
    
    # Step 3: Add chapter markers
    if not add_chapter_markers():
        print("❌ Failed to add chapter markers")
        return False
    
    print("=" * 60)
    print("✅ Recovery process completed successfully!")
    print("📁 Updated file: final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html")
    
    return True

if __name__ == "__main__":
    main()
