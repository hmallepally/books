#!/usr/bin/env python3
"""
Comprehensive script to convert remaining chapters to table structures
"""

import re
import os

def convert_remaining_chapters():
    """Convert remaining chapters to table structures"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print("🚀 Converting remaining chapters to table structures...")
    print("=" * 60)
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define conversion patterns
    conversions = [
        # Chapter 7: Total Quality Management
        {
            'pattern': r'<h4 id="tqm-principles">TQM Principles</h4>.*?<p><div class="tech-label">Key Principles:</div>.*?</p>',
            'replacement': '''<h4 id="tqm-principles">TQM Principles</h4>
<table class="comparison-table">
    <thead>
        <tr>
            <th style="width: 30%;">Principle</th>
            <th style="width: 70%;">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Customer Focus</strong></td>
            <td>Understanding and meeting customer requirements</td>
        </tr>
        <tr>
            <td><strong>Continuous Improvement</strong></td>
            <td>Ongoing enhancement of processes and products</td>
        </tr>
        <tr>
            <td><strong>Employee Involvement</strong></td>
            <td>Engaging all employees in quality improvement</td>
        </tr>
        <tr>
            <td><strong>Process Approach</strong></td>
            <td>Managing activities as interconnected processes</td>
        </tr>
        <tr>
            <td><strong>System Approach</strong></td>
            <td>Understanding and managing interrelated processes</td>
        </tr>
    </tbody>
</table>''',
            'description': 'TQM Principles table'
        },
        
        # Chapter 8: Customer Satisfaction
        {
            'pattern': r'<h4 id="ai-powered-customer-analytics">AI-Powered Customer Analytics</h4>.*?<p><div class="tech-label">Key Capabilities:</div>.*?</p>',
            'replacement': '''<h4 id="ai-powered-customer-analytics">AI-Powered Customer Analytics</h4>
<table class="comparison-table">
    <thead>
        <tr>
            <th style="width: 30%;">Capability</th>
            <th style="width: 70%;">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Sentiment Analysis</strong></td>
            <td>Analyzing customer emotions and opinions from text data</td>
        </tr>
        <tr>
            <td><strong>Predictive Analytics</strong></td>
            <td>Forecasting customer behavior and preferences</td>
        </tr>
        <tr>
            <td><strong>Personalization</strong></td>
            <td>Creating tailored experiences for individual customers</td>
        </tr>
        <tr>
            <td><strong>Churn Prediction</strong></td>
            <td>Identifying customers likely to leave</td>
        </tr>
    </tbody>
</table>''',
            'description': 'AI-Powered Customer Analytics table'
        },
        
        # Chapter 9: Software Development
        {
            'pattern': r'<h4 id="ai-in-software-testing">AI in Software Testing</h4>.*?<p><div class="tech-label">Key Applications:</div>.*?</p>',
            'replacement': '''<h4 id="ai-in-software-testing">AI in Software Testing</h4>
<table class="comparison-table">
    <thead>
        <tr>
            <th style="width: 30%;">Application</th>
            <th style="width: 70%;">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Automated Test Generation</strong></td>
            <td>AI generates test cases automatically</td>
        </tr>
        <tr>
            <td><strong>Intelligent Test Execution</strong></td>
            <td>AI optimizes test execution order and coverage</td>
        </tr>
        <tr>
            <td><strong>Defect Prediction</strong></td>
            <td>AI predicts where defects are likely to occur</td>
        </tr>
        <tr>
            <td><strong>Performance Testing</strong></td>
            <td>AI optimizes performance test scenarios</td>
        </tr>
    </tbody>
</table>''',
            'description': 'AI in Software Testing table'
        },
        
        # Chapter 10: Leadership
        {
            'pattern': r'<h4 id="ai-leadership-competencies">AI Leadership Competencies</h4>.*?<p><div class="tech-label">Key Competencies:</div>.*?</p>',
            'replacement': '''<h4 id="ai-leadership-competencies">AI Leadership Competencies</h4>
<table class="comparison-table">
    <thead>
        <tr>
            <th style="width: 30%;">Competency</th>
            <th style="width: 70%;">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>AI Literacy</strong></td>
            <td>Understanding AI capabilities and limitations</td>
        </tr>
        <tr>
            <td><strong>Data-Driven Decision Making</strong></td>
            <td>Using data and analytics to guide decisions</td>
        </tr>
        <tr>
            <td><strong>Change Management</strong></td>
            <td>Leading organizational transformation</td>
        </tr>
        <tr>
            <td><strong>Ethical AI Leadership</strong></td>
            <td>Ensuring responsible AI implementation</td>
        </tr>
    </tbody>
</table>''',
            'description': 'AI Leadership Competencies table'
        }
    ]
    
    # Apply conversions
    total_conversions = 0
    for conversion in conversions:
        if re.search(conversion['pattern'], content, re.DOTALL):
            content = re.sub(conversion['pattern'], conversion['replacement'], content, flags=re.DOTALL)
            total_conversions += 1
            print(f"✅ Converted: {conversion['description']}")
        else:
            print(f"⚠️  Pattern not found: {conversion['description']}")
    
    # Write the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n🎉 Conversion completed!")
    print(f"📊 Total conversions applied: {total_conversions}")
    print(f"📄 Updated file: {html_file}")
    
    return True

def main():
    """Main function"""
    print("🚀 Comprehensive Chapter Conversion Script")
    print("=" * 50)
    
    success = convert_remaining_chapters()
    
    if success:
        print("\n✅ All conversions completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Review the updated HTML file")
        print("   2. Generate a new PDF to see all improvements")
        print("   3. Check for any remaining sections that could be converted")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
