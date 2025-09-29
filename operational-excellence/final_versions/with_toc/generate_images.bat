@echo off
echo 🎨 Generating missing images from Mermaid diagrams...
echo ============================================================

REM Check if mermaid-cli is installed
where mmdc >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ mermaid-cli not found!
    echo Please install it with: npm install -g @mermaid-js/mermaid-cli
    echo.
    echo 💡 Alternative: Use Mermaid Live Editor at https://mermaid.live/
    pause
    exit /b 1
)

REM Generate AI Enhanced KPI System
echo Generating ai_enhanced_kpi_system.png...
mmdc -i ai_enhanced_kpi_system.mmd -o images/ai_enhanced_kpi_system.png -w 1200 -H 900 -b white -s 2
if %errorlevel% equ 0 (
    echo ✅ Successfully generated ai_enhanced_kpi_system.png
) else (
    echo ❌ Failed to generate ai_enhanced_kpi_system.png
)

REM Generate Human-AI Collaboration Diagram
echo Generating human_ai_collaboration_diagram.png...
mmdc -i human_ai_collaboration_diagram.mmd -o images/human_ai_collaboration_diagram.png -w 1200 -H 900 -b white -s 2
if %errorlevel% equ 0 (
    echo ✅ Successfully generated human_ai_collaboration_diagram.png
) else (
    echo ❌ Failed to generate human_ai_collaboration_diagram.png
)

REM Generate AI Ethics Framework
echo Generating ai_ethics_framework.png...
mmdc -i ai_ethics_framework.mmd -o images/ai_ethics_framework.png -w 1200 -H 900 -b white -s 2
if %errorlevel% equ 0 (
    echo ✅ Successfully generated ai_ethics_framework.png
) else (
    echo ❌ Failed to generate ai_ethics_framework.png
)

echo ============================================================
echo 🎉 Image generation complete!
echo.
echo 📝 Next steps:
echo 1. Verify the generated images in the 'images/' directory
echo 2. Test PDF generation to ensure images load correctly
echo 3. Check print quality of the final PDF
echo.
pause


