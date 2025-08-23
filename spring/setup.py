#!/usr/bin/env python3
"""
Spring Framework Book Setup Script
This script helps set up the development environment for the Spring Framework book.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("Spring Framework Book - Development Environment Setup")
    print("=" * 60)
    print()

def check_java():
    """Check if Java is installed and get version"""
    print("Checking Java installation...")
    
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, stderr=subprocess.PIPE)
        if result.returncode == 0:
            # Extract version from stderr
            version_line = result.stderr.split('\n')[0]
            print(f"✅ Java found: {version_line}")
            
            # Check if it's Java 8 or higher
            if 'version "1.8' in version_line or 'version "9' in version_line or 'version "10' in version_line or 'version "11' in version_line or 'version "12' in version_line or 'version "13' in version_line or 'version "14' in version_line or 'version "15' in version_line or 'version "16' in version_line or 'version "17' in version_line or 'version "18' in version_line or 'version "19' in version_line or 'version "20' in version_line or 'version "21' in version_line:
                print("✅ Java version is compatible (8 or higher)")
                return True
            else:
                print("⚠️  Java version might be too old. Recommended: Java 8 or higher")
                return False
        else:
            print("❌ Java not found or not working properly")
            return False
    except FileNotFoundError:
        print("❌ Java not found in PATH")
        return False

def check_maven():
    """Check if Maven is installed"""
    print("\nChecking Maven installation...")
    
    try:
        result = subprocess.run(['mvn', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Extract version from output
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Maven found: {version_line}")
            return True
        else:
            print("❌ Maven not working properly")
            return False
    except FileNotFoundError:
        print("❌ Maven not found in PATH")
        return False

def check_gradle():
    """Check if Gradle is installed"""
    print("\nChecking Gradle installation...")
    
    try:
        result = subprocess.run(['gradle', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Extract version from output
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Gradle found: {version_line}")
            return True
        else:
            print("❌ Gradle not working properly")
            return False
    except FileNotFoundError:
        print("❌ Gradle not found in PATH")
        return False

def check_ide():
    """Check for common IDEs"""
    print("\nChecking for IDEs...")
    
    ides = {
        'IntelliJ IDEA': [
            'C:\\Program Files\\JetBrains\\IntelliJ IDEA*',
            'C:\\Users\\*\\AppData\\Local\\JetBrains\\IntelliJIdea*',
            '/Applications/IntelliJ IDEA.app',
            '~/Library/Application Support/JetBrains/IntelliJIdea*'
        ],
        'Eclipse': [
            'C:\\Program Files\\Eclipse*',
            'C:\\Users\\*\\eclipse*',
            '/Applications/Eclipse.app',
            '~/eclipse*'
        ],
        'VS Code': [
            'C:\\Users\\*\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe',
            '/Applications/Visual Studio Code.app',
            '/usr/bin/code'
        ]
    }
    
    found_ides = []
    
    for ide_name, paths in ides.items():
        for path_pattern in paths:
            if os.path.exists(os.path.expanduser(path_pattern)):
                found_ides.append(ide_name)
                break
    
    if found_ides:
        print(f"✅ Found IDEs: {', '.join(found_ides)}")
    else:
        print("⚠️  No common IDEs found. Consider installing IntelliJ IDEA, Eclipse, or VS Code")
    
    return found_ides

def create_sample_project():
    """Create a sample Spring Boot project"""
    print("\nCreating sample Spring Boot project...")
    
    project_dir = Path("sample-spring-app")
    
    if project_dir.exists():
        print(f"⚠️  Directory {project_dir} already exists. Skipping project creation.")
        return
    
    try:
        # Create project directory
        project_dir.mkdir()
        
        # Create pom.xml
        pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.1.0</version>
        <relativePath/>
    </parent>
    
    <groupId>com.example</groupId>
    <artifactId>sample-spring-app</artifactId>
    <version>1.0.0</version>
    <name>Sample Spring Boot Application</name>
    <description>Sample application for Spring Framework Book</description>
    
    <properties>
        <java.version>17</java.version>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
        
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
        </dependency>
        
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
        
        <dependency>
            <groupId>com.h2database</groupId>
            <artifactId>h2</artifactId>
            <scope>runtime</scope>
        </dependency>
        
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>'''
        
        with open(project_dir / "pom.xml", "w") as f:
            f.write(pom_content)
        
        # Create main application class
        src_main_java = project_dir / "src" / "main" / "java" / "com" / "example"
        src_main_java.mkdir(parents=True)
        
        main_app_content = '''package com.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class SampleSpringApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(SampleSpringApplication.class, args);
    }
}'''
        
        with open(src_main_java / "SampleSpringApplication.java", "w") as f:
            f.write(main_app_content)
        
        # Create application.properties
        src_main_resources = project_dir / "src" / "main" / "resources"
        src_main_resources.mkdir(parents=True)
        
        properties_content = '''# Server Configuration
server.port=8080

# Database Configuration
spring.datasource.url=jdbc:h2:mem:testdb
spring.datasource.driver-class-name=org.h2.Driver
spring.datasource.username=sa
spring.datasource.password=

# JPA Configuration
spring.jpa.hibernate.ddl-auto=create-drop
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true

# H2 Console
spring.h2.console.enabled=true
spring.h2.console.path=/h2-console

# Logging
logging.level.org.springframework.security=DEBUG
logging.level.org.hibernate.SQL=DEBUG

# Actuator
management.endpoints.web.exposure.include=health,info,metrics'''
        
        with open(src_main_resources / "application.properties", "w") as f:
            f.write(properties_content)
        
        # Create README
        readme_content = '''# Sample Spring Boot Application

This is a sample Spring Boot application created for the Spring Framework Book.

## Features
- Spring Boot 3.1.0
- Spring Security
- Spring Data JPA
- H2 Database
- Spring Boot Actuator

## Running the Application

1. Make sure you have Java 17+ and Maven installed
2. Navigate to this directory
3. Run: `mvn spring-boot:run`
4. Open http://localhost:8080 in your browser
5. H2 Console: http://localhost:8080/h2-console

## Project Structure
```
src/
├── main/
│   ├── java/
│   │   └── com/
│   │       └── example/
│   │           └── SampleSpringApplication.java
│   └── resources/
│       └── application.properties
└── test/
    └── java/
        └── com/
            └── example/
                └── SampleSpringApplicationTests.java
```

## Next Steps
- Add controllers, services, and repositories
- Implement security configurations
- Add unit and integration tests
- Explore Spring Boot features'''
        
        with open(project_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print(f"✅ Sample Spring Boot project created in {project_dir}")
        print("   - Maven project with Spring Boot 3.1.0")
        print("   - Basic configuration and main class")
        print("   - H2 database for development")
        print("   - Spring Security enabled")
        
    except Exception as e:
        print(f"❌ Error creating sample project: {e}")

def print_setup_instructions():
    """Print setup instructions"""
    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS")
    print("=" * 60)
    print()
    
    print("1. JAVA SETUP:")
    print("   - Download and install Java 17 or higher from:")
    print("     https://adoptium.net/ or https://www.oracle.com/java/")
    print("   - Set JAVA_HOME environment variable")
    print("   - Add Java bin directory to PATH")
    print()
    
    print("2. BUILD TOOL SETUP:")
    print("   - Maven: Download from https://maven.apache.org/download.cgi")
    print("   - Gradle: Download from https://gradle.org/releases/")
    print("   - Add build tool bin directory to PATH")
    print()
    
    print("3. IDE SETUP:")
    print("   - IntelliJ IDEA: https://www.jetbrains.com/idea/")
    print("   - Eclipse: https://www.eclipse.org/downloads/")
    print("   - VS Code: https://code.visualstudio.com/")
    print()
    
    print("4. SPRING INITIALIZR:")
    print("   - Use https://start.spring.io/ to create new projects")
    print("   - Select Spring Boot 3.x and Java 17+")
    print("   - Choose appropriate dependencies")
    print()
    
    print("5. SAMPLE PROJECT:")
    print("   - A sample Spring Boot project has been created")
    print("   - Navigate to 'sample-spring-app' directory")
    print("   - Run 'mvn spring-boot:run' to start the application")
    print()
    
    print("6. LEARNING PATH:")
    print("   - Start with Chapter 1: Dependency Injection")
    print("   - Progress through AOP, Spring Boot, and advanced topics")
    print("   - Practice with the provided examples")
    print("   - Build the suggested projects")
    print()
    
    print("7. RESOURCES:")
    print("   - Spring Documentation: https://spring.io/guides")
    print("   - Spring Boot Reference: https://docs.spring.io/spring-boot/docs/current/reference/html/")
    print("   - Spring Security Reference: https://docs.spring.io/spring-security/site/docs/current/reference/html5/")
    print()

def main():
    """Main setup function"""
    print_header()
    
    # Check prerequisites
    java_ok = check_java()
    maven_ok = check_maven()
    gradle_ok = check_gradle()
    ides = check_ide()
    
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if java_ok:
        print("✅ Java: Ready")
    else:
        print("❌ Java: Needs installation")
    
    if maven_ok or gradle_ok:
        print("✅ Build Tool: Ready")
    else:
        print("❌ Build Tool: Needs installation")
    
    if ides:
        print("✅ IDE: Found")
    else:
        print("⚠️  IDE: Consider installing")
    
    print()
    
    # Create sample project
    create_sample_project()
    
    # Print setup instructions
    print_setup_instructions()
    
    print("🎉 Setup complete! You're ready to start learning Spring Framework.")
    print("   Happy coding! 🚀")

if __name__ == "__main__":
    main()
