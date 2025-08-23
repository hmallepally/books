/**
 * Chapter 1: Java Fundamentals - Exercise Problems
 * These exercises reinforce the concepts learned in Chapter 1.
 */

import java.util.*;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;

public class Chapter1Exercises {
    
    public static void main(String[] args) {
        System.out.println("🎯 Chapter 1: Java Fundamentals - Exercise Problems");
        System.out.println("==================================================");
        
        // Run all exercises
        exercise1_1();
        exercise1_2();
        exercise1_3();
        exercise1_4();
        exercise1_5();
        exercise1_6();
        exercise1_7();
        exercise1_8();
        
        System.out.println("\n🎉 All exercises completed!");
        System.out.println("Check the solutions section for detailed explanations.");
    }
    
    /**
     * Exercise 1.1: Memory Leak Detection
     * Create a program that demonstrates a memory leak and then fixes it.
     */
    public static void exercise1_1() {
        System.out.println("\n📝 Exercise 1.1: Memory Leak Detection");
        System.out.println("----------------------------------------");
        System.out.println("Problem: Create a program that demonstrates a memory leak");
        System.out.println("and then shows how to fix it.");
        System.out.println();
        
        // TODO: Implement this exercise
        System.out.println("Your task:");
        System.out.println("1. Create a class that holds references to objects");
        System.out.println("2. Demonstrate how forgetting to clear references causes memory leaks");
        System.out.println("3. Show the proper way to prevent memory leaks");
        System.out.println("4. Measure memory usage before and after");
        
        // Hint: Use a static collection that grows without bounds
        // Hint: Use MemoryMXBean to measure memory usage
        // Hint: Show the difference between proper cleanup and no cleanup
    }
    
    /**
     * Exercise 1.2: String Optimization
     * Write a program that efficiently concatenates 100,000 strings and measures performance.
     */
    public static void exercise1_2() {
        System.out.println("\n📝 Exercise 1.2: String Optimization");
        System.out.println("--------------------------------------");
        System.out.println("Problem: Write a program that efficiently concatenates");
        System.out.println("100,000 strings and measures performance.");
        System.out.println();
        
        // TODO: Implement this exercise
        System.out.println("Your task:");
        System.out.println("1. Implement string concatenation using the + operator");
        System.out.println("2. Implement string concatenation using StringBuilder");
        System.out.println("3. Implement string concatenation using StringBuffer");
        System.out.println("4. Measure and compare performance of all three methods");
        System.out.println("5. Explain why StringBuilder is generally the best choice");
        
        // Hint: Use System.nanoTime() for precise timing
        // Hint: Create a large number of strings to concatenate
        // Hint: Consider thread safety vs performance trade-offs
    }
    
    /**
     * Exercise 1.3: Primitive vs Wrapper Performance
     * Compare the performance of primitive types vs wrapper classes in a loop.
     */
    public static void exercise1_3() {
        System.out.println("\n📝 Exercise 1.3: Primitive vs Wrapper Performance");
        System.out.println("--------------------------------------------------");
        System.out.println("Problem: Compare the performance of primitive types");
        System.out.println("vs wrapper classes in a loop of 10 million iterations.");
        System.out.println();
        
        // TODO: Implement this exercise
        System.out.println("Your task:");
        System.out.println("1. Create a loop that performs arithmetic operations with primitives");
        System.out.println("2. Create a loop that performs arithmetic operations with wrappers");
        System.out.println("3. Measure execution time for both approaches");
        System.out.println("4. Calculate the performance ratio");
        System.out.println("5. Explain the performance difference");
        
        // Hint: Use different operations: addition, multiplication, comparison
        // Hint: Consider the overhead of autoboxing and unboxing
        // Hint: Use a large number of iterations to see significant differences
    }
    
    /**
     * Exercise 1.4: String Pool Investigation
     * Investigate how the String Pool works and demonstrate its benefits.
     */
    public static void exercise1_4() {
        System.out.println("\n📝 Exercise 1.4: String Pool Investigation");
        System.out.println("--------------------------------------------");
        System.out.println("Problem: Investigate how the String Pool works");
        System.out.println("and demonstrate its memory benefits.");
        System.out.println();
        
        // TODO: Implement this exercise
        System.out.println("Your task:");
        System.out.println("1. Create multiple String literals with the same content");
        System.out.println("2. Create multiple String objects with new String()");
        System.out.println("3. Compare memory usage and reference equality");
        System.out.println("4. Demonstrate compile-time string concatenation");
        System.out.println("5. Show how the String Pool saves memory");
        
        // Hint: Use == to compare references, equals() to compare content
        // Hint: Use System.identityHashCode() to see object identity
        // Hint: Show the difference between literal and new String()
    }
    
    /**
     * Exercise 1.5: Memory Model Understanding
     * Create a program that demonstrates different memory areas in Java.
     */
    public static void exercise1_5() {
        System.out.println("\n📝 Exercise 1.5: Memory Model Understanding");
        System.out.println("---------------------------------------------");
        System.out.println("Problem: Create a program that demonstrates");
        System.out.println("different memory areas in Java.");
        System.out.println();
        
        // TODO: Implement this exercise
        System.out.println("Your task:");
        System.out.println("1. Demonstrate stack memory (local variables)");
        System.out.println("2. Demonstrate heap memory (objects)");
        System.out.println("3. Demonstrate method area (static variables)");
        System.out.println("4. Show memory usage patterns");
        System.out.println("5. Explain when each memory area is used");
        
        // Hint: Use MemoryMXBean to get memory information
        // Hint: Create objects of different sizes
        // Hint: Show the difference between stack and heap allocation
    }
    
    /**
     * Exercise 1.6: Type System Mastery
     * Demonstrate advanced features of Java's type system.
     */
    public static void exercise1_6() {
        System.out.println("\n📝 Exercise 1.6: Type System Mastery");
        System.out.println("--------------------------------------");
        System.out.println("Problem: Demonstrate advanced features of Java's type system.");
        System.out.println();
        
        // TODO: Implement this exercise
        System.out.println("Your task:");
        System.out.println("1. Show autoboxing and unboxing in action");
        System.out.println("2. Demonstrate type conversion and casting");
        System.out.println("3. Show how to handle null values safely");
        System.out.println("4. Use utility methods from wrapper classes");
        System.out.println("5. Explain when to use primitives vs wrappers");
        
        // Hint: Show different ways to convert between types
        // Hint: Demonstrate safe null checking
        // Hint: Use wrapper class utility methods like parseInt, toBinaryString
    }
    
    /**
     * Exercise 1.7: Performance Profiling
     * Create a simple performance profiling tool for Java programs.
     */
    public static void exercise1_7() {
        System.out.println("\n📝 Exercise 1.7: Performance Profiling");
        System.out.println("----------------------------------------");
        System.out.println("Problem: Create a simple performance profiling tool");
        System.out.println("for Java programs.");
        System.out.println();
        
        // TODO: Implement this exercise
        System.out.println("Your task:");
        System.out.println("1. Create a Profiler class that measures execution time");
        System.out.println("2. Add memory usage measurement");
        System.out.println("3. Create a simple benchmarking framework");
        System.out.println("4. Profile different algorithms or approaches");
        System.out.println("5. Generate performance reports");
        
        // Hint: Use System.nanoTime() for timing
        // Hint: Use MemoryMXBean for memory measurement
        // Hint: Create a fluent API for easy profiling
        // Hint: Support multiple measurement types
    }
    
    /**
     * Exercise 1.8: Best Practices Implementation
     * Implement and demonstrate Java best practices from Chapter 1.
     */
    public static void exercise1_8() {
        System.out.println("\n📝 Exercise 1.8: Best Practices Implementation");
        System.out.println("------------------------------------------------");
        System.out.println("Problem: Implement and demonstrate Java best practices");
        System.out.println("from Chapter 1.");
        System.out.println();
        
        // TODO: Implement this exercise
        System.out.println("Your task:");
        System.out.println("1. Create a class that follows all best practices");
        System.out.println("2. Show proper encapsulation and data hiding");
        System.out.println("3. Demonstrate efficient string handling");
        System.out.println("4. Show proper memory management");
        System.out.println("5. Include comprehensive error handling");
        
        // Hint: Use private fields with public getters/setters
        // Hint: Implement proper equals(), hashCode(), and toString()
        // Hint: Show defensive copying where appropriate
        // Hint: Include input validation and error messages
    }
    
    /**
     * Helper method to measure memory usage
     */
    private static long getCurrentMemoryUsage() {
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        return memoryBean.getHeapMemoryUsage().getUsed();
    }
    
    /**
     * Helper method to format memory size
     */
    private static String formatMemorySize(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
        return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}
