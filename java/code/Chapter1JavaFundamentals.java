/**
 * Chapter 1: Java Fundamentals & Philosophy
 * This file demonstrates core Java concepts with practical examples.
 */

import java.util.*;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;

public class Chapter1JavaFundamentals {
    
    public static void main(String[] args) {
        System.out.println("🚀 Chapter 1: Java Fundamentals & Philosophy");
        System.out.println("==================================================");
        
        // 1.1 The Java Philosophy
        demonstrateJavaPhilosophy();
        
        // 1.2 Java Memory Model
        demonstrateMemoryModel();
        
        // 1.3 Primitive Types vs Wrapper Classes
        demonstrateTypeSystem();
        
        // 1.4 String Immutability & String Pool
        demonstrateStringImmutability();
        
        // 1.5 Performance Comparisons
        demonstratePerformanceComparisons();
        
        // 1.6 Memory Management Best Practices
        demonstrateMemoryManagement();
    }
    
    /**
     * 1.1 Demonstrate Java's core philosophy
     */
    public static void demonstrateJavaPhilosophy() {
        System.out.println("\n📚 1.1 The Java Philosophy");
        System.out.println("------------------------------");
        
        // Platform independence demonstration
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("Java Vendor: " + System.getProperty("java.vendor"));
        System.out.println("OS Name: " + System.getProperty("os.name"));
        System.out.println("OS Architecture: " + System.getProperty("os.arch"));
        
        // Object-oriented nature
        Object obj = "This is an object";
        System.out.println("Everything is an object: " + obj.getClass().getSimpleName());
        
        // Strong typing
        int number = 42;
        // number = "string"; // This would cause compilation error
        System.out.println("Strong typing prevents type errors at compile time");
    }
    
    /**
     * 1.2 Demonstrate Java memory model
     */
    public static void demonstrateMemoryModel() {
        System.out.println("\n🧠 1.2 Java Memory Model");
        System.out.println("------------------------------");
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        
        // Stack memory (local variables)
        int localVar = 100;
        String localString = "Local string";
        System.out.println("Stack variables created: " + localVar + ", " + localString);
        
        // Heap memory (objects)
        String heapString = new String("Heap string");
        Integer heapInteger = new Integer(200);
        System.out.println("Heap objects created: " + heapString + ", " + heapInteger);
        
        // Method area (static variables)
        staticVariable = "Static variable value";
        System.out.println("Static variable: " + staticVariable);
        
        // Memory usage information
        long heapUsed = memoryBean.getHeapMemoryUsage().getUsed();
        long heapMax = memoryBean.getHeapMemoryUsage().getMax();
        System.out.printf("Heap memory used: %d bytes (%.2f%% of max)\n", 
                         heapUsed, (double) heapUsed / heapMax * 100);
    }
    
    /**
     * 1.3 Demonstrate primitive vs wrapper types
     */
    public static void demonstrateTypeSystem() {
        System.out.println("\n🔢 1.3 Primitive Types vs Wrapper Classes");
        System.out.println("------------------------------------------");
        
        // Primitive types
        int primitiveInt = 42;
        double primitiveDouble = 3.14;
        boolean primitiveBoolean = true;
        
        // Wrapper classes
        Integer wrapperInt = Integer.valueOf(42);
        Double wrapperDouble = Double.valueOf(3.14);
        Boolean wrapperBoolean = Boolean.valueOf(true);
        
        // Auto-boxing and unboxing
        Integer autoBoxed = 100;        // int -> Integer
        int autoUnboxed = autoBoxed;    // Integer -> int
        
        System.out.println("Primitive int: " + primitiveInt);
        System.out.println("Wrapper Integer: " + wrapperInt);
        System.out.println("Auto-boxed: " + autoBoxed);
        System.out.println("Auto-unboxed: " + autoUnboxed);
        
        // Null handling
        Integer nullableInt = null;
        // int nullPrimitive = nullableInt; // This would cause NullPointerException
        
        // Utility methods
        System.out.println("Integer.parseInt(\"123\"): " + Integer.parseInt("123"));
        System.out.println("Integer.toBinaryString(42): " + Integer.toBinaryString(42));
        System.out.println("Double.isNaN(0.0/0.0): " + Double.isNaN(0.0/0.0));
    }
    
    /**
     * 1.4 Demonstrate string immutability and string pool
     */
    public static void demonstrateStringImmutability() {
        System.out.println("\n🔤 1.4 String Immutability & String Pool");
        System.out.println("------------------------------------------");
        
        // String literal (goes to string pool)
        String s1 = "Hello";
        String s2 = "Hello";
        System.out.println("s1 == s2: " + (s1 == s2));  // true (same reference)
        System.out.println("s1.equals(s2): " + s1.equals(s2));  // true (same content)
        
        // New String object (heap memory)
        String s3 = new String("Hello");
        System.out.println("s1 == s3: " + (s1 == s3));  // false (different references)
        System.out.println("s1.equals(s3): " + s1.equals(s3));  // true (same content)
        
        // String concatenation creates new objects
        String s4 = s1 + " World";
        System.out.println("s4: " + s4);
        System.out.println("s1 unchanged: " + s1);
        
        // StringBuilder for efficient concatenation
        StringBuilder sb = new StringBuilder();
        sb.append("Hello").append(" ").append("World");
        String s5 = sb.toString();
        System.out.println("StringBuilder result: " + s5);
        
        // String methods return new objects
        String upper = s1.toUpperCase();
        String lower = s1.toLowerCase();
        System.out.println("Original: " + s1);
        System.out.println("Uppercase: " + upper);
        System.out.println("Lowercase: " + lower);
        System.out.println("Original unchanged: " + s1);
        
        // String pool optimization
        String s6 = "Hello" + " World";  // Compile-time concatenation
        String s7 = "Hello World";
        System.out.println("s6 == s7: " + (s6 == s7));  // true (same reference)
    }
    
    /**
     * 1.5 Demonstrate performance comparisons
     */
    public static void demonstratePerformanceComparisons() {
        System.out.println("\n⚡ 1.5 Performance Comparisons");
        System.out.println("--------------------------------");
        
        // String concatenation performance
        int iterations = 10000;
        
        // Method 1: String concatenation with +
        long startTime = System.nanoTime();
        String result1 = "";
        for (int i = 0; i < iterations; i++) {
            result1 += "String" + i;
        }
        long time1 = System.nanoTime() - startTime;
        
        // Method 2: StringBuilder
        startTime = System.nanoTime();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < iterations; i++) {
            sb.append("String").append(i);
        }
        String result2 = sb.toString();
        long time2 = System.nanoTime() - startTime;
        
        System.out.printf("String concatenation (+): %d ns\n", time1);
        System.out.printf("StringBuilder: %d ns\n", time2);
        System.out.printf("Performance ratio: %.2fx faster\n", (double) time1 / time2);
        
        // Primitive vs Wrapper performance
        iterations = 10000000;
        
        // Primitive loop
        startTime = System.nanoTime();
        int sum1 = 0;
        for (int i = 0; i < iterations; i++) {
            sum1 += i;
        }
        long primitiveTime = System.nanoTime() - startTime;
        
        // Wrapper loop
        startTime = System.nanoTime();
        Integer sum2 = 0;
        for (Integer i = 0; i < iterations; i++) {
            sum2 += i;
        }
        long wrapperTime = System.nanoTime() - startTime;
        
        System.out.printf("Primitive loop: %d ns\n", primitiveTime);
        System.out.printf("Wrapper loop: %d ns\n", wrapperTime);
        System.out.printf("Performance ratio: %.2fx faster\n", (double) wrapperTime / primitiveTime);
    }
    
    /**
     * 1.6 Demonstrate memory management best practices
     */
    public static void demonstrateMemoryManagement() {
        System.out.println("\n💾 1.6 Memory Management Best Practices");
        System.out.println("----------------------------------------");
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        
        // Memory before creating objects
        long memoryBefore = memoryBean.getHeapMemoryUsage().getUsed();
        System.out.println("Memory before: " + memoryBefore + " bytes");
        
        // Create many objects
        List<String> objects = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            objects.add("Object " + i);
        }
        
        // Memory after creating objects
        long memoryAfter = memoryBean.getHeapMemoryUsage().getUsed();
        System.out.println("Memory after creating objects: " + memoryAfter + " bytes");
        System.out.println("Memory used: " + (memoryAfter - memoryBefore) + " bytes");
        
        // Clear references
        objects.clear();
        objects = null;
        
        // Suggest garbage collection
        System.gc();
        
        // Wait a bit for GC to complete
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Memory after garbage collection
        long memoryAfterGC = memoryBean.getHeapMemoryUsage().getUsed();
        System.out.println("Memory after GC: " + memoryAfterGC + " bytes");
        System.out.println("Memory freed: " + (memoryAfter - memoryAfterGC) + " bytes");
        
        // Best practices demonstration
        System.out.println("\nMemory Management Best Practices:");
        System.out.println("1. Use primitives for performance-critical code");
        System.out.println("2. Use StringBuilder for string concatenation in loops");
        System.out.println("3. Clear references when objects are no longer needed");
        System.out.println("4. Be aware of autoboxing overhead in loops");
        System.out.println("5. Use appropriate collection types for your use case");
    }
    
    // Static variable for demonstration
    private static String staticVariable;
}
