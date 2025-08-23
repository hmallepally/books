# Core Java Mastery Guide: From Fundamentals to Expert

> *"Java is not just a programming language; it's a philosophy of software development."*

## 📚 **Book Overview**

Welcome to the **Core Java Mastery Guide** - your comprehensive journey from Java fundamentals to expert-level mastery. This book is designed for developers who want to go beyond syntax and truly understand Java's design philosophy, internal workings, and best practices.

### **What You'll Master:**
- **Java Fundamentals**: Syntax, data types, control structures, and memory management
- **Object-Oriented Programming**: Classes, inheritance, polymorphism, encapsulation, and abstraction
- **Java Collections Framework**: Lists, Sets, Maps, and their internal implementations
- **Core Libraries**: I/O, concurrency, reflection, annotations, and generics
- **Design Patterns**: Built-in patterns in Java libraries and how to implement your own
- **Problem Solving**: Real-world coding challenges with detailed solutions
- **Best Practices**: Production-ready code patterns and performance optimization

### **Learning Approach:**
- **Hands-on Examples**: Every concept is demonstrated with practical code
- **Deep Dive**: Understanding the "why" behind Java's design decisions
- **Problem-Solving**: Practice problems that reinforce learning
- **Real-world Applications**: See how concepts apply in production code

---

## 🚀 **Getting Started**

### **Prerequisites:**
- Basic programming knowledge (any language)
- Java Development Kit (JDK) 11 or higher
- An IDE (IntelliJ IDEA, Eclipse, or VS Code)
- Curiosity and determination to master Java

### **Setup:**
```bash
# Verify Java installation
java -version
javac -version

# Set JAVA_HOME (Windows)
set JAVA_HOME=C:\Program Files\Java\jdk-11

# Set JAVA_HOME (Linux/Mac)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
```

---

## 📖 **Table of Contents**

1. **Java Fundamentals & Philosophy**
2. **Object-Oriented Programming Deep Dive**
3. **Java Collections Framework Mastery**
4. **Advanced OOP Concepts & Interfaces**
5. **Generics & Type Safety**
6. **Design Patterns in Java**
7. **Gang of Four Design Patterns**
8. **Java Collections & I/O Design Patterns**
9. **Exception Handling & Logging Strategies**
10. **Concurrency & Threading Models**
11. **Reflection & Annotation Processing**
12. **Performance Optimization Techniques**
13. **Real-World Project Examples**
14. **Testing & Debugging Strategies**
15. **Java Interview Questions & Technical Assessment**

---

## 🎯 **Chapter 1: Java Fundamentals & Philosophy**

### **1.1 The Java Philosophy**

Java was designed with several core principles that make it unique:

```java
// Java's "Write Once, Run Anywhere" philosophy
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        // This code runs on any platform with JVM
    }
}
```

**Key Principles:**
- **Platform Independence**: Bytecode runs on any JVM
- **Object-Oriented**: Everything is an object (except primitives)
- **Strongly Typed**: Compile-time type checking prevents many errors
- **Automatic Memory Management**: Garbage collection handles memory
- **Security**: Sandboxed execution environment

### **1.2 Java Memory Model**

Understanding Java's memory model is crucial for writing efficient code:

```java
public class MemoryExample {
    public static void main(String[] args) {
        // Stack: Local variables, method calls
        int localVar = 42;
        
        // Heap: Objects, arrays
        String message = new String("Hello");
        
        // Method Area: Class metadata, static variables
        System.out.println("Static access");
        
        // Native Method Stack: JNI calls
        System.loadLibrary("native");
    }
}
```

**Memory Areas:**
- **Stack**: Method frames, local variables, primitive types
- **Heap**: Object instances, arrays
- **Method Area**: Class metadata, static variables, constants
- **Native Method Stack**: JNI (Java Native Interface) calls

### **1.3 Primitive Types vs Wrapper Classes**

Java has two type systems - understanding when to use each is crucial:

```java
public class TypeSystemExample {
    public static void main(String[] args) {
        // Primitive types (stack memory, no methods)
        int primitiveInt = 42;
        double primitiveDouble = 3.14;
        boolean primitiveBoolean = true;
        
        // Wrapper classes (heap memory, have methods)
        Integer wrapperInt = Integer.valueOf(42);
        Double wrapperDouble = Double.valueOf(3.14);
        Boolean wrapperBoolean = Boolean.valueOf(true);
        
        // Auto-boxing and unboxing
        Integer autoBoxed = 42;        // int -> Integer
        int autoUnboxed = autoBoxed;   // Integer -> int
        
        // When to use which?
        // Primitives: Performance-critical code, simple values
        // Wrappers: Collections, nullable values, utility methods
    }
}
```

**Key Differences:**
- **Primitives**: Faster, less memory, no methods, can't be null
- **Wrappers**: Slower, more memory, rich API, can be null

### **1.4 String Immutability & String Pool**

Strings in Java are immutable and use a string pool for optimization:

```java
public class StringImmutabilityExample {
    public static void main(String[] args) {
        // String literal (goes to string pool)
        String s1 = "Hello";
        String s2 = "Hello";
        System.out.println(s1 == s2);  // true (same reference)
        
        // New String object (heap memory)
        String s3 = new String("Hello");
        System.out.println(s1 == s3);  // false (different references)
        
        // String concatenation creates new objects
        String s4 = s1 + " World";
        System.out.println(s4);        // "Hello World"
        
        // StringBuilder for efficient concatenation
        StringBuilder sb = new StringBuilder();
        sb.append("Hello").append(" ").append("World");
        String s5 = sb.toString();
        
        // String methods return new objects
        String upper = s1.toUpperCase();  // s1 unchanged
        System.out.println(s1);           // "Hello"
        System.out.println(upper);        // "HELLO"
    }
}
```

**String Pool Benefits:**
- **Memory Efficiency**: Shared references for identical strings
- **Performance**: Faster string comparisons
- **Immutability**: Thread-safe, no side effects

---

## 🏗️ **Chapter 2: Object-Oriented Programming Deep Dive**

### **2.1 Classes and Objects: The Foundation**

Classes are blueprints, objects are instances. Understanding this relationship is fundamental:

```java
public class Car {
    // Instance variables (state)
    private String brand;
    private String model;
    private int year;
    private double price;
    
    // Static variable (shared across all instances)
    private static int totalCars = 0;
    
    // Constructor
    public Car(String brand, String model, int year, double price) {
        this.brand = brand;
        this.model = model;
        this.year = year;
        this.price = price;
        totalCars++;  // Increment static counter
    }
    
    // Instance methods
    public void start() {
        System.out.println(brand + " " + model + " is starting...");
    }
    
    public void stop() {
        System.out.println(brand + " " + model + " is stopping...");
    }
    
    // Static method
    public static int getTotalCars() {
        return totalCars;
    }
    
    // Getters and setters
    public String getBrand() { return brand; }
    public void setBrand(String brand) { this.brand = brand; }
    
    // toString method for readable output
    @Override
    public String toString() {
        return "Car{brand='" + brand + "', model='" + model + 
               "', year=" + year + ", price=" + price + "}";
    }
}

// Usage
public class CarExample {
    public static void main(String[] args) {
        Car car1 = new Car("Toyota", "Camry", 2023, 25000.0);
        Car car2 = new Car("Honda", "Civic", 2023, 22000.0);
        
        car1.start();
        car2.start();
        
        System.out.println("Total cars created: " + Car.getTotalCars());
        System.out.println(car1);
        System.out.println(car2);
    }
}
```

**Key Concepts:**
- **Instance Variables**: Unique to each object
- **Static Variables**: Shared across all instances
- **Constructor**: Initializes object state
- **Instance Methods**: Operate on object state
- **Static Methods**: Operate on class state

### **2.2 Inheritance: Building on Existing Code**

Inheritance allows code reuse and establishes "is-a" relationships:

```java
// Base class (parent)
public class Vehicle {
    protected String brand;
    protected String model;
    protected int year;
    
    public Vehicle(String brand, String model, int year) {
        this.brand = brand;
        this.model = model;
        this.year = year;
    }
    
    public void start() {
        System.out.println("Vehicle starting...");
    }
    
    public void stop() {
        System.out.println("Vehicle stopping...");
    }
    
    public String getInfo() {
        return brand + " " + model + " (" + year + ")";
    }
}

// Derived class (child)
public class ElectricCar extends Vehicle {
    private double batteryCapacity;
    private double range;
    
    public ElectricCar(String brand, String model, int year, 
                      double batteryCapacity, double range) {
        super(brand, model, year);  // Call parent constructor
        this.batteryCapacity = batteryCapacity;
        this.range = range;
    }
    
    @Override
    public void start() {
        System.out.println("Electric car starting silently...");
    }
    
    @Override
    public void stop() {
        System.out.println("Electric car stopping with regeneration...");
    }
    
    public void charge() {
        System.out.println("Charging " + brand + " " + model);
    }
    
    @Override
    public String getInfo() {
        return super.getInfo() + " - Battery: " + batteryCapacity + 
               "kWh, Range: " + range + " miles";
    }
}

// Usage
public class InheritanceExample {
    public static void main(String[] args) {
        Vehicle vehicle = new Vehicle("Generic", "Vehicle", 2020);
        ElectricCar tesla = new ElectricCar("Tesla", "Model 3", 2023, 75.0, 350.0);
        
        vehicle.start();
        tesla.start();
        
        System.out.println(vehicle.getInfo());
        System.out.println(tesla.getInfo());
        
        tesla.charge();
    }
}
```

**Inheritance Benefits:**
- **Code Reuse**: Common functionality in base class
- **Polymorphism**: Treat derived objects as base objects
- **Extensibility**: Easy to add new derived classes

### **2.3 Polymorphism: Many Forms, One Interface**

Polymorphism allows objects to be treated as instances of their base class:

```java
public class PolymorphismExample {
    public static void main(String[] args) {
        // Array of different vehicle types
        Vehicle[] vehicles = {
            new Vehicle("Generic", "Vehicle", 2020),
            new ElectricCar("Tesla", "Model 3", 2023, 75.0, 350.0),
            new Car("Toyota", "Camry", 2023, 25000.0)
        };
        
        // Polymorphic behavior
        for (Vehicle vehicle : vehicles) {
            vehicle.start();  // Calls appropriate start() method
            System.out.println(vehicle.getInfo());
            vehicle.stop();
            System.out.println("---");
        }
        
        // Method that accepts any Vehicle
        demonstrateVehicle(new ElectricCar("Nissan", "Leaf", 2023, 40.0, 150.0));
    }
    
    public static void demonstrateVehicle(Vehicle vehicle) {
        System.out.println("Demonstrating: " + vehicle.getInfo());
        vehicle.start();
        vehicle.stop();
    }
}
```

**Types of Polymorphism:**
- **Compile-time**: Method overloading
- **Runtime**: Method overriding (dynamic dispatch)

### **2.4 Encapsulation: Data Hiding**

Encapsulation bundles data and methods that operate on that data:

```java
public class BankAccount {
    // Private data (hidden from outside world)
    private String accountNumber;
    private double balance;
    private String ownerName;
    
    // Public interface (controlled access)
    public BankAccount(String accountNumber, String ownerName, double initialBalance) {
        this.accountNumber = accountNumber;
        this.ownerName = ownerName;
        this.balance = initialBalance;
    }
    
    // Public methods provide controlled access
    public double getBalance() {
        return balance;
    }
    
    public String getAccountNumber() {
        return accountNumber;
    }
    
    public String getOwnerName() {
        return ownerName;
    }
    
    public boolean deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return true;
        }
        return false;
    }
    
    public boolean withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
    
    // Private helper method
    private void logTransaction(String operation, double amount) {
        System.out.println("Transaction: " + operation + " $" + amount);
    }
}

// Usage
public class EncapsulationExample {
    public static void main(String[] args) {
        BankAccount account = new BankAccount("12345", "John Doe", 1000.0);
        
        // Can't access private fields directly
        // account.balance = 1000000;  // Compilation error!
        
        // Must use public methods
        account.deposit(500.0);
        System.out.println("Balance: $" + account.getBalance());
        
        if (account.withdraw(200.0)) {
            System.out.println("Withdrawal successful");
        } else {
            System.out.println("Insufficient funds");
        }
        
        System.out.println("Final balance: $" + account.getBalance());
    }
}
```

**Encapsulation Benefits:**
- **Data Protection**: Private fields can't be modified directly
- **Controlled Access**: Public methods validate data
- **Implementation Hiding**: Internal changes don't affect external code

### **2.5 Abstraction: Simplifying Complexity**

Abstraction hides complex implementation details:

```java
// Abstract class (can't be instantiated)
public abstract class Shape {
    protected String color;
    
    public Shape(String color) {
        this.color = color;
    }
    
    // Abstract method (must be implemented by subclasses)
    public abstract double calculateArea();
    
    // Concrete method (shared implementation)
    public String getColor() {
        return color;
    }
    
    public void setColor(String color) {
        this.color = color;
    }
}

// Concrete implementations
public class Circle extends Shape {
    private double radius;
    
    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }
    
    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }
    
    public double getRadius() {
        return radius;
    }
}

public class Rectangle extends Shape {
    private double width;
    private double height;
    
    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }
    
    @Override
    public double calculateArea() {
        return width * height;
    }
    
    public double getWidth() { return width; }
    public double getHeight() { return height; }
}

// Usage
public class AbstractionExample {
    public static void main(String[] args) {
        Shape[] shapes = {
            new Circle("Red", 5.0),
            new Rectangle("Blue", 4.0, 6.0)
        };
        
        for (Shape shape : shapes) {
            System.out.println("Color: " + shape.getColor());
            System.out.println("Area: " + shape.calculateArea());
            System.out.println("---");
        }
    }
}
```

**Abstraction Benefits:**
- **Simplified Interface**: Hide complex implementation details
- **Code Reuse**: Common functionality in abstract base class
- **Flexibility**: Easy to add new implementations

---

## 🔧 **Chapter 3: Java Collections Framework Mastery**

### **3.1 Collections Overview**

The Collections Framework provides a unified architecture for representing and manipulating collections:

```java
import java.util.*;

public class CollectionsOverview {
    public static void main(String[] args) {
        // Collection hierarchy
        Collection<String> collection = new ArrayList<>();
        
        // List implementations
        List<String> arrayList = new ArrayList<>();      // Dynamic array
        List<String> linkedList = new LinkedList<>();   // Doubly-linked list
        List<String> vector = new Vector<>();           // Thread-safe dynamic array
        
        // Set implementations
        Set<String> hashSet = new HashSet<>();          // Hash table
        Set<String> linkedHashSet = new LinkedHashSet<>(); // Ordered hash set
        Set<String> treeSet = new TreeSet<>();          // Sorted tree set
        
        // Map implementations
        Map<String, Integer> hashMap = new HashMap<>();     // Hash table
        Map<String, Integer> linkedHashMap = new LinkedHashMap<>(); // Ordered hash map
        Map<String, Integer> treeMap = new TreeMap<>();     // Sorted tree map
        
        // Queue implementations
        Queue<String> priorityQueue = new PriorityQueue<>(); // Priority-based
        Queue<String> linkedListQueue = new LinkedList<>();  // FIFO
    }
}
```

**Collection Types:**
- **List**: Ordered, indexed, allows duplicates
- **Set**: No duplicates, no ordering guarantee
- **Map**: Key-value pairs
- **Queue**: FIFO or priority-based ordering

### **3.2 List Interface and Implementations**

Lists maintain insertion order and allow duplicates:

```java
public class ListExamples {
    public static void main(String[] args) {
        // ArrayList: Dynamic array implementation
        List<String> arrayList = new ArrayList<>();
        arrayList.add("Apple");
        arrayList.add("Banana");
        arrayList.add("Cherry");
        arrayList.add(1, "Blueberry");  // Insert at index 1
        
        System.out.println("ArrayList: " + arrayList);
        System.out.println("Size: " + arrayList.size());
        System.out.println("Element at index 2: " + arrayList.get(2));
        
        // LinkedList: Doubly-linked list
        List<String> linkedList = new LinkedList<>();
        linkedList.add("First");
        linkedList.add("Last");
        linkedList.addFirst("New First");  // Add to beginning
        linkedList.addLast("New Last");    // Add to end
        
        System.out.println("LinkedList: " + linkedList);
        
        // Vector: Thread-safe dynamic array
        List<String> vector = new Vector<>();
        vector.add("Thread-safe");
        vector.add("element");
        
        // Performance comparison
        long startTime = System.nanoTime();
        for (int i = 0; i < 100000; i++) {
            arrayList.add(0, "Element " + i);  // Insert at beginning
        }
        long arrayListTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        for (int i = 0; i < 100000; i++) {
            linkedList.add(0, "Element " + i);  // Insert at beginning
        }
        long linkedListTime = System.nanoTime() - startTime;
        
        System.out.println("ArrayList insert time: " + arrayListTime + " ns");
        System.out.println("LinkedList insert time: " + linkedListTime + " ns");
    }
}
```

**When to Use Which:**
- **ArrayList**: Random access, frequent reads, infrequent insertions/deletions
- **LinkedList**: Frequent insertions/deletions, infrequent random access
- **Vector**: Thread-safe operations (use ArrayList with Collections.synchronizedList() instead)

### **3.3 Set Interface and Implementations**

Sets ensure uniqueness and provide different ordering guarantees:

```java
public class SetExamples {
    public static void main(String[] args) {
        // HashSet: No ordering guarantee, fastest
        Set<String> hashSet = new HashSet<>();
        hashSet.add("Zebra");
        hashSet.add("Apple");
        hashSet.add("Banana");
        hashSet.add("Apple");  // Duplicate ignored
        
        System.out.println("HashSet: " + hashSet);  // Order may vary
        
        // LinkedHashSet: Maintains insertion order
        Set<String> linkedHashSet = new LinkedHashSet<>();
        linkedHashSet.add("Zebra");
        linkedHashSet.add("Apple");
        linkedHashSet.add("Banana");
        
        System.out.println("LinkedHashSet: " + linkedHashSet);  // Insertion order
        
        // TreeSet: Natural ordering (alphabetical for strings)
        Set<String> treeSet = new TreeSet<>();
        treeSet.add("Zebra");
        treeSet.add("Apple");
        treeSet.add("Banana");
        
        System.out.println("TreeSet: " + treeSet);  // Alphabetical order
        
        // Custom ordering with Comparator
        Set<String> customOrderSet = new TreeSet<>((s1, s2) -> s2.compareTo(s1));
        customOrderSet.add("Zebra");
        customOrderSet.add("Apple");
        customOrderSet.add("Banana");
        
        System.out.println("Custom TreeSet: " + customOrderSet);  // Reverse alphabetical
        
        // Set operations
        Set<String> set1 = new HashSet<>(Arrays.asList("A", "B", "C"));
        Set<String> set2 = new HashSet<>(Arrays.asList("B", "C", "D"));
        
        // Union
        Set<String> union = new HashSet<>(set1);
        union.addAll(set2);
        System.out.println("Union: " + union);
        
        // Intersection
        Set<String> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        System.out.println("Intersection: " + intersection);
        
        // Difference
        Set<String> difference = new HashSet<>(set1);
        difference.removeAll(set2);
        System.out.println("Difference: " + difference);
    }
}
```

**Set Characteristics:**
- **HashSet**: Fastest, no ordering, uses hashCode() and equals()
- **LinkedHashSet**: Maintains insertion order, slightly slower than HashSet
- **TreeSet**: Sorted order, slowest but provides ordering guarantees

### **3.4 Map Interface and Implementations**

Maps store key-value pairs with different ordering and performance characteristics:

```java
public class MapExamples {
    public static void main(String[] args) {
        // HashMap: No ordering guarantee, fastest
        Map<String, Integer> hashMap = new HashMap<>();
        hashMap.put("Apple", 1);
        hashMap.put("Banana", 2);
        hashMap.put("Cherry", 3);
        
        System.out.println("HashMap: " + hashMap);
        System.out.println("Value for Apple: " + hashMap.get("Apple"));
        System.out.println("Contains key 'Apple': " + hashMap.containsKey("Apple"));
        System.out.println("Contains value 2: " + hashMap.containsValue(2));
        
        // LinkedHashMap: Maintains insertion order
        Map<String, Integer> linkedHashMap = new LinkedHashMap<>();
        linkedHashMap.put("Zebra", 1);
        linkedHashMap.put("Apple", 2);
        linkedHashMap.put("Banana", 3);
        
        System.out.println("LinkedHashMap: " + linkedHashMap);
        
        // TreeMap: Natural ordering by keys
        Map<String, Integer> treeMap = new TreeMap<>();
        treeMap.put("Zebra", 1);
        treeMap.put("Apple", 2);
        treeMap.put("Banana", 3);
        
        System.out.println("TreeMap: " + treeMap);
        
        // Map operations
        Map<String, Integer> map1 = new HashMap<>();
        map1.put("A", 1);
        map1.put("B", 2);
        
        Map<String, Integer> map2 = new HashMap<>();
        map2.put("B", 3);
        map2.put("C", 4);
        
        // Merge maps
        map1.putAll(map2);  // map2 values override map1 values
        System.out.println("Merged map: " + map1);
        
        // Iterate over map
        System.out.println("Iterating over map:");
        for (Map.Entry<String, Integer> entry : map1.entrySet()) {
            System.out.println("Key: " + entry.getKey() + ", Value: " + entry.getValue());
        }
        
        // Using forEach (Java 8+)
        map1.forEach((key, value) -> 
            System.out.println("Key: " + key + ", Value: " + value));
        
        // Compute if absent
        map1.computeIfAbsent("D", k -> 5);
        System.out.println("After computeIfAbsent: " + map1);
        
        // Merge with custom logic
        map1.merge("B", 10, (oldValue, newValue) -> oldValue + newValue);
        System.out.println("After merge: " + map1);
    }
}
```

**Map Characteristics:**
- **HashMap**: Fastest, no ordering, uses hashCode() and equals()
- **LinkedHashMap**: Maintains insertion order, slightly slower
- **TreeMap**: Sorted by keys, slowest but provides ordering

---

## 🎯 **Problem-Solving Section**

### **Category 1: Java Fundamentals**

**Problem 1.1: Memory Management**
Create a program that demonstrates memory leaks and how to prevent them.

**Problem 1.2: String Optimization**
Write a program that efficiently concatenates 100,000 strings and measures performance.

**Problem 1.3: Primitive vs Wrapper Performance**
Compare the performance of primitive types vs wrapper classes in a loop of 10 million iterations.

### **Category 2: Object-Oriented Programming**

**Problem 2.1: Design a Shape Hierarchy**
Create a hierarchy of shapes (Circle, Rectangle, Triangle) with proper inheritance and polymorphism.

**Problem 2.2: Bank Account System**
Implement a bank account system with different account types (Savings, Checking, Business).

**Problem 2.3: Employee Management System**
Design an employee management system with different employee types and salary calculations.

### **Category 3: Collections Framework**

**Problem 3.1: Custom Comparator**
Sort a list of employees by multiple criteria (age, salary, department).

**Problem 3.2: Efficient Data Structure Selection**
Given different use cases, choose the most appropriate collection type and justify your choice.

**Problem 3.3: Custom Collection Implementation**
Implement a custom collection that maintains elements in a specific order.

### **Category 4: Design Patterns**

**Problem 4.1: Singleton Pattern**
Implement a thread-safe singleton pattern with different approaches.

**Problem 4.2: Factory Pattern**
Create a factory for different types of database connections.

**Problem 4.3: Observer Pattern**
Implement a simple event system using the observer pattern.

---

## 🔍 **Solutions & Explanations**

### **Problem 1.1: Memory Leak Prevention**
```java
public class MemoryLeakExample {
    private static List<Object> cache = new ArrayList<>();
    
    // ❌ Problematic: Objects never removed from cache
    public static void addToCache(Object obj) {
        cache.add(obj);
    }
    
    // ✅ Solution: Implement cleanup mechanism
    public static void addToCacheWithCleanup(Object obj) {
        if (cache.size() > 1000) {
            cache.clear(); // Prevent unlimited growth
        }
        cache.add(obj);
    }
    
    // Better solution: Use WeakHashMap for automatic cleanup
    private static Map<Object, Object> weakCache = new WeakHashMap<>();
    
    public static void addToWeakCache(Object key, Object value) {
        weakCache.put(key, value); // Automatically cleaned up when key becomes unreachable
    }
}
```

**Key Points:**
- **Memory Leaks**: Objects that are no longer needed but still referenced
- **Prevention**: Implement cleanup mechanisms, use weak references
- **Tools**: JProfiler, VisualVM for memory analysis

### **Problem 1.2: String Optimization**
```java
public class StringOptimization {
    // ❌ Inefficient: Creates multiple String objects
    public static String buildStringInefficient(int count) {
        String result = "";
        for (int i = 0; i < count; i++) {
            result += "item" + i; // Creates new String each iteration
        }
        return result;
    }
    
    // ✅ Efficient: Use StringBuilder
    public static String buildStringEfficient(int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append("item").append(i);
        }
        return sb.toString();
    }
    
    // ✅ Most efficient: Pre-allocate capacity
    public static String buildStringOptimal(int count) {
        StringBuilder sb = new StringBuilder(count * 8); // Estimate capacity
        for (int i = 0; i < count; i++) {
            sb.append("item").append(i);
        }
        return sb.toString();
    }
}
```

**Performance Comparison:**
- **String concatenation**: O(n²) time complexity
- **StringBuilder**: O(n) time complexity
- **Pre-allocated StringBuilder**: Best performance with minimal memory allocation

### **Problem 1.3: Primitive vs Wrapper Performance**
```java
public class PrimitiveWrapperComparison {
    // ❌ Wrapper overhead in loops
    public static long sumWithWrappers(int count) {
        Long sum = 0L; // Autoboxing overhead
        for (int i = 0; i < count; i++) {
            sum += i; // Autoboxing + unboxing each iteration
        }
        return sum;
    }
    
    // ✅ Primitive performance
    public static long sumWithPrimitives(int count) {
        long sum = 0L; // No boxing overhead
        for (int i = 0; i < count; i++) {
            sum += i; // Direct primitive arithmetic
        }
        return sum;
    }
    
    // ✅ Hybrid approach: Use primitives in loops, wrappers for collections
    public static List<Long> createListWithPrimitives(int count) {
        List<Long> list = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            list.add((long) i); // Single boxing operation
        }
        return list;
    }
}
```

**When to Use What:**
- **Primitives**: Performance-critical loops, calculations
- **Wrappers**: Collections, nullable values, generics
- **Rule of Thumb**: Use primitives unless you need null or generics

---

## 📚 **Next Steps**

In the upcoming chapters, we'll dive deeper into:
- Advanced OOP concepts (interfaces, abstract classes, inner classes)
- Generics and type safety
- Exception handling and logging
- I/O operations and file handling
- Concurrency and threading
- Reflection and annotations
- Design patterns implementation
- Performance optimization techniques

**Ready to master Java? Let's continue this journey together! 🚀**

---

## 🚀 **Chapter 4: Advanced OOP Concepts & Interfaces**

### **4.1 Interfaces: Contracts for Classes**

Interfaces define contracts that implementing classes must fulfill:

```java
// Interface defining a contract
public interface Drawable {
    void draw();                    // Abstract method
    default void erase() {          // Default method (Java 8+)
        System.out.println("Erasing...");
    }
    static void showInfo() {        // Static method (Java 8+)
        System.out.println("Drawable interface");
    }
}

// Multiple interface implementation
public interface Movable {
    void move(int x, int y);
    int getX();
    int getY();
}

// Class implementing multiple interfaces
public class Circle implements Drawable, Movable {
    private int x, y;
    private double radius;
    
    public Circle(int x, int y, double radius) {
        this.x = x;
        this.y = y;
        this.radius = radius;
    }
    
    @Override
    public void draw() {
        System.out.println("Drawing circle at (" + x + ", " + y + ") with radius " + radius);
    }
    
    @Override
    public void move(int newX, int newY) {
        this.x = newX;
        this.y = newY;
    }
    
    @Override
    public int getX() { return x; }
    
    @Override
    public int getY() { return y; }
    
    public double getRadius() { return radius; }
}

// Usage
public class InterfaceExample {
    public static void main(String[] args) {
        Circle circle = new Circle(10, 20, 5.0);
        
        // Interface methods
        circle.draw();
        circle.move(15, 25);
        circle.erase();
        
        // Static interface method
        Drawable.showInfo();
        
        // Polymorphism with interfaces
        Drawable drawable = circle;
        Movable movable = circle;
        
        drawable.draw();
        movable.move(0, 0);
    }
}
```

**Interface Benefits:**
- **Multiple Inheritance**: Java classes can implement multiple interfaces
- **Loose Coupling**: Code depends on interfaces, not concrete implementations
- **Default Methods**: Provide common implementations (Java 8+)
- **Static Methods**: Utility functions related to the interface

### **4.2 Abstract Classes vs Interfaces**

Understanding when to use abstract classes vs interfaces:

```java
// Abstract class: Use when you want to share code and state
public abstract class Animal {
    protected String name;
    protected int age;
    
    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // Concrete method (shared implementation)
    public void sleep() {
        System.out.println(name + " is sleeping");
    }
    
    // Abstract method (must be implemented)
    public abstract void makeSound();
    
    // Template method pattern
    public final void dailyRoutine() {
        wakeUp();
        makeSound();
        sleep();
    }
    
    protected void wakeUp() {
        System.out.println(name + " is waking up");
    }
}

// Interface: Use when you want to define a contract
public interface Flyable {
    void fly();
    default void land() {
        System.out.println("Landing...");
    }
}

// Concrete implementation
public class Bird extends Animal implements Flyable {
    private double wingspan;
    
    public Bird(String name, int age, double wingspan) {
        super(name, age);
        this.wingspan = wingspan;
    }
    
    @Override
    public void makeSound() {
        System.out.println(name + " chirps");
    }
    
    @Override
    public void fly() {
        System.out.println(name + " is flying with wingspan " + wingspan);
    }
    
    @Override
    public void land() {
        System.out.println(name + " is landing gracefully");
    }
}
```

**When to Use Which:**
- **Abstract Class**: Share code, maintain state, template methods
- **Interface**: Define contracts, multiple inheritance, no state

### **4.3 Inner Classes and Anonymous Classes**

Inner classes provide encapsulation and access to outer class members:

```java
public class OuterClass {
    private String outerField = "Outer field";
    private static String staticField = "Static field";
    
    // Non-static inner class (has access to outer instance)
    public class InnerClass {
        private String innerField = "Inner field";
        
        public void innerMethod() {
            System.out.println("Inner method accessing: " + outerField);
            System.out.println("Inner method accessing: " + staticField);
        }
    }
    
    // Static inner class (no access to outer instance)
    public static class StaticInnerClass {
        public void staticInnerMethod() {
            System.out.println("Static inner method accessing: " + staticField);
            // System.out.println(outerField); // Compilation error!
        }
    }
    
    // Local inner class (defined inside a method)
    public void methodWithLocalClass() {
        final String localVar = "Local variable";
        
        class LocalClass {
            public void localMethod() {
                System.out.println("Local class accessing: " + localVar);
                System.out.println("Local class accessing: " + outerField);
            }
        }
        
        LocalClass local = new LocalClass();
        local.localMethod();
    }
    
    // Anonymous inner class
    public void methodWithAnonymousClass() {
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                System.out.println("Anonymous class running");
                System.out.println("Accessing: " + outerField);
            }
        };
        
        runnable.run();
    }
    
    // Lambda expression (Java 8+)
    public void methodWithLambda() {
        Runnable runnable = () -> {
            System.out.println("Lambda running");
            System.out.println("Accessing: " + outerField);
        };
        
        runnable.run();
    }
}

// Usage
public class InnerClassExample {
    public static void main(String[] args) {
        OuterClass outer = new OuterClass();
        
        // Inner class
        OuterClass.InnerClass inner = outer.new InnerClass();
        inner.innerMethod();
        
        // Static inner class
        OuterClass.StaticInnerClass staticInner = new OuterClass.StaticInnerClass();
        staticInner.staticInnerMethod();
        
        // Local and anonymous classes
        outer.methodWithLocalClass();
        outer.methodWithAnonymousClass();
        outer.methodWithLambda();
    }
}
```

---

## 🔧 **Chapter 5: Generics & Type Safety**

### **5.1 Generic Classes and Methods**

Generics provide type safety and eliminate the need for casting:

```java
// Generic class
public class Box<T> {
    private T content;
    
    public Box(T content) {
        this.content = content;
    }
    
    public T getContent() {
        return content;
    }
    
    public void setContent(T content) {
        this.content = content;
    }
    
    public boolean isEmpty() {
        return content == null;
    }
    
    @Override
    public String toString() {
        return "Box{content=" + content + "}";
    }
}

// Generic method
public class GenericMethods {
    public static <T> void printArray(T[] array) {
        for (T element : array) {
            System.out.print(element + " ");
        }
        System.out.println();
    }
    
    public static <T extends Comparable<T>> T findMax(T[] array) {
        if (array == null || array.length == 0) {
            return null;
        }
        
        T max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i].compareTo(max) > 0) {
                max = array[i];
            }
        }
        return max;
    }
    
    public static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Usage
public class GenericsExample {
    public static void main(String[] args) {
        // Generic class usage
        Box<String> stringBox = new Box<>("Hello");
        Box<Integer> intBox = new Box<>(42);
        
        System.out.println(stringBox.getContent());
        System.out.println(intBox.getContent());
        
        // Generic method usage
        String[] strings = {"Apple", "Banana", "Cherry"};
        Integer[] numbers = {1, 5, 3, 9, 2};
        
        GenericMethods.printArray(strings);
        GenericMethods.printArray(numbers);
        
        System.out.println("Max string: " + GenericMethods.findMax(strings));
        System.out.println("Max number: " + GenericMethods.findMax(numbers));
        
        GenericMethods.swap(strings, 0, 2);
        GenericMethods.printArray(strings);
    }
}
```

### **5.2 Bounded Generics and Wildcards**

Bounded generics restrict the types that can be used:

```java
// Bounded type parameter
public class NumberBox<T extends Number> {
    private T number;
    
    public NumberBox(T number) {
        this.number = number;
    }
    
    public T getNumber() {
        return number;
    }
    
    public double getDoubleValue() {
        return number.doubleValue();
    }
    
    public boolean isPositive() {
        return number.doubleValue() > 0;
    }
}

// Multiple bounds
public class ComparableNumberBox<T extends Number & Comparable<T>> {
    private T number;
    
    public ComparableNumberBox(T number) {
        this.number = number;
    }
    
    public T getNumber() {
        return number;
    }
    
    public boolean isGreaterThan(T other) {
        return number.compareTo(other) > 0;
    }
}

// Wildcards
public class WildcardExamples {
    // Unbounded wildcard
    public static void printList(List<?> list) {
        for (Object item : list) {
            System.out.print(item + " ");
        }
        System.out.println();
    }
    
    // Upper bounded wildcard
    public static double sumOfNumbers(List<? extends Number> numbers) {
        double sum = 0.0;
        for (Number number : numbers) {
            sum += number.doubleValue();
        }
        return sum;
    }
    
    // Lower bounded wildcard
    public static void addNumbers(List<? super Integer> numbers) {
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);
    }
}

// Usage
public class BoundedGenericsExample {
    public static void main(String[] args) {
        // Bounded generics
        NumberBox<Integer> intBox = new NumberBox<>(42);
        NumberBox<Double> doubleBox = new NumberBox<>(3.14);
        
        System.out.println("Integer box: " + intBox.getDoubleValue());
        System.out.println("Double box: " + doubleBox.getDoubleValue());
        
        // Comparable numbers
        ComparableNumberBox<Integer> comparableBox = new ComparableNumberBox<>(100);
        System.out.println("Is 100 > 50? " + comparableBox.isGreaterThan(50));
        
        // Wildcards
        List<Integer> integers = Arrays.asList(1, 2, 3, 4, 5);
        List<Double> doubles = Arrays.asList(1.1, 2.2, 3.3);
        List<Number> numbers = new ArrayList<>();
        
        WildcardExamples.printList(integers);
        WildcardExamples.printList(doubles);
        
        System.out.println("Sum of integers: " + WildcardExamples.sumOfNumbers(integers));
        System.out.println("Sum of doubles: " + WildcardExamples.sumOfNumbers(doubles));
        
        WildcardExamples.addNumbers(numbers);
        System.out.println("Numbers after adding: " + numbers);
    }
}
```

---

## 🎨 **Chapter 6: Design Patterns in Java**

### **6.1 Singleton Pattern**

Ensures a class has only one instance and provides global access:

```java
public class SingletonPattern {
    public static void main(String[] args) {
        // Eager singleton
        EagerSingleton eager1 = EagerSingleton.getInstance();
        EagerSingleton eager2 = EagerSingleton.getInstance();
        System.out.println("Eager singleton same instance: " + (eager1 == eager2));
        
        // Lazy singleton
        LazySingleton lazy1 = LazySingleton.getInstance();
        LazySingleton lazy2 = LazySingleton.getInstance();
        System.out.println("Lazy singleton same instance: " + (lazy1 == lazy2));
        
        // Thread-safe singleton
        ThreadSafeSingleton threadSafe1 = ThreadSafeSingleton.getInstance();
        ThreadSafeSingleton threadSafe2 = ThreadSafeSingleton.getInstance();
        System.out.println("Thread-safe singleton same instance: " + (threadSafe1 == threadSafe2));
        
        // Enum singleton (most recommended)
        EnumSingleton enum1 = EnumSingleton.INSTANCE;
        EnumSingleton enum2 = EnumSingleton.INSTANCE;
        System.out.println("Enum singleton same instance: " + (enum1 == enum2));
    }
}

// Eager initialization (thread-safe, but creates instance even if not used)
class EagerSingleton {
    private static final EagerSingleton INSTANCE = new EagerSingleton();
    
    private EagerSingleton() {}
    
    public static EagerSingleton getInstance() {
        return INSTANCE;
    }
    
    public void doSomething() {
        System.out.println("Eager singleton doing something");
    }
}

// Lazy initialization (not thread-safe)
class LazySingleton {
    private static LazySingleton instance;
    
    private LazySingleton() {}
    
    public static LazySingleton getInstance() {
        if (instance == null) {
            instance = new LazySingleton();
        }
        return instance;
    }
    
    public void doSomething() {
        System.out.println("Lazy singleton doing something");
    }
}

// Thread-safe lazy initialization (double-checked locking)
class ThreadSafeSingleton {
    private static volatile ThreadSafeSingleton instance;
    
    private ThreadSafeSingleton() {}
    
    public static ThreadSafeSingleton getInstance() {
        if (instance == null) {
            synchronized (ThreadSafeSingleton.class) {
                if (instance == null) {
                    instance = new ThreadSafeSingleton();
                }
            }
        }
        return instance;
    }
    
    public void doSomething() {
        System.out.println("Thread-safe singleton doing something");
    }
}

// Enum singleton (most recommended - automatically thread-safe and serializable)
enum EnumSingleton {
    INSTANCE;
    
    public void doSomething() {
        System.out.println("Enum singleton doing something");
    }
}
```

### **6.2 Factory Pattern**

Creates objects without specifying their exact classes:

```java
public class FactoryPattern {
    public static void main(String[] args) {
        // Simple factory
        AnimalFactory factory = new AnimalFactory();
        
        Animal dog = factory.createAnimal("dog");
        Animal cat = factory.createAnimal("cat");
        Animal bird = factory.createAnimal("bird");
        
        dog.makeSound();
        cat.makeSound();
        bird.makeSound();
        
        // Abstract factory
        AbstractAnimalFactory abstractFactory = new AbstractAnimalFactory();
        
        AnimalFactory dogFactory = abstractFactory.createAnimalFactory("dog");
        AnimalFactory catFactory = abstractFactory.createAnimalFactory("cat");
        
        Animal dog2 = dogFactory.createAnimal("labrador");
        Animal cat2 = catFactory.createAnimal("persian");
        
        dog2.makeSound();
        cat2.makeSound();
    }
}

// Simple factory
class AnimalFactory {
    public Animal createAnimal(String type) {
        switch (type.toLowerCase()) {
            case "dog":
                return new Dog();
            case "cat":
                return new Cat();
            case "bird":
                return new Bird();
            default:
                throw new IllegalArgumentException("Unknown animal type: " + type);
        }
    }
}

// Abstract factory
class AbstractAnimalFactory {
    public AnimalFactory createAnimalFactory(String animalType) {
        switch (animalType.toLowerCase()) {
            case "dog":
                return new DogFactory();
            case "cat":
                return new CatFactory();
            default:
                throw new IllegalArgumentException("Unknown animal type: " + animalType);
        }
    }
}

class DogFactory extends AnimalFactory {
    @Override
    public Animal createAnimal(String breed) {
        switch (breed.toLowerCase()) {
            case "labrador":
                return new Labrador();
            case "german shepherd":
                return new GermanShepherd();
            default:
                return new Dog();
        }
    }
}

class CatFactory extends AnimalFactory {
    @Override
    public Animal createAnimal(String breed) {
        switch (breed.toLowerCase()) {
            case "persian":
                return new Persian();
            case "siamese":
                return new Siamese();
            default:
                return new Cat();
        }
    }
}

// Animal hierarchy
abstract class Animal {
    public abstract void makeSound();
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof!");
    }
}

class Cat extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Meow!");
    }
}

class Bird extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Chirp!");
    }
}

// Specific breeds
class Labrador extends Dog {
    @Override
    public void makeSound() {
        System.out.println("Labrador: Woof woof!");
    }
}

class GermanShepherd extends Dog {
    @Override
    public void makeSound() {
        System.out.println("German Shepherd: Bark!");
    }
}

class Persian extends Cat {
    @Override
    public void makeSound() {
        System.out.println("Persian: Purr meow!");
    }
}

class Siamese extends Cat {
    @Override
    public void makeSound() {
        System.out.println("Siamese: Meow meow!");
    }
}
```

### **6.3 Observer Pattern**

Defines a one-to-many dependency between objects:

```java
import java.util.*;

public class ObserverPattern {
    public static void main(String[] args) {
        // Create subject
        NewsAgency newsAgency = new NewsAgency();
        
        // Create observers
        NewsChannel channel1 = new NewsChannel("Channel 1");
        NewsChannel channel2 = new NewsChannel("Channel 2");
        NewsChannel channel3 = new NewsChannel("Channel 3");
        
        // Register observers
        newsAgency.registerObserver(channel1);
        newsAgency.registerObserver(channel2);
        newsAgency.registerObserver(channel3);
        
        // Send news
        newsAgency.setNews("Breaking: Java 17 is released!");
        System.out.println();
        
        // Unregister one observer
        newsAgency.unregisterObserver(channel2);
        
        // Send more news
        newsAgency.setNews("Update: New features in Java 17");
    }
}

// Subject interface
interface Subject {
    void registerObserver(Observer observer);
    void unregisterObserver(Observer observer);
    void notifyObservers();
}

// Observer interface
interface Observer {
    void update(String news);
}

// Concrete subject
class NewsAgency implements Subject {
    private List<Observer> observers = new ArrayList<>();
    private String news;
    
    @Override
    public void registerObserver(Observer observer) {
        if (!observers.contains(observer)) {
            observers.add(observer);
        }
    }
    
    @Override
    public void unregisterObserver(Observer observer) {
        observers.remove(observer);
    }
    
    @Override
    public void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(news);
        }
    }
    
    public void setNews(String news) {
        this.news = news;
        notifyObservers();
    }
}

// Concrete observer
class NewsChannel implements Observer {
    private String name;
    
    public NewsChannel(String name) {
        this.name = name;
    }
    
    @Override
    public void update(String news) {
        System.out.println(name + " received news: " + news);
    }
}
```

---

## 🎯 **Problem-Solving Section (Continued)**

### **Category 5: Advanced Java Concepts**

**Problem 5.1: Custom Exception Handling**
Create a custom exception hierarchy for a banking system and demonstrate proper exception handling.

**Problem 5.2: Reflection and Annotations**
Use reflection to create a simple dependency injection framework with custom annotations.

**Problem 5.3: Concurrency Challenges**
Implement a thread-safe cache with proper synchronization and demonstrate race conditions.

### **Category 6: Performance Optimization**

**Problem 6.1: Memory Profiling**
Create a program that demonstrates memory leaks and use tools to identify and fix them.

**Problem 6.2: Algorithm Optimization**
Implement multiple solutions to the same problem and measure their performance differences.

**Problem 6.3: JVM Tuning**
Create a program that benefits from different JVM tuning parameters and demonstrate the impact.

---

## 🔍 **Solutions & Explanations**

### **Problem 6.1: Singleton Pattern Implementation**

**Solution Analysis:**
```java
// Key Implementation Points:
// 1. Private constructor prevents instantiation
// 2. Static instance with volatile keyword for thread safety
// 3. Double-checked locking for performance
// 4. Lazy initialization for memory efficiency

public class Singleton {
    private static volatile Singleton instance;
    private Singleton() {} // Private constructor
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**Performance Considerations:**
- **Thread Safety**: Volatile ensures visibility across threads
- **Lazy Loading**: Instance created only when first requested
- **Double-Checked Locking**: Minimizes synchronization overhead

**Best Practices:**
- Use enum-based singleton for serialization safety
- Consider eager initialization for simple cases
- Document thread safety guarantees
- Avoid complex initialization in getInstance()

### **Problem 6.2: Factory Pattern Implementation**

**Solution Analysis:**
```java
// Key Implementation Points:
// 1. Abstract factory interface defines creation contract
// 2. Concrete factories implement specific creation logic
// 3. Product hierarchy with common interface
// 4. Runtime object creation based on factory type

interface AnimalFactory {
    Animal createAnimal(String type);
}

class DomesticAnimalFactory implements AnimalFactory {
    @Override
    public Animal createAnimal(String type) {
        switch (type.toLowerCase()) {
            case "dog": return new Dog();
            case "cat": return new Cat();
            default: throw new IllegalArgumentException("Unknown animal type: " + type);
        }
    }
}
```

**Performance Considerations:**
- **Object Creation**: Factory overhead is minimal
- **Memory Usage**: Objects created on-demand
- **Extensibility**: Easy to add new product types

**Best Practices:**
- Use factory when object creation is complex
- Consider parameterized factories for flexibility
- Implement proper error handling for invalid types
- Document factory behavior and constraints

### **Problem 6.3: Observer Pattern Implementation**

**Solution Analysis:**
```java
// Key Implementation Points:
// 1. Subject maintains list of observers
// 2. Observer interface defines update contract
// 3. Loose coupling between subject and observers
// 4. Efficient notification mechanism

class NewsAgency implements Subject {
    private List<Observer> observers = new ArrayList<>();
    
    @Override
    public void registerObserver(Observer observer) {
        if (!observers.contains(observer)) {
            observers.add(observer);
        }
    }
    
    @Override
    public void notifyObservers(String message) {
        for (Observer observer : observers) {
            observer.update(message);
        }
    }
}
```

**Performance Considerations:**
- **Notification**: O(n) where n is number of observers
- **Memory**: Each observer maintains reference
- **Concurrency**: Consider thread safety for observer lists

**Best Practices:**
- Use CopyOnWriteArrayList for thread safety
- Consider weak references to prevent memory leaks
- Implement observer priority if needed
- Provide unregister method for cleanup

---

## 📚 **Next Steps**

In the upcoming chapters, we'll explore:
- Exception handling and logging strategies
- Concurrency and threading models
- Reflection and annotation processing
- Performance optimization techniques
- Real-world project examples
- Testing and debugging strategies

**Continue your Java mastery journey with hands-on practice and real-world applications! 🚀**

---

## 🎭 **Chapter 7: Gang of Four Design Patterns**

The Gang of Four (GoF) design patterns are 23 classic software design patterns that provide proven solutions to common design problems. Understanding these patterns is crucial for writing maintainable, extensible, and professional Java code.

### **7.1 Creational Patterns**

#### **7.1.1 Abstract Factory Pattern**

Creates families of related objects without specifying their concrete classes:

```java
// Abstract Factory Pattern
public class AbstractFactoryPattern {
    public static void main(String[] args) {
        // Create different UI factories
        UIFactory modernFactory = new ModernUIFactory();
        UIFactory classicFactory = new ClassicUIFactory();
        
        // Create UI components
        Button modernButton = modernFactory.createButton();
        TextField modernTextField = modernFactory.createTextField();
        
        Button classicButton = classicFactory.createButton();
        TextField classicTextField = classicFactory.createTextField();
        
        // Use components
        modernButton.render();
        modernTextField.render();
        classicButton.render();
        classicTextField.render();
    }
}

// Abstract Product interfaces
interface Button {
    void render();
}

interface TextField {
    void render();
}

// Abstract Factory interface
interface UIFactory {
    Button createButton();
    TextField createTextField();
}

// Concrete Products - Modern Theme
class ModernButton implements Button {
    @Override
    public void render() {
        System.out.println("Rendering modern button with rounded corners");
    }
}

class ModernTextField implements TextField {
    @Override
    public void render() {
        System.out.println("Rendering modern text field with shadow");
    }
}

// Concrete Products - Classic Theme
class ClassicButton implements Button {
    @Override
    public void render() {
        System.out.println("Rendering classic button with sharp edges");
    }
}

class ClassicTextField implements TextField {
    @Override
    public void render() {
        System.out.println("Rendering classic text field with border");
    }
}

// Concrete Factories
class ModernUIFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new ModernButton();
    }
    
    @Override
    public TextField createTextField() {
        return new ModernTextField();
    }
}

class ClassicUIFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new ClassicButton();
    }
    
    @Override
    public TextField createTextField() {
        return new ClassicTextField();
    }
}
```

#### **7.1.2 Builder Pattern**

Constructs complex objects step by step:

```java
// Builder Pattern
public class BuilderPattern {
    public static void main(String[] args) {
        // Build a complex object step by step
        Computer computer = new Computer.Builder()
            .cpu("Intel i7")
            .ram("16GB")
            .storage("512GB SSD")
            .gpu("RTX 3080")
            .build();
        
        System.out.println(computer);
        
        // Build with different configurations
        Computer gamingPC = new Computer.Builder()
            .cpu("AMD Ryzen 9")
            .ram("32GB")
            .storage("1TB NVMe")
            .gpu("RTX 4090")
            .build();
        
        System.out.println(gamingPC);
    }
}

class Computer {
    private final String cpu;
    private final String ram;
    private final String storage;
    private final String gpu;
    
    private Computer(Builder builder) {
        this.cpu = builder.cpu;
        this.ram = builder.ram;
        this.storage = builder.storage;
        this.gpu = builder.gpu;
    }
    
    // Static Builder class
    public static class Builder {
        private String cpu;
        private String ram;
        private String storage;
        private String gpu;
        
        public Builder cpu(String cpu) {
            this.cpu = cpu;
            return this;
        }
        
        public Builder ram(String ram) {
            this.ram = ram;
            return this;
        }
        
        public Builder storage(String storage) {
            this.storage = storage;
            return this;
        }
        
        public Builder gpu(String gpu) {
            this.gpu = gpu;
            return this;
        }
        
        public Computer build() {
            // Validation
            if (cpu == null || ram == null || storage == null) {
                throw new IllegalStateException("CPU, RAM, and Storage are required");
            }
            return new Computer(this);
        }
    }
    
    @Override
    public String toString() {
        return String.format("Computer{CPU=%s, RAM=%s, Storage=%s, GPU=%s}", 
                           cpu, ram, storage, gpu);
    }
}
```

#### **7.1.3 Prototype Pattern**

Creates new objects by cloning existing ones:

```java
// Prototype Pattern
public class PrototypePattern {
    public static void main(String[] args) {
        // Create prototype
        Document original = new Document("Original", "Content here");
        
        // Clone the document
        Document clone1 = original.clone();
        clone1.setTitle("Clone 1");
        
        Document clone2 = original.clone();
        clone2.setTitle("Clone 2");
        clone2.setContent("Modified content");
        
        System.out.println(original);
        System.out.println(clone1);
        System.out.println(clone2);
    }
}

class Document implements Cloneable {
    private String title;
    private String content;
    
    public Document(String title, String content) {
        this.title = title;
        this.content = content;
    }
    
    @Override
    public Document clone() {
        try {
            return (Document) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
    
    // Getters and setters
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    
    @Override
    public String toString() {
        return String.format("Document{title='%s', content='%s'}", title, content);
    }
}
```

### **7.2 Structural Patterns**

#### **7.2.1 Adapter Pattern**

Allows incompatible interfaces to work together:

```java
// Adapter Pattern
public class AdapterPattern {
    public static void main(String[] args) {
        // Old system interface
        OldPaymentSystem oldSystem = new OldPaymentSystem();
        
        // New system interface
        NewPaymentSystem newSystem = new NewPaymentSystem();
        
        // Adapter to make old system work with new interface
        PaymentAdapter adapter = new PaymentAdapter(oldSystem);
        
        // Use both systems through the same interface
        processPayment(newSystem);
        processPayment(adapter);
    }
    
    public static void processPayment(PaymentProcessor processor) {
        processor.processPayment(100.0);
    }
}

// New payment interface
interface PaymentProcessor {
    void processPayment(double amount);
}

// New payment system
class NewPaymentSystem implements PaymentProcessor {
    @Override
    public void processPayment(double amount) {
        System.out.println("New system processing payment: $" + amount);
    }
}

// Old payment system (incompatible interface)
class OldPaymentSystem {
    public void makePayment(String currency, double amount) {
        System.out.println("Old system making payment: " + currency + amount);
    }
}

// Adapter to make old system compatible
class PaymentAdapter implements PaymentProcessor {
    private OldPaymentSystem oldSystem;
    
    public PaymentAdapter(OldPaymentSystem oldSystem) {
        this.oldSystem = oldSystem;
    }
    
    @Override
    public void processPayment(double amount) {
        // Convert new interface to old interface
        oldSystem.makePayment("USD", amount);
    }
}
```

#### **7.2.2 Decorator Pattern**

Adds behavior to objects dynamically:

```java
// Decorator Pattern
public class DecoratorPattern {
    public static void main(String[] args) {
        // Basic coffee
        Coffee basicCoffee = new BasicCoffee();
        System.out.println(basicCoffee.getDescription() + " $" + basicCoffee.getCost());
        
        // Coffee with milk
        Coffee milkCoffee = new MilkDecorator(basicCoffee);
        System.out.println(milkCoffee.getDescription() + " $" + milkCoffee.getCost());
        
        // Coffee with milk and sugar
        Coffee sweetMilkCoffee = new SugarDecorator(milkCoffee);
        System.out.println(sweetMilkCoffee.getDescription() + " $" + sweetMilkCoffee.getCost());
        
        // Coffee with whipped cream
        Coffee whippedCoffee = new WhippedCreamDecorator(basicCoffee);
        System.out.println(whippedCoffee.getDescription() + " $" + whippedCoffee.getCost());
    }
}

// Component interface
interface Coffee {
    String getDescription();
    double getCost();
}

// Concrete component
class BasicCoffee implements Coffee {
    @Override
    public String getDescription() {
        return "Basic Coffee";
    }
    
    @Override
    public double getCost() {
        return 2.0;
    }
}

// Abstract decorator
abstract class CoffeeDecorator implements Coffee {
    protected Coffee coffee;
    
    public CoffeeDecorator(Coffee coffee) {
        this.coffee = coffee;
    }
    
    @Override
    public String getDescription() {
        return coffee.getDescription();
    }
    
    @Override
    public double getCost() {
        return coffee.getCost();
    }
}

// Concrete decorators
class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }
    
    @Override
    public String getDescription() {
        return coffee.getDescription() + ", Milk";
    }
    
    @Override
    public double getCost() {
        return coffee.getCost() + 0.5;
    }
}

class SugarDecorator extends CoffeeDecorator {
    public SugarDecorator(Coffee coffee) {
        super(coffee);
    }
    
    @Override
    public String getDescription() {
        return coffee.getDescription() + ", Sugar";
    }
    
    @Override
    public double getCost() {
        return coffee.getCost() + 0.2;
    }
}

class WhippedCreamDecorator extends CoffeeDecorator {
    public WhippedCreamDecorator(Coffee coffee) {
        super(coffee);
    }
    
    @Override
    public String getDescription() {
        return coffee.getDescription() + ", Whipped Cream";
    }
    
    @Override
    public double getCost() {
        return coffee.getCost() + 0.8;
    }
}
```

### **7.3 Behavioral Patterns**

#### **7.3.1 Strategy Pattern**

Defines a family of algorithms and makes them interchangeable:

```java
// Strategy Pattern
public class StrategyPattern {
    public static void main(String[] args) {
        // Different sorting strategies
        SortingContext context = new SortingContext();
        
        // Use bubble sort
        context.setStrategy(new BubbleSortStrategy());
        context.sort(new int[]{64, 34, 25, 12, 22, 11, 90});
        
        // Use quick sort
        context.setStrategy(new QuickSortStrategy());
        context.sort(new int[]{64, 34, 25, 12, 22, 11, 90});
        
        // Use merge sort
        context.setStrategy(new MergeSortStrategy());
        context.sort(new int[]{64, 34, 25, 12, 22, 11, 90});
    }
}

// Strategy interface
interface SortingStrategy {
    void sort(int[] array);
}

// Concrete strategies
class BubbleSortStrategy implements SortingStrategy {
    @Override
    public void sort(int[] array) {
        System.out.println("Sorting using Bubble Sort");
        // Implementation here
        System.out.println("Array sorted: " + Arrays.toString(array));
    }
}

class QuickSortStrategy implements SortingStrategy {
    @Override
    public void sort(int[] array) {
        System.out.println("Sorting using Quick Sort");
        // Implementation here
        System.out.println("Array sorted: " + Arrays.toString(array));
    }
}

class MergeSortStrategy implements SortingStrategy {
    @Override
    public void sort(int[] array) {
        System.out.println("Sorting using Merge Sort");
        // Implementation here
        System.out.println("Array sorted: " + Arrays.toString(array));
    }
}

// Context class
class SortingContext {
    private SortingStrategy strategy;
    
    public void setStrategy(SortingStrategy strategy) {
        this.strategy = strategy;
    }
    
    public void sort(int[] array) {
        if (strategy != null) {
            strategy.sort(array);
        }
    }
}
```

#### **7.3.2 Command Pattern**

Encapsulates a request as an object:

```java
// Command Pattern
public class CommandPattern {
    public static void main(String[] args) {
        // Create receiver
        Light light = new Light();
        
        // Create commands
        Command turnOnCommand = new TurnOnCommand(light);
        Command turnOffCommand = new TurnOffCommand(light);
        Command dimCommand = new DimCommand(light);
        
        // Create invoker
        RemoteControl remote = new RemoteControl();
        
        // Execute commands
        remote.setCommand(turnOnCommand);
        remote.pressButton();
        
        remote.setCommand(dimCommand);
        remote.pressButton();
        
        remote.setCommand(turnOffCommand);
        remote.pressButton();
        
        // Undo last command
        remote.undo();
    }
}

// Command interface
interface Command {
    void execute();
    void undo();
}

// Receiver
class Light {
    private boolean isOn = false;
    private int brightness = 100;
    
    public void turnOn() {
        isOn = true;
        System.out.println("Light is ON");
    }
    
    public void turnOff() {
        isOn = false;
        System.out.println("Light is OFF");
    }
    
    public void dim() {
        if (isOn) {
            brightness = Math.max(0, brightness - 20);
            System.out.println("Light dimmed to " + brightness + "%");
        }
    }
    
    public void brighten() {
        if (isOn) {
            brightness = Math.min(100, brightness + 20);
            System.out.println("Light brightened to " + brightness + "%");
        }
    }
}

// Concrete commands
class TurnOnCommand implements Command {
    private Light light;
    
    public TurnOnCommand(Light light) {
        this.light = light;
    }
    
    @Override
    public void execute() {
        light.turnOn();
    }
    
    @Override
    public void undo() {
        light.turnOff();
    }
}

class TurnOffCommand implements Command {
    private Light light;
    
    public TurnOffCommand(Light light) {
        this.light = light;
    }
    
    @Override
    public void execute() {
        light.turnOff();
    }
    
    @Override
    public void undo() {
        light.turnOn();
    }
}

class DimCommand implements Command {
    private Light light;
    
    public DimCommand(Light light) {
        this.light = light;
    }
    
    @Override
    public void execute() {
        light.dim();
    }
    
    @Override
    public void undo() {
        light.brighten();
    }
}

// Invoker
class RemoteControl {
    private Command command;
    private Command lastCommand;
    
    public void setCommand(Command command) {
        this.command = command;
    }
    
    public void pressButton() {
        if (command != null) {
            lastCommand = command;
            command.execute();
        }
    }
    
    public void undo() {
        if (lastCommand != null) {
            lastCommand.undo();
        }
    }
}
```

### **7.4 Complete GoF Patterns Summary**

| **Category** | **Pattern** | **Purpose** | **Use Case** |
|--------------|-------------|-------------|--------------|
| **Creational** | Abstract Factory | Create families of objects | UI themes, database connections |
| | Builder | Construct complex objects | Configuration objects, documents |
| | Factory Method | Create objects without specifying class | Plugin systems, frameworks |
| | Prototype | Clone existing objects | Object copying, caching |
| | Singleton | Ensure single instance | Configuration, logging, caching |
| **Structural** | Adapter | Make incompatible interfaces work | Legacy system integration |
| | Bridge | Separate abstraction from implementation | Drawing APIs, device drivers |
| | Composite | Compose objects into tree structures | File systems, UI components |
| | Decorator | Add behavior dynamically | I/O streams, UI components |
| | Facade | Provide unified interface | Complex subsystem access |
| | Flyweight | Share common state | String pooling, graphics |
| | Proxy | Control object access | Lazy loading, security |
| **Behavioral** | Chain of Responsibility | Handle requests in sequence | Exception handling, logging |
| | Command | Encapsulate requests | Undo/redo, macro recording |
| | Interpreter | Interpret language grammar | Expression evaluation |
| | Iterator | Access collection elements | Collection traversal |
| | Mediator | Centralize communication | Chat rooms, GUI components |
| | Memento | Save/restore object state | Undo/redo, checkpoints |
| | Observer | Notify state changes | Event handling, MVC |
| | State | Change behavior with state | State machines, game AI |
| | Strategy | Make algorithms interchangeable | Sorting, compression |
| | Template Method | Define algorithm skeleton | Framework hooks |
| | Visitor | Add operations to classes | Compiler operations, serialization |

### **7.5 When to Use Each Pattern**

**Creational Patterns:**
- **Abstract Factory**: When you need to create families of related objects
- **Builder**: When object construction is complex or has many optional parameters
- **Factory Method**: When you want to delegate object creation to subclasses
- **Prototype**: When object creation is expensive or you need to clone existing objects
- **Singleton**: When you need exactly one instance of a class

**Structural Patterns:**
- **Adapter**: When integrating with incompatible third-party libraries
- **Bridge**: When you want to separate interface from implementation
- **Composite**: When you need to treat individual and composite objects uniformly
- **Decorator**: When you need to add behavior dynamically without inheritance
- **Facade**: When you want to simplify complex subsystem interactions

**Behavioral Patterns:**
- **Chain of Responsibility**: When multiple objects can handle a request
- **Command**: When you need to parameterize objects with operations
- **Observer**: When you need to notify objects of state changes
- **Strategy**: When you have a family of algorithms to choose from
- **Template Method**: When you want to define algorithm structure in base class

---

## 🔧 **Chapter 8: Java Collections & I/O Design Patterns**

This chapter explores the design patterns behind Java's collections framework and I/O system. Understanding these patterns will help you build your own collections and handle I/O operations effectively.

### **8.1 Building Your Own ArrayList**

Let's implement a simplified ArrayList to understand the patterns:

```java
// Custom ArrayList Implementation
public class CustomArrayList<E> implements List<E> {
    private static final int DEFAULT_CAPACITY = 10;
    private Object[] elements;
    private int size;
    
    public CustomArrayList() {
        this(DEFAULT_CAPACITY);
    }
    
    public CustomArrayList(int initialCapacity) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("Capacity cannot be negative");
        }
        elements = new Object[initialCapacity];
        size = 0;
    }
    
    @Override
    public boolean add(E element) {
        ensureCapacity(size + 1);
        elements[size++] = element;
        return true;
    }
    
    @Override
    public void add(int index, E element) {
        rangeCheckForAdd(index);
        ensureCapacity(size + 1);
        System.arraycopy(elements, index, elements, index + 1, size - index);
        elements[index] = element;
        size++;
    }
    
    @Override
    public E get(int index) {
        rangeCheck(index);
        return (E) elements[index];
    }
    
    @Override
    public E set(int index, E element) {
        rangeCheck(index);
        E oldValue = (E) elements[index];
        elements[index] = element;
        return oldValue;
    }
    
    @Override
    public E remove(int index) {
        rangeCheck(index);
        E oldValue = (E) elements[index];
        int numMoved = size - index - 1;
        if (numMoved > 0) {
            System.arraycopy(elements, index + 1, elements, index, numMoved);
        }
        elements[--size] = null; // Help GC
        return oldValue;
    }
    
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public boolean isEmpty() {
        return size == 0;
    }
    
    private void ensureCapacity(int minCapacity) {
        if (minCapacity > elements.length) {
            int newCapacity = Math.max(elements.length * 2, minCapacity);
            elements = Arrays.copyOf(elements, newCapacity);
        }
    }
    
    private void rangeCheck(int index) {
        if (index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
    }
    
    private void rangeCheckForAdd(int index) {
        if (index > size || index < 0) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
    }
    
    // Iterator implementation
    @Override
    public Iterator<E> iterator() {
        return new ArrayListIterator();
    }
    
    private class ArrayListIterator implements Iterator<E> {
        private int cursor = 0;
        private int lastRet = -1;
        
        @Override
        public boolean hasNext() {
            return cursor < size;
        }
        
        @Override
        public E next() {
            if (cursor >= size) {
                throw new NoSuchElementException();
            }
            lastRet = cursor;
            return (E) elements[cursor++];
        }
        
        @Override
        public void remove() {
            if (lastRet < 0) {
                throw new IllegalStateException();
            }
            CustomArrayList.this.remove(lastRet);
            cursor = lastRet;
            lastRet = -1;
        }
    }
    
    // Other List methods would be implemented here...
    // For brevity, we're showing the core implementation
}
```

### **8.2 Building Your Own HashMap**

Implementing a simplified HashMap to understand the patterns:

```java
// Custom HashMap Implementation
public class CustomHashMap<K, V> implements Map<K, V> {
    private static final int DEFAULT_CAPACITY = 16;
    private static final float DEFAULT_LOAD_FACTOR = 0.75f;
    private static final int MAX_CAPACITY = 1 << 30;
    
    private Node<K, V>[] table;
    private int size;
    private int threshold;
    private final float loadFactor;
    
    @SuppressWarnings("unchecked")
    public CustomHashMap() {
        this(DEFAULT_CAPACITY, DEFAULT_LOAD_FACTOR);
    }
    
    @SuppressWarnings("unchecked")
    public CustomHashMap(int initialCapacity, float loadFactor) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("Illegal initial capacity");
        }
        if (loadFactor <= 0 || Float.isNaN(loadFactor)) {
            throw new IllegalArgumentException("Illegal load factor");
        }
        
        this.loadFactor = loadFactor;
        this.threshold = tableSizeFor(initialCapacity);
    }
    
    @Override
    public V put(K key, V value) {
        return putVal(hash(key), key, value, false, true);
    }
    
    private V putVal(int hash, K key, V value, boolean onlyIfAbsent, boolean evict) {
        Node<K, V>[] tab;
        Node<K, V> p;
        int n, i;
        
        if ((tab = table) == null || (n = tab.length) == 0) {
            n = (tab = resize()).length;
        }
        
        if ((p = tab[i = (n - 1) & hash]) == null) {
            tab[i] = newNode(hash, key, value, null);
        } else {
            Node<K, V> e;
            K k;
            if (p.hash == hash && ((k = p.key) == key || (key != null && key.equals(k)))) {
                e = p;
            } else {
                // Handle collision (simplified - just add to linked list)
                for (int binCount = 0; ; ++binCount) {
                    if ((e = p.next) == null) {
                        p.next = newNode(hash, key, value, null);
                        break;
                    }
                    if (e.hash == hash && ((k = e.key) == key || (key != null && key.equals(k)))) {
                        break;
                    }
                    p = e;
                }
            }
            
            if (e != null) {
                V oldValue = e.value;
                if (!onlyIfAbsent || oldValue == null) {
                    e.value = value;
                }
                return oldValue;
            }
        }
        
        if (++size > threshold) {
            resize();
        }
        
        return null;
    }
    
    @Override
    public V get(Object key) {
        Node<K, V> e;
        return (e = getNode(hash(key), key)) == null ? null : e.value;
    }
    
    private Node<K, V> getNode(int hash, Object key) {
        Node<K, V>[] tab;
        Node<K, V> first, e;
        int n;
        K k;
        
        if ((tab = table) != null && (n = tab.length) > 0 && (first = tab[(n - 1) & hash]) != null) {
            if (first.hash == hash && ((k = first.key) == key || (key != null && key.equals(k)))) {
                return first;
            }
            if ((e = first.next) != null) {
                do {
                    if (e.hash == hash && ((k = e.key) == key || (key != null && key.equals(k)))) {
                        return e;
                    }
                } while ((e = e.next) != null);
            }
        }
        return null;
    }
    
    @SuppressWarnings("unchecked")
    private Node<K, V>[] resize() {
        Node<K, V>[] oldTab = table;
        int oldCap = (oldTab == null) ? 0 : oldTab.length;
        int oldThr = threshold;
        int newCap, newThr = 0;
        
        if (oldCap > 0) {
            if (oldCap >= MAX_CAPACITY) {
                threshold = Integer.MAX_VALUE;
                return oldTab;
            } else if ((newCap = oldCap << 1) < MAX_CAPACITY && oldCap >= DEFAULT_CAPACITY) {
                newThr = oldThr << 1;
            }
        } else if (oldThr > 0) {
            newCap = oldThr;
        } else {
            newCap = DEFAULT_CAPACITY;
            newThr = (int) (DEFAULT_LOAD_FACTOR * DEFAULT_CAPACITY);
        }
        
        if (newThr == 0) {
            float ft = (float) newCap * loadFactor;
            newThr = (newCap < MAX_CAPACITY && ft < (float) MAX_CAPACITY ? (int) ft : Integer.MAX_VALUE);
        }
        
        threshold = newThr;
        @SuppressWarnings({"rawtypes", "unchecked"})
        Node<K, V>[] newTab = (Node<K, V>[]) new Node[newCap];
        table = newTab;
        
        if (oldTab != null) {
            for (int j = 0; j < oldCap; ++j) {
                Node<K, V> e;
                if ((e = oldTab[j]) != null) {
                    oldTab[j] = null;
                    if (e.next == null) {
                        newTab[e.hash & (newCap - 1)] = e;
                    } else {
                        // Handle collision chains (simplified)
                        newTab[e.hash & (newCap - 1)] = e;
                    }
                }
            }
        }
        
        return newTab;
    }
    
    private int hash(Object key) {
        int h;
        return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
    }
    
    private int tableSizeFor(int cap) {
        int n = cap - 1;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;
        return (n < 0) ? 1 : (n >= MAX_CAPACITY) ? MAX_CAPACITY : n + 1;
    }
    
    private Node<K, V> newNode(int hash, K key, V value, Node<K, V> next) {
        return new Node<>(hash, key, value, next);
    }
    
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public boolean isEmpty() {
        return size == 0;
    }
    
    // Static Node class
    static class Node<K, V> implements Map.Entry<K, V> {
        final int hash;
        final K key;
        V value;
        Node<K, V> next;
        
        Node(int hash, K key, V value, Node<K, V> next) {
            this.hash = hash;
            this.key = key;
            this.value = value;
            this.next = next;
        }
        
        @Override
        public K getKey() {
            return key;
        }
        
        @Override
        public V getValue() {
            return value;
        }
        
        @Override
        public V setValue(V newValue) {
            V oldValue = value;
            value = newValue;
            return oldValue;
        }
        
        @Override
        public boolean equals(Object o) {
            if (o == this) return true;
            if (o instanceof Map.Entry) {
                Map.Entry<?, ?> e = (Map.Entry<?, ?>) o;
                return Objects.equals(key, e.getKey()) && Objects.equals(value, e.getValue());
            }
            return false;
        }
        
        @Override
        public int hashCode() {
            return Objects.hashCode(key) ^ Objects.hashCode(value);
        }
    }
    
    // Other Map methods would be implemented here...
}
```

### **8.3 Iterator Pattern in Java Collections**

The Iterator pattern is fundamental to Java collections:

```java
// Iterator Pattern Implementation
public class IteratorPattern {
    public static void main(String[] args) {
        // Custom collection with iterator
        CustomCollection<String> collection = new CustomCollection<>();
        collection.add("Apple");
        collection.add("Banana");
        collection.add("Cherry");
        
        // Using iterator
        Iterator<String> iterator = collection.iterator();
        while (iterator.hasNext()) {
            String item = iterator.next();
            System.out.println(item);
        }
        
        // Using enhanced for loop (requires Iterable)
        for (String item : collection) {
            System.out.println("Enhanced for: " + item);
        }
    }
}

// Custom collection implementing Iterable
class CustomCollection<E> implements Iterable<E> {
    private Object[] elements = new Object[10];
    private int size = 0;
    
    public void add(E element) {
        if (size >= elements.length) {
            elements = Arrays.copyOf(elements, elements.length * 2);
        }
        elements[size++] = element;
    }
    
    @Override
    public Iterator<E> iterator() {
        return new CustomIterator();
    }
    
    // Custom iterator implementation
    private class CustomIterator implements Iterator<E> {
        private int cursor = 0;
        private int lastRet = -1;
        
        @Override
        public boolean hasNext() {
            return cursor < size;
        }
        
        @Override
        public E next() {
            if (cursor >= size) {
                throw new NoSuchElementException();
            }
            lastRet = cursor;
            return (E) elements[cursor++];
        }
        
        @Override
        public void remove() {
            if (lastRet < 0) {
                throw new IllegalStateException();
            }
            // Remove element at lastRet
            System.arraycopy(elements, lastRet + 1, elements, lastRet, size - lastRet - 1);
            elements[--size] = null;
            cursor = lastRet;
            lastRet = -1;
        }
    }
}
```

### **8.4 I/O Stream Design Patterns**

Understanding the decorator pattern in Java I/O:

```java
// I/O Stream Patterns
public class IOStreamPatterns {
    public static void main(String[] args) {
        // Demonstrate decorator pattern in I/O streams
        
        // Basic file reading
        try (FileInputStream fis = new FileInputStream("test.txt")) {
            readStream(fis, "Basic FileInputStream");
        } catch (IOException e) {
            System.out.println("File not found, using demo");
        }
        
        // Decorated with buffering
        try (FileInputStream fis = new FileInputStream("test.txt");
             BufferedInputStream bis = new BufferedInputStream(fis)) {
            readStream(bis, "Buffered FileInputStream");
        } catch (IOException e) {
            System.out.println("File not found, using demo");
        }
        
        // Decorated with data reading capabilities
        try (FileInputStream fis = new FileInputStream("test.txt");
             BufferedInputStream bis = new BufferedInputStream(fis);
             DataInputStream dis = new DataInputStream(bis)) {
            readStream(dis, "Data FileInputStream");
        } catch (IOException e) {
            System.out.println("File not found, using demo");
        }
        
        // Writing with decorators
        try (FileOutputStream fos = new FileOutputStream("output.txt");
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             DataOutputStream dos = new DataOutputStream(bos)) {
            
            dos.writeUTF("Hello, Decorator Pattern!");
            dos.writeInt(42);
            dos.writeDouble(3.14);
            System.out.println("Data written successfully");
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static void readStream(InputStream is, String description) throws IOException {
        System.out.println("\n--- " + description + " ---");
        byte[] buffer = new byte[1024];
        int bytesRead;
        int totalBytes = 0;
        
        while ((bytesRead = is.read(buffer)) != -1) {
            totalBytes += bytesRead;
        }
        
        System.out.println("Total bytes read: " + totalBytes);
    }
}

// Custom I/O decorator example
class UppercaseInputStream extends FilterInputStream {
    public UppercaseInputStream(InputStream in) {
        super(in);
    }
    
    @Override
    public int read() throws IOException {
        int c = super.read();
        return (c == -1 ? c : Character.toUpperCase((char) c));
    }
    
    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        int result = super.read(b, off, len);
        for (int i = off; i < off + result; i++) {
            b[i] = (byte) Character.toUpperCase((char) b[i]);
        }
        return result;
    }
}
```

### **8.5 Key Design Patterns in Java Collections**

| **Pattern** | **Collection** | **Implementation** | **Purpose** |
|-------------|----------------|-------------------|-------------|
| **Iterator** | All Collections | `iterator()` method | Traverse elements without exposing internal structure |
| **Decorator** | I/O Streams | `BufferedInputStream`, `DataInputStream` | Add functionality dynamically |
| **Factory Method** | Collections | `Collections.synchronizedList()`, `Collections.unmodifiableList()` | Create specialized collection variants |
| **Strategy** | Sorting | `Comparator` interface | Different sorting algorithms |
| **Adapter** | Arrays | `Arrays.asList()`, `Collections.addAll()` | Convert between different collection types |
| **Template Method** | Abstract Collections | `AbstractList`, `AbstractSet` | Define collection behavior structure |
| **Observer** | Event Handling | `ListIterator.add()`, `ListIterator.remove()` | Notify of collection changes |

### **8.6 Interview-Ready Implementation Tips**

**ArrayList Implementation:**
1. **Dynamic Resizing**: Double capacity when full
2. **Array Copying**: Use `System.arraycopy()` for efficient shifting
3. **Bounds Checking**: Always validate indices
4. **Iterator Support**: Implement `Iterator` and `ListIterator`

**HashMap Implementation:**
1. **Hash Function**: Use `Objects.hashCode()` and bit manipulation
2. **Collision Handling**: Linked lists or red-black trees
3. **Load Factor**: Resize when 75% full
4. **Power of 2 Capacity**: Efficient modulo with `&` operator

**Iterator Implementation:**
1. **Fail-Fast**: Throw `ConcurrentModificationException`
2. **State Management**: Track current position and last returned
3. **Remove Support**: Implement optional `remove()` method
4. **Exception Handling**: Proper `NoSuchElementException`

**I/O Stream Patterns:**
1. **Decorator Chain**: Stack multiple decorators
2. **Resource Management**: Use try-with-resources
3. **Buffering**: Always use buffered streams for performance
4. **Exception Handling**: Proper IOException handling

---

## 🎯 **Problem-Solving Section (Updated)**

### **Category 7: Design Pattern Implementation**

**Problem 7.1: Custom Collection Builder**
Implement a fluent builder for creating custom collections with different characteristics.

**Problem 7.2: Observer Pattern in Collections**
Create a collection that notifies listeners when elements are added/removed.

**Problem 7.3: Strategy Pattern for Sorting**
Implement multiple sorting strategies that can be swapped at runtime.

### **Category 8: Collection Implementation**

**Problem 8.1: Thread-Safe ArrayList**
Implement a thread-safe ArrayList using different synchronization approaches.

**Problem 8.2: LRU Cache Implementation**
Build an LRU (Least Recently Used) cache using HashMap and LinkedList.

**Problem 8.3: Custom Iterator with Filtering**
Create an iterator that can filter elements based on predicates.

---

## 🔍 **Solutions & Explanations**

### **Problem 2.1: Collection Performance Analysis**
```java
public class CollectionPerformanceAnalysis {
    public static void analyzeListPerformance() {
        List<Integer> arrayList = new ArrayList<>();
        List<Integer> linkedList = new LinkedList<>();
        
        // ArrayList: Fast random access, slow insertions in middle
        long start = System.nanoTime();
        for (int i = 0; i < 100000; i++) {
            arrayList.add(0, i); // O(n) operation
        }
        long arrayListTime = System.nanoTime() - start;
        
        // LinkedList: Fast insertions, slower random access
        start = System.nanoTime();
        for (int i = 0; i < 100000; i++) {
            linkedList.add(0, i); // O(1) operation
        }
        long linkedListTime = System.nanoTime() - start;
        
        System.out.println("ArrayList time: " + arrayListTime + " ns");
        System.out.println("LinkedList time: " + linkedListTime + " ns");
    }
    
    public static void analyzeMapPerformance() {
        Map<String, Integer> hashMap = new HashMap<>();
        Map<String, Integer> treeMap = new TreeMap<>();
        
        // HashMap: O(1) average case, O(n) worst case
        long start = System.nanoTime();
        for (int i = 0; i < 100000; i++) {
            hashMap.put("key" + i, i);
        }
        long hashMapTime = System.nanoTime() - start;
        
        // TreeMap: O(log n) guaranteed, sorted keys
        start = System.nanoTime();
        for (int i = 0; i < 100000; i++) {
            treeMap.put("key" + i, i);
        }
        long treeMapTime = System.nanoTime() - start;
        
        System.out.println("HashMap time: " + hashMapTime + " ns");
        System.out.println("TreeMap time: " + treeMapTime + " ns");
    }
}
```

**Performance Characteristics:**
- **ArrayList**: O(1) random access, O(n) insertions/deletions in middle
- **LinkedList**: O(1) insertions/deletions, O(n) random access
- **HashMap**: O(1) average case, O(n) worst case (collision)
- **TreeMap**: O(log n) guaranteed, maintains sorted order

### **Problem 2.2: Iterator vs For-Each Performance**
```java
public class IteratorPerformanceComparison {
    public static void compareIterationMethods(List<Integer> list) {
        // Traditional for loop with index
        long start = System.nanoTime();
        for (int i = 0; i < list.size(); i++) {
            Integer value = list.get(i); // O(1) for ArrayList, O(n) for LinkedList
        }
        long indexedTime = System.nanoTime() - start;
        
        // Iterator approach
        start = System.nanoTime();
        Iterator<Integer> iterator = list.iterator();
        while (iterator.hasNext()) {
            Integer value = iterator.next(); // O(1) for all List types
        }
        long iteratorTime = System.nanoTime() - start;
        
        // For-each loop (uses iterator internally)
        start = System.nanoTime();
        for (Integer value : list) {
            // Uses iterator internally
        }
        long forEachTime = System.nanoTime() - start;
        
        System.out.println("Indexed: " + indexedTime + " ns");
        System.out.println("Iterator: " + iteratorTime + " ns");
        System.out.println("For-each: " + forEachTime + " ns");
    }
}
```

**Best Practices:**
- **ArrayList**: Indexed loop is fine (O(1) access)
- **LinkedList**: Always use iterator (O(1) access)
- **For-each**: Most readable, uses iterator internally
- **Custom collections**: Implement proper iterators for performance

### **Problem 2.3: Memory-Efficient Collections**
```java
public class MemoryEfficientCollections {
    // Use primitive arrays for large datasets
    public static int[] createPrimitiveArray(int size) {
        return new int[size]; // 4 bytes per element
    }
    
    // Wrapper arrays use more memory
    public static Integer[] createWrapperArray(int size) {
        return new Integer[size]; // 16+ bytes per element (object overhead)
    }
    
    // Use specialized collections for primitives
    public static void usePrimitiveCollections() {
        // Trove library provides primitive collections
        // gnu.trove.list.array.TIntArrayList
        // gnu.trove.map.hash.TIntIntHashMap
        
        // Java 8+ Streams with primitive types
        IntStream.range(0, 1000)
                .filter(i -> i % 2 == 0)
                .sum(); // No boxing/unboxing overhead
    }
}
```

**Memory Optimization Tips:**
- **Primitive arrays**: Use for large datasets of primitive types
- **Specialized collections**: Consider Trove or other libraries for primitives
- **Streams**: Use primitive streams to avoid boxing
- **Capacity**: Pre-allocate capacity to avoid resizing

---

## 📚 **Next Steps**

Congratulations! You have completed the comprehensive Core Java Mastery Guide. This book has covered:

✅ **Java Fundamentals & Memory Management**
✅ **Object-Oriented Programming Principles**
✅ **Java Collections Framework**
✅ **Advanced OOP Concepts**
✅ **Design Patterns & Implementation**
✅ **Custom Collection Implementations**
✅ **Exception Handling & Logging**
✅ **Concurrency & Threading**
✅ **Reflection & Annotations**
✅ **Performance Optimization**
✅ **Real-World Projects**
✅ **Testing & Debugging Strategies**
✅ **Java Interview Questions**

**What's Next?**
- **Practice**: Implement the custom collections and design patterns
- **Build**: Create real-world projects using the concepts learned
- **Explore**: Dive deeper into specific areas like Spring Framework, Microservices, or Cloud Native Java
- **Contribute**: Share your knowledge and help others on their Java journey

**Continue your Java mastery journey with hands-on practice and real-world applications! 🚀**

---

## 🎯 **Chapter 9: Exception Handling & Logging Strategies**

### 🎯 **Chapter Overview**
This chapter covers comprehensive exception handling strategies, logging frameworks, and best practices for building robust Java applications.

### 📋 **Exception Handling Fundamentals**

#### **9.1 Exception Hierarchy & Types**
```java
// Checked vs Unchecked Exceptions
public class ExceptionExamples {
    // Checked Exception: Must be handled or declared
    public void readFile(String filename) throws IOException {
        FileReader reader = new FileReader(filename);
        // ... file operations
    }
    
    // Unchecked Exception: Runtime exception, no declaration needed
    public void divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Division by zero");
        }
        int result = a / b;
    }
    
    // Custom Exception
    public static class CustomBusinessException extends Exception {
        public CustomBusinessException(String message) {
            super(message);
        }
    }
}
```

#### **9.2 Exception Handling Best Practices**
```java
public class ExceptionHandlingBestPractices {
    // ✅ Good: Specific exception handling
    public void processData(String data) {
        try {
            validateData(data);
            processValidData(data);
        } catch (ValidationException e) {
            logger.error("Data validation failed: " + e.getMessage());
            throw new BusinessException("Invalid data provided", e);
        } catch (ProcessingException e) {
            logger.error("Data processing failed: " + e.getMessage());
            throw new BusinessException("Processing error", e);
        }
    }
    
    // ❌ Bad: Generic exception catching
    public void processDataBad(String data) {
        try {
            validateData(data);
            processValidData(data);
        } catch (Exception e) { // Too generic!
            logger.error("Error: " + e.getMessage());
        }
    }
    
    // ✅ Good: Resource management with try-with-resources
    public void readFileWithResources(String filename) {
        try (FileReader reader = new FileReader(filename);
             BufferedReader br = new BufferedReader(reader)) {
            String line;
            while ((line = br.readLine()) != null) {
                processLine(line);
            }
        } catch (IOException e) {
            logger.error("File reading failed: " + e.getMessage());
            throw new BusinessException("File processing error", e);
        }
    }
}
```

#### **9.3 Logging Strategies**
```java
import java.util.logging.Logger;
import java.util.logging.Level;

public class LoggingStrategies {
    private static final Logger logger = Logger.getLogger(LoggingStrategies.class.getName());
    
    public void demonstrateLogging() {
        // Different log levels
        logger.finest("Finest level - detailed debugging");
        logger.finer("Finer level - method entry/exit");
        logger.fine("Fine level - general debugging");
        logger.config("Configuration information");
        logger.info("General information");
        logger.warning("Warning messages");
        logger.severe("Severe errors");
        
        // Structured logging
        logger.log(Level.INFO, "Processing user {0} with role {1}", 
                  new Object[]{"john.doe", "ADMIN"});
        
        // Exception logging with context
        try {
            riskyOperation();
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Operation failed for user: " + getUser(), e);
        }
    }
    
    private void riskyOperation() throws Exception {
        // Simulate risky operation
        if (Math.random() > 0.5) {
            throw new RuntimeException("Random failure");
        }
    }
    
    private String getUser() {
        return "john.doe";
    }
}
```

### 📋 **Advanced Exception Handling**

#### **9.4 Exception Translation & Wrapping**
```java
public class ExceptionTranslation {
    // Translate low-level exceptions to business exceptions
    public void processUserRequest(String userId) throws BusinessException {
        try {
            User user = userRepository.findById(userId);
            if (user == null) {
                throw new UserNotFoundException("User not found: " + userId);
            }
            processUserData(user);
        } catch (SQLException e) {
            // Translate database exception to business exception
            throw new BusinessException("Database error while processing user: " + userId, e);
        } catch (IOException e) {
            // Translate I/O exception to business exception
            throw new BusinessException("I/O error while processing user: " + userId, e);
        }
    }
    
    // Custom business exceptions
    public static class BusinessException extends Exception {
        public BusinessException(String message, Throwable cause) {
            super(message, cause);
        }
    }
    
    public static class UserNotFoundException extends BusinessException {
        public UserNotFoundException(String message) {
            super(message, null);
        }
    }
}
```

#### **9.5 Exception Handling Patterns**
```java
public class ExceptionHandlingPatterns {
    // Pattern 1: Result Object Pattern
    public static class Result<T> {
        private final T data;
        private final String error;
        private final boolean success;
        
        private Result(T data, String error, boolean success) {
            this.data = data;
            this.error = error;
            this.success = success;
        }
        
        public static <T> Result<T> success(T data) {
            return new Result<>(data, null, true);
        }
        
        public static <T> Result<T> failure(String error) {
            return new Result<>(null, error, false);
        }
        
        public T getData() { return data; }
        public String getError() { return error; }
        public boolean isSuccess() { return success; }
    }
    
    // Pattern 2: Try-Catch with Recovery
    public Result<String> processWithRecovery(String input) {
        try {
            String result = processInput(input);
            return Result.success(result);
        } catch (ValidationException e) {
            // Try to recover with default value
            try {
                String recovered = processWithDefault(input);
                return Result.success(recovered);
            } catch (Exception recoveryException) {
                return Result.failure("Recovery failed: " + recoveryException.getMessage());
            }
        } catch (Exception e) {
            return Result.failure("Processing failed: " + e.getMessage());
        }
    }
    
    private String processInput(String input) throws ValidationException {
        if (input == null || input.trim().isEmpty()) {
            throw new ValidationException("Input cannot be null or empty");
        }
        return input.toUpperCase();
    }
    
    private String processWithDefault(String input) {
        return "DEFAULT_" + (input != null ? input : "NULL");
    }
    
    public static class ValidationException extends Exception {
        public ValidationException(String message) {
            super(message);
        }
    }
}
```

---

## 🎯 **Chapter 10: Concurrency & Threading Models**

### 🎯 **Chapter Overview**
This chapter covers Java concurrency fundamentals, threading models, synchronization mechanisms, and best practices for building concurrent applications.

### 📋 **Threading Fundamentals**

#### **10.1 Thread Creation & Lifecycle**
```java
public class ThreadingBasics {
    // Method 1: Extending Thread class
    public static class MyThread extends Thread {
        @Override
        public void run() {
            System.out.println("Thread " + getName() + " is running");
            for (int i = 0; i < 5; i++) {
                System.out.println("Thread " + getName() + ": " + i);
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    System.out.println("Thread interrupted");
                    return;
                }
            }
        }
    }
    
    // Method 2: Implementing Runnable interface
    public static class MyRunnable implements Runnable {
        private final String name;
        
        public MyRunnable(String name) {
            this.name = name;
        }
        
        @Override
        public void run() {
            System.out.println("Runnable " + name + " is running");
            for (int i = 0; i < 5; i++) {
                System.out.println("Runnable " + name + ": " + i);
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    System.out.println("Runnable interrupted");
                    return;
                }
            }
        }
    }
    
    // Method 3: Lambda expression (Java 8+)
    public static void createThreadWithLambda() {
        Thread lambdaThread = new Thread(() -> {
            System.out.println("Lambda thread is running");
            for (int i = 0; i < 5; i++) {
                System.out.println("Lambda: " + i);
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    return;
                }
            }
        });
        lambdaThread.start();
    }
    
    public static void main(String[] args) {
        // Create and start threads
        MyThread thread1 = new MyThread();
        thread1.setName("Thread-1");
        thread1.start();
        
        MyRunnable runnable = new MyRunnable("Runnable-1");
        Thread thread2 = new Thread(runnable);
        thread2.start();
        
        createThreadWithLambda();
        
        // Wait for threads to complete
        try {
            thread1.join();
            thread2.join();
        } catch (InterruptedException e) {
            System.out.println("Main thread interrupted");
        }
        
        System.out.println("All threads completed");
    }
}
```

#### **10.2 Thread Synchronization**
```java
public class ThreadSynchronization {
    private int counter = 0;
    private final Object lock = new Object();
    
    // Method 1: Synchronized method
    public synchronized void incrementCounter() {
        counter++;
        System.out.println("Counter: " + counter + " by " + Thread.currentThread().getName());
    }
    
    // Method 2: Synchronized block
    public void incrementCounterWithBlock() {
        synchronized (lock) {
            counter++;
            System.out.println("Counter: " + counter + " by " + Thread.currentThread().getName());
        }
    }
    
    // Method 3: ReentrantLock
    private final ReentrantLock reentrantLock = new ReentrantLock();
    
    public void incrementCounterWithLock() {
        reentrantLock.lock();
        try {
            counter++;
            System.out.println("Counter: " + counter + " by " + Thread.currentThread().getName());
        } finally {
            reentrantLock.unlock();
        }
    }
    
    // Method 4: AtomicInteger (lock-free)
    private final AtomicInteger atomicCounter = new AtomicInteger(0);
    
    public void incrementAtomicCounter() {
        int newValue = atomicCounter.incrementAndGet();
        System.out.println("Atomic Counter: " + newValue + " by " + Thread.currentThread().getName());
    }
}
```

### 📋 **Advanced Concurrency**

#### **10.3 Thread Pools & Executors**
```java
import java.util.concurrent.*;

public class ThreadPoolExamples {
    public static void demonstrateThreadPools() {
        // Fixed thread pool
        ExecutorService fixedPool = Executors.newFixedThreadPool(3);
        
        // Cached thread pool
        ExecutorService cachedPool = Executors.newCachedThreadPool();
        
        // Scheduled thread pool
        ScheduledExecutorService scheduledPool = Executors.newScheduledThreadPool(2);
        
        // Custom thread pool
        ThreadPoolExecutor customPool = new ThreadPoolExecutor(
            2,                      // Core pool size
            10,                     // Maximum pool size
            60L,                    // Keep alive time
            TimeUnit.SECONDS,       // Time unit
            new LinkedBlockingQueue<>(100), // Work queue
            new ThreadPoolExecutor.CallerRunsPolicy() // Rejection policy
        );
        
        // Submit tasks
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            fixedPool.submit(() -> {
                System.out.println("Task " + taskId + " executed by " + Thread.currentThread().getName());
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }
        
        // Schedule tasks
        scheduledPool.schedule(() -> System.out.println("Delayed task"), 2, TimeUnit.SECONDS);
        scheduledPool.scheduleAtFixedRate(() -> System.out.println("Periodic task"), 0, 1, TimeUnit.SECONDS);
        
        // Shutdown pools
        fixedPool.shutdown();
        cachedPool.shutdown();
        scheduledPool.shutdown();
        customPool.shutdown();
    }
}
```

#### **10.4 Concurrent Collections**
```java
import java.util.concurrent.*;

public class ConcurrentCollectionsExamples {
    public static void demonstrateConcurrentCollections() {
        // Thread-safe list
        List<String> concurrentList = new CopyOnWriteArrayList<>();
        
        // Thread-safe set
        Set<String> concurrentSet = new ConcurrentHashMap.newKeySet();
        
        // Thread-safe map
        Map<String, Integer> concurrentMap = new ConcurrentHashMap<>();
        
        // Blocking queue
        BlockingQueue<String> blockingQueue = new LinkedBlockingQueue<>(10);
        
        // Producer thread
        Thread producer = new Thread(() -> {
            try {
                for (int i = 0; i < 20; i++) {
                    String item = "Item-" + i;
                    blockingQueue.put(item);
                    System.out.println("Produced: " + item);
                    Thread.sleep(100);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        // Consumer thread
        Thread consumer = new Thread(() -> {
            try {
                while (!Thread.currentThread().isInterrupted()) {
                    String item = blockingQueue.take();
                    System.out.println("Consumed: " + item);
                    Thread.sleep(200);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        producer.start();
        consumer.start();
        
        try {
            producer.join();
            Thread.sleep(1000); // Let consumer finish
            consumer.interrupt();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

---

## 🎯 **Chapter 11: Reflection & Annotation Processing**

### 🎯 **Chapter Overview**
This chapter covers Java reflection capabilities, annotation processing, and how to use these powerful features for dynamic programming and metadata-driven applications.

### 📋 **Reflection Fundamentals**

#### **11.1 Class Information & Inspection**
```java
import java.lang.reflect.*;

public class ReflectionBasics {
    public static void inspectClass(Class<?> clazz) {
        System.out.println("=== Class Information ===");
        System.out.println("Name: " + clazz.getName());
        System.out.println("Simple Name: " + clazz.getSimpleName());
        System.out.println("Package: " + clazz.getPackage());
        System.out.println("Modifiers: " + Modifier.toString(clazz.getModifiers()));
        System.out.println("Superclass: " + clazz.getSuperclass());
        System.out.println("Interfaces: " + Arrays.toString(clazz.getInterfaces()));
        
        // Fields
        System.out.println("\n=== Fields ===");
        Field[] fields = clazz.getDeclaredFields();
        for (Field field : fields) {
            System.out.println("Field: " + field.getName() + " (" + field.getType() + ")");
        }
        
        // Methods
        System.out.println("\n=== Methods ===");
        Method[] methods = clazz.getDeclaredMethods();
        for (Method method : methods) {
            System.out.println("Method: " + method.getName() + " (" + Arrays.toString(method.getParameterTypes()) + ")");
        }
        
        // Constructors
        System.out.println("\n=== Constructors ===");
        Constructor<?>[] constructors = clazz.getDeclaredConstructors();
        for (Constructor<?> constructor : constructors) {
            System.out.println("Constructor: " + constructor.getName() + " (" + Arrays.toString(constructor.getParameterTypes()) + ")");
        }
    }
    
    public static void main(String[] args) {
        // Inspect a sample class
        inspectClass(String.class);
        
        // Inspect our own class
        inspectClass(ReflectionBasics.class);
    }
}
```

#### **11.2 Dynamic Object Creation & Method Invocation**
```java
public class DynamicObjectCreation {
    public static void createObjectDynamically() {
        try {
            // Get class reference
            Class<?> stringClass = String.class;
            
            // Create object using default constructor
            String defaultString = (String) stringClass.getDeclaredConstructor().newInstance();
            System.out.println("Default string: '" + defaultString + "'");
            
            // Create object using parameterized constructor
            Constructor<?> stringConstructor = stringClass.getDeclaredConstructor(String.class);
            String customString = (String) stringConstructor.newInstance("Hello Reflection!");
            System.out.println("Custom string: " + customString);
            
            // Invoke methods dynamically
            Method lengthMethod = stringClass.getMethod("length");
            int length = (int) lengthMethod.invoke(customString);
            System.out.println("String length: " + length);
            
            Method substringMethod = stringClass.getMethod("substring", int.class, int.class);
            String substring = (String) substringMethod.invoke(customString, 0, 5);
            System.out.println("Substring: " + substring);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static void accessPrivateMembers() {
        try {
            // Access private fields
            Field[] fields = String.class.getDeclaredFields();
            for (Field field : fields) {
                field.setAccessible(true); // Make private field accessible
                System.out.println("Field: " + field.getName() + " = " + field.get(null));
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### 📋 **Annotation Processing**

#### **11.3 Custom Annotations**
```java
import java.lang.annotation.*;

// Custom annotation for validation
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.FIELD, ElementType.METHOD, ElementType.PARAMETER})
public @interface Validation {
    String message() default "Validation failed";
    int minLength() default 0;
    int maxLength() default Integer.MAX_VALUE;
    boolean required() default false;
}

// Custom annotation for logging
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD})
public @interface Loggable {
    String level() default "INFO";
    boolean logParameters() default true;
    boolean logReturnValue() default false;
}

// Custom annotation for caching
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD})
public @interface Cacheable {
    String key() default "";
    int ttl() default 3600; // Time to live in seconds
}
```

#### **11.4 Annotation Processing at Runtime**
```java
public class AnnotationProcessor {
    public static void processValidationAnnotations(Object obj) {
        Class<?> clazz = obj.getClass();
        Field[] fields = clazz.getDeclaredFields();
        
        for (Field field : fields) {
            Validation validation = field.getAnnotation(Validation.class);
            if (validation != null) {
                field.setAccessible(true);
                try {
                    Object value = field.get(obj);
                    validateField(value, validation, field.getName());
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    
    private static void validateField(Object value, Validation validation, String fieldName) {
        if (validation.required() && value == null) {
            throw new ValidationException(fieldName + ": " + validation.message());
        }
        
        if (value instanceof String) {
            String stringValue = (String) value;
            if (stringValue.length() < validation.minLength()) {
                throw new ValidationException(fieldName + ": Minimum length is " + validation.minLength());
            }
            if (stringValue.length() > validation.maxLength()) {
                throw new ValidationException(fieldName + ": Maximum length is " + validation.maxLength());
            }
        }
    }
    
    public static void processLoggingAnnotations(Object obj, String methodName, Object... args) {
        try {
            Method method = obj.getClass().getMethod(methodName, 
                Arrays.stream(args).map(Object::getClass).toArray(Class[]::new));
            
            Loggable loggable = method.getAnnotation(Loggable.class);
            if (loggable != null) {
                System.out.println("[" + loggable.level() + "] Calling method: " + methodName);
                
                if (loggable.logParameters()) {
                    System.out.println("Parameters: " + Arrays.toString(args));
                }
                
                // Invoke method
                Object result = method.invoke(obj, args);
                
                if (loggable.logReturnValue()) {
                    System.out.println("Return value: " + result);
                }
                
                System.out.println("[" + loggable.level() + "] Method completed: " + methodName);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static class ValidationException extends RuntimeException {
        public ValidationException(String message) {
            super(message);
        }
    }
}

// Example usage
class User {
    @Validation(required = true, minLength = 2, maxLength = 50)
    private String name;
    
    @Validation(required = true, minLength = 6)
    private String password;
    
    public User(String name, String password) {
        this.name = name;
        this.password = password;
    }
    
    @Loggable(level = "INFO", logParameters = true, logReturnValue = true)
    public String getDisplayName() {
        return "User: " + name;
    }
    
    @Cacheable(key = "user", ttl = 1800)
    public String getProfile() {
        return "Profile for user: " + name;
    }
}
```

---

## 🎯 **Chapter 12: Performance Optimization Techniques**

### 🎯 **Chapter Overview**
This chapter covers Java performance optimization techniques, profiling tools, memory management, and best practices for building high-performance applications.

### 📋 **Performance Profiling & Measurement**

#### **12.1 Performance Measurement Tools**
```java
public class PerformanceMeasurement {
    public static void measureExecutionTime(Runnable task, String taskName) {
        long startTime = System.nanoTime();
        long startMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        task.run();
        
        long endTime = System.nanoTime();
        long endMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        long duration = endTime - startTime;
        long memoryUsed = endMemory - startMemory;
        
        System.out.println("Task: " + taskName);
        System.out.println("Duration: " + duration + " ns (" + (duration / 1_000_000.0) + " ms)");
        System.out.println("Memory used: " + memoryUsed + " bytes (" + (memoryUsed / 1024.0) + " KB)");
        System.out.println();
    }
    
    public static void benchmarkCollections() {
        // ArrayList vs LinkedList performance
        measureExecutionTime(() -> {
            List<Integer> arrayList = new ArrayList<>();
            for (int i = 0; i < 100000; i++) {
                arrayList.add(0, i); // O(n) operation
            }
        }, "ArrayList add at beginning");
        
        measureExecutionTime(() -> {
            List<Integer> linkedList = new LinkedList<>();
            for (int i = 0; i < 100000; i++) {
                linkedList.add(0, i); // O(1) operation
            }
        }, "LinkedList add at beginning");
        
        // HashMap vs TreeMap performance
        measureExecutionTime(() -> {
            Map<String, Integer> hashMap = new HashMap<>();
            for (int i = 0; i < 100000; i++) {
                hashMap.put("key" + i, i);
            }
        }, "HashMap insertion");
        
        measureExecutionTime(() -> {
            Map<String, Integer> treeMap = new TreeMap<>();
            for (int i = 0; i < 100000; i++) {
                treeMap.put("key" + i, i);
            }
        }, "TreeMap insertion");
    }
}
```

#### **12.2 Memory Optimization Techniques**
```java
public class MemoryOptimization {
    // Object pooling for expensive objects
    public static class ObjectPool<T> {
        private final Queue<T> pool;
        private final Supplier<T> factory;
        private final int maxSize;
        
        public ObjectPool(Supplier<T> factory, int maxSize) {
            this.factory = factory;
            this.maxSize = maxSize;
            this.pool = new ConcurrentLinkedQueue<>();
        }
        
        public T borrow() {
            T obj = pool.poll();
            return obj != null ? obj : factory.get();
        }
        
        public void returnObject(T obj) {
            if (pool.size() < maxSize) {
                pool.offer(obj);
            }
        }
    }
    
    // Weak references for caching
    public static class WeakCache<K, V> {
        private final Map<K, WeakReference<V>> cache = new ConcurrentHashMap<>();
        
        public V get(K key) {
            WeakReference<V> ref = cache.get(key);
            if (ref != null) {
                V value = ref.get();
                if (value != null) {
                    return value;
                } else {
                    cache.remove(key); // Clean up garbage collected references
                }
            }
            return null;
        }
        
        public void put(K key, V value) {
            cache.put(key, new WeakReference<>(value));
        }
    }
    
    // Primitive collections to avoid boxing
    public static void usePrimitiveCollections() {
        // Instead of List<Integer>, use int[]
        int[] numbers = new int[1000000];
        for (int i = 0; i < numbers.length; i++) {
            numbers[i] = i;
        }
        
        // Use IntStream for operations
        int sum = Arrays.stream(numbers).sum();
        int[] evenNumbers = Arrays.stream(numbers)
                .filter(n -> n % 2 == 0)
                .toArray();
    }
}
```

### 📋 **JVM Tuning & Optimization**

#### **12.3 JVM Flags & Garbage Collection**
```java
public class JVMOptimization {
    public static void demonstrateGCBehavior() {
        // Force garbage collection (for demonstration only)
        System.gc();
        
        // Memory information
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        long maxMemory = runtime.maxMemory();
        
        System.out.println("Memory Information:");
        System.out.println("Total Memory: " + (totalMemory / 1024 / 1024) + " MB");
        System.out.println("Free Memory: " + (freeMemory / 1024 / 1024) + " MB");
        System.out.println("Used Memory: " + (usedMemory / 1024 / 1024) + " MB");
        System.out.println("Max Memory: " + (maxMemory / 1024 / 1024) + " MB");
        
        // Garbage collection information
        List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
        for (GarbageCollectorMXBean gcBean : gcBeans) {
            System.out.println("GC: " + gcBean.getName() + 
                             " - Collections: " + gcBean.getCollectionCount() +
                             " - Time: " + gcBean.getCollectionTime() + " ms");
        }
    }
    
    public static void optimizeStringOperations() {
        // StringBuilder for multiple concatenations
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 10000; i++) {
            sb.append("Item ").append(i).append(", ");
        }
        String result = sb.toString();
        
        // String interning for repeated strings
        String[] strings = new String[1000];
        for (int i = 0; i < strings.length; i++) {
            strings[i] = ("CommonPrefix" + (i % 100)).intern();
        }
    }
}
```

#### **12.4 Algorithmic Optimization**
```java
public class AlgorithmicOptimization {
    // Efficient sorting for small arrays
    public static void optimizedSort(int[] array) {
        if (array.length < 10) {
            insertionSort(array); // O(n²) but good for small arrays
        } else if (array.length < 1000) {
            Arrays.sort(array); // Dual-pivot quicksort
        } else {
            parallelSort(array); // Parallel sort for large arrays
        }
    }
    
    private static void insertionSort(int[] array) {
        for (int i = 1; i < array.length; i++) {
            int key = array[i];
            int j = i - 1;
            while (j >= 0 && array[j] > key) {
                array[j + 1] = array[j];
                j--;
            }
            array[j + 1] = key;
        }
    }
    
    private static void parallelSort(int[] array) {
        Arrays.parallelSort(array);
    }
    
    // Lazy evaluation for expensive operations
    public static class LazyEvaluator<T> {
        private final Supplier<T> supplier;
        private volatile T value;
        private volatile boolean initialized = false;
        
        public LazyEvaluator(Supplier<T> supplier) {
            this.supplier = supplier;
        }
        
        public T get() {
            if (!initialized) {
                synchronized (this) {
                    if (!initialized) {
                        value = supplier.get();
                        initialized = true;
                    }
                }
            }
            return value;
        }
    }
    
    // Example usage
    public static void demonstrateLazyEvaluation() {
        LazyEvaluator<ExpensiveObject> lazyObject = new LazyEvaluator<>(() -> {
            System.out.println("Creating expensive object...");
            return new ExpensiveObject();
        });
        
        System.out.println("Lazy evaluator created");
        System.out.println("Getting object...");
        ExpensiveObject obj = lazyObject.get(); // Object created only now
        System.out.println("Object retrieved: " + obj);
    }
    
    private static class ExpensiveObject {
        public ExpensiveObject() {
            try {
                Thread.sleep(1000); // Simulate expensive creation
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
```

---

## 🎯 **Chapter 13: Real-World Project Examples**

### 🎯 **Chapter Overview**
This chapter presents real-world Java projects that demonstrate the practical application of all the concepts covered in this book.

### 📋 **Project 1: E-Commerce Order Management System**

#### **13.1 System Architecture & Design**
```java
// Domain Models
public class Order {
    private String orderId;
    private Customer customer;
    private List<OrderItem> items;
    private OrderStatus status;
    private LocalDateTime createdAt;
    private BigDecimal totalAmount;
    
    // Constructor, getters, setters
    public Order(String orderId, Customer customer) {
        this.orderId = orderId;
        this.customer = customer;
        this.items = new ArrayList<>();
        this.status = OrderStatus.PENDING;
        this.createdAt = LocalDateTime.now();
        this.totalAmount = BigDecimal.ZERO;
    }
    
    public void addItem(Product product, int quantity) {
        OrderItem item = new OrderItem(product, quantity);
        items.add(item);
        recalculateTotal();
    }
    
    private void recalculateTotal() {
        this.totalAmount = items.stream()
                .map(OrderItem::getSubtotal)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }
    
    public void process() {
        if (status != OrderStatus.PENDING) {
            throw new IllegalStateException("Order cannot be processed in current status: " + status);
        }
        
        // Validate inventory
        validateInventory();
        
        // Process payment
        processPayment();
        
        // Update status
        status = OrderStatus.CONFIRMED;
        
        // Notify customer
        notifyCustomer();
    }
    
    private void validateInventory() {
        for (OrderItem item : items) {
            if (!item.getProduct().hasSufficientStock(item.getQuantity())) {
                throw new InsufficientStockException("Insufficient stock for product: " + item.getProduct().getName());
            }
        }
    }
    
    private void processPayment() {
        // Payment processing logic
        PaymentProcessor.process(this);
    }
    
    private void notifyCustomer() {
        NotificationService.sendOrderConfirmation(customer, this);
    }
}

public class OrderItem {
    private Product product;
    private int quantity;
    private BigDecimal unitPrice;
    
    public OrderItem(Product product, int quantity) {
        this.product = product;
        this.quantity = quantity;
        this.unitPrice = product.getPrice();
    }
    
    public BigDecimal getSubtotal() {
        return unitPrice.multiply(BigDecimal.valueOf(quantity));
    }
    
    // Getters and setters
}

public enum OrderStatus {
    PENDING, CONFIRMED, SHIPPED, DELIVERED, CANCELLED
}
```

#### **13.2 Service Layer Implementation**
```java
@Service
public class OrderService {
    private final OrderRepository orderRepository;
    private final ProductService productService;
    private final CustomerService customerService;
    private final PaymentService paymentService;
    
    public OrderService(OrderRepository orderRepository, 
                       ProductService productService,
                       CustomerService customerService,
                       PaymentService paymentService) {
        this.orderRepository = orderRepository;
        this.productService = productService;
        this.customerService = customerService;
        this.paymentService = paymentService;
    }
    
    @Transactional
    public Order createOrder(CreateOrderRequest request) {
        // Validate customer
        Customer customer = customerService.getCustomerById(request.getCustomerId());
        if (customer == null) {
            throw new CustomerNotFoundException("Customer not found: " + request.getCustomerId());
        }
        
        // Create order
        String orderId = generateOrderId();
        Order order = new Order(orderId, customer);
        
        // Add items
        for (OrderItemRequest itemRequest : request.getItems()) {
            Product product = productService.getProductById(itemRequest.getProductId());
            if (product == null) {
                throw new ProductNotFoundException("Product not found: " + itemRequest.getProductId());
            }
            order.addItem(product, itemRequest.getQuantity());
        }
        
        // Save order
        Order savedOrder = orderRepository.save(order);
        
        // Publish event
        eventPublisher.publishEvent(new OrderCreatedEvent(savedOrder));
        
        return savedOrder;
    }
    
    @Transactional
    public Order processOrder(String orderId) {
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new OrderNotFoundException("Order not found: " + orderId));
        
        try {
            order.process();
            return orderRepository.save(order);
        } catch (Exception e) {
            // Log error and update status
            order.setStatus(OrderStatus.FAILED);
            orderRepository.save(order);
            throw e;
        }
    }
    
    private String generateOrderId() {
        return "ORD-" + System.currentTimeMillis() + "-" + ThreadLocalRandom.current().nextInt(1000, 9999);
    }
}
```

### 📋 **Project 2: RESTful API with Spring Boot**

#### **13.3 Controller & Exception Handling**
```java
@RestController
@RequestMapping("/api/orders")
@Validated
public class OrderController {
    private final OrderService orderService;
    private final OrderMapper orderMapper;
    
    public OrderController(OrderService orderService, OrderMapper orderMapper) {
        this.orderService = orderService;
        this.orderMapper = orderMapper;
    }
    
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public OrderResponse createOrder(@Valid @RequestBody CreateOrderRequest request) {
        Order order = orderService.createOrder(request);
        return orderMapper.toResponse(order);
    }
    
    @GetMapping("/{orderId}")
    public OrderResponse getOrder(@PathVariable String orderId) {
        Order order = orderService.getOrderById(orderId);
        return orderMapper.toResponse(order);
    }
    
    @PutMapping("/{orderId}/process")
    public OrderResponse processOrder(@PathVariable String orderId) {
        Order order = orderService.processOrder(orderId);
        return orderMapper.toResponse(order);
    }
    
    @GetMapping
    public Page<OrderResponse> getOrders(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) OrderStatus status,
            @RequestParam(required = false) String customerId) {
        
        OrderSearchCriteria criteria = OrderSearchCriteria.builder()
                .status(status)
                .customerId(customerId)
                .build();
        
        Page<Order> orders = orderService.searchOrders(criteria, PageRequest.of(page, size));
        return orders.map(orderMapper::toResponse);
    }
    
    @ExceptionHandler(OrderNotFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ErrorResponse handleOrderNotFound(OrderNotFoundException e) {
        return ErrorResponse.builder()
                .error("Order not found")
                .message(e.getMessage())
                .timestamp(LocalDateTime.now())
                .build();
    }
    
    @ExceptionHandler(ValidationException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ErrorResponse handleValidation(ValidationException e) {
        return ErrorResponse.builder()
                .error("Validation failed")
                .message(e.getMessage())
                .timestamp(LocalDateTime.now())
                .build();
    }
}
```

#### **13.4 Repository & Data Access Layer**
```java
@Repository
public class OrderRepositoryImpl implements OrderRepository {
    private final JdbcTemplate jdbcTemplate;
    private final OrderRowMapper orderRowMapper;
    
    public OrderRepositoryImpl(JdbcTemplate jdbcTemplate, OrderRowMapper orderRowMapper) {
        this.jdbcTemplate = jdbcTemplate;
        this.orderRowMapper = orderRowMapper;
    }
    
    @Override
    public Order save(Order order) {
        if (order.getOrderId() == null) {
            return insert(order);
        } else {
            return update(order);
        }
    }
    
    private Order insert(Order order) {
        String sql = """
            INSERT INTO orders (order_id, customer_id, status, created_at, total_amount)
            VALUES (?, ?, ?, ?, ?)
            """;
        
        jdbcTemplate.update(sql,
                order.getOrderId(),
                order.getCustomer().getId(),
                order.getStatus().name(),
                order.getCreatedAt(),
                order.getTotalAmount());
        
        // Insert order items
        insertOrderItems(order);
        
        return order;
    }
    
    private void insertOrderItems(Order order) {
        String sql = """
            INSERT INTO order_items (order_id, product_id, quantity, unit_price)
            VALUES (?, ?, ?, ?)
            """;
        
        for (OrderItem item : order.getItems()) {
            jdbcTemplate.update(sql,
                    order.getOrderId(),
                    item.getProduct().getId(),
                    item.getQuantity(),
                    item.getUnitPrice());
        }
    }
    
    @Override
    public Optional<Order> findById(String orderId) {
        String sql = """
            SELECT o.*, c.*, oi.*, p.*
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            LEFT JOIN products p ON oi.product_id = p.id
            WHERE o.order_id = ?
            """;
        
        List<Order> orders = jdbcTemplate.query(sql, orderRowMapper, orderId);
        return orders.isEmpty() ? Optional.empty() : Optional.of(orders.get(0));
    }
    
    @Override
    public Page<Order> searchOrders(OrderSearchCriteria criteria, Pageable pageable) {
        StringBuilder sql = new StringBuilder("""
            SELECT o.*, c.*, oi.*, p.*
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            LEFT JOIN products p ON oi.product_id = p.id
            WHERE 1=1
            """);
        
        List<Object> params = new ArrayList<>();
        
        if (criteria.getStatus() != null) {
            sql.append(" AND o.status = ?");
            params.add(criteria.getStatus().name());
        }
        
        if (criteria.getCustomerId() != null) {
            sql.append(" AND o.customer_id = ?");
            params.add(criteria.getCustomerId());
        }
        
        sql.append(" ORDER BY o.created_at DESC");
        
        // Add pagination
        sql.append(" LIMIT ? OFFSET ?");
        params.add(pageable.getPageSize());
        params.add(pageable.getPageNumber() * pageable.getPageSize());
        
        List<Order> orders = jdbcTemplate.query(sql.toString(), orderRowMapper, params.toArray());
        
        // Get total count for pagination
        String countSql = buildCountSql(criteria);
        Long total = jdbcTemplate.queryForObject(countSql, Long.class, 
                params.subList(0, params.size() - 2).toArray());
        
        return new PageImpl<>(orders, pageable, total != null ? total : 0);
    }
}
```

---

## 🎯 **Chapter 14: Testing & Debugging Strategies**

### 🎯 **Chapter Overview**
This chapter covers comprehensive testing strategies, debugging techniques, and best practices for building reliable Java applications.

### 📋 **Unit Testing with JUnit 5**

#### **14.1 Test Structure & Best Practices**
```java
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Order Service Tests")
class OrderServiceTest {
    
    private OrderService orderService;
    private OrderRepository mockOrderRepository;
    private ProductService mockProductService;
    private CustomerService mockCustomerService;
    
    @BeforeEach
    void setUp() {
        mockOrderRepository = mock(OrderRepository.class);
        mockProductService = mock(ProductService.class);
        mockCustomerService = mock(CustomerService.class);
        
        orderService = new OrderService(
            mockOrderRepository, 
            mockProductService, 
            mockCustomerService, 
            mock(PaymentService.class)
        );
    }
    
    @Test
    @DisplayName("Should create order successfully with valid data")
    void shouldCreateOrderSuccessfully() {
        // Given
        String customerId = "CUST-001";
        String productId = "PROD-001";
        int quantity = 2;
        
        Customer customer = new Customer(customerId, "John Doe", "john@example.com");
        Product product = new Product(productId, "Laptop", BigDecimal.valueOf(999.99));
        
        CreateOrderRequest request = CreateOrderRequest.builder()
                .customerId(customerId)
                .items(List.of(new OrderItemRequest(productId, quantity)))
                .build();
        
        when(mockCustomerService.getCustomerById(customerId)).thenReturn(customer);
        when(mockProductService.getProductById(productId)).thenReturn(product);
        when(mockOrderRepository.save(any(Order.class))).thenAnswer(invocation -> invocation.getArgument(0));
        
        // When
        Order result = orderService.createOrder(request);
        
        // Then
        assertNotNull(result);
        assertEquals(customerId, result.getCustomer().getId());
        assertEquals(1, result.getItems().size());
        assertEquals(quantity, result.getItems().get(0).getQuantity());
        assertEquals(OrderStatus.PENDING, result.getStatus());
        
        verify(mockOrderRepository).save(any(Order.class));
    }
    
    @Test
    @DisplayName("Should throw exception when customer not found")
    void shouldThrowExceptionWhenCustomerNotFound() {
        // Given
        String customerId = "INVALID-CUST";
        CreateOrderRequest request = CreateOrderRequest.builder()
                .customerId(customerId)
                .items(new ArrayList<>())
                .build();
        
        when(mockCustomerService.getCustomerById(customerId)).thenReturn(null);
        
        // When & Then
        CustomerNotFoundException exception = assertThrows(
            CustomerNotFoundException.class,
            () -> orderService.createOrder(request)
        );
        
        assertEquals("Customer not found: " + customerId, exception.getMessage());
        verify(mockOrderRepository, never()).save(any(Order.class));
    }
    
    @ParameterizedTest
    @ValueSource(ints = {0, -1, -100})
    @DisplayName("Should throw exception for invalid quantities")
    void shouldThrowExceptionForInvalidQuantities(int invalidQuantity) {
        // Given
        String customerId = "CUST-001";
        String productId = "PROD-001";
        
        Customer customer = new Customer(customerId, "John Doe", "john@example.com");
        Product product = new Product(productId, "Laptop", BigDecimal.valueOf(999.99));
        
        CreateOrderRequest request = CreateOrderRequest.builder()
                .customerId(customerId)
                .items(List.of(new OrderItemRequest(productId, invalidQuantity)))
                .build();
        
        when(mockCustomerService.getCustomerById(customerId)).thenReturn(customer);
        when(mockProductService.getProductById(productId)).thenReturn(product);
        
        // When & Then
        assertThrows(IllegalArgumentException.class, () -> orderService.createOrder(request));
    }
    
    @Test
    @DisplayName("Should calculate order total correctly")
    void shouldCalculateOrderTotalCorrectly() {
        // Given
        String customerId = "CUST-001";
        Customer customer = new Customer(customerId, "John Doe", "john@example.com");
        
        Product laptop = new Product("PROD-001", "Laptop", BigDecimal.valueOf(999.99));
        Product mouse = new Product("PROD-002", "Mouse", BigDecimal.valueOf(29.99));
        
        CreateOrderRequest request = CreateOrderRequest.builder()
                .customerId(customerId)
                .items(List.of(
                    new OrderItemRequest("PROD-001", 1),
                    new OrderItemRequest("PROD-002", 2)
                ))
                .build();
        
        when(mockCustomerService.getCustomerById(customerId)).thenReturn(customer);
        when(mockProductService.getProductById("PROD-001")).thenReturn(laptop);
        when(mockProductService.getProductById("PROD-002")).thenReturn(mouse);
        when(mockOrderRepository.save(any(Order.class))).thenAnswer(invocation -> invocation.getArgument(0));
        
        // When
        Order result = orderService.createOrder(request);
        
        // Then
        BigDecimal expectedTotal = laptop.getPrice()
                .add(mouse.getPrice().multiply(BigDecimal.valueOf(2)));
        assertEquals(0, expectedTotal.compareTo(result.getTotalAmount()));
    }
}
```

#### **14.2 Integration Testing**
```java
@SpringBootTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@Transactional
class OrderServiceIntegrationTest {
    
    @Autowired
    private OrderService orderService;
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private CustomerRepository customerRepository;
    
    @Autowired
    private ProductRepository productRepository;
    
    @Test
    @DisplayName("Should create and persist order in database")
    void shouldCreateAndPersistOrderInDatabase() {
        // Given
        Customer customer = customerRepository.save(new Customer("John Doe", "john@example.com"));
        Product product = productRepository.save(new Product("Laptop", BigDecimal.valueOf(999.99)));
        
        CreateOrderRequest request = CreateOrderRequest.builder()
                .customerId(customer.getId())
                .items(List.of(new OrderItemRequest(product.getId(), 2)))
                .build();
        
        // When
        Order createdOrder = orderService.createOrder(request);
        
        // Then
        assertNotNull(createdOrder.getOrderId());
        
        // Verify persistence
        Optional<Order> persistedOrder = orderRepository.findById(createdOrder.getOrderId());
        assertTrue(persistedOrder.isPresent());
        
        Order order = persistedOrder.get();
        assertEquals(customer.getId(), order.getCustomer().getId());
        assertEquals(1, order.getItems().size());
        assertEquals(2, order.getItems().get(0).getQuantity());
        assertEquals(product.getId(), order.getItems().get(0).getProduct().getId());
    }
    
    @Test
    @DisplayName("Should process order and update status")
    void shouldProcessOrderAndUpdateStatus() {
        // Given
        Customer customer = customerRepository.save(new Customer("John Doe", "john@example.com"));
        Product product = productRepository.save(new Product("Laptop", BigDecimal.valueOf(999.99)));
        
        Order order = new Order("TEST-ORD-001", customer);
        order.addItem(product, 1);
        order = orderRepository.save(order);
        
        // When
        Order processedOrder = orderService.processOrder(order.getOrderId());
        
        // Then
        assertEquals(OrderStatus.CONFIRMED, processedOrder.getStatus());
        
        // Verify database update
        Order updatedOrder = orderRepository.findById(order.getOrderId()).orElseThrow();
        assertEquals(OrderStatus.CONFIRMED, updatedOrder.getStatus());
    }
}
```

### 📋 **Debugging & Troubleshooting**

#### **14.3 Debugging Techniques**
```java
public class DebuggingExamples {
    
    public static void demonstrateDebugging() {
        // 1. Strategic logging
        Logger logger = LoggerFactory.getLogger(DebuggingExamples.class);
        
        try {
            processComplexData();
        } catch (Exception e) {
            logger.error("Failed to process data", e);
            // Log additional context
            logger.error("Current state: {}", getCurrentState());
            logger.error("Input parameters: {}", getInputParameters());
        }
        
        // 2. Assertions for debugging
        String result = processString("test");
        assert result != null : "Result should not be null";
        assert result.length() > 0 : "Result should not be empty";
        
        // 3. Conditional breakpoints (use in IDE)
        for (int i = 0; i < 1000; i++) {
            if (i == 500) { // Set breakpoint here
                System.out.println("Debug point reached at i = " + i);
            }
            processItem(i);
        }
    }
    
    private static void processComplexData() {
        // Simulate complex processing
        if (Math.random() > 0.5) {
            throw new RuntimeException("Random processing failure");
        }
    }
    
    private static String processString(String input) {
        return input.toUpperCase();
    }
    
    private static void processItem(int item) {
        // Simulate item processing
        if (item % 100 == 0) {
            System.out.println("Processing item: " + item);
        }
    }
    
    private static String getCurrentState() {
        return "Processing state";
    }
    
    private static String getInputParameters() {
        return "Input parameters";
    }
}
```

#### **14.4 Performance Testing**
```java
@SpringBootTest
class OrderServicePerformanceTest {
    
    @Autowired
    private OrderService orderService;
    
    @Test
    @DisplayName("Should handle high load efficiently")
    void shouldHandleHighLoadEfficiently() {
        int numberOfOrders = 1000;
        List<CreateOrderRequest> requests = generateTestRequests(numberOfOrders);
        
        long startTime = System.currentTimeMillis();
        
        // Process orders in parallel
        List<Order> orders = requests.parallelStream()
                .map(orderService::createOrder)
                .collect(Collectors.toList());
        
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        
        // Performance assertions
        assertEquals(numberOfOrders, orders.size());
        assertTrue(totalTime < 10000, "Should process 1000 orders in less than 10 seconds");
        
        double ordersPerSecond = (double) numberOfOrders / (totalTime / 1000.0);
        System.out.println("Performance: " + String.format("%.2f", ordersPerSecond) + " orders/second");
    }
    
    private List<CreateOrderRequest> generateTestRequests(int count) {
        // Generate test data
        return IntStream.range(0, count)
                .mapToObj(i -> CreateOrderRequest.builder()
                        .customerId("CUST-" + i)
                        .items(List.of(new OrderItemRequest("PROD-" + i, 1)))
                        .build())
                .collect(Collectors.toList());
    }
}
```

---

## 🎯 **Chapter 15: Java Interview Questions & Technical Assessment**
```java
String s1 = "Hello";
String s2 = new String("Hello");
String s3 = "Hello";

System.out.println(s1 == s2);      // false (different objects)
System.out.println(s1 == s3);      // true (string pool)
System.out.println(s1.equals(s2)); // true (same content)
```

**Q2: What is the String Pool and how does it work?**
- String Pool is a special memory area in the heap
- String literals are automatically interned
- `String.intern()` method adds strings to the pool
- Saves memory by reusing string objects

**Q3: Explain Java Memory Model components.**
```java
// Stack: Local variables, method calls
// Heap: Objects, arrays
// Method Area: Class metadata, static variables
// Native Method Stack: Native method calls
```

**Q4: What is the difference between primitive types and wrapper classes?**
```java
int primitive = 42;           // Stack, no null
Integer wrapper = 42;         // Heap, can be null
Integer autoBoxed = 42;       // Autoboxing
int unboxed = wrapper;        // Unboxing
```

#### **15.2 Object-Oriented Programming**

**Q5: Explain the difference between method overloading and overriding.**
```java
// Overloading: Same method name, different parameters
public void process(int value) { }
public void process(String value) { }
public void process(int value, String description) { }

// Overriding: Same method signature in subclass
@Override
public void process(int value) { 
    super.process(value); // Call parent method
}
```

**Q6: What is the difference between abstract classes and interfaces?**
```java
// Abstract Class: Can have state, constructors, concrete methods
abstract class Animal {
    protected String name; // State
    public Animal(String name) { this.name = name; } // Constructor
    public abstract void makeSound(); // Abstract method
    public void sleep() { System.out.println("Sleeping..."); } // Concrete
}

// Interface: Pure contract, default methods, static methods
interface Movable {
    void move(); // Abstract method
    default void stop() { System.out.println("Stopped"); } // Default
    static boolean isValidSpeed(int speed) { return speed > 0; } // Static
}
```

**Q7: Explain polymorphism with examples.**
```java
// Compile-time polymorphism (overloading)
public class Calculator {
    public int add(int a, int b) { return a + b; }
    public double add(double a, double b) { return a + b; }
}

// Runtime polymorphism (overriding)
Animal animal = new Dog(); // Animal reference, Dog object
animal.makeSound(); // Calls Dog's makeSound method
```

#### **15.3 Collections Framework**

**Q8: What is the difference between ArrayList and LinkedList?**
```java
// ArrayList: Dynamic array, fast random access
ArrayList<String> arrayList = new ArrayList<>();
arrayList.add("Element");           // O(1) amortized
String element = arrayList.get(0);  // O(1)
arrayList.remove(0);               // O(n) - shifting

// LinkedList: Doubly linked list, fast insertion/deletion
LinkedList<String> linkedList = new LinkedList<>();
linkedList.add("Element");          // O(1)
linkedList.addFirst("First");       // O(1)
linkedList.removeLast();            // O(1)
```

**Q9: How does HashMap handle collisions?**
```java
// HashMap uses chaining (linked list) for collisions
// Java 8+: Converts to tree when list length > 8
// Resize when load factor threshold is reached
Map<String, Integer> map = new HashMap<>();
map.put("key1", 1);
map.put("key2", 2);
// If hash(key1) == hash(key2), collision occurs
```

**Q10: Explain the difference between HashSet and TreeSet.**
```java
// HashSet: Hash table, O(1) operations, no ordering
HashSet<String> hashSet = new HashSet<>();
hashSet.add("Zebra");
hashSet.add("Apple");
// Order not guaranteed

// TreeSet: Red-black tree, O(log n) operations, sorted
TreeSet<String> treeSet = new TreeSet<>();
treeSet.add("Zebra");
treeSet.add("Apple");
// Always sorted: Apple, Zebra
```

#### **15.4 Generics & Type Safety**

**Q11: What are bounded generics and how do you use them?**
```java
// Upper bound: T must be Number or subclass
class NumberBox<T extends Number> {
    private T value;
    public NumberBox(T value) { this.value = value; }
    public double getDoubleValue() { return value.doubleValue(); }
}

// Multiple bounds: T must implement multiple interfaces
class ComparableNumberBox<T extends Number & Comparable<T>> {
    public T max(T a, T b) {
        return a.compareTo(b) > 0 ? a : b;
    }
}
```

**Q12: Explain wildcards in generics.**
```java
// Upper bound wildcard: ? extends Type
public static double sumOfNumbers(List<? extends Number> numbers) {
    return numbers.stream()
                 .mapToDouble(Number::doubleValue)
                 .sum();
}

// Lower bound wildcard: ? super Type
public static void addNumbers(List<? super Integer> numbers) {
    numbers.add(42); // Can add Integer
    numbers.add(100); // Can add Integer
}
```

#### **15.5 Exception Handling**

**Q13: What is the difference between checked and unchecked exceptions?**
```java
// Checked exceptions: Must be handled or declared
public void readFile() throws IOException {
    FileReader reader = new FileReader("file.txt"); // Compiler error if not handled
}

// Unchecked exceptions: Don't need explicit handling
public void divide(int a, int b) {
    if (b == 0) {
        throw new IllegalArgumentException("Division by zero"); // No throws needed
    }
    return a / b;
}
```

**Q14: Explain try-with-resources.**
```java
// Automatic resource management
try (FileReader reader = new FileReader("file.txt");
     BufferedReader br = new BufferedReader(reader)) {
    String line = br.readLine();
    // Resources automatically closed
} catch (IOException e) {
    // Handle exception
}
```

#### **15.6 Multithreading & Concurrency**

**Q15: What is the difference between Thread and Runnable?**
```java
// Extending Thread
class MyThread extends Thread {
    @Override
    public void run() {
        System.out.println("Thread running: " + Thread.currentThread().getName());
    }
}

// Implementing Runnable (preferred)
class MyRunnable implements Runnable {
    @Override
    public void run() {
        System.out.println("Runnable running: " + Thread.currentThread().getName());
    }
}

// Usage
Thread thread1 = new MyThread();
Thread thread2 = new Thread(new MyRunnable());
```

**Q16: Explain synchronized keyword and its usage.**
```java
class Counter {
    private int count = 0;
    
    // Synchronized method
    public synchronized void increment() {
        count++;
    }
    
    // Synchronized block
    public void decrement() {
        synchronized(this) {
            count--;
        }
    }
    
    // Synchronized static method
    public static synchronized void reset() {
        // Synchronizes on class object
    }
}
```

#### **15.7 Design Patterns**

**Q17: Implement a Singleton pattern.**
```java
public class Singleton {
    private static volatile Singleton instance;
    private Singleton() {} // Private constructor
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**Q18: Explain the Observer pattern with Java implementation.**
```java
interface Observer {
    void update(String message);
}

interface Subject {
    void registerObserver(Observer observer);
    void removeObserver(Observer observer);
    void notifyObservers(String message);
}

class NewsAgency implements Subject {
    private List<Observer> observers = new ArrayList<>();
    
    @Override
    public void registerObserver(Observer observer) {
        observers.add(observer);
    }
    
    @Override
    public void notifyObservers(String message) {
        for (Observer observer : observers) {
            observer.update(message);
        }
    }
}
```

#### **15.8 Java 8+ Features**

**Q19: Explain Stream API with examples.**
```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");

// Filter and map
List<String> filteredNames = names.stream()
    .filter(name -> name.length() > 4)
    .map(String::toUpperCase)
    .collect(Collectors.toList());

// Reduce operation
int totalLength = names.stream()
    .mapToInt(String::length)
    .sum();

// Parallel processing
List<String> parallelNames = names.parallelStream()
    .filter(name -> name.startsWith("A"))
    .collect(Collectors.toList());
```

**Q20: What are Optional and how do you use them?**
```java
// Avoid null pointer exceptions
Optional<String> name = Optional.of("John");
Optional<String> emptyName = Optional.empty();

// Safe operations
String result = name.orElse("Unknown");
String upperCase = name.map(String::toUpperCase).orElse("");

// Chaining
Optional<String> processed = name
    .filter(n -> n.length() > 3)
    .map(String::toUpperCase);
```

### 🎯 **Advanced Interview Questions**

#### **15.9 Performance & Optimization**

**Q21: How would you optimize a Java application for better performance?**
- Use appropriate data structures
- Implement caching strategies
- Optimize memory usage
- Use profiling tools
- Consider JVM tuning

**Q22: Explain garbage collection in Java.**
```java
// Different GC algorithms
// - Serial GC: Single thread, good for small applications
// - Parallel GC: Multiple threads, throughput focused
// - G1 GC: Low latency, good for large heaps
// - ZGC: Ultra-low latency, Java 11+

// GC tuning flags
// -Xms2g -Xmx4g -XX:+UseG1GC
```

#### **15.10 System Design Questions**

**Q23: Design a custom ArrayList implementation.**
```java
// Key considerations:
// - Dynamic resizing
// - Efficient element shifting
// - Iterator implementation
// - Thread safety (if needed)
// - Performance characteristics
```

**Q24: Design a thread-safe cache.**
```java
// Key considerations:
// - Eviction policy (LRU, LFU, FIFO)
// - Concurrency control
// - Memory management
// - Performance optimization
// - Monitoring and metrics
```

### 📝 **Interview Preparation Tips**

#### **15.11 Before the Interview**
1. **Review Fundamentals**: Ensure solid understanding of core concepts
2. **Practice Coding**: Solve problems on platforms like LeetCode, HackerRank
3. **Understand Trade-offs**: Know when to use different approaches
4. **Review Your Code**: Be ready to explain your implementation choices

#### **15.12 During the Interview**
1. **Clarify Requirements**: Ask questions before starting
2. **Think Aloud**: Explain your thought process
3. **Consider Edge Cases**: Think about boundary conditions
4. **Optimize Incrementally**: Start simple, then improve

#### **15.13 Common Interview Mistakes to Avoid**
1. **Not Testing Code**: Always test with examples
2. **Ignoring Edge Cases**: Consider null, empty, invalid inputs
3. **Poor Communication**: Explain your approach clearly
4. **Not Asking Questions**: Clarify requirements when unclear

### 🔍 **Sample Interview Scenarios**

#### **Scenario 1: String Processing**
**Problem**: Implement a method to find the first non-repeating character in a string.

**Approach**:
```java
public static char firstNonRepeating(String str) {
    Map<Character, Integer> charCount = new HashMap<>();
    
    // Count characters
    for (char c : str.toCharArray()) {
        charCount.put(c, charCount.getOrDefault(c, 0) + 1);
    }
    
    // Find first non-repeating
    for (char c : str.toCharArray()) {
        if (charCount.get(c) == 1) {
            return c;
        }
    }
    
    return '\0'; // No non-repeating character
}
```

#### **Scenario 2: Data Structure Design**
**Problem**: Design a data structure that supports insert, delete, and getRandom in O(1) time.

**Approach**:
```java
class RandomizedSet {
    private List<Integer> list;
    private Map<Integer, Integer> map; // value -> index
    
    public RandomizedSet() {
        list = new ArrayList<>();
        map = new HashMap<>();
    }
    
    public boolean insert(int val) {
        if (map.containsKey(val)) return false;
        
        list.add(val);
        map.put(val, list.size() - 1);
        return true;
    }
    
    public boolean remove(int val) {
        if (!map.containsKey(val)) return false;
        
        int index = map.get(val);
        int lastElement = list.get(list.size() - 1);
        
        list.set(index, lastElement);
        map.put(lastElement, index);
        
        list.remove(list.size() - 1);
        map.remove(val);
        return true;
    }
    
    public int getRandom() {
        return list.get((int)(Math.random() * list.size()));
    }
}
```

### 📚 **Additional Resources**

#### **15.14 Recommended Reading**
- "Effective Java" by Joshua Bloch
- "Java Concurrency in Practice" by Brian Goetz
- "Clean Code" by Robert C. Martin
- "Design Patterns" by Gang of Four

#### **15.15 Online Practice Platforms**
- LeetCode (Java problems)
- HackerRank (Java track)
- CodeSignal
- InterviewBit

#### **15.16 Mock Interview Resources**
- Pramp
- Interviewing.io
- Triplebyte
- Local coding meetups

### 🎯 **Chapter Summary**

This chapter has provided you with:
- **Comprehensive coverage** of Java interview topics
- **Practical examples** for each concept
- **Common interview questions** with solutions
- **Preparation strategies** and tips
- **Real-world scenarios** to practice

Remember: **Practice makes perfect!** The more you code and solve problems, the more confident you'll become in technical interviews.
