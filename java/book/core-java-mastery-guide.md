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
4. **Core Libraries & Utilities**
5. **Design Patterns in Java**
6. **Advanced Java Concepts**
7. **Problem-Solving Challenges**
8. **Solutions & Explanations**

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

*Solutions will be provided at the end of the book with detailed explanations, code analysis, and performance considerations.*

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

*Detailed solutions with code analysis, performance considerations, and best practices will be provided at the end of the book.*

---

## 📚 **Next Steps**

In the upcoming chapters, we'll explore:
- Exception handling and logging strategies
- I/O operations and file handling
- Concurrency and threading models
- Reflection and annotation processing
- Performance optimization techniques
- Real-world project examples
- Advanced design patterns
- Testing and debugging strategies

**Continue your Java mastery journey with hands-on practice and real-world applications! 🚀**
