import java.util.*;
import java.util.function.Predicate;

/**
 * Custom Iterator with Filtering Capabilities
 * Demonstrates advanced iterator patterns and functional programming concepts
 */
public class FilteringIterator {
    
    public static void main(String[] args) {
        System.out.println("=== Filtering Iterator Demo ===\n");
        
        // Create a list of numbers
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println("Original list: " + numbers);
        
        // Filter even numbers
        System.out.println("\n--- Even Numbers ---");
        FilteredIterator<Integer> evenIterator = new FilteredIterator<>(numbers.iterator(), n -> n % 2 == 0);
        while (evenIterator.hasNext()) {
            System.out.print(evenIterator.next() + " ");
        }
        
        // Filter odd numbers
        System.out.println("\n\n--- Odd Numbers ---");
        FilteredIterator<Integer> oddIterator = new FilteredIterator<>(numbers.iterator(), n -> n % 2 != 0);
        while (oddIterator.hasNext()) {
            System.out.print(oddIterator.next() + " ");
        }
        
        // Filter numbers greater than 5
        System.out.println("\n\n--- Numbers > 5 ---");
        FilteredIterator<Integer> greaterThan5Iterator = new FilteredIterator<>(numbers.iterator(), n -> n > 5);
        while (greaterThan5Iterator.hasNext()) {
            System.out.print(greaterThan5Iterator.next() + " ");
        }
        
        // Chain multiple filters
        System.out.println("\n\n--- Even Numbers > 5 ---");
        FilteredIterator<Integer> evenAndGreaterThan5 = new FilteredIterator<>(
            new FilteredIterator<>(numbers.iterator(), n -> n % 2 == 0),
            n -> n > 5
        );
        while (evenAndGreaterThan5.hasNext()) {
            System.out.print(evenAndGreaterThan5.next() + " ");
        }
        
        // String filtering example
        System.out.println("\n\n--- String Filtering ---");
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry", "fig");
        System.out.println("Original words: " + words);
        
        // Filter words starting with 'a' or 'b'
        FilteredIterator<String> abWordsIterator = new FilteredIterator<>(words.iterator(), 
            word -> word.startsWith("a") || word.startsWith("b"));
        System.out.print("Words starting with 'a' or 'b': ");
        while (abWordsIterator.hasNext()) {
            System.out.print(abWordsIterator.next() + " ");
        }
        
        // Filter words longer than 5 characters
        System.out.println("\n\nWords longer than 5 characters:");
        FilteredIterator<String> longWordsIterator = new FilteredIterator<>(words.iterator(), 
            word -> word.length() > 5);
        while (longWordsIterator.hasNext()) {
            System.out.print(longWordsIterator.next() + " ");
        }
        
        // Custom object filtering
        System.out.println("\n\n--- Custom Object Filtering ---");
        List<Person> people = Arrays.asList(
            new Person("Alice", 25, "Engineer"),
            new Person("Bob", 30, "Manager"),
            new Person("Charlie", 22, "Developer"),
            new Person("Diana", 35, "Designer"),
            new Person("Eve", 28, "Engineer")
        );
        
        System.out.println("All people:");
        people.forEach(p -> System.out.println("  " + p));
        
        // Filter engineers
        System.out.println("\nEngineers:");
        FilteredIterator<Person> engineersIterator = new FilteredIterator<>(people.iterator(), 
            p -> "Engineer".equals(p.getProfession()));
        while (engineersIterator.hasNext()) {
            System.out.println("  " + engineersIterator.next());
        }
        
        // Filter people under 30
        System.out.println("\nPeople under 30:");
        FilteredIterator<Person> youngPeopleIterator = new FilteredIterator<>(people.iterator(), 
            p -> p.getAge() < 30);
        while (youngPeopleIterator.hasNext()) {
            System.out.println("  " + youngPeopleIterator.next());
        }
        
        // Chain filters: Young engineers
        System.out.println("\nYoung engineers (under 30):");
        FilteredIterator<Person> youngEngineersIterator = new FilteredIterator<>(
            new FilteredIterator<>(people.iterator(), p -> "Engineer".equals(p.getProfession())),
            p -> p.getAge() < 30
        );
        while (youngEngineersIterator.hasNext()) {
            System.out.println("  " + youngEngineersIterator.next());
        }
        
        // Demonstrate lazy evaluation
        System.out.println("\n--- Lazy Evaluation Demo ---");
        System.out.println("Creating expensive filter...");
        FilteredIterator<Integer> expensiveIterator = new FilteredIterator<>(numbers.iterator(), n -> {
            // Simulate expensive operation
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return n % 3 == 0;
        });
        
        System.out.println("Filter created, but not executed yet!");
        System.out.println("Now executing filter:");
        while (expensiveIterator.hasNext()) {
            System.out.print(expensiveIterator.next() + " ");
        }
    }
}

/**
 * Generic Filtered Iterator that filters elements based on a predicate
 */
class FilteredIterator<T> implements Iterator<T> {
    private final Iterator<T> sourceIterator;
    private final Predicate<T> predicate;
    private T nextElement;
    private boolean hasNextElement;
    
    public FilteredIterator(Iterator<T> sourceIterator, Predicate<T> predicate) {
        this.sourceIterator = sourceIterator;
        this.predicate = predicate;
        this.hasNextElement = false;
        findNext();
    }
    
    @Override
    public boolean hasNext() {
        return hasNextElement;
    }
    
    @Override
    public T next() {
        if (!hasNextElement) {
            throw new NoSuchElementException();
        }
        
        T result = nextElement;
        findNext();
        return result;
    }
    
    @Override
    public void remove() {
        throw new UnsupportedOperationException("Remove operation not supported by FilteredIterator");
    }
    
    /**
     * Find the next element that matches the predicate
     */
    private void findNext() {
        hasNextElement = false;
        while (sourceIterator.hasNext()) {
            T element = sourceIterator.next();
            if (predicate.test(element)) {
                nextElement = element;
                hasNextElement = true;
                break;
            }
        }
    }
}

/**
 * Advanced Filtered Iterator with multiple predicates and operations
 */
class AdvancedFilteredIterator<T> implements Iterator<T> {
    private final Iterator<T> sourceIterator;
    private final List<Predicate<T>> predicates;
    private final boolean requireAll; // true = AND, false = OR
    private T nextElement;
    private boolean hasNextElement;
    
    public AdvancedFilteredIterator(Iterator<T> sourceIterator, List<Predicate<T>> predicates, boolean requireAll) {
        this.sourceIterator = sourceIterator;
        this.predicates = predicates;
        this.requireAll = requireAll;
        this.hasNextElement = false;
        findNext();
    }
    
    public AdvancedFilteredIterator(Iterator<T> sourceIterator, Predicate<T> predicate) {
        this(sourceIterator, Arrays.asList(predicate), true);
    }
    
    @Override
    public boolean hasNext() {
        return hasNextElement;
    }
    
    @Override
    public T next() {
        if (!hasNextElement) {
            throw new NoSuchElementException();
        }
        
        T result = nextElement;
        findNext();
        return result;
    }
    
    @Override
    public void remove() {
        throw new UnsupportedOperationException("Remove operation not supported by AdvancedFilteredIterator");
    }
    
    private void findNext() {
        hasNextElement = false;
        while (sourceIterator.hasNext()) {
            T element = sourceIterator.next();
            if (matchesPredicates(element)) {
                nextElement = element;
                hasNextElement = true;
                break;
            }
        }
    }
    
    private boolean matchesPredicates(T element) {
        if (requireAll) {
            // All predicates must be true (AND)
            return predicates.stream().allMatch(p -> p.test(element));
        } else {
            // At least one predicate must be true (OR)
            return predicates.stream().anyMatch(p -> p.test(element));
        }
    }
}

/**
 * Mutable Filtered Iterator that allows changing filters dynamically
 */
class MutableFilteredIterator<T> implements Iterator<T> {
    private final Iterator<T> sourceIterator;
    private Predicate<T> currentPredicate;
    private T nextElement;
    private boolean hasNextElement;
    
    public MutableFilteredIterator(Iterator<T> sourceIterator, Predicate<T> initialPredicate) {
        this.sourceIterator = sourceIterator;
        this.currentPredicate = initialPredicate;
        this.hasNextElement = false;
        findNext();
    }
    
    /**
     * Change the filter predicate dynamically
     */
    public void setPredicate(Predicate<T> newPredicate) {
        this.currentPredicate = newPredicate;
        // Reset and find next element with new predicate
        findNext();
    }
    
    @Override
    public boolean hasNext() {
        return hasNextElement;
    }
    
    @Override
    public T next() {
        if (!hasNextElement) {
            throw new NoSuchElementException();
        }
        
        T result = nextElement;
        findNext();
        return result;
    }
    
    @Override
    public void remove() {
        throw new UnsupportedOperationException("Remove operation not supported by MutableFilteredIterator");
    }
    
    private void findNext() {
        hasNextElement = false;
        while (sourceIterator.hasNext()) {
            T element = sourceIterator.next();
            if (currentPredicate.test(element)) {
                nextElement = element;
                hasNextElement = true;
                break;
            }
        }
    }
}

/**
 * Person class for demonstration
 */
class Person {
    private String name;
    private int age;
    private String profession;
    
    public Person(String name, int age, String profession) {
        this.name = name;
        this.age = age;
        this.profession = profession;
    }
    
    public String getName() { return name; }
    public int getAge() { return age; }
    public String getProfession() { return profession; }
    
    @Override
    public String toString() {
        return String.format("%s (%d, %s)", name, age, profession);
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Person person = (Person) o;
        return age == person.age && 
               Objects.equals(name, person.name) && 
               Objects.equals(profession, person.profession);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(name, age, profession);
    }
}

/**
 * Utility class for creating common filters
 */
class Filters {
    
    /**
     * Create a filter for even numbers
     */
    public static <T extends Number> Predicate<T> evenNumbers() {
        return n -> n.intValue() % 2 == 0;
    }
    
    /**
     * Create a filter for odd numbers
     */
    public static <T extends Number> Predicate<T> oddNumbers() {
        return n -> n.intValue() % 2 != 0;
    }
    
    /**
     * Create a filter for numbers greater than a threshold
     */
    public static <T extends Number> Predicate<T> greaterThan(T threshold) {
        return n -> n.doubleValue() > threshold.doubleValue();
    }
    
    /**
     * Create a filter for numbers less than a threshold
     */
    public static <T extends Number> Predicate<T> lessThan(T threshold) {
        return n -> n.doubleValue() < threshold.doubleValue();
    }
    
    /**
     * Create a filter for strings starting with a prefix
     */
    public static Predicate<String> startsWith(String prefix) {
        return s -> s != null && s.startsWith(prefix);
    }
    
    /**
     * Create a filter for strings containing a substring
     */
    public static Predicate<String> contains(String substring) {
        return s -> s != null && s.contains(substring);
    }
    
    /**
     * Create a filter for strings with length greater than threshold
     */
    public static Predicate<String> longerThan(int length) {
        return s -> s != null && s.length() > length;
    }
    
    /**
     * Create a filter for strings with length less than threshold
     */
    public static Predicate<String> shorterThan(int length) {
        return s -> s != null && s.length() < length;
    }
    
    /**
     * Create a filter that negates another filter
     */
    public static <T> Predicate<T> not(Predicate<T> predicate) {
        return predicate.negate();
    }
    
    /**
     * Create a filter that combines multiple filters with AND logic
     */
    @SafeVarargs
    public static <T> Predicate<T> and(Predicate<T>... predicates) {
        return Arrays.stream(predicates).reduce(Predicate::and).orElse(t -> true);
    }
    
    /**
     * Create a filter that combines multiple filters with OR logic
     */
    @SafeVarargs
    public static <T> Predicate<T> or(Predicate<T>... predicates) {
        return Arrays.stream(predicates).reduce(Predicate::or).orElse(t -> false);
    }
}
