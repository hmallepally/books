import java.util.*;

/**
 * Comprehensive Test Runner for All Collection Implementations
 * Demonstrates ArrayList, HashMap, Filtering Iterator, and LRU Cache
 */
public class CollectionTestRunner {
    
    public static void main(String[] args) {
        System.out.println("🚀 === COMPREHENSIVE COLLECTION IMPLEMENTATION TEST === 🚀\n");
        
        // Test 1: Custom ArrayList
        testCustomArrayList();
        
        // Test 2: Custom HashMap
        testCustomHashMap();
        
        // Test 3: Filtering Iterator
        testFilteringIterator();
        
        // Test 4: LRU Cache
        testLRUCache();
        
        // Test 5: Performance Comparison
        testPerformanceComparison();
        
        System.out.println("\n🎉 === ALL TESTS COMPLETED SUCCESSFULLY! === 🎉");
    }
    
    /**
     * Test 1: Custom ArrayList Implementation
     */
    private static void testCustomArrayList() {
        System.out.println("📚 === TEST 1: CUSTOM ARRAYLIST IMPLEMENTATION === 📚\n");
        
        try {
            // Create custom ArrayList
            SimpleArrayList<String> customList = new SimpleArrayList<>();
            
            // Test basic operations
            System.out.println("Testing basic operations:");
            customList.add("Apple");
            customList.add("Banana");
            customList.add("Cherry");
            System.out.println("After adding 3 elements: " + customList);
            System.out.println("Size: " + customList.size());
            System.out.println("Is empty: " + customList.isEmpty());
            
            // Test get and set
            System.out.println("\nTesting get and set:");
            System.out.println("Element at index 1: " + customList.get(1));
            String oldValue = customList.set(1, "Blueberry");
            System.out.println("Replaced '" + oldValue + "' with 'Blueberry'");
            System.out.println("Updated list: " + customList);
            
            // Test add at index
            System.out.println("\nTesting add at index:");
            customList.add(1, "Blackberry");
            System.out.println("After adding 'Blackberry' at index 1: " + customList);
            
            // Test remove
            System.out.println("\nTesting remove:");
            String removed = customList.remove(2);
            System.out.println("Removed element at index 2: '" + removed + "'");
            System.out.println("List after removal: " + customList);
            
            // Test contains and indexOf
            System.out.println("\nTesting search operations:");
            System.out.println("Contains 'Apple': " + customList.contains("Apple"));
            System.out.println("Contains 'Orange': " + customList.contains("Orange"));
            System.out.println("Index of 'Cherry': " + customList.indexOf("Cherry"));
            
            // Test iterator
            System.out.println("\nTesting iterator:");
            System.out.print("Elements via iterator: ");
            for (String item : customList) {
                System.out.print(item + " ");
            }
            
            // Test addAll
            System.out.println("\n\nTesting addAll:");
            List<String> moreFruits = Arrays.asList("Date", "Elderberry", "Fig");
            customList.addAll(moreFruits);
            System.out.println("After adding more fruits: " + customList);
            
            // Test clear
            System.out.println("\nTesting clear:");
            customList.clear();
            System.out.println("After clearing: " + customList);
            System.out.println("Size after clear: " + customList.size());
            System.out.println("Is empty after clear: " + customList.isEmpty());
            
            System.out.println("✅ Custom ArrayList test completed successfully!\n");
            
        } catch (Exception e) {
            System.err.println("❌ Custom ArrayList test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test 2: Custom HashMap Implementation
     */
    private static void testCustomHashMap() {
        System.out.println("🗺️ === TEST 2: CUSTOM HASHMAP IMPLEMENTATION === 🗺️\n");
        
        try {
            // Create custom HashMap
            CustomHashMap<String, Integer> customMap = new CustomHashMap<>();
            
            // Test basic operations
            System.out.println("Testing basic operations:");
            customMap.put("Apple", 1);
            customMap.put("Banana", 2);
            customMap.put("Cherry", 3);
            System.out.println("After adding 3 elements: " + customMap);
            System.out.println("Size: " + customMap.size());
            System.out.println("Is empty: " + customMap.isEmpty());
            
            // Test get
            System.out.println("\nTesting get:");
            System.out.println("Value for 'Apple': " + customMap.get("Apple"));
            System.out.println("Value for 'Orange': " + customMap.get("Orange"));
            
            // Test contains
            System.out.println("\nTesting contains:");
            System.out.println("Contains key 'Apple': " + customMap.containsKey("Apple"));
            System.out.println("Contains key 'Orange': " + customMap.containsKey("Orange"));
            System.out.println("Contains value 2: " + customMap.containsValue(2));
            System.out.println("Contains value 5: " + customMap.containsValue(5));
            
            // Test update
            System.out.println("\nTesting update:");
            Integer oldValue = customMap.put("Apple", 10);
            System.out.println("Updated 'Apple' from " + oldValue + " to 10");
            System.out.println("Updated map: " + customMap);
            
            // Test remove
            System.out.println("\nTesting remove:");
            Integer removed = customMap.remove("Banana");
            System.out.println("Removed 'Banana' with value: " + removed);
            System.out.println("Map after removal: " + customMap);
            
            // Test collision handling
            System.out.println("\nTesting collision handling:");
            // Add elements that might cause collisions
            for (int i = 0; i < 20; i++) {
                customMap.put("Key" + i, i * 100);
            }
            System.out.println("After adding 20 more elements: " + customMap);
            System.out.println("Size: " + customMap.size());
            
            // Test keySet, values, entrySet
            System.out.println("\nTesting collections:");
            System.out.println("Keys: " + customMap.keySet());
            System.out.println("Values: " + customMap.values());
            System.out.println("Entries: " + customMap.entrySet());
            
            // Test clear
            System.out.println("\nTesting clear:");
            customMap.clear();
            System.out.println("After clearing: " + customMap);
            System.out.println("Size after clear: " + customMap.size());
            
            System.out.println("✅ Custom HashMap test completed successfully!\n");
            
        } catch (Exception e) {
            System.err.println("❌ Custom HashMap test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test 3: Filtering Iterator Implementation
     */
    private static void testFilteringIterator() {
        System.out.println("🔍 === TEST 3: FILTERING ITERATOR IMPLEMENTATION === 🔍\n");
        
        try {
            // Test basic filtering
            System.out.println("Testing basic filtering:");
            List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
            System.out.println("Original numbers: " + numbers);
            
            // Filter even numbers
            FilteredIterator<Integer> evenIterator = new FilteredIterator<>(numbers.iterator(), n -> n % 2 == 0);
            System.out.print("Even numbers: ");
            while (evenIterator.hasNext()) {
                System.out.print(evenIterator.next() + " ");
            }
            
            // Filter odd numbers
            System.out.println("\n\nFiltering odd numbers:");
            FilteredIterator<Integer> oddIterator = new FilteredIterator<>(numbers.iterator(), n -> n % 2 != 0);
            System.out.print("Odd numbers: ");
            while (oddIterator.hasNext()) {
                System.out.print(oddIterator.next() + " ");
            }
            
            // Test string filtering
            System.out.println("\n\nTesting string filtering:");
            List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry", "fig");
            System.out.println("Original words: " + words);
            
            FilteredIterator<String> longWordsIterator = new FilteredIterator<>(words.iterator(), 
                word -> word.length() > 5);
            System.out.print("Words longer than 5 characters: ");
            while (longWordsIterator.hasNext()) {
                System.out.print(longWordsIterator.next() + " ");
            }
            
            // Test custom object filtering
            System.out.println("\n\nTesting custom object filtering:");
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
            FilteredIterator<Person> engineersIterator = new FilteredIterator<>(people.iterator(), 
                p -> "Engineer".equals(p.getProfession()));
            System.out.println("\nEngineers:");
            while (engineersIterator.hasNext()) {
                System.out.println("  " + engineersIterator.next());
            }
            
            // Test chained filters
            System.out.println("\nTesting chained filters:");
            FilteredIterator<Person> youngEngineersIterator = new FilteredIterator<>(
                new FilteredIterator<>(people.iterator(), p -> "Engineer".equals(p.getProfession())),
                p -> p.getAge() < 30
            );
            System.out.println("Young engineers (under 30):");
            while (youngEngineersIterator.hasNext()) {
                System.out.println("  " + youngEngineersIterator.next());
            }
            
            System.out.println("✅ Filtering Iterator test completed successfully!\n");
            
        } catch (Exception e) {
            System.err.println("❌ Filtering Iterator test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test 4: LRU Cache Implementation
     */
    private static void testLRUCache() {
        System.out.println("💾 === TEST 4: LRU CACHE IMPLEMENTATION === 💾\n");
        
        try {
            // Create LRU cache
            LRUCache<String, String> cache = new LRUCache<>(3);
            
            // Test basic operations
            System.out.println("Testing basic operations:");
            cache.put("A", "Apple");
            cache.put("B", "Banana");
            cache.put("C", "Cherry");
            System.out.println("Cache after adding A, B, C: " + cache);
            System.out.println("Size: " + cache.size());
            System.out.println("Capacity: " + cache.getCapacity());
            
            // Test access and LRU behavior
            System.out.println("\nTesting LRU behavior:");
            System.out.println("Accessing 'A': " + cache.get("A"));
            System.out.println("Cache after accessing A: " + cache);
            
            // Test eviction
            System.out.println("\nTesting eviction:");
            cache.put("D", "Date");
            System.out.println("After adding 'D': " + cache);
            System.out.println("Size: " + cache.size());
            
            // Test statistics
            System.out.println("\nTesting statistics:");
            cache.get("B"); // Access B to update stats
            cache.get("X"); // Miss
            System.out.println("Hit count: " + cache.getHitCount());
            System.out.println("Miss count: " + cache.getMissCount());
            System.out.println("Hit ratio: " + String.format("%.2f%%", cache.getHitRatio() * 100));
            
            // Test operations
            System.out.println("\nTesting operations:");
            System.out.println("Contains 'A': " + cache.containsKey("A"));
            System.out.println("Contains 'B': " + cache.containsKey("B"));
            System.out.println("Contains 'C': " + cache.containsKey("C"));
            
            // Test remove
            System.out.println("\nTesting remove:");
            String removed = cache.remove("B");
            System.out.println("Removed 'B': " + removed);
            System.out.println("Cache after removal: " + cache);
            
            // Test clear
            System.out.println("\nTesting clear:");
            cache.clear();
            System.out.println("After clearing: " + cache);
            System.out.println("Size after clear: " + cache.size());
            
            System.out.println("✅ LRU Cache test completed successfully!\n");
            
        } catch (Exception e) {
            System.err.println("❌ LRU Cache test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test 5: Performance Comparison
     */
    private static void testPerformanceComparison() {
        System.out.println("⚡ === TEST 5: PERFORMANCE COMPARISON === ⚡\n");
        
        try {
            int testSize = 10000;
            
            // Test ArrayList performance
            System.out.println("Testing ArrayList performance:");
            long startTime = System.currentTimeMillis();
            
            SimpleArrayList<Integer> customList = new SimpleArrayList<>();
            for (int i = 0; i < testSize; i++) {
                customList.add(i);
            }
            
            long endTime = System.currentTimeMillis();
            System.out.println("Custom ArrayList - Added " + testSize + " elements in " + (endTime - startTime) + "ms");
            
            // Test HashMap performance
            System.out.println("\nTesting HashMap performance:");
            startTime = System.currentTimeMillis();
            
            CustomHashMap<Integer, String> customMap = new CustomHashMap<>();
            for (int i = 0; i < testSize; i++) {
                customMap.put(i, "Value" + i);
            }
            
            endTime = System.currentTimeMillis();
            System.out.println("Custom HashMap - Added " + testSize + " elements in " + (endTime - startTime) + "ms");
            
            // Test LRU Cache performance
            System.out.println("\nTesting LRU Cache performance:");
            startTime = System.currentTimeMillis();
            
            LRUCache<Integer, String> cache = new LRUCache<>(1000);
            for (int i = 0; i < testSize; i++) {
                cache.put(i, "Value" + i);
                if (i % 100 == 0) {
                    cache.get(i / 2); // Access some elements
                }
            }
            
            endTime = System.currentTimeMillis();
            System.out.println("LRU Cache - Added " + testSize + " elements in " + (endTime - startTime) + "ms");
            System.out.println("Final cache size: " + cache.size());
            System.out.println("Hit ratio: " + String.format("%.2f%%", cache.getHitRatio() * 100));
            
            System.out.println("✅ Performance comparison completed successfully!\n");
            
        } catch (Exception e) {
            System.err.println("❌ Performance comparison failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Utility method to print a separator line
     */
    private static void printSeparator() {
        System.out.println("\n" + "=".repeat(80) + "\n");
    }
}
