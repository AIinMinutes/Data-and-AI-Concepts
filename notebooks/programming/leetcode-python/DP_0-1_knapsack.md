# Optimizing the 0/1 Knapsack Problem: From Recursive to Memoized Solutions

The 0/1 Knapsack problem is a classic algorithmic challenge that perfectly illustrates the power of dynamic programming optimization. In this post, we'll analyze three different implementations, starting with a basic recursive solution and progressively optimizing it through memoization techniques.

## Problem Setup

We begin with a scenario where we have:
- 30 items with random profits between 1 and 10
- Corresponding weights between 1 and 10
- A knapsack capacity of 10 units

```python
import numpy as np

profits = np.random.randint(1, 10, size=30)
weights = np.random.randint(1, 10, size=30)
capacity = 10
```

## Implementation 1: Basic Recursive Solution

### Step-by-Step Execution
1. **Initial Call**: Function is called with starting index 0 and full capacity
2. **At Each Step**:
   - Consider current item at index i
   - Make two recursive calls:
     a. Skip current item (move to i+1)
     b. Include current item if it fits (move to i+1 with reduced capacity)
   - Return maximum of the two possibilities
3. **Base Case**: When index reaches end of array, return 0
4. **Decision Making**:
   - If item weight > remaining capacity: only consider skip option
   - If item fits: take max of skip and include options

```python
def dfs(i, capacity, profits, weights):
    if i == len(profits):  # Base case
        return 0
    
    # Try skipping current item
    profit_w_skip = dfs(i + 1, capacity, profits, weights)
    
    # Try including current item if it fits
    new_capacity = capacity - weights[i]
    if new_capacity >= 0:
        profit_w_include = profits[i] + dfs(i + 1, new_capacity, profits, weights)
        max_profit = max(profit_w_skip, profit_w_include)
    else:
        max_profit = profit_w_skip
    
    return max_profit
```

**Performance: 6.52 ms per execution**

## Implementation 2: Dictionary Memoization

### Step-by-Step Execution
1. **Initialize**: Create empty dictionary for memoization
2. **At Each Call**:
   - Create key from current (index, capacity)
   - Check if result already in memo dictionary
   - If found: return memoized result
   - If not: compute as before but store result in memo
3. **Memoization Key**: Use tuple of (index, capacity) as dictionary key
4. **Result Storage**: Store max profit for each unique (index, capacity) combination
5. **Memory Usage**: Dictionary grows as new states are explored

```python
def dfs(i, capacity, profits, weights, memo=None):
    if memo is None:
        memo = {}  # Initialize memo dictionary
    
    if i == len(profits):  # Base case
        return 0
    
    key = (i, capacity)  # Create unique key
    if key in memo:  # Check if already computed
        return memo[key]
    
    # Compute as before but store results
    profit_w_skip = dfs(i + 1, capacity, profits, weights, memo)
    profit_w_include = 0
    
    if capacity >= weights[i]:
        profit_w_include = profits[i] + dfs(i + 1, capacity - weights[i], profits, weights, memo)
    
    memo[key] = max(profit_w_skip, profit_w_include)
    return memo[key]
```

**Performance: 105 μs per execution**

## Implementation 3: Array-based Memoization

### Step-by-Step Execution
1. **Initialize**: Create 2D array filled with -1
   - Rows: number of items (n)
   - Columns: capacity + 1
2. **At Each Call**:
   - Use [index][capacity] as direct array lookup
   - Check if value != -1 (indicates computed result)
   - If found: return array value
   - If not: compute and store in array
3. **Array Access**: Direct indexing with O(1) access
4. **Memory Management**: Fixed-size array allocated upfront
5. **Result Updates**: Replace -1 values with computed results

```python
def dfs(i, capacity, profits, weights, memo=None):
    if memo is None:
        # Initialize 2D array with -1
        memo = [[-1] * (capacity + 1) for _ in range(len(profits))]
    
    if i == len(profits):  # Base case
        return 0
    
    # Check if already computed
    if memo[i][capacity] != -1:
        return memo[i][capacity]
    
    # Compute as before but store results
    profit_w_skip = dfs(i + 1, capacity, profits, weights, memo)
    profit_w_include = 0
    
    if capacity >= weights[i]:
        profit_w_include = profits[i] + dfs(i + 1, capacity - weights[i], profits, weights, memo)
    
    memo[i][capacity] = max(profit_w_skip, profit_w_include)
    return memo[i][capacity]
```

**Performance: 86.8 μs per execution**

## Performance Comparison

Let's summarize the performance improvements:

1. Basic Recursive: 6,520 μs
2. Dictionary Memoization: 105 μs (62x faster)
3. Array Memoization: 86.8 μs (75x faster than basic)

## Analysis of Optimizations

### Why Dictionary Memoization Works
The dictionary approach provides a significant speedup by:
- Storing computed results using (index, capacity) tuples as keys
- Eliminating redundant calculations of subproblems
- Reducing time complexity from O(2^n) to O(n*capacity)

### Why Array Memoization is Even Better
The array-based approach provides additional benefits:
- Faster memory access compared to dictionary lookups
- More predictable memory layout
- Reduced memory overhead without the need for tuple key creation
- Better cache locality due to contiguous memory storage

## Example Walkthrough

Let's trace a small example with profits=[2,3,4] weights=[3,2,1] capacity=4:

1. **Basic Recursive**:
   - Explores all possible combinations
   - Makes redundant calls for same states
   - Total recursive calls: ~2^n

2. **Dictionary Memoization**:
   - First call: memo = {}
   - After (0,4) computed: memo[(0,4)] = 7
   - After (1,4) computed: memo[(1,4)] = 7
   - Reuses results instead of recomputing

3. **Array Memoization**:
   - Initial array: [[-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1]]
   - After computing (0,4): memo[0][4] = 7
   - After computing (1,4): memo[1][4] = 7
   - Direct array access for lookups

## Conclusion

This optimization journey demonstrates several key principles of algorithm optimization:
1. The immense power of memoization in dynamic programming
2. The importance of data structure selection for performance
3. How small implementation changes can lead to significant performance gains

While all three implementations produce the same correct result (maximum profit of 21 in our test case), the performance differences are substantial. The final array-based memoization solution provides the best balance of readability and performance.
