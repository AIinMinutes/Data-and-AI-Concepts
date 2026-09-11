# Mastering Stack Operations for Technical Interviews: A Comprehensive Guide

## Introduction to Stacks
A stack is a fundamental data structure that follows the Last-In-First-Out (LIFO) principle. In Python, we can implement stacks using lists, which provide efficient operations for stack manipulation.

### Basic Stack Operations
```python
stack = []
# Push operation - O(1)
stack.append(1)      # stack: [1]
stack.append(2)      # stack: [1, 2]

# Pop operation - O(1)
stack.pop()         # Returns 2, stack: [1]

# Peek/Top operation - O(1)
stack[-1]           # Returns top element

# Check if empty - O(1)
not stack           # Returns True if empty

# Get size - O(1)
len(stack)          # Returns stack length
```

## Common Stack Interview Problems

### 1. Valid Parentheses
**Problem Definition:** Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. The brackets must close in the correct order.

**Example:**
- Input: "{([])}" → Output: True
- Input: "([)]" → Output: False

```python
def check_valid_parantheses(s):
    if len(s) == 0: return False
    if len(s) % 2 == 1: return False
        
    info = {'(': ')', '[': ']', '{': '}'}
    stack = []
    
    for c in s:
        if c in info:
            stack.append(c)
        else:
            if not stack or info[stack[-1]] != c:
                return False
            stack.pop()
    return not stack
```

### 2. Remove Adjacent Duplicates
**Problem Definition:** Given a string, remove all adjacent duplicate characters recursively.

**Example:**
- Input: "abbaca" → Output: "ca"
- Input: "azxxzy" → Output: "ay"

```python
def remove_duplicates(s):
    stack = []
    for c in s:
        if not stack:
            stack.append(c)
        elif stack[-1] == c:
            stack.pop()
        else:
            stack.append(c)
    return ''.join(stack)
```

### 3. Backspace String Compare
**Problem Definition:** Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character.

**Example:**
- Input: s = "ab#c", t = "ad#c" → Output: True
- Input: s = "ab##", t = "c#d#" → Output: True

```python
def process_string(s):
    stack = []
    for c in s:
        if c != '#':
            stack.append(c)
        elif stack:
            stack.pop()
    return ''.join(stack)

def backspace_compare(s, t):
    return process_string(s) == process_string(t)
```

### 4. Simplify Path
**Problem Definition:** Given a string path, which is an absolute path (starting with '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

**Example:**
- Input: "/home/user/Documents/../Pictures" → Output: "/home/user/Pictures"
- Input: "/a/./b/../../c/" → Output: "/c"

```python
def simplify_path(path):
    stack = []
    for portion in path.split('/'):
        if portion in ('', '.'): continue
        elif portion == '..':
            if stack: stack.pop()
        else:
            stack.append(portion)
    return '/' + '/'.join(stack)
```

### 5. Next Greater Element
**Problem Definition:** Given an array, print the Next Greater Element (NGE) for every element. The NGE for an element x is the first greater element on the right side of x in the array.

**Example:**
- Input: [4,1,2,3] → Output: [-1,2,3,-1]
- Input: [2,7,3,1,5] → Output: [7,-1,5,5,-1]

```python
def next_greater_element(arr):
    n = len(arr)
    result = [-1] * n
    stack = []
    
    for i in range(n-1, -1, -1):
        while stack and stack[-1] <= arr[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(arr[i])
    
    return result
```

### 6. Min Stack
**Problem Definition:** Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

**Example Operations:**
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
        
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
            
    def pop(self):
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        return self.stack.pop()
        
    def top(self):
        return self.stack[-1]
        
    def getMin(self):
        return self.min_stack[-1]
```

### 7. Daily Temperatures
**Problem Definition:** Given an array of integers temperatures represents daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.

**Example:**
- Input: [73,74,75,71,69,72,76,73] → Output: [1,1,4,2,1,1,0,0]

```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    stack = []
    
    for i in range(n):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev_day = stack.pop()
            result[prev_day] = i - prev_day
        stack.append(i)
    
    return result
```

### 8. Evaluate Reverse Polish Notation
**Problem Definition:** Evaluate the value of an arithmetic expression in Reverse Polish Notation. Valid operators are +, -, *, and /. Each operand may be an integer or another expression.

**Example:**
- Input: ["2","1","+","3","*"] → Output: 9
- Input: ["4","13","5","/","+"] → Output: 6

```python
def evalRPN(tokens):
    stack = []
    operations = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: int(x / y)
    }
    
    for token in tokens:
        if token in operations:
            b = stack.pop()
            a = stack.pop()
            stack.append(operations[token](a, b))
        else:
            stack.append(int(token))
            
    return stack[0]
```

## Common Stack Patterns and Techniques

### 1. Monotonic Stack Pattern
Used in problems like Next Greater Element and Daily Temperatures. Key characteristics:
- Maintains elements in increasing/decreasing order
- Pops elements that violate the order
- Useful for finding next/previous greater/smaller elements

### 2. Two Stack Pattern
Used in problems like Min Stack and Calculator problems. Key characteristics:
- One stack maintains actual elements
- Second stack maintains additional information
- Both stacks work in sync

### 3. String Processing Pattern
Used in problems like Remove Duplicates and Backspace Compare. Key characteristics:
- Process string character by character
- Stack maintains state of valid characters
- Pop when finding matching/canceling characters

## Interview Tips and Best Practices

### 1. Problem Analysis
- Always clarify input constraints
- Discuss edge cases before coding
- Consider stack when you see:
  - Matching pairs (parentheses, tags)
  - Next greater/smaller elements
  - Processing strings character by character
  - Need to maintain ordered state
  - Need to track historical information

### 2. Implementation Tips
- Use clear variable names
- Handle edge cases first
- Consider using helper functions for clarity
- Test with example cases
- Consider space and time complexity

### 3. Time/Space Complexity
Most stack operations are:
- Push/Pop/Top: O(1) time
- Space complexity usually O(n) where n is input size
- Some problems might require multiple passes: O(n) time

## Conclusion
Understanding stack operations and common patterns is crucial for technical interviews. Practice these problems to build intuition for when and how to use stacks effectively.
