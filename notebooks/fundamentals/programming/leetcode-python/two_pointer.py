import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two Pointers Algorithm Technique
    ---

    ## Pre-requisite Concepts
    1. **Array Data Structure**
       - Understanding of array indexing
       - Array traversal
       - Array manipulation

    2. **String Operations**
       - Basic string manipulation
       - String indexing

    3. **Sorting**
       - Understanding of sorted arrays
       - Properties of sorted sequences

    4. **Time Complexity**
       - Basic understanding of Big O notation
       - Understanding of O(n) vs O(n²) complexity

    ---

    ## Detailed Notes

    ### 1. Two Pointers Overview
    - Two pointers is a general algorithmic technique
    - Sliding window is considered a subset of two pointer problems
    - Key distinction:
      - Two Pointers: Focus on individual elements at pointer positions
      - Sliding Window: Focus on entire window between pointers

    ### 2. Palindrome Check Example
    #### Approach
    - Initialize two pointers:
      - Left pointer at start
      - Right pointer at end
    - Compare elements at both pointers
    - Move pointers toward center
    - Stop when pointers cross

    #### Time Complexity: O(n)
    #### Space Complexity: O(1)

    ### 3. Two Sum in Sorted Array
    #### Problem Description
    - Given: Sorted array
    - Task: Find two numbers that sum to target
    - Assumption: Exactly one solution exists

    #### Solution Approach
    1. Initialize pointers at array edges
    2. Calculate current sum
    3. If sum > target: decrease right pointer
    4. If sum < target: increase left pointer
    5. If sum == target: found solution

    #### Key Insights
    - Utilizes sorted property of array
    - Eliminates impossible combinations
    - More efficient than brute force O(n²)
    - Called "shrinking window" pattern

    #### Time Complexity: O(n)
    #### Space Complexity: O(1)

    ---

    ## Python Examples

    ### 1. Palindrome Check
    ```python
    def isPalindrome(s: str) -> bool:
        left, right = 0, len(s) - 1

        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1

        return True

    # Example usage
    print(isPalindrome("racecar"))  # True
    print(isPalindrome("hello"))    # False
    ```

    ### 2. Two Sum in Sorted Array
    ```python
    def findTwoSum(nums: list[int], target: int) -> list[int]:
        left, right = 0, len(nums) - 1

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == target:
                return [left, right]
            elif current_sum > target:
                right -= 1
            else:
                left += 1

        return []  # No solution found

    # Example usage
    sorted_array = [-1, 2, 3, 4, 7, 8, 9]
    target = 7
    print(findTwoSum(sorted_array, target))  # [2, 4] (3 + 4 = 7)
    ```

    ### 3. Generic Two Pointer Template
    ```python
    def twoPointerTemplate(arr: list) -> None:
        left = 0
        right = len(arr) - 1

        while left < right:
            # Process elements at left and right pointers

            # Update pointers based on condition
            if some_condition:
                left += 1
            else:
                right -= 1

        # Return result
    ```

    ---

    ## Key Takeaways
    1. Two pointers is an efficient technique for reducing time complexity from O(n²) to O(n)
    2. Particularly useful for:
       - Palindrome problems
       - Finding pairs in sorted arrays
       - Problems requiring comparison of elements from opposite ends
    3. Often eliminates need for extra space, achieving O(1) space complexity
    4. Most effective when input has some order (like sorting)
    """)
    return


if __name__ == "__main__":
    app.run()
