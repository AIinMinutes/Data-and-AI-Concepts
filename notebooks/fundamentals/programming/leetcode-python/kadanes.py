import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import random
    random.seed(47)
    return (random,)


@app.cell
def _(random):
    nums = [random.randint(-100, 100) for _ in range(100)]
    return (nums,)


@app.function
def kadane(nums):
    if not nums:
        return 0  # Handling edge case for an empty list

    current_sum = 0
    max_sum = nums[0]  # Start with the first element

    for num in nums:
        # Max of starting a new subarray or extending the current one
        current_sum = max(num, current_sum + num)  
        max_sum = max(max_sum, current_sum)  # Update max_sum

    return max_sum


@app.function
def kadane_sliding_window(nums):
    if not nums:  # Handle edge case for empty input
        return 0, 0

    max_sum = -float('inf')
    cur_sum = 0
    max_L, max_R = 0, 0
    L = 0

    for R, num in enumerate(nums):
        # Reset cur_sum and update L if cur_sum goes negative
        if cur_sum < 0:
            cur_sum = 0
            L = R
        
        # Add the current number to cur_sum
        cur_sum += num

        # Update max_sum and the indices if we find a new maximum
        if cur_sum > max_sum:
            max_sum = cur_sum
            max_L = L
            max_R = R

    return max_L, max_R


@app.cell
def _(nums):
    max_subarray_sum = kadane(nums)
    indices = kadane_sliding_window(nums)
    assert (
        sum(nums[indices[0]: indices[1] + 1]) == max_subarray_sum
    )
    return


if __name__ == "__main__":
    app.run()
