def fibonacci(n):
    """Generate the first n numbers of the Fibonacci sequence."""
    if n <= 0:
        return []
    
    fib_sequence = [0, 1]
    
    while len(fib_sequence) < n:
        next_value = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_value)
    
    return fib_sequence

# Example usage
if __name__ == "__main__":
    try:
        # Get number of terms from user input
        num_terms = int(input("Enter the number of Fibonacci terms to generate: "))
        
        if num_terms < 0:
            print("Please enter a non-negative integer.")
        else:
            fib_series = fibonacci(num_terms)
            
            print(f"\nFirst {num_terms} Fibonacci numbers:")
            for i, num in enumerate(fib_series):
                print(f"F({i}) = {num}")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
