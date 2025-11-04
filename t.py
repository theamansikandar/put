# Import the NumPy library for matrix math
import numpy as np

def run_real_pagerank_with_steps():
    
    # --- 1. Our "Real Data" ---
    page_names = [
        'google.com',
        'wikipedia.org',
        'facebook.com',
        'microsoft.com'
    ]
    num_pages = len(page_names)

    # The "Link Matrix" (L) from our research
    L = np.array([
        [0.0,   1/3,     0.0,     0.0],   # TO google.com
        [0.0,   0.0,     0.0,     0.0],   # TO wikipedia.org
        [0.0,   1/3,     0.0,     0.0],   # TO facebook.com
        [0.0,   1/3,     0.0,     0.0]    # TO microsoft.com
    ])
    
    # --- 2. The Real-World Damping Factor ---
    d = 0.85 # The standard PageRank damping factor
    
    # The "Jump Matrix" (J)
    J = np.full((num_pages, num_pages), 1.0 / num_pages)
    
    # Create the final Stochastic Matrix (M)
    M = np.zeros((num_pages, num_pages))
    
    for i in range(num_pages):
        column = L[:, i]
        if np.sum(column) == 0:
            M[:, i] = J[:, i]
        else:
            M[:, i] = (d * L[:, i]) + ((1 - d) * J[:, i])

    # --- 3. The PageRank Algorithm (Power Iteration) ---
    
    # Start with equal ranks
    ranks = np.full(num_pages, 1.0 / num_pages)
    max_iterations = 100
    tolerance = 0.00001

    print("--- Starting PageRank Calculation ---")
    print(f"Initial Ranks: {ranks}\n")
    
    for i in range(max_iterations):
        
        # "Multiply ranks by the Matrix M"
        new_ranks = M @ ranks
        
        # Calculate the change (norm)
        change = np.linalg.norm(new_ranks - ranks)
        
        # --- NEW CODE: Print results for this iteration ---
        print(f"--- Iteration {i+1} ---")
        for j in range(num_pages):
            print(f"  {page_names[j]}: {new_ranks[j] * 100:.4f}%")
        print(f"  Change (Tolerance): {change:.8f}")
        print("-" * 20) # Separator
        # --- END OF NEW CODE ---

        # Check for stabilization
        if change < tolerance:
            print(f"\nStabilized after {i+1} iterations (Change is less than tolerance).\n")
            ranks = new_ranks
            break
            
        # Update ranks for the next iteration
        ranks = new_ranks

    # --- 4. Final Ranks Found! ---
    print("--- Final Authority Scores Found ---")
    
    final_scores = sorted(zip(page_names, ranks), key=lambda item: item[1], reverse=True)
    
    for page, score in final_scores:
        print(f"{page}: {score * 100:.2f}% Authority")

if __name__ == "__main__":
    run_real_pagerank_with_steps()
