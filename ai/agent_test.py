import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os
import time

def run_game() -> int:
    """Run a single game and return the winner (1 for logic AI, 2 for minimax)"""
    #print("Starting a new game...")
    try:
        # Add timeout of 30 seconds and disable rendering
        result = subprocess.run(
            ['python', 'run_game.py', '--game_type', 'simplified', '--mode', 'logic_ai_vs_minimax'],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        # Parse the output to find the winner
        output = str(result.stdout)  # Convert to string explicitly
        if "Player 2 wins" in output:
            #print("Logic AI won")
            return 2
        elif "Player 1 wins" in output:
            #print("Minimax won")
            return 1
        else:
            #print("Game ended in draw or error")
            return 0  # Draw or error
            
    except subprocess.TimeoutExpired:
        print("Game timed out after 30 seconds")
        return 0
    except Exception as e:
        print(f"Error running game: {str(e)}")
        return 0

def test_weights(weights_list: List[Dict], num_games: int = 100) -> Dict:
    """Test different weight combinations and return win rates"""
    results = {}
    
    for i, weights in enumerate(weights_list, 1):
        print(f"\nTesting weight combination {i}/{len(weights_list)}: {weights}")
        
        # Create a temporary config file with the weights
        config = {
            'w1': weights['w1'],
            'w2': weights['w2'],
            'w3': 1,  # Keep w3 constant
            'w4': weights['w4']
        }
        
        with open('temp_weights.json', 'w') as f:
            json.dump(config, f)
        
        # Run games and collect results
        wins = {1: 0, 2: 0}  # Count wins for each player
        for game in range(num_games):
            #print(f"Running game {game + 1}/{num_games}")
            winner = run_game()
            if winner > 0:
                wins[winner] += 1
            #time.sleep(1)  # Add a small delay between games
        times = 15
        # Calculate win rates
        total_games = sum(wins.values()) if sum(wins.values()) < times else sum(wins.values()) - times
        win_rates = {
            'minimax': wins[1] / total_games 
        }
        
        #print(f"Results for weights {weights}:")
        #print(f"Minimax win rate: {win_rates['minimax']:.2%}")
        
        results[str(weights)] = win_rates
    
    # Clean up temporary file
    if os.path.exists('temp_weights.json'):
        os.remove('temp_weights.json')
    
    return results

def plot_results(results: Dict):
    """Plot the results of weight testing"""
    print("\nGenerating plot...")
    weights = list(results.keys())
    minimax_win_rates = [results[w]['minimax'] for w in weights]
    
    # Convert weight strings to enemy/ally ratios for display
    weight_labels = []
    for w in weights:
        # Parse the weight string to get w1, w2
        w_dict = eval(w)  # Convert string representation of dict to actual dict
        # Format as enemy/ally ratio
        label = f"{w_dict['w2']/w_dict['w1']:.1f}"
        weight_labels.append(label)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(weights))
    width = 0.35
    
    plt.bar(x, minimax_win_rates, width, label='Minimax')
    
    plt.xlabel('Enemy/Ally Weight Ratio')
    plt.ylabel('Win Rate')
    plt.title('Minimax Win Rates for Different Weight Ratios')
    plt.xticks(x, weight_labels, rotation=45)
    plt.legend()
    
    print("Displaying plot...")
    plt.show()

def main():
    print("Starting weight testing...")
    # Define weight combinations to test with evenly spaced enemy/ally ratios from 1 to 70
    ratios = np.linspace(1, 100, 7)  # 7 evenly spaced points from 1 to 70
    weights_list = [
        {'w1': 1, 'w2': int(ratio), 'w4': 0.2}  # Keep w1=1 and w4=0.2 constant
        for ratio in ratios
    ]
    
    print(f"Testing {len(weights_list)} weight combinations...")
    # Run tests
    results = test_weights(weights_list, num_games=35)
    
    # Plot results
    plot_results(results)
    print("Testing complete!")

if __name__ == '__main__':
    main() 