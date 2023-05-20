import matplotlib.pyplot as plt

# Create lists of depth values and execution times for Minimax and Alpha-Beta Pruning
depths = [1, 2, 3]
minimax_vs_random_times = [3.106706142425537, 4.458906173706055, 9.944153547286987]
minimax_vs_alphabeta_times = [3.972822904586792, 6.217182159423828, 21.860193490982056]
alphabeta_times_vs_random_times = [2.492675304412842, 3.502676010131836, 5.263047933578491]

# Create a line graph with two lines, one for Minimax and one for Alpha-Beta Pruning
plt.plot(depths, minimax_vs_random_times, label="Minimax vs Random", color="#1f77b4", linewidth=2)
plt.plot(depths, alphabeta_times_vs_random_times, label="Alpha-Beta Pruning vs Random", color="#ff7f0e", linewidth=2)
plt.plot(depths, minimax_vs_alphabeta_times, label="Minimax vs Alpha-Beta Pruning", color="#B0E0E6", linewidth=2)

# Add a legend to the graph
plt.legend()

# Add labels for the x-axis and y-axis
plt.xlabel("Depth", fontsize=14, fontweight="bold")
plt.ylabel("Execution Time (s)", fontsize=14, fontweight="bold")

# Add a title for the graph
plt.title("Performance Graph", fontsize=16, fontweight="bold")

# Customize the background color and grid lines
plt.rcParams["axes.facecolor"] = "#1f77b4"
plt.grid(axis="y", color="white")

# Display the graph
plt.show()