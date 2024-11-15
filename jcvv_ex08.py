import csv
import numpy as np                                                  # Functions we need in this code
import random
import matplotlib.pyplot as plt

def load_players(file_path):                                        # Function for loading the players_id file 
    players = {}                                                    # Create dictionary to store player detais        

    with open(file_path, 'r', encoding='utf-8-sig') as file:        # Open csv file
        reader = csv.reader(file)                                   # Read the csv file

        for row in reader:                                          # Loop through each row in file
            player_id = row[0]                                      # Get first column as player ID
            player_name = row[1]                                    # Get second column as player name
            players[player_id] = player_name                        # Store ID as key and name as value in dict
            
    return players

def load_stats(file_path):                                          # Function for loading the player_stats file
    stats = {}                                                      # Create dictionary to store the player stats

    with open(file_path, 'r', encoding='utf-8-sig') as file:        # Opening the csv file
        reader = csv.reader(file)                                   # Reading file

        for row in reader:                                          # Loop through each line
            player_id = row[0]                                      # Store 1st coln as player id
            points = float(row[1])                                  # Store 2nd coln as player pts
            assists = float(row[2])                                 # Store 3rd coln as player ast
            rebounds = float(row[3])                                # Store 4th coln as player reb
            stats[player_id] = (points, assists, rebounds)          # Store stats and id as key in dict

    return stats

def initialize_centroids(data, k):                                  # Function to initialize centroid
    centroids = []                                                  # For storing centroids
    for i in range(k):                                              # Traversing number of clusters (k)
        random_index = random.randint(0, len(data)-1)               # Randomly choose index in data set
        centroids.append(data[random_index])                        # Pick that point as centroid
    return centroids

    # predefined_centroids = [                                        # For testing only
    #     (367.0, 249.0, 799.0),
    #     (89.0, 145.0, 251.0),
    #     (83.0, 25.0, 94.0),
    #     (19.0, 25.0, 31.0),
    #     (60.0, 24.0, 222.0),
    #     (554.0, 55.0, 545.0)
    # ]
    # return predefined_centroids[:k]

def compute_distance(point1, point2):                               # Function to compute the distance between 2 points
    distance = 0                                                    # Initialize distance
    for i in range(len(point1)):                                    # Loop through stats (pts, ast, reb)
        distance += (point1[i] - point2[i]) ** 2                    # Calculate the square difference for each dimension
    return distance ** 0.5                                          # Return the square root of sum of squared

def assign_clusters(data, centroids):                               # Function to assign data points to the nearest centroid
    clusters = []                                                   # For storing clusters

    for i in range(len(centroids)):                                 # Initializing empty clusters for each centroid
        clusters.append([])

    for point in data:                                              # Loop through each data point
        min_distance = float('inf')                                 # Start with very large number for comparison
        cluster_index = -1                                          # Start with invalid cluster index

        for i in range(len(centroids)):                             # Loop in each centroid
            distance = compute_distance(point, centroids[i])        # Calculate distance between point and centroid

            if distance < min_distance:                             # Check if distance is the closest centroid
                min_distance = distance                             # Update minimum distance if closest
                cluster_index = i                                   # Update cluster index

        clusters[cluster_index].append(point)                       # Assign point to closest cluster

    return clusters

def update_centroids(clusters):                                     # Function to update centroid by taking average points in each cluster
    new_centroids = []                                              # List to store new centroids

    for cluster in clusters:                                        # Loop through each cluster
        if len(cluster) == 0:                                       # If cluster is empty
            new_centroids.append([0, 0, 0])                         # Assign zero centroid
        else:
            centroid = []                                           # For storing new centroid
            num_points = len(cluster)                               # Get the number of points in cluster

            for i in range(len(cluster[0])):                        # Loop through each dimension (pts, ast, reb)
                sum = 0                                             # Initialize sum for each dimension

                for point in cluster:                               # Loop through each point in cluster
                    sum += point[i]                                 # Add the value of the dimention
                
                centroid.append(sum / num_points)                   # Calculate the average and add to centroid
            new_centroids.append(centroid)                          # Add new centroid in list

    return new_centroids                                            # Return updated centroids

def kmeans(data, k, max_iterations=100):                                # Function to perform k means clustering
    centroids = initialize_centroids(data, k)                           # Call initialize centroid function to initialize centroid first
    initial_centroids = centroids.copy()                                # Save the initial centroid for comparison later

    for iteration in range(max_iterations):                             # Loop for number of iterations
        clusters = assign_clusters(data, centroids)                     # Call for assign cluster to assign data points to cluster based on closest centroid
        new_centroids = update_centroids(clusters)                      # Call for update centroinds to update centroid based on new clusters

        if centroids == new_centroids:                                  # If centroids are not changed, break
            break

        centroids = new_centroids                                       # Set new centroid for next iteration

    return initial_centroids, centroids, clusters                       # Return the following

                                                                        # Function to write output.txt
def save_output(filename, k, initial_centroids, final_centroids, num_iterations, labeled_data):
    with open(filename, 'w') as file:                                   # Open file in write mode
        file.write(f"K-Means Clustering Output\n")                      # Header

        file.write(f"k = {k}\n")                                        # Write the number of clusters (k)

        file.write(f"\nInitial Centroids\n")
        for centroid in initial_centroids:                              # Write the initial centroids
            file.write(f"{tuple(centroid)}\n")

        file.write(f"\nFinal Centroids\n")
        for centroid in final_centroids:                                # Write the final centroid after clustering
            file.write(f"{tuple(centroid)}\n")

        file.write(f"\nNumber of Iterations: {num_iterations}\n")       # Write number of iteration

        file.write(f"\nLabeled Dataset:\n")                             # Write the dataset with cluster labels
        for label, player_id, player_name, point in labeled_data:
            file.write(f"{label} : {player_id} - {player_name} {list(point)}\n")

def plot_clusters(clusters, centroids, output_filename):                # Function to plot cluster in 3d
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#FFA500',             # Color list of clusters (max of 10)
              '#8A2BE2', '#00FF00']
    fig = plt.figure()                                                  # Create new figure
    ax = fig.add_subplot(111, projection='3d')                          # Create 3d plot

    for cluster_index, cluster in enumerate(clusters):                  # Loop through each cluster
        cluster_points = np.array(cluster)                              # Convert cluster to numpy for easy manipulation

        if len(cluster_points) > 0:                                     # If cluster has points
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],      # Plot the points in 3d
                       cluster_points[:, 2], c=colors[cluster_index], label=f"Cluster {cluster_index}")

    for centroid in centroids:                                          # Loop for each centroid
        ax.scatter(*centroid, c='black', marker='x', s=100, label="Centroid")   # Plot centroid as black x
    
    ax.set_xlabel('Points')
    ax.set_ylabel('Assists')                                            # Label the x,y,z axis as the points, assists, rebounds
    ax.set_zlabel('Rebounds')
    ax.legend()                                                         # Show legend

    plt.savefig(output_filename)                                        # Save plot as png file
    plt.show()                                                          # Show plot

def main():
    players = load_players('players_id.csv')                        # Load players id from csv
    stats = load_stats('players_stat.csv')                          # Load players stats from csv
    
    data = []                                                       # Array to store play stats as data points
    player_ids = []                                                 # List of player IDs

    for key in stats:                                               # We will get all the keys in stats
        data.append(stats[key])                                     # Adding keys in stats to data array
        player_ids.append(key)                                      # Appending also that key in player ids

    k = int(input("Enter the number of clusters (k): "))            # Ask for K
    
    if k < 1 or k > 10:                                             # K value should be 1 to 10 only
        print("k must be between 1 and 10.")
        return

    initial_centroids, final_centroids, clusters = kmeans(data, k)  # Perform k means clustering
    
    cluster_labels = {}                                             # List to store player stats with assigned cluster label
    for cluster_index in range(len(clusters)):                      # Loop in each cluster
        for point in clusters[cluster_index]:                       # Loop through each point in cluster
            cluster_labels[tuple(point)] = cluster_index            # Assign cluster label to each data point
    
    labeled_data = []                                               # List to store data with player info and cluster label
    for index, point in enumerate(data):                            # Loop through each player's stats                              # 
        player_id = player_ids[index]                               # Get player id

        if player_id in players:                                    # Check if play is exist
            player_name = players[player_id]                        # Get player name from dict
            cluster_label = cluster_labels[tuple(point)]            # Get assign cluster label for player stats
            labeled_data.append((cluster_label, player_id, player_name, point)) # Store the following in labeled data
    
                                                                    # Call for save output to write a text file
    save_output('output.txt', k, initial_centroids, final_centroids, len(clusters), labeled_data)
    print("output.txt is saved.")                                   # Inform user

    plot_clusters(clusters, final_centroids, 'scatterplot.png')     # Call to generate scatter plot
    print("scatterplot saved as 'scatterplot.png'")                 # Inform user

main()                                                              # Call for menu function