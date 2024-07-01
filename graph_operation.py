# For graph format
# graph = {
#     (1, 2): [(3, 4), (5, 6)],
#     (3, 4): [(1, 2), (4, 5), (3, 6)],
#     (4, 5): []
# }

# Function to add connections
def add_connection(graph, from_node, to_node):
    if from_node in graph:
        graph[from_node].append(to_node)
    else:
        graph[from_node] = [to_node]

# Function to delete a specific connection
def delete_connection(graph, from_node, to_node):
    if from_node in graph and to_node in graph[from_node]:
        graph[from_node].remove(to_node)
        # Optionally, remove the node key if no more connections left
        if not graph[from_node]:
            del graph[from_node]

# Function to delete an entire node
def delete_node(graph, node):
    if node in graph:
        del graph[node]
    # Also remove any connections to this node from other nodes
    for key in list(graph.keys()):  # Use list to avoid dictionary size change during iteration
        if node in graph[key]:
            graph[key].remove(node)
            if not graph[key]:  # Remove key if no connections left
                del graph[key]
