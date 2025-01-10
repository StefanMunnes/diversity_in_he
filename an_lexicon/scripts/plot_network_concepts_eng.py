import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


df_pd = data_lookedup.to_pandas()

edges = []
for idx, row in df_pd.iterrows():
    concept1_tokens = row['individual']
    concept2_tokens = row['collective']
    for token1 in concept1_tokens:
        for token2 in concept2_tokens:
            edges.append((token1, token2))

# Step 2: Graph Creation
edge_counts = Counter(edges)
G_weighted = nx.Graph()
for edge, weight in edge_counts.items():
    G_weighted.add_edge(edge[0], edge[1], weight=weight)

# Step 3: Customizations
# Define colors for concepts
concept1_tokens_set = set(token for tokens in df_pd['individual'] for token in tokens)
concept2_tokens_set = set(token for tokens in df_pd['collective'] for token in tokens)

node_colors = []
for node in G_weighted.nodes():
    if node in concept1_tokens_set:
        node_colors.append('skyblue')
    elif node in concept2_tokens_set:
        node_colors.append('lightgreen')
    else:
        node_colors.append('gray')  # Default color

# Node sizes based on degree
node_sizes = [G_weighted.degree(node) * 50 for node in G_weighted.nodes()]

# Edge widths based on weight
weights = [G_weighted[u][v]['weight'] for u,v in G_weighted.edges()]

# Step 4: Visualization
pos = nx.spring_layout(G_weighted, k=0.5, iterations=50)
nx.draw_networkx_nodes(G_weighted, pos, node_size=node_sizes, node_color=node_colors)
nx.draw_networkx_edges(G_weighted, pos, width=weights)
nx.draw_networkx_labels(G_weighted, pos, font_size=8, font_family='sans-serif')
plt.axis('off')
plt.show()

plt.close()
