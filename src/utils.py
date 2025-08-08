"""
This file provides utility functions for text processing and graph manipulation
that support the main pipeline scripts (stage1_train.py and stage2_train.py).

The functions handle tasks such as extracting specific sections from clinical
reports and creating or modifying graph structures using NetworkX and PyTorch
Geometric.
"""

import re
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def extract_impression(text: str) -> str:
    """
    Extracts the 'IMPRESSION' section from the full text of a radiology report.

    The 'IMPRESSION' section typically contains the most critical diagnostic
    summary, making it the most valuable part of the report for many analyses.

    Args:
        text (str): The full, unstructured text of the radiology report.

    Returns:
        str: The extracted impression text, stripped of leading/trailing
             whitespace. If the 'IMPRESSION' section is not found, the
             original text is returned as a fallback.
    """
    if not isinstance(text, str):
        return ""

    # This regular expression looks for the 'IMPRESSION:' header and captures
    # all text until it hits another major section header (e.g., 'FINDINGS:')
    # or the end of the string. It's case-insensitive and handles multiline text.
    match = re.search(r'IMPRESSION:\s*(.*?)(?=\n[A-Z\s]+:|$)', text, re.IGNORECASE | re.DOTALL)

    return match.group(1).strip() if match else text.strip()


def create_pruned_visual_graph(patch_features, edges: list) -> Data:
    """
    Constructs a PyTorch Geometric graph from visual patch features and edges.

    This function takes the features of the diagnostically important image
    patches (nodes) and their spatial connections (edges) and converts them
    into a PyG `Data` object, which is the standard graph format for the library.

    Args:
        patch_features (list or np.ndarray): A list of feature vectors, where
                                             each vector corresponds to a node.
        edges (list): A list of tuples, where each tuple `(u, v)` represents
                      an edge between node `u` and node `v`.

    Returns:
        torch_geometric.data.Data: A graph object containing node features (`x`)
                                   and edge connectivity (`edge_index`).
    """
    # Convert node features to a floating-point tensor.
    x = torch.tensor(patch_features, dtype=torch.float)

    # Convert the list of edge tuples into the COO format required by PyG,
    # which is a tensor of shape [2, num_edges].
    # Handle the case where there are no edges.
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def create_text_knowledge_graph(text: str, nlp_model) -> Data:
    """
    Converts a piece of clinical text into a knowledge graph.

    The graph is constructed by:
    1.  Identifying clinical entities (e.g., 'pleural effusion') using an NLP model.
    2.  Creating a node for each unique entity.
    3.  Creating edges between entities that co-occur within the same sentence.

    Args:
        text (str): The input clinical text (e.g., the impression section).
        nlp_model: A loaded spaCy model used for Named Entity Recognition (NER).

    Returns:
        torch_geometric.data.Data: The resulting knowledge graph. Node features
                                   are initialized to 1s as a placeholder.
    """
    # Process the text with the spaCy model to find entities.
    doc = nlp_model(text)
    entities = list(set([ent.text.lower() for ent in doc.ents]))

    # If no entities are found, return an empty graph.
    if not entities:
        data = from_networkx(nx.Graph())
        # Ensure the feature matrix has the correct shape for later concatenation.
        data.x = torch.empty((0, 1), dtype=torch.float)
        return data

    # Build a graph using NetworkX.
    G = nx.Graph()
    entity_map = {entity: i for i, entity in enumerate(entities)}
    for entity, i in entity_map.items():
        G.add_node(i, label=entity)

    # Add edges for entities that co-occur in the same sentence.
    for sentence in [sent.text.lower() for sent in doc.sents]:
        sent_entities = [e for e in entities if e in sentence]
        # Create a fully connected clique between all entities in the sentence.
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                G.add_edge(entity_map[sent_entities[i]], entity_map[sent_entities[j]])

    # Convert the NetworkX graph to a PyG Data object.
    data = from_networkx(G)
    # Assign a dummy feature vector of [1] to each node.
    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
    return data


def fuse_graphs(g1: Data, g2: Data) -> Data:
    """
    Fuses two graphs (e.g., a visual and a text graph) into a single graph.

    The fusion strategy connects the two graphs via a single, semantically
    meaningful edge. This edge links the most central node from each graph,
    determined by 'betweenness centrality'. This is an efficient method to
    integrate modalities without creating quadratic complexity.

    Args:
        g1 (torch_geometric.data.Data): The first graph (e.g., visual graph).
        g2 (torch_geometric.data.Data): The second graph (e.g., text graph).

    Returns:
        torch_geometric.data.Data: The new, unified graph.
    """
    # Handle cases where one of the graphs is empty.
    if g1.num_nodes == 0: return g2
    if g2.num_nodes == 0: return g1

    # Find the most central node in the first graph.
    node1_to_connect = 0
    if g1.num_nodes > 1:
        # Create a temporary NetworkX graph to calculate centrality.
        g1_nx = nx.Graph()
        g1_nx.add_nodes_from(range(g1.num_nodes))
        g1_nx.add_edges_from(g1.edge_index.t().tolist())
        node1_to_connect = max(nx.betweenness_centrality(g1_nx), key=nx.betweenness_centrality(g1_nx).get)

    # Find the most central node in the second graph.
    node2_to_connect = 0
    if g2.num_nodes > 1:
        g2_nx = nx.Graph()
        g2_nx.add_nodes_from(range(g2.num_nodes))
        g2_nx.add_edges_from(g2.edge_index.t().tolist())
        node2_to_connect = max(nx.betweenness_centrality(g2_nx), key=nx.betweenness_centrality(g2_nx).get)

    # Combine node features.
    x_fused = torch.cat([g1.x, g2.x], dim=0)

    # Combine edge indices, shifting the indices of the second graph.
    num_nodes_g1 = g1.num_nodes
    edge_index_g2_shifted = g2.edge_index + num_nodes_g1 if g2.num_nodes > 0 else g2.edge_index
    edge_index_fused = torch.cat([g1.edge_index, edge_index_g2_shifted], dim=1)

    # Create the single fusion edge connecting the two central nodes.
    fusion_edge = torch.tensor([[node1_to_connect], [node2_to_connect + num_nodes_g1]], dtype=torch.long)
    final_edge_index = torch.cat([edge_index_fused, fusion_edge], dim=1)

    return Data(x=x_fused, edge_index=final_edge_index)