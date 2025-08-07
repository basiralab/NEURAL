# utils.py

import re
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def extract_impression(text):
    if not isinstance(text, str): return ""
    match = re.search(r'IMPRESSION:\s*(.*?)(?=\n[A-Z\s]+:|$)', text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def create_pruned_visual_graph(patch_features, edges):
    x = torch.tensor(patch_features, dtype=torch.float)
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def create_text_knowledge_graph(text, nlp_model):
    doc = nlp_model(text)
    entities = list(set([ent.text.lower() for ent in doc.ents]))
    if not entities: 
        data = from_networkx(nx.Graph())
        data.x = torch.empty((0, 1), dtype=torch.float) # Ensure feature matrix has a feature dimension
        return data

    G = nx.Graph()
    entity_map = {entity: i for i, entity in enumerate(entities)}
    for entity, i in entity_map.items():
        G.add_node(i, label=entity)

    for sentence in [sent.text.lower() for sent in doc.sents]:
        sent_entities = [e for e in entities if e in sentence]
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                G.add_edge(entity_map[sent_entities[i]], entity_map[sent_entities[j]])
    
    data = from_networkx(G)
    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
    return data

def fuse_graphs(g1, g2):
    if g1.num_nodes == 0: return g2
    if g2.num_nodes == 0: return g1
    
    node1_to_connect, node2_to_connect = 0, 0
    if g1.num_nodes > 1:
        g1_nx = nx.Graph()
        g1_nx.add_nodes_from(range(g1.num_nodes))
        g1_nx.add_edges_from(g1.edge_index.t().tolist())
        node1_to_connect = max(nx.betweenness_centrality(g1_nx), key=nx.betweenness_centrality(g1_nx).get)

    if g2.num_nodes > 1:
        g2_nx = nx.Graph()
        g2_nx.add_nodes_from(range(g2.num_nodes))
        g2_nx.add_edges_from(g2.edge_index.t().tolist())
        node2_to_connect = max(nx.betweenness_centrality(g2_nx), key=nx.betweenness_centrality(g2_nx).get)

    num_nodes_g1 = g1.num_nodes
    x_fused = torch.cat([g1.x, g2.x], dim=0)
    
    edge_index_g2_shifted = g2.edge_index + num_nodes_g1 if g2.num_nodes > 0 else g2.edge_index
    edge_index_fused = torch.cat([g1.edge_index, edge_index_g2_shifted], dim=1)
    
    fusion_edge = torch.tensor([[node1_to_connect], [node2_to_connect + num_nodes_g1]], dtype=torch.long)
    final_edge_index = torch.cat([edge_index_fused, fusion_edge], dim=1)
    
    return Data(x=x_fused, edge_index=final_edge_index)