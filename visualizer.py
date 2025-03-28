import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

class ANNVisualizer:
    def __init__(self, model):
        self.model = model
        self.graph = nx.DiGraph()
        self._build_graph()
    
    def _build_graph(self):
        # Add nodes for each layer
        for layer_idx, num_neurons in enumerate(self.model.neurons):
            for neuron_idx in range(num_neurons):
                self.graph.add_node((layer_idx, neuron_idx), 
                                  layer=layer_idx,
                                  neuron_idx=neuron_idx)
        
        # Add edges between layers with weights
        for layer_idx in range(len(self.model.neurons)-1):
            for src_neuron in range(self.model.neurons[layer_idx]):
                for dest_neuron in range(self.model.neurons[layer_idx+1]):
                    weight = self.model.weights[layer_idx][src_neuron, dest_neuron]
                    self.graph.add_edge(
                        (layer_idx, src_neuron),
                        (layer_idx+1, dest_neuron),
                        weight=weight
                    )
    
    def visualize_network(self, figsize=(15, 10), node_size=1000, font_size=8):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create layered layout
        pos = {}
        layer_dist = 2.0  # Increased distance for better visibility
        neuron_dist = 1.2
        
        for layer_idx, num_neurons in enumerate(self.model.neurons):
            y_offset = -(num_neurons * neuron_dist) / 2
            for neuron_idx in range(num_neurons):
                pos[(layer_idx, neuron_idx)] = (
                    layer_idx * layer_dist,
                    y_offset + neuron_idx * neuron_dist
                )
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.model.neurons)))
        node_colors = [colors[data['layer']] for _, data in self.graph.nodes(data=True)]
        
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_size,
            node_color=node_colors,
            alpha=0.8,
            ax=ax
        )
        
        # Prepare edge visualization
        edges = self.graph.edges()
        
        # Draw weight edges
        weights = [self.graph[u][v]['weight'] for u,v in edges]
        weight_mappable = None
        if len(weights) > 0:
            # Normalize weights for coloring and width
            weights_norm = (np.array(weights) - min(weights)) / (max(weights) - min(weights) + 1e-10)
            weight_colors = plt.cm.coolwarm(weights_norm)
            weight_widths = 0.5 + 2 * weights_norm  # Vary width by weight magnitude
            
            weight_edges = nx.draw_networkx_edges(
                self.graph, pos,
                edge_color=weight_colors,
                width=weight_widths,
                edge_cmap=plt.cm.coolwarm,
                edge_vmin=min(weights),
                edge_vmax=max(weights),
                arrows=False,
                alpha=0.7,
                label='Weights',
                ax=ax
            )
            weight_mappable = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                                                norm=plt.Normalize(min(weights), max(weights)))
        
        # Draw gradient edges if available
        grad_mappable = None
        if hasattr(self.model, 'weight_gradients'):
            grads = []
            for i, (u, v) in enumerate(edges):
                layer = u[0]  # Get layer index from node tuple
                src_neuron = u[1]
                dest_neuron = v[1]
                grad = self.model.weight_gradients[layer][src_neuron, dest_neuron]
                grads.append(grad)
            
            if len(grads) > 0:
                # Normalize gradients for coloring and style
                grads_norm = (np.array(grads) - min(grads)) / (max(grads) - min(grads) + 1e-10)
                grad_colors = plt.cm.viridis(grads_norm)
                grad_widths = 0.5 + 2 * grads_norm
                
                # Draw gradient edges as dashed lines
                grad_edges = nx.draw_networkx_edges(
                    self.graph, pos,
                    edge_color=grad_colors,
                    width=grad_widths,
                    style='dashed',
                    edge_cmap=plt.cm.viridis,
                    edge_vmin=min(grads),
                    edge_vmax=max(grads),
                    arrows=False,
                    alpha=0.7,
                    label='Gradients',
                    ax=ax
                )
                grad_mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                                    norm=plt.Normalize(min(grads), max(grads)))
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=font_size,
            font_color='white',
            font_weight='bold',
            ax=ax
        )
        
        # Create legends
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'Layer {i}',
                markerfacecolor=colors[i], markersize=10)
            for i in range(len(self.model.neurons))
        ]
        
        # Add edge legends
        legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='Weights'))
        if hasattr(self.model, 'weight_gradients'):
            legend_elements.append(Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='Gradients'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add colorbars
        if weight_mappable is not None:
            plt.colorbar(weight_mappable, ax=ax, orientation='horizontal', 
                        pad=0.05, aspect=40, label='Weight Values')
        
        if grad_mappable is not None:
            plt.colorbar(grad_mappable, ax=ax, orientation='horizontal', 
                        pad=0.15, aspect=40, label='Gradient Values')
        
        ax.set_title("Neural Network Architecture with Weights and Gradients")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_weight_distribution(self, layers=None):
        if layers is None:
            layers = range(len(self.model.weights))
        
        plt.figure(figsize=(12, 6))
        
        for layer_idx in layers:
            weights = self.model.weights[layer_idx].flatten()
            plt.hist(
                weights,
                bins=50,
                alpha=0.5,
                label=f'Layer {layer_idx}'
            )
        
        plt.title("Weight Distribution by Layer")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_gradient_distribution(self, layers=None):
        if not hasattr(self.model, 'weight_gradients'):
            print("No gradient data available - train the model first")
            return
            
        if layers is None:
            layers = range(len(self.model.weight_gradients))
        
        plt.figure(figsize=(12, 6))
        
        for layer_idx in layers:
            grads = self.model.weight_gradients[layer_idx].flatten()
            plt.hist(
                grads,
                bins=50,
                alpha=0.5,
                label=f'Layer {layer_idx}'
            )
        
        plt.title("Gradient Distribution by Layer")
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_neuron_weights(self, layer_idx, neuron_idx):
        plt.figure(figsize=(12, 5))
        
        # Plot incoming weights (from previous layer)
        if layer_idx > 0:
            incoming = self.model.weights[layer_idx-1][:, neuron_idx]
            plt.bar(range(len(incoming)), incoming, 
                   alpha=0.7, label='Incoming Weights')
        
        # Plot outgoing weights (to next layer)
        if layer_idx < len(self.model.weights)-1:
            outgoing = self.model.weights[layer_idx][neuron_idx, :]
            offset = len(incoming) if layer_idx > 0 else 0
            plt.bar(range(offset, offset+len(outgoing)), outgoing,
                   alpha=0.7, label='Outgoing Weights')
        
        plt.title(f"Weights for Neuron {neuron_idx} in Layer {layer_idx}")
        plt.xlabel("Connection Index")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_neuron_gradients(self, layer_idx, neuron_idx):
        if not hasattr(self.model, 'weight_gradients'):
            print("No gradient data available - train the model first")
            return
            
        plt.figure(figsize=(12, 5))
        
        # Plot incoming gradients (from previous layer)
        if layer_idx > 0:
            incoming = self.model.weight_gradients[layer_idx-1][:, neuron_idx]
            plt.bar(range(len(incoming)), incoming,
                   alpha=0.7, label='Incoming Gradients')
        
        # Plot outgoing gradients (to next layer)
        if layer_idx < len(self.model.weight_gradients)-1:
            outgoing = self.model.weight_gradients[layer_idx][neuron_idx, :]
            offset = len(incoming) if layer_idx > 0 else 0
            plt.bar(range(offset, offset+len(outgoing)), outgoing,
                   alpha=0.7, label='Outgoing Gradients')
        
        plt.title(f"Gradients for Neuron {neuron_idx} in Layer {layer_idx}")
        plt.xlabel("Connection Index")
        plt.ylabel("Gradient Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def visualize_all(self):
        self.visualize_network()
        self.plot_weight_distribution()
        
        if hasattr(self.model, 'weight_gradients'):
            self.plot_gradient_distribution()