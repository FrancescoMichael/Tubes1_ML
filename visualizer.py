import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class ANNVisualizer:
    def __init__(self, model):
        self.model = model
        self.graph = nx.DiGraph()
        self._build_graph()
    
    def _build_graph(self):
        for layer_idx, num_neurons in enumerate(self.model.neurons):
            for neuron_idx in range(num_neurons):
                self.graph.add_node(
                    (layer_idx, neuron_idx),
                    layer=layer_idx,
                    neuron_idx=neuron_idx,
                )

        for layer_idx in range(len(self.model.neurons) - 1):
            for src_neuron in range(self.model.neurons[layer_idx]):
                for dest_neuron in range(self.model.neurons[layer_idx + 1]):
                    weight = self.model.weights[layer_idx][src_neuron, dest_neuron]
                    self.graph.add_edge(
                        (layer_idx, src_neuron),
                        (layer_idx + 1, dest_neuron),
                        weight=weight
                    )
    
    def visualize_network(self, figsize=(20, 10), node_size=1000, font_size=8):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

        activations_used = set(self.model.activations)

        pos = {}
        layer_dist = 2.0
        neuron_dist = 1.2

        for layer_idx, num_neurons in enumerate(self.model.neurons):
            y_offset = -(num_neurons * neuron_dist) / 2
            for neuron_idx in range(num_neurons):
                pos[(layer_idx, neuron_idx)] = (
                    layer_idx * layer_dist,
                    y_offset + neuron_idx * neuron_dist
                )

        activation_colors = {
            "linear": "blue",
            "relu": "red",
            "sigmoid": "green",
            "tanh": "purple",
            "softmax": "orange",
            "softplus": "cyan",
            "leaky_relu": "pink",
            "mish": "brown"
        }

        node_colors = []
        for layer_idx in range(len(self.model.neurons)):
            activation = self.model.activations[layer_idx] if layer_idx < len(self.model.activations) else "linear"
            color = activation_colors.get(activation, "gray")
            node_colors.extend([color] * self.model.neurons[layer_idx])

        for ax in (ax1, ax2):
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_size=node_size,
                node_color=node_colors,
                alpha=0.8,
                ax=ax
            )
            nx.draw_networkx_labels(
                self.graph, pos,
                font_size=font_size,
                font_color='white',
                font_weight='bold',
                ax=ax
            )

        edges = self.graph.edges()

        weights = [self.graph[u][v]['weight'] for u, v in edges]
        if weights:
            weights_norm = (np.array(weights) - min(weights)) / (max(weights) - min(weights) + 1e-10)
            weight_colors = plt.cm.coolwarm(weights_norm)
            weight_widths = 0.5 + 2 * weights_norm

            nx.draw_networkx_edges(
                self.graph, pos,
                edge_color=weight_colors,
                width=weight_widths,
                edge_cmap=plt.cm.coolwarm,
                edge_vmin=min(weights),
                edge_vmax=max(weights),
                arrows=False,
                alpha=0.7,
                ax=ax1,
                connectionstyle="arc3,rad=0"
            )
            weight_mappable = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                                                norm=plt.Normalize(min(weights), max(weights)))
            plt.colorbar(weight_mappable, ax=ax1, orientation='horizontal', 
                        pad=0.05, aspect=40, label='Weight Values')

        ax1.set_title("Neural Network Weights")
        ax1.axis('off')

        if hasattr(self.model, 'weight_gradients'):
            grads = []
            for i, (u, v) in enumerate(edges):
                layer = u[0]
                src_neuron = u[1]
                dest_neuron = v[1]
                grad = self.model.weight_gradients[layer][src_neuron, dest_neuron]
                grads.append(grad)

            if grads:
                grads_norm = (np.array(grads) - min(grads)) / (max(grads) - min(grads) + 1e-10)
                grad_colors = plt.cm.PiYG(grads_norm)
                grad_widths = 0.5 + 2 * grads_norm

                nx.draw_networkx_edges(
                    self.graph, pos,
                    edge_color=grad_colors,
                    width=grad_widths,
                    edge_cmap=plt.cm.PiYG,
                    edge_vmin=min(grads),
                    edge_vmax=max(grads),
                    arrows=False,
                    alpha=0.7,
                    ax=ax2,
                    connectionstyle="arc3,rad=0"
                )
                grad_mappable = plt.cm.ScalarMappable(cmap=plt.cm.PiYG, 
                                                    norm=plt.Normalize(min(grads), max(grads)))
                plt.colorbar(grad_mappable, ax=ax2, orientation='horizontal', 
                            pad=0.05, aspect=40, label='Gradient Values')

        ax2.set_title("Neural Network Gradients")
        ax2.axis('off')

        legend_labels = {act: color for act, color in activation_colors.items() if act in activations_used}

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=act)
                for act, color in legend_labels.items()]

        ax1.legend(handles=handles, loc='upper right', title="Activation Functions")

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

    def plot_neuron_analysis(self, layer_idx, neuron_idx):
        if not hasattr(self.model, 'weights'):
            print("No weight data available")
            return
        
        has_gradients = hasattr(self.model, 'weight_gradients')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.set_title(f"Weights for Neuron {neuron_idx} in Layer {layer_idx}")
        
        if layer_idx > 0:
            incoming = self.model.weights[layer_idx-1][:, neuron_idx]
            ax.bar(range(len(incoming)), incoming, alpha=0.7, label="Incoming Weights")
        
        if layer_idx < len(self.model.weights) - 1:
            outgoing = self.model.weights[layer_idx][neuron_idx, :]
            offset = len(incoming) if layer_idx > 0 else 0
            ax.bar(range(offset, offset + len(outgoing)), outgoing, alpha=0.7, label="Outgoing Weights")

        ax.set_xlabel("Connection Index")
        ax.set_ylabel("Weight Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.set_title(f"Gradients for Neuron {neuron_idx} in Layer {layer_idx}")

        if has_gradients:
            if layer_idx > 0:
                incoming = self.model.weight_gradients[layer_idx-1][:, neuron_idx]
                ax.bar(range(len(incoming)), incoming, alpha=0.7, label="Incoming Gradients")
            
            if layer_idx < len(self.model.weight_gradients) - 1:
                outgoing = self.model.weight_gradients[layer_idx][neuron_idx, :]
                offset = len(incoming) if layer_idx > 0 else 0
                ax.bar(range(offset, offset + len(outgoing)), outgoing, alpha=0.7, label="Outgoing Gradients")
        
        else:
            ax.text(0.5, 0.5, "No gradient data available", fontsize=12, ha='center', va='center')
        
        ax.set_xlabel("Connection Index")
        ax.set_ylabel("Gradient Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def visualize_all(self):
        self.visualize_network()
        self.plot_weight_distribution()
        
        if hasattr(self.model, 'weight_gradients'):
            self.plot_gradient_distribution()