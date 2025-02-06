from django.shortcuts import render

# Create your views here.
import random
import string
from rest_framework.views import APIView
from rest_framework.response import Response
import os
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from io import BytesIO
import json
import urllib.request
import ast
import networkx as nx
import numpy as np
import scipy.sparse as sp
from tensorflow.keras.models import load_model
import tensorflow as tf
import traceback

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
tf.random.set_seed(42)

help_dict = {
    'Module': 0,
    'In': 0,
    'Out': 1,
    'If': 2,
    'While': 3,
    'For': 4,
    'Expr': 5,
    'recursive': 6
}

highest_values = {
    "Out": 0,
    "If": 0,
    "While": 0,
    "For": 0,
    "Expr": 0,
    "recursive": 0,
}



class TreeNodeLater:
    def __init__(self, node_type, cognitive_weight, children=None):
        self.node_type = node_type
        self.cognitive_weight = cognitive_weight
        self.children = children if children is not None else []
        self.node_feature = [0,0,0,0,0,0,0]

    def __repr__(self):
        return f"TreeNodeLater({self.node_type}, {self.cognitive_weight}, {self.children})"

class TreeNode:
    def __init__(self, node_type, cognitive_weight):
        self.node_type = node_type
        self.cognitive_weight = cognitive_weight
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self, level=0):
        ret = " " * (level * 2) + f"{self.node_type} (Weight: {self.cognitive_weight})\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def to_dict(self):
        return {
            "node_type": self.node_type,
            "cognitive_weight": self.cognitive_weight,
            "children": [child.to_dict() for child in self.children],
        }

def build_tree(node, current_function=None):
    cognitive_weights = {
        ast.For: 3,
        ast.While: 3,
        ast.If: 2,
        ast.Expr: 2,
    }

    if isinstance(node, ast.Module):
        tree_node = TreeNode("Module", 0)
        for child in ast.iter_child_nodes(node):
            child_tree_node = build_tree(child, current_function)
            if child_tree_node:
                tree_node.add_child(child_tree_node)

        if tree_node.children:
            sum_child_weights = sum(child.cognitive_weight for child in tree_node.children)
            tree_node.cognitive_weight += sum_child_weights

        return tree_node

    elif isinstance(node, ast.FunctionDef):
        tree_node = TreeNode("FunctionDef", 0)
        current_function = node.name

        for child in ast.iter_child_nodes(node):
            child_tree_node = build_tree(child, current_function)
            if child_tree_node:
                tree_node.add_child(child_tree_node)

        if tree_node.children:
            sum_child_weights = sum(child.cognitive_weight for child in tree_node.children)
            tree_node.cognitive_weight += sum_child_weights

        return tree_node

    elif isinstance(node, (ast.For, ast.If, ast.While)):
        weight = cognitive_weights[type(node)]
        tree_node = TreeNode(type(node).__name__, weight)

        for child in ast.iter_child_nodes(node):
            child_tree_node = build_tree(child, current_function)
            if child_tree_node:
                tree_node.add_child(child_tree_node)

        if tree_node.children:
            sum_child_weights = sum(child.cognitive_weight for child in tree_node.children)
            tree_node.cognitive_weight *= sum_child_weights

        return tree_node

    elif isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == current_function:
                return TreeNode("recursive", 3)
            else:
                return TreeNode("Expr", 2)
        return None

    return None

def parse_tree(node, highest_values):
    # Get the node type and cognitive weight
    node_type = node.get("node_type")
    cognitive_weight = node.get("cognitive_weight", 0)

    # Update the highest value for the respective node type
    if node_type in highest_values:
        highest_values[node_type] = max(highest_values[node_type], cognitive_weight)
    # else:
    #     highest_values[node_type] = cognitive_weight

    # Recursively process children if they exist
    children = node.get("children", [])
    for child in children:
        parse_tree(child, highest_values)

def build_tree_later(data):
    # Base case: if data is not a dict, return None
    if isinstance(data, dict):
        node_type = data.get("node_type")
        cognitive_weight = data.get("cognitive_weight", 0)
        children_data = data.get("children", [])
        # Create a TreeNodeLater and recursively build children
        children = [build_tree_later(child) for child in children_data]
        return TreeNodeLater(node_type, cognitive_weight, children)
    return None

def calculate_node_features(node,parent_type):
    if node.node_type == 'Expr' and parent_type == 'For':
        node.node_feature = [0,0,0,0,highest_values['For'],2,0]
    if node.node_type == 'FunctionDef':
        node.node_feature = [0,0,0,0,0,0,0]
        for child in node.children:
            calculate_node_features(child, node.node_type)
    elif node is not None:
        if node.node_type == 'FunctionDef':
            node.node_feature = [0,0,0,0,0,0,0]
        else:
            node.node_feature[help_dict[node.node_type]] = highest_values[node.node_type]
        for child in node.children:
            if child.node_type != node.node_type and child.node_type != 'FunctionDef':
                node.node_feature[help_dict[child.node_type]] = highest_values[child.node_type]
            calculate_node_features(child, node.node_type)

def create_tree_data_with_features(node):
    if node is not None:
        new_data = {
            "node_type": node.node_type,
            "cognitive_weight": node.cognitive_weight,
            "node_feature": node.node_feature,
            "children": [create_tree_data_with_features(child) for child in node.children]
        }
        return new_data
    return None

def create_tree_from_code(code):
    tree = ast.parse(code)
    return build_tree(tree)

def build_graph_from_dict(data, graph=None, parent_id=None, node_id=0):
    """
    Recursively builds a NetworkX graph from a nested dictionary.

    :param data: The dictionary defining the graph structure.
    :param graph: An instance of a NetworkX graph.
    :param parent_id: The ID of the parent node (used to add edges).
    :param node_id: The current node ID.
    :return: A tuple containing the graph and the next available node ID.
    """
    if graph is None:
        graph = nx.DiGraph()

    # Add the current node with attributes
    graph.add_node(node_id,
                   node_type=data['node_type'],
                   cognitive_weight=data['cognitive_weight'],
                   node_feature=data['node_feature'])

    # If there is a parent, add an edge from the parent to the current node
    if parent_id is not None:
        graph.add_edge(parent_id, node_id)

    # Process children recursively
    for child in data.get('children', []):
        node_id += 1
        graph, node_id = build_graph_from_dict(child, graph, parent_id=node_id-1, node_id=node_id)

    return graph, node_id

def get_adjacency_and_features(graph):
    """
    Generates the adjacency matrix A and feature matrix X from a NetworkX graph.

    :param graph: A NetworkX graph.
    :return: A tuple (A, X) where A is the adjacency matrix and X is the node feature matrix.
    """
    # Adjacency matrix
    A = nx.adjacency_matrix(graph).toarray()

    # Add self-loops to A
    np.fill_diagonal(A, 1)

    # Feature matrix
    features = []
    for _, attr in graph.nodes(data=True):
        features.append(attr['node_feature'])
    X = np.array(features)

    return A, X

def normalize_adjacency(A):
    """
    Normalizes the adjacency matrix A using D^-1/2 * A * D^-1/2.

    :param A: Adjacency matrix.
    :return: Normalized adjacency matrix.
    """
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    return D_inv_sqrt @ A @ D_inv_sqrt

def forward_pass(A, X, W):
    """
    Performs a single forward pass for a GCN layer.

    :param A: Normalized adjacency matrix.
    :param X: Node feature matrix.
    :param W: Weight matrix.
    :return: Updated feature matrix after the forward pass.
    """
    return np.tanh(A @ X @ W)  # Using tanh as activation function

# Step 1: Add self-loops to the adjacency matrix
def add_self_loops(A):
    I = np.eye(A.shape[0])  # Identity matrix
    return A + I

# Step 2: Normalize the adjacency matrix
def normalize_adjacency_matrix(A):
    D = np.diag(np.sum(A, axis=1))  # Degree matrix
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))  # D^(-1/2)
    return np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)

# Step 3: Compute A'X
def compute_AX_with_self_loops(A, X):
    A_with_loops = add_self_loops(A)  # Add self-loops
    A_normalized = normalize_adjacency_matrix(A_with_loops)  # Normalize A'
    return np.dot(A_normalized, X)  # A'X

def normalize_features(A, X):
    """
    Normalize the features AX using the degree matrix D.

    :param A: Adjacency matrix (normalized or unnormalized).
    :param X: Feature matrix (result of AX).
    :return: Normalized features (D^-1 * AX).
    """
    # Compute the degree matrix D
    degrees = np.sum(A, axis=1)
    D_inv = np.diag(1 / degrees)  # Inverse of the degree matrix

    # Compute the normalized feature matrix DAX
    return np.dot(D_inv, X)

def symmetric_normalization(A):
    """
    Applies symmetric normalization to the adjacency matrix A.

    A' = D^(-1/2) * A * D^(-1/2)

    :param A: The adjacency matrix.
    :return: The normalized adjacency matrix A'.
    """
    # Degree matrix D (diagonal matrix with the sum of each row of A)
    D = np.diag(np.sum(A, axis=1))

    # Compute D^(-1/2), the inverse square root of the degree matrix
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    # Apply symmetric normalization: A' = D^(-1/2) * A * D^(-1/2)
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt

    return A_normalized

def relu(X):
    return np.maximum(0, X)

# Initialize weight matrices for the 2 layers
def initialize_weights(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(input_dim, hidden_dim)  # Weight matrix for first layer
    W2 = np.random.randn(hidden_dim, output_dim)  # Weight matrix for second layer
    return W1, W2

# 2-layer GCN forward pass
def forward_pass_2layer(A, X, W1, W2):
    """
    Performs a forward pass through a 2-layer GCN.

    :param A: Normalized adjacency matrix.
    :param X: Node feature matrix.
    :param W1: Weight matrix for the first layer.
    :param W2: Weight matrix for the second layer.
    :return: Output feature matrix after the second layer.
    """
    # First GCN layer
    H1 = np.dot(A, X)  # Apply adjacency matrix to feature matrix
    H1 = np.dot(H1, W1)  # Apply first weight matrix
    H1 = relu(H1)  # Apply ReLU activation

    # Second GCN layer
    H2 = np.dot(A, H1)  # Apply adjacency matrix again
    H2 = np.dot(H2, W2)  # Apply second weight matrix

    return H2

def max_pooling(X):
    """
    Applies one-way Max Pooling to the node feature matrix X to generate a fixed-sized vector.

    :param X: Node feature matrix (nodes x features).
    :return: A fixed-sized vector obtained by pooling the maximum value of each feature.
    """
    # Apply max pooling along the node dimension (axis=0 means across rows for each feature)
    return np.max(X, axis=0)

def predict_defect_status(model, input_data):
    """
    Given an input data, predict if the system is defective or not.

    :param model: Trained CNN model
    :param input_data: Pooled vector or new input data (shape should match model input)
    :return: Predicted class (defective or not defective)
    """
    # Ensure the input is reshaped for prediction (samples, features, 1)
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    # Predict using the trained model
    prediction = model.predict(input_data)
    # Output is a probability, 0.5 is the threshold to classify as defective (1) or not defective (0)
    return (prediction > 0.5).astype(int)

@method_decorator(csrf_exempt, name='dispatch')
class CognitiveComplexityView(APIView):
    def post(self, request):
        try:
            code = request.data.get("code")
            if not code:
                return JsonResponse({"error": "No code provided"}, status=400)

            tree_root = create_tree_from_code(code)
            cognitive_complexity = sum(child.cognitive_weight for child in tree_root.children)
            tree_dict = tree_root.to_dict()

            return JsonResponse({
                "tree": tree_dict,
                "cognitive_complexity": cognitive_complexity,
            }, status=200)
        except Exception as e:
            return JsonResponse({
                "error":"Syntatic Error in the code"
            }, status=504)

@method_decorator(csrf_exempt, name='dispatch')
class CognitiveComplexityViewNodeFeatures(APIView):
    def post(self, request):
        try:
            tree_dict = request.data.get("tree_dict")
            if not tree_dict:
                return JsonResponse({"error": "No tree_dict provided"}, status=400)

            # Debug log for tree_dict
            # print("Received tree_dict:", tree_dict)

            parse_tree(tree_dict, highest_values)  # Ensure these are defined
            highest_values['Module'] = 1
            highest_values['FunctionDef'] = 0

            tree_root_later = build_tree_later(tree_dict)
            tree_root_later.node_feature = [1, 0, 0, 0, 0, 0, 0]
            calculate_node_features(tree_root_later, None)

            # Create new tree data structure with node features
            updated_tree_data = create_tree_data_with_features(tree_root_later)

            return JsonResponse({
                "node_feature_dict": updated_tree_data,
                "high":highest_values
            }, status=200)
        except Exception as e:
            error_details = traceback.format_exc()
            return JsonResponse({
                "error": str(e),
                "traceback": error_details  # Include traceback in response
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class CognitiveComplexityViewFixedSizeVector(APIView):
    def post(self, request):
        try:
            # Set seed values
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(42)
            tf.random.set_seed(42)

            # Load data
            data = request.data.get("node_feature_dict")
            G, _ = build_graph_from_dict(data)
            A, X = get_adjacency_and_features(G)

            # Normalize adjacency matrix
            A_normalized = normalize_adjacency(A)

            # Initialize weight matrix W
            input_dim, output_dim = X.shape[1], 16
            W = np.random.randn(input_dim, output_dim)

            # Perform forward pass
            H = forward_pass(A_normalized, X, W)

            # Compute AX with self-loops and normalize
            np.fill_diagonal(A, 1)  # Add self-loops directly to A
            A_normalized_symmetric = symmetric_normalization(A)
            AX_with_loops = np.dot(A_normalized_symmetric, X)
            DAX = normalize_features(A, AX_with_loops)

            # Initialize weights for 2-layer GCN
            input_dim, hidden_dim, output_dim = 7, 32, 7
            W1, W2 = initialize_weights(input_dim, hidden_dim, output_dim)

            # Forward pass through 2-layer GCN
            output_features = forward_pass_2layer(A_normalized_symmetric, X, W1, W2)

            # Use first element and round
            pooled_vector = np.round(np.array([output_features[0]]), 2)
            print(pooled_vector)
            # Load model once (could be optimized further by caching)
            model = load_model("complexipy.keras", compile=False)
            defect_status = predict_defect_status(model, pooled_vector)

            return JsonResponse({
                "fixed_size_vector": pooled_vector.tolist(),
                "prediction": int(defect_status[0][0])
            }, status=200)

        except KeyError as e:
            return JsonResponse({"error": f"Missing key: {e}"}, status=400)
        except ValueError as e:
            return JsonResponse({"error": f"Value error: {e}"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e), "traceback": traceback.format_exc()}, status=500)

