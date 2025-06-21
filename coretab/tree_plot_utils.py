import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from copy import deepcopy

from coretab.coreset_algorithms import CoreTabDT


class TreeAttributeAccessor:
    def __init__(self, tree, edges):
        """
        Initialize the TreeAttributeAccessor with a sklearn.tree._tree.Tree object.
        
        Parameters:
        tree (sklearn.tree._tree.Tree): The decision tree object (from sklearn.tree).
        """
        self.tree = tree
        self.left_childs = np.array([val if (node_id, val, 'left') in edges else -1 for node_id, val in enumerate(tree.children_left)])
        self.right_childs = np.array([val if (node_id, val, 'right') in edges else -1 for node_id, val in enumerate(tree.children_right)])

        
    def __getattr__(self, attr):
        """
        Override the __getattr__ method to fetch attributes from the underlying Tree object.
        
        Parameters:
        attr (str): Name of the attribute to fetch.
        
        Returns:
        Any: Value of the specified attribute from the underlying Tree object.
        """
        if hasattr(self.tree, attr):
            if attr == 'children_left':
                return self.left_childs
            elif attr == 'children_right':
                return self.right_childs
            return getattr(self.tree, attr)
        else:
            raise AttributeError(f"'{type(self.tree).__name__}' object has no attribute '{attr}'")
        

def extract_subset_tree(tree, hom_node_indexes, node_index=0):

    is_hom_leaf = True if node_index in hom_node_indexes else False
    
    if node_index <= -1:
        return [], [], is_hom_leaf

    left_child = tree.children_left[node_index]
    right_child = tree.children_right[node_index]

    # Recursive traversal to extract subset
    nodes = [node_index]
    edges = []

    is_hom_leaf_left = False
    if left_child >= -1:
        left_nodes, left_edges, is_hom_leaf_left = extract_subset_tree(tree, hom_node_indexes, left_child)
        if is_hom_leaf_left:
            nodes.extend(left_nodes)
            edges.extend([(node_index, left_child, 'left')])
            edges.extend([(node_index, right_child, 'right')])
            edges.extend(left_edges)
            

    is_hom_leaf_right = False
    if right_child >= -1:
        right_nodes, right_edges, is_hom_leaf_right = extract_subset_tree(tree, hom_node_indexes, right_child)
        if is_hom_leaf_right:
            nodes.extend(right_nodes)
            edges.extend([(node_index, right_child, 'right')])
            edges.extend([(node_index, left_child, 'left')])
            edges.extend(right_edges)

    return nodes, edges, (is_hom_leaf or is_hom_leaf_right or is_hom_leaf_left)


def search_for_leaves_with_feature(coretab_dt: CoreTabDT, feature_name):
    relevant_nodes = []
    i = 0

    for node_id in coretab_dt.hom_groups.keys():
        if i % 100 == 0:
            print(f"Processing node {i} of {len(coretab_dt.hom_groups.keys())}")
        i += 1
        node_description = get_rules_for_leaf(coretab_dt.model.tree_, node_id, coretab_dt.columns)[0]
        if any([s.startswith(feature_name + ' ') for s in node_description]):
            relevant_nodes.append(node_id)
    return relevant_nodes


def get_rules_for_leaf(tree, node_to_search, feature_names, node_id=0):
    """
    Recursively traverse the decision tree to extract rules leading to a specific leaf node.
    
    Parameters:
    tree (sklearn.tree._tree.Tree): The trained decision tree object.
    node_id (int): The node ID to start traversal (typically the root node ID).
    feature_names (list): List of feature names.
    threshold_sign (list): List of threshold signs ('<=', '>') for each node.
    
    Returns:
    list: List of rule strings leading to the specified leaf node.
    """
    # Initialize an empty list to store the rules
    rules = []

    # Check if the current node is a leaf node
    if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
        # Return the list of rules when reaching a leaf node
        return rules, node_id == node_to_search
    
    # Get the feature index and threshold value for the current node
    feature_index = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    
    # Get the feature name and threshold sign ('<=', '>')
    feature_name = feature_names[feature_index]
    
    # Recursively traverse the left and right child nodes
    left_rules, is_node_there_left = get_rules_for_leaf(tree, node_to_search, feature_names, tree.children_left[node_id])
    right_rules, is_node_there_right = get_rules_for_leaf(tree, node_to_search, feature_names, tree.children_right[node_id])
    
    # Append the rules from left and right subtrees
    if is_node_there_left:
        rules.extend([f"{feature_name} <= {threshold:.2f}"] + left_rules)
    if is_node_there_right:
        rules.extend([f"{feature_name} >= {threshold:.2f}"] + right_rules)
    
    return rules, is_node_there_left or is_node_there_right


def plot_path_to_leaves(coretab_dt: CoreTabDT, leaves=None, max_depth=12, show_fig=True):
    if leaves is None:
        leaves = list(coretab_dt.hom_groups.keys())

    nodes, edges, _ = extract_subset_tree(coretab_dt.model.tree_, leaves)
    new_t = TreeAttributeAccessor(coretab_dt.model.tree_, edges)
    coretab_dt_copy = deepcopy(coretab_dt)
    coretab_dt_copy.model.tree_ = new_t
    plt.figure(figsize=(20,10))  # Adjust the figure size as needed
    plot_tree(coretab_dt_copy.model, filled=True, feature_names=coretab_dt.columns,
              impurity=False, max_depth=max_depth, node_ids=True)
    if show_fig:
        plt.show()


def plot_coreset(coretab_dt, green_nodes=[], grey_nodes=[], max_depth=25):
    
    if not green_nodes:
        green_nodes = coretab_dt.hom_groups.keys()

    highlight_nodes = green_nodes + grey_nodes    
    plot_path_to_leaves(coretab_dt, highlight_nodes, show_fig=False, max_depth=max_depth)
    
    ax = plt.gca()
    # Iterate over each node in the tree plot
    for node in ax.get_children():
        if hasattr(node, 'get_text') and node.get_text().startswith('node '):
            node_id = int(node.get_text().split()[1][1:])  # Extract the node ID from the text
            node.set_text(node.get_text().replace('value = ', 'label count='))
        # Highlight the node if it's in the list of nodes to color
            if node.get_text().count('\n') == 2:
                node.set_text('\n'.join([node.get_text().split('\n')[0], 'leaf node - no condition'] +  node.get_text().split('\n')[1:]))
            sizes = eval(node.get_text()[node.get_text().find('['): ])
            if sizes[0] > sizes[1]:
                if sizes[1] == 0:
                    node.set_backgroundcolor('tomato')
                elif sizes[0] / sizes[1] >= 10:
                    node.set_backgroundcolor('salmon')
                else:
                    node.set_backgroundcolor('lightsalmon')
            
            if node_id in green_nodes:
                node.set_backgroundcolor('lime')  # Change node color to highlight
            if node_id in grey_nodes:
                node.set_backgroundcolor('gray')  # Change node color to highlight

