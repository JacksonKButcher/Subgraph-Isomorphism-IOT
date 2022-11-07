from os import kill
import networkx as nx
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class TreeNode:
    def __init__(self):
        self.row_col = []
        self.child = []
        self.tier = 0
        self.graphs = []

    def insert_child(self,data):
        self.child.append(data)
    
    #Print row-column of itself and all child nodes
    def print(self):
        print(f"Level: {self.tier} \nGraphs: {self.graphs} \nRowCol: {self.row_col}\n")
        for child in self.child:
            child.print()

#Implement properly after finishing the two functions below
class GraphNode:
    def __init__(self,name):
        self.label = name
        self.value = 0
    def __str__(self):
        return self.label
    def __hash__(self):
        return hash(str(self))
    def __eq__(self,other):
        return self.label == other.label

#Generates every version of an n x n permutation matrix
# Input: n (integer)
# Output: List of every n x n permutation matrix
def per_mat(n):
    Permutation_Matrix = np.zeros((n,n),dtype=int)
    Permutations_List = []
    for x in range(n):
        Permutation_Matrix[x][x] = 1
    for m in it.permutations(Permutation_Matrix):
        Permutations_List.append(m)
    return Permutations_List

#Generate all-permutated adjacency matrices of Graph G
# Input: Adjacency matrix of graph G
# Output: List of permutated adjacency matrices
def adj_mat(Adjacency_Matrix,Permutation_Matrices):
    Subgraphs = []
    for p in Permutation_Matrices:
        Permutation_Transpose = np.transpose(p)
        Subgraphs.append(np.linalg.multi_dot([p,Adjacency_Matrix,Permutation_Transpose]))
    return Subgraphs

#Generate a row column representation of a graph
# Input: adjacency matrix, number of nodes
# Output: List of row-column representation
def row_col(M, nodes):
    row_col_List = []
    for x in range(nodes):
        r = []
        #traverse down column x
        for y in range(0,x+1,+1):
            r.append(M[y, x])
        #traverse down row x
        for y in range(x,0,-1):
            r.append(M[x, y-1])
        row_col_List.append(r)
    return row_col_List

#Generate a decision tree
# Input: 
#   adj_Matrix = row-column representation of permutated adjacency subgraph
#   root = TreeNode Object, self-isolated root of method
#   index = identifier for how deep the call is into the tree
#   graph_Number = identifier for what graph the method is processing
#   tree_Graph = networkx graph that will visualize tree
# Output: 
#   No official output. 
#   Root node orginially passed should now have branches.
def merge_tree(Adj_Matrix, root, index, graph_number):
    #Case: row-column is represented in one of the childs of root
    for children in root.child:
        if(Adj_Matrix[index] == children.row_col):
            root.graphs.append(graph_number)
            merge_tree(Adj_Matrix,children,index+1,graph_number)
            return 0
    #Case: recursion has reached a branch node at the bottom of the tree
    if(index == len(Adj_Matrix)):
        root.graphs.append(graph_number)
        return 0
    #Case: a row-column representation isn't currently a child
    else:
        ChildNode = TreeNode()
        ChildNode.row_col = Adj_Matrix[index]
        ChildNode.tier = index + 1
        root.insert_child(ChildNode)
        merge_tree(Adj_Matrix,ChildNode,index+1,graph_number)
    root.graphs.append(graph_number)
    return 0

#Generate a decision tree to identify subgraph isomorphism
#Input: Adjacency Matrix of Graph
#Output: Root node of the decision tree
def create_tree(Adj_Matrix, model_vertices):
    #
    Permutation_Matrix_List = per_mat(model_vertices)
    #
    Permutated_Subgraphs = adj_mat(Adj_Matrix,Permutation_Matrix_List)
    
    #
    row_columns = []
    for i in range(len(Permutated_Subgraphs)):
        x = row_col(Permutated_Subgraphs[i],model_vertices)
        row_columns.append(x)


    root = TreeNode()
    #loop through each size of subgraph until 1 is hit
    for k in range(model_vertices):
        merge_tree2(Permutated_Subgraphs, root, k)
    return root

#Fill out a layer of the decision tree of the subgraphs up to k vertices
def merge_tree2(row_columns, root, k):

    for graph in row_columns:
        node = TreeNode()
        for n in range(k):
            #Case 1:
            #
            if(node.check_children(graph[n])):
                break
            #Case 2:
            #
            else:
                ChildNode = TreeNode()
                ChildNode.row_col = graph[n]
                root.insert_child(ChildNode)
    return 0

#Documentation needed here
def traverse_tree(root, Input_RowCol, input_size):

    #If leaf node has been reached return the graphs associated with it
    if(root.tier == input_size):
        return root.graphs

    for children in root.child:
        if(children.row_col == Input_RowCol[root.tier]):
            return traverse_tree(children, Input_RowCol, input_size)
    
    return []
#Input:
#I = Input Graph/Matrix
#root = root of decision tree
def input_permutation_traversal(I,root):

    I_AdjMatrix = nx.adjacency_matrix(I)
    input_nodes = nx.number_of_nodes(I)
    input_rowcol = row_col(I_AdjMatrix,input_nodes)
    
    Permutation_Matrix_List = per_mat(input_nodes)
    Permutated_Subgraphs = adj_mat(I_AdjMatrix.todense(),Permutation_Matrix_List)
    print(f"Input Graph Row Column Representation\n{input_rowcol}")
    testing = []

    for i in range(input_nodes*2):
        #Printing Subgrahps with formatting

        x = row_col(Permutated_Subgraphs[i],input_nodes)
        print(x)
        tree_logic_test = traverse_tree(root,x,len(x))

        testing.append(tree_logic_test)

    return testing

#Documentation needed here
def draw_tree(root, tree_visualization):
    node_number = tree_visualization.number_of_nodes()
    for children in root.child:
        stringconvert = " ".join(map(str,children.row_col))
        tree_visualization.add_edge(int(node_number),int(tree_visualization.number_of_nodes()+1),rc = stringconvert)
        draw_tree(children,tree_visualization)

    return

#main 
#Graph Instantiation
G_Model = nx.DiGraph()

#Random Graph of 10 nodes with 20 edges
#Test run of current program with decision tree creation 7/7/2022
#Total time taken was 1 minute 27 seconds
#G_Model = nx.gnm_random_graph(10, 20, seed=109389)

edgelist_3node = [(0, 1),(0, 2),(1, 2)]
edgelist_test = [(0,1)]
edgelist_4node = [(0,1),(0,2),(1,2),(2,3),(0,3)]
edgelist_5node = [(0,1),(0,2),(1,3),(2,3),(2,4),(3,4)]
G_Model.add_edges_from(edgelist_3node)
Model_nodes = nx.number_of_nodes(G_Model)

#Graph Drawing utilizing pyplot
nx.draw_networkx(G_Model,arrowsize=30,font_color = "w",node_color = "k")
plt.show()

#Adjacency Matrix of Model Graph Generation
AdjMatrix_Model = nx.adjacency_matrix(G_Model)

#function calls

Permutation_Matrix_List = per_mat(Model_nodes)

Permutated_Subgraphs = adj_mat(AdjMatrix_Model.todense(),Permutation_Matrix_List)


#Decision Tree Instantation
tree_root = TreeNode()
tree_Graph = nx.DiGraph()
tree_Graph.add_node(1)

#Printing Subgraphs and Row-Column Representation
for i in range(Model_nodes*2):
    #Printing Subgrahps with formatting
    print(f"Subgraph {i}:\n{Permutated_Subgraphs[i]}")
    x = row_col(Permutated_Subgraphs[i],Model_nodes)
    print(f"Row Column Representation: {x}\n")

    #P = nx.DiGraph(Permutated_Subgraphs[i])
    #nx.draw_networkx(P)
    #plt.show()

    #Creating Decision Tree with create_tree
    merge_tree(x,tree_root,0,i)

#tree_root.print()

#decision_tree = create_tree(AdjMatrix.todense(),Model_nodes)


#Traversal of Decision tree to Analyze an Input Graph Subgraph Isomorphism

#Input graph creation and handling
G_Input = nx.DiGraph()
edgelist_input = [(0, 1),(0, 2),(2,1)]
edgelist_input2 = [(0,1),(0,2)]
G_Input.add_edges_from(edgelist_input2)


#Draw Input Graph
nx.draw_networkx(G_Input,arrowsize=30,font_color = "w",node_color = "k")
plt.show()


tree_logic_test = input_permutation_traversal(G_Input,tree_root)
if not(tree_logic_test):
    print("No Subgraph Match")
else:
    print("\nThe traversal of the decision tree has detected a subgraph isomorphism")
    print(f"Input Graph Matches Graphs {tree_logic_test}\n")



#Decision tree visualization
#method call to generate networkx graph of row col, similar to Subgraph Isomorphism Paper
draw_tree(tree_root,tree_Graph)
#generate optimal positions of nodes in format of multilevel tree
pos = nx.nx_pydot.graphviz_layout(tree_Graph, prog="dot")
#convert dictionary keys to integers since they're returned as strings of ints(aren't compatible with graph drawing)
pos = {int(k):v for k,v in pos.items()}
#draw the graph with slight formatting
nx.draw_networkx(tree_Graph,pos,arrowsize=5,font_color = "w",node_color = "k")
edge_labels = nx.get_edge_attributes(tree_Graph,'rc')
nx.draw_networkx_edge_labels(tree_Graph, pos,edge_labels = edge_labels,font_size=11,horizontalalignment='center',rotate=False,label_pos = 0.285)
plt.show()


#Misc code leftovers
#stringconvert = " ".join(map(str,Adj_Matrix[index]))
#stringconvert2 = " ".join(map(str,Adj_Matrix[index-1]))
#if(index == 0):
#tree_Graph.add_edge(stringconvert2, "root")
#else:
#tree_Graph.add_edge(stringconvert2, stringconvert)
