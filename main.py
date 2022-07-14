import networkx as nx
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
#import pydot
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
    for x in Permutation_Matrices:
        Permutation_Transpose = np.transpose(x)
        Subgraphs.append(np.linalg.multi_dot([x,Adjacency_Matrix,Permutation_Transpose]))
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
def create_tree(Adj_Matrix, root, index, graph_number):
    #Case: row-column is represented in one of the childs of root
    for children in root.child:
        if(Adj_Matrix[index] == children.row_col):
            root.graphs.append(graph_number)
            create_tree(Adj_Matrix,children,index+1,graph_number)
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
        create_tree(Adj_Matrix,ChildNode,index+1,graph_number)
    root.graphs.append(graph_number)
    return 0

#Finish before 7/14
def traverse_tree(root, Input_RowCol):

    for children in root.child:
        if(children.row_col[root.tier] == Input_RowCol[root.tier]):
            traverse_tree(children, Input_RowCol)
    
    #Two if conditions
    #1. Has reached a leaf node
    #2. 

    #
    return 

#Finish before 7/14
def draw_tree(root):
    return 0

#main 
#Graph Creation
#sample adj matrix from paper Subgraph Isomorphism in polynomial time
G_Model = nx.DiGraph()

#Random Graph of 10 nodes with 20 edges
#Test run of current program with decision tree creation 7/7/2022
#Total time taken was 1 minute 27 seconds
#G_Model = nx.gnm_random_graph(10, 20, seed=109389)

edgelist = [(0, 1),(0, 2),(1, 2)]
G_Model.add_edges_from(edgelist)
Model_nodes = nx.number_of_nodes(G_Model)

#Graph Drawing utilizing pyplot
#nx.draw_networkx(G_Model)
#plt.show()

#Adjacency Matrix of Model Graph Generation
#print(f"Graph Adjacency: {G_Model.adj}")
AdjMatrix_Model = nx.adjacency_matrix(G_Model)
#Cause for SparseEfficiencyWarning
AdjMatrix_Model[0,0] = 3
AdjMatrix_Model[1,1] = 2
AdjMatrix_Model[2,2] = 2

#function calls
Permutation_Matrix_List = per_mat(Model_nodes)

Permutated_Subgraphs = adj_mat(AdjMatrix_Model.todense(),Permutation_Matrix_List)

tree_root = TreeNode()
tree_Graph = nx.DiGraph()
tree_Graph.add_node("root")

#Input_RowCol = row_col(Graph,len(Graph))


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
    create_tree(x,tree_root,0,i)

#tree_root.print()

#pos = nx.nx_pydot.graphviz_layout(tree_Graph, prog="dot")
#nx.draw_networkx(tree_Graph, pos)
#plt.show()


#stringconvert = " ".join(map(str,Adj_Matrix[index]))
#stringconvert2 = " ".join(map(str,Adj_Matrix[index-1]))
#if(index == 0):
#tree_Graph.add_edge(stringconvert2, "root")
#else:
#tree_Graph.add_edge(stringconvert2, stringconvert)