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

#Documentation needed here
def traverse_tree(root, Input_RowCol, input_size):

    #If leaf node has been reached return the graphs associated with it
    if(root.tier == input_size):
        return root.graphs

    for children in root.child:
        if(children.row_col == Input_RowCol[root.tier]):
            return traverse_tree(children, Input_RowCol, input_size)
    
    return []

#Documentation needed here
def draw_tree(root, tree_visualization):
    node_number = tree_visualization.number_of_nodes()
    for children in root.child:
        stringconvert = " ".join(map(str,children.row_col))
        tree_visualization.add_edge(node_number,tree_visualization.number_of_nodes()+1,rowcol = stringconvert)
        draw_tree(children,tree_visualization)

    return

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
tree_Graph.add_node(1)

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



#Traversal of Decision tree to Analyze an Input Graph Subgraph Isomorphism


input_rowcol = [[2], [0, 3, 1], [0, 1, 2, 0, 1]]
#input_rowcol = [[2], [0, 3, 1]]
#input_rowcol = [[2]]

print(f"Input Graph Row Column Representation\n{input_rowcol}")

test = traverse_tree(tree_root,input_rowcol,len(input_rowcol))
if not(test):
    print("No Subgraph Match")
else:
    print(f"Input Graph Matches Graphs {test}")








draw_tree(tree_root,tree_Graph)
#pos = nx.nx_pydot.graphviz_layout(tree_Graph, prog="dot")
#nx.draw_networkx_edge_labels(tree_Graph, pos = nx.spring_layout(tree_Graph))
nx.draw_networkx(tree_Graph)

#G = nx.complete_graph(4)

# = nx.nx_pydot.graphviz_layout(G)

#pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
#nx.draw(G)
plt.show()

#T = nx.balanced_tree(2, 5)

#pos = graphviz_layout(T, prog="twopi")
#nx.draw(T, pos)
#plt.show()


#stringconvert = " ".join(map(str,Adj_Matrix[index]))
#stringconvert2 = " ".join(map(str,Adj_Matrix[index-1]))
#if(index == 0):
#tree_Graph.add_edge(stringconvert2, "root")
#else:
#tree_Graph.add_edge(stringconvert2, stringconvert)