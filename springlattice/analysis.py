import numpy as np
from scipy.spatial import Delaunay, Voronoi
from icecream import ic

# class LineConnectivityGraph:
#     def __init__(self, lines):
#         self.lines = lines
#         self.node_count = 0
#         self.chain = []
#         self.node_list = []
#         self.node_positions = np.empty((0, 2), dtype=float)
#         self._build_graph()

#     def _build_graph(self):
#         while self.lines.shape[0] > 0:
#             line_end1 = self.lines[0, :2]
#             line_end2 = self.lines[0, 2:]


#             connected = self._check_connection(line_end1, line_end2)

#             if not connected:
#                 self._add_new_nodes(line_end1, line_end2)

#             self.lines = np.delete(self.lines, 0, axis=0)

#         # Convert lists to NumPy arrays
#         self.chain = np.array(self.chain, dtype=int)
#         self.node_list = np.array(self.node_list, dtype=int)

#     def _check_connection(self, node1, node2):
#         connected = False

#         for idx, current_node in enumerate(self.node_positions):
#             if np.linalg.norm(node1 - current_node) <= 1e-9:
#                 for n, other_end in enumerate(self.node_positions):
#                     if np.linalg.norm(node2 - other_end) <= 1e-9:
#                         self.chain.append([self.node_list[idx], self.node_list[n]])
#                         connected = True
#                         break

#                 if not connected:
#                     self._add_node_and_update_chain(node2, idx)
#                     connected = True
#                 break

#             elif np.linalg.norm(node2 - current_node) <= 1e-9:
#                 for n, other_end in enumerate(self.node_positions):
#                     if np.linalg.norm(node1 - other_end) <= 1e-9:
#                         self.chain.append([self.node_list[idx], self.node_list[n]])
#                         connected = True
#                         break

#                 if not connected:
#                     self._add_node_and_update_chain(node1, idx)
#                     connected = True
#                 break

#         return connected

#     def _add_node_and_update_chain(self, new_node, idx):
#         self.node_list.append(self.node_count)
#         self.node_positions = np.vstack([self.node_positions, new_node])
#         self.chain.append([self.node_list[idx], self.node_count])
#         self.node_count += 1

#     def _add_new_nodes(self, node1, node2):
#         self.node_list.append(self.node_count)
#         self.node_positions = np.vstack([self.node_positions, node1])
#         self.node_positions = np.vstack([self.node_positions, node2])
#         self.chain.append([self.node_count, self.node_count+1])
#         self.node_list.append(self.node_count+1)
#         self.node_count += 2



class LineConnectivityGraph:
    def __init__(self, lines, precision=9):
        self.lines = lines
        self.node_count = 0
        self.chain = []
        self.node_positions = np.empty((0, 2), dtype=float)
        self.node_dict = {}
        self.precision = precision
        self._build_graph()

    def _build_graph(self):
        for line in self.lines:
            line_end1 = tuple(round(x,self.precision) for x in line[:2])
            line_end2 = tuple(round(x,self.precision) for x in line[2:])

            connected = self._check_connection(line_end1, line_end2)

            if not connected:
                self._add_new_nodes(line_end1, line_end2)

        # Convert lists to NumPy arrays
        self.chain = np.array(self.chain, dtype=int)


    def _check_connection(self, node1, node2):
        connected = False

        idx_node1 = self.node_dict.get(node1)
        idx_node2 = self.node_dict.get(node2)

        if idx_node1 is not None:
            if idx_node2 is not None:
                self.chain.append([idx_node1, idx_node2])
                connected = True
            else:
                self._add_node_and_update_chain(node2, idx_node1)
                connected = True
        elif idx_node2 is not None:
            self._add_node_and_update_chain(node1, idx_node2)
            connected = True

        return connected

    def _add_node_and_update_chain(self, new_node, idx):
        self.node_dict[new_node] = self.node_count
        self.node_positions = np.vstack([self.node_positions, np.array(new_node)])
        self.chain.append([idx, self.node_count])
        self.node_count += 1

    def _add_new_nodes(self, node1, node2):
        self.node_dict[node1] = self.node_count
        self.node_dict[node2] = self.node_count + 1
        self.node_positions = np.vstack([self.node_positions, np.array(node1), np.array(node2)])
        self.chain.append([self.node_count, self.node_count + 1])
        self.node_count += 2




# Usage:
# lines_data = np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...]) # Replace with your line data
# where each row represents the coordinates of the two ends of a line segment
# For example, [x1, y1, x2, y2] represents a line from (x1, y1) to (x2, y2)

# # Create the LineConnectivityGraph object
# line_graph = LineConnectivityGraph(lines_data)

# # Access the resulting chain and node lists if needed
# chain = line_graph.chain
# node_list = line_graph.node_list



# Define TotalNodes, a, N, and M before using this code
def VoronoiGenerator(NodePos: np.ndarray, a: float, rect_box: tuple):
    nx, ny = rect_box
    dx = a
    dy = 0.5 * np.sqrt(3) * a

    # Artificial points to define the rectangular boundary
    artificial_points = []

    # Bottom of the lattice
    for j in range(nx + 1):
        x = j * dx - 0.5 * dx
        y = -dy
        artificial_points.append([x, y])

    # Left of the lattice
    for i in range(0,ny+ 1,2):
        x = -1 * dx
        y = i * dy 
        artificial_points.append([x, y])

    # Right of the lattice
    for i in range(0,ny, 2):
        x = nx * dx - 0.5 * dx
        y = (i+1) * dy 
        artificial_points.append([x,y])    

    # Top of the lattice
    for j in range(nx + 1):
        if ny % 2 == 0:
            x = j * dx
        else:
            x = j * dx - 0.5 * dx
        y = ny * dy
        artificial_points.append([x, y])

    NodePos = np.vstack((NodePos, artificial_points))

    vor = Voronoi(NodePos)
    # Voro_Edges = vor.ridge_vertices
    # Voro_Region = vor.regions

    return vor

"""Function to find the Voronoi edges connecting nodes i and j"""
def find_voronoi_edges(vor, i: int, j: int):
    for idx, (p1, p2) in enumerate(vor.ridge_points):
        if {i, j} == {p1, p2}:
            ridge_vertices_idx = vor.ridge_vertices[idx]
            ridge_vertices = vor.vertices[ridge_vertices_idx]
            return ridge_vertices

    print(f"No edges found between nodes {i} and {j}")
    return None

def preprocess_voronoi_edges(vor):
    edge_map = {}
    for idx, (p1, p2) in enumerate(vor.ridge_points):
        if p1 != -1 and p2 != -1:
            if p1 not in edge_map:
                edge_map[p1] = set()
            if p2 not in edge_map:
                edge_map[p2] = set()
            edge_map[p1].add(idx)
            edge_map[p2].add(idx)
    return edge_map

def find_voronoi_edges_optimized(vor, i: int, j: int, edge_map):
    common_edges = edge_map.get(i, set()).intersection(edge_map.get(j, set()))
    
    if common_edges:
        idx = common_edges.pop()  # Retrieve the single index from the set
        ridge_vertices_idx = vor.ridge_vertices[idx]
        return vor.vertices[ridge_vertices_idx]

    print(f"No edges found between nodes {i} and {j}")
    return None


