import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle
import os
from shapely.geometry import Point, Polygon
from dataclasses import dataclass


@dataclass
class mesh:
    nx: int
    ny: int
    a: float
    lattice: str
    pos: np.ndarray
    neighbors: list
    angles: list
    normal_stiffness: list
    tangential_stiffness: list
    bottom: list
    top: list
    left: list
    right: list
    u: np.ndarray

    def __post_init__(self):
        self.domain = [
            np.min(self.pos[:, 0]), np.max(self.pos[:, 0]),
            np.min(self.pos[:, 1]), np.max(self.pos[:, 1])
        ]
        self.folder = ''
        self.solver = {'dt':0.05, 'name':'default', 'skipsteps':0,
                       'endtime':1, 'zeta':0}
        self.crack_param = {}
        self.bond_prop = {}
        self.bcs = bcs()
        self.lbcs = lbcs()

# displacement boundary condition class
class bcs:
    def __init__(self) -> None:
        self.ids = []
        self.comp = []
        self.parser = []
        self.fun = []
# load boundary condition class
class lbcs:
    def __init__(self) -> None: 
        self.ids = []
        self.fx = []
        self.fy = []
        self.fun = []




""" Function for generating triangular lattice mesh"""
def mesh_generator(M,N,a = 1):
    """
    Generate a triangular mesh lattice with specified dimensions and properties.

    Parameters:
    - M (int): Number of rows in the lattice.
    - N (int): Number of columns in the lattice.
    - a (float, optional): Lattice constant or node spacing (default is 1).

    Returns:
    - mesh: Mesh object containing the generated lattice information.

    This function generates a 2D triangular lattice based on the provided dimensions and lattice constants.
    It develops nodes for the lattice, establishing connections (bonds) between them based on the lattice structure.
    Node properties such as positions, neighbors, angles, and stiffness are defined according to the lattice configuration.
    The function creates a mesh object representing the generated lattice and returns it.
    """
    lattice = 'triangle'

    # defining node properties
    neighbors = [0]*M*N
    angles = [0]*M*N
    normal_stiffness = [0]*M*N
    tangential_stiffness = [0]*M*N
    pos = np.zeros(shape = (M*N,2))
    disp = np.zeros(shape = (M*N,2))

    # default bond stiffness
    Kn = 1
    # Node index 
    ID = 0
    # Variables to store the nodes of left and right boundary
    left =[]
    right = []

    # defining inline lambda function
    Lex = lambda i,j: i*N + j

    # Developing nodes for 2D triangular lattice
    for i in range(0,M):
        for j in range(0,N):
            # bottom row
            if i==0:
                # left most node
                if j==0:
                    # 3 bonds connectivity
                    neighbors[ID] = [Lex(i,j+1), Lex(i+1,j+1), Lex(i+1,j)]
                    angles[ID] = [0, 60, 120]
                    left.append(ID)
                # rightmost node    
                elif j == N-1:
                    # 2 bonds connectivity
                    neighbors[ID] = [Lex(i+1,j), Lex(i,j-1)]
                    angles[ID] = [120, 180]
                    right.append(ID)
                # in-between nodes    
                else:
                    # 4 bonds connectivity
                    neighbors[ID] = [Lex(i,j+1), Lex(i+1,j+1), Lex(i+1,j), Lex(i,j-1)]
                    angles[ID] = [0,60, 120, 180]

            # top row
            elif i == M - 1:
                # leftmost node
                if j == 0:
                    left.append(ID)
                    if i%2 != 0:
                        # 2 bonds connectivity
                        neighbors[ID] = [Lex(i,j+1), Lex(i-1,j)]
                        angles[ID] = [0, -60]
                    else:
                        # 3 bonds connectivity
                        neighbors[ID] = [Lex(i,j+1), Lex(i-1,j), Lex(i-1,j+1)]
                        angles[ID] = [0, -120, -60]
                elif j == N - 1:
                    right.append(ID)
                    if i%2 != 0:
                        # 3 bonds
                        neighbors[ID] = [Lex(i,j-1), Lex(i-1,j-1), Lex(i-1,j)]
                        angles[ID] = [180, -120, -60]
                    else:
                        # 2 bonds
                        neighbors[ID] = [Lex(i,j-1), Lex(i-1,j)] 
                        angles[ID] = [180, -120] 
                else:
                    if i%2 != 0:
                        # 4 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i,j-1), Lex(i-1,j-1), Lex(i-1,j)]
                    else:
                        # 4 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i,j-1), Lex(i-1,j), Lex(i-1,j+1)] 

                    angles[ID] = [0, 180, -120, -60]
            else:
                if j == 0:
                    left.append(ID)
                    if i%2 != 0:
                        # 3 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i+1,j), Lex(i-1,j)]
                        angles[ID] = [0, 60, -60]
                    else:
                        # 5 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i+1,j+1), Lex(i+1,j), Lex(i-1,j), Lex(i-1,j+1)]
                        angles[ID] = [0, 60, 120, -120, -60]
                elif j == N - 1:
                    right.append(ID)
                    if i%2 != 0:
                        # 5 bonds
                        neighbors[ID] = [Lex(i+1,j), Lex(i+1,j-1), Lex(i,j-1), Lex(i-1,j-1), Lex(i-1,j)]
                        angles[ID] = [60, 120, 180, -120, -60]
                    else:
                        # 3 bonds
                        neighbors[ID] = [Lex(i+1,j), Lex(i,j-1), Lex(i-1,j)]
                        angles[ID] = [120, 180, -120] 
                else:
                    if i%2 != 0:
                        # 6 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i+1,j), Lex(i+1,j-1), Lex(i,j-1), Lex(i-1,j-1), Lex(i-1,j)]
                    else:
                        # 6 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i+1,j+1), Lex(i+1,j), Lex(i,j-1), Lex(i-1,j), Lex(i-1,j+1)]

                    angles[ID] = [0, 60, 120, 180, -120, -60]
            
            normal_stiffness[ID] = [Kn]*len(neighbors[ID]) 
            tangential_stiffness[ID] = [0.2*Kn]*len(neighbors[ID])
            if i%2 != 0:
                pos[ID] = np.array([(j*a - 0.5*a), (i*a*0.5*math.sqrt(3))])
            else:
                pos[ID] = np.array([(j*a), (i*a*0.5*math.sqrt(3))])

            ID = ID + 1

    
    total_nodes = ID -1
    bottom = list(range(0,N))
    top = list(range(total_nodes-N+1,total_nodes+1))
    return mesh(nx= M,ny=N,a=a,lattice=lattice,pos= pos,neighbors= neighbors,angles= angles,
                normal_stiffness= normal_stiffness, tangential_stiffness=tangential_stiffness,
                 bottom= bottom, top=top, left=left, right=right, u=disp)

"""Function to create folders"""
def _create_directory(folder):
    """
    Create a directory within the current working directory.

    Parameters:
    - folder (str): Name of the directory to be created.

    Returns:
    - str: Absolute path of the created directory.

    This function creates a directory within the current working directory.
    It takes the specified folder name and attempts to create a new directory with that name.
    If a directory with the same name already exists, it prints a message indicating its existence.
    Returns the absolute path of the created directory or the existing directory if no new directory was created.
    """
    path = os.getcwd()
    folder = os.path.join(path, folder)

    try:
        os.mkdir(folder)
    except FileExistsError:
        print(f"Directory with name '{folder}' already exists")

    return folder
   

"""Function to create sub-directory"""
def _sub_directory(dir_name, path):
    """
    Create a subdirectory within the specified path.

    Parameters:
    - dir_name (str): Name of the subdirectory to be created.
    - path (str): Path where the subdirectory will be created.

    Returns:
    - str: Absolute path of the created subdirectory.

    This function creates a subdirectory within the specified path.
    It combines the path and the subdirectory name to create the absolute path.
    If the subdirectory does not exist at that path, it creates a new subdirectory.
    Returns the absolute path of the created subdirectory or the existing subdirectory if no new subdirectory was created.
    """
    dir = os.path.join(path, f'{dir_name}/')

    if not os.path.isdir(dir):
        os.mkdir(dir)

    return dir

def _convert_to_dict(obj):
    """
    Convert a mesh object into a dictionary representation.

    Parameters:
    - obj: The mesh object to be converted.

    Returns:
    - dict: A dictionary representation of the mesh object.

    This function converts the given mesh object into a dictionary representation.
    It iterates through the attributes of the object and includes them in the dictionary.
    If certain attributes are instances of specific subclasses (like 'bcs', 'lbcs', 'crack'),
    it extracts and includes their attributes in the dictionary under the respective attribute names.
    It also handles a dynamically assigned dictionary object named 'circle' if present in the mesh object.
    """
    class_dict = {}

    for attr_name, attr_value in vars(obj).items():
        is_subclass_instance = False
        for subclass in [bcs, lbcs]:
            if isinstance(attr_value, subclass):
                class_dict.update({f"{attr_name}.{sub_attr}": sub_value for sub_attr, sub_value in vars(attr_value).items()})
                is_subclass_instance = True
                break  # Exit loop if a subclass instance is found

        if not is_subclass_instance:
            class_dict[attr_name] = attr_value

    # # Handle a dynamically assigned dictionary object named 'circle' if present in the mesh object
    # dynamic_dict_name = 'circle'
    # if hasattr(obj, dynamic_dict_name):
    #     sub_instance = getattr(obj, dynamic_dict_name)
    #     class_dict[f"{dynamic_dict_name}"] = sub_instance         

    return class_dict


def save_mesh(mesh_obj, obj_name=None, path=None):
    """
    Serialize and save a mesh object to a file.

    Args:
    - mesh_obj (Mesh): The mesh object to be serialized and saved.
    - obj_name (str, optional): Name of the file to save the mesh object. 
        If not provided, the default filename 'meshobj' will be used.
    - path (str, optional): Directory path where the file will be saved. 
        If specified, the file will be saved in this directory. 
        If not provided, the mesh object's 'folder' attribute will be checked.

    This function converts the given mesh object into a dictionary representation and serializes it, 
    saving it as a file. If no filename is specified (obj_name=None), the default filename 'meshobj' is used.
    
    If 'path' is provided, the file will be saved in that directory. If 'path' is not provided, 
    the function checks if the mesh object has a 'folder' attribute to determine the save location.
    
    Raises:
    - FileNotFoundError: If no valid directory is found to save the mesh object.
    - AttributeError: If the mesh cannot be saved in the specified directory.
    """
    dict_obj = _convert_to_dict(mesh_obj)

    filename = obj_name if obj_name else 'meshobj'

    if path and os.path.exists(path):
        dir_path = os.path.join(path, filename)
    elif mesh_obj.folder:
        dir_path = os.path.join(mesh_obj.folder, filename)
    else:
        raise FileNotFoundError(f"No valid directory found to save the mesh object")

    try:
        with open(dir_path, 'wb') as f:
            pickle.dump(dict_obj, f)
    except FileNotFoundError:
        raise AttributeError(f"Mesh cannot be saved in the specified directory: {dir_path}")


def load_mesh(dir, objname=None):
    """
    Loads a mesh object from a serialized file.

    Parameters:
    - dir (str): The directory path where the serialized mesh object is located.
    - objname (str, optional): The name of the serialized mesh object file. If not provided, defaults to 'meshobj'.

    Returns:
    - meshobj: The loaded mesh object.

    The function loads a serialized mesh object stored in a file and reconstructs the mesh object from its serialized form.
    The serialized file should contain essential mesh information in a dictionary format.

    Example usage:
    ```python
    loaded_mesh = load_mesh('/path/to/directory', 'my_mesh_object')
    ```
    """

    # If objname is not provided, set default value to 'meshobj'
    if objname is None:
        objname = 'meshobj'

    # Open the file in binary mode
    with open(dir + f'/{objname}', 'rb') as f:
        dict_obj = pickle.load(f)

    # Reconstruct mesh object from the loaded dictionary
    meshobj = mesh(
        nx=dict_obj['ny'],
        ny=dict_obj['nx'],
        a =dict_obj['a'],
        lattice=dict_obj['lattice'],
        pos=dict_obj['pos'],
        neighbors=dict_obj['neighbors'],
        angles=dict_obj['angles'],
        normal_stiffness=dict_obj['normal_stiffness'],
        tangential_stiffness=dict_obj['tangential_stiffness'],
        bottom=dict_obj['bottom'],
        top=dict_obj['top'],
        left=dict_obj['left'],
        right=dict_obj['right'],
        u=dict_obj['u']
    )

    # Assign additional attributes to the reconstructed mesh object
    meshobj.folder = dict_obj['folder']
    meshobj.domain = dict_obj['domain']
    meshobj.solver = dict_obj['solver']
    meshobj.crack_param = dict_obj['crack_param']
    meshobj.bond_prop = dict_obj['bond_prop']

    # If 'circle' information is present in the loaded dictionary, assign it to the mesh object
    if 'circle' in dict_obj.keys():
        meshobj.circle = dict_obj['circle']


    # Define a mapping of attributes between dictionary keys and subclasses' attributes
    attribute_map = {
        'bcs': {'comp': 'bcs.comp', 'ids': 'bcs.ids', 'parser': 'bcs.parser', 'fun': 'bcs.fun'},
        'lbcs': {'ids': 'lbcs.ids', 'fx': 'lbcs.fx', 'fy': 'lbcs.fy', 'fun': 'lbcs.fun'},
        }

    # Assign attributes from the dictionary to the respective sub-classes of mesh_obj
    for subclass, attributes in attribute_map.items():
        if hasattr(meshobj, subclass):
            subclass_obj = getattr(meshobj, subclass)
            for attr_name, dict_key in attributes.items():
                if dict_key in dict_obj:
                    setattr(subclass_obj, attr_name, dict_obj[dict_key])
                else:
                    print(f"Missing key '{dict_key}' for {subclass}")
        else:
            print(f"Subclass '{subclass}' doesn't exist in mesh_obj")    

    return meshobj



def _calculate_stiffness_poisson(nu):
    """
    Calculate stiffness parameters (kn, R) based on Poisson's ratio.

    Args:
    - nu (float): Poisson's ratio

    Returns:
    - kn (float): Normal stiffness
    - R (float): Tangential stiffness ratio
    """
    cl = math.sqrt((1 - nu) / ((1 + nu) * (1 - 2 * nu)))
    cs = math.sqrt(1 / (2 * (1 + nu)))
    cr = cs * (0.874 + 0.162 * nu)
    kn = (cl / cr) ** 2 - (1 / 3) * (cs / cr) ** 2
    R = ((cs / cr) ** 2 - (1 / 3) * (cl / cr) ** 2) / kn
    return kn, R

def _calculate_stiffness_properties(E, rho, a, nu):
    """
    Calculate stiffness parameters (kn, R) based on material properties.

    Args:
    - E (float): Young's modulus
    - rho (float): Density
    - a (float): Lattice parameter
    - nu (float): Poisson's ratio

    Returns:
    - kn (float): Normal stiffness
    - R (float): Tangential stiffness ratio
    """
    cl = math.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))
    cs = math.sqrt(E / (2 * rho * (1 + nu)))
    kn = (cl ** 2 - (1 / 3) * cs ** 2) / a ** 2
    R = (cs ** 2 - (1 / 3) * cl ** 2) / (kn * a ** 2)
    return kn, R

def bond_stiffness(mesh_obj:mesh, **kwargs):
    """
    Assign material properties for bond stiffness in the mesh.

    Args:
    - mesh_obj (Mesh): Mesh object containing node information
    - kn (float): Default normal stiffness (default: 1)
    - R (float): Default tangential stiffness ratio (default: 0.2)
    - nondim (bool): Flag indicating if the properties are non-dimensional (default: False)
    - **kwargs: Additional keyword arguments:
        - poisson_ratio (float): Poisson's ratio
        - youngs_modulus (float): Young's modulus
        - density (float): Density

    This function allows setting bond stiffness properties for the mesh. It accepts various keyword arguments to define the stiffness or calculate it based on material properties.
    If 'kn' and 'R' are both provided, it directly assigns these values. Alternatively, if 'poisson_ratio,' 'density,' and 'youngs_modulus' are passed, it calculates corresponding 'kn' and 'R.'
    If only 'poisson_ratio' is provided and 'nondim' is True, it calculates 'kn' and 'R' in non-dimensional units and initializes the mesh bond stiffnesses.
    """

    if 'kn' in kwargs and 'R' in kwargs:
        kn = kwargs['kn']
        R = kwargs['R']
    else:
        try:
            nu = kwargs['poisson_ratio']
            nondim = kwargs.get('nondim', False)
            if nondim:
                kn, R = _calculate_stiffness_poisson(nu)
            else:
                E = kwargs['youngs_modulus']
                rho = kwargs['density']
                a = mesh_obj.a
                kn, R = _calculate_stiffness_properties(E, rho, a, nu)
        except KeyError:
            raise KeyError("kn and R both must be directly specified, or poisson_ratio, density, youngs_modulus must be specified")

    # updating the bond_property parameter
    mesh_obj.bond_prop['poisson_ratio'] = nu
    mesh_obj.bond_prop['nondim'] = nondim

    # assigning to mesh object
    for idx in range(len(mesh_obj.pos)):
        mesh_obj.normal_stiffness[idx] = [kn] * len(mesh_obj.neighbors[idx])
        mesh_obj.tangential_stiffness[idx] = [R * kn] * len(mesh_obj.neighbors[idx])

          
"""function to view the mesh lattice"""
def mesh_plot(mesh_obj, filename = 'latticeview', title = 'latticeview', vectorfield = 'off', save = False):
    """
    Visualizes a triangular mesh lattice.

    Parameters:
    - mesh: The mesh object containing node positions, neighbors, etc.
    - filename (str): The filename for the saved plot (default: 'latticeview').
    - title (str): The title of the plot (default: 'latticeview').
    - vectorfield (str): Toggle for displaying vector field ('on' or 'off', default: 'off').
    - save (bool): Indicates whether to save the plot (default: False).

    Returns:
    - If save is False, displays the plot. Otherwise, saves the plot with the provided filename.
    """
    # Creating empty graph
    G = nx.Graph()
    Edgelist = []

    for id in range(0,len(mesh_obj.pos)):
        G.add_node(id, pos = mesh_obj.pos[id])
        for neigh, ns, aph in zip(mesh_obj.neighbors[id], mesh_obj.normal_stiffness[id], mesh_obj.angles[id]):
            if ns:
                if aph in [0, 60, 120]:
                    Edgelist.append([id, neigh])
                    
    G.add_edges_from(Edgelist)
    pos = nx.get_node_attributes(G, 'pos')

    fig, ax = plt.subplots(figsize = (10,10), dpi  = 300)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlim([mesh_obj.domain[0]-2, mesh_obj.domain[1]+2])
    ax.set_ylim([mesh_obj.domain[2]-2, mesh_obj.domain[3]+2])
    g = nx.draw_networkx_edges(G, pos, width = 0.1, edge_color='black')
    if vectorfield == 'on':
        ax.quiver(mesh_obj.pos[:,0], mesh_obj.pos[:,1], mesh_obj.u[:,0], mesh_obj.u[:,1], scale = 10, color = 'blue')
    if save:   
        if not mesh_obj.folder:
            mesh_obj.folder = _create_directory(f"{mesh_obj.ny}X{mesh_obj.nx}")

        # create snapshots directory insider folder
        snapshots = _sub_directory('snapshots', mesh_obj.folder)
        
        fig.savefig(snapshots + filename, bbox_inches = 'tight', dpi = 300)
        plt.close()    
    else:    
        plt.show()

def return_node_prop(mesh_obj, attr, *ij):
    """
    Retrieves node properties (IDs or positions) based on indices provided.

    Parameters:
    - mesh_obj: The mesh object containing node information.
    - attr (str): The property to retrieve ('id' or 'pos').
    - *ij: Variable number of indices (i, j) to retrieve node properties.

    Returns:
    - Tuple containing IDs or positions of nodes based on the provided indices.
    """
    if mesh_obj.lattice != 'triangle':
        raise ValueError("Unsupported lattice type. Expected 'triangle'.")

    if attr not in ('id', 'pos'):
        raise ValueError("Invalid attribute provided. Use 'id' or 'pos'.")

    def triangular_index_to_id(i, j):
        return i * mesh_obj.nx + j

    def get_node_id_pos(i, j):
        node_id = triangular_index_to_id(i, j)
        return node_id if attr == 'id' else mesh_obj.pos[node_id]

    return tuple(get_node_id_pos(i, j) for i, j in ij)


"""Function to create a circle at the center of the lattice"""
def create_hole(mesh_obj:mesh,center:tuple, radius:float):
    """
    Deletes nodes lying inside a given circle from the mesh.

    Parameters:
    - mesh_obj: The mesh object containing node information.
    - center (tuple): Center coordinates of the circle.
    - radius (float): Radius of the circle.
    """
    # define a coordinate of the circle
    theta = np.linspace(0,2*np.pi,100)
    circle_coord = np.zeros(shape=(100,2))
    circle_coord[:,0] = center[0] + radius*np.cos(theta)
    circle_coord[:,1] = center[1] + radius*np.sin(theta)
    circle = Polygon(circle_coord)

    # find nodes inside the circle
    total_nodes = mesh_obj.nx*mesh_obj.ny
    nodes_inside_circle = []
    for id in range(total_nodes):
        point = Point(mesh_obj.pos[id])
        if circle.contains(point):
            nodes_inside_circle.append(id)

    # find the nodes and normal vector on circle boundary
    circ_bound_nodes = []
    norm_vec = []
    for id in range(total_nodes):
        if id not in nodes_inside_circle:
            common = set(nodes_inside_circle).intersection(set(mesh_obj.neighbors[id]))
            if common:
                circ_bound_nodes.append(id)
                vec = mesh_obj.pos[id] - center
                norm_vec.append(vec/np.linalg.norm(vec,2))
    # create hole
    for id in nodes_inside_circle:
        delete_bonds(mesh_obj, id, neighbors = mesh_obj.neighbors[id])

    # dynamic dictionary initialization
    mesh_obj.circle = {'center': center, 'radius':radius,
                       'norm_vec':norm_vec, 'circ_bound_nodes':circ_bound_nodes}    


def delete_bonds(mesh_obj, node_id, k=0, **kwargs):
    """
    Delete bonds of the given node in the lattice by specifying stiffness value 'k'.

    Parameters:
    - mesh_obj: The mesh object containing node information.
    - node_id: The ID of the node whose bonds need to be deleted.
    - k (float): The stiffness value for deleted bonds (default: 0).
    - kwargs: Additional arguments. Either 'angles' or 'neighbors' should be provided.

    Keyword Arguments:
    - angles (list): List of angles corresponding to bonds to be deleted.
    - neighbors (list): List of neighbor IDs corresponding to bonds to be deleted.
    """
    if 'angles' in kwargs:
        angles_to_delete = kwargs['angles']
        bonds_to_delete = [i for i, angle in enumerate(mesh_obj.angles[node_id]) if angle in angles_to_delete]
    elif 'neighbors' in kwargs:
        neighbors_to_delete = kwargs['neighbors']
        bonds_to_delete = [i for i, neighbor in enumerate(mesh_obj.neighbors[node_id]) if neighbor in neighbors_to_delete]
    else:
        raise ValueError('KeywordError: Either "angles" or "neighbors" should be provided.')

    for bond_idx in bonds_to_delete:
        neighbor_id = mesh_obj.neighbors[node_id][bond_idx]
        neighbor_bond_idx = mesh_obj.neighbors[neighbor_id].index(node_id)

        mesh_obj.normal_stiffness[node_id][bond_idx] = k
        mesh_obj.tangential_stiffness[node_id][bond_idx] = k
        mesh_obj.normal_stiffness[neighbor_id][neighbor_bond_idx] = k
        mesh_obj.tangential_stiffness[neighbor_id][neighbor_bond_idx] = k

"""Function to analyse the mesh object from the script to choose the solver"""
def analyse_mesh(mesh_obj: mesh):
    """
    Analyzes the mesh object to check various parameters.

    Parameters:
    - mesh_obj (mesh): The mesh object to be analyzed.

    Returns:
    - crack_allowed (bool): Indicates if cracks are allowed based on the mesh configuration.
    - multiprocess (bool): Indicates if multiprocessing can be utilized based on mesh size.

    This function checks the mesh object for specific attributes to determine the following:
    - If any boundary conditions (displacement or load) are applied. Raises exceptions if not properly defined.
    - Whether cracks are allowed based on the mesh configuration.
    - Determines if multiprocessing can be utilized based on the mesh size.
    """    
    disp_boundary_cond = bool(getattr(mesh_obj, 'bcs', False) and mesh_obj.bcs.comp)
    load_boundary_cond = bool(getattr(mesh_obj, 'lbcs', False) and mesh_obj.lbcs.fun)

    if not disp_boundary_cond and not load_boundary_cond:
        raise Exception("No boundary condition applied")
    elif not disp_boundary_cond and load_boundary_cond:
        raise Exception("Lattice needs to be pinned")

    crack_allowed = bool(mesh_obj.crack_param)
    multiprocess = mesh_obj.nx * mesh_obj.ny > 10000

    return crack_allowed, multiprocess
         

"""Function to compute nodal strian tensor"""
def compute_nodal_stress_tensor(mesh_obj):
    """
    Compute the nodal strain tensor for a given mesh object.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.

    Returns:
    - strain (numpy.ndarray): Nodal strain tensor of shape (total_nodes, 4).

    This function computes the nodal strain tensor for a given mesh object. It initializes an array 'strain' to store
    the computed strain values for each node. The computation is performed for triangular lattice nodes. It iterates
    through each node, calculates the strain contributions from its neighbors, and computes the total strain at each
    node based on the node's connectivity, angles, and normal stiffness.
    """    
    total_nodes = mesh_obj.nx*mesh_obj.ny
    nu = mesh_obj.bond_prop['poisson_ratio']
    stress = np.zeros(shape = (total_nodes,4))
    if mesh_obj.lattice == 'triangle':
        for id in range(0,total_nodes):
            si = 0
            for neigh, aph, ns in zip(mesh_obj.neighbors[id],mesh_obj.angles[id], mesh_obj.normal_stiffness[id]):
                uij = mesh_obj.u[neigh] - mesh_obj.u[id]
                rij = np.array([np.cos(np.pi*np.array(aph)/180), np.sin(np.pi*np.array(aph)/180)])
                si += (ns!=0)*(1/6)*(uij[:,np.newaxis]@rij[np.newaxis,:] + \
                rij[:,np.newaxis]@uij[np.newaxis,:])

            stress[id,0] = (1/(1+nu))*(si[0,0] + nu/(1-2*nu)*(si[0,0]+si[1,1]))
            stress[id,3] = (1/(1+nu))*(si[1,1] + nu/(1-2*nu)*(si[0,0]+si[1,1]))
            stress[id,1] = stress[id,2] = (1/(1+nu))*si[0,1]
    return stress             

