from . import mesh, bcs, solver, crack, crackmp
from typing import Union

# Function #01: mesh generator
def MeshGenerator(nx: int, ny: int, a: float = 1.0):
    """
    Generate a triangular mesh lattice with specified dimensions and properties.

    Parameters:
    - nx (int): Number of columns in the lattice.
    - ny (int): Number of rows in the lattice.
    - a (float, optional): Lattice constant or node spacing (default is 1).

    Returns:
    - mesh: Mesh object containing the generated lattice information.

    This function generates a 2D triangular lattice based on the provided dimensions and lattice constants.
    It develops nodes for the lattice, establishing connections (bonds) between them based on the lattice structure.
    Node properties such as positions, neighbors, angles, and stiffness are defined according to the lattice configuration.
    The function creates a mesh object representing the generated lattice and returns it.
    """
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise TypeError("nx and ny must be integers")
    if not isinstance(a, (int, float)):
        raise TypeError("Lattice unit 'a' must be an integer or float")

    return mesh.mesh_generator(ny, nx, a)

# Function #02: create hole
def CreateHole(mesh_obj: mesh.mesh, center: tuple, radius: float):
    """
    Deletes nodes lying inside a given circle from the mesh.

    Parameters:
    - mesh_obj: The mesh object containing node information.
    - center (tuple): Center coordinates of the circle.
    - radius (float): Radius of the circle.
    """    
    if not isinstance(mesh_obj, mesh.mesh):
        raise TypeError("mesh_obj must be an instance of the 'mesh.mesh' class")

    if not isinstance(center, tuple) or len(center) != 2 or not all(isinstance(val, (int, float)) for val in center):
        raise TypeError("Center coordinate must be a tuple (xc, yc) with numeric values")

    if not isinstance(radius, (int, float)):
        raise TypeError("Radius must be an integer or float")    
    
    mesh.create_hole(mesh_obj, center, radius)



# Function #03: visualise the mesh lattice
def MeshPlot(mesh_obj: mesh.mesh, **kwargs):
    """
    Visualizes a triangular mesh lattice.

    Parameters:
    - mesh_obj (mesh.mesh): The mesh object containing node positions, neighbors, etc.
    - filename (str): The filename for the saved plot (default: 'latticeview').
    - title (str): The title of the plot (default: 'lattice view').
    - vectorfield (str): Toggle for displaying vector field ('on' or 'off', default: 'off').
    - save (bool): Indicates whether to save the plot (default: False).

    Returns:
    - If save is False, displays the plot. Otherwise, saves the plot with the provided filename.
    """
    filename = kwargs.get('filename', 'latticeview')
    title = kwargs.get('title', 'lattice view')
    vectorfield = kwargs.get('vectorfield', 'off')
    save = kwargs.get('save', False)
    
    if not isinstance(mesh_obj, mesh.mesh):
        raise TypeError("mesh_obj must be an instance of the 'mesh.mesh' class")
    
    mesh.mesh_plot(mesh_obj, filename=filename, title=title, vectorfield=vectorfield, save=save)

    
# Function #04: assigning the bond properties
def BondStiffness(mesh_obj:mesh.mesh,**kwargs):
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
    if not isinstance(mesh_obj, mesh.mesh):
        raise TypeError("mesh_obj must be of class mesh")    
                       
    mesh.bond_stiffness(mesh_obj, **kwargs) 

# Function #05: Save mesh object
def SaveMesh(mesh_obj:mesh.mesh, obj_name=None, path=None):
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
    if not isinstance(mesh_obj, mesh.mesh):
        raise TypeError("mesh_obj must be of class mesh")

    if not isinstance(obj_name, str) and obj_name is not None:
        raise TypeError("Invalid file name: must be string")  
        
    mesh.save_mesh(mesh_obj, obj_name, path)

# Function #06: Load the existing mesh
def LoadMesh(path:str, name=None)-> mesh.mesh:
    """
    Loads a mesh object from a serialized file.

    Parameters:
    - path (str): The directory path where the serialized mesh object is located.
    - name (str, optional): The name of the serialized mesh object file. If not provided, defaults to 'meshobj'.

    Returns:
    - meshobj: The loaded mesh object.

    The function loads a serialized mesh object stored in a file and reconstructs the mesh object from its serialized form.
    The serialized file should contain essential mesh information in a dictionary format.

    Example usage:
    ```python
    loaded_mesh = LoadMesh('/path/to/directory', 'my_mesh_object')
    ```
    """

    return mesh.load_mesh(path, name)

# Function #07: impose diplacement constant boundary conditions
def DirichletConstant(mesh_obj:mesh.mesh, ids:list, comp:str, value=0):
    """
    Assign constant displacement boundary conditions to specified node IDs of the mesh object.

    Parameters:
    - mesh_obj (class mesh): The mesh object to which the boundary condition is applied.
    - ids (list): Node IDs to which the constant displacement is applied.
    - comp (str): The direction of displacement - 'u' for horizontal, 'v' for vertical, 
                  or 'pin' for pinning the nodes.
    - value (float): The constant displacement value (default 0).

    This function assigns constant displacement boundary conditions to specified nodes 
    in the provided mesh object. The 'ids' parameter denotes the node IDs where the 
    boundary conditions will be applied. 'comp' specifies the direction of displacement 
    - 'u' for horizontal, 'v' for vertical, or 'pin' for pinning the nodes. The 'value' 
    parameter represents the constant displacement magnitude.

    Example:
    DirichletConstant(mesh_object, [2, 5, 9], 'u', 0.5)
    This example imposes a constant horizontal displacement of magnitude 0.5 to nodes 2, 5, and 9 in the mesh object.
    """  

    if not isinstance(mesh_obj,mesh.mesh):
        raise TypeError("mesh object must be of class mesh")

    if not isinstance(ids, list) or not all(isinstance(id,int) for id in ids):
        raise TypeError("ids must be list of integers")
    
    if comp not in ['u', 'v', 'pin']:
            raise TypeError("comp can take 'u', 'v', or 'pin' only")
        
    bcs.dirichlet_constant(mesh_obj, ids, comp, value)

# Function #08: Impose time-dependent boundary condtions
def DirichletFunction(mesh_obj:mesh.mesh, ids:list, comp:str, parser:str):
    """
    Assign time-dependent displacement boundary conditions to specified node IDs of the mesh object.

    Parameters:
    - mesh_obj (class mesh): The mesh object to which the boundary condition is applied.
    - ids (list): Node IDs to which the time-dependent displacement is applied.
    - comp (str): The direction of displacement - 'u' for horizontal and 'v' for vertical.
    - parser (str): A valid Python expression representing the displacement over time. 
                    It should contain the time variable 't', like '10*t', 'sin(t)', etc.

    Example:
    DirichletFunction(mesh_object, [2, 5, 9], 'u', '10 * t + 5 * sin(t)')
    This example imposes a time-dependent horizontal displacement to nodes 2, 5, and 9 in the mesh object.
    The displacement is represented as '10 * t + 5 * sin(t)', where 't' is the time variable.
    """
    if not isinstance(mesh_obj,mesh.mesh):
        raise TypeError("mesh object must be of class mesh")

    if not isinstance(ids, list) or not all(isinstance(id,int) for id in ids):
        raise TypeError("ids must be list of integers")
    
    if comp not in ['u', 'v', 'pin']:
            raise TypeError("comp can take 'u' and 'v' only")
    
    bcs.dirichlet_function(mesh_obj, ids, comp, parser)   

# Function #09: Impose Load boundary conditions
def LoadParserFunction(mesh_obj, ids, fx='0', fy='0', fun='parser'):
    """
    Assign time-dependent load boundary conditions to specified node IDs of the mesh object.

    Parameters:
    - mesh_obj (class mesh): The mesh object to which the boundary condition is applied.
    - ids (list): Node IDs to which the time-dependent load is applied.
    - fx (str or float): Horizontal force applied to the node. 
                         For constant force magnitude, pass a float; for time-dependent force magnitude, 
                         provide a valid Python expression with the time variable 't', like '0.1*t', 'sin(5*t)', etc. 
                         (default '0')
    - fy (str or float): Vertical force applied to the node. 
                         Passed in the same manner as parameter 'fx'. (default '0')
    - fun (str): Kind of the load (default 'parser')
                 Available options:
                 - 'parser': continuous time-dependent function
                 - 'impulse': load applied only at the first iteration and then removed
                 - 'ramp_impulse': time-dependent load function applied until t = 1 and then removed
                 - 'ramp': time-dependent load function applied until t = 10 and then removed

    This function assigns time-dependent load boundary conditions to specified nodes in the provided mesh object.
    The 'ids' parameter denotes the node IDs where the load conditions will be applied. 'fx' and 'fy' represent 
    the horizontal and vertical forces, respectively. They can be either constant float values or valid Python 
    expressions involving the time variable 't'. The 'fun' parameter defines the type of load applied, with 
    options like 'parser', 'impulse', 'ramp_impulse', or 'ramp'.

    Example:
    LoadParserFunction(mesh_object, [2, 5, 9], '0.1 * t', 'sin(t)', 'ramp')
    This example applies a time-dependent load on nodes 2, 5, and 9 in the mesh object. The horizontal force 
    is '0.1 * t', representing a ramp-type load, and the vertical force is 'sin(t)', a continuous time-dependent function.
    """
    if not isinstance(mesh_obj,mesh.mesh):
        raise TypeError("mesh object must be of class mesh")

    if not isinstance(ids, list) or not all(isinstance(id,int) for id in ids):
        raise TypeError("ids must be list of integers")
    
    bcs.load_parser_function(mesh_obj, ids, fx, fy, fun)  


# Function #10: Impose circular boundary conditions
"""Function to read boundary condition for circle"""

def HoleBoundaryCondition(mesh_obj:mesh.mesh, force: Union[str, float], fun='radial'):
    """
    Assign time-dependent load boundary conditions to the hole created in the triangular lattice mesh.

    Parameters:
    - mesh_obj (class mesh): The mesh object containing the hole.
    - force (str or float): The horizontal force applied to the hole.
                            For constant force magnitude, pass a float; for time-dependent force
                            magnitude, provide a valid Python expression with the time variable 't', 
                            like '0.1*t', 'sin(5*t)', etc.
    - fun (str): Kind of the load (default 'radial')
                Available options:
                - 'radial': radial load applied to the hole surface
                - 'tangential': load applied tangential to the hole surface
                - 'radial_impulse': radial time-dependent load function applied until t = 10 and then removed
                - 'tangential_impulse': tangential time-dependent load function applied until t = 10 and then removed

    This function assigns time-dependent load boundary conditions to the hole within the triangular lattice mesh.
    The 'mesh_obj' parameter should contain the information about the hole.
    'force' denotes the horizontal force applied to the hole; it can be a constant float value or a valid Python 
    expression involving the time variable 't'. The 'fun' parameter defines the type of load applied to the hole, 
    with options like 'radial', 'tangential', 'radial_impulse', or 'tangential_impulse'.

    Example:
    HoleBoundaryCondition(mesh_object, '0.5 * t', 'tangential')
    This example applies a time-dependent tangential load of '0.5 * t' to the hole in the mesh object.
    """
    if not isinstance(mesh_obj,mesh.mesh):
        raise TypeError("mesh object must be of class mesh")
    
    bcs.hole_boundary_condition(mesh_obj, force, fun)   

# Function #11: Breaking parameters for bond breaking
def BreakingParameters(mesh_obj:mesh.mesh, prop:str, threshold:float, **kwargs):
    """
    Set parameters for mesh cracking within the mesh object.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - prop (str): The property affecting the mesh cracking behavior and can take 'stretch' or 'strain'.
    - threshold (float): The threshold value for the specified property.
    - **kwargs: Additional keyword arguments:
        - comp (str, optional): Required if 'prop' is 'stretch'. Specifies the component ('normal' or 'tangential')
          for stretching.

    Returns:
    - None

    This function configures the parameters related to mesh cracking within the given mesh object.
    'prop' defines the property influencing the cracking behavior, 'threshold' sets the threshold value
    for this property. If 'prop' is 'stretch', the 'comp' keyword argument is required to specify the
    stretching component. 
    """

    crack.breaking_parameters(mesh_obj,prop, threshold, **kwargs)

# Function #12: Edge crack
def EdgeCrack(mesh_obj:mesh.mesh, crack_length:int, row=0, right=False):
    """
    Create an edge crack in the mesh.

    Parameters:
    - mesh_obj (object): Mesh object containing the geometry and connectivity.
    - crack_length (int): Length of the crack to be created.
    - row (int, optional): Row index where the crack starts or ends. Default is 0.
    - right (bool, optional): Indicates if the crack starts from the right side. Default is False.

    This function creates an edge crack in the mesh by deleting bonds based on the specified crack length,
    starting row index, and side (right or left). It identifies the crack node IDs and deletes bonds
    associated with those nodes to create the crack.
    """
    crack.edge_crack(mesh_obj, crack_length, row, right)

# Function #13: Solver
def Solver(mesh_obj, dt, endtime, **kwargs):
    """
    Solve the mesh system dynamics over time.

    Parameters:
    - mesh_obj (object): Mesh object containing system information.
    - dt (float): Time step for integration.
    - endtime (float): End time for simulation.
    - zeta (float, optional): Damping coefficient. Default is 0.
    - vectorfield (str, optional): Vector field visualization mode. Default is 'off'.
    - folder (str, optional): Folder name to store data. Default is None.
    - **kwargs: Additional keyword arguments.

    This function integrates the mesh system dynamics over time using Verlet integration.
    It imposes boundary conditions, computes load boundary conditions, updates the system state,
    and saves displacement data and deleted bonds during the simulation.
    """
    static = kwargs.get('static', False)
    
    # Analyze the mesh and decide the solver
    crack_allowed, multiprocess = mesh.analyse_mesh(mesh_obj)

    if crack_allowed:
        if multiprocess:
            crackmp.solve(mesh_obj, dt, endtime, **kwargs)
        else:
            crack.solve(mesh_obj, dt, endtime, **kwargs)
    else:
        if static:
            solver.static_solve(mesh_obj, dt, endtime, **kwargs)
        else:    
            solver.solve(mesh_obj, dt, endtime, **kwargs)
