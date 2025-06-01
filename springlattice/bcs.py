import numpy as np
from math import sin, cos
import ast



def is_valid_expression(expr):
    """
    Check if the provided string is a valid Python mathematical expression 
    containing specific functions and variables.

    Args:
    - expr (str): The expression to be validated.

    Returns:
    - bool: True if the expression is valid according to the specified criteria, False otherwise.
    """
    try:
        parsed = ast.parse(expr, mode='eval')

        # Custom check for valid functions
        allowed_functions = {'sin', 'cos', 'tan'}  # Add more as needed
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in allowed_functions:
                    return False  # Invalid function found

                if not node.args:
                    return False  # Empty function call

        # Check for 't' variable if any Name nodes are present
        names = {node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)}
        for var in names:
            if len(var)==1 and var!='t':
                return False

        return True

    except SyntaxError:
        return False



"""function to obtain Dirichlet displacement boundary condition"""
def dirichlet_constant(mesh_obj, ids, comp, value=0):
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
    # update the mesh object
    parser = f"{value}"
    mesh_obj.bcs.ids.append(ids)
    mesh_obj.bcs.parser.append(parser)
    mesh_obj.bcs.comp.append(comp)
    mesh_obj.bcs.fun.append('constant')
     
  
"""Function to obtain Dirichlet function boundary condition"""
def dirichlet_function(mesh_obj, ids, comp, parser):
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

    if not is_valid_expression(parser):
        raise SyntaxError("Parser contains invalid python expression") 
    
    # update the mesh object
    mesh_obj.bcs.ids.append(ids)
    mesh_obj.bcs.parser.append(parser)
    mesh_obj.bcs.comp.append(comp)
    mesh_obj.bcs.fun.append('variable')
  
           


"""Parser function for loading"""  
def load_parser_function(mesh_obj, ids, fx='0', fy='0', fun='parser'):
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
    fun_args = ['parser', 'impulse', 'ramp_impulse', 'ramp']
    if fun not in fun_args:
        raise NameError(f"Valid arguments for 'fun' keyword: {fun_args}")

    # convert fx and fy to string if float value is passed
    if not isinstance(fx, str):
        fx = str(fx)
    if not isinstance(fy,str):
        fy = str(fy)   
    # check the parser fx and fy are valid python expression
    if not is_valid_expression(fx) or not is_valid_expression(fy):
        raise SyntaxError("Either fx or fy contains invalid python expression")   
      
    # update the mesh object
    mesh_obj.lbcs.ids.append(ids)
    mesh_obj.lbcs.fx.append(fx)
    mesh_obj.lbcs.fy.append(fy)
    mesh_obj.lbcs.fun.append(fun)

"""Function to read boundary condition for circle"""
def hole_boundary_condition(mesh_obj, force, fun='radial'):
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
    
    try:
        ids = mesh_obj.circle['circ_bound_nodes']
    except KeyError:
        raise Exception("The mesh object does not contain any hole")  
    
    if not isinstance(force, str):
        force = str(force)

    if not is_valid_expression(force):
        raise SyntaxError("Force is not valid Python mathematical expression")
    
    fun_args = ['radial', 'radial_impulse', 'tangential', 'tangential_impulse']
    if fun not in fun_args:
        raise NameError(f"Valid arguments for 'fun' keyword: {fun_args}")
    
    # update the mesh object  
    mesh_obj.lbcs.ids.append(ids)
    mesh_obj.lbcs.fx.append(force)  # store force to fx
    mesh_obj.lbcs.fy.append('0')   # set fy as 0 (tangential direction)
    mesh_obj.lbcs.fun.append(fun)
 


"""Function to impose load boundary conditions for verlet integrator"""
def impose_loads(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t, norm_vec = None):
    """
    Update load boundary conditions for specified node IDs in the mesh.

    Parameters:
    - total_nodes (int): Total number of nodes in the mesh.
    - lbcs_ids (list of lists): List of node IDs to which the load is applied.
    - lbcs_fx (list): List of expressions or values for horizontal force.
    - lbcs_fy (list): List of expressions or values for vertical force.
    - lbcs_fun (list): List of load types.
    - t (float): Time variable.
    - norm_vec (numpy array, optional): Normal vector for radial loads applied to inside hole.

    Returns:
    - load (numpy array): Updated load array.

    This function updates the load boundary conditions for specified node IDs in the mesh. It takes arrays or lists 
    of node IDs ('lbcs_ids'), horizontal ('lbcs_fx') and vertical ('lbcs_fy') force expressions or values, load 
    types ('lbcs_fun'), and the time variable 't'. The 'norm_vec' parameter is an optional argument representing 
    the normal vector for radial loads.
    """
    # define lambda for getting indices
    u = lambda i: 2*i
    v = lambda i: 2*i + 1

    if norm_vec is not None:
        norm_vec = np.array(norm_vec)
 
    load = np.zeros(shape=(2*total_nodes,1))

    for i, fun in enumerate(lbcs_fun):
        id = np.array(lbcs_ids[i])
        if fun == 'parser':
            load[u(id)] = eval(lbcs_fx[i])
            load[v(id)] = eval(lbcs_fy[i])
        elif fun == 'impulse' and t == 0:
            load[u(id)] = eval(lbcs_fx[i])
            load[v(id)] = eval(lbcs_fy[i])
        elif fun == 'ramp_impulse':
            if t<=1:
                load[u(id)] = (t/10)*eval(lbcs_fx[i])
                load[v(id)] = (t/10)*eval(lbcs_fy[i])
        elif fun == 'ramp':
            if t<=10:
                load[u(id)] = (t/10)*eval(lbcs_fx[i])
                load[v(id)] = (t/10)*eval(lbcs_fy[i])
            else:
                load[u(id)] = eval(lbcs_fx[i])
                load[v(id)] = eval(lbcs_fy[i])
        elif fun == 'radial':
            load[u(id),0] = eval(lbcs_fx[i])*norm_vec[:,0]
            load[v(id),0] = eval(lbcs_fx[i])*norm_vec[:,1]  
        elif fun == 'radial_impulse':
            if t<10:
                load[u(id),0] = eval(lbcs_fx[i])*norm_vec[:,0]
                load[v(id),0] = eval(lbcs_fx[i])*norm_vec[:,1]

    return load  

"""impose boundary conditons for verlet integrator"""
def impose_displacement(disp:np.ndarray,ids:list,comp:list,parser:list,fun:list,t:float):  
    """
    Update the displacement boundary conditions for specified node IDs in the mesh.

    Parameters:
    - disp (numpy array): Displacement array containing the displacements for nodes.
    - ids (list of lists): List of node ID lists to which the displacement is applied.
    - comp (list): List of directions ('u' or 'v') for horizontal and vertical displacements.
    - parser (list): List of valid Python expressions as strings with the time variable 't'.
    - fun (list): List of functions indicating the type of displacement ('constant', 'variable', or 'pin').
    - t (float): Time variable.

    Returns:
    - disp (numpy array): Updated displacement array.

    This function updates the displacement boundary conditions for specified node IDs in the mesh. It takes arrays 
    or lists of node IDs ('ids'), directions ('comp'), Python expressions ('parser'), and functions ('fun') to update 
    the displacement boundary conditions for these nodes. The 't' parameter represents the time variable used in the 
    Python expressions provided.

    Example:
    update_displacement(displacement_array, [[1, 2, 3], [5, 6, 9]], ['u', 'v', 'u'], ['0.5 * t', 'sin(t)', '0'], ['variable', 'variable', 'pin'], 10)
    This example updates the displacement array with time-dependent displacements for nodes 1, 2, and 3 in the 'u' and 'v'
    directions based on provided Python expressions at time t = 10.
    """
    # variables suffix corresponding to node id 'i'
    u = lambda i: 2*i   # indices of the u displacement of node i
    v = lambda i: 2*i + 1   # indices of the u displacement of node i

    for i, f in enumerate(fun):
        id = np.array(ids[i])   # required to convert list to numpy array to use vectorization
        if f == 'constant' or f == 'variable':
            if comp[i] == 'u':
                disp[u(id)] = eval(parser[i]) 
            elif comp[i] == 'v':
                disp[v(id)] = eval(parser[i]) 
            elif comp[i] == 'pin':
                disp[u(id)] = 0
                disp[v(id)] = 0           
               
    return disp 



