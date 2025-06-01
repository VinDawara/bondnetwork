from . import mesh, bcs, solver
import numpy as np
import pickle
import h5py
import copy
import math

 
def breaking_parameters(mesh_obj, prop, threshold, **kwargs):
    """
    Set parameters for mesh cracking within the mesh object.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - prop (str): The property affecting the mesh cracking behavior.
    - threshold (float): The threshold value for the specified property.
    - **kwargs: Additional keyword arguments:
        - comp (str, optional): Required if 'prop' is 'stretch'. Specifies the component for stretching.

    Returns:
    - None

    This function configures the parameters related to mesh cracking within the given mesh object.
    'prop' defines the property influencing the cracking behavior, 'threshold' sets the threshold value
    for this property. If 'prop' is 'stretch', the 'comp' keyword argument is required to specify the
    stretching component. 
    """
    if not isinstance(mesh_obj,mesh.mesh):
        raise TypeError("mesh object must be of class mesh")
    
    if prop not in ['strain', 'stretch']:
        raise NameError("Prop can take 'stretch' or 'strain")
    
    if not isinstance(threshold, (int,float)):
        raise TypeError("threshold must be either integer or float")
    
    mesh_obj.crack_param.update({'prop':prop, 'threshold':threshold})


    if prop == 'stretch':
        if 'comp' not in kwargs:
            raise ValueError("Keyword argument 'comp' is required when 'prop' is 'stretch'")
        mesh_obj.crack_param['comp'] = kwargs['comp']


"""Updated verlet integrator using matrix multiplication"""
def solve(mesh_obj, dt, endtime, **kwargs):
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
    # set keyward options
    zeta = kwargs.get('zeta', 0)
    vectorfield = kwargs.get('vectorfield', 'off')
    folder = kwargs.get('folder', "sl_crack")
    interval = kwargs.get('interval', False)
    save_ratio = kwargs.get('save_ratio', 0)    

    # creating directory for storing data
    mesh_obj.folder = mesh._create_directory(f"{folder}_{mesh_obj.ny}X{mesh_obj.nx}")

    mesh_obj.solver.update({'dt': dt, 'endtime': endtime, 'name': 'verlet','zeta':zeta})
    maxsteps = math.ceil(endtime/dt)
    skipsteps = math.ceil((save_ratio/100)*maxsteps) if save_ratio else 1
    mesh_obj.solver['skipsteps'] = skipsteps

    # save initial mesh
    mesh.save_mesh(mesh_obj)

    # initial mesh position
    pos = copy.deepcopy(mesh_obj.pos)

    # extract displacement boundary conditions
    bcs_ids = copy.copy(mesh_obj.bcs.ids)
    bcs_parser = copy.copy(mesh_obj.bcs.parser)
    bcs_comp = copy.copy(mesh_obj.bcs.comp)
    bcs_fun = copy.copy(mesh_obj.bcs.fun)

    # if lattice contains hole, radial unit vector inside the hole
    norm_vec = mesh_obj.circle.get('norm_vec', None) if hasattr(mesh_obj, 'circle') else None
   
    # extract load boundary conditions
    lbcs_ids = copy.copy(mesh_obj.lbcs.ids)
    lbcs_fx = copy.copy(mesh_obj.lbcs.fx)
    lbcs_fy = copy.copy(mesh_obj.lbcs.fy)
    lbcs_fun = copy.copy(mesh_obj.lbcs.fun)

    # create a matrix A
    A = solver.generate_matrix(mesh_obj)

    # displacement variables for time step 't-1', 't' and 't+1', respectively
    u_prev, u_curr, u_next = (solver.flatten(mesh_obj.u) for _ in range(3))

    # impose boundary condition at t = 0
    u_curr = bcs.impose_displacement(u_curr,ids=bcs_ids, comp=bcs_comp, parser=bcs_parser,
                                fun=bcs_fun, t=0)
    u_prev = copy.deepcopy(u_curr)

    # Creating handler to file for displacement data
    total_nodes = len(pos)
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'w')
    dset_u = disp_file.create_dataset(
        name = 'u',
        shape=(0,total_nodes),
        maxshape = (None, total_nodes),
        dtype='float64',
        compression = 'gzip',
        compression_opts = 9
    )
    dset_v = disp_file.create_dataset(
        name = 'v',
        shape=(0,total_nodes),
        maxshape = (None, total_nodes),
        dtype='float64',
        compression = 'gzip',
        compression_opts = 9
    )

    # displacement bucket to write to file
    bucket = 0
    fill_steps = 0
    bucket_idx = 0
    remain_steps = int(maxsteps/skipsteps)
    bucket_size = min(1000,remain_steps)
    # bucket size to be stored in file
    U = np.zeros(shape=(bucket_size, total_nodes))
    V = np.zeros(shape=(bucket_size, total_nodes))

    # file handler for saving deleted bonds
    bonds_file = open(mesh_obj.folder + '/delbonds','wb')
    
    # integration begins
    for step in range(maxsteps):
        # time variable
        t = step*dt

        # velocity of the nodes
        v = (u_next - u_prev)/(2*dt)

        # impose load boundary condtions
        load = bcs.impose_loads(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t,
                               norm_vec=norm_vec)

        # time integration
        u_next= 2*u_curr - u_prev - (A@u_curr - load + zeta*v)*dt**2

        # impose boundary conditions
        u_next = bcs.impose_displacement(u_next, ids=bcs_ids, comp=bcs_comp, parser=bcs_parser,
                                fun=bcs_fun, t=t)

        mesh_obj.u = solver.reshape2vector(u_next)
        deleted_bonds = activate_breaking(mesh_obj)
        if deleted_bonds:
            A = solver.update_A(mesh_obj,A, deleted_bonds)
            # deleted bonds
            pickle.dump([t,deleted_bonds], bonds_file)

        # update node object
        if interval and step%int(interval/dt) == 0: 
            mesh_obj.pos = pos + mesh_obj.u
            mesh.mesh_plot(mesh_obj,filename = f"step_{step}.png", title = f'T = {np.round(t,3)}', vectorfield = vectorfield, save=True)
            
        
        u_prev = copy.deepcopy(u_curr)
        u_curr = copy.deepcopy(u_next)
        print('Time step = ',step, 'T = %0.4f' % t, 'Progress = %0.2f' % (100*step/maxsteps))

        # saving displacement data to disk
        if step%skipsteps == 0:
            # saving displacements fields
            u_shape = solver.reshape2vector(u_next)
            U[bucket_idx] = u_shape[:,0]
            V[bucket_idx] = u_shape[:,1] 
            bucket_idx += 1

            if bucket_idx == bucket_size: # if variable is full, empty bucket 
                dset_u.resize(dset_u.shape[0]+bucket_size, axis = 0)
                dset_u[-bucket_size:] = U
                dset_v.resize(dset_v.shape[0]+bucket_size, axis = 0)
                dset_v[-bucket_size:] = V
                bucket += 1
                fill_steps += bucket_size
                remain_steps += -bucket_size 
                bucket_idx = 0
                
                if  remain_steps < bucket_size:
                    bucket_size = remain_steps
                    U = np.zeros(shape = (bucket_size,total_nodes))
                    V = np.zeros(shape = (bucket_size,total_nodes))
                
    disp_file.close()
    bonds_file.close()
    print(f'Solver completed. Data saved to {mesh_obj.folder}')

        
"""Function to include bond breaking condition for dyanmic solver"""
def activate_breaking(mesh_obj):
    """
    Activate breaking of bonds in the mesh based on specified properties.

    Parameters:
    - mesh_obj (object): An object representing the mesh system with crack parameters.

    Returns:
    - deleted_bonds (list): List of deleted bond pairs. If no bonds are deleted, returns an empty list.

    This function activates the breaking of bonds within the mesh based on defined properties.
    It checks specific bond properties such as 'stretch' or 'strain' and their associated thresholds.
    Bonds exceeding the threshold are marked as deleted.
    """    
    prop = mesh_obj.crack_param['prop']
    
    if prop == 'stretch':
        comp = mesh_obj.crack_param['comp']
        threshold = mesh_obj.crack_param['threshold']
        deleted_bonds = check_stretched_bonds(mesh_obj, comp=comp, threshold=threshold) 

    elif prop == 'strain':
        threshold = mesh_obj.crack_param['threshold']
        deleted_bonds = check_strained_bonds(mesh_obj, threshold = threshold)  

    if deleted_bonds:
        for id, neigh in deleted_bonds:
            update_bond_state(mesh_obj, node_id=id, neighbor_id = neigh)

    return deleted_bonds          

"""Function to find critical bonds based on stretch"""
def check_stretched_bonds(mesh_obj,comp, threshold):
    """
    Check for stretched bonds in the mesh based on specified properties.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - comp (str): Component to calculate stretching ('normal', 'tangential', or 'abs').
    - threshold (float): Threshold value for bond stretching.

    Returns:
    - critical_bonds (list): List of bond pairs that exceed the threshold.

    This function checks for stretched bonds in the mesh based on the specified component and threshold.
    It iterates through all nodes and their neighbors, calculating bond stretching based on the given component.
    Bonds exceeding the threshold are marked as critical and returned as a list of bond pairs.
    """    
    critical_bonds = []       
    total_nodes = mesh_obj.nx*mesh_obj.ny

    for id in range(total_nodes):
        for neigh, aph, ns in zip(mesh_obj.neighbors[id],mesh_obj.angles[id], mesh_obj.normal_stiffness[id]):
            uij = mesh_obj.u[neigh] - mesh_obj.u[id]

            if comp == 'normal':
                rij = np.array([np.cos(np.pi*np.array(aph)/180), np.sin(np.pi*np.array(aph)/180)])
                value = (ns!=0)*np.dot(uij,rij)

            elif comp == 'tangential':
                tij = np.array([-np.sin(np.pi*np.array(aph)/180), np.cos(np.pi*np.array(aph)/180)])
                value = (ns!=0)*np.dot(uij,tij)
    
            elif comp == 'abs':
                value = np.linalg.norm(uij)/mesh_obj.a

            else:
                raise Exception('Wrongly assigned argument to \'comp\' keyword ')    

            if value >= threshold:
                critical_bonds.append([id, neigh])               

    return critical_bonds

             
def check_strained_bonds(mesh_obj, threshold):
    """
    Check for strained bonds in the mesh based on a specified threshold.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - threshold (float): Threshold value for bond strain.

    Returns:
    - critical_bonds (list): List of bond pairs that exceed the threshold.

    This function checks for strained bonds in the mesh based on the provided strain tensor and threshold.
    It computes the nodal strain tensor, then iterates through nodes and their neighbors to calculate
    the bond strain. If the principal strain of a bond exceeds the given threshold, it marks it as critical.
    The function returns a list of bond pairs that exceed the threshold.
    """    
    strain = compute_nodal_strain_tensor(mesh_obj)
    critical_bonds = []

    for node_id, node_neighbors in enumerate(mesh_obj.neighbors):
        for neigh, ns in zip(node_neighbors, mesh_obj.normal_stiffness[node_id]):
            if ns:
                bond_strain = 0.5 * (strain[node_id] + strain[neigh])
                principal_strain = principal_value(bond_strain)

                if principal_strain >= threshold:
                    critical_bonds.append([node_id, neigh])

    return critical_bonds


"""Function to compute nodal strian tensor"""
def compute_nodal_strain_tensor(mesh_obj):
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
    strain = np.zeros(shape = (total_nodes,4))
    if mesh_obj.lattice == 'triangle':
        for id in range(0,total_nodes):
            si = 0
            for neigh, aph, ns in zip(mesh_obj.neighbors[id],mesh_obj.angles[id], mesh_obj.normal_stiffness[id]):
                uij = mesh_obj.u[neigh] - mesh_obj.u[id]
                rij = np.array([np.cos(np.pi*np.array(aph)/180), np.sin(np.pi*np.array(aph)/180)])
                si += (ns!=0)*(1/6)*(uij[:,np.newaxis]@rij[np.newaxis,:] + \
                rij[:,np.newaxis]@uij[np.newaxis,:])

            strain[id,:] = si.reshape(1,4)
    return strain        

"""Function to compute principal strain (Not mesh object)"""
def principal_value(t, type = 'maximum'):
    """
    Compute the principal value of a tensor.

    Parameters:
    - t (list or numpy.ndarray): Input tensor of length 4 representing strain components.
    - type (str): Type of principal value to compute. Default is 'maximum'.

    Returns:
    - ps (float): Principal value of the tensor.

    This function computes the principal value of a tensor given as input, specifically designed for tensors
    representing strain components. The 'type' parameter determines the computation type. For 'maximum' type,
    it calculates the maximum principal value based on the input tensor components (t[0], t[1], t[3]).
    """    
    if type == 'maximum':
        ps = 0.5*(t[0] + t[3]) + math.sqrt((0.5*(t[0]-t[3]))**2 + \
            t[1]**2)
    return ps 

    

def update_bond_state(mesh_obj, node_id, neighbor_id, k=0):
    """
    Update bond states of the mesh for the specified bond.

    Parameters:
    - mesh_obj (object): Mesh object containing the bond states and properties.
    - node_id (int): Node ID of the bond.
    - neighbor_id (int): Neighbor node ID of the bond.
    - k (int, optional): New stiffness value to assign. Default is 0.

    This function updates the bond states (stiffness) in the mesh object for the specified bond.
    It updates the stiffness (both normal and tangential) for the given bond and its corresponding neighbor
    with the provided stiffness value 'k'.
    """
    id_idx = mesh_obj.neighbors[node_id].index(neighbor_id)
    neighbor_idx = mesh_obj.neighbors[neighbor_id].index(node_id)

    mesh_obj.normal_stiffness[node_id][id_idx] = k
    mesh_obj.tangential_stiffness[node_id][id_idx] = k
    mesh_obj.normal_stiffness[neighbor_id][neighbor_idx] = k
    mesh_obj.tangential_stiffness[neighbor_id][neighbor_idx] = k

   

def edge_crack(mesh_obj, crack_length, row=0, right=False):
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
    if not isinstance(mesh_obj, mesh.mesh):
        raise TypeError("mesh_obj must be of class mesh")
    
    if not isinstance(crack_length, int):
        raise TypeError("crack_length must be integer")

    if right:
        end_id = mesh_obj.right[row] + 1 if row else mesh_obj.right[math.ceil(0.5 * len(mesh_obj.right)) - 1] + 1
        start_id = end_id - crack_length
    else:
        start_id = mesh_obj.left[row] if row else mesh_obj.left[math.ceil(0.5 * len(mesh_obj.left)) - 1]
        end_id = start_id + crack_length

    crack_ids = range(start_id, end_id)
    for node_id in crack_ids:
        neighbors = [neighbor for neighbor in mesh_obj.neighbors[node_id] if neighbor > node_id + 1]
        mesh.delete_bonds(mesh_obj, node_id, neighbors=neighbors)

