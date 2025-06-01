from . import mesh, bcs, solver
import numpy as np
import pickle
import h5py
import copy
import math
import multiprocessing as mp


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
    folder = kwargs.get('folder', "slmp_crack")
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
    u_prev, u_curr, u_next = (solver.flatten(mesh_obj.u) for i in range(3))

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
        deleted_bonds = par_check_strained_bonds(mesh_obj, threshold = threshold)  

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


"""Function to chunk the mesh object"""
def chunks(total_nodes:int, num_processor:int) -> list:
    """
    Divide the nodes into chunks for parallel processing.

    Parameters:
    - total_nodes (int): Total number of nodes in the dataset.
    - num_processor (int): Number of processors to split the nodes.

    Returns:
    - batchs (list): List of tuples containing start and end node indices for each chunk.

    This function divides a dataset into chunks to distribute computational workload
    among multiple processors for parallel processing. It determines the chunk size based on
    the total number of nodes and the specified number of processors. The returned list
    contains tuples indicating the start and end indices of nodes for each chunk.
    """    
    chunk_size = np.ceil(total_nodes/num_processor).astype(int)
    batchs = []
    start = 0
    for i in range(num_processor):
        end = start + chunk_size - 1
        batchs.append((start,end))
        start = end + 1
        # Adjust the remaining nodes to the last batch
        remain = total_nodes-start
        if remain <= chunk_size:
            chunk_size = remain      
    return batchs        


"""Function to find critical bonds based on strain tensor"""
def par_check_strained_bonds(mesh_obj, threshold):
    """
    Parallel computation to check for strained bonds in the mesh based on a specified threshold.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - threshold (float): Threshold value for bond strain.

    Returns:
    - criticalbonds (list): List of bond pairs that exceed the threshold.

    This function divides the nodes of the mesh into chunks and performs parallel computation
    to check for strained bonds in the mesh based on the provided strain tensor and threshold.
    It computes the nodal strain tensor in parallel and then identifies critical bonds using
    parallel processing. Finally, it aggregates the results and returns a list of bond pairs that
    exceed the threshold.
    """           
    total_nodes = mesh_obj.nx*mesh_obj.ny
    # chunking the total nodes into 6 pieces
    batches = chunks(total_nodes, num_processor=6)
    # computing strain in parallel
    with mp.Pool(processes=6) as pool:
        strain = pool.starmap(compute_nodal_strain_tensor, [(mesh_obj, batch) for batch in batches])
    # stacking all the results
    strain = np.vstack(tuple(strain))

    # finding the critical bonds in parallel
    with mp.Pool(processes=6) as pool:
        par_critical_bonds = pool.starmap(critical_strained_bonds, [(mesh_obj, strain, threshold, batch)
                                     for batch in batches])
    # combining all the batch results
    critical_bonds = []
    for bonds in par_critical_bonds:
        critical_bonds += bonds    
    return critical_bonds        
  

"""Function to check critical bonds on strain tensor"""
def compute_nodal_strain_tensor(mesh_obj, ij):
    """
    Compute the nodal strain tensor for a specified range of nodes.

    Parameters:
    - mesh_obj (mesh.mesh): The mesh object containing node information.
    - ij (tuple): Tuple indicating the start and end indices of nodes for computation.

    Returns:
    - strain (numpy.ndarray): Array containing the computed strain tensor for the specified nodes.

    This function calculates the nodal strain tensor for a given range of nodes within a mesh object.
    The 'ij' parameter defines the start and end indices of the nodes for which the strain tensor
    needs to be computed. The function iterates through the specified node range, calculating the
    strain tensor based on neighboring nodes' information, stiffness, and angles. The computed
    strain tensor is returned as a NumPy array.
    """    
    start = ij[0]; end = ij[1]
    chunk_nodes = end-start+1
    strain = np.zeros(shape=(chunk_nodes,4))
    for i in range(chunk_nodes):
        id = start+i
        si = 0
        for neigh, aph, ns in zip(mesh_obj.neighbors[id],mesh_obj.angles[id], mesh_obj.normal_stiffness[id]):
            uij = mesh_obj.u[neigh] - mesh_obj.u[id]
            rij = np.array([np.cos(np.pi*np.array(aph)/180), np.sin(np.pi*np.array(aph)/180)])
            si += (ns!=0)*(1/6)*(uij[:,np.newaxis]@rij[np.newaxis,:] + \
            rij[:,np.newaxis]@uij[np.newaxis,:])

        strain[i,:] = si.reshape(1,4)
    return strain

"""Function to find critically strained bond"""
def critical_strained_bonds(mesh_obj, strain, threshold, ij):
    """
    Identify critical strained bonds within a specified range of nodes.

    Parameters:
    - mesh_obj (mesh.mesh): The mesh object containing node information.
    - strain (numpy.ndarray): Strain tensor data for the entire mesh.
    - threshold (float): Threshold value for critical bond strain.
    - ij (tuple): Tuple indicating the start and end indices of nodes for analysis.

    Returns:
    - critical_bonds (list): List of bond pairs that exceed the specified threshold.

    This function identifies critical strained bonds within a specified range of nodes
    in a mesh. It utilizes the strain tensor information for the entire mesh and iterates
    through the nodes within the specified range ('ij') to compute bond strains. Bonds
    with principal strains surpassing the specified threshold are considered critical,
    and the function returns a list of bond pairs that exceed this threshold.
    """    
    start = ij[0]; end = ij[1]
    batch_nodes = end-start+1
    critical_bonds = []
    for i in range(batch_nodes):
        id = start + i
        for neigh, ns in zip(mesh_obj.neighbors[id], mesh_obj.normal_stiffness[id]):
            if ns:
                bond_strain = 0.5*(strain[id,:] + strain[neigh,:])
                principal_strain = principal_value(bond_strain)

                if principal_strain >= threshold:
                    critical_bonds.append([id, neigh])
    return critical_bonds                



    

