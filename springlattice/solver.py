from . import mesh, bcs
import numpy as np
import math
import copy
import h5py
from scipy.sparse import csc_matrix


"""Generate A matrix for verlet computation"""
def generate_matrix(mesh_obj):
    """
    Generate a sparse matrix (A) for Verlet computation based on a mesh object.
    
    Args:
    - mesh_obj: Mesh object containing attributes like neighbors, angles,
                normal_stiffness, tangential_stiffness, and pos.
    
    Returns:
    - A: Sparse matrix representing the coefficients for Verlet computation.
    """  
    # create the copy of the mesh object
    neighbors = mesh_obj.neighbors
    angles = mesh_obj.angles
    normal_stiff = mesh_obj.normal_stiffness
    tang_stiff = mesh_obj.tangential_stiffness

    # declare variables to store A and b
    total_nodes = len(mesh_obj.pos)
    I,J, value = ([] for i in range(3))

    # variables suffix corresponding to mesh id 'i'
    u = lambda i: 2*i
    v = lambda i: 2*i + 1
    
    # coefficeints u_i and v_i
    coeff_ux = lambda ns,ts,aph: np.array(ns)*np.cos(np.pi*np.array(aph)/180)**2 + np.array(ts)*np.sin(np.pi*np.array(aph)/180)**2
    coeff_uv = lambda ns,ts,aph: (np.array(ns)-np.array(ts))*np.multiply(np.sin(np.pi*np.array(aph)/180),np.cos(np.pi*np.array(aph)/180))
    coeff_vy = lambda ns, ts, aph: np.array(ns)*np.sin(np.pi*np.array(aph)/180)**2 + np.array(ts)*np.cos(np.pi*np.array(aph)/180)**2

    for id in range(total_nodes):
        # coefficient of u_i in X equilibrium equation
        coeff_ux_val = np.sum(coeff_ux(normal_stiff[id], tang_stiff[id], angles[id]))
        # coefficient of v_i and u_i in X and Y equilibrium equation
        coeff_uv_val = np.sum(coeff_uv(normal_stiff[id], tang_stiff[id], angles[id]))
        # coefficient of v_i in Y equilibrium equation
        coeff_vy_val = np.sum(coeff_vy(normal_stiff[id], tang_stiff[id], angles[id]))

        I.extend([u(id), u(id), v(id), v(id)])
        J.extend([u(id), v(id), u(id), v(id)])
        value.extend([coeff_ux_val, coeff_uv_val, coeff_uv_val, coeff_vy_val])

        # coefficient of u_j and v_j in X and Y force equation
        for neigh, aph, ns, ts in zip(neighbors[id],angles[id],normal_stiff[id],tang_stiff[id]):
            # coefficient of u_j, v_j in X equation and v_j, u_j in Y equation
            I.extend([u(id), u(id), v(id), v(id)])
            J.extend([u(neigh), v(neigh), u(neigh), v(neigh)])
            value.extend([-coeff_ux(ns,ts,aph), -coeff_uv(ns,ts,aph), -coeff_uv(ns,ts,aph), -coeff_vy(ns,ts,aph)])

    value = np.array(value)
    I = np.array(I)
    J = np.array(J)
    
    # Creating a sparse matrix
    A = csc_matrix((value, (I,J)), shape=(2*total_nodes,2*total_nodes))

    return A


"""Function to update A based on broken bond list for verlet integrator"""
def update_A(mesh_obj, A, bondlist):
    """
    Update the sparse matrix A based on a list of broken bonds for the Verlet integrator.
    
    Args:
    - mesh_obj: Mesh object containing attributes like neighbors, angles,
                normal_stiffness, tangential_stiffness.
    - A: Sparse matrix representing coefficients for Verlet computation.
    - bondlist: List of broken bond pairs (id, neigh).
    
    Returns:
    - Updated sparse matrix A after modification.
    """ 
    # create the copy of the node object
    neighbors = mesh_obj.neighbors
    angles = mesh_obj.angles
    normal_stiff = mesh_obj.normal_stiffness
    tang_stiff = mesh_obj.tangential_stiffness

    # declare variables to store A and b
    I,J, value = ([] for i in range(3))

    # variables suffix corresponding to node id 'i'
    u = lambda i: 2*i
    v = lambda i: 2*i + 1
    # coefficeints u_i and v_i
    coeff_ux = lambda ns,ts,aph: np.round(np.array(ns)*np.cos(np.pi*np.array(aph)/180)**2 + np.array(ts)*np.sin(np.pi*np.array(aph)/180)**2, decimals = 12)
    coeff_uv = lambda ns,ts,aph: np.round((np.array(ns)-np.array(ts))*np.multiply(np.sin(np.pi*np.array(aph)/180),np.cos(np.pi*np.array(aph)/180)), decimals = 12)
    coeff_vy = lambda ns, ts, aph: np.round(np.array(ns)*np.sin(np.pi*np.array(aph)/180)**2 + np.array(ts)*np.cos(np.pi*np.array(aph)/180)**2, decimals = 12)

    for id, neigh in bondlist:  
        # coefficient of u_i in X equilibrium equation
        coeff_ux_val = np.sum(coeff_ux(normal_stiff[id], tang_stiff[id], angles[id]))
        # coefficient of v_i and u_i in X and Y equilibrium equation
        coeff_uv_val = np.sum(coeff_uv(normal_stiff[id], tang_stiff[id], angles[id]))
        # coefficient of v_i in Y equilibrium equation
        coeff_vy_val = np.sum(coeff_vy(normal_stiff[id], tang_stiff[id], angles[id]))

        I.extend([u(id), u(id), v(id), v(id)])
        J.extend([u(id), v(id), u(id), v(id)])
        value.extend([coeff_ux_val, coeff_uv_val, coeff_uv_val, coeff_vy_val])

        # index of neighbor in id neighbor list
        idx = neighbors[id].index(neigh)
        ns = normal_stiff[id][idx]
        ts = tang_stiff[id][idx]
        aph = angles[id][idx]

        # coefficient of u_j, v_j in X equation and v_j, u_j in Y equation
        I.extend([u(id), u(id), v(id), v(id)])
        J.extend([u(neigh), v(neigh), u(neigh), v(neigh)])
        value.extend([-coeff_ux(ns,ts,aph), -coeff_uv(ns,ts,aph), -coeff_uv(ns,ts,aph), -coeff_vy(ns,ts,aph)])

    value = np.array(value)
    I = np.array(I)
    J = np.array(J)
    # update matrix A
    A[I,J] = value

    return A

"""reshape displacement to array(total_nodes,2) to make it callable with other functions""" 
def reshape2vector(u):
    """
    Reshape a vector 'u' into a 2D array representing node positions (u, v) for each node 'id'.
    
    Args:
    - u: Input vector representing node positions with alternating u and v coordinates.

    Returns:
    - u_reshape: 2D array with shape (n, 2) representing reshaped node positions,
                 where 'n' is the number of nodes.
    """
    # variables suffix corresponding to node id 'i'
    u_idx = lambda i: 2*i
    v_idx = lambda i: 2*i + 1
    n = int(0.5*len(u))
    u_reshape = np.zeros(shape = (n,2))
    for id in range(n):
        u_reshape[id,0] = u[u_idx(id)]
        u_reshape[id,1] = u[v_idx(id)]
    return u_reshape  

"""reshape displacement from array(total_nodes,2) to array(2*total_nodes,1)""" 
def flatten(u):
    """
    Flatten a 2D array representing node positions into a vector 'u'.

    This function reshapes a 2D array containing node positions (u, v) into a
    flattened vector, where the nodes' u and v coordinates are alternately
    positioned in the resulting vector.

    Args:
    - u: Input 2D array with shape (n, 2) representing node positions.

    Returns:
    - u_reshape: Flattened vector of node positions, alternating u and v coordinates.
    """
    # variables suffix corresponding to node id 'i'
    u_idx = lambda i: 2*i
    v_idx = lambda i: 2*i + 1
    n = len(u)
    u_reshape = np.zeros(shape = (2*n,1))
    for id in range(n):
        u_reshape[u_idx(id)] = u[id,0]
        u_reshape[v_idx(id)] = u[id,1]
    return u_reshape 


"""Updated verlet integrator using matrix multiplication"""
def solve(mesh_obj, dt, endtime, **kwargs):
    """
    Perform Verlet integration on a given mesh object.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - dt (float): The time step used in the integration process.
    - endtime (float): The end time for the integration.
    - zeta (float, optional): The damping coefficient (default is 0).
    - vectorfield (str, optional): Indicates the vector field visualization status ('off' by default).
    - folder (str, optional): The folder name to save integration-related data (default is "sl_nx_ny").
    - **kwargs: Additional keyword arguments:
        - interval (float, optional): The time interval for mesh visulaization during integration (default is False).
        - save_ratio (float, optional): The ratio of steps to save data (default is 5).

    Returns:
    - None

    This function integrates a mesh object using the Verlet method, simulating the physical system
    over a given time period (from 0 to 'endtime') with the provided time step 'dt'.
    It handles boundary conditions, performs time integration, and saves data during the process.
    Vector field visualization can be toggled with 'vectorfield', and data can be saved in the specified 'folder'.
    """    
    # set keyward options
    zeta = kwargs.get('zeta', 0)
    vectorfield = kwargs.get('vectorfield', 'off')
    folder = kwargs.get('folder', "sl")
    interval = kwargs.get('interval', False)
    save_ratio = kwargs.get('save_ratio', 0)   

    # creating directory for storing data
    mesh_obj.folder = mesh._create_directory(f"{folder}_{mesh_obj.ny}X{mesh_obj.nx}")
    
    # updating the solver values
    mesh_obj.solver.update({'dt': dt, 'endtime': endtime, 'name': 'verlet_integrator',
                            'zeta': zeta})

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
    A = generate_matrix(mesh_obj)

    # displacement variables for time step 't-1', 't' and 't+1', respectively
    u_prev, u_curr, u_next = (flatten(mesh_obj.u) for i in range(3))

    # impose boundary condition at t = 0
    u_curr = bcs.impose_displacement(u_curr,ids=bcs_ids, comp=bcs_comp, parser=bcs_parser,
                                fun=bcs_fun, t=0)
    u_prev = copy.deepcopy(u_curr)

    # writing contact-time data to file
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

 
        # update node object
        if interval and step%int(interval/dt) == 0: 
            mesh_obj.u = reshape2vector(u_next)
            mesh_obj.pos = pos + mesh_obj.u
            mesh.mesh_plot(mesh_obj,filename = f"step_{step}.png", title = f'T = {np.round(t,3)}', vectorfield = vectorfield, save=True)
            
        
        u_prev = copy.deepcopy(u_curr)
        u_curr = copy.deepcopy(u_next)
        print('Time step = ',step, 'T = %0.4f' % t, 'Progress = %0.2f' % (100*step/maxsteps))

        if step%skipsteps == 0:
            # saving displacements fields
            u_shape = reshape2vector(u_next)
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
    print(f'Solver completed. Data saved to {mesh_obj.folder}')


"""Updated verlet integrator using matrix multiplication"""
def static_solve(mesh_obj, dt, endtime, **kwargs):
    """
    Perform Verlet integration on a given mesh object.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - dt (float): The time step used in the integration process.
    - endtime (float): The end time for the integration.
    - zeta (float, optional): The damping coefficient (default is 0).
    - vectorfield (str, optional): Indicates the vector field visualization status ('off' by default).
    - folder (str, optional): The folder name to save integration-related data (default is "sl_nx_ny").
    - **kwargs: Additional keyword arguments:
        - interval (float, optional): The time interval for mesh visulaization during integration (default is False).
        - save_ratio (float, optional): The ratio of steps to save data (default is 5).

    Returns:
    - None

    This function integrates a mesh object using the Verlet method, simulating the physical system
    over a given time period (from 0 to 'endtime') with the provided time step 'dt'.
    It handles boundary conditions, performs time integration, and saves data during the process.
    Vector field visualization can be toggled with 'vectorfield', and data can be saved in the specified 'folder'.
    """    
    # set keyward options
    zeta = kwargs.get('zeta', 0.1)
    vectorfield = kwargs.get('vectorfield', 'off')
    folder = kwargs.get('folder', "sl")
    interval = kwargs.get('interval', False)
    

    # creating directory for storing data
    mesh_obj.folder = mesh._create_directory(f"{folder}_{mesh_obj.ny}X{mesh_obj.nx}")
    
    # updating the solver values
    mesh_obj.solver.update({'dt': dt, 'endtime': endtime, 'name': 'static',
                            'zeta': zeta})

    maxsteps = math.ceil(endtime/dt)
    
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
    A = generate_matrix(mesh_obj)

    # displacement variables for time step 't-1', 't' and 't+1', respectively
    u_prev, u_curr, u_next = (flatten(mesh_obj.u) for _ in range(3))

    # impose boundary condition at t = 0
    u_curr = bcs.impose_displacement(u_curr,ids=bcs_ids, comp=bcs_comp, parser=bcs_parser,
                                fun=bcs_fun, t=0)
    u_prev = copy.deepcopy(u_curr)

    # writing contact-time data to file
    total_nodes = len(pos)

    # time step for convergence check
    conv_time = endtime/10

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

 
        # update node object
        if interval and step%int(interval/dt) == 0: 
            mesh_obj.u = reshape2vector(u_next)
            mesh_obj.pos = pos + mesh_obj.u
            mesh.mesh_plot(mesh_obj,filename = f"step_{step}.png", title = f'T = {np.round(t,3)}', vectorfield = vectorfield, save=True)
            
        # next step
        if t >= conv_time:
            if np.sum(np.abs(u_next - u_curr)) <= 1e-9:
                print('Solution converged at time T = %0.4f' % t)
                break
            conv_time += endtime/10

        u_prev = copy.deepcopy(u_curr)
        u_curr = copy.deepcopy(u_next)
        print('Time step = ',step, 'T = %0.4f' % t, 'Progress = %0.2f' % (100*step/maxsteps))

    # save initial mesh
    mesh_obj.u = reshape2vector(u_next)
    mesh_obj.pos = pos
    mesh.save_mesh(mesh_obj, 'static')    
    print(f'Solver completed. Data saved to {mesh_obj.folder}')


    



