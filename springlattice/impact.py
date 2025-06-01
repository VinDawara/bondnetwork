from . import mesh, bcs, solver, crack
import numpy as np
import pickle
import h5py
import copy
import math
import multiprocessing as mp

"""Defining impact parameters"""
mesh.mesh.impact_param = {}

"""Function to create dictionary object for a mesh class object"""
def create_dict_obj(mesh_obj):
    
    dict_obj = crack.create_dict_obj(mesh_obj)
    dict_obj['impact_param'] = mesh_obj.crack.t 
    return dict_obj

"""Function to save the mesh object""" 
def save_mesh(mesh_obj, arg = None):
    # create a dictionary object
    dict_obj = create_dict_obj(mesh_obj)
    
    if arg is None:
        filename = 'meshobj'
    dir = mesh_obj.folder + f'/{filename}'
    try:
        with open(dir, 'wb') as f:
            pickle.dump(dict_obj,f)
    except: 
        raise AttributeError(f"mesh cannot be saved in specified directory {dir}") 
    
"""Function to load mesh object"""   
def load_mesh(dir,objname = None):
    # load the dictionary file
    if objname is None:
        objname = 'meshobj'

    with open(dir + f'/{objname}', 'rb') as f:
        dict_obj = pickle.load(f)

    # first load the mesh object
    meshobj = crack.load_mesh(dir, objname)

    # add impact dictionary object to dictionay object
    meshobj.impact = dict_obj['impact_param']

"""Function to read impact paramters"""
def impact_param(mesh_obj:mesh.mesh, ids:list, velocity:float, mass = 1, orient = 'vertical'):
    
    mesh_obj.impact_param = {'ids':ids, 'vel':velocity, 'm':mass, 'orient':orient}


"""Function to specify the impact boundary condition"""
def impactBC(disp, force, param:dict, dt):
    ids = np.array(param['ids'])
    vel = param['vel']
    m = param['m']
    orient = param['orient']

    # function attribute that retains value at each call
    if not hasattr(impactBC,'vprev'):
        impactBC.uprev = 0
        impactBC.vprev = vel
        impactBC.sign = np.copysign(1,vel)

    # variables suffix corresponding to node id 'i'
    u = lambda i: 2*i
    v = lambda i: 2*i + 1

    if np.copysign(1,impactBC.vprev) == impactBC.sign:      
        if orient == 'horizontal':
            ft = np.sum(force[u(ids)])
            impactBC.vprev = impactBC.vprev - (ft/m)*dt
            impactBC.uprev = impactBC.uprev + impactBC.vprev*dt
            disp[u(ids)] = impactBC.uprev
        else:
            ft = np.sum(force[v(ids)])
            impactBC.vprev = impactBC.vprev - (ft/m)*dt
            impactBC.uprev = impactBC.uprev + impactBC.vprev*dt
            disp[v(ids)] = impactBC.uprev
                              
    return disp 


"""Solver for impact loading"""
def solve(mesh_obj, dt, endtime, c = 0,  vectorfield = 'off', folder = None, **kwargs):
    # analyse the script completeness for mesh
    analyse_mesh(mesh_obj)
    # set keyward options
    if kwargs:
        try:
            interval = kwargs['interval']
        except:
            interval = False

        try:
            save_ratio = kwargs['save_ratio']
        except:
            save_ratio = 0    

    # creating directory for storing data
    if not mesh_obj.folder:
        mesh_obj.folder = mesh.create_directory(f"{folder}_{mesh_obj.ny}X{mesh_obj.nx}")
    mesh_obj.solver.dt = dt
    mesh_obj.solver.endtime = endtime
    mesh_obj.solver.name = 'verlet'
    maxsteps = math.ceil(endtime/dt)

    if save_ratio == 0:
        skipsteps = 1
    else:    
        skipsteps = math.ceil((save_ratio/100)*maxsteps)

    mesh_obj.solver.skipsteps = skipsteps
    # save initial mesh
    save_mesh(mesh_obj)

    # initial mesh position
    pos = copy.deepcopy(mesh_obj.pos)

    # extract displacement boundary conditions
    bcs_ids = copy.copy(mesh_obj.bcs.ids)
    bcs_parser = copy.copy(mesh_obj.bcs.parser)
    bcs_comp = copy.copy(mesh_obj.bcs.comp)
    bcs_fun = copy.copy(mesh_obj.bcs.fun)
    if hasattr(mesh_obj,'circle'):
        norm_vec = mesh_obj.circle['norm_vec']
    else:
        norm_vec = None 

    # extract load boundary conditions
    lbcs_ids = copy.copy(mesh_obj.lbcs.ids)
    lbcs_fx = copy.copy(mesh_obj.lbcs.fx)
    lbcs_fy = copy.copy(mesh_obj.lbcs.fy)
    lbcs_fun = copy.copy(mesh_obj.lbcs.fun)

    # extract impact boundary conditions
    impact_par = mesh_obj.impact_param

    # create a matrix A
    A = solver.generate_matrix(mesh_obj)

    # displacement variables for time step 't-1', 't' and 't+1', respectively
    u_prev, u_curr, u_next = (solver.flatten(mesh_obj.u) for i in range(3))

    # impose boundary condition at t = 0
    u_curr = bcs.dispbcs(u_curr,ids=bcs_ids, comp=bcs_comp, parser=bcs_parser,
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

    # file handler for saving deleted bonds
    bonds_file = open(mesh_obj.folder + '/delbonds','wb')
    # displacement bucket to write to file
    bucket = 0
    fill_steps = 0
    bucket_size = 1000
    bucket_idx = 0
    remain_steps = int(maxsteps/skipsteps)
    if remain_steps < bucket_size:
        bucket_size = remain_steps
    U = np.zeros(shape=(bucket_size, total_nodes))
    V = np.zeros(shape=(bucket_size, total_nodes))

    # integration begins
    for step in range(maxsteps):
        # time variable
        t = step*dt

        # velocity of the nodes
        v = (u_next - u_prev)/(2*dt)

        # impose load boundary condtions
        load = bcs.loadbcs(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t,
                               norm_vec=norm_vec)

        # time integration
        u_next= 2*u_curr - u_prev - (A@u_curr - load + c*v)*dt**2

        # impose boundary conditions
        u_next = bcs.dispbcs(u_next, ids=bcs_ids, comp=bcs_comp, parser=bcs_parser,
                                fun=bcs_fun, t=t)
        
        # impose impact boundary conditions
        u_next = impactBC(u_next, force=A@u_curr, param=impact_par, dt=dt)

        mesh_obj.u = solver.reshape2vector(u_next)
        deleted_bonds = crack.activatebreaking(mesh_obj)
        if deleted_bonds:
            A = solver.update_A(mesh_obj,A, deleted_bonds)
            # deleted bonds
            pickle.dump([t,deleted_bonds], bonds_file)

        # update node object
        if interval and step%int(interval/dt) == 0: 
            mesh_obj.pos = pos + mesh_obj.u
            mesh.meshplot(mesh_obj,filename = f"step_{step}.png", title = f'T = {np.round(t,3)}', vectorfield = vectorfield, save=True)
            
        
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

"""Function to analyse the mesh_obj object from the user-scripts"""
def analyse_mesh(mesh_obj):
    if not hasattr(mesh_obj,'bcs'):
        raise Exception('Boundary conditions not specified')
    
    if hasattr(mesh_obj, 'crack'):
        if not mesh_obj.crack.param:
            raise Exception('Breaking criterion required')
        



    

