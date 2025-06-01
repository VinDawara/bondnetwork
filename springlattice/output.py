from . import mesh, bcs, solver, crack, analysis
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import h5py
import copy
import matplotlib.pyplot as  plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
from typing import List
from matplotlib.collections import LineCollection
import matplotlib.tri as tri

# """Function to load mesh"""
# def load_mesh_old(dir, objname='meshobj', arg=None):
#     # Load the dictionary file
#     with open(f"{dir}/{objname}", 'rb') as f:
#         dict_obj = pickle.load(f)

#     # Create a mesh_obj using mesh.load_mesh (assuming it's a factory function)
#     mesh_obj = mesh.load_mesh_old(dir, arg) # main class
#     mesh_obj.bcs = bcs.bcs()    # subclass
#     mesh_obj.solver = solver.solver()   # subclass
#     mesh_obj.crack = crack.crack()  # subclass


#     # Assign attributes from the dictionary to the respective sub-classes of mesh_obj
#     if 'bcs.comp' in dict_obj:
#         mesh_obj.bcs.comp = dict_obj['bcs.comp']
#     if 'bcs.ids' in dict_obj:
#         mesh_obj.bcs.ids = dict_obj['bcs.ids']
#     if 'bcs.parser' in dict_obj:
#         mesh_obj.bcs.parser = dict_obj['bcs.parser']

#     if 'lbcs.ids' in dict_obj:
#         mesh_obj.lbcs.ids = dict_obj['lbcs.ids']
#     if 'lbcs.fx' in dict_obj:
#         mesh_obj.lbcs.fx = dict_obj['lbcs.fx']
#     if 'lbcs.fy' in dict_obj:
#         mesh_obj.lbcs.fy = dict_obj['lbcs.fy']
#     if 'lbcs.fun' in dict_obj:
#         mesh_obj.lbcs.fun = dict_obj['lbcs.fun']

#     if 'solver.dt' in dict_obj:
#         mesh_obj.solver.dt = dict_obj['solver.dt']
#     if 'solver.name' in dict_obj:
#         mesh_obj.solver.name = dict_obj['solver.name']
#     if 'solver.skipsteps' in dict_obj:
#         mesh_obj.solver.skipsteps = dict_obj['solver.skipsteps']
#     if 'solver.endtime' in dict_obj:
#         mesh_obj.solver.endtime = dict_obj['solver.endtime']

#     if 'crack.t' in dict_obj:
#         mesh_obj.crack.t = dict_obj['crack.t']
#     if 'crack.ids' in dict_obj:
#         mesh_obj.crack.ids = dict_obj['crack.ids']  
#     if 'crack.param' in dict_obj:
#         mesh_obj.crack.param = dict_obj['crack.param']     

#     return mesh_obj

def load_mesh(dir, objname='meshobj', arg=None):
    # Load the dictionary file
    with open(f"{dir}/{objname}", 'rb') as f:
        dict_obj = pickle.load(f)

    # Create a mesh_obj using mesh.load_mesh (assuming it's a factory function)
    mesh_obj = mesh.load_mesh(dir, arg) #  main class in module "mesh"

    # Define a mapping of attributes between dictionary keys and subclasses' attributes
    attribute_map = {
        'bcs': {'comp': 'bcs.comp', 'ids': 'bcs.ids', 'parser': 'bcs.parser'},
        'lbcs': {'ids': 'lbcs.ids', 'fx': 'lbcs.fx', 'fy': 'lbcs.fy', 'fun': 'lbcs.fun'},
    }

    # Assign attributes from the dictionary to the respective sub-classes of mesh_obj
    for subclass, attributes in attribute_map.items():
        if hasattr(mesh_obj, subclass):
            subclass_obj = getattr(mesh_obj, subclass)
            for attr_name, dict_key in attributes.items():
                if dict_key in dict_obj:
                    setattr(subclass_obj, attr_name, dict_obj[dict_key])
                else:
                    print(f"Missing key '{dict_key}' for {subclass}")
        else:
            print(f"Subclass '{subclass}' doesn't exist in mesh_obj")

    return mesh_obj


"""Function to plot field""" 
def plotfield(x,y,f, title = 'fieldview',cbarlim = [], save = False):
    fig, ax = plt.subplots()
    # set axes properties
    ax.set_xlim([np.min(x)-2, np.max(x)+2])
    ax.set_ylim([np.min(y), np.max(y)])
    ax.set_title(title)
    ax.set_aspect('equal')
    ax_divider= make_axes_locatable(ax)
    # add an axes to the right of the main axes
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    if not cbarlim:
        ul = np.nanmax(f)
        ll = np.nanmin(f)
        cbarlim = [ll, ul]
        
    surf = ax.pcolormesh(x,y,f,cmap='jet', shading='auto', vmin=cbarlim[0], vmax=cbarlim[1])
    surf.set_clim(cbarlim)
    fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],6),3))
    if save:
        plt.close()
        return fig, ax
    else:
        plt.show()  

"""Function to compute the strain fields"""
def computefield(comp, extract_data,start = 0,end = None,step = 1, range = None, cbar = None):
    # load mesh data
    mesh_obj = load_mesh(extract_data)
    mesh_obj.folder = extract_data

    # simulation parameters
    dt = mesh_obj.solver.dt
    skipsteps = mesh_obj.solver.skipsteps   
    # time steps
    if end is None:
        end = mesh_obj.solver.endtime
    if range is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((range/(dt*skipsteps)))
    t = time_steps*skipsteps*dt

    # copy of the initial node positions
    pos = copy.deepcopy(mesh_obj.pos)

    # load displacement data
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')
    
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # create strain directory insider folder
    sub_dir = mesh.sub_directory(f'{comp}', mesh_obj.folder)
    
    # create mesh grid
    xmin, xmax = np.min(pos[:,0])+1, np.max(pos[:,0])-1
    ymin, ymax = np.min(pos[:,1])+0.1, np.max(pos[:,1])-0.5
    nx = 4*mesh_obj.nx
    ny = 4*mesh_obj.ny
    x,y = np.meshgrid(np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax, ny))

    for i,step in enumerate(time_steps):
        # read displacement data in buckets
        if step >= fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            idx = fill_step
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            
        bucket_idx = int(step - idx)   
        
        up = griddata(pos, u[bucket_idx,:], (x, y), method='cubic')
        vp = griddata(pos, v[bucket_idx,:], (x, y), method='cubic')
        
        if comp == 'u':
            f = up
        else:
            f = vp    
        
        fig,ax = plotfield(x+up,y+vp,f, title=f"T = {t[i]}", cbarlim=cbar, save=True)
        # set axes properties
        ax.set_xlim([xmin-2, xmax+2])
        ax.set_ylim([ymin-2, ymax+2])

        fig.savefig(sub_dir+f"{int(step)}", bbox_inches = 'tight', dpi = 300)
        plt.close()
      
    disp_file.close()


"""Function to time history of nodes"""
# Takes 'ids' kwargs as a list argument specified in square bracket
# Takes 'ij' kwargs also as list argument specified as [(x1,y1),..]
def timehistory(comp, extract_data,**kwargs):
    # read the mesh object
    mesh_obj = load_mesh(extract_data)
    mesh_obj.folder = extract_data
    
    # reading keywords
    if kwargs:
        try:
            ids = kwargs['ids']
        except:
            ids = []
            ij = kwargs['ij']
            for i,j in ij:
                ids.append(mesh.return_node_prop(mesh_obj, 'id',(i,j)))

    # load displacement fields
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')
    maxsteps = disp_file['u'][:,0].shape[0]
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # output variables
    t = mesh_obj.solver.dt*mesh_obj.solver.skipsteps*np.arange(0,maxsteps)
    f = np.zeros(shape=(len(ids),maxsteps))

    while fill_step != maxsteps:
        var = disp_file[f'{comp}'][fill_step:fill_step+bucket_size]
        i = 0
        for idx in ids:
            f[i,fill_step:fill_step+bucket_size] = var[:,idx].T
            i += 1

        fill_step += bucket_size
        remain_step = maxsteps - fill_step
        if remain_step < bucket_size:
            bucket_size = remain_step
    disp_file.close()
    return t,f        

"""Function to compute friction force"""
def compute_boundary_force(extract_data, comp, boundary):
    # load mesh object
    mesh_obj = load_mesh(extract_data)
    mesh_obj.folder = extract_data

    # extract bond properties
    neighbors = mesh_obj.neighbors
    norm_stiff = mesh_obj.normal_stiffness
    tang_stiff = mesh_obj.tangential_stiffness
    angles = mesh_obj.angles

    # extract simulation parameters
    dt = mesh_obj.solver.dt
    skipsteps = mesh_obj.solver.skipsteps
    

    # load displacement fields
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')
    
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # output variables    
    t = np.zeros(maxsteps)
    f = np.zeros(maxsteps)

    # set boundary
    if boundary == 'top':
        ids = mesh_obj.top
    elif boundary == 'right':
        ids = mesh_obj.right
    elif boundary == 'bottom':
        ids = mesh_obj.bottom
    elif boundary == 'left':
        ids = mesh_obj.left
    else:
        raise Exception(f'Invalid \'{boundary}\' keyword')            

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0

        t[step] = skipsteps*step*dt
        sum = 0
        for id in ids:
            id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
            for i,neigh in enumerate(neighbors[id]):
                ns = norm_stiff[id][i]
                ts = tang_stiff[id][i]
                aph = angles[id][i]               
                neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
                ro = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])
                rf = mesh_obj.pos[neigh] + neigh_disp - mesh_obj.pos[id] - id_disp
                fb = (ns-ts)*np.dot(rf-ro,ro)*ro + ts*(rf-ro)
                if comp == 'fx':
                    sum += fb[0]
                else:
                    sum += fb[1]    
          
        f[step] = -sum  
        bucket_idx += 1    
    return t,f      


"""Function to interplolate the data along the line drawn inside the domain"""
def line_interpolation(field, line_start, line_end, extract_data , interval = 1, animate = False):
    # read the mesh object
    mesh_obj = load_mesh(extract_data)
    mesh_obj.folder = extract_data  

    # read displacement fields
    if field == 'u':
        data = np.load(mesh_obj.folder+'/u.npy')
    elif field == 'v':    
        data = np.load(mesh_obj.folder+'/v.npy')   
    else:
        raise Exception(f'Unidentified field: {field}')    

    # check number of steps
    try:
        max_steps = np.size(data,1)
    except:
        max_steps = 1
        data = data[:,np.newaxis]

    # extract the coordinates of the nodes
    x = mesh_obj.pos[:,0]
    y = mesh_obj.pos[:,1]
    z = np.zeros(shape=(len(mesh_obj.pos),1))

    # generate line coordinates
    npoints = 300
    lx,ly = create_line(line_start,line_end,npoints)

    # radial distance from start of the line
    r = np.sqrt((lx-lx[0])**2 + (ly-ly[0])**2)

    # time variables
    t = np.arange(0,max_steps*mesh_obj.dt,interval)

    # field variables
    f = np.zeros(shape=(len(t),npoints))

    for i,tt in enumerate(t):
        z = data[:,int(tt/mesh_obj.dt)]   
        interp = LinearNDInterpolator(list(zip(x,y)),z)
        f[i,:] = interp(lx,ly).reshape(1,npoints)

    if animate:
        line_animation(r,f,t, folder=mesh_obj.folder, ylabel=field)
        print(f"Creating movie {field}.mp4...Done")

    return [r,f,t]  


"""Function to create line between two start and end points"""
def create_line(start,end,npoints=2):
    if np.all(start == end):
        raise Exception('Line needs two distinct start and end points')
    else:
        if start[0] == end[0]:  # horizontal line
            y = np.linspace(start[1],end[1],npoints)
            x = start[0]*np.ones(y.shape)
        elif start[1] == end[1]:    # vertical line
            x = np.linspace(start[0],end[0],npoints)
            y = start[1]*np.ones(x.shape)
        else:
            x = np.linspace(start[0],end[0],npoints)
            m = (end[1]-start[1])/(end[0]-start[0])
            y = start[1] + m*(x - start[0])
    return x,y                

"""Function for line animation"""
# This animation is only for line plot where ydata varies with time
def  line_animation(xdata, ydata, time, folder, ylabel = 'y'):
    # round the time float variable
    time = np.round(time,2)

    # create a figure and axes
    fig, ax = plt.subplots()
    # create an empty plot. Note the comma after line variable because it return multiple values
    line, = ax.plot([],[])

    # This function will clear the screen
    def init():
        line.set_data([],[])
        return line,
    
    # initialize the axes
    ax.set_xlabel('r')
    ax.set_ylabel(f'{ylabel}')
    ax.set_xlim([xdata.min(), xdata.max()])
    ax.set_ylim([np.nanmin(ydata), np.nanmax(ydata)])

    # function to update the screen
    def animate(i):
        line.set_data(xdata,ydata[i,:])
        ax.set_title(f'T = {time[i]}')
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(ydata[:,0]), blit = True)

    # saving the animation
    anim.save(folder+f'/{ylabel}.mp4',writer='ffmpeg', fps=10)

"""Function to compute the strain field"""
def plotstrainfield(comp, extract_data, start = 0,end = None,step = 1, range = None, cbar = None):
    pass

"""Function to generate the crack path"""
def generate_crack_path(extract_data: str, till_time=None):
    # Step 1: Load mesh data
    mesh_obj = load_mesh(extract_data)
    mesh_obj.folder = extract_data
    pos = copy.deepcopy(mesh_obj.pos)

    # Step 2: Create Voronoi construction for the triangular grid
    vor = analysis.VoronoiGenerator(pos, mesh_obj.a, (mesh_obj.nx, mesh_obj.ny))

    # Step 3: Read the ids of the broken bonds till the specified time
    broken_ids = []
    for read_data in BrokenBondGenerator(extract_data):
        t, ids = read_data[0], read_data[1]
        if till_time is not None and t > till_time:
            break

        broken_ids.extend(ids)
        
    # Step 4: Get the Voronoi edges corresponding to broken node ids
    edge_map = analysis.preprocess_voronoi_edges(vor)
    x_segs = np.zeros((len(broken_ids), 2))
    y_segs = np.zeros((len(broken_ids), 2))

    for k, (i, j) in enumerate(broken_ids):
        edge_vertices = analysis.find_voronoi_edges_optimized(vor, i, j, edge_map)
        if edge_vertices is not None:
            x_segs[k, :] = edge_vertices[:, 0]
            y_segs[k, :] = edge_vertices[:, 1]
        else:
            print(f"No edges found between nodes {i} and {j}")

    return x_segs, y_segs


"""Generator to read the broken bond data"""
def BrokenBondGenerator(extract_data:str):
    with open(extract_data + '/delbonds', 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                yield data 
            except EOFError:
                break
                   

"""Function to plot displacement fields"""
# def visualize_displacement_field(dir_path, comp, **kwargs):
#     # step 0: extract time
#     time = kwargs.get('time', None)
#     start = kwargs.get('start', None)
#     end = kwargs.get('end', None)
#     step = kwargs.get('step', None)
#     save = kwargs.get('save', True)
    
#     # step 1: load mesh
#     mesh_obj = load_mesh(dir_path)
#     pos = mesh_obj.pos
#     dt = mesh_obj.solver['dt']
#     skipsteps = mesh_obj.solver['skipsteps']

#     # create sub directory to save plots
#     path = mesh._sub_directory('disp', mesh_obj.folder)

#     # create a mesh grid
#     nx, ny = 2*mesh_obj.nx, 2*mesh_obj.ny
#     xo, yo = pos[:,0], pos[:,1]
#     xo_min, xo_max = xo.min(), xo.max()
#     yo_min, yo_max = yo.min(), yo.max()
#     xm, ym = np.meshgrid(np.linspace(xo_min, xo_max, nx), 
#                          np.linspace(yo_min, yo_max, ny))

#     # handler for displacement data file
#     disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

#     # time steps corresponding to the time intervals
#     if time is None:
#         time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
#     else:
#         time_steps = np.rint((time/(dt*skipsteps)))    
#     t = time_steps*skipsteps*dt

#     # preparing the batch reading for displacement data
#     maxsteps = disp_file['u'][:,0].shape[0]
#     bucket = 0
#     fill_step = int(time_steps[0])
#     remain_step = maxsteps - fill_step
#     bucket_size = min(10000, maxsteps)

#     for i,step in enumerate(time_steps):
#         # read displacement data in buckets
#         while True:
#             if step >= fill_step:
#                 u = disp_file['u'][fill_step:fill_step+bucket_size]
#                 v = disp_file['v'][fill_step:fill_step+bucket_size]
#                 bucket += 1
#                 idx = fill_step
#                 fill_step += bucket_size
#                 remain_step = maxsteps - fill_step 
#                 if remain_step < bucket_size:
#                     bucket_size = remain_step
#                 print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
#                       'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
#             bucket_idx = int(step - idx) 
#             if bucket_idx < fill_step-idx:
#                 break   
            
#         if comp == 'u':
#             disp = u[bucket_idx,:]
#         elif comp == 'v':
#             disp = v[bucket_idx,:]
#         elif comp == 'abs':
#             disp = np.sqrt(u[bucket_idx,:]**2 + v[bucket_idx,:]**2)

#         # interpolation in the mesh grid
#         disp_grid = griddata(pos, disp, (xm, ym), method='cubic')

#         # creating figure
#         viewfield(xm, ym, disp_grid, xlim = (xo_min,xo_max), 
#                             ylim = (yo_min, yo_max), save = save, path = path,
#                             filename = f'disp_{step}', title = f'T = {t[i]}')
            
#     disp_file.close()
      
"""Function to visualize the field"""
def viewfield(x:np.ndarray,y:np.ndarray,f:np.ndarray, **kwargs):
    # read optional keywords
    title = kwargs.get('title', 'field')
    cbarlim = kwargs.get('cbarlim', None)
    save = kwargs.get('save', False)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    path = kwargs.get('path', None)
    filename= kwargs.get('filename', 'disp_field')
    mask = kwargs.get('mask', None)
    crackpattern = kwargs.get('crackpattern', None)

    if mask is not None:
        f = mask*f    

    if not xlim:
        xlim = (np.min(x)-2, np.max(x)+2)

    if not ylim:
        ylim = (np.min(y), np.max(y))     

    fig, ax = plt.subplots()
    # set axes properties
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax_divider= make_axes_locatable(ax)
    # add an axes to the right of the main axes
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    if not cbarlim:
        ul = np.nanmax(f)
        ll = np.nanmin(f)
        cbarlim = [ll, ul]
        
    surf = ax.pcolormesh(x,y,f,cmap='jet', shading='auto', vmin=cbarlim[0], vmax=cbarlim[1])
    surf.set_clim(cbarlim)
    fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],6),3))

    if crackpattern is not None:
        crack_segs = LineCollection(crackpattern, linewidths = 2, color = 'white')
        ax.add_collection(crack_segs)

    if save:
        fig.savefig(path + f'/{filename}.png', bbox_inches = 'tight' , dpi = 300)
        plt.close()
    else:
        plt.show()  


"""Function to read the data in batches"""
def batch_data_reader(disp_file, time_steps):
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = min(10000, maxsteps)

    for step in time_steps:
        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file['u'][fill_step:fill_step+bucket_size]
                v = disp_file['v'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step 
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
            bucket_idx = int(step - idx) 
            if bucket_idx < fill_step-idx:
                break  
        yield u,v,bucket_idx 

def mask_hole(xm,ym, hole_param):
    center = hole_param['center']
    radius = hole_param['radius']
    xc = xm - center[0]
    yc = ym - center[1]
    mask = (xc**2 + yc**2) >= radius**2 
    # Assigning 1 to points outside the circle
    mask = np.where(mask, 1, np.nan)
    return mask

def generate_crack_segs(data_path, time):
    x_segs, y_segs = generate_crack_path(data_path, time)

    segs = np.zeros((len(x_segs),2,2))
    for i, (x, y) in enumerate(zip(x_segs, y_segs)):
        segs[i,:,0] = x
        segs[i,:,1] = y

    return segs   



def exclude_nodes(mesh_obj:mesh.mesh):
    # load nodes
    load_node_ids = mesh_obj.lbcs.ids

    mask_ids = []
    for id_list in load_node_ids:
        for node_id in id_list:
            ns = mesh_obj.normal_stiffness[node_id]
            if all(value == 0 for value in ns):
                mask_ids.append(node_id)

    # displacement nodes
    disp_node_ids = mesh_obj.bcs.ids
    for id_list in disp_node_ids:
        for node_id in id_list:
            ns = mesh_obj.normal_stiffness[node_id]
            if all(value == 0 for value in ns):
                mask_ids.append(node_id)

    # # if contains hole
    # if hasattr(mesh_obj, 'circle'):
    #     mask_ids.extend(mesh_obj.circle['circ_bound_nodes'])

    mask_node = np.ones(mesh_obj.nx * mesh_obj.ny, dtype=bool)
    mask_node[mask_ids] = False    

    return mask_node

"""Generator to read the broken bond data"""
def mesh_connectvity_updator(mesh_obj:mesh.mesh, time = None) -> List:
    broken_ids = []
    for read_data in BrokenBondGenerator(mesh_obj.folder):
        t, ids = read_data[0], read_data[1]
        if time is not None and t > time:
            break
        broken_ids.extend(ids)

        for id, neigh in ids:
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)


    mask_ids = []
    for id, neigh in broken_ids:
        ns = mesh_obj.normal_stiffness[id]
        if all(value == 0 for value in ns):
                mask_ids.append(id)

    return mask_ids            


def visualize_displacement_field(dir_path, comp, **kwargs):
    # step 0: extract time
    time = kwargs.get('time', None)
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    step = kwargs.get('step', None)
    save = kwargs.get('save', True)
    cbarlim = kwargs.get('cbarlim', None)
    
    # step 1: load mesh
    mesh_obj = load_mesh(dir_path)
    pos = mesh_obj.pos
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # create sub directory to save plots
    path = mesh._sub_directory('disp', mesh_obj.folder)

    # create a mesh grid
    nx, ny = 2*mesh_obj.nx, 2*mesh_obj.ny
    xo, yo = pos[:,0], pos[:,1]
    xo_min, xo_max = xo.min(), xo.max()
    yo_min, yo_max = yo.min(), yo.max()
    xm, ym = np.meshgrid(np.linspace(xo_min, xo_max, nx), 
                         np.linspace(yo_min, yo_max, ny))

    # create mask
    mask = mask_hole(xm,ym, mesh_obj.circle) if hasattr(mesh_obj, 'circle') else None
        
    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # time steps corresponding to the time intervals
    if time is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((time/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # reading data
    data_generator = batch_data_reader(disp_file, time_steps)

    for i, (u,v,bucket_idx) in enumerate(data_generator):       
        if comp == 'u':
            disp = u[bucket_idx,:]
        elif comp == 'v':
            disp = v[bucket_idx,:]
        elif comp == 'abs':
            disp = np.sqrt(u[bucket_idx,:]**2 + v[bucket_idx,:]**2)

        # mesh connectivity updator
        mesh_connectvity_updator(mesh_obj, t[i]) 

        # masked nodes
        mask_nodes = exclude_nodes(mesh_obj)

        # interpolation in the mesh grid
        disp_grid = griddata(pos[mask_nodes], disp[mask_nodes], (xm, ym), method='cubic')

        # generate crack segments
        crack_segs = generate_crack_segs(mesh_obj.folder, t[i])

        # creating figure
        viewfield(xm, ym, disp_grid, xlim = (xo_min,xo_max), 
                            ylim = (yo_min, yo_max), save = save, path = path,
                            filename = f'disp_{time_steps[i]}', title = f'T = {t[i]}',
                            mask = mask, crackpattern = crack_segs, cbarlim = cbarlim)  
    
    disp_file.close()



def trifield(x:np.ndarray,y:np.ndarray,f:np.ndarray, **kwargs):
    # read optional keywords
    title = kwargs.get('title', 'field')
    cbarlim = kwargs.get('cbarlim', None)
    save = kwargs.get('save', False)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    path = kwargs.get('path', None)
    filename= kwargs.get('filename', 'disp_field')
    mask = kwargs.get('mask', None)
    crackpattern = kwargs.get('crackpattern', None)

    triang = tri.Triangulation(x,y)

    if mask is not None:
        f[mask] = np.nan    

    if not xlim:
        xlim = (np.min(x)-2, np.max(x)+2)

    if not ylim:
        ylim = (np.min(y), np.max(y))     

    fig, ax = plt.subplots()
    # set axes properties
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax_divider= make_axes_locatable(ax)
    # add an axes to the right of the main axes
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    if not cbarlim:
        ul = np.nanmax(f)
        ll = np.nanmin(f)
        cbarlim = [ll, ul]
        
    surf = ax.tripcolor(x,y,f,cmap='jet', vmin=cbarlim[0], vmax=cbarlim[1])
    surf.set_clim(cbarlim)
    fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],6),3))

    if crackpattern is not None:
        crack_segs = LineCollection(crackpattern, linewidths = 2, color = 'white')
        ax.add_collection(crack_segs)

    if save:
        fig.savefig(path + f'/{filename}.png', bbox_inches = 'tight' , dpi = 300)
        plt.close()
    else:
        plt.show()  

def triplot_displacement_field(dir_path, comp, **kwargs):
    # step 0: extract time
    time = kwargs.get('time', None)
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    step = kwargs.get('step', None)
    save = kwargs.get('save', True)
    cbarlim = kwargs.get('cbarlim', None)
    
    # step 1: load mesh
    mesh_obj = load_mesh(dir_path)
    pos = mesh_obj.pos
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # create sub directory to save plots
    path = mesh._sub_directory('disp', mesh_obj.folder)

    # create a mesh grid
    xo, yo = pos[:,0], pos[:,1]
    xo_min, xo_max = xo.min(), xo.max()
    yo_min, yo_max = yo.min(), yo.max()


    # mask hole ids
    mask_hole_ids = nodes_inside_hole(mesh_obj, pos) if hasattr(mesh_obj, 'circle') else []
        
    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # time steps corresponding to the time intervals
    if time is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((time/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # reading data
    data_generator = batch_data_reader(disp_file, time_steps)

    for i, (u,v,bucket_idx) in enumerate(data_generator):       
        if comp == 'u':
            disp = u[bucket_idx,:]
        elif comp == 'v':
            disp = v[bucket_idx,:]
        elif comp == 'abs':
            disp = np.sqrt(u[bucket_idx,:]**2 + v[bucket_idx,:]**2)

        # mesh connectivity updator
        mask_nodes = mesh_connectvity_updator(mesh_obj, t[i]) 

        # masked nodes
        mask_nodes = np.concatenate((mask_nodes_triplot(mesh_obj), mask_hole_ids)).astype(int)
        # generate crack segments
        crack_segs = generate_crack_segs(mesh_obj.folder, t[i])

        # creating figure
        trifield(xo,yo, disp, xlim = (xo_min,xo_max), 
                            ylim = (yo_min, yo_max), save = save, path = path,
                            filename = f'disp_{time_steps[i]}', title = f'T = {t[i]}',
                            mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim)  
    
    disp_file.close()

"""Function to find hole nodes"""
def nodes_inside_hole(mesh_obj: mesh.mesh, pos: np.ndarray):
    center = mesh_obj.circle['center']
    radius = mesh_obj.circle['radius']

    pos_from_center = pos - center
    distances_sq = np.sum(pos_from_center ** 2, axis=1)
    ids = np.where(distances_sq <= radius**2)[0]

    return ids

def mask_nodes_triplot(mesh_obj):
    # load nodes
    load_node_ids = mesh_obj.lbcs.ids

    mask_ids = []
    for id_list in load_node_ids:
        for node_id in id_list:
            ns = mesh_obj.normal_stiffness[node_id]
            if all(value == 0 for value in ns):
                mask_ids.append(node_id)

    # displacement nodes
    disp_node_ids = mesh_obj.bcs.ids
    for id_list in disp_node_ids:
        for node_id in id_list:
            ns = mesh_obj.normal_stiffness[node_id]
            if all(value == 0 for value in ns):
                mask_ids.append(node_id)    

    return mask_ids

"""Function to visualize the strain field"""
def triplot_strain_field(dir_path, comp, **kwargs):
    # step 0: extract time
    time = kwargs.get('time', None)
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    step = kwargs.get('step', None)
    save = kwargs.get('save', True)
    cbarlim = kwargs.get('cbarlim', None)
    
    # step 1: load mesh
    mesh_obj = load_mesh(dir_path)
    pos = mesh_obj.pos
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # create sub directory to save plots
    path = mesh._sub_directory(f'strain_{comp}', mesh_obj.folder)

    # create a mesh grid
    xo, yo = pos[:,0], pos[:,1]
    xo_min, xo_max = xo.min(), xo.max()
    yo_min, yo_max = yo.min(), yo.max()


    # mask hole ids
    mask_hole_ids = nodes_inside_hole(mesh_obj, pos) if hasattr(mesh_obj, 'circle') else []
        
    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # time steps corresponding to the time intervals
    if time is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((time/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # reading data
    data_generator = batch_data_reader(disp_file, time_steps)

    for i, (u,v,bucket_idx) in enumerate(data_generator):   
        # mesh connectivity updator
        # mask_nodes = mesh_connectvity_updator(mesh_obj, t[i])
        # masked nodes
        mask_nodes = np.concatenate((mesh_connectvity_updator(mesh_obj, t[i]), mask_hole_ids)).astype(int)
            
        # compute strain tensor
        mesh_obj.u[:,0] = u[bucket_idx,:]
        mesh_obj.u[:,1] = v[bucket_idx,:]

        strain = crack.compute_nodal_strain_tensor(mesh_obj)

        if comp == 'xx':
            field = strain[:,0]
        elif  comp == 'xy':
            field = strain[:,1]
        elif comp == 'yy':
            field = strain[:,3]
        elif comp == 'max':
            field = compute_evalues(strain)             

        # generate crack segments
        crack_segs = generate_crack_segs(mesh_obj.folder, t[i])

        # creating figure
        trifield(xo,yo, field, xlim = (xo_min,xo_max), 
                            ylim = (yo_min, yo_max), save = save, path = path,
                            filename = f'disp_{time_steps[i]}', title = f'T = {t[i]}',
                            mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim)  
    
    disp_file.close()

"""Function to compute principal strain"""
def compute_evalues(strain):
    return np.array([crack.principal_value(s) for s in strain])


"""Function to visualize the strain field"""
def triplot_stress_field(dir_path, comp, **kwargs):
    # step 0: extract time
    time = kwargs.get('time', None)
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    step = kwargs.get('step', None)
    save = kwargs.get('save', True)
    cbarlim = kwargs.get('cbarlim', None)
    
    # step 1: load mesh
    mesh_obj = load_mesh(dir_path)
    pos = mesh_obj.pos
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # create sub directory to save plots
    path = mesh._sub_directory(f'stress_{comp}', mesh_obj.folder)

    # create a mesh grid
    xo, yo = pos[:,0], pos[:,1]
    xo_min, xo_max = xo.min(), xo.max()
    yo_min, yo_max = yo.min(), yo.max()


    # mask hole ids
    mask_hole_ids = nodes_inside_hole(mesh_obj, pos) if hasattr(mesh_obj, 'circle') else []
        
    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # time steps corresponding to the time intervals
    if time is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((time/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # reading data
    data_generator = batch_data_reader(disp_file, time_steps)

    for i, (u,v,bucket_idx) in enumerate(data_generator):   

        # updating mesh connectivity and extracting nodes with zero bond connectivity
        mask_nodes = np.concatenate((mesh_connectvity_updator(mesh_obj, t[i]), mask_hole_ids)).astype(int)
            
        # compute strain tensor
        mesh_obj.u[:,0] = u[bucket_idx,:]
        mesh_obj.u[:,1] = v[bucket_idx,:]
        stress = mesh.compute_nodal_stress_tensor(mesh_obj)

        if comp == 'xx':
            field = stress[:,0]
        elif  comp == 'xy':
            field = stress[:,1]
        elif comp == 'yy':
            field = stress[:,3]
        elif comp == 'max':
            field = compute_evalues(stress)             

        # generate crack segments
        crack_segs = generate_crack_segs(mesh_obj.folder, t[i])

        # creating figure
        trifield(xo,yo, field, xlim = (xo_min,xo_max), 
                            ylim = (yo_min, yo_max), save = save, path = path,
                            filename = f'disp_{time_steps[i]}', title = f'T = {t[i]}',
                            mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim)  
    
    disp_file.close()



