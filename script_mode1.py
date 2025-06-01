import springlattice as sl
import numpy as np
# create mesh
mesh = sl.MeshGenerator(500,350,1)  

# assign bond properties
sl.BondStiffness(mesh, poisson_ratio = 0.2, nondim = True)

# create edge crack
sl.EdgeCrack(mesh, crack_length=100, row = 175) 

# boundary conditions
sl.LoadParserFunction(mesh, mesh.bottom, fy = '-0.5', fun = 'ramp')
sl.LoadParserFunction(mesh, mesh.top, fy = '0.5', fun = 'ramp')

# brealomg criterion
sl.crack.breaking_parameters(mesh, prop = 'strain', threshold=0.02)

# solve the system
sl.crackmp.solve(mesh, dt=0.05, endtime = 1000, zeta = 0, 
          vectorfield = 'off', folder = 'DEC_MI', interval = 10)