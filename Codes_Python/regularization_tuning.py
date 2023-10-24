import numpy as np
import scipy as sp
import KTCFwd
import KTCMeshing
import KTCRegularization
import KTCScoring
import KTCAux
import matplotlib.pyplot as plt
import glob

def load_data(file_name):
    mat_dict2 = sp.io.loadmat(file_name)
    Inj = mat_dict2["Inj"]
    Uel = mat_dict2["Uel"]
    Mpat = mat_dict2["Mpat"]
    deltaU = Uel - Uelref
    return (Inj, Uel, Mpat, deltaU)

def segment(deltareco_pixgrid):
    level, x = KTCScoring.Otsu2(deltareco_pixgrid.flatten(), 256, 7)

    deltareco_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)

    ind0 = deltareco_pixgrid < x[level[0]]
    ind1 = np.logical_and(deltareco_pixgrid >= x[level[0]],deltareco_pixgrid <= x[level[1]])
    ind2 = deltareco_pixgrid > x[level[1]]
    inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
    bgclass = inds.index(max(inds)) #background class

    if bgclass == 0:
        deltareco_pixgrid_segmented[ind1] = 2
        deltareco_pixgrid_segmented[ind2] = 2
    elif bgclass == 1:
        deltareco_pixgrid_segmented[ind0] = 1
        deltareco_pixgrid_segmented[ind2] = 2
    elif bgclass == 2:
        deltareco_pixgrid_segmented[ind0] = 1
        deltareco_pixgrid_segmented[ind1] = 1
        
    return deltareco_pixgrid_segmented

#%% Setup mesh
inputFolder = "TrainingData"
categoryNbr = 7

Nel = 32  # number of electrodes
z = (1e-6) * np.ones((Nel, 1))  # contact impedances
mat_dict = sp.io.loadmat(inputFolder + '/ref.mat') #load the reference data
Injref = mat_dict["Injref"] #current injections
Uelref = mat_dict["Uelref"] #measured voltages from water chamber
Mpat = mat_dict["Mpat"] #voltage measurement pattern
vincl = np.ones(((Nel - 1),76), dtype=bool) #which measurements to include in the inversion
rmind = np.arange(0,2 * (categoryNbr - 1),1) #electrodes whose data is removed

#remove measurements according to the difficulty level
for ii in range(0,75):
   for jj in rmind:
       if Injref[jj,ii]:
               vincl[:,ii] = 0
       vincl[jj,:] = 0

# load premade finite element mesh (made using Gmsh, exported to Matlab and saved into a .mat file)
mat_dict_mesh = sp.io.loadmat('Mesh_sparse.mat')
g = mat_dict_mesh['g'] #node coordinates
H = mat_dict_mesh['H'] #indices of nodes making up the triangular elements
elfaces = mat_dict_mesh['elfaces'][0].tolist() #indices of nodes making up the boundary electrodes

#Element structure
ElementT = mat_dict_mesh['Element']['Topology'].tolist()
for k in range(len(ElementT)):
    ElementT[k] = ElementT[k][0].flatten()
ElementE = mat_dict_mesh['ElementE'].tolist() #marks elements which are next to boundary electrodes
for k in range(len(ElementE)):
    if len(ElementE[k][0]) > 0:
        ElementE[k] = [ElementE[k][0][0][0], ElementE[k][0][0][1:len(ElementE[k][0][0])]]
    else:
        ElementE[k] = []

#Node structure
NodeC = mat_dict_mesh['Node']['Coordinate']
NodeE = mat_dict_mesh['Node']['ElementConnection'] #marks which elements a node belongs to
nodes = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC]
for k in range(NodeC.shape[0]):
    nodes[k].ElementConnection = NodeE[k][0].flatten()
elements = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT]
for k in range(len(ElementT)):
    elements[k].Electrode = ElementE[k]

#2nd order mesh data
H2 = mat_dict_mesh['H2']
g2 = mat_dict_mesh['g2']
elfaces2 = mat_dict_mesh['elfaces2'][0].tolist()
ElementT2 = mat_dict_mesh['Element2']['Topology']
ElementT2 = ElementT2.tolist()
for k in range(len(ElementT2)):
    ElementT2[k] = ElementT2[k][0].flatten()
ElementE2 = mat_dict_mesh['Element2E']
ElementE2 = ElementE2.tolist()
for k in range(len(ElementE2)):
    if len(ElementE2[k][0]) > 0:
        ElementE2[k] = [ElementE2[k][0][0][0], ElementE2[k][0][0][1:len(ElementE2[k][0][0])]]
    else:
        ElementE2[k] = []

NodeC2 = mat_dict_mesh['Node2']['Coordinate']  # ok
NodeE2 = mat_dict_mesh['Node2']['ElementConnection']  # ok
nodes2 = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC2]
for k in range(NodeC2.shape[0]):
    nodes2[k].ElementConnection = NodeE2[k][0].flatten()
elements2 = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT2]
for k in range(len(ElementT2)):
    elements2[k].Electrode = ElementE2[k]

Mesh = KTCMeshing.Mesh(H,g,elfaces,nodes,elements)
Mesh2 = KTCMeshing.Mesh(H2,g2,elfaces2,nodes2,elements2)

print(f'Nodes in inversion 1st order mesh: {len(Mesh.g)}')

# set up the forward solver for inversion
solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)

vincl = vincl.T.flatten()

# set up the noise model for inversion
noise_std1 = 0.05;  # standard deviation for first noise component (relative to each voltage measurement)
noise_std2 = 0.01;  # standard deviation for second noise component (relative to the largest voltage measurement)
solver.SetInvGamma(noise_std1, noise_std2, Uelref)

# %% Create regularization
sigma0 = np.ones((len(Mesh.g), 1)) #linearization point
corrlength = 1 * 0.115 #used in the prior
var_sigma = 0.05 ** 2 #prior variance
mean_sigma = sigma0
smprior = KTCRegularization.SMPrior(Mesh.g, corrlength, var_sigma, mean_sigma)
reg1_par_list = [1]#[0.5, 0.1, 0.05, 0.01] #[0.5, 1, 5, 10, 20]
reg2_par_list = np.logspace(1,12,30)#[0, 8e6]#[0.5, 0.1, 0.05, 0.01] #[0.5, 1, 5, 10, 20]

L = smprior.L
radius = np.max(np.linalg.norm(Mesh.g, axis = 1))
m = Mesh.g.shape[0]
num_el =  32 - (2*categoryNbr - 1)
electrodes = np.zeros((num_el, 2))
angle = 2*np.pi/Nel
for i in range(num_el):
    electrodes[i] = radius*np.array([np.sin(i*angle), np.cos(i*angle)])

"""
D = np.zeros(m)
for i in range(m):
    v = Mesh.g[i]
    dist = np.zeros(num_el)
    for k, e in enumerate(electrodes):
        dist[k] = np.linalg.norm(v - e)
    D[i] = np.linalg.norm(dist, ord = 0.5)
       

D = np.diag(D)  
reg_vis = KTCAux.interpolateRecoToPixGrid(np.diag(D), Mesh)
plt.figure()
plt.imshow(reg_vis)
for electrode in electrodes:
    plt.scatter(255*(0.5 + electrode[0]/(2*radius)), 255-255*(0.5 + electrode[1]/(2*radius)), s=10, c=k)
plt.colorbar()
plt.title("Local regularization")
plt.show()
""" 
radius = np.max(np.linalg.norm(Mesh.g, axis = 1))
top = np.array([0, radius])
angle = 2*np.pi*(categoryNbr - 1)/Nel
mid = radius*np.array([np.sin(-angle), np.cos(-angle)])
small_radius = np.linalg.norm(top-mid)

D = np.zeros(m)
for i in range(m):
    if np.linalg.norm(Mesh.g[i] - mid) <= small_radius:
        D[i] = np.linalg.norm(Mesh.g[i])**2
      
D = np.diag(D)  
reg_vis = KTCAux.interpolateRecoToPixGrid(np.diag(D), Mesh)
plt.figure()
plt.imshow(reg_vis)
plt.scatter(255*(0.5 + mid[0]/(2*radius)), 255-255*(0.5 + mid[1]/(2*radius)), s=10, c=k)
plt.colorbar()
plt.title("Local regularization")
plt.show()

# Get a list of .mat files in the input folder
mat_files = glob.glob(inputFolder + '/data*.mat')
for objectno in range(0,len(mat_files)): #compute the reconstruction forc each input file
    (Inj, Uel, Mpat, deltaU) = load_data(mat_files[objectno])

    Usim = solver.SolveForward(sigma0, z) #forward solution at the linearization point
    J = solver.Jacobian(sigma0, z)

    mask = np.array(vincl, bool)
    
    truth = sp.io.loadmat('GroundTruths/true' + str(objectno+1) + '.mat')
    truth = truth["truth"]
    
    plt.figure()
    plt.imshow(truth)
    plt.title(f"Truth: {mat_files[objectno]}")
    plt.show()
    
    scores = np.zeros((len(reg1_par_list), len(reg2_par_list)))
    for (ind1, reg1) in enumerate(reg1_par_list):
        for (ind2, reg2) in enumerate(reg2_par_list):
            deltareco = np.linalg.solve(J.T @ solver.InvGamma_n[np.ix_(mask,mask)] @ J + reg1*L.T @ L + reg2*D.T @ D,
                                        J.T @ solver.InvGamma_n[np.ix_(mask,mask)] @ deltaU[vincl])
                
            # interpolate the reconstruction into a pixel image
            deltareco_pixgrid = KTCAux.interpolateRecoToPixGrid(deltareco, Mesh)
            
            plt.figure()
            plt.imshow(deltareco_pixgrid)
            plt.colorbar()
            plt.title(f"obj_num = {objectno}, reg1_par = {reg1}, reg2_par = {reg2}")
            plt.show()
            
            # threshold the image histogram using Otsu's method
            reconstruction = segment(deltareco_pixgrid)
    
            plt.figure()
            plt.imshow(reconstruction)
            plt.colorbar()
            plt.title(f"obj_num = {objectno}, reg1_par = {reg1}, reg2_par = {reg2}")
            plt.show()
    
            score = KTCScoring.scoringFunction(truth, reconstruction)
            #scores[ind1, ind2] = KTCScoring.scoringFunction(truth, reconstruction)
            print(f"obj_num = {objectno}, reg1_par = {reg1}, reg2_par = {reg2}, score = {score}")
  
        
    plt.figure()
    plt.imshow(scores)
    plt.title(mat_files[objectno] + f", difficulty = {categoryNbr}")
    plt.colorbar()
    plt.show()
    