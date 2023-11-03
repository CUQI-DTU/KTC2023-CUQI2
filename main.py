import argparse
import numpy as np
import scipy as sp
import KTCFwd
import KTCMeshing
import KTCRegularization
import KTCPlotting
import KTCScoring
import KTCAux
import matplotlib.pyplot as plt
import glob
from skimage.segmentation import chan_vese

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("inputFolder")
    parser.add_argument("outputFolder")
    parser.add_argument("categoryNbr", type=int)
    args = parser.parse_args()

    inputFolder = args.inputFolder
    outputFolder = args.outputFolder
    categoryNbr = args.categoryNbr

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

    # print(f'Nodes in inversion 1st order mesh: {len(Mesh.g)}')

    sigma0 = np.ones((len(Mesh.g), 1)) #linearization point
    corrlength = 1 * 0.115 #used in the prior
    var_sigma = 0.05 ** 2 #prior variance
    mean_sigma = sigma0
    smprior = KTCRegularization.SMPrior(Mesh.g, corrlength, var_sigma, mean_sigma)

    # set up the forward solver for inversion
    solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)

    vincl = vincl.T.flatten()

    # set up the noise model for inversion
    noise_std1 = 0.05;  # standard deviation for first noise component (relative to each voltage measurement)
    noise_std2 = 0.01;  # standard deviation for second noise component (relative to the largest voltage measurement)
    solver.SetInvGamma(noise_std1, noise_std2, Uelref)

    # Get a list of .mat files in the input folder
    mat_files = sorted(glob.glob(inputFolder + '/data*.mat'))
    for objectno in range (0,len(mat_files)): #compute the reconstruction for each input file
        mat_dict2 = sp.io.loadmat(mat_files[objectno])
        Inj = mat_dict2["Inj"]
        Uel = mat_dict2["Uel"]
        Mpat = mat_dict2["Mpat"]
        deltaU = Uel - Uelref

        #############################  Changed code
        reg1 = 0.6
        reg2 = 6*1e6
        smprior.L = reg1*smprior.L
        m = Mesh.g.shape[0]
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

        mask = np.array(vincl, bool) # Mask the removed electrodes

        deltaU = Uel - Uelref
        J = solver.Jacobian(sigma0, z)
        deltareco = np.linalg.solve(J.T @ solver.InvGamma_n[np.ix_(mask,mask)] @ J + smprior.L.T @ smprior.L + reg2*D.T@D,
                                    J.T @ solver.InvGamma_n[np.ix_(mask,mask)] @ deltaU[vincl])

        # interpolate the reconstruction into a pixel image
        deltareco_pixgrid = KTCAux.interpolateRecoToPixGrid(deltareco, Mesh)
        # Do Chan-Vese segmentation

        mu = np.mean(deltareco_pixgrid)

        # Feel free to play around with the parameters to see how they impact the result
        cv = chan_vese(abs(deltareco_pixgrid), mu=0.1, lambda1=1, lambda2=1, tol=1e-6,
                    max_num_iter=1000, dt=2.5, init_level_set="checkerboard",
                    extended_output=True)

        labeled_array, num_features = sp.ndimage.label(cv[0])
        # Initialize a list to store masks for each region
        region_masks = []

        # Loop through each labeled region
        deltareco_pixgrid_segmented = np.zeros((256,256))

        for label in range(1, num_features + 1):
            # Create a mask for the current region
            region_mask = labeled_array == label
            region_masks.append(region_mask)
            if np.mean(deltareco_pixgrid[region_mask]) < mu:
                deltareco_pixgrid_segmented[region_mask] = 1
            else:
                deltareco_pixgrid_segmented[region_mask] = 2
        ###################################  End of changed code

        reconstruction = deltareco_pixgrid_segmented
        mdic = {"reconstruction": reconstruction}
        print(outputFolder + '/' + str(objectno + 1) + '.mat')
        sp.io.savemat( outputFolder + '/' + str(objectno + 1) + '.mat',mdic)

if __name__ == "__main__":
    main()
