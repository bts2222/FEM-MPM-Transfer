import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

#Timestep to transfer from (0 = gravity, 1 = first dynamic step (after one timestep), 2 = second  dynamic step, etc)
ts = [0,4,8,12,16]
#ts = [0]

#Materials
mtags = [1]
rhos = [1700]

def eVol(x1,y1,x2,y2,x3,y3,x4,y4):
    return(0.5*((x1*y2+x2*y3+x3*y4+x4*y1)-(x2*y1+x3*y2+x4*y3+x1*y4)))

#Import Files
nodes = np.genfromtxt("nodeinfo.dat", dtype = float)
eles = np.genfromtxt("elementinfo.dat", dtype = float)
Gstress = np.genfromtxt("Gstress.out", dtype = float)
Gstrain = np.genfromtxt("Gstrain.out", dtype = float)
Gpp = np.genfromtxt("GporePressure.out",dtype=float)
vel = np.genfromtxt("velocity.out", dtype = float)
stress = np.genfromtxt("stress.out", dtype = float)
strain = np.genfromtxt("strain.out", dtype = float)
disp = np.genfromtxt("displacement.out", dtype = float)
pp = np.genfromtxt("porePressure.out",dtype=float)
neles = len(eles)
nnodes = len(nodes)

etn = np.zeros([nnodes,neles+1], dtype = float) #Which elements are used by each node, nodes x elements binary 1 if connected
etn[:,0] = nodes[:,0] #node ids
volumes = np.zeros([neles], dtype = float) #initial element volumes from FEM
N = np.empty([16,4], dtype = float)

#Calculate initial volume of each element
for e in range(0,neles):
    n1 = int(eles[e,1])
    n1 = np.where(nodes[:,0] == n1)
    n1 = n1[0]
    n2 = int(eles[e,2])
    n2 = np.where(nodes[:,0] == n2)
    n2 = n2[0]
    n3 = int(eles[e,3])
    n3 = np.where(nodes[:,0] == n3)
    n3 = n3[0]   
    n4 = int(eles[e,4])
    n4 = np.where(nodes[:,0] == n4)
    n4 = n4[0]
    volumes[e] = eVol(nodes[n1,1],nodes[n1,2],nodes[n2,1],nodes[n2,2],nodes[n3,1],nodes[n3,2],nodes[n4,1],nodes[n4,2])


#Create Shape Function, going right and then rows up starting at BL particle, nodes CCW from BL node, each row is different particle, each column a node
N[0,0] = .25*(1+0.8611)*(1+0.8611)
N[0,1] = .25*(1-0.8611)*(1+0.8611)
N[0,2] = .25*(1-0.8611)*(1-0.8611)
N[0,3] = .25*(1+0.8611)*(1-0.8611)
N[1,0] = .25*(1+0.3400)*(1+0.8611)
N[1,1] = .25*(1-0.3400)*(1+0.8611)
N[1,2] = .25*(1-0.3400)*(1-0.8611)
N[1,3] = .25*(1+0.3400)*(1-0.8611)
N[2,0] = .25*(1-0.3400)*(1+0.8611)
N[2,1] = .25*(1+0.3400)*(1+0.8611)
N[2,2] = .25*(1+0.3400)*(1-0.8611)
N[2,3] = .25*(1-0.3400)*(1-0.8611)
N[3,0] = .25*(1-0.8611)*(1+0.8611)
N[3,1] = .25*(1+0.8611)*(1+0.8611)
N[3,2] = .25*(1+0.8611)*(1-0.8611)
N[3,3] = .25*(1-0.8611)*(1-0.8611)
N[4,0] = .25*(1+0.8611)*(1+0.3400)
N[4,1] = .25*(1-0.8611)*(1+0.3400)
N[4,2] = .25*(1-0.8611)*(1-0.3400)
N[4,3] = .25*(1+0.8611)*(1-0.3400)
N[5,0] = .25*(1+0.3400)*(1+0.3400)
N[5,1] = .25*(1-0.3400)*(1+0.3400)
N[5,2] = .25*(1-0.3400)*(1-0.3400)
N[5,3] = .25*(1+0.3400)*(1-0.3400)
N[6,0] = .25*(1-0.3400)*(1+0.3400)
N[6,1] = .25*(1+0.3400)*(1+0.3400)
N[6,2] = .25*(1+0.3400)*(1-0.3400)
N[6,3] = .25*(1-0.3400)*(1-0.3400)
N[7,0] = .25*(1-0.8611)*(1+0.3400)
N[7,1] = .25*(1+0.8611)*(1+0.3400)
N[7,2] = .25*(1+0.8611)*(1-0.3400)
N[7,3] = .25*(1-0.8611)*(1-0.3400)
N[8,0] = .25*(1+0.8611)*(1-0.3400)
N[8,1] = .25*(1-0.8611)*(1-0.3400)
N[8,2] = .25*(1-0.8611)*(1+0.3400)
N[8,3] = .25*(1+0.8611)*(1+0.3400)
N[9,0] = .25*(1+0.3400)*(1-0.3400)
N[9,1] = .25*(1-0.3400)*(1-0.3400)
N[9,2] = .25*(1-0.3400)*(1+0.3400)
N[9,3] = .25*(1+0.3400)*(1+0.3400)
N[10,0] = .25*(1-0.3400)*(1-0.3400)
N[10,1] = .25*(1+0.3400)*(1-0.3400)
N[10,2] = .25*(1+0.3400)*(1+0.3400)
N[10,3] = .25*(1-0.3400)*(1+0.3400)
N[11,0] = .25*(1-0.8611)*(1-0.3400)
N[11,1] = .25*(1+0.8611)*(1-0.3400)
N[11,2] = .25*(1+0.8611)*(1+0.3400)
N[11,3] = .25*(1-0.8611)*(1+0.3400)
N[12,0] = .25*(1+0.8611)*(1-0.8611)
N[12,1] = .25*(1-0.8611)*(1-0.8611)
N[12,2] = .25*(1-0.8611)*(1+0.8611)
N[12,3] = .25*(1+0.8611)*(1+0.8611)
N[13,0] = .25*(1+0.3400)*(1-0.8611)
N[13,1] = .25*(1-0.3400)*(1-0.8611)
N[13,2] = .25*(1-0.3400)*(1+0.8611)
N[13,3] = .25*(1+0.3400)*(1+0.8611)
N[14,0] = .25*(1-0.3400)*(1-0.8611)
N[14,1] = .25*(1+0.3400)*(1-0.8611)
N[14,2] = .25*(1+0.3400)*(1+0.8611)
N[14,3] = .25*(1-0.3400)*(1+0.8611)
N[15,0] = .25*(1-0.8611)*(1-0.8611)
N[15,1] = .25*(1+0.8611)*(1-0.8611)
N[15,2] = .25*(1+0.8611)*(1+0.8611)
N[15,3] = .25*(1-0.8611)*(1+0.8611)


for a in range(0,len(ts)):
    print('Beginning transfer at step '+str(ts[a]))
    Gicor = np.empty([neles,2], dtype = float) #Coordinates of initial gauss points and x and y distances of elements
    Gnew = np.zeros([16*neles,21], dtype = float) #Particle Data, element(0), coordinates(1-2), velocities(3-4), stresses(5-7), strain(8-10), 
    # vol(11), volstrain(12), devstrain(13), pore pressure(14), sigv0(15), ru(16), materialtag(17), displacements(18,19)
    dvolumes = np.zeros([neles], dtype = float) #deformed element volumes from FEM 
    ### Shape Functions Write Nodal State Variables to Particles
    for e in range(0,neles): #loop over elements
        nxy = np.empty([4,8], dtype = float) #for use in for loop, node info
        #positions (0,1), velocities (2,3), pore pressure (4), ru (5), displacements
        #node x parameter, used in shape function
        for n in range(0,4): #4 nodes/element
            #convert from node # to row in node info
            node = int(eles[e,n+1])
            node = np.where(nodes[:,0] == node)
            node = int(node[0])
            #identify values of state variables of each connected node
            if ts[a] == 0:
                nxy[n,0] = nodes[node,1]
                nxy[n,1] = nodes[node,2]
                nxy[n,2] = 0
                nxy[n,3] = 0
                nxy[n,4] = Gpp[len(Gpp)-1,node+1]
                nxy[n,5] = 0 #ru
                nxy[n,6] = 0
                nxy[n,7] = 0
            else:
                nxy[n,0] = nodes[node,1]+disp[ts[a],node*2+1]
                nxy[n,1] = nodes[node,2]+disp[ts[a],node*2+2]
                nxy[n,2] = vel[ts[a],node*2+1]
                nxy[n,3] = vel[ts[a],node*2+2]
                nxy[n,4] = pp[ts[a],node+1]
                nxy[n,5] = (pp[ts[a],node+1]-Gpp[len(Gpp)-1,node+1])/-Gstress[len(Gstress)-1,e+2]
                nxy[n,6] = disp[ts[a],node*2+1]
                nxy[n,7] = disp[ts[a],node*2+2]
        Gicor[e,0] = sum(nxy[:,0])/4
        Gicor[e,1] = sum(nxy[:,1])/4
        dvolumes[e] = eVol(nxy[0,0],nxy[0,1],nxy[1,0],nxy[1,1],nxy[2,0],nxy[2,1],nxy[3,0],nxy[3,1])
        for p in range(0,16):
            Gnew[e*16+p,0] = e+1 #Which element it is a part of
            Gnew[e*16+p,1:5] = np.matmul(N[p,0:4],nxy[0:4,0:4]) #disp and vel
            #volume for each particle based on Gauss weight of location
            if p == 0 or p == 3 or p == 12 or p == 15:
                Gnew[e*16+p,11] = dvolumes[e]*(.347855/2)**2
                Gnew[e*16+p,20] = volumes[e]*(.347855/2)**2
            elif p == 5 or p == 6 or p == 9 or p == 10:
                Gnew[e*16+p,11] = dvolumes[e]*(.652145/2)**2
                Gnew[e*16+p,20] = volumes[e]*(.652145/2)**2
            else:
                Gnew[e*16+p,11] = dvolumes[e]*(.652145/2)*(.347855/2)
                Gnew[e*16+p,20] = volumes[e]*(.652145/2)*(.347855/2)
            Gnew[e*16+p,12] = (dvolumes[e]-volumes[e])/volumes[e] #volstrain
            Gnew[e*16+p,14] = np.matmul(N[p,0:4],nxy[0:4,4]) #pwp
            Gnew[e*16+p,15] = Gstress[len(Gstress)-1,e+2] #sigv0
            Gnew[e*16+p,16] = np.matmul(N[p,0:4],nxy[0:4,5]) #ru
            Gnew[e*16+p,17] = int(eles[e,5]) #material tag
            Gnew[e*16+p,18:20] = np.matmul(N[p,0:4],nxy[0:4,6:8])
    del nxy

    ### Stress and Strain RBF fields
    print('   Creating stress and strain fields')
    stress3 = np.empty([neles,3]) #x,y,shear
    strain3 = np.empty([neles,3]) #x,y,shear,dev
    if ts[a] == 0:
        for e in range(0,neles):
            stress3[e,0] = Gstress[ts[a],3*e+1]
            stress3[e,1] = Gstress[ts[a],3*e+2]
            stress3[e,2] = Gstress[ts[a],3*e+3]
            strain3[e,0] = Gstrain[ts[a],3*e+1]
            strain3[e,1] = Gstrain[ts[a],3*e+2]
            strain3[e,2] = Gstrain[ts[a],3*e+3]
    else:
        for e in range(0,neles):
            stress3[e,0] = stress[ts[a],3*e+1]
            stress3[e,1] = stress[ts[a],3*e+2]
            stress3[e,2] = stress[ts[a],3*e+3]
            strain3[e,0] = strain[ts[a],3*e+1]
            strain3[e,1] = strain[ts[a],3*e+2]
            strain3[e,2] = strain[ts[a],3*e+3]
    hsf = Rbf(Gicor[:,0],Gicor[:,1],stress3[:,0])
    Gnew[:,5] = hsf(Gnew[:,1],Gnew[:,2])
    del hsf
    vsf = Rbf(Gicor[:,0],Gicor[:,1],stress3[:,1])
    Gnew[:,6] = vsf(Gnew[:,1],Gnew[:,2])
    del vsf
    ssf = Rbf(Gicor[:,0],Gicor[:,1],stress3[:,2])
    Gnew[:,7] = ssf(Gnew[:,1],Gnew[:,2])
    del ssf
    hef = Rbf(Gicor[:,0],Gicor[:,1],strain3[:,0])
    Gnew[:,8] = hef(Gnew[:,1],Gnew[:,2])
    del hef
    vef = Rbf(Gicor[:,0],Gicor[:,1],strain3[:,1])
    Gnew[:,9] = vef(Gnew[:,1],Gnew[:,2])
    del vef
    sef = Rbf(Gicor[:,0],Gicor[:,1],strain3[:,2])
    Gnew[:,10] = sef(Gnew[:,1],Gnew[:,2])
    del sef
    #deviatoric strain
    for p in range(0,neles*16):
        e11 = Gnew[p,8] - Gnew[p,12]/3
        e22 = Gnew[p,9] - Gnew[p,12]/3
        Gnew[p,13] = (2/3)**0.5*(e11**2+e22**2+Gnew[p,10]**2/2)**0.5 

    np.savetxt("Gnew.txt", Gnew, fmt="%s", delimiter=' ')

    ### Create h5 file
    print('   Creating h5 file')
    data = np.zeros([neles*16,53], dtype = object)
    for p in range(0,neles*16): #row in respective hdf5 file
        data[p,0] = p #id
        for m in range(0,len(mtags)+1):
            if Gnew[p,17] == mtags[m]:
                data[p,1] = Gnew[p,20]*rhos[m]
                break
            elif m > len(mtags):
                print('  Unrecognized Material tag '+str(Gnew[p,17]))
        data[p,2] = Gnew[p,11] #volume
        #pressure left as zero
        data[p,4] = Gnew[p,1] #xcor
        data[p,5] = Gnew[p,2] #ycor
        #zcor = 0
        data[p,7] = Gnew[p,18] #xdisp
        data[p,8] = Gnew[p,19] #ydisp
        #zdisp = 0
        data[p,10] = Gnew[p,11]**0.5 #xsize
        data[p,11] = Gnew[p,11]**0.5 #ysize
         #zsize = 0
        data[p,13] = Gnew[p,3] #xvel
        data[p,14] = Gnew[p,4] #yvel
        #zvel left as zero
        data[p,16] = (Gnew[p,5]-Gnew[p,14])*1000 #sigmaxx (convert to Pa)
        data[p,17] = (Gnew[p,6]-Gnew[p,14])*1000 #sigmayy
        #sigmazz left as zero
        data[p,19] = Gnew[p,7]*1000 #tauxy
        #tauyz left as zero
        #tauxz left as zero
        data[p,22] = Gnew[p,8] #epsilonxx
        data[p,23] = Gnew[p,9] #epsilonyy
        #epsilonzz left as zero
        data[p,25] = Gnew[p,10] #gammaxy
        #gammayz left as zero
        #gammaxz left as zero
        data[p,28] = Gnew[p,12] #epsilonv
        data[p,29] = int(1) #status = 1
        data[p,30] = int(18446744073709551615) #cell id = 18446744073709551615
        data[p,31] = int(Gnew[p,17]) #material id
        #nstatevectors = 0
        #svars_0-5 (33,34,35,36,37,38) left as zero
        data[p,39] = Gnew[p,13] #svars_6, pdstrain
        #svars_6-19 (39-52) left as zero
        ##input array into data frame
    frame = pd.DataFrame(data, columns=['id','mass','volume','pressure','coord_x','coord_y','coord_z',
                                    'displacement_x','displacement_y','displacement_z','nsize_x',
                                    'nsize_y','nsize_z','velocity_x','velocity_y','velocity_z',
                                    'stress_xx','stress_yy','stress_zz','tau_xy','tau_yz','tau_xz',
                                    'strain_xx','strain_yy','strain_zz','gamma_xy','gamma_yz','gamma_xz',
                                    'epsilon_v','status','cell_id','material_id','nstate_vars','svars_0',
                                    'svars_1','svars_2','svars_3','svars_4','svars_5','svars_6','svars_7',
                                    'svars_8','svars_9','svars_10','svars_11','svars_12','svars_13',
                                    'svars_14','svars_15','svars_16','svars_17','svars_18','svars_19'])
    ##save data frame as hdf5
    frame.to_csv('particles'+str(ts[a])+'.csv',index=False,mode='w')
    print('   Finished')


