import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

#Timestep to transfer from (0 = gravity, 1 = first dynamic step (after one timestep), 2 = second  dynamic step, etc)
#ts = [30,40,43,44,45,46,47,48,49,50,51,52,53,54,60,69] #PGA 0.4
#ts = [30,40,41,42,43,44,45,46,47,48,49,50,51,60] #PGA 0.5
#ts = [50,51,52,53,54,55,56,57] #PGA 0.3 early
#ts = [60,70,80,90,100,110,120,130] #PGA 0.2
#ts = [46,47,48,49]
ts = [80]

#Materials
mtags = [167,168,170]
rhos = [1690,1870,1900]
#Bottom left node - for removing base displacement
blnode = 176 #node number, not row
#Truncation Limits
#tx = [-340,87.5] #upstream
tx = [-340,140] #downstream
ty = [-5,100]
#Liquefaction Geometry PGA 0.4
# Lx1 = [0,-31.2,-31.2,-31.2,-31,-31,-31,-61,-61,-61,-61,-61,-61,-61,-31.3,-31.1]
# Ly1 = [25,21,21,21,12,12,12,11.2,11.2,11.2,11.2,11.2,11.2,11.2,9,4.5]
# Lx2 = [16.1,12.3,12.3,12.3,-3.1,-3.1,-3.1,4.3,4.3,4.3,4.3,4.3,4.3,4.3,2.8,-6.6]
# Ly2 = [25,9.8,9.8,9.8,10.3,10.3,10.3,7.5,7.5,7.5,7.5,7.5,7.5,7.5,4.8,1.6]
#Liquefaction Geometry PGA 0.5
# Lx1 = [0,-22.6,-22.6,-55.67,-55.67,-55.67,-55.67,0,0,0,0,0,0,-24.7]
# Ly1 = [24,18.4,18.4,14.29,14.29,14.29,14.29,5,5,5,5,4.5,4.5,3.8]
# Lx2 = [1,8.9,8.9,-7.64,-7.64,-7.64,-7.64,1,1,1,1,1,1,1.8]
# Ly2 = [24,10.2,10.2,6.08,6.08,6.08,6.08,5,5,5,5,4.5,4.5,1.8]
#PGA 0.3 early times
# Lx1 = [-12.6,-12.6,-12.6,-12.6,-12.6,-12.6,-12.6,-12.6]
# Ly1 = [16.3,16.3,16.3,16.3,16.3,16.3,16.3,16.3]
# Lx2 = [10.6,10.6,10.6,10.6,10.6,10.6,10.6,10.6]
# Ly2 = [10.9,10.9,10.9,10.9,10.9,10.9,10.9,10.9]
# PGA 0.3 Even Earlier Times
# Lx1=[-28.6,-28.6,-28.6,-28.6]
# Ly1 = [19.6,19.6,19.6,19.6]
# Lx2 = [-4.4,-4.4,-4.4,-4.4,]
# Ly2 = [14.2,14.2,14.2,14.2]
#PGA 0.2 
# Lx1 = [0,-45.1,-41.37,-67,-71.8,-79,-123,135.5]
# Ly1 = [24,22.7,22.1,22.9,22.7,21.6,22,21.2]
# Lx2 = [1,-7.5,8.13,-0.2,-14.3,-1.8,-25.5,-14.4]
# Ly2 = [24,13.1,9.63,6.5,6.9,4.6,6.9,4.6]

Lx1 = [-1.3]
Ly1 = [3.3]
Lx2 = [9.5]
Ly2 = [6.1]

#MPM MC Properties
#[sat emb,dry emb,unL tail,L tail 7,found,L tail 6,L tail 5,L tail 4,L tail 3,L tail 2,L tail 1]
mpmtags = [166,167,168,169,170,171,172,173,174,175,176] #material tag
mpmphi0 = [35,35,30,0,0,0,0,0,0,0,0] #peak friction angle
mpmphiR = [20,20,0,0,0,0,0,0,0,0,0] #residual friction angle
mpmC0 = [20,20,1,12,0,11.5,10.6,9.1,7.4,5.6,3] #peak cohesion
mpmCR = [1,1,5,6,0,5.75,5.3,4.55,3.7,2.8,1.5] #residual cohesion
mpmeR = [.05,.05,1,1,1,1,1,1,1,1,1] #residual strain

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
print('FEM files imported')

etn = np.zeros([nnodes,neles+1], dtype = float) #Which elements are used by each node, nodes x elements binary 1 if connected
etn[:,0] = nodes[:,0] #node ids
volumes = np.zeros([neles], dtype = float) #initial element volumes from FEM
N = np.empty([4,4], dtype = float)

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
N[0,0] = .25*(1+0.57735)*(1+0.57735)
N[0,1] = .25*(1-0.57735)*(1+0.57735)
N[0,2] = .25*(1-0.57735)*(1-0.57735)
N[0,3] = .25*(1+0.57735)*(1-0.57735)
N[1,0] = .25*(1-0.57735)*(1+0.57735)
N[1,1] = .25*(1+0.57735)*(1+0.57735)
N[1,2] = .25*(1+0.57735)*(1-0.57735)
N[1,3] = .25*(1-0.57735)*(1-0.57735)
N[2,0] = .25*(1-0.57735)*(1-0.57735)
N[2,1] = .25*(1+0.57735)*(1-0.57735)
N[2,2] = .25*(1+0.57735)*(1+0.57735)
N[2,3] = .25*(1-0.57735)*(1+0.57735)
N[3,0] = .25*(1+0.57735)*(1-0.57735)
N[3,1] = .25*(1-0.57735)*(1-0.57735)
N[3,2] = .25*(1-0.57735)*(1+0.57735)
N[3,3] = .25*(1+0.57735)*(1+0.57735)



for a in range(0,len(ts)):
    print('Beginning transfer at step '+str(ts[a]))
    Gicor = np.empty([neles,2], dtype = float) #Coordinates of initial gauss points and x and y distances of elements
    Gnew = np.zeros([4*neles,21], dtype = float) #Particle Data, element(0), coordinates(1-2), velocities(3-4), stresses(5-7), strain(8-10), 
    # vol(11), volstrain(12), devstrain(13), pore pressure(14), sigv0(15), ru(16), materialtag(17), displacements(18,19), (20) vol0
    dvolumes = np.zeros([neles], dtype = float) #deformed element volumes from FEM 
    ### Shape Functions Write Nodal State Variables to Particles
    for e in range(0,neles): #loop over elements
        nxy = np.empty([4,8], dtype = float) #for use in for loop, node info
        #positions (0,1), velocities (2,3), pore pressure (4), ru (5)
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
                nxy[n,0] = nodes[node,1]+disp[ts[a],node*2+1] #coord x
                nxy[n,1] = nodes[node,2]+disp[ts[a],node*2+2] #coord y
                nxy[n,2] = vel[ts[a],node*2+1] #vel x
                nxy[n,3] = vel[ts[a],node*2+2] #vel y
                nxy[n,4] = pp[ts[a],node+1] #pp
                nxy[n,5] = (pp[ts[a],node+1]-Gpp[len(Gpp)-1,node+1])/-Gstress[len(Gstress)-1,3*e+2] #ru
                nxy[n,6] = disp[ts[a],node*2+1] #disp x
                nxy[n,7] = disp[ts[a],node*2+2] #disp y
        Gicor[e,0] = sum(nxy[:,0])/4
        Gicor[e,1] = sum(nxy[:,1])/4
        dvolumes[e] = eVol(nxy[0,0],nxy[0,1],nxy[1,0],nxy[1,1],nxy[2,0],nxy[2,1],nxy[3,0],nxy[3,1])
        for p in range(0,4):
            Gnew[e*4+p,0] = e+1 #Which element it is a part of
            Gnew[e*4+p,1:5] = np.matmul(N[p,0:4],nxy[0:4,0:4]) #disp and vel
            #volume for each particle based on Gauss weight of location
            Gnew[e*4+p,11] = dvolumes[e]/4
            Gnew[e*4+p,12] = (dvolumes[e]-volumes[e])/volumes[e] #volstrain
            Gnew[e*4+p,14] = np.matmul(N[p,0:4],nxy[0:4,4]) #pwp
            Gnew[e*4+p,15] = Gstress[len(Gstress)-1,3*e+2] #sigv0
            Gnew[e*4+p,16] = np.matmul(N[p,0:4],nxy[0:4,5]) #ru
            Gnew[e*4+p,17] = int(eles[e,5]) #material tag
            Gnew[e*4+p,18:20] = np.matmul(N[p,0:4],nxy[0:4,6:8])
            Gnew[e*4+p,20] = volumes[e]/4
    del nxy
    np.savetxt("Gnew.txt", Gnew, fmt="%s", delimiter=' ')
    np.savetxt("Gicor.txt", Gicor, fmt="%s", delimiter=' ')

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
    for p in range(0,len(Gnew)):
        e11 = Gnew[p,8] - Gnew[p,12]/3
        e22 = Gnew[p,9] - Gnew[p,12]/3
        Gnew[p,13] = (2/3)**0.5*(e11**2+e22**2+Gnew[p,10]**2/2)**0.5
    #np.savetxt("Gnew.txt", Gnew, fmt="%s", delimiter=' ')
        
    ## Truncate Particles outside specified zone
    ptparts = len(Gnew)
    baseD = disp[ts[a],blnode*2-1] #displacement of base
    for p in range(len(Gnew)-1,-1,-1):
        if (Gnew[p,1]-baseD)<tx[0] or (Gnew[p,1]-baseD)>tx[1] or Gnew[p,2]<ty[0] or Gnew[p,2]>ty[1]:
            Gnew = np.delete(Gnew,p,0)
    atparts = len(Gnew)
    print('   '+str(ptparts-atparts)+'/'+str(ptparts)+' particles truncated')

    ## Calculations for Liquefaction Geometry
    Lm = (Ly2[a]-Ly1[a])/(Lx2[a]-Lx1[a])
    Lb = Ly1[a] - Lm*Lx1[a]

    ### Create h5 file --------------------------------------------------------------------------------------------
    print('   Creating h5 file')
    data = np.zeros([len(Gnew),53], dtype = object)
    for p in range(0,len(Gnew)): #row in respective hdf5 file
        data[p,0] = p #id
        for m in range(0,len(mtags)+1):
            if Gnew[p,17] == mtags[m]:
                data[p,1] = Gnew[p,20]*rhos[m]
                break
            elif m > len(mtags):
                print('   Unrecognized Material tag '+str(Gnew[p,17]))
        data[p,2] = Gnew[p,11] #volume
        #pressure left as zero
        data[p,4] = Gnew[p,1]-baseD #xcor, with base displacement removed
        data[p,5] = Gnew[p,2] #ycor
        #zcor = 0
        data[p,7] = Gnew[p,18]-baseD #xdisp
        data[p,8] = Gnew[p,19] #ydisp
        #zdisp = 0
        data[p,10] = Gnew[p,11]**0.5 #xsize
        data[p,11] = Gnew[p,11]**0.5 #ysize
         #zsize = 0
        data[p,13] = Gnew[p,3] #xvel
        data[p,14] = Gnew[p,4] #yvel
        #zvel left as zero
        data[p,16] = (Gnew[p,5]-Gnew[p,14])*1000 #sigmaxx (Total stress, converted to Pa, negative compression)
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
        #if liquefied and tailings, give different mat tag:
        if (Gnew[p,16]>0.7 and Gnew[p,17]==168) or (Gnew[p,17]==168 and Gnew[p,1]<Lx1[a] and Gnew[p,2]>Ly1[a]) or (Gnew[p,17]==168 and Gnew[p,1]>Lx1[a] and Gnew[p,2]>(Lm*Gnew[p,1]+Lb)): 
            if Gnew[p,15] > -50:
                data[p,31] = int(176)
            elif Gnew[p,15] > -100:
                data[p,31] = int(175)
            elif Gnew[p,15] > -150:
                data[p,31] = int(174)
            elif Gnew[p,15] > -200:
                data[p,31] = int(173)
            elif Gnew[p,15] > -250:
                data[p,31] = int(172)
            elif Gnew[p,15] > -300:
                data[p,31] = int(171)
            elif Gnew[p,15] < -300:
                data[p,31] = int(169)
        elif (Gnew[p,17]==167 and Gnew[p,2]<(-0.303*Gnew[p,2]+18.427)):
            data[p,31] = int(166)
        else:
            data[p,31] = int(Gnew[p,17]) #original material id
        if Gnew[p,17] == 170: #170 (foundation) is LE so no state variables to transfer
            data[p,32] = 0
        else: #transfer of state variables for softening MC models
            for m in range(0,len(mpmtags)): #identify material tag
                if data[p,31] == mpmtags[m]:
                    if Gnew[p,13] < mpmeR[m]: #in process of softening
                        data[p,32] = 7
                        data[p,33] = mpmphiR[m]+(mpmphi0[m]-mpmphiR[m])*(mpmeR[m]-Gnew[p,13]) #svars_0, phi
                        data[p,34] = 0 #svars_1, dilation (0)
                        data[p,35] = mpmCR[m]+(mpmC0[m]-mpmCR[m])*(mpmeR[m]-Gnew[p,13]) #svars_2, cohesion
                        data[p,36] = 0 #svars_3, epsilon (0)
                        data[p,37] = 0 #svars_4, rho (0)
                        data[p,38] = 0 #svars_5, theta (0)
                        data[p,39] = Gnew[p,13] #svars_6, pdstrain
                    elif Gnew[p,13] >= mpmeR[m]: #or fully softened
                        data[p,32] = 7
                        data[p,33] = mpmphiR[m] #svars_0, phi
                        data[p,34] = 0 #svars_1, dilation (0)
                        data[p,35] = mpmCR[m] #svars_2, cohesion
                        data[p,36] = 0 #svars_3, epsilon (0)
                        data[p,37] = 0 #svars_4, rho (0)
                        data[p,38] = 0 #svars_5, theta (0)
                        data[p,39] = Gnew[p,13] #svars_6, pdstrain
                else:
                    print('MPM material not found')
        #svars_7-19 (39-52) left as zero
        #NOTE: These state variables are recalculated every timestep in the MPM code, however deviatoric strain is only added.
        #i.e. the amount of deviatoric strain expereinced in that timestep is added to the total deviatoric strain, so you must transfer 
            #the amount developed in the FEM phase. dilation, epsilon, rho, and theta and recalced after one timestep in MPM
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


