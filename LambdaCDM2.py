
# import usefull libraries:

import camb
from camb import model, initialpower
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from CosmoFunc import *
from BF_OptMC import *
import time
import os
from numba import jit, prange

#######################################################################################

# cosmological parameters:
OmegaM= 0.3151
OmegaLambda = 1.0-OmegaM
Omegarad = 10**(-5)
c = 299792.458 # speed of light in km/s
wo = -1 # for lambdaCDM
wa = 0 # for lambdaCDM
H0 = 100. 
littleh=0.6727

#########################################################################################

# 1. We calculate the linear matter power spectrum using CAMB:
pars = camb.CAMBparams()
pars.set_cosmology(H0*littleh,ombh2=0.022, omch2=0.1198)
pars.set_dark_energy() #re-set defaults
pars.InitPower.set_params(As=2.114940245149156e-09,ns=0.9645)
pars.set_matter_power(redshifts=[0.], kmax=0.4)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
khs, z, pks = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 300) 
Sig8_fid = results.get_sigma8()[0]
Pk_linear=pks[0,:]
k_linear=khs

#################################################################################################

#2. get the coordinates of r from the (Ra,dec) coordinates from data:

# extract the Ra,Dec and z from the data file:
#data_PC = 'C:/Users/lucil/Documents/Stage M2/data_DESI/PV_clustering_data_v5_v13.fits'  # PC portable
#data_PC_fix = '/renoir/fromenti/Documents/data_DESI/combinedpv/Y1/PV_clustering_data_v5_v13.fits' 
data_mardec = '/datadec/desi/pv/combinedpv/Y1/PV_clustering_data_v5_v13.fits'

hdul = fits.open(data_mardec)
with fits.open(data_mardec) as hfile:
    infile = hfile[1].data


ra          = infile['RA']
dec         = infile['DEC']
rsf         = infile['Z']
rsf_err     = infile['ZCMB'] # the error on the redshift ?
logdist     = infile['LOGDIST']
err_logdist = infile['LOGDIST_ERR']
weights     = infile['WEIGHT'] # But we have one weight for one galaxy and for one component Bx,By;Bz,Qxx,Qxy,... 
pv          = infile['PV'] 
pv_err      = infile['PV_ERR'] 

n_galaxies = len(rsf)

# Now we want to extract the comoving distances of the galaxie assuming a model from the redshift:
d_comoving = []

for i in range(n_galaxies):
    d_comoving.append(DistDc(rsf[i], OmegaM, OmegaLambda, Omegarad, H0, wo, wa, ap=1)) # in Mpc ap does not matter for the distance

d_rsf = []
for i in range(n_galaxies):
    d_rsf.append(DistDc(rsf[i], OmegaM, OmegaLambda, Omegarad, H0, wo, wa, ap=1)) # in Mpc ap does not matter for the distance
d_comoving = d_rsf*10.**(-logdist)   

# Now that we have the comobile distance we can reconstruct the 3D r vector using the Ra and Dec: r_n = d_comoving[n] * r^n
# We convert the Ra and Dec in cartesian coordinates :

x_coord =  np.cos(dec/180*np.pi) * np.cos(ra/180*np.pi)  # we divide RA and DEC by 180 and multiply by pi to convert the angles in radians
y_coord =  np.cos(dec/180*np.pi) * np.sin(ra/180*np.pi)
z_coord =  np.sin(dec/180*np.pi)

# We stack the x,y,z in a single array:
r_hat = np.vstack((x_coord, y_coord, z_coord)).T
d_comoving = np.array(d_comoving) # the comoving distance in Mpc must be an array to be able to convert it in a [N,1] array to multiply it with r_hat [N,3]
r = r_hat * d_comoving[:, None]


###########################################################################################################
#4.Weights calculation:

# calculate the error on the peculiar velocity using the error on the logdistance and the redshift of the first galaxy:

alpha_n = np.log(10) * c * rsf * err_logdist / (1+rsf) # error on the peculiar velocity in km/s
alpha_star = 300 # in km/s is a estimation of the typical value of the error of the peculiar velocity of the galaxies

# Lets calculate the weights with the formula from the paper:

# calculating the g function for each galaxy:

g = np.column_stack([x_coord,                       # x
                     y_coord,                       # y
                     z_coord,                       # z 
                     d_comoving*x_coord**2,         # xx
                     2*d_comoving * x_coord*y_coord,# xy
                     2*d_comoving * x_coord*z_coord,# xz
                     d_comoving * y_coord**2,       # yy
                     2*d_comoving * y_coord*z_coord,# yz
                     d_comoving * z_coord**2])      # zz

# calculating the A matrix:
alpha_tot2 = alpha_n**2 + alpha_star**2  # (N,)
A = g.T @ (g / alpha_tot2[:, None])

# calculating the weights:
A_inv = np.linalg.inv(A)
w = 1 / (alpha_n**2 + alpha_star**2)
weights_calc = A_inv @ (g * w[:, None]).T

#########################################################################################################################

# Version Numba de f_nm - PAS de scipy, PAS de np.clip la fonction marche bien elle a été tester dans plot_CDM2 
@jit(nopython=True)
def f_nm_numba(k, n, m, d_comoving, r_hat):
    """Version Numba pure - n'utilise que des fonctions supportées"""
    
    if n == m:
        return 1.0 / 3.0
    
    # Vecteurs unitaires
    rn_x = r_hat[0, n]
    rn_y = r_hat[1, n]
    rn_z = r_hat[2, n]
    rm_x = r_hat[0, m]
    rm_y = r_hat[1, m]
    rm_z = r_hat[2, m]
    
    # Produit scalaire manuel
    cosanm = rn_x * rm_x + rn_y * rm_y + rn_z * rm_z
    
    # Clip manuel
    if cosanm > 1.0:
        cosanm = 1.0
    if cosanm < -1.0:
        cosanm = -1.0
    
    # Arccos
    alphanm = np.arccos(cosanm)
    
    # Distances
    rn = d_comoving[n]
    rm = d_comoving[m]
    
    # Calcul Aij2
    Aij2 = rn * rn + rm * rm - 2.0 * rn * rm * cosanm
    
    if Aij2 <= 1e-10:
        return 1.0 / 3.0
    
    Aij = np.sqrt(Aij2)
    C1 = cosanm / 3.0
    sinanm = np.sin(alphanm)
    C2 = (rn * rm * sinanm * sinanm) / (Aij * Aij)
    
    kA = k * Aij
    kA2 = kA * kA
    
    if kA2 < 1e-10:
        return 1.0 / 3.0
    
    sinkA = np.sin(kA)
    coskA = np.cos(kA)
    
    term1 = sinkA/kA * (3.0 - 6.0/kA2) + 6.0*coskA/kA2
    term2 = (3.0/kA2 - 1.0)*sinkA/kA - 3.0*coskA/kA2
    
    fmnk = C1 * term1 + C2 * term2
    
    return fmnk


# Version Numba de window_func
@jit(nopython=True, parallel=True)
def window_func_numba(k_array, p, q, N, weights, d_comoving, r_hat):
    """Version Numba avec parallélisation"""
    n_k = len(k_array)
    W = np.zeros(n_k)
    
    # Paralléliser sur les valeurs de k
    for i in prange(n_k):
        k = k_array[i]
        total = 0.0
        
        for n in range(N):
            w_pn = weights[p, n]
            for m in range(N):
                if m == n:
                    continue
                w_qm = weights[q, m]
                total += w_pn * w_qm * f_nm_numba(k, n, m, d_comoving, r_hat)
        
        W[i] = total
    
    return W

# ============ UTILISATION ============
# Paramètres
N = 100
k_max = 100
n_k = 200
k = np.linspace(0.001, k_max,n_k)


# Préparer les données pour Numba (format correct)
# Numba préfère les tableaux contigus en mémoire
d_comoving_numba = np.ascontiguousarray(d_comoving[:N])  # shape (N,)
r_hat_numba = np.ascontiguousarray(r_hat[:N].T)  # shape (3, N) - transposé !
weights_numba = np.ascontiguousarray(weights_calc[:, :N])  # shape (9, N)


###########################################################################################################################

# 6. integration of all the term of R :


def R_matrix(N, method): # N is the number of galaxies and method is : 'trapezoid' or 'simpson' or 'quad'
    
    k = np.linspace(0.001, 100, len(Pk_linear))
    R_matrix = np.zeros((9, 9))

    for p in range(9):
        for q in range(9):
            
            start_w = time.time()
            W = window_func_numba(k_linear, p, q, N, weights_numba, d_comoving_numba, r_hat_numba)
            end_w = time.time()
            print(f'The time to calculate the window function for p={p} and q={q} is:',end_w-start_w)
            
            R = W * Pk_linear * OmegaM * H0**2 / (2*np.pi**2)
            result = method(R, k)
            R_matrix[p, q] = result
            
    return R_matrix

# results of the R matrix with N galaxies and the trapèze method:

output_file = '/renoir/fromenti/Documents/codes_Bulk_flow/tests/R_matrices_testtt_klinear.txt'

with open(output_file, "w") as f:

    # ========================
    # Simpson       
    # ========================
    #est la methode la plus précsie et rapide des 2 car la window function est lisse
    
    start_2 = time.time()
    R_matrix_simp = R_matrix(N, method=simpson)
    end_2 = time.time()

    f.write("==== SIMPSON METHOD ====\n")
    f.write(f"N = {N}\n")
    f.write(f"Time = {end_2 - start_2:.4f} seconds\n\n")

    for row in R_matrix_simp:
        f.write("  ".join(f"{x:10.4f}" for x in row) + "\n")

print(f"Results saved in: {output_file}")
