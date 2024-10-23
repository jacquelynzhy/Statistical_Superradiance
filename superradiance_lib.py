import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams.update({'font.size': 20})
import  matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import h5py
import scipy.constants as s

#########################
# MODELLING THE BH SPIN #
#########################
def model(m_WR,T,state='c_depletion'): 
    '''modelling the spin of the black hole assuming that it is forming from a WR star (Equation 5)
    INPUT:
        m_WR = mass of the Wolf Rayet star (can be approximanted with the BH)
        T = period in days
        state = c_deplation/He_deplation change slightly the parameters
    OUTPUT:
        spin of the black hole'''
    if state == 'c_depletion':
        c1a = 0.051237
        c2a = 0.029928
        c3a = 0.282998
        c1b = 0.027090
        c2b = 0.010905
        c3b = 0.422213
    elif state == 'he_depletion':
        c1a = 0.059305
        c2a = 0.035552
        c3a = 0.270245
        c1b = 0.026960
        c2b = 0.011001
        c3b = 0.420739
    else:
        raise ValueError('state not supported!')
    
    #a_BH2(T >= 1.) = 0 
    a_BH2 = np.zeros(len(T))
    
    def constant(m_WR, c1, c2, c3):
        return -c1/(c2+np.exp(-c3*m_WR))
    
    alpha = constant(m_WR[T<1.], c1a, c2a, c3a)
    beta = constant(m_WR[T<1.], c1b, c2b, c3b)
    a_BH2[T<1.] = alpha*np.log10(T[T<1.])**2+beta*np.log10(T[T<1.])
    
    return a_BH2

######################
# SUPERRADIANCE RATE #
######################
def GAMMA_322(at,Mb,mu = 1.34e-12,verbose=False):    
    '''
    INPUT:
        at = spin of the black hole
        Mb = mass of the black hole in solar masses
        mu = mass of the boson in electron volts
    OUTPUT:
        GAMMA = superradiance rate'''
    
    import scipy.constants as s
    # scipy constants
    # s.G = 6.6743e-11  # m3 kg-1 s-2
    # s.c = 299792458.0 # m/s
    # s.hbar = 1.0545718176461565e-34 joule seconds
    
    mu = mu * 1.78e-36 #  conversion to from eV to kg
    Mb = Mb * 2e30 # conversion from solar masses to kg
    alpha = (s.G * Mb * mu ) / (s.c * s.hbar)
    
    gamma = s.c**3 * (8 * alpha**13 * (1 + np.sqrt(1 - at**2)) * (-alpha + alpha**3 / 18. + (23 * alpha**5)/1080. - (1.6e10 * alpha**6 * at) / 1.620000000081e12 + at / (1 + np.sqrt(1 - at**2))) * (4 - 4 * at**2 + (2 * at - 2 * alpha * (1 + np.sqrt(1 - at**2)))**2) * (1 - at**2 + (2 * at - 2 * alpha * (1 + np.sqrt(1 - at**2)))**2)) / (885735. * s.G * Mb) 
    
    if verbose:
        print('Boson mass:\t\t %.3e kg (%.3e eV)'%(mu,mu/1.78e-36))
        print('Black Hole mass:\t %.3e kg (%.0f Msun)'%(Mb, Mb/2e30))
        print('Alpha:\t\t\t %.3e '%alpha ) 
        print("Gamma:\t\t\t %.3e"%gamma)
    return gamma

################################
# SUPERRADIANCE RATE 211 STATE #
################################
def GAMMA_211(at,Mb,mu = 1.34e-12,verbose=False):
    '''
    INPUT:
        at = spin of the black hole
        Mb = mass of the black hole in solar masses
        mu = mass of the boson in electron volts
    OUTPUT:
        GAMMA = superradiance rate for 211 state
    ''' 
    
    import scipy.constants as s
    # scipy constants
    # s.G = 6.6743e-11  # m3 kg-1 s-2
    # s.c = 299792458.0 # m/s
    # s.hbar = 1.0545718176461565e-34 joule seconds
    
    mu = mu * 1.78e-36 #  conversion to from eV to kg
    Mb = Mb * 2e30 # conversion from solar masses to kg
    alpha = (s.G * Mb * mu ) / (s.c * s.hbar)
    
    gamma211 = s.c**3 * (alpha**9 * (np.sqrt(1 - at**2) + 1) * ((at - 2 * alpha * (np.sqrt(1 - at**2) + 1))**2 - at**2 + 1) * ((17 * alpha**5) / 64 + alpha**3 / 4 - 2 * alpha + at * (1 / (np.sqrt(1 - at**2) + 1) - (5000000000 * alpha**6) / 30000000003))) / (48 * s.G * Mb)

    if verbose:
        print('Boson mass:\t\t %.3e kg (%.3e eV)'%(mu,mu/1.78e-36))
        print('Black Hole mass:\t %.3e kg (%.0f Msun)'%(Mb, Mb/2e30))
        print('Alpha:\t\t\t %.3e '%alpha ) 
        print("Gamma:\t\t\t %.3e"%gamma211)
        
    return gamma211

################################
# SUPERRADIANCE RATE VARIATION #
################################
def DELTA_GAMMA_322(e,Mb,q,a,mu = 1.34e-12,verbose=False):
    '''
    INPUT:
        e = eccentricity
        Mb = mass of the black hole in solar masses
        q = ratio between lower mass and higher mass 
        a = semimajor axis in astronomical units solar radii
        mu = mass of the boson in electron volts
    OUTPUT:
        DELTA GAMMA = superradiance rate variation 
    '''
    
    # scipy constants
    # G = 6.6743e-11  # m3 kg-1 s-2
    # c = 299792458.0 # m/s
    # hbar = 1.0545718176461565e-34 joule seconds
    
    mu = mu * 1.78e-36 #  conversion to from eV to kg
    Mb = Mb * 2e30 # conversion from solar masses to kg    
    a = a * 696340*10**3 # conversion from solar radii to meters
    #Semi latus rectum 
    p = a * (1 - e**2) # same units of a
    
    alpha = (s.G * Mb * mu ) / (s.c * s.hbar)
    
    delta_gamma =  -s.G**5 * s.c**(-9)* (455625. * (8 + 3*e**2 * (8+e**2))*Mb**5  * q**2 ) / (512. * p**6 *alpha**10) #Checked: correct

    if verbose:
        print('Boson mass:\t\t %.3e kg (%.3e eV)'%(mu,mu/1.78e-36))
        print('Black Hole mass:\t %.3e kg (%.0f Msun)'%(Mb, Mb/2e30))
        print('Alpha:\t\t\t %.3e '%alpha ) 
        print('eccentricity:\t\t %.3e '%e ) 
        print('Mass ratio:\t\t %.3f '%q ) 
        print('Semi-major axis:\t %.3e meters'%a ) 
        print('Semi-latus rectum:\t %.3e meters'%p ) 
        print("Delta Gamma:\t\t %.3e"%delta_gamma)
    
    return delta_gamma



####################################
# SUPERRADIANCE RATE VARIATION ACR #
####################################
def DELTA_GAMMA_ACR_322(e,Mb,q,a,at,mu = 1.34e-12,verbose=False):
    '''
    INPUT:
        e = eccentricity
        Mb = mass of the black hole in solar masses
        q = ratio between lower mass and higher mass 
        a = semimajor axis in solar radii
        mu = mass of the boson in electron volts
        at = spin of the black hole
    OUTPUT:
        DELTA GAMMA = superradiance rate variation ACR
    '''
    # scipy constants
    # G = 6.6743e-11  # m3 kg-1 s-2
    # c = 299792458.0 # m/s
    # hbar = 1.0545718176461565e-34 joule seconds
    
    mu = mu * 1.78e-36 #  conversion to from eV to kg
    Mb = Mb * 2e30 # conversion from solar masses to kg    
    a = a * 696340*10**3 # conversion from solar radii to meters
    #Semi latus rectum 
    p = a * (1 - e**2) # same units of a
    
    alpha = (s.G * Mb * mu ) / (s.c * s.hbar)
    
    # numerator components
    num1 = 5 * (1 + np.sqrt(1 - at**2)) * (8 + 3 * e**2 * (8 + e**2)) * s.G**3 * np.sqrt(Mb**3) * q**2
    num21 = 98415. / (np.sqrt(9 - alpha**2))
    num221 = (2 * alpha**7 * (4 - 4 * at**2 + (2 * at - 2 * (1 + np.sqrt(1 - at**2)) * alpha)**2))
    num222= (1 - at**2 + (2 * at-2 * (1 +np.sqrt(1 - at**2)) * alpha)**2)
    num223 = (-alpha + alpha**3 / 18. + (23 * alpha**5) / 1080 + at * (1 / (1 + np.sqrt(1 - at**2)) - (16e9 * alpha**6) / (1620000000081) ) ) 
    #denumerator component 
    den=5832* np.sqrt(1/(Mb- e**2 * Mb))**3 * p**6 *(((s.c**3 * 4 * s.G * Mb * (1+q))/p**3)+((s.c**9 * 64 * alpha**10 * (180000000009+10000000000 * at * alpha)**2)/(65610000006561000000164025 * s.G**2 * Mb**2))) 

    #DELTA_GAMMA_ACR = -num1*(num21+num221*num222*num223)/(den * c**5)  #Now correct
    delta_gamma_acr = -num1*(num21+num221*num222*num223)/den
    
    if verbose:
        print('Boson mass:\t\t %.3e kg (%.3e eV)'%(mu,mu/1.78e-36))
        print('Black Hole mass:\t %.3e kg (%.0f Msun)'%(Mb, Mb/2e30))
        print('Alpha:\t\t\t %.3e '%alpha ) 
        print('eccentricity:\t\t %.3e '%e ) 
        print('Mass ratio:\t\t %.3f '%q ) 
        print('Semi-major axis:\t %.3e meters'%a ) 
        print('Semi-latus rectum:\t %.3e meters'%p ) 
        print("Delta Gamma:\t\t %.3e"%delta_gamma_acr)
    
    return delta_gamma_acr
    
    
########################################
# SUPERRADIANCE RATE VARIATION ACR 211 #
########################################
def DELTA_GAMMA_ACR_211(e,Mb,q,a,at,mu = 1.34e-12,verbose=False):
    '''
    INPUT:
        e = eccentricity
        Mb = mass of the black hole in solar masses
        q = ratio between lower mass and higher mass 
        a = semimajor axis in solar radii
        mu = mass of the boson in electron volts
        at = spin of the black hole
    OUTPUT:
        DELTA GAMMA = superradiance rate variation ACR  211 for 211 to 21-1 
    '''
    
    # scipy constants
    # G = 6.6743e-11  # m3 kg-1 s-2
    # c = 299792458.0 # m/s
    # hbar = 1.0545718176461565e-34 joule seconds
    
    mu = mu * 1.78e-36 #  conversion to from eV to kg
    Mb = Mb * 2e30 # conversion from solar masses to kg    
    a = a * 696340*10**3 # conversion from solar radii to meters
    #Semi latus rectum 
    p = a * (1 - e**2) # same units of a
    
    alpha = (s.G * Mb * mu ) / (s.c * s.hbar)


    #ACR for 211 to 21-1 
    numm1=(810000000081 * alpha**3 * at * (3 * (e**2 + 8) * e**2 + 8) * s.G**5 * (Mb**3)**(3/2) * q**2)
    numm21=(-2 * (640000000000 * alpha**6 + 550000000051 * alpha**4 + 480000000048 * alpha**2 - 5760000000576) * (np.sqrt(1 - at**2) + 1))
    numm22=at**2 * (510000000051 * alpha**4 + 480000000048 * alpha**2 + 320000000000 * alpha**6 * (np.sqrt(1 - at**2) + 3) - 5760000000576)
    denn=4096 * p**3 * np.sqrt(-1 / ((e**2 - 1)**3 * Mb)) * (6250000000000000000 * alpha**12 * at**2 * p**3 * s.c**9 + 900000000180000000009 * s.G**3 * Mb**3 * (q + 1) * s.c**3)

    delta_gamma_acr211=-(numm1*( 480000000048 + alpha**2 * (numm21+numm22)))/(denn)
    
    if verbose:
        print('Boson mass:\t\t %.3e kg (%.3e eV)'%(mu,mu/1.78e-36))
        print('Black Hole mass:\t %.3e kg (%.0f Msun)'%(Mb, Mb/2e30))
        print('Alpha:\t\t\t %.3e '%alpha ) 
        print('eccentricity:\t\t %.3e '%e ) 
        print('Mass ratio:\t\t %.3f '%q ) 
        print('Semi-major axis:\t %.3e meters'%a ) 
        print('Semi-latus rectum:\t %.3e meters'%p ) 
        print("Delta Gamma:\t\t %.3e"%delta_gamma_acr211)
    
    return delta_gamma_acr211


def DELTA_GAMMA_OVER_GAMMA_322(at,Mb,q,mu,e,a,verbose=False):
    '''Ratio between DELTA_GAMMA and GAMMA
    INPUT:
        e = eccentricity
        Mb = mass of the black hole in solar masses
        q = ratio between lower mass and higher mass 
        a = semimajor axis in solar radii
        mu = mass of the boson in electron volts
        at = spin of the black hole
    OUTPUT:
        DELTA GAMMA / GAMMA  
    '''
    gamma = GAMMA_322(at,Mb,mu,verbose=verbose)
    delta_gamma = DELTA_GAMMA_322(e,Mb,q,a,mu,verbose=verbose)
    delta_gamma_over_gamma = delta_gamma/gamma
    
    
    return delta_gamma_over_gamma

def DELTA_GAMMA_ACR_OVER_GAMMA_322(at,Mb,q,mu,e,a,verbose=False):
    '''Ratio between DELTA_GAMMA_ACR_322 and GAMMA_ACR_322
    INPUT:
        e = eccentricity
        Mb = mass of the black hole in solar masses
        q = ratio between lower mass and higher mass 
        a = semimajor axis in solar radii
        mu = mass of the boson in electron volts
        at = spin of the black hole
    OUTPUT:
        DELTA GAMMA_ACR_322 / GAMMA_ACR_322 
    '''
    gamma322 = GAMMA_322(at,Mb,mu,verbose=verbose)
    delta_gamma_acr322 = DELTA_GAMMA_ACR_322(e,Mb,q,a,mu,verbose=verbose)
    delta_gamma_over_gamma_acr322 = delta_gamma_acr322/gamma322
    return delta_gamma_over_gamma_acr322
    
    
def DELTA_GAMMA_ACR_OVER_GAMMA211(at,Mb,q,mu,e,a,verbose=False):
    '''Ratio between DELTA_GAMMA_ACR_211 and GAMMA_ACR_211
    INPUT:
        e = eccentricity
        Mb = mass of the black hole in solar masses
        q = ratio between lower mass and higher mass 
        a = semimajor axis in solar radii
        mu = mass of the boson in electron volts
        at = spin of the black hole
    OUTPUT:
        DELTA GAMMA_ACR_211 / GAMMA_ACR_211  
    '''
    gamma211 = GAMMA_211(at,Mb,mu,verbose=verbose)
    delta_gamma_acr211 = DELTA_GAMMA_ACR_211(e,Mb,q,a,mu,verbose=verbose)
    delta_gamma_over_gamma_acr211 = delta_gamma_acr211/gamma211
    return delta_gamma_over_gamma_acr211
    
