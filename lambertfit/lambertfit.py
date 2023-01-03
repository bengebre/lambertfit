import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA
from astroquery.jplhorizons import Horizons
from astropy import units as u
from astropy.time import Time
from poliastro import __version__ as poli_version
#from poliastro.twobody.propagation import cowell
from poliastro.frames import Planes
from poliastro.iod import izzo
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter

def angle_between(u1,u2):
    """
    Calculates angle between two unit vectors in radians.
    
    Parameters
    ----------
    u1 : array-like
        unit vector 1.
    u2 : array-like
        unit vector 2.
        
    Returns
    -------
    float
        Angle between u1 and u2 in radians.
    """
    
    return np.arcsin(LA.norm(np.cross(u1,u2)))

def rmse(arr):
    """
    Calculates the RMS error of an array of residuals.
    
    Parameters
    ----------
    arr : numpy.ndarray
        array of residuals.
        
    Returns
    -------
    float
        RMS error of residuals.
    """
    return np.sqrt(np.mean(arr**2))

def circular_lfit(radecs,times,r_so,pts=[0,-1],rgs=[],max_ecc=0.1):
    """
    Calculate the circular(ish) orbit that has the lowest RMS error for the observations.  Used as an initial guess 
    for Lambert fitting.
    
    Parameters
    ----------
    radecs : numpy.ndarray
        Nx2 array of observations in right ascension and declination degrees.
    times : array-like
        N length array of observation times.
    r_so : numpy.ndarray
        Nx3 array of position vectors in the equitorial plane pointing from the Sun to the observer location.
    pts : array-like, optional
        The two indicies of the observation enpoints to fit around [m,n].
    rgs : array-like, optional
        Range guesses for the fit in AU.
    max_ecc : float, optional
        Maximum eccentricity to allow in Lambert fit.
        
    Returns
    -------
    numpy.ndarray
        Position and velocity for the circular orbit at time 'm'.
    numpy.ndarray
        Position and velocity for the circular orbit at time 'n'.
    float
        The RMS error of the fit in arc seconds.
    """
    
    #some definitions and inits
    aukm = 149597870.7 #km/1AU
    rmses = []
    rv0s = []
    rv1s = []
    
    if len(rgs)==0:
        rgs = np.concatenate((np.arange(1.1,6,0.1),np.arange(6,101,1)))
    
    #observer->object equitorial unit vectors from radecs
    ru_ob = eq2cart(radecs[:,0],radecs[:,1]) 
    
    #endpoints
    m = pts[0]
    n = pts[1]
    dt0n = times[n]-times[m]
    
    for rg in rgs:
        r0 = r_so[m,:] + geo2helio(r_so[m,:],ru_ob[m,:],rg)*ru_ob[m,:]
        r1 = r_so[n,:] + geo2helio(r_so[n,:],ru_ob[n,:],rg)*ru_ob[n,:]
        
        if np.any(np.isnan(r0)) or np.any(np.isnan(r1)):
            rmses.append(1e10)
            rv0s.append([0,0,0,0,0,0])
            rv1s.append([0,0,0,0,0,0])
            continue
        
        v0, vn = lambert(Sun.k, r0*u.AU, r1*u.AU, dt0n*u.day, rtol=1e-8, numiter=50)
        orb = Orbit.from_vectors(Sun, r0*u.AU, v0, Time(times[m],format='jd'))
        
        if orb.ecc > max_ecc:
            rmses.append(1e10)
            rv0s.append([0,0,0,0,0,0])
            rv1s.append([0,0,0,0,0,0])
            continue

        rv0 = np.concatenate((r0,v0.value))
        rv1 = np.concatenate((r1,vn.value))

        try:
            rms = 3600*rmse(calc_residuals(rv1,times[n],ru_ob,times,r_so)) #*orb.ecc.value**2
            #print(rg,rms,orb.ecc,rms*orb.ecc**2)

            rmses.append(rms)
            rv0s.append(rv0)
            rv1s.append(rv1)
        except:            
            rmses.append(1e10)
            rv0s.append([0,0,0,0,0,0])
            rv1s.append([0,0,0,0,0,0])
            continue
            
    min_idx = np.nanargmin(rmses)
    rmses = np.array(rmses)
    rv0s = np.array(rv0s)
    rv1s = np.array(rv1s)
        
    return (rv0s[min_idx],rv1s[min_idx],rmses[min_idx])

def vec2unit(v):
    """
    Convert a vector (or vectors) into a unit vector(s).
    
    Parameters
    ----------
    v : numpy.ndarray
        Either a 1 or 2 dimensional array.
        
    Returns
    -------
    numpy.ndarray
        An array of the same shape as the input with length equal to 1.
    """
    if len(np.shape(v))==1:
        return v/LA.norm(v)
    else:
        return v/LA.norm(v,axis=1)[:,None]

def calc_residuals(rv,time,ru_ob,times,r_so):
    """
    Calculate the residuals for an orbit [state] against a set of observations.
    
    Parameters
    ----------
    rv : numpy.ndarray
        Heliocentric position [AU] and velocity [km/s] of an orbit to calculate the residuals of.
    time : float
        Julian date of the position and velocity.
    ru_ob : numpy.ndarray
        Nx3 array of unit vectors pointing from observer->body.
    times : array-like
        N length array of observation times.
    r_so : numpy.ndarray
        Nx3 array of position vectors in the equitorial plane pointing from the Sun to the observer location.

    Returns
    -------
    numpy.ndarray
        An array or residuals in degrees with the same length as the number of observations.
    """
    
    orb = Orbit.from_vectors(Sun, rv[0:3]*u.AU, rv[3:6]*(u.km/u.s), Time(time,format='jd'))
    
    residuals = []
    for i,ot in enumerate(times):
        dt = ot - time
        porb = orb.propagate(dt*u.day)
        
        r = (porb.r.to('AU')).value
        pr = r - r_so[i,:]
        pru = pr/LA.norm(pr)
            
        residuals.append(angle_between(ru_ob[i,:],pru))
          
    return np.rad2deg(residuals)

def lambert(*args, **kwargs):
    """
    Wraps Poliastro's izzo.lambert() to handle an interface change.
    """
    if poli_version=='0.16.0':
        (v0, vn), = izzo.lambert(*args, **kwargs)
    else:
        (v0, vn) = izzo.lambert(*args, **kwargs)

    return v0, vn

def fit(radecs,times,r_so,pts=[0,-1],rg=None,tol=1e-5,max_iter=100000,min_rms=0.001):
    """
    Fit an orbit to a set of observations in RA and DEC (radec) at times (times) and observer positions (r_so)
    by using a Lambert solver to minimize the RMS error.

    Parameters
    ----------
    radecs : numpy.ndarray
        Nx2 array of observations in right ascension and declination degrees.
    times : array-like
        N length array of observation times.
    r_so : numpy.ndarray
        Nx3 array of position vectors in the equitorial plane pointing from the Sun to the observer location.
    pts : array-like, optional
        The two indicies of the observation endpoints to fit around [m,n].
    rg : float, optional
        Initial range guess in AU.
    tol : float, optional
        Stop when RMS delta between checks is smaller than this (arc seconds).
    max_iter : int, optional
        Stop when fit iterations exceed this count.
    min_rms : float, optional
        Stop when an RMS better than this is achieved (arc seconds).

    Returns
    -------
    numpy.ndarray
        Position and velocity for the circular orbit at time 'm'.
    numpy.ndarray
        Position and velocity for the circular orbit at time 'n'.
    float
        The RMS error of the fit in arc seconds.
    """

    #solve time
    fit_start = time.time()

    #define
    aukm = 149597870.7 #km/1AU
    i = 0 #iteration
    last_rms = 1e10 #prior RMS measured
    stopping_step = 100 #km
    sign0 = [1,-1] #step0 direction
    signn = [1,-1] #stepn direction

    if rg==None:
        #guess best fit circular(ish) starting orbit
        max_ecc = 0.1
        rvg_rms = 1e5
        while rvg_rms >= 1e5 and max_ecc <= 0.9:
            rvmg_sb, rvng_sb, rvg_rms = circular_lfit(radecs,times,r_so,pts=pts,max_ecc=max_ecc)
            max_ecc+=0.1
        start_range = LA.norm(rvng_sb[0:3])
    else:
        start_range = rg

    #observer->body relative unit vectors
    ru_ob = eq2cart(radecs[:,0],radecs[:,1],1)

    #endpoints
    m = pts[0]
    n = pts[1]

    #get initial observer ranges from geo2helio
    obs_range0 = geo2helio(r_so[m,:],ru_ob[m,:],start_range)
    obs_rangen = geo2helio(r_so[n,:],ru_ob[n,:],start_range)

    #initial heliocentric range vectors
    r0g = r_so[m,:] + obs_range0*ru_ob[m,:]
    rng = r_so[n,:] + obs_rangen*ru_ob[n,:]

    #initial Lambert solution to connect endpoints to pointings [not connected with circular guess]
    dt0n = times[n]-times[m]
    v0, vn = lambert(Sun.k, r0g*u.AU, rng*u.AU, dt0n*u.day, rtol=1e-8, numiter=50)

    #starting guess orbit [will be slightly different from circular because using geo2helio to point]
    lrv0 = np.concatenate((r0g,v0.value)) #lambert rv at t=0
    current_rms = rmse(calc_residuals(lrv0,times[m],ru_ob,times,r_so))*3600

    #how much to vary range guess (step size)
    step = 1e-2 #AU
    last_step = step

    #init no improvement counter
    ni = 0

    #LambertFit
    while (step*aukm > stopping_step):

        for j,s in enumerate(sign0):
            #Observation 0 (aka observation m)
            lr0 = r_so[m,:] + geo2helio(r_so[m,:],ru_ob[m,:],LA.norm(r0g)-step*s)*ru_ob[m,:]
            lv0, vn = lambert(Sun.k, lr0*u.AU, rng*u.AU, dt0n*u.day, rtol=1e-8, numiter=50)
            lrv0 = np.concatenate((lr0,lv0.value))
            em = rmse(calc_residuals(lrv0,times[m],ru_ob,times,r_so))*3600

            if em < current_rms:
                ni = 0
                current_rms = em
                r0g = lrv0[0:3]
                if j==1:
                    sign0 = np.flip(sign0)
                break
            elif j==1:
                ni += 1 #no improvement

        for j,s in enumerate(signn):
            #Observation n
            lrn = r_so[n,:] + geo2helio(r_so[n,:],ru_ob[n,:],LA.norm(rng)-step*s)*ru_ob[n,:]
            v0, lvn = lambert(Sun.k, r0g*u.AU, lrn*u.AU, dt0n*u.day, rtol=1e-8, numiter=50)
            lrvn = np.concatenate((lrn,lvn.value))
            em = rmse(calc_residuals(lrvn,times[n],ru_ob,times,r_so))*3600

            if em < current_rms:
                ni = 0
                current_rms = em
                rng = lrvn[0:3]
                if j==1:
                    signn = np.flip(signn)
                break
            elif j==1:
                ni+=1 #no improvement

        if ni>=2:
            last_step = step
            step = 0.5*step

        #break out of loop conditions; try to increase step size
        if i%10==0:

            ts = (time.time() - fit_start)
            print("RMS: {:.5f} arcsec; Step: {:,.0f} km; {:.2f}s          ".format(current_rms,step*aukm,ts),end='\r')

            if (i >= max_iter):
                print("\nStopped: {:d} iterations reached".format(i),end='\r')
                break

            if (current_rms <= min_rms):
                print("\nStopped: RMS minimum achieved",end='\r')
                break

            if np.abs(last_rms-current_rms) <= tol:
                print("\nStopped: tolerance achieved",end='\r')
                break

            if step < last_step:
                step = 1.1*step

            last_rms = current_rms

        #iteration counter
        i+=1

    #final orbit positions and velocities
    v0, vn = lambert(Sun.k, r0g*u.AU, rng*u.AU, dt0n*u.day, rtol=1e-8, numiter=50)
    rv0g = np.concatenate((r0g,v0.value))
    rvng = np.concatenate((rng,vn.value))

    print("RMS: {:.5f} arcsec; Step: {:,.0f} km; {:.2f}s; Done          ".format(current_rms,step*aukm,ts))

    return (rv0g,rvng,current_rms)

def sun_observer(obs_id,times,body_id='299'):
    """
    Get the vectors pointing from the Solar System Barycenter to the observer location 
    at the specified times.
    
    Parameters
    ----------
    obs_id : str
        The MPC code of the observatory.
    times : float
        Observation times (Julian dates).
    body_id : str
        JPL body id to use to calculate positon against. (e.g. '299' is Venus)
    
    Returns
    -------
    numpy.ndarray
        Heliocentric position of the observer at the observation times.
    
    """
    
    for i in np.arange(0,len(times),25):
        hv = Horizons(id=body_id,location='500@0',epochs=times[i:i+25],id_type='id').vectors(aberrations='astrometric',refplane='earth')
        if i==0:
            r_sb = np.asarray([[h['x'],h['y'],h['z']] for h in hv])
        else:
            r_sb = np.vstack((r_sb,np.asarray([[h['x'],h['y'],h['z']] for h in hv])))

    for i in np.arange(0,len(times),25):
        hv = Horizons(id=body_id,location=obs_id,epochs=times[i:i+25],id_type='id').vectors(aberrations='astrometric',refplane='earth')
        if i==0:
            r_ob = np.asarray([[h['x'],h['y'],h['z']] for h in hv])
        else:
            r_ob = np.vstack((r_ob,np.asarray([[h['x'],h['y'],h['z']] for h in hv])))
    
    r_so = r_sb-r_ob
    
    return r_so

def eq2cart(ra,dec,d=1):
    """
    Equitorial spherical coordinates (RA/DEC) to equitorial vectors.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    def : float
        Declination in degrees.
    d : float, optional
        Magnitude of vector to return.

    Returns
    -------
    numpy.ndarray
        Equitorial position vectors scaled to 'd'

    """
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    x = d*np.cos(ra_rad)*np.cos(dec_rad)
    y = d*np.sin(ra_rad)*np.cos(dec_rad)
    z = d*np.sin(dec_rad)

    eq_cart = np.column_stack((x,y,z))

    return eq_cart

def geo2helio(r_so,rhat_ob,d):
    """
    Return the scalar multiplier for 'rhat_ob' that yields an 'r_so' of length 'd'
    
    Parameters
    ----------
    r_so : numpy.ndarray
        Nx3 array of position vectors in the equitorial plane pointing from the Sun to the observer location.
    rhat_ob : numpy.ndarray
        Nx3 unit vector pointing from observer to body observed.
    d : float
        Heliocentric distance or range to assert.

    Returns
    -------
    float
        The scalar multiplier for 'rhat_ob' that yields a heliocentric vector of length 'd'.
    """

    is_arr = isinstance(rhat_ob,np.ndarray) and rhat_ob.ndim==2

    if is_arr:
        b = 2*np.sum(r_so*rhat_ob,axis=1)
        c = -1*(d**2 - np.sum(r_so*r_so,axis=1))
    else:
        b = 2*np.dot(rhat_ob,r_so)
        c = -1*(d**2 - np.dot(r_so,r_so))

    s = np.sqrt(b**2 - 4*c)
    r0 = (-b + s)/2.0
    r1 = (-b - s)/2.0

    if is_arr:
        alpha = np.max(np.vstack((r0,r1)).T,axis=1)
        alpha[alpha<0] = np.nan
    else:
        alpha = np.max((r0,r1))
        if alpha<0:
            alpha = np.nan

    return alpha

