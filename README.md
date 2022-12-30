# LambertFit
A Python-based angles-only initial orbit determination tool for solar system objects using a Lambert solver.

## About
LambertFit takes equitorial angular observations of solar system objects in right ascension and declination and finds an orbit that minimizes the  residuals.  As the name implies, LambertFit leverages the solution to Lambert's problem to fit the orbit.  The conceptual outline of the OD process is as follows:

1. Lambert solve an initial orbit between two observations (*endpoints*: e.g. the first and the last observation) by guessing a number of constant, scalar heliocentric ranges at the two endpoints.
2. Take the Lambert solution for the singular heliocentric range guess that yields the best (i.e. smallest) residuals as your starting orbit.
3. Update each endpoint range estimate, one at a time, by stepping in the direction that reduces the residuals as measured by the RMS error until the orbit residuals stop getting smaller.

## Installation
```
pip install git+https://github.com/bengebre/lambertfit
```

## Example usage

#### Getting some simulated RA/DEC observations from Horizons
```python
import numpy as np
import lambertfit as lf
from astroquery.jplhorizons import Horizons
from astropy.time import Time

#define observer location and body to observe
loc = 'G96' #MPC observatory location code (eg. 'G96' or '500@0')
body_id = '2000002' #small body designation (e.g. '2000002' == Pallas)

#how many (simulated) observations do we want and at what time
tstart = 2459000.5 #julian start date
dt = 14 #number of days of observations
nobs = 30 #number of observations

#observation times
times = np.linspace(tstart,tstart+dt,nobs) #observation times
tdbs = Time(times,format='jd').tdb.value #barycenter dynamical times

#get simulated observation data from Horizons
ephs = Horizons(id=body_id,location=loc,epochs=times,id_type='designation').ephemerides()
radecs = np.array([[eph['RA'],eph['DEC']] for eph in ephs])
```

#### Orbit Determination via LambertFit
```python
#sun->observer position vector at observation times
r_so = lf.sun_observer(loc,tdbs)

#OD fit from Lambert solutions for RA/DEC observations 
#returns heliocentric vectors in AU and km/s at the endpoints (defaults to first and last observations)
rvmf_sb, rvnf_sb, fit_rms = lf.fit(radecs,times,r_so)
```

## Results
The blue orbit is the earth.  The The green orbit is the true orbit of the body we're trying to find the orbit for.  The orange orbit is the LambertFit solution.  The reported RMS errors are in arc seconds.  The ```rvmf_sb``` variable above yields the LambertFit orbit on the right.  The left orbit is the initial guess orbit that LambertFit starts with and refines.  The observations are equally spaced (in this instance) between the diamond and the circle.

![download](https://user-images.githubusercontent.com/882036/210093698-9225f7b0-753c-4d20-b5db-ebefd7308ad0.png)

## Caveats, Limitations and Todos

- LambertFit uses two body propagation around the solar system barycenter.  In my testing, trying to solve for observation durations greater than 90 days or so starts to be limited by the fidelity of the orbit propagator.
- LambertFit requires that you choose two observations (endpoints) to solve around - by default the first and last observation.  More than any other observations, the quality of the orbit LambertFit finds is dependent on how good these two endpoint observations are.  The orbit is in a sense 'pinned' at these two endpoint observations and fit to the remaining observations.
- LambertFit is slow.  I'm essentially using the dumbest implementation of gradient descent possible right now.

## Acknowledgements
- LambertFit uses [Poliastro](https://github.com/poliastro/poliastro) for both orbit propagation and for Lambert solving ([Izzo, 2014](https://arxiv.org/abs/1403.2705)).  It's an excellent astrodynamics package.
- While my approach is not the same as the authors, I did find the paper [Initial orbit determination methods for track-to-track association](https://www.sciencedirect.com/science/article/pii/S0273117721005287#n0010) (Pastor et. al, 2021) very helpful for thinking about how to use a Lambert solver for initial orbit determination.
