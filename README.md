# LambertFit
A Python-based angles-only initial orbit determination tool for solar system objects using a Lambert solver.

## About
LambertFit takes equitorial angular observations in right ascension and declination of solar system objects and finds an orbit that minimizes the observation residuals.  As the name implies, LambertFit leverages the solution to Lambert's problem to fit the orbit.  The conceptual outline of the OD process is as follows:

1. Lambert solve an initial orbit between two observations (*endpoints*: e.g. the first and the last observation) by guessing a number of constant, scalar heliocentric ranges at the two endpoints.
2. Take the Lambert solution for the singular heliocentric range guess that yields the best (i.e. smallest) residuals as your starting orbit.
3. Update each endpoint range estimate, one at a time, by stepping in the direction that reduces the residuals as measured by the RMS error until the orbit residuals stop getting smaller.

## Installation
```
pip install git+https://github.com/bengebre/lambertfit
```

## Example usage (using simulated observations from Horizons)

```python
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

#sun->observer position vector at observation times
r_so = sun_observer(loc,tdbs)

#OD fit from Lambert solutions for RA/DEC observations 
#returns heliocentric vectors in the AU and km/s at the endpoints
rvmf_sb, rvnf_sb, fit_rms = fit(radecs,times,r_so)
```

## Results
The ```rvmf_sb``` variable above yields the orbit on the right.

![download](https://user-images.githubusercontent.com/882036/210093698-9225f7b0-753c-4d20-b5db-ebefd7308ad0.png)

## Caveats, Limitations and Todos

- LambertFit uses two body propagation around the solar system barycenter.  In my testing, trying to solve for observation durations greater than 90 days or so starts to be limited by the fidelity of the orbit propagation.
- LambertFit requires that you choose two observations (endpoints) to solve around - by default the first and last observation.  More than any other observations, the quality of the orbit LambertFit finds is dependent on how good these two endpoint observations are.  The orbit is in a sense 'pinned' at these two endpoint observations and fit to the remaining observations.
- LambertFit is slow.  I'm essentially using the dumbest implementation of gradient descent possible right now.
