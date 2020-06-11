# IE-MCMC
Importance extruction based on Markov Chain Monte Carlo methods (IE-MCMC)

# Requirements
Python >= 3.7

numpy >= 1.16.2

scikit-learn >= 0.20.3

emcee >= 2.2.1 

matplotlib >= 2.2.3

# Installation
Download or clone the github repository, e.g. git clone https://github.com/rtmr/IE-MCMC

# Usage

## Parameters

**nwalkers: int** (Number of walkers in MCMC sampling)

**nstep: int** (Number of step in MCMC sampling)

**Temp: real** (The value of T in probability distribuion)

## Target dataset

Target dataset is set to the features with label information.
(Target.csv contains the temper designations and composition elements dependence of proof stress in the 5000 series aluminum alloys.)

## Execution
```
python IE-MCMC.py 
```

# License
This project is licensed under the terms of the MIT license.
