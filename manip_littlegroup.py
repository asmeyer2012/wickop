"""
    Copyright 2019 Aaron S Meyer

    This file is part of ``wickop''.

    ``wickop'' is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ``wickop'' is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ``wickop''.  If not, see <https://www.gnu.org/licenses/>.

"""
import numpy as np

from defines import *
from manip_array import *
from manip_group import *
from manip_momentum import *

## momentum (littlegroup) specific extract_generators routines
def extract_generators_100(rep):
  return [rep[2],rep[-4]] ## r12, r13.r13.pp
def extract_generators_110(rep):
  return [rep[2],rep[-2]] ## r21.r13.r13, r12.r12.pp
def extract_generators_111(rep):
  #return [rep[4],rep[-6]] ## r21.r13.r13, r12.r12.pp
  return [rep[4],rep[6]] ## r21.r13.r13, r12.r12.pp
def extract_generators_xyz(rep): ## all higher momenta
  return [rep[1]]

## from a full list of representation matrices in little group, extract only the generators
## this depends on how indices in character_tables are defined!
def extract_generators_littlegroup(rep,pref):
  tp = classify_momentum_type(pref)
  if   tp == '000':
    return extract_generators(rep)
  elif tp == '100':
    return extract_generators_100(rep)
  elif tp == '110':
    return extract_generators_110(rep)
  elif tp == '111':
    return extract_generators_111(rep)
  elif tp == '210' or tp == '211' or tp == '321':
    return extract_generators_xyz(rep)
  else:
    raise ValueError("Undefined momentum type!")

### for an input projected representation matrix and requested eigenvalue,
###  get eigenvectors in matrix subspace
#def construct_littlegroup_evec(proj,el):
#  if all_zero(proj):
#    return [] ## projection matrix is zero
#  ev = eig_subspace(proj,el) ## get COLUMN eigenvectors with that eigenvalue, reduced form
#  return ev

## from little group COLUMN evecs, expand to evecs of full representation matrix
## sort momentum list, expand in simpler list, then undo sorting
def expand_littlegroup_evec(pl,pref,ev):
  if len(ev) == 0:
    return ev ## nothing to see here, move along
  pref = reference_momentum(pref)
  perm = momentum_permutation_block(pl)
  plx = full_momentum_sort(pl)
  i0 = plx.index(pref)
  i1 = len(plx) - plx[::-1].index(pref)
  slc = slice(i0,i1)
  evx = np.zeros((len(pl),ev.shape[1]),dtype=complex)
  evx[slc] = ev
  return herm(perm).dot(evx)

## helper function: compute the correction phase between two eigenvectors given rep matrix
## will give val after correction
def inverse_phase(x,y,gen,val):
 return np.exp(-I*np.angle((herm(x).dot(gen).dot(y))[0,0]/val))

## use COLUMN evecs
## use some generator or the group symmetries to collect the evecs that belong in the same irrep
## return the reordered v1, paired so it matches v0 ordering
def collect_evec(pgen,v0,v1):
 v1 = orthogonalize( v1)
 v1x = []
 for v in herm(v0):
   vnew = normalize( v.dot( pgen).dot( v1).dot( herm( v1)))
   v1x.append( vnew)
 return herm( np.array(v1x))

## zero momentum need index i to distinguish positive and negative parity
def construct_lgevec_e000(pgen,i):
 ev0 = eig_subspace(pgen[0], 1.) ## R12
 ev1 = eig_subspace(pgen[0],-1.)
 ev1 = collect_evec(pgen[1],ev0,ev1) ## match up the eigenvectors with each other
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])] ## separate columns
 ev1 = [ev1[:,i:i+1] for i in range(ev1.shape[1])]
 ## fix phases between ev0 and ev1
 ev1 = [y* inverse_phase(x,y,pgen[1],-np.sqrt(3.)/2.) for x,y in zip(ev0,ev1)]
 ev = [chop(np.hstack([x,y])) for x,y in zip(ev0,ev1)] ## stack
 return ev
def construct_lgevec_t000(pgen,i):
 rsgn = (-1. if (i == 4 or i == 12) else 1.)
 ev0 = eig_subspace(pgen[1],rsgn) ## R23
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])]
 ## use rotations to construct the other elements, easier
 ev1 = [rsgn*pgen[0].dot(x) for x in ev0] ## R12 x -> y
 ev2 = [rsgn*pgen[1].dot(x) for x in ev1] ## R23 y -> z
 ev = [chop(np.hstack([x,y,z])) for x,y,z in zip(ev0,ev1,ev2)]
 return ev
def construct_lgevec_g000(pgen,i):
 rsgn = (-1. if (i == 6 or i == 14) else 1.)
 ev0 = eig_subspace(pgen[0],rsgn*(1.-I)/sr2) ## R12
 ev1 = eig_subspace(pgen[0],rsgn*(1.+I)/sr2)
 ev1 = collect_evec(pgen[1],ev0,ev1)
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])]
 ev1 = [ev1[:,i:i+1] for i in range(ev1.shape[1])]
 ev1 = [y* inverse_phase(x,y,pgen[1],-rsgn*I/sr2) for x,y in zip(ev0,ev1)]
 ev = [chop(np.hstack([x,y])) for x,y in zip(ev0,ev1)]
 return ev
def construct_lgevec_h000(pgen,i):
 ev0 = eig_subspace(pgen[0], (1.-I)/sr2) ## R12
 ev1 = eig_subspace(pgen[0], (1.+I)/sr2)
 ev2 = eig_subspace(pgen[0],-(1.-I)/sr2)
 ev3 = eig_subspace(pgen[0],-(1.+I)/sr2)
 ev1 = collect_evec(pgen[1],ev0,ev1)
 ev2 = collect_evec(pgen[1],ev0,ev2)
 ev3 = collect_evec(pgen[1],ev0,ev3)
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])]
 ev1 = [ev1[:,i:i+1] for i in range(ev1.shape[1])]
 ev2 = [ev2[:,i:i+1] for i in range(ev2.shape[1])]
 ev3 = [ev3[:,i:i+1] for i in range(ev3.shape[1])]
 ev1 = [y* inverse_phase(x,y,pgen[1],I*np.sqrt(1./8.)) for x,y in zip(ev0,ev1)]
 ev2 = [y* inverse_phase(x,y,pgen[1],  np.sqrt(3./8.)) for x,y in zip(ev0,ev2)]
 ev3 = [y* inverse_phase(x,y,pgen[1],I*np.sqrt(3./8.)) for x,y in zip(ev0,ev3)]
 ev = [chop(np.hstack([w,x,y,z])) for w,x,y,z in zip(ev0,ev1,ev2,ev3)]
 return ev
def construct_lgevec_b100(pgen):
 ev0 = eig_subspace(pgen[0],I) ## R12
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])]
 ev1 = [pgen[1].dot(x) for x in ev0] ## simple rotation
 ev = [chop(np.hstack([x,y])) for x,y in zip(ev0,ev1)]
 return ev
def construct_lgevec_g100(pgen,i):
 rsgn = (-1. if i == 6 else 1.)
 ev0 = eig_subspace(pgen[0],rsgn*(1.-I)/sr2) ## R12
 ev1 = eig_subspace(pgen[0],rsgn*(1.+I)/sr2)
 ev1 = collect_evec(pgen[1],ev0,ev1)
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])]
 ev1 = [ev1[:,i:i+1] for i in range(ev1.shape[1])]
 ev1 = [y* inverse_phase(x,y,pgen[1],1.) for x,y in zip(ev0,ev1)] ## R13.R13.P
 ev = [chop(np.hstack([x,y])) for x,y in zip(ev0,ev1)]
 return ev
def construct_lgevec_g110(pgen):
 ev0 = eig_subspace(pgen[1],-I) ## R12.R12.P
 ev1 = eig_subspace(pgen[1], I)
 ev1 = collect_evec(pgen[0],ev0,ev1)
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])]
 ev1 = [ev1[:,i:i+1] for i in range(ev1.shape[1])]
 ev1 = [y* inverse_phase(x,y,pgen[0],(1.-I)/sr2) for x,y in zip(ev0,ev1)] ## R21.R13.R13
 ev = [chop(np.hstack([x,y])) for x,y in zip(ev0,ev1)]
 return ev
def construct_lgevec_b111(pgen):
 ev0 = eig_subspace(pgen[0],0.5*(-1.+I*np.sqrt(3.))) ## R13.R12
 ev1 = eig_subspace(pgen[0],0.5*(-1.-I*np.sqrt(3.)))
 ev1 = collect_evec(pgen[1],ev0,ev1)
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])]
 ev1 = [ev1[:,i:i+1] for i in range(ev1.shape[1])]
 ev1 = [y* inverse_phase(x,y,pgen[1],1.) for x,y in zip(ev0,ev1)] ## R12.R13.R13.P
 ev = [chop(np.hstack([x,y])) for x,y in zip(ev0,ev1)]
 return ev
def construct_lgevec_g111(pgen):
 ev0 = eig_subspace(pgen[1],-I) ## R12.R13.R13.P
 ev1 = eig_subspace(pgen[1], I)
 ev1 = collect_evec(pgen[0],ev0,ev1)
 ev0 = [ev0[:,i:i+1] for i in range(ev0.shape[1])]
 ev1 = [ev1[:,i:i+1] for i in range(ev1.shape[1])]
 ev1 = [y* inverse_phase(x,y,pgen[0],(1.-I)/2.) for x,y in zip(ev0,ev1)] ## R13.R12
 ev = [chop(np.hstack([x,y])) for x,y in zip(ev0,ev1)]
 return ev

## 1-dimensionals: eigenvectors of projection are all that are needed
def construct_littlegroup_evec_xyz(pj):
 ev = eig_subspace(pj,1.)
 ev = [ev[:,i:i+1] for i in range(ev.shape[1])]
 return ev

def construct_littlegroup_evec_000(pgen,pref,i):
  if   i == 2 or i == 10:
    return construct_lgevec_e000(pgen,i)
  elif i == 3 or i == 4 or i == 11 or i == 12:
    return construct_lgevec_t000(pgen,i)
  elif i == 5 or i == 6 or i == 13 or i == 14:
    return construct_lgevec_g000(pgen,i)
  elif i == 7 or i == 15:
    return construct_lgevec_h000(pgen,i)
  else:
    raise ValueError("unknown index: "+str(i))
def construct_littlegroup_evec_100(pgen,pref,i):
  if   i == 4:
    return construct_lgevec_b100(pgen)
  elif i == 5 or i == 6:
    return construct_lgevec_g100(pgen,i) ## need i to distinguish G1 and G2
  else:
    raise ValueError("unknown index: "+str(i))
def construct_littlegroup_evec_110(pgen,pref,i):
  if   i == 4:
    return construct_lgevec_g110(pgen)
  else:
    raise ValueError("unknown index: "+str(i))
def construct_littlegroup_evec_111(pgen,pref,i):
  if   i == 2:
    return construct_lgevec_b111(pgen)
  elif i == 5:
    return construct_lgevec_g111(pgen)
  else:
    raise ValueError("unknown index: "+str(i))

## for input representation generators, reference momentum, projection matrix, and irrep index,
##  get eigenvectors in matrix subspace
## 
def construct_littlegroup_evec(gen,pref,pj,i):
  pref = reference_momentum(pref)
  pt = classify_momentum_type(pref)
  pgen = [x.dot(pj) for x in gen] ## project out the subspace
  if   pt == '000':
    if i < 2 or i == 8 or i == 9: ## postive and negative parity
      return construct_littlegroup_evec_xyz(pj)
    return construct_littlegroup_evec_000(pgen,pref,i)
  elif pt == '100':
    if i < 4:
      return construct_littlegroup_evec_xyz(pj)
    return construct_littlegroup_evec_100(pgen,pref,i)
  elif pt == '110':
    if i < 4:
      return construct_littlegroup_evec_xyz(pj)
    return construct_littlegroup_evec_110(pgen,pref,i)
  elif pt == '111':
    if i < 2 or i == 3 or i == 4: ## index 2 is 2-dimensional
      return construct_littlegroup_evec_xyz(pj)
    return construct_littlegroup_evec_111(pgen,pref,i)
  elif pt == '210' or pt == '211' or pt == '321':
    return construct_littlegroup_evec_xyz(pj)
  else:
    raise ValueError("unknown momentum type!")

## #build full set of representation matrices
## #extract little group elements
## #find little group reference momentum subspace
## #compute little group projection matrix
## #extract eigenvectors in little group
## #separate orthogonal eigenvector spaces
## #align eigenvectors with reference representations
## #expand eigenvectors to full space
## #compute eigenvectors for full orbit
## ensure that orbit eigenvectors are defined consistently
