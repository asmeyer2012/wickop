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

## momentum magnitude, an additional quantum number to sort with
def momentum_magnitude(p0):
  return np.dot(p0,p0)

## convert momentum to canonical form
## order by magnitude and all positive
## assumes list form
def canonical_momentum(p0):
  pc = [np.abs(px) for px in p0]
  pc.sort()
  return pc[::-1]

## convert to a reference momentum
## assumes list form
def reference_momentum(p0):
  momType = classify_momentum_type(p0)
  p0c = canonical_momentum(p0) ## all positive, decreasing order
  if momType == '100':
    return p0c[::-1] ## increasing order
  elif momType == '211':
    ## want the unique direction last
    if p0c[0] > p0c[1]:
      return p0c[::-1] ## increasing order
    else:
      return p0c ## decreasing order
  else:
    return p0c

## convert to a generic momentum type
## preserve rotation from canonical as well
def generic_momentum(p0):
  momType = classify_momentum_type(p0)
  genmom = reference_momentum([int(x) for x in momType])
  ## rotate to match direction of given
  i0 = all_permutations(p0).index(p0)
  return all_permutations(genmom)[i0]

## classify momenta by type, 7 types in total
def classify_momentum_type(p0):
  pc = canonical_momentum(p0)
  cnt0 = pc.count(0) ## count zeroes
  cntq = len(set(pc)) ## count unique elements
  if cnt0 == 3:
    return '000'
  elif cnt0 == 2:
    return '100'
  elif (cnt0 == 1) and (cntq == 2):
    return '110'
  elif (cnt0 == 1) and (cntq == 3):
    return '210'
  elif (cnt0 == 0) and (cntq == 2):
    return '211'
  elif (cnt0 == 0) and (cntq == 1):
    return '111'
  else:
    return '321'

## given momentum type, get dimension
def momentum_type_dimension(p0):
  if isinstance(p0,list):
    ## use the type instead of the momentum itself
    return momentum_type_dimension( classify_momentum_type(p0))
  else:
    ## pseudo-switch statement
    return {
      '000':  1,
      '100':  6,
      '110': 12,
      '111':  8,
      '210': 24,
      '211': 24,
      '321': 48
    }[p0]

## sorting algorithm for the momentum of one specific type and magnitude
## want to pick out the z-direction whenever possible
#
## this works for the base cases of momenta
## need smarter choices for eg 221, but otherwise general
def momentum_sort(pl):
  momType = classify_momentum_type(pl[0])
  if   momType == '000':
    return pl ## 0,0,0 only
  elif momType == '100':
    return sorted(pl, key=lambda x: (x[2],x[1],x[0])) ## 0,0,1 last
  elif momType == '110':
    return sorted(pl, key=lambda x: (x[0],x[1],-np.abs(x[2]),x[2])) ## 1,1,0 last
  elif momType == '111':
    return sorted(pl) ## 1,1,1 last
  elif momType == '210':
    return sorted(pl, key=lambda x: (x[0],x[1],-np.abs(x[2]),x[2])) ## 2,1,0 last
  elif momType == '211':
    pc = canonical_momentum(pl[0])
    if pc.count(pc[0]) == 1:
      ## largest aligned with z
      return sorted(pl, key=lambda x: (x[2],x[1],x[0])) ## 1,1,2 last (old)
    else:
      ## smallest must be aligned with z, do a replacement to get this
      psmall = pc[2]
      largeVal = 1000
      ## replace with large value, sort, then replace back
      plr = [[(np.sign(y)*largeVal if np.abs(y) == psmall else y) for y in x] for x in pl]
      pnew = sorted(plr, key=lambda x: (x[2],x[1],x[0]))
      plr = [[(np.sign(y)*psmall if np.abs(y) == largeVal else y) for y in x] for x in pnew]
      return plr
  elif momType == '321':
    return sorted(pl, key=lambda x: (x[2],x[1],x[0])) ## 1,2,3 last

## sort by momentum, and also sort vals too
## zip(* ) is the inverse of zip()
def simultaneous_momentum_sort(pl,vals):
  momType = classify_momentum_type(pl[0])
  if   momType == '000':
    return pl,vals ## 0,0,0 only, no sorting
  elif momType == '100':
    return zip(*sorted(zip(pl,vals), key=lambda (x,y): (x[2],x[1],x[0]))) ## 0,0,1 last
  elif momType == '110':
    ## put x[2] in twice to order (1,0,1) and (1,0,-1), but make (1,1,0) last
    return zip(*sorted(zip(pl,vals), key=lambda (x,y): (x[0],x[1],-np.abs(x[2]),x[2])))
  elif momType == '111':
    return zip(*sorted(zip(pl,vals), key=lambda (x,y): x)) ## 1,1,1 last
  elif momType == '210':
    return zip(*sorted(zip(pl,vals), key=lambda (x,y): (x[0],x[1],-np.abs(x[2]),x[2])))
  elif momType == '211':
    #return sorted(pl, key=lambda x: (x[2],x[1],x[0])) ## 1,1,2 last (old)
    pc = canonical_momentum(pl[0])
    if pc.count(pc[0]) == 1:
      ## largest aligned with z
      return zip(*sorted(zip(pl,vals), key=lambda (x,y): (x[2],x[1],x[0]))) ## 1,1,2 last (old)
    else:
      ## smallest must be aligned with z, do a replacement to get this
      psmall = pc[2]
      largeVal = 1000
      ## replace with large value, sort, then replace back
      plr = [[(np.sign(y)*largeVal if np.abs(y) == psmall else y) for y in x] for x in pl]
      pnew,newvals = zip(*sorted(zip(plr,vals), key=lambda (x,y): (x[2],x[1],x[0])))
      plr = [[(np.sign(y)*psmall if np.abs(y) == largeVal else y) for y in x] for x in pnew]
      return plr,newvals
  elif momType == '321':
    return zip(*sorted(zip(pl,vals), key=lambda (x,y): (x[2],x[1],x[0]))) ## 1,2,3 last

## sort momentum for all momentum types and magnitudes
## first sort by magnitude and type, then within the type
def full_momentum_sort(pl):
  pm = [momentum_magnitude(x) for x in pl]
  pt = [classify_momentum_type(x) for x in pl]
  plx = sorted(zip(pm,pt,pl), key=lambda x: (x[0],x[1])) ## won't change order within type
  ply = [x[:2] for x in plx] ## cut actual momentum and then sort over sublists
  plz = []
  for x in remove_duplicates_set(ply):
    i0 = ply.index(x)
    i1 = len(ply) - ply[::-1].index(x)
    plq = [y[2] for y in plx[i0:i1]]
    plz.append( momentum_sort(plq) )
  plz = [list(x) for x in np.concatenate(plz)]
  return plz

## get all permutations of a given momentum
def all_permutations(p0):
  ## get negative momentum combinations
  plist = [p0]
  for i in range(3):
    pnew = []
    for px in plist:
      pnew.append(px)
      py = list(px)
      py[i] = -py[i]
      pnew.append(py)
    plist = pnew
  ## get permutations of axes
  pnew = []
  for px in plist:
    for i in range(3):
      py = list(px)
      pz = list(px)
      for j in range(3):
        py[j] = px[(i+j)%3]   ## even permutations
        pz[j] = px[(i-j+3)%3] ## odd permutations
      pnew.append(py)
      pnew.append(pz)
  ## some elements double counted, get indices of unique elements
  return momentum_sort( remove_duplicates( pnew ))

## build a representation matrix for the given 3-rotation for this momentum list
def rep_matrix_mom(pl,r):
 lenp = len(pl)
 plr = [[int(x) for x in r.dot(px)] for px in pl]
 #ix = np.array([ (i,pl.index(prx)) for i,prx in enumerate(plr) ]).T
 ix = np.array([ (pl.index(prx),i) for i,prx in enumerate(plr) ]).T
 ix = [tuple(x) for x in ix] ## convert to tuples for indexing
 rmat = np.zeros((lenp,lenp))
 rmat[ix] = 1.
 return rmat

## make ncpy duplicates of each momentum from list pl
def expand_momentum_list(pl,ncpy):
  pList = []
  for p in pl:
   for i in range(ncpy):
    pList.append(p)
  return pList

## get a list of the summed momenta after a tensor product
## may not agree with conventional ordering of momenta, or all be the same type
def momentum_list_tensor(pl0,pl1):
  pl2 = np.concatenate( [[list( np.array(p0)+np.array(p1)) for p1 in pl1] for p0 in pl0] )
  return [list(x) for x in pl2]

## append a unique index to each momenta in list to track
## useful for momentum_permutation_block
def index_momentum_list(pl):
 idx = {}
 plx = []
 for p in pl:
  pstr = str(p)
  plx.append(p+[idx.get(pstr,0)])
  idx[pstr] = idx.get(pstr,0) + 1
 return plx

## from a list of momenta, build a permutation matrix
## matrix blocks all of the same magnitude/type momenta together
def momentum_permutation_block(pl):
 ## give each momentum a unique index
 pli = index_momentum_list(pl)
 plx = full_momentum_sort(pl)
 plxi = index_momentum_list(plx)
 ## very similar to rep_matrix_mom from here on out
 lenp = len(pl)
 ix = np.array([ (i,pli.index(px)) for i,px in enumerate(plxi) ]).T
 ix = [tuple(x) for x in ix] ## convert to tuples for indexing
 rmat = np.zeros((lenp,lenp))
 rmat[ix] = 1.
 ## can get block-diagonal matrix by applying mconj(rep,rmat)
 return rmat

## for a given momentum list and momentum type+magnitude,
## extract the corresponding subset of momentum
def extract_momentum_subset(pl,pref):
  plt = full_momentum_sort(pl)
  plx = [canonical_momentum(x) for x in pl]
  pls = full_momentum_sort(plx)
  pref = canonical_momentum(pref)
  i0 = pls.index(pref)
  i1 = len(pls) - pls[::-1].index(pref)
  slc = slice(i0,i1)
  return plt[slc]

## for a given momentum list and momentum type+magnitude,
## extract the corresponding subset of momentum
def extract_reference_subset(pl,pref):
  plt = full_momentum_sort(pl)
  #pref = reference_momentum(pref) ## don't enforce this
  i0 = plt.index(pref)
  i1 = len(plt) - plt[::-1].index(pref)
  slc = slice(i0,i1)
  return plt[slc]

## for a given momentum list and momentum type+magnitude,
## extract the corresponding block of the representation matrix for that momentum
def extract_momentum_block(pl,pref,mat):
  plx = [canonical_momentum(x) for x in pl]
  pls = full_momentum_sort(plx)
  pref = canonical_momentum(pref)
  i0 = pls.index(pref)
  i1 = len(pls) - pls[::-1].index(pref)
  slc = slice(i0,i1)
  return mat[slc,slc]

## for a given momentum list and reference momentum,
## extract the corresponding block of the representation matrix for the little group
def extract_reference_block(pl,pref,mat):
  pls = full_momentum_sort(pl) ## just to be sure
  i0 = pls.index(pref)
  i1 = len(pls) - pls[::-1].index(pref)
  slc = slice(i0,i1)
  return mat[slc,slc]

