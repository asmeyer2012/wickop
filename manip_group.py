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

## expand list of characters by centralizers, for generating projection matrices
def extend_char(charRow,charCent):
  return np.concatenate([np.array([x]*int(y)) for x,y in zip(charRow,charCent)])

### expand list of characters by centralizers for all rows in character table
#def extend_char_table(charTab,charCent):
#  return np.array([ extend_char(x,charCent) for x in charTab ])

## move LG spin/helicity indices to the right, momentum indices to the left
def push_lg_indices(r0,p0d,p1d,lg0d,lg1d):
  totDim = p0d*p1d*lg0d*lg1d
  if len(r0[0]) != totDim:
    ## dimensions don't match
    print "dimensions: ",len(r0[0]),totDim
    raise DimensionError("dimensions don't match representation matrices!")
  if (lg0d == 1) or (p1d == 1): ## nothing to do here, move along
    return r0
  ry = np.reshape( r0, (p0d,lg0d,p1d,lg1d,p0d,lg0d,p1d,lg1d) )
  rz = np.transpose( ry, axes=(0,2,1,3,4,6,5,7) )
  r1 = rz.reshape((totDim,totDim))
  return r1

## compute tensor product of representation matrices
def get_tensor(r0,r1):
 return [np.kron(rx,ry) for rx,ry in zip(r0,r1)]

## compute tensor product of representation matrices, pushing LG spin/helicity indices to outside
def get_tensor_spinmom(r0,r1,p0d,p1d,lg0d,lg1d):
 r2 = [np.kron(rx,ry) for rx,ry in zip(r0,r1)]
 r2x = [push_lg_indices(x,p0d,p1d,lg0d,lg1d) for x in r2]
 return r2x

## generate a projection matrix for the given representation and chosen row of character table
def proj_matrix(r0,charRow,charCent):
 charExt = extend_char(charRow,charCent)
 charOrd = int(np.sum(charCent))
 ## charTab[0] is dimension of representation
 return sum([x*y for x,y in zip(r0,charExt)])*charRow[0]/charOrd

## need something smarter, does not distinguish between degenerate eigenvalues
#
## get permutation matrix to make order of di0 match l0, assuming di0 is diagonal of matrix
def get_permutation(di0,l0):
  li0 = di0
  len0 = len(li0)
  ix = []
  if not( all_zero(np.array(sorted(di0)) - np.array(sorted(l0)) )):
    print sorted(di0), sorted(l0)
    raise SimilarityError("Eigenvalues not similar to each other!")
  for i,x in enumerate(l0):
    ## find all values that are same
    x0 = list(np.abs(chop(li0 - x)) < smallNum)
    ## rule out those that don't match and that have already been used
    if len(ix) == 0:
      x1 = [j for j,y in enumerate(x0) if y ]
    else: ## check for already used entries
      x1 = [j for j,y in enumerate(x0) if (y and not(j in np.array(ix).T[1])) ]
    ix.append( (i, x1[0]) )
  ix = [tuple(x) for x in np.array(ix).T]
  perm = np.zeros((len0,len0),dtype=int)
  perm[ix] = 1
  ## now diagonal of perm.dot(np.diag(di0)).dot(perm.T) should match l0
  return perm

## given some representation matrix and a reference
## find similarity transform needed to get to the reference
def get_similarity(mat,ref):
  elm,evm = sane_eig(mat)
  elr,evr = sane_eig(ref)
  perm = get_permutation(elm,elr)
  sim = evr.dot(perm).dot(herm(evm))
  ## similarity transform is ref = sim.dot(mat).dot(herm(sim))
  return sim

## if one element guaranteed aligned, then next similarity must commute with that element
## similarity is a diagonal phase matrix sandwiched between the eigenvector matrices
## one diagonal entry unconstrained, set first to 1
## all others determined iteratively
## not clean code, but seems to work
#
## if V is evec matrix of refz, and Vi is inverse; determine diagonal unitary D by:
## V.refx.Vi = D.V.matx.Vi.Di
## new similarity is Vi.D.V
## TODO: check this, is V <-> Vi?
## TODO: have replaced V <-> Vi as check
#
## to determine D, compute V.refx.Vi and V.matx.Vi, compare off-diagonal entries
## off-diagonals must be related by phases only, these are the products of phases of D
def similarity_phase(matx,refx,refz):
  elz,evz = sane_eig(refz)
  vmv = herm(evz).dot(matx).dot(evz)
  vrv = herm(evz).dot(refx).dot(evz)
  phs = [complex(1.,0.)] ## keep track of phases
  idx = [0] ## keep track of done indices
  it = 0
  while len(idx) < len(elz):
   if it > 10: 
     raise IterationError("no convergence on phases")
   it = it +1
   ## use the previous to determine the phase of the next
   for i,(xm,xr) in enumerate(zip(vmv[idx[-1]],vrv[idx[-1]])):
     if i in idx:
       continue ## already done, find something different
     if np.abs(xm) < smallNum:
       continue ## zero, no information
     if i < idx[-1]:
       phs.append( (xm/xr)*phs[idx[-1]] )
     else:
       phs.append( (xm/xr)/phs[idx[-1]] )
     idx.append( i )
  ## rearrange the phases to match the index order
  phs = [phs[t] for t in idx]
  ## compute the new similarity matrix
  sim = evz.dot(np.diag(phs)).dot(herm(evz))
  return sim

## build representation matrices from the 0-momentum conjugacy classes and generators
## group products must be the same for all momenta
## 1, +R, +R'.R, +-R.R, +-R.R'.R, -1, -R, -R'.R
def build_from_generators(gR12,gR23,gPP):
  gR13 = mconj(gR23,gR12)
  gR21 = np.linalg.inv(gR12)
  gR32 = np.linalg.inv(gR23)
  gR31 = np.linalg.inv(gR13)
  rep_list = []
  ## identity, 1 total
  gId = gPP.dot(gPP)
  rep_list.append(gId)
  ## +R operators, 6 total
  for x in [gR23,gR13,gR12,gR32,gR31,gR21]:
    rep_list.append(x)
  ## +R'.R operators, 8 total
  for x in [gR13,gR12,gR31,gR21]:
    rep_list.append(gR23.dot(x))
  for x in [gR13,gR12,gR31,gR21]:
    rep_list.append(x.dot(gR32))
  ## +-R.R operators, 6 total
  for x in [gR23,gR13,gR12,gR32,gR31,gR21]:
    rep_list.append(x.dot(x))
  ## +-R.R'.R operators, 12 total
  ## get these from conjugation
  for x in [gId,gR23,gR13,gR12,gR32,gR31,gR21]:
    rep_list.append( mconj(gR12.dot(gR23).dot(gR12),x) )
  for x in [gId,gR12,gR32,gR31,gR31.dot(gR12)]:
    rep_list.append( mconj(gR13.dot(gR32).dot(gR13),x) )
  ## -1 operator, 1 total
  nId = np.linalg.matrix_power(gR12,4)
  rep_list.append(nId)
  ## -R, -R'.R, just multiply existing by -1 operator
  for x in list(rep_list[1:15]):
    rep_list.append(nId.dot(x))
  ## parity, multiply all by gPP
  for x in list(rep_list):
    rep_list.append(gPP.dot(x))
  return rep_list

## from a full list of representation matrices, extract only the generators of octahedral group
## this depends on how build_from_generators is defined!
def extract_generators(rep):
  return [rep[3],rep[1],rep[48]] ## r12,r23,pp

## same as build_from_generators, except generators applied in reverse
## application of matrix operators satisfy:
## L(v) = R^{-1}.L(R.v) = S^{-1}.R^{-1}.L(R.S.v)
## => S.L(R.v) = L(R.S.v)
## S.R.L(v) = L(R.S.v)
## important implications for choosing elements that belong to little groups
def build_reverse_from_generators(gR12,gR23,gPP):
  gR13 = mconj(gR23,herm(gR12)) ## R12 needs to be conjugated to fix for reversed
  gR21 = np.linalg.inv(gR12)
  gR32 = np.linalg.inv(gR23)
  gR31 = np.linalg.inv(gR13)
  rep_list = []
  ## identity, 1 total
  gId = gPP.dot(gPP)
  rep_list.append(gId)
  ## +R operators, 6 total
  for x in [gR23,gR13,gR12,gR32,gR31,gR21]:
    rep_list.append(x)
  ## +R'.R operators, 8 total
  for x in [gR13,gR12,gR31,gR21]:
    rep_list.append(x.dot(gR23))
  for x in [gR13,gR12,gR31,gR21]:
    rep_list.append(gR32.dot(x))
  ## +-R.R operators, 6 total
  for x in [gR23,gR13,gR12,gR32,gR31,gR21]:
    rep_list.append(x.dot(x))
  ## +-R.R'.R operators, 12 total
  ## get these from conjugation
  for x in [gId,gR23,gR13,gR12,gR32,gR31,gR21]:
    rep_list.append( mconj(gR12.dot(gR23).dot(gR12),herm(x)) )
  for x in [gId,gR12,gR32,gR31,gR12.dot(gR31)]:
    rep_list.append( mconj(gR13.dot(gR32).dot(gR13),herm(x)) )
  ## -1 operator, 1 total
  nId = np.linalg.matrix_power(gR12,4)
  rep_list.append(nId)
  ## -R, -R'.R, just multiply existing by -1 operator
  for x in list(rep_list[1:15]):
    rep_list.append(nId.dot(x))
  ## parity, multiply all by gPP
  for x in list(rep_list):
    rep_list.append(gPP.dot(x))
  return rep_list

#la = []
#lb = []
#for i,x in enumerate([r23,r13,r12,r32,r31,r21]):
# for j,y in enumerate([r23,r13,r12,r32,r31,r21]):
#  if i == j or i == j+3 or i+3 == j:
#   continue
#  if np.all([not(all_zero(x.dot(y)-z)) for z in la]) and\
#     np.all([not(all_zero(y.dot(x)-z)) for z in lb]):
#   la.append(x.dot(y))
#   lb.append(y.dot(x))
#   print len(la),i,j
#
#la = []
#lb = []
#for i,x in enumerate([r23v,r13v,r12v,r32v,r31v,r21v]):
# for j,y in enumerate([r23v,r13v,r12v,r32v,r31v,r21v]):
#  if i == j or i == j+3 or i+3 == j:
#   continue
#  if np.all([not(all_zero(x.dot(y)-z)) for z in la]) and\
#     np.all([not(all_zero(y.dot(x)-z)) for z in lb]):
#   la.append(x.dot(y))
#   lb.append(y.dot(x))
#   print len(la),i,j

