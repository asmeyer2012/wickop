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

## hermitian conjugate of matrix
def herm(x):
 return np.conj(x).T

## conjugate by matrix
## all matrices are unitary, so use hermitian conjugate instead of inverse
def mconj(x,y):
 return y.dot(x).dot(herm(y))
 #return y.dot(x).dot(np.linalg.inv(y))

## check if all entries of np.array are close to zero
def all_zero(mat):
  return np.all( np.abs(mat) < smallNum )

## check if a square matrix is identity
def is_identity(mat):
  return all_zero( mat- np.diag(np.ones(mat.shape[0])) )

## remove duplicates from list
def remove_duplicates(lst):
 nlst = list()
 for x in lst:
  if not(np.any([ all_zero(np.array(x)-np.array(y)) for y in nlst])):
   nlst.append(x)
 return nlst

## remove duplicates from list using sets
def remove_duplicates_set(lst):
 return list(set(lst))

## remove duplicates from first list, taking out elements from second list too
def simultaneous_remove_duplicates(lst0,lst1):
 nlst0 = list()
 nlst1 = list()
 for x0,x1 in zip(lst0,lst1):
  if not(np.any([ all_zero(x0-y) for y in nlst0])):
   nlst0.append(x0)
   nlst1.append(x1)
 return nlst0,nlst1

## set small values of np.array to zero
## separately handled for real,complex
def chop(v):
 if isinstance(v,float):
   if np.abs(v) < smallNum:
     return 0.
   return v
 if isinstance(v,complex):
   if np.abs(np.real(v)) < smallNum:
     v = complex(0.,np.imag(v))
   if np.abs(np.imag(v)) < smallNum:
     v = complex(np.real(v),0.)
   return v
 v = np.array(v)
 if isinstance(v[0],float):
   v[np.abs(v) < smallNum] = 0.
   return v
 rcpx = np.array(np.real(v))
 icpx = np.array(np.imag(v))
 rcpx[np.abs(rcpx) < smallNum] = 0.
 icpx[np.abs(icpx) < smallNum] = 0.
 vchop = np.vectorize(complex)(rcpx,icpx)
 return vchop

## normalize a set of COLUMN vectors in np.ndarray
def normalize(v):
 if len(v.shape) > 1:
  return herm(np.array([normalize(vx) for vx in herm(v)]))
 return (v/np.sqrt(np.vdot(v,v)))

## orthogonalize a set of COLUMN vectors
## np.linalg.eig returns non-orthogonal vectors when eigenvalues same, fix that
def orthogonalize(v):
 if isinstance(v,np.matrix):
  return orthogonalize(np.array(v))
 if len(v) == 0:
  return v
 v = herm(v)
 vr = [v[0]]
 ln = len(v[0])
 for vx in v[1:]:
  ## need to take care that space is spanned by vectors, np.linalg.eig not always guaranteed
  vnew = chop(vx - sum([vy*np.vdot(vy,vx)/np.vdot(vy,vy) for vy in vr]))
  ## completely projected out, pick a new random vector and try again
  if np.abs(np.linalg.norm(vnew)) < smallNum:
    vx = np.random.rand(ln)
    vnew = chop(vx - sum([vy*np.vdot(vy,vx)/np.vdot(vy,vy) for vy in vr]))
  vr.append( vnew )
 return chop(normalize(herm(np.array(vr))))

## orthongonalize a set of COLUMN vectors
## remove vectors if they are completely projected out
def hard_orthogonalize(v):
 if isinstance(v,np.matrix):
  return hard_orthogonalize(np.array(v))
 if len(v) == 0:
  return v
 v = herm(v)
 vr = [v[0]]
 ln = len(v[0])
 for vx in v[1:]:
  ## need to take care that space is spanned by vectors, np.linalg.eig not always guaranteed
  vnew = chop(vx - sum([vy*np.vdot(vy,vx)/np.vdot(vy,vy) for vy in vr]))
  if np.abs(np.linalg.norm(vnew)) < smallNum:
    continue ## completely projected out, skip
  vr.append( vnew )
 return chop(normalize(herm(np.array(vr))))

## takes a set of COLUMN vectors
## returns only vectors parallel to set in par
def parallelize(v,par):
 if isinstance(v,np.matrix) or isinstance(par,np.matrix):
  return parallelize(np.array(v),np.array(par))
 if len(v) == 0:
  return v
 if len(par) == 0:
  return np.array([]).reshape((len(v[0]),0))
 vr = []
 for vx in herm(v):
  vnew = chop(sum([vy*np.vdot(vy,vx)/np.vdot(vy,vy) for vy in par]))
  vr.append(vnew)
 return chop(normalize(herm(np.array(vr))))

## takes a set of COLUMN vectors
## returns only vectors parallel to set in par
def hard_parallelize(v,par):
 if isinstance(v,np.matrix) or isinstance(par,np.matrix):
  return hard_parallelize(np.array(v),np.array(par))
 if len(v) == 0:
  return v
 if len(par) == 0:
  return np.array([]).reshape((len(v[0]),0))
 vr = []
 for vx in herm(v):
  vnew = chop(sum([vy*np.vdot(vy,vx)/np.vdot(vy,vy) for vy in herm(par)]))
  if np.abs(np.linalg.norm(vnew)) < smallNum:
    continue ## completely projected out, skip
  vr.append(vnew)
 return hard_orthogonalize(herm(np.array(vr)))

## rotate two vectors into each other with a unitary mtx such that one vector has a 0 in index i
def unitary_reduce(v0,v1,i):
  ux0 = herm(normalize(np.array([v0,v1]).T[i] ))
  ux1 = np.array([ ux0, I*s2.dot(herm(ux0)) ]) ## construct the unitary 2x2 matrix
  return tuple( herm(herm(np.array([v0,v1])).dot( herm(ux1)) ) ) ## return rotated v0,v1

## get rid of phase on first nonzero element
def rephase_vector(v0):
  y = list(np.abs(v0) > smallNum).index(True) ## find first nonzero entry
  return chop( np.abs(v0[y])*v0/v0[y] ) ## remove the phase

## input set of COLUMN vectors
## rotate within given subspace to zero out as many entries as possible
def vector_reduction(ev):
 ln = ev.shape[1]
 if ln < 1:
  raise ValueError("no vector to reduce")
  return ev
 elif ln == 1:
  x = herm(ev)[0]
  y = list(np.abs(x) > smallNum).index(True) ## find first nonzero entry
  return chop( np.abs(x[y])*ev/x[y] ) ## remove the phase
 ## in general, vectors will contain lots of zeros
 ## find subset of rows that have nonzero entries and use those
 ix = [] ## list of indices to include
 evcut = [] ## list of eigenvector components to consider
 for x in herm(ev):
  tst = (np.abs(x) > smallNum) ## find nonzero entries
  for i,y in enumerate(tst):
    if y and not(i in ix): ## nonzero and not already included
      ## check that this is adding a linearly independent vector
      evnext = list(evcut)
      evnext.append(ev[i]) ## add this component of vectors
      if np.linalg.matrix_rank(herm(evnext)) > np.linalg.matrix_rank(herm(evcut)):
        ix.append(i) ## track this index
        evcut.append(ev[i]) ## add this component of vectors
        break ## check next vector x
 ## now use unitary transformations to rotate away nonzero elements
 newv = herm(ev)
 for i0,j0 in enumerate( ix):
  for i1 in range( i0+1,ln):
   v0 = newv[i0]
   v1 = newv[i1]
   v0x,v1x = unitary_reduce(v0,v1,j0)
   newv[i0] = rephase_vector( v0x)
   newv[i1] = rephase_vector( v1x)
 ## return the new vectors
 return chop(herm(newv))

## given an incomplete set of COLUMN eigenvectors, find missing subspace
## return only the new eigenvectors
def complete_eigenbasis(ev):
 evl = ev.shape[0] ## vector length
 evb = ev.shape[1] ## basis size
 ## compute new vectors orthogonal to old
 newVecs = hard_orthogonalize( np.hstack(( ev,
  np.vstack(( np.diag(np.ones(evl-evb)), np.zeros((evb,evl-evb)) )) )))[:,evb:]
 if (len(newVecs.shape) < 2) or (newVecs.shape[1] < evl-evb): ## try again
   newVecs = hard_orthogonalize( np.hstack(( ev, np.random.random((evl,evl-evb)) )))[:,evb:]
 #return vector_reduction( newVecs) ## doesn't work, degenerate vectors
 return newVecs ## reduction unnecessary

## format np.linalg.eig output in sane way: eigenvector matrix orthonormal, eigenvalues sorted
## eigenvector matrix is a set of COLUMN vectors
def sane_eig(mat):
 l,v = np.linalg.eig(np.array(mat)) ## eigenvectors as columns
 l = chop(l)
 v = herm(np.array(v)) ## eigenvectors as rows
 ## sort eigenvalues by absolute size in decreasing order
 lv = sorted(zip(l,np.array(v)),key=lambda pair: -np.abs(pair[0]))
 ## unpack
 ln = np.array([x[0] for x in lv])
 vn = np.array([x[1] for x in lv])
 ## recreate matrix with vn.dot(ln).dot(np.conj(vn).T)
 return ln,orthogonalize(herm(vn))

## get only the subspace that has the requested eigenvalue
def eig_subspace(mat,elx):
 el,ev = sane_eig(mat)
 ev = herm(ev)
 evsub = []
 for l,v in zip(el,ev):
  if np.abs(l - elx) < smallNum:
   ## belongs to requested subspace
   evsub.append(v)
 if len(evsub) == 0:
   print "No component with eigenvalue "+str(elx)+" in matrix:"
   print mat
   raise ValueError
 evsub = herm(np.array(evsub))
 return vector_reduction(evsub)

## take noncontiguous blocked slices of matrix and put them together
def noncontig_2d_slice(mat,ixList,blockSize):
  sliceList = [slice(blockSize*i,blockSize*(i+1)) for i in ixList]
  return np.hstack(np.hstack(np.array( [[ mat[i,j] for j in sliceList] for i in sliceList] )))

