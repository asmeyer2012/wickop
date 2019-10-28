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

from itertools import permutations

## this is a really dumb isospin implementation
## it will later be replaced

## isospin object
## isospin QNs are multiplied by 2 to always be integers
class isospin_base:
  def __init__(self,I2tot,I2comp):
    self.I2tot = I2tot
    self.I2comp = I2comp
  ## apply lowering operator
  def ilower(self):
    if self.I2comp > -self.I2tot:
      self.I2comp = self.I2comp - 2
      return True
    return False
  ## apply raising operator
  def iraise(self):
    if self.I2comp < self.I2tot:
      self.I2comp = self.I2comp + 2
      return True
    return False
  ## set component to highest weight
  def highest_weight(self):
    self.I2comp = self.I2tot
  ## copy
  def copy(self):
    return isospin_base(self.I2tot,self.I2comp)

## combination of several isospin_base classes
## compute tensor products and decompose into isospin irreps
class isospin_term:
  def __init__(self,iso2=None,pf=None):
    self.iso2 = []
    self.pf = complex(1.)
    if isinstance(iso2,list):
      self.iso2 = iso2
    elif not(iso2 is None):
      self.iso2 = [iso2]
    if not(pf is None):
      self.pf = complex(pf)

  ## apply lowering operator to all isospin_bases and create a new set of isospin_terms
  def ilower(self):
    terms = []
    for i in range(len(self.iso2)):
      term = [i2.copy() for i2 in self.iso2]
      if (term[i].ilower()):
        terms.append( isospin_term(term, pf=self.pf))
    return terms
  ## apply lowering operator to all isospin_bases and create a new set of isospin_terms
  def iraise(self):
    terms = []
    for i in range(len(self.iso2)):
      term = [i2.copy() for i2 in self.iso2]
      if (term[i].iraise()):
        terms.append( isospin_term(term, pf=self.pf))
    return terms

  ## scale the term by a factor
  def scale(self,factor):
    self.pf = self.pf * factor
  ## set all components to highest weight
  def highest_weight(self):
    [i2.highest_weight() for i2 in self.iso2]
  ## return the isospin components of each element of the term
  def get_isospin_components(self):
    return tuple(i2.I2comp for i2 in self.iso2)
  ## quickly change many isospin components
  def set_isospin_components(self,I2comp):
    #[i2.I2comp = ic for zip(i2,ic) in zip(self.iso2,I2comp)]
    pass
  ## copy
  def copy(self):
    return isospin_term(self.iso2,self.pf)

## sum of several isospin_term classes
class isospin:
  def __init__(self,iso2=None):
    self.iso2 = []
    if isinstance(iso2,list):
      self.iso2 = iso2
    elif not(iso2 is None):
      self.iso2 = [iso2]

  ## normalize prefactors to give unit norm
  def normalize(self):
    pf = [i2.pf for i2 in self.iso2]
    pfn = 1./np.sqrt(np.abs(np.vdot(pf,pf)))
    [i2.scale(pfn) for i2 in self.iso2]
  ## reduce isospin operator to combine all like terms
  def ireduce(self):
    icomps = [i2.get_isospin_components() for i2 in self.iso2]
    terms = []
    for ic in set(icomps):
      pf0 = sum([(i2.pf if i2.get_isospin_components() == ic else complex(0.)) for i2 in self.iso2])
      if np.abs(pf0) < smallNum: ## get rid of zeros:
        continue
      i2 = self.iso2[ icomps.index(ic) ]
      i2.pf = pf0
      terms.append(i2)
    self.iso2 = sorted(terms, lambda x: x.get_isospin_components())
    self.normalize()

  ## apply lowering to all operators
  def ilower(self):
    terms = []
    for i2 in self.iso2:
      terms = terms + i2.ilower()
    self.iso2 = terms
    self.ireduce()
  ## apply raising to all operators
  def iraise(self):
    terms = []
    for i2 in self.iso2:
      terms = terms + i2.iraise()
    self.iso2 = terms
    self.ireduce()

## use highest weight decomposition
i1 = [[1.,[1]]]

## copy a list so changes don't propagate further than intended
def iso_copy(x):
 return [[x0[0],list(x0[1])] for x0 in x]

## do a tensor product of isospin representations
def iso_tensor(x,y):
 return [(x0[0]*y0[0],x0[1]+y0[1]) for x0 in x for y0 in y]

## sum terms, get rid of zeros, sort
def iso_reduce(x):
 terms = [x0[1] for x0 in x]
 uterms = sorted([list(y) for y in set([tuple(x0) for x0 in terms])])
 newx = []
 for term in uterms:
  ## compare isospin terms, find all that are the same
  idx = [i for i in range(len(terms)) if term == terms[i]]
  newfac =sum([x[i][0] for i in idx])
  #if np.abs(newfac) > smallNum:
  newx.append([newfac,term])
 return newx

def iso_scale(x,fac):
 return [[x0[0]*fac,x0[1]] for x0 in x]

def iso_replace(x,v):
 return [[vi,x0[1]] for vi,x0 in zip(v,x)]

def iso_extract_vector(x):
 return np.array([x0[0] for x0 in x])

def iso_normalize(x):
 vec = iso_extract_vector(x)
 norm = np.sqrt(vec.dot(vec))
 if np.abs(norm) < smallNum:
  return iso_scale(x,0.)
 else:
  return iso_scale(x,1./norm)

def iso_extract_vector_list(xl):
 vl = np.array([ iso_extract_vector(x) for x in xl ])
 return vl.T

## return an orthogonal vector to those already supplied
## input are COLUMN vectors
def iso_orthogonal(v):
 nextv = np.zeros((v.shape[0],1))
 nextv[0] = 1.
 outv = orthogonalize(np.hstack([v,nextv]))
 return np.real( outv.T[-1] )

### iso_reduce check
#ix = iso_copy(i1) + iso_copy(i1)
#print iso_reduce(ix)
#iy = iso_copy(i1) + iso_copy(i1)
#iy[0][0] = -1.
#print iso_reduce(iy)

def iso_upper(x):
 newIso = []
 x = iso_reduce(x) ## sort and reduce
 for x0 in x:
  for i in range(len(x0[1])):
   y = list(x0[1])
   y[i] = y[i] + 1
   if y[i] > 1:
    continue
   newIso.append([x0[0],y])
 return iso_normalize(iso_reduce(newIso))

def iso_lower(x):
 newIso = []
 x = iso_reduce(x) ## sort and reduce
 for x0 in x:
  for i in range(len(x0[1])):
   y = list(x0[1])
   y[i] = y[i] - 1
   if y[i] < -1:
    continue
   newIso.append([x0[0],y])
 return iso_normalize(iso_reduce(newIso))

## input total isospin, build a full set of operators and compute CG coefficients
def isospin_set(ni):
 ## build from the highest weight state
 iso_vn = [ [[1.,[1]*ni]] ] ## list of states
 for i in range(ni):
  next_vn = [ iso_lower(x) for x in iso_vn ]
  vnext = iso_orthogonal( iso_extract_vector_list( next_vn ))
  inext = iso_replace( next_vn[0], chop(vnext) )
  iso_vn = next_vn
  iso_vn.append( inext )
 return iso_vn[::-1] ## reverse, so index corresponds to isospin

def iso_permutation(iterm,perm):
 xset = []
 ## permute all terms in isospin set
 for y in iterm:
  xset.append([y[0],[y[1][t] for t in perm]])
 return xset

def isospin_permutation(iset,perm):
 xset = []
 ## permute all terms in isospin set
 for x0 in iset:
  #xset.append( iso_permutation( x0,perm))
  xset.append([])
  for y in x0:
   xset[-1].append([y[0],[y[1][t] for t in perm]])
 return xset

## try to symmetrize the isospin set over particle exchanges
def isospin_symmetrize(iset):
  blen = len(iset) ## basis length
  vlen = len(iset[0]) ## vector length
  tlen = len(iset[0][0][1]) ## isospin term length
  isetNext = iset
  #for i in range(tlen):
  # for j in range(i+1,tlen):
  #  perm = range(tlen)
  #  perm[i] = j
  #  perm[j] = i
  for perm in permutations(range(tlen)):
    for k in range(blen):
     x = isetNext[k]
     iperm = iso_permutation(x,perm)
     ipos = iso_reduce( x + iperm)
     ineg = iso_reduce( x + iso_scale( iperm, -1.) )
     vlist = iso_extract_vector_list( isetNext)
     vpos = hard_parallelize( iso_extract_vector( ipos).reshape((len(ipos),1)), vlist)
     vneg = hard_parallelize( iso_extract_vector( ineg).reshape((len(ipos),1)), vlist)
     if (vpos.shape == (0,)):
      vpos = vpos.reshape((vlen,0))
     if (vneg.shape == (0,)):
      vneg = vneg.reshape((vlen,0))
     vnext = (hard_orthogonalize( np.hstack(( vpos,vneg,vlist))))
     if (len(vnext[0]) != blen): ## don't lose vectors
       raise ValueError("isospin_symmetrize reduced matrix rank!")
     isetNext = [iso_replace( isetNext[0], chop(x) ) for x in vnext.T]
  return isetNext

## input total isospin, build a full set of operators and compute CG coefficients
def isospin_set_full(ni):
 rdict = {}
 if ni <= 1:
  rdict[ni] = [ [[1.,[0]]] ] ## only one operator
  return rdict
 ## build from the highest weight state
 iso_vn = [ [[1.,[1]*ni]] ] ## list of states
 rdict[ni] = iso_vn
 for i in range(ni):
  ## handle the return values
  for key in rdict.keys():
    rdict[key] = [ iso_reduce( iso_lower(x)) for x in rdict[key] ]
  next_vn = [ iso_reduce( iso_lower(x)) for x in iso_vn ]
  vnext = complete_eigenbasis( iso_extract_vector_list( next_vn ))
  inext = [iso_replace( next_vn[0], chop(x) ) for x in vnext.T]
  inext = isospin_symmetrize( inext)
  #print "isospin ",ni-i-1,":"
  #for j in range(int(np.ceil(len(inext)/3))):
  # print iso_extract_vector_list(inext)[:,int(3*j):int(3*(j+1))]
  iso_vn = next_vn + inext
  rdict[ni-i-1] = inext
 return rdict

## input total isospin, build a full set of operators and compute CG coefficients
def isospin_set_full_components(ni):
 rdict = {}
 if ni <= 1: ## trivial
  for nij in [-1,0,1]:
    rdict[ni,nij] = [ [[1.,[nij]]] ]
  rdict[ni] = rdict[ni,0]
  return rdict
 ## build from the highest weight state
 iso_vn = [ [[1.,[1]*ni]] ] ## list of states
 rdict[ni] = iso_vn
 rdict[ni,ni] = iso_vn
 for i in range(2*ni):
  ## handle the return values
  for key in rdict.keys():
    if not( isinstance( key, tuple)):
      if np.abs(ni-i-1) > key:
        continue
      rdict[key] = [ iso_reduce( iso_lower(x)) for x in rdict[key] ]
      rdict[key,ni-i-1] = [ iso_copy(x) for x in rdict[key]]
  if i < ni: ## only until we reach isospin 0
    next_vn = [ iso_reduce( iso_lower(x)) for x in iso_vn ]
    vnext = complete_eigenbasis( iso_extract_vector_list( next_vn ))
    inext = [iso_replace( next_vn[0], chop(x) ) for x in vnext.T]
    inext = isospin_symmetrize( inext)
    #print "isospin ",ni-i-1,":"
    #for j in range(int(np.ceil(len(inext)/3))):
    # print iso_extract_vector_list(inext)[:,int(3*j):int(3*(j+1))]
    iso_vn = next_vn + inext
    rdict[ni-i-1] = inext
    rdict[ni-i-1,ni-i-1] = [iso_copy(x) for x in inext]
 ## update the non-tuple values to isospin 0 for backward compatibility
 for i in range(ni+1):
  rdict[i] = [iso_copy(x) for x in rdict[i,0]]
 return rdict

### isospin 2
#ni = 2
#i2set = isospin_set(ni)
#for i,x in enumerate(i2set):
# print i
# print x
#
### isospin 3
#ni = 3
#i3set = isospin_set(ni)
#for i,x in enumerate(i3set):
# print i
# print x

### isospin 4
#ni = 4
#i4set = isospin_set_full(ni)
#for nix in range(ni+1):
# for i,x in enumerate(i4set[nix]):
#  print i
#  print x
#

### some test junk
#i3vecs = np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])
#i3vecs = orthogonalize( i3vecs)
#i2vecs = np.array([
#       [ 0.18898224,  0.28867513,  0.28867513,  0.28867513, -0.40824829,
#        -0.40824829, -0.40824829,  0.        ,  0.        , -0.46291005],
#       [ 0.18898224,  0.28867513, -0.28867513, -0.28867513, -0.40824829,
#         0.40824829,  0.40824829,  0.        ,  0.        , -0.46291005],
#       [ 0.18898224, -0.28867513,  0.28867513, -0.28867513,  0.40824829,
#        -0.40824829,  0.40824829,  0.        ,  0.        , -0.46291005],
#       [ 0.18898224, -0.28867513, -0.28867513,  0.28867513,  0.40824829,
#         0.40824829, -0.40824829,  0.        ,  0.        , -0.46291005],
#       [ 0.37796447,  0.57735027,  0.        ,  0.        ,  0.40824829,
#         0.        ,  0.        ,  0.57735027,  0.        ,  0.15430335],
#       [ 0.37796447,  0.        ,  0.57735027,  0.        ,  0.        ,
#         0.40824829,  0.        , -0.28867513,  0.5       ,  0.15430335],
#       [ 0.37796447,  0.        ,  0.        ,  0.57735027,  0.        ,
#         0.        ,  0.40824829, -0.28867513, -0.5       ,  0.15430335],
#       [ 0.37796447,  0.        ,  0.        , -0.57735027,  0.        ,
#         0.        , -0.40824829, -0.28867513, -0.5       ,  0.15430335],
#       [ 0.37796447,  0.        , -0.57735027,  0.        ,  0.        ,
#        -0.40824829,  0.        , -0.28867513,  0.5       ,  0.15430335],
#       [ 0.37796447, -0.57735027,  0.        ,  0.        , -0.40824829,
#         0.        ,  0.        ,  0.57735027,  0.        ,  0.15430335]])
#
#i1vecs = np.array([
#       [ 0.37796447, -0.25819889, -0.25819889, -0.25819889, -0.28867513,
#        -0.28867513, -0.28867513,  0.        ,  0.        ,  0.32732684,
#         0.31622777,  0.31622776,  0.31622777,  0.        ,  0.        ,
#         0.        ],
#       [ 0.37796447, -0.25819889,  0.25819889,  0.25819889,  0.28867513,
#         0.28867513, -0.28867513,  0.        ,  0.        ,  0.32732684,
#         0.31622777, -0.31622777, -0.31622776,  0.        ,  0.        ,
#         0.        ],
#       [ 0.37796447,  0.25819889, -0.25819889,  0.25819889,  0.28867513,
#        -0.28867513,  0.28867513,  0.        ,  0.        ,  0.32732684,
#        -0.31622777,  0.31622777, -0.31622777,  0.        ,  0.        ,
#         0.        ],
#       [ 0.37796447,  0.25819889,  0.25819889, -0.25819889, -0.28867513,
#         0.28867513,  0.28867513,  0.        ,  0.        ,  0.32732684,
#        -0.31622777, -0.31622776,  0.31622776,  0.        ,  0.        ,
#         0.        ],
#       [ 0.18898224,  0.38729833, -0.12909944, -0.12909944,  0.28867514,
#         0.28867514,  0.        ,  0.40824829,  0.        , -0.21821789,
#         0.31622777,  0.15811389,  0.15811387,  0.35355339, -0.35355339,
#         0.        ],
#       [ 0.18898224, -0.12909944,  0.38729833, -0.12909944,  0.28867514,
#         0.        ,  0.28867514, -0.20412414,  0.35355339, -0.21821789,
#         0.15811388,  0.31622777,  0.15811387, -0.35355339,  0.        ,
#        -0.35355339],
#       [ 0.18898224, -0.12909944, -0.12909944,  0.38729833,  0.        ,
#         0.28867514,  0.28867514, -0.20412414, -0.35355339, -0.21821789,
#         0.15811388,  0.15811388,  0.31622776,  0.        ,  0.3535534 ,
#         0.3535534 ],
#       [ 0.18898224,  0.38729833,  0.12909944,  0.12909944, -0.28867514,
#        -0.28867514,  0.        ,  0.40824829,  0.        , -0.21821789,
#         0.31622777, -0.15811388, -0.15811389, -0.35355339,  0.3535534 ,
#         0.        ],
#       [ 0.18898224, -0.12909944,  0.12909944, -0.38729833,  0.        ,
#        -0.28867514,  0.28867514, -0.20412414, -0.35355339, -0.21821789,
#         0.15811388, -0.15811388, -0.31622777,  0.        , -0.35355341,
#         0.35355338],
#       [ 0.18898224, -0.12909944, -0.38729833,  0.12909944, -0.28867514,
#         0.        ,  0.28867514, -0.20412414,  0.35355339, -0.21821789,
#         0.15811388, -0.31622777, -0.15811388,  0.35355339,  0.        ,
#        -0.35355339],
#       [ 0.18898224,  0.12909944,  0.38729833,  0.12909944, -0.28867514,
#         0.        , -0.28867514, -0.20412414,  0.35355339, -0.21821789,
#        -0.15811388,  0.31622777, -0.15811389,  0.35355339,  0.        ,
#         0.3535534 ],
#       [ 0.18898224,  0.12909944, -0.12909944, -0.38729833,  0.        ,
#         0.28867514, -0.28867514, -0.20412414, -0.35355339, -0.21821789,
#        -0.15811388,  0.15811388, -0.31622777,  0.        ,  0.35355338,
#        -0.35355341],
#       [ 0.18898224, -0.38729833, -0.12909944,  0.12909944, -0.28867514,
#         0.28867514,  0.        ,  0.40824829,  0.        , -0.21821789,
#        -0.31622777,  0.15811388, -0.15811387, -0.3535534 , -0.35355339,
#         0.        ],
#       [ 0.18898224,  0.12909944,  0.12909944,  0.38729833,  0.        ,
#        -0.28867514, -0.28867514, -0.20412414, -0.35355339, -0.21821789,
#        -0.15811388, -0.15811388,  0.31622777,  0.        , -0.35355337,
#        -0.35355336],
#       [ 0.18898224,  0.12909944, -0.38729833, -0.12909944,  0.28867514,
#         0.        , -0.28867514, -0.20412414,  0.35355339, -0.21821789,
#        -0.15811388, -0.31622777,  0.15811389, -0.3535534 ,  0.        ,
#         0.3535534 ],
#       [ 0.18898224, -0.38729833,  0.12909944, -0.12909944,  0.28867514,
#        -0.28867514,  0.        ,  0.40824829,  0.        , -0.21821789,
#        -0.31622777, -0.15811389,  0.1581139 ,  0.35355338,  0.35355339,
#         0.        ]])
#
#
#isoterms3 = np.ones((4,4)) - np.diag(np.ones(4))
#isoterms2 = np.vstack((np.ones((4,4))-2.*np.diag(np.ones(4)),np.array([[0,0,1,1],[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0],[1,1,0,0]])))
#isoterms0 = [
# [0, 0, 0, 0], [-1, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0],
# [0, -1, 0, 1], [0, -1, 1, 0], [1, -1, 0, 0], [0, 0, -1, 1],
# [0, 1, -1, 0], [1, 0, -1, 0], [0, 0, 1, -1], [0, 1, 0, -1],
# [1, 0, 0, -1], [-1, -1, 1, 1], [-1, 1, -1, 1], [-1, 1, 1, -1],
# [1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1]]
#
#termorder2 = [0, 4, 7, 9, 1, 2, 3, 5, 6, 8]
#termorder1 = [12, 7, 5, 4, 3, 6, 8, 0, 11, 13, 1, 9, 15, 2, 10, 14]
#termorder0 = [9, 1, 2, 4, 6, 7, 14, 8, 11, 16, 10, 12, 17, 0, 3, 5, 13, 15, 18]
#
#def f(x,y):
# for i,j in enumerate(y):
#  print i,x[j]
#
#def g(x,y):
# return np.array([x[j][0] for j in y])
#
#def h(x,y):
# return [x[j] for j in y]
#
#i403 = [[v,list(w)] for v,w in zip(i3vecs.T[0],isoterms3)]
#i303 = [[v,list(w)] for v,w in zip(i3vecs.T[1],isoterms3)]
#i313 = [[v,list(w)] for v,w in zip(i3vecs.T[2],isoterms3)]
#i323 = [[v,list(w)] for v,w in zip(i3vecs.T[3],isoterms3)]
#
#i402 = [[v,list(w)] for v,w in zip(i2vecs.T[0],isoterms2)]
#i302 = [[v,list(w)] for v,w in zip(i2vecs.T[1],isoterms2)]
#i312 = [[v,list(w)] for v,w in zip(i2vecs.T[2],isoterms2)]
#i322 = [[v,list(w)] for v,w in zip(i2vecs.T[3],isoterms2)]
#i202 = [[v,list(w)] for v,w in zip(i2vecs.T[4],isoterms2)]
#i212 = [[v,list(w)] for v,w in zip(i2vecs.T[5],isoterms2)]
#i222 = [[v,list(w)] for v,w in zip(i2vecs.T[6],isoterms2)]
#i232 = [[v,list(w)] for v,w in zip(i2vecs.T[7],isoterms2)]
#i242 = [[v,list(w)] for v,w in zip(i2vecs.T[8],isoterms2)]
#i252 = [[v,list(w)] for v,w in zip(i2vecs.T[9],isoterms2)]
#
#i401 = h(iso_lower(i402),termorder1)
#i301 = h(iso_lower(i302),termorder1)
#i311 = h(iso_lower(i312),termorder1)
#i321 = h(iso_lower(i322),termorder1)
#i201 = h(iso_lower(i202),termorder1)
#i211 = h(iso_lower(i212),termorder1)
#i221 = h(iso_lower(i222),termorder1)
#i231 = h(iso_lower(i232),termorder1)
#i241 = h(iso_lower(i242),termorder1)
#i251 = h(iso_lower(i252),termorder1)
#i101 = iso_replace(i401,i1vecs.T[10])
#i111 = iso_replace(i401,i1vecs.T[11])
#i121 = iso_replace(i401,i1vecs.T[12])
#i131 = iso_replace(i401,i1vecs.T[13])
#i141 = iso_replace(i401,i1vecs.T[14])
#i151 = iso_replace(i401,i1vecs.T[15])
#
#i400 = h(iso_lower(i401),termorder0)
#i300 = h(iso_lower(i301),termorder0)
#i310 = h(iso_lower(i311),termorder0)
#i320 = h(iso_lower(i321),termorder0)
#i200 = h(iso_lower(i201),termorder0)
#i210 = h(iso_lower(i211),termorder0)
#i220 = h(iso_lower(i221),termorder0)
#i230 = h(iso_lower(i231),termorder0)
#i240 = h(iso_lower(i241),termorder0)
#i250 = h(iso_lower(i251),termorder0)
#i100 = h(iso_lower(i101),termorder0)
#i110 = h(iso_lower(i111),termorder0)
#i120 = h(iso_lower(i121),termorder0)
#i130 = h(iso_lower(i131),termorder0)
#i140 = h(iso_lower(i141),termorder0)
#i150 = h(iso_lower(i151),termorder0)
#
#f(iso_lower(i403),termorder2)
##f(iso_lower(i402),termorder1)
#
#t0 = np.real(chop(np.vstack([g(iso_lower(x),termorder2) \
# for x in [i403,i303,i313,i323]])))
#
#x0 = np.real(chop(np.vstack([g(iso_lower(x),termorder1) \
# for x in [i402,i302,i312,i322,i202,i212,i222,i232,i242,i252]])))
#
#y0 = np.real(chop(np.vstack([g(iso_lower(x),termorder0) \
# for x in [i401,i301,i311,i321,i201,i211,i221,i231,i241,i251,i101,i111,i121,i131,i141,i151]])))
#
#t0 = t0.T
#x0 = x0.T
#y0 = y0.T
#
#test1 = isospin_set(4)[1] ## isospin 1
#test1 = [test1[j] for j in termorder0]
#test1v = np.array([test1[j][0] for j in range(19)])
#
#for i in range(19):
# ostr = (" %2d [%2d,%2d,%2d,%2d] : " % ((i,)+tuple(isoterms0[i])))
# for j in range(6):
#  ostr = ostr + (" %8.6f" % y0[i][10+j])
# print ostr
#
#z0 = np.array([ np.real(chop(
# g( iso_reduce(testsym+iso_scale(isospin_permutation([testsym],[1,0,3,2])[0],-1.)), termorder0)))
# for testsym in [i100,i110,i120,i130,i140,i150]])
#
#z1 = np.array([ np.real(chop(
# g( iso_reduce(testsym+iso_scale(isospin_permutation([testsym],[1,0,2,3])[0],-1.)), termorder0)))
# for testsym in [i100,i110,i120,i130,i140,i150]])
#
#z2 = np.array([ np.real(chop(
# g( iso_reduce(
# testsym
# +isospin_permutation([testsym],[0,1,3,2])[0]
# +iso_scale(isospin_permutation([testsym],[1,0,2,3])[0],-1.)
# +iso_scale(isospin_permutation( isospin_permutation([testsym],[0,1,3,2]), [1,0,2,3])[0],-1.)
# ), termorder0)))
# for testsym in [i100,i110,i120,i130,i140,i150]])
#
#print chop(y0.T.dot(test1v))
#print np.real(chop(z1.dot(z1.T)))[:,1:]
#print np.real(chop(z2.dot(z2.T)))[:,1:]

