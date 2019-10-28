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
from manip_littlegroup import *
from character_tables import *

## object that represents and irreducible representation
## contains the eigenvectors that correspond to the projected space that transforms in this irrep
## the list of momenta is a mapping of what the eigenvector elements mean
## irrep tag is an identifier that indicates what the momentum-helicity lattice symmetries are
class irrep:
 def __init__(self,irrepTag,evec,refMom):
  self.irrepTag = irrepTag ## named irrep tag
  self.evec = chop(evec) ## list of representation evecs
  self.refMom = refMom ## reference momentum
  self.pList = self.build_irrep_momentum_list() ## list of momenta
  if len(self.pList) != self.evec.shape[1]:
   raise DimensionError("dimension of evecs does not match expected irrep dimension!")
  pass

 ## from the representation matrices of the parent rep_object, construct the irrep matrices
 def build_irrep_matrices(self,repMat):
  if np.any([(repMat[0].shape[0] != x.shape[0]) or (repMat[0].shape[0] != x.shape[1])\
    for x in repMat]):
   raise DimensionError("representation matrix dimensions do not match!")
  if self.evec.shape[0] != repMat[0].shape[0]:
   print self.evec.shape[0],repMat[0].shape[0]
   raise DimensionError("dimension of evecs does not match representation matrix dimension!")
  if self.evec.shape[1] != len(self.pList):
   print self.evec.shape[1],len(self.pList)
   raise DimensionError("number of evecs does not match momentum list dimension!")
  return [chop(mconj(x,herm(self.evec))) for x in repMat]

 ## from the reference momentum, build the momentum list
 ## need to account for both orbit and littlegroup dimensions
 def build_irrep_momentum_list(self):
  plist = all_permutations(self.refMom)
  return expand_momentum_list(plist,LGDimension[self.irrepTag])

pass ## end class irrep

## general representation object
## not necessarily irreducible, decomposes itself before moving on
class rep_object:
 def __init__(self,repGen,pList,irrepTag=None,refMom=None):
  self.repGen = repGen ## representation generators
  self.pList = pList ## list of momentum and ordering
  if np.any([(self.repGen[0].shape[0] != x.shape[0]) or (self.repGen[0].shape[0] != x.shape[1])\
    for x in self.repGen]):
   raise DimensionError("representation generator dimensions do not match!")
  if len(self.pList) != self.repGen[0].shape[0]:
   raise DimensionError("dimension of momentum does not match representation matrix dimension!")
  self.irreps = [] ## list of computed irreps
  ## for tracking daughters later
  self.irrepTag=irrepTag
  if self.irrepTag is None:
    self.irrepTag='unnamed_irrep'
  self.refMom=refMom
  ## saved internally
  self.perm = None ## permutation puts momentum into block-diagonal order, though not irreps
  self.daughters = None ## pointers to daughter representations, if they exist
  self.fatherIrrepTag = None
  ## do the computation we've all been waiting for
  self.compute_irreps()
  pass

 def compute_irreps(self):
  ## some initial sorting steps
  pListSort = full_momentum_sort(self.pList)
  self.perm = momentum_permutation_block(self.pList)
  pListUnique = remove_duplicates( [reference_momentum(p) for p in pListSort] )
  permGen = [mconj(x,self.perm) for x in self.repGen]
  repFull = build_from_generators(*permGen) ## build all octahedral group rep matrices
  ## loop over all momentum present
  for p in pListUnique:
   pcan = canonical_momentum(p)
   pref = reference_momentum(p)
   ctab = get_littlegroup_character_table(pref)
   cent = get_littlegroup_centralizer(pref)
   ## restrict to only the subspace of momentum p
   ## evecs will be expanded back to full space before end
   momListCut = extract_momentum_subset(pListSort,pcan)
   momGen = [extract_momentum_block(pListSort,pcan,x) for x in permGen]
   momFull = build_from_generators(*momGen)
   momLG = get_littlegroup_matrices(pcan,momFull) ## cut out the non-LG matrices
   ## get the subspace for reference momentum only
   momLGRef = [extract_reference_block(momListCut,pref,x) for x in momLG]
   momLGGen = extract_generators_littlegroup(momLGRef,pref) ## get only the generators of LG
   pSlice = momSlice[momClass.index( classify_momentum_type(pref) )]
   for i,irTag in enumerate(rep_list_spinmom[pSlice]): ## loop over all possible irreps
    projMatrix = proj_matrix(momLGRef,ctab[i],cent)
    if all_zero(projMatrix):
     ## this irrep is not present in decomposition
     continue
    ## get the eigenvectors for non-degenerate irreps
    evList = construct_littlegroup_evec(momLGGen,pref,projMatrix,i)
    evList = [expand_littlegroup_evec(pListSort,pref,ev) for ev in evList] ## expand
    for ev in evList: ## compute full orbits of eigenvectors
     evOrbit = []
     for j in get_littlegroup_cosets(pref):
      ## x defined such that momentum p lands in pref slot, want inverse of that
      #x = herm(repFull[j])
      x = repFull[j]
      evOrbit.append(x.dot(ev))
     evOrbit = np.hstack(evOrbit) ## full orbit, all evecs!
     ## save complete irrep object
     self.irreps.append( irrep(irTag,evOrbit,pref) )

 ## compute a tensor product with another representation!
 ## if the other representation has daughters, raise an error
 ## nontrivial to handle due to eigenvector chaining
 def compute_tensor_product(self,repObj):
  if not(repObj.daughters is None):
   raise GroupTheoryError("cannot handle tensor product with daughters in second representation!")
  if self.daughters is None:
   self.fatherIrrepTag = repObj.irrepTag
   self.daughters = []
   for irrep in self.irreps:
    irGenM = irrep.build_irrep_matrices([mconj(x,self.perm) for x in self.repGen])
    repGenD = get_tensor(irGenM,repObj.repGen) ## new generators from repObj
    pListM = irrep.pList
    pListD = momentum_list_tensor(pListM,repObj.pList)
    ## build a new rep_object for a daughter
    daughter = rep_object(repGenD,pListD,irrepTag=irrep.irrepTag,refMom=irrep.refMom)
    self.daughters.append(daughter)
  else:
   for daughter in self.daughters:
    daughter.compute_tensor_product(repObj)

 ## print irreps in a clean way
 def list_irreps(self,**kwargs):
  if self.daughters is None:
   for i,x in enumerate(self.irreps):
    print ('%s %2d %s %s' % (kwargs.get('fmt',''),i,x.irrepTag,x.refMom) )
  else:
   for i,daughter in enumerate(self.daughters):
    print ('%s daughter %d: %s %s' % (kwargs.get('fmt',''),i,daughter.irrepTag,daughter.refMom) )
    daughter.list_irreps(fmt=(kwargs.get('fmt','')+' '))

 ## return the daughter irreps such that they can be
 ##  easily extracted with extract_irrep_from_product()
 def daughter_irreps(self):
  if not(self.daughters is None):
   rdict = {}
   for i,daughter in enumerate(self.daughters):
    for j,irrep in enumerate(daughter.irreps):
     rdict[irrep.irrepTag] = (i,j)
  else:
   raise GroupTheoryError("no daughter irreps!")
  return rdict

 ## return a copy of self
 def copy(self):
  return rep_object(self.repGen,self.pList,self.irrepTag,self.refMom)

 ## return a rep_object from a daughter irrep
 def copy_daughter_irrep(self,i,j):
  ## need to permute momenta too
  repGen = self.daughters[i].irreps[j].build_irrep_matrices(\
   [mconj(x,self.daughters[i].perm) for x in self.daughters[i].repGen])
  pList = self.daughters[i].irreps[j].pList
  irrepTag = self.daughters[i].irreps[j].irrepTag
  refMom = self.daughters[i].irreps[j].refMom
  return rep_object(repGen,pList,irrepTag,refMom)

 ## helper function to combine mother and daughter evecs
 def combine_evecs(self,dEvec,mTag=None,mEvec=None):
  if mEvec is None:
   return dEvec ## first mother
  dim = (1 if mTag is None else irrepDimension[ mTag ])
  shp = dEvec.shape
  dvec = dEvec.reshape((dim,shp[0]/dim,shp[1]))
  xvec = np.tensordot( mEvec, dvec, axes=(1,0) )
  shp = xvec.shape
  return xvec.reshape((shp[0]*shp[1],shp[2]))

 ## compute the combined eigenvector for all representations in the tensor product
 def compute_full_eigenvectors(self,mTag=None,mEvec=None):
  evecList = []
  tagList = []
  momList = []
  if self.daughters is None:
   ## last stop
   for irrep in self.irreps:
    ## undo momentum permutation
    evec = self.combine_evecs( herm(self.perm).dot(irrep.evec), mTag, mEvec)
    evecList.append(evec)
    tagList.append(irrep.irrepTag) ## only last tag matters
    momList.append(irrep.refMom) ## only last momentum matters too
  else:
   ## loop through all daughters
   for irrep,daughter in zip(self.irreps,self.daughters):
    ## combine with mother evecs and pass along to daughters
    evec = self.combine_evecs( herm(self.perm).dot(irrep.evec), mTag, mEvec)
    evec,tags,moms = daughter.compute_full_eigenvectors( irrep.irrepTag, evec )
    evecList = evecList + evec
    tagList = tagList + tags
    momList = momList + moms
  ## return
  return evecList,tagList,momList

 ## compute the tags to trace full ancestry of irrep
 def compute_long_taglist(self,mTag=None):
  tagList = []
  if self.daughters is None:
   ## last stop
   for irrep in self.irreps:
     if mTag is None:
       tag = ('%s[%d,%d,%d]' %((irrep.irrepTag,) +tuple( irrep.refMom)))
     else:
       tag = mTag +(' ->%s[%d,%d,%d]' %((irrep.irrepTag,) +tuple( irrep.refMom)))
     tagList.append( tag) ## only last tag matters
  else:
   ## loop through all daughters
   for irrep,daughter in zip(self.irreps,self.daughters):
    ## combine with mother tags and pass along to daughters
    if mTag is None:
      tag = ('(%s * %s)' %(self.irrepTag, self.fatherIrrepTag))
    else:
      tag = ('(%s ->%s[%d,%d,%d] * %s)' %(
       (mTag, irrep.irrepTag) +tuple( irrep.refMom) +( self.fatherIrrepTag,)))
    tagList += daughter.compute_long_taglist( tag)
  ## return
  return tagList

pass ## end class rep_object

## helper function for quickly creating reference irrep objects
def extract_irrep_from_product(rep0,rep1,irrepTag):
  ## copy to prevent overwriting
  r0 = rep0.copy()
  r1 = rep1.copy()
  r0.compute_tensor_product(r1)
  irrepDict = r0.daughter_irreps()
  return r0.copy_daughter_irrep(*irrepDict[irrepTag])

