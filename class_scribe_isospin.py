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
import os
import time

from defines import *
from manip_array import *
from manip_group import *
from manip_momentum import *
from manip_littlegroup import *
from character_tables import *
from class_irrep import *
from isospin import *
#from reference_reps import *

## build the representations that will be used in this file

## base momenta
p000 = [0,0,0]
p100 = [1,0,0]
p110 = [1,1,0]
p111 = [1,1,1]
p210 = [2,1,0]
p211 = [2,1,1]
p321 = [3,2,1]
## full lists of momenta
pl000 = all_permutations(p000)
pl100 = all_permutations(p100)
pl110 = all_permutations(p110)
pl111 = all_permutations(p111)
pl210 = all_permutations(p210)
pl211 = all_permutations(p211)
pl321 = all_permutations(p321)
## expanded 0-momentum lists
pl000_2 = expand_momentum_list(pl000,2)
pl000_3 = expand_momentum_list(pl000,3)
## generators
a1m_000 = [np.array([[1.]])]*2 +[np.array([[-1.]])] ## for quick parity change
a1p_000 = [np.array([[1.]])]*3
a1p_100 = [rep_matrix_mom(pl100,x) for x in [r12v,r23v,ppv]]
a1p_110 = [rep_matrix_mom(pl110,x) for x in [r12v,r23v,ppv]]
aap_111 = [rep_matrix_mom(pl111,x) for x in [r12v,r23v,ppv]]
aap_210 = [rep_matrix_mom(pl210,x) for x in [r12v,r23v,ppv]]
aap_211 = [rep_matrix_mom(pl211,x) for x in [r12v,r23v,ppv]]
aa0_321 = [rep_matrix_mom(pl321,x) for x in [r12v,r23v,ppv]]
## turn these generators into momentum irreps
rep_a1p_100 = rep_object(a1p_100,pl100,irrepTag='100_a1+')
rep_a1p_110 = rep_object(a1p_110,pl110,irrepTag='110_a1+')
rep_aap_111 = rep_object(aap_111,pl111,irrepTag='111_aa+')
rep_aap_210 = rep_object(aap_210,pl210,irrepTag='210_aa+')
rep_aap_211 = rep_object(aap_211,pl211,irrepTag='211_aa+')
rep_aa0_321 = rep_object(aa0_321,pl321,irrepTag='321_aa0')

## make trivial 0 momentum irreps into rep_objects
rep_a1p_000 = rep_object(a1p_000,pl000,irrepTag='000_a1+')
rep_a1m_000 = rep_object(a1m_000,pl000,irrepTag='000_a1-')

## build 0 momentum representation matrices from generators of G1-,T1-
g1m_000 = [r12,r23,pp]
t1m_000 = [r12v,r23v,ppv]
g1p_000 = get_tensor(g1m_000,a1m_000)
t1p_000 = get_tensor(t1m_000,a1m_000)
t2p_000 = [-r12v,-r23v,-ppv]
## make them into rep_objects
rep_g1p_000 = rep_object(g1p_000,pl000_2,irrepTag='000_g1+')
rep_t1p_000 = rep_object(t1p_000,pl000_3,irrepTag='000_t1+')
rep_g1m_000 = rep_object(g1m_000,pl000_2,irrepTag='000_g1-')
rep_t1m_000 = rep_object(t1m_000,pl000_3,irrepTag='000_t1-')
rep_t2p_000 = rep_object(t2p_000,pl000_3,irrepTag='000_t2+')

## for switch statements that will be used
switch_gamma_rep = {
 'Identity':     rep_a1p_000,
 'GammaT':       rep_a1p_000,
 'GammaI':       rep_t1m_000,
 'GammaIGammaT': rep_t1m_000,
 'GammaIGammaJ': rep_t1p_000,
 'GammaIGamma5': rep_t1p_000,
 'GammaTGamma5': rep_a1m_000,
 'Gamma5':       rep_a1m_000 }
switch_gamma_repname = {
 'Identity':     '000_a1+',
 'GammaT':       '000_a1+',
 'GammaI':       '000_t1-',
 'GammaIGammaT': '000_t1-',
 'GammaIGammaJ': '000_t1+',
 'GammaIGamma5': '000_t1+',
 'GammaTGamma5': '000_a1-',
 'Gamma5':       '000_a1-' }
switch_gamma_tag = {
 'Identity':     [''],
 'GammaT':       ['GAMMA 3'],
 'GammaI':       ['GAMMA 0','GAMMA 1','GAMMA 2'],
 'GammaIGammaT': [ x +'\n GAMMA 3' for x in ['GAMMA 0','GAMMA 1','GAMMA 2']],
 'GammaIGammaJ': ['GAMMA 1\n GAMMA 2', 'GAMMA 2\n GAMMA 0', 'GAMMA 0\n GAMMA 1'],
 'GammaIGamma5': [ x +'\n GAMMA 5' for x in ['GAMMA 0','GAMMA 1','GAMMA 2']],
 'GammaTGamma5': ['GAMMA 3\n GAMMA 5'],
 'Gamma5':       ['GAMMA 5'] }
switch_gamma_herm_sign= {
 'Identity':      1.,
 'GammaT':        1.,
 'GammaI':       -1.,
 'GammaIGammaT':  1.,
 'GammaIGammaJ': -1.,
 'GammaIGamma5': -1.,
 'GammaTGamma5':  1.,
 'Gamma5':       -1. }

switch_momentum_rep = {
 classify_momentum_type(reference_momentum(p000)): rep_a1p_000,
 classify_momentum_type(reference_momentum(p100)): rep_a1p_100,
 classify_momentum_type(reference_momentum(p110)): rep_a1p_110,
 classify_momentum_type(reference_momentum(p111)): rep_aap_111,
 classify_momentum_type(reference_momentum(p210)): rep_aap_210,
 classify_momentum_type(reference_momentum(p211)): rep_aap_211,
 classify_momentum_type(reference_momentum(p321)): rep_aa0_321 }

## list of fields to test for classes
allowed_arguments = set(['gamma','reference_momentum','antiquark_flavor','quark_flavor',\
 'antiquark_modifier','quark_modifier','t','factor','hermitian_conjugate','deriv_operator'])
allowed_flavors = set(['U','D'])
allowed_modifiers = set(['local'])

class meson_scribe:
 #def __init__(self,flavor0,flavor1,gamma,refMom,time,factor):
 def __init__(self,**kwargs):
  self.gamma = kwargs.get('gamma','Identity')
  self.refMom = reference_momentum(kwargs.get('reference_momentum',p000))
  self.hermConj = kwargs.get('hermitian_conjugate',False)
  self.flavor0 = kwargs.get('antiquark_flavor','U')
  self.flavor1 = kwargs.get('quark_flavor','U')
  self.modifier0 = kwargs.get('antiquark_modifier','')
  self.modifier1 = kwargs.get('quark_modifier','')
  self.derivOp = kwargs.get('deriv_operator',None)
  self.time = kwargs.get('t','t')
  if self.derivOp is None:
    self.pList = all_permutations( self.refMom) ## for generate_term
  else:
    self.pList = momentum_list_tensor( all_permutations( self.refMom),
     expand_momentum_list( pl000, LGDimension[ switch_deriv_repname[ self.derivOp]]))
  #self.factor = kwargs.get('factor',1.)
  ## some checks
  if not( self.flavor0 in allowed_flavors ):
   raise ValueError("antiquark flavor 0 not in allowed flavors!")
  if not( self.flavor1 in allowed_flavors ):
   raise ValueError("quark flavor 1 not in allowed flavors!")
  if not( self.modifier0 in allowed_modifiers ):
   raise ValueError("antiquark modifier 0 not in allowed modifiers!")
  if not( self.modifier1 in allowed_modifiers ):
   raise ValueError("quark modifier 1 not in allowed modifiers!")
  if not( self.gamma in switch_gamma_tag.keys() ):
   raise ValueError("gamma not in allowed gamma keys!")
  pass

 #def generate_term(self,factor,i,**kwargs):
 def generate_term(self,i,**kwargs):
  gList = switch_gamma_tag[self.gamma]
  pList = self.pList
  glen = len( gList)
  ip = i / glen
  ig = i % glen
  ### generate just the term text, factor comes later
  term = ''
  if self.hermConj:
    term = term + (' %sBAR %s %s\n ' % (self.flavor1, self.time, self.modifier0))
  else:
    term = term + (' %sBAR %s %s\n ' % (self.flavor0, self.time, self.modifier0))
  if self.gamma != 'Identity':
   term = term + ('%s\n ' % gList[ig] )
  if not( self.derivOp is None):
   term = term +('MOM%s [%d,%d,%d,%s/%d] %s\n ' % ( ('DAG' if self.hermConj else ''),
    pList[ip][0],pList[ip][1],pList[ip][2],self.derivOp,ip,self.time))
  elif self.refMom != [0,0,0]:
   if self.hermConj:
     term = term + ('MOMDAG [%d,%d,%d,mom] %s\n ' % (
      pList[ip][0],pList[ip][1],pList[ip][2],self.time) )
   else:
     term = term + ('MOM [%d,%d,%d,mom] %s\n ' % (
      pList[ip][0],pList[ip][1],pList[ip][2],self.time) )
  if self.hermConj:
    term = term + ('%s %s %s\n' % (self.flavor0, self.time, self.modifier1))
  else:
    term = term + ('%s %s %s\n' % (self.flavor1, self.time, self.modifier1))
  #
  return term

 def generate_term_list(self,**kwargs):
  if self.derivOp is None:
    nterm = len( switch_gamma_tag[self.gamma]) *len( self.pList)
  else:
    nterm = len( switch_gamma_tag[self.gamma]) \
     *irrepDimension[ switch_deriv_repname[ self.derivOp]]
  return [ self.generate_term(i,**kwargs) for i in range(nterm) ]

## build a bunch of operators with different isospin
## inputs are now arrays rather than values, each array will be used to generate
##  a separate meson object
## isospin inferred from the number of terms in lists
isospin_wrapper_allowed_arguments = [
 'gamma','reference_momentum','hermitian_conjugate','deriv_operator']
switch_gamma_string = {
 'Identity':     '1',
 'GammaT':       't',
 'GammaI':       'i',
 'GammaIGammaT': 'it',
 'GammaIGammaJ': 'ij',
 'GammaIGamma5': 'i5',
 'GammaTGamma5': 't5',
 'Gamma5':       '5' }

### this file is written as plaintext in the python dictionary syntax
#with open( 'deriv_op_file', 'r') as f:
#  switch_deriv_repname = eval( f.read())
switch_deriv_repname = {} ## dummy

switch_reference_rep = {}
#for irrep in rep_list_spinmom: ## requires import reference_reps
#  switch_reference_rep[irrep] = eval( 'rep_' +irrepCleanName[ irrep])

class meson_isospin_wrapper:
 def __init__(self,**kwargs):
  ## checks
  if not( all([key in isospin_wrapper_allowed_arguments for key in kwargs.keys()]) ):
   mkey = kwargs.keys()[[key in isospin_wrapper_allowed_arguments\
    for key in kwargs.keys()].index(False)]
   raise ValueError("unknown keyword argument \""+mkey+"\"!")
  ## attributes
  self.gamma = kwargs.get('gamma',[])
  self.refMom = kwargs.get('reference_momentum',[])
  self.hermConj = kwargs.get('hermitian_conjugate',False)
  self.derivOp = kwargs.get('deriv_operator',None)
  self.refMom = [reference_momentum(x) for x in self.refMom] ## fix to reference momentum
  self.isospin = len(self.gamma)
  self.nmeson = [] ## list of meson objects
  if self.isospin < 1:
   raise ValueError("no gamma matrices included for mesons!")
  if len(self.gamma) != len(self.refMom):
   raise ValueError("different numbers of momenta and gammas specified!")
  if not( self.derivOp is None) and len(self.gamma) != len(self.derivOp):
   raise ValueError("different numbers of derivative operators and gammas specified!")
  ## do the isospin algebra
  self.isoSet = isospin_set_full_components(self.isospin)
  self.kwdef = {'antiquark_modifier':'local', 'quark_modifier':'local', 't':'t', 'factor':1.}
  ## set up the class objects
  startTime = time.time()
  print "-starting group theory computation"
  if self.derivOp is None:
    repGenDeriv = [ switch_momentum_rep[ classify_momentum_type(p)].repGen \
     for p in self.refMom ]
    momListDeriv = [ all_permutations( p) for p in self.refMom]
  else:
    ## do check of derivative operators and momentum class
    for i,(p,dop) in enumerate( zip(self.refMom,self.derivOp)):
      if dop is None:
        continue
      dop = switch_deriv_repname[ dop]
      ixd = rep_list_spinmom.index( dop)
      ixp = momClass.index( classify_momentum_type( p))
      if not( ixd in range( 44)[ momSlice[ ixp]]):
        raise ValueError("reference momentum does not match derivative operator momentum class!")
    repGenDeriv = [ ( switch_momentum_rep[ classify_momentum_type(p)].repGen \
     if dop is None else switch_reference_rep[ switch_deriv_repname[ dop]].repGen ) \
     for p,dop in zip( self.refMom, self.derivOp) ]
    momListDeriv = [ (all_permutations(p) \
     if dop is None else momentum_list_tensor( all_permutations(p), \
     expand_momentum_list(pl000, LGDimension[ switch_deriv_repname[dop] ])) \
     ) for dop,p in zip(self.derivOp,self.refMom) ]
  ## representation generators for the combined gamma + deriv + momentum representation
  repGen = [ get_tensor( pgen, switch_gamma_rep[g].repGen ) \
   for g,pgen in zip(self.gamma,repGenDeriv) ]
  momList = [ momentum_list_tensor( pl, \
   expand_momentum_list(pl000, LGDimension[ switch_gamma_repname[g] ]) \
   ) for g,pl in zip(self.gamma,momListDeriv) ]
  self.repGen = repGen
  self.momList = momList
  ## assign to class attributes
  self.repObjAll = [ rep_object(g,p) for g,p in zip(repGen,momList) ]
  self.repObjMother = self.repObjAll[0].copy()
  print "-starting tensor products"
  for itns,x in enumerate(self.repObjAll[1:]):
   print "-- tensor product "+str(itns)
   self.repObjMother.compute_tensor_product(x)
  endTime = time.time()
  print "-done with group theory computation"
  print "-time: "+str(np.round(endTime-startTime,2))

 ## do the full generation for one specific combination of flavors
 def generate_flavor_term(self,evecList,tagList,momList,factor,scribes,**kwargs):
  ## do this the stupid way, doesn't take too much memory
  termList = [ scribe.generate_term_list(**kwargs) for scribe in scribes ]
  hermFac = ( np.product([ switch_gamma_herm_sign[scribe.gamma] for scribe in scribes ])
    if self.hermConj else 1. )
  fullTermList = termList[0]
  for tl in termList[1:]:
   nextTermList = []
   for xterm in fullTermList:
    for term in tl:
     nextTermList.append(xterm+term)
   fullTermList = nextTermList
  ## put in the factors 
  output_operators = {}
  for filei,(evl,tag,pref) in enumerate(zip(evecList,tagList,momList)):
   pList = expand_momentum_list( all_permutations(pref), LGDimension[ tag ] )
   if len(pList) != evl.shape[1]:
    raise ValueError("momentum list and eigenvector length do not match!")
   for i,(mom,ev) in enumerate( zip(pList, evl.T) ):
     ev = factor*ev
     hermStr = (' (herm)' if self.hermConj else '')
     newOp = '# '+tag+hermStr+' p='+','.join([str(x) for x in mom])\
      +' LG_index='+str(i %LGDimension[tag]) +'\n'
     for term,vfactor in zip(fullTermList,ev):
       vfactor = vfactor * hermFac
       ## don't include zero factors or nan
       if np.abs(vfactor) < smallNum or not(vfactor == vfactor):
        continue
       ## add the factor for the term
       if isinstance(vfactor,complex):
         if np.abs(vfactor.imag) > smallNum:
           if self.hermConj:
             newOp = newOp + ('\nFACTOR %s %s\n' % (vfactor.real, -vfactor.imag))
           else:
             newOp = newOp + ('\nFACTOR %s %s\n' % (vfactor.real, vfactor.imag))
         else:
           newOp = newOp + ('\nFACTOR %s\n' % vfactor.real)
       else:
         newOp = newOp + ('\nFACTOR %s\n' % vfactor)
       ## add the term
       newOp = newOp + term
       pass
     ## spit the output so it can be combined with isospin operations
     pcan = ''.join([str(x) for x in canonical_momentum(mom)])
     irName = irrepCleanName[tag][:4]+pcan
     irRow = i
     irID = filei
     key = (irID,irName,irRow)
     output_operators[key] = newOp
  return output_operators

 ## for each isospin term, do the generation of all flavor terms
 ## set up the scribes for these flavors, then call the generate_flavor_term() for each
 def generate_isospin_term(self,evecList,tagList,momList,isoTerm,**kwargs):
  iso_output_operators = {}
  factorList = [] ## list of factors to go with each scribe list
  scribeList = [] ## list of sets of scribes to use
  ## generate the full set of scribes and associated factors factors
  ## this would be so easy if it weren't for the DD-UU combinations
  derivOp = ([None for i in range( len( self.gamma))]  if self.derivOp is None else self.derivOp)
  for i,(g,p,dop,iso) in enumerate(zip(self.gamma,self.refMom,derivOp,isoTerm[1])):
   nextFactorList = factorList
   nextScribeList = scribeList
   kwnext = dict(self.kwdef) ## copy the default dictionary
   kwnext['gamma'] = g
   kwnext['reference_momentum'] = p
   kwnext['hermitian_conjugate'] = self.hermConj
   kwnext['deriv_operator'] = dop
   #kwnext['t'] = ('t%d' % i)
   if   iso == 1: ## +DbarU
     kwnext['antiquark_flavor'] = 'D'
     kwnext['quark_flavor'] = 'U'
     if i == 0:
      factorList = [isoTerm[0]]
      scribeList = [[meson_scribe(**kwnext)]]
     else:
      #factorList = nextFactorList ## no change
      scribeList = [x+ [meson_scribe(**kwnext)] for x in nextScribeList]
   elif iso == -1: ## -UbarD
     kwnext['antiquark_flavor'] = 'U'
     kwnext['quark_flavor'] = 'D'
     if i == 0:
      factorList = [-isoTerm[0]]
      scribeList = [[meson_scribe(**kwnext)]]
     else:
      factorList = [-x for x in nextFactorList] ## multiply all terms by -1
      scribeList = [x+ [meson_scribe(**kwnext)] for x in nextScribeList]
   elif iso == 0: ## (+DbarD -UbarU)/np.sqrt(2.)
     kwnext['antiquark_flavor'] = 'D'
     kwnext['quark_flavor'] = 'D'
     scb0 = [meson_scribe(**kwnext)]
     kwnext['antiquark_flavor'] = 'U'
     kwnext['quark_flavor'] = 'U'
     scb1 = [meson_scribe(**kwnext)]
     if i == 0:
      factorList = [isoTerm[0]/sr2,-isoTerm[0]/sr2]
      scribeList = [scb0,scb1]
     else:
      factorList = [x/sr2 for x in nextFactorList] + [-x/sr2 for x in nextFactorList]
      scribeList = [x+scb0 for x in nextScribeList] + [x+scb1 for x in nextScribeList]
  ## append all the isospin terms
  for factor,sList in zip(factorList,scribeList):
   if np.abs(factor) < smallNum:
    continue
   output_operators = self.generate_flavor_term(evecList,tagList,momList,factor,sList,**kwargs)
   for key in output_operators:
    iso_output_operators[key] = iso_output_operators.get(key,'') +'\n'+ output_operators[key]
  return iso_output_operators

 ## generate the full operators, including all isospins
 def generate(self,**kwargs):
  ## generate all the terms, eigenvectors, tags
  evecList,tagList,momList = self.repObjMother.compute_full_eigenvectors()
  iso_output_operators = {}
  startTime = time.time()
  for isokey in self.isoSet.keys():
   if not( isinstance( isokey, tuple)): ## need all components, not just I3=0
    continue
   for i,isoOp in enumerate(self.isoSet[isokey]):
    ## build each isospin operator individually
    for isoTerm in isoOp:
     output_operators = self.generate_isospin_term(evecList,tagList,momList,isoTerm,**kwargs)
     endTime = time.time()
     print "- generate_isospin_term time: "+str(np.round(endTime-startTime,2))
     for key in output_operators:
      ikey = (str(isokey[0]),str(isokey[1]),str(i),)+key
      iso_output_operators[ikey] = iso_output_operators.get(ikey,'') + output_operators[key]
  ## write to files
  startTime = time.time()
  for ikey in iso_output_operators:
   isokey = ikey[0]
   isocomp = ikey[1]
   i = ikey[2]
   irID = ikey[3]
   irName = ikey[4]
   irRow = ikey[5]
   pcom = irName[-3:]
   gamstr = 'g'+'g'.join([switch_gamma_string[g] for g in self.gamma])
   pstr = ('p'+'p'.join([''.join([str(x) for x in canonical_momentum(mom)]) for mom in self.refMom])
    if self.derivOp is None else '')
   dstr = ''
   if not(self.derivOp is None):
     dstr = ''
     for dop in self.derivOp:
       if dop is None:
         dop = dop +'d0a1pp000c00'
       else:
         dspl = dop.split('_')
         dstr = dstr +''.join([ dspl[0],dspl[1],'p',dspl[2],'c',dspl[3] ])
   ## build the directory structure
   directory = 'output_baseop00/' +('i%sc%s'%(str(ikey[0]),str(ikey[1]))) \
     +'/pcom'+pcom+'/'+gamstr+pstr+dstr
   if not os.path.exists(directory):
     os.makedirs(directory)
   if self.hermConj:
     fsuffix = '.hm'
   else:
     fsuffix = '.op'
   f = open(directory+'/'+irName+'.'+str(1000*int(i)+int(irID)).zfill(5)\
    +'.'+str(irRow).zfill(2)+fsuffix,'w')
   f.write( iso_output_operators[ikey] )
   f.close()
  endTime = time.time()
  print "- file write time: "+str(np.round(endTime-startTime,2))

## compute single bilinear operators, no derivatives
gamma_list = switch_gamma_string.keys() ## should be all of them
momentum_list = [[0,0,0],[1,0,0],[1,1,0],[1,1,1],[2,0,0]]
kwargs = {}
kwargs['hermitian_conjugate'] = False
for g in gamma_list:
 for p in momentum_list:
   ptag = ''.join([ str(x) for x in p ])
   kwargs['gamma'] = [g]
   kwargs['reference_momentum'] = [p]
   kwargs['deriv_operator'] = None
   print g+' momentum ' +str(p)
   mi1 = meson_isospin_wrapper(**kwargs)
   mi1.generate() ## write to files

