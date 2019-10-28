import numpy as np

from defines import *
from manip_array import *
from manip_group import *
from manip_momentum import *
from manip_littlegroup import *
from character_tables import *
from class_irrep import *

## build trivial momentum representations
#
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

## make trivial 0 momentum irreps into rep_objects
rep_a1p_000 = rep_object(a1p_000,pl000,irrepTag='000_a1+')
rep_a1m_000 = rep_object(a1m_000,pl000,irrepTag='000_a1-')

## build 0 momentum representation matrices from generators of G1-,T1-
g1m_000 = [r12,r23,pp]
t1m_000 = [r12v,r23v,ppv]
g1p_000 = get_tensor(g1m_000,a1m_000)
t1p_000 = get_tensor(t1m_000,a1m_000)
## make them into rep_objects
rep_g1p_000 = rep_object(g1p_000,pl000_2,irrepTag='000_g1+')
rep_t1p_000 = rep_object(t1p_000,pl000_3,irrepTag='000_t1+')
rep_t1m_000 = rep_object(t1m_000,pl000_3,irrepTag='000_t1-')

## build +parity fermionic representations of zero momentum
rep_hhp_000 = extract_irrep_from_product(rep_g1p_000,rep_t1p_000,'000_hh+')
rep_g2p_000 = extract_irrep_from_product(rep_hhp_000,rep_t1p_000,'000_g2+')
rep_g2m_000 = extract_irrep_from_product(rep_g2p_000,rep_a1m_000,'000_g2-')
rep_hhm_000 = extract_irrep_from_product(rep_hhp_000,rep_a1m_000,'000_hh-')

## build +parity bosonic representations of zero momentum
## can get multiple at once, so do that
r0 = rep_t1p_000.copy()
r1 = rep_t1p_000.copy()
r0.compute_tensor_product(r1)
rdict = r0.daughter_irreps()
rep_eep_000 = r0.copy_daughter_irrep(*rdict['000_ee+'])
rep_t2p_000 = r0.copy_daughter_irrep(*rdict['000_t2+'])
rep_a2p_000 = extract_irrep_from_product(rep_eep_000,rep_eep_000,'000_a2+')
rep_a2m_000 = extract_irrep_from_product(rep_a2p_000,rep_a1m_000,'000_a2-')
rep_eem_000 = extract_irrep_from_product(rep_eep_000,rep_a1m_000,'000_ee-')
rep_t2m_000 = extract_irrep_from_product(rep_t2p_000,rep_a1m_000,'000_t2-')

## build the trivial nonzero momentum irreps
rep_a1p_100 = rep_object(a1p_100,pl100,irrepTag='100_a1+')
rep_a1p_110 = rep_object(a1p_110,pl110,irrepTag='110_a1+')
rep_aap_111 = rep_object(aap_111,pl111,irrepTag='111_aa+')
rep_aap_210 = rep_object(aap_210,pl210,irrepTag='210_aa+')
rep_aap_211 = rep_object(aap_211,pl211,irrepTag='211_aa+')
rep_aa0_321 = rep_object(aa0_321,pl321,irrepTag='321_aa0')

### build nontrivial nonzero momentum irreps
## 100 momentum
rep_a2p_100 = extract_irrep_from_product(rep_a1p_100,rep_a2p_000,'100_a2+')
rep_bb0_100 = extract_irrep_from_product(rep_a1p_100,rep_t1p_000,'100_bb0')
rep_g10_100 = extract_irrep_from_product(rep_g1p_000,rep_a1p_100,'100_g10')
rep_g20_100 = extract_irrep_from_product(rep_g2p_000,rep_a1p_100,'100_g20')
rep_a1m_100 = extract_irrep_from_product(rep_a1p_100,rep_a1m_000,'100_a1-')
rep_a2m_100 = extract_irrep_from_product(rep_a2p_100,rep_a1m_000,'100_a2-')
## 110 momentum
rep_a2p_110 = extract_irrep_from_product(rep_a1p_110,rep_a2p_000,'110_a2+')
rep_gg0_110 = extract_irrep_from_product(rep_a1p_110,rep_g1p_000,'110_gg0')
rep_a1m_110 = extract_irrep_from_product(rep_a1p_110,rep_a1m_000,'110_a1-')
rep_a2m_110 = extract_irrep_from_product(rep_a2p_110,rep_a1m_000,'110_a2-')
### 111 momentum
r0 = rep_hhp_000.copy()
r1 = rep_aap_111.copy()
r0.compute_tensor_product(r1) ## TODO: fix
rdict = r0.daughter_irreps()
rep_llp_111 = r0.copy_daughter_irrep(*rdict['111_ll+'])
rep_llm_111 = r0.copy_daughter_irrep(*rdict['111_ll-'])
rep_gg0_111 = r0.copy_daughter_irrep(*rdict['111_gg0'])
rep_bb0_111 = extract_irrep_from_product(rep_aap_111,rep_eep_000,'111_bb0')
rep_aam_111 = extract_irrep_from_product(rep_aap_111,rep_a1m_000,'111_aa-')
## 210 momentum
r0 = rep_g1p_000.copy()
r1 = rep_aap_210.copy()
r0.compute_tensor_product(r1)
rdict = r0.daughter_irreps()
rep_llp_210 = r0.copy_daughter_irrep(*rdict['210_ll+'])
rep_llm_210 = r0.copy_daughter_irrep(*rdict['210_ll-'])
rep_aam_210 = extract_irrep_from_product(rep_aap_210,rep_a1m_000,'210_aa-')
## 211 momentum
r0 = rep_g1p_000.copy()
r1 = rep_aap_211.copy()
r0.compute_tensor_product(r1)
rdict = r0.daughter_irreps()
rep_llp_211 = r0.copy_daughter_irrep(*rdict['211_ll+'])
rep_llm_211 = r0.copy_daughter_irrep(*rdict['211_ll-'])
rep_aam_211 = extract_irrep_from_product(rep_aap_211,rep_a1m_000,'211_aa-')
## 321 momentum
rep_ll0_321 = extract_irrep_from_product(rep_aa0_321,rep_g1p_000,'321_ll0')

### get indices of little groups for each momentum in orbit
### sorted by how momentum are rotated by each element of octahedral group
#t1m_000_rev = build_reverse_from_generators(*t1m_000)
#t1m_000_full = build_from_generators(*t1m_000)
#pLGdict = {}
#for p in [p000,p100,p110,p111,p210,p211,p321]:
# pref = reference_momentum(p)
# plist = [[int(y) for y in x.dot(pref)] for x in t1m_000_rev]
# for i,px in enumerate(plist):
#  if not( tuple(px) in pLGdict):
#    pLGdict[tuple(px)] = []
#  pLGdict[tuple(px)].append(i)
#
### dictionary of coset indices
### used later to build representation eigenvectors
#pCoset = {}
#for pl in [pl000,pl100,pl110,pl111,pl210,pl211,pl321]:
# pref = tuple(reference_momentum(pl[0]))
# pCoset[pref] = [pLGdict[tuple(p)][0] for p in pl]
# print pref,pCoset[pref]

### full representation matrices
#a1m_000_full = build_from_generators(*a1m_000)
#a1p_000_full = build_from_generators(*a1p_000)
#a1p_100_full = build_from_generators(*a1p_100)
#a1p_110_full = build_from_generators(*a1p_110)
#aap_111_full = build_from_generators(*aap_111)
#aap_210_full = build_from_generators(*aap_210)
#aap_211_full = build_from_generators(*aap_211)
#aa0_321_full = build_from_generators(*aa0_321)
##
#g1m_000_full = build_from_generators(*g1m_000)
#g1p_000_full = build_from_generators(*g1p_000)
#t1m_000_full = build_from_generators(*t1m_000)
#t1p_000_full = build_from_generators(*t1p_000)

#r0p = rep_g1p_000.copy()
#r0 = r0p.copy()
#r0.compute_tensor_product( rep_a1m_000.copy())
#r0m = r0.get_daughter_irrep(0,0)
#
#r0 = r0p.copy()
##r1 = r0p.copy()
##r1 = r0m.copy()
#r1 = rep_t1p_000.copy()
#r0.compute_tensor_product( r1) ## r0 indices increment slowest

#p200 = [2,0,0]
#pl200 = all_permutations(p200)
#a1p_200 = [rep_matrix_mom(pl200,x) for x in [r12v,r23v,ppv]]
#rep_a1p_200 = rep_object(a1p_100,pl200,irrepTag='100_a1+')
#
#rmom = [rep_a1p_000.copy(), rep_a1p_100.copy(), rep_a1p_110.copy(),
#  rep_aap_111.copy(), rep_aap_210.copy(), rep_aap_211.copy(),
#  rep_aa0_321.copy(), rep_a1p_200.copy()]
#pdef = [p000, p100, p110, p111, p210, p211, p321, p200]
#rdefN = [rep_g1p_000.copy() for i in range( len( rmom))]
#rdefD = [rep_hhp_000.copy() for i in range( len( rmom))]
#rdefM = [rep_a1m_000.copy() for i in range( len( rmom))]
#
### moving frames for single particles
#for x0,x1,x2,y in zip(rdefN,rdefD,rdefM,rmom):
#  x0.compute_tensor_product( y)
#  x1.compute_tensor_product( y)
#  x2.compute_tensor_product( y)
#
### organize into dictionary
#pd = {}
#Nkeys = []
#Dkeys = []
#Mkeys = []
#
#for x0,x1,x2,y in zip(rdefN,rdefD,rdefM,pdef):
#  nkey = ('N',)  +tuple(y)
#  dkey = ('D',)  +tuple(y)
#  mkey = ('pi',) +tuple(y)
#  pd[nkey] = x0
#  pd[dkey] = x1
#  pd[mkey] = x2
#  Nkeys.append( nkey)
#  Dkeys.append( dkey)
#  Mkeys.append( mkey)
#
##for nkey in Nkeys:
##  rdict = pd[ nkey].daughter_irreps()
##  for pkey in sorted( rdict.keys()):
##    for mkey in Mkeys:
##      r1 = pd[ nkey].copy_daughter_irrep( *rdict[ pkey])
##      r2 = pd[ mkey].copy_daughter_irrep(0,0) ## there can only be one
##      r1.compute_tensor_product( r2)
##      pd[ nkey +(pkey,) +mkey[1:]] = r1
##      print nkey +(pkey,) +mkey[1:]
##      r1.list_irreps()
#
#keyPairs = []
### 000
#keyPairs.append( (('N',) +tuple( p000), ('pi',) +tuple( p000)) )
#keyPairs.append( (('N',) +tuple( p100), ('pi',) +tuple( p100)) )
#keyPairs.append( (('N',) +tuple( p110), ('pi',) +tuple( p110)) )
#keyPairs.append( (('N',) +tuple( p111), ('pi',) +tuple( p111)) )
#keyPairs.append( (('N',) +tuple( p200), ('pi',) +tuple( p200)) )
### 100
#keyPairs.append( (('N',) +tuple( p100), ('pi',) +tuple( p000)) )
#keyPairs.append( (('N',) +tuple( p000), ('pi',) +tuple( p100)) )
#keyPairs.append( (('N',) +tuple( p110), ('pi',) +tuple( p100)) )
#keyPairs.append( (('N',) +tuple( p200), ('pi',) +tuple( p100)) )
#keyPairs.append( (('N',) +tuple( p100), ('pi',) +tuple( p110)) )
#keyPairs.append( (('N',) +tuple( p111), ('pi',) +tuple( p110)) )
#keyPairs.append( (('N',) +tuple( p110), ('pi',) +tuple( p111)) )
#keyPairs.append( (('N',) +tuple( p100), ('pi',) +tuple( p200)) )
### 110
#keyPairs.append( (('N',) +tuple( p110), ('pi',) +tuple( p000)) )
#keyPairs.append( (('N',) +tuple( p111), ('pi',) +tuple( p100)) )
#keyPairs.append( (('N',) +tuple( p000), ('pi',) +tuple( p110)) )
#keyPairs.append( (('N',) +tuple( p200), ('pi',) +tuple( p110)) )
#keyPairs.append( (('N',) +tuple( p100), ('pi',) +tuple( p111)) )
#keyPairs.append( (('N',) +tuple( p110), ('pi',) +tuple( p200)) )
##keyPairs.append( (('N',) +tuple( p100), ('pi',) +tuple( p100)) )
##keyPairs.append( (('N',) +tuple( p110), ('pi',) +tuple( p110)) )
### 111
#keyPairs.append( (('N',) +tuple( p111), ('pi',) +tuple( p000)) )
#keyPairs.append( (('N',) +tuple( p110), ('pi',) +tuple( p100)) )
#keyPairs.append( (('N',) +tuple( p100), ('pi',) +tuple( p110)) )
#keyPairs.append( (('N',) +tuple( p000), ('pi',) +tuple( p111)) )
#keyPairs.append( (('N',) +tuple( p111), ('pi',) +tuple( p200)) )
#
### 000
#keyPairs.append( (('D',) +tuple( p000), ('pi',) +tuple( p000)) )
#keyPairs.append( (('D',) +tuple( p100), ('pi',) +tuple( p100)) )
#keyPairs.append( (('D',) +tuple( p110), ('pi',) +tuple( p110)) )
#keyPairs.append( (('D',) +tuple( p111), ('pi',) +tuple( p111)) )
#keyPairs.append( (('D',) +tuple( p200), ('pi',) +tuple( p200)) )
### 100
#keyPairs.append( (('D',) +tuple( p100), ('pi',) +tuple( p000)) )
#keyPairs.append( (('D',) +tuple( p000), ('pi',) +tuple( p100)) )
#keyPairs.append( (('D',) +tuple( p110), ('pi',) +tuple( p100)) )
#keyPairs.append( (('D',) +tuple( p200), ('pi',) +tuple( p100)) )
#keyPairs.append( (('D',) +tuple( p100), ('pi',) +tuple( p110)) )
#keyPairs.append( (('D',) +tuple( p111), ('pi',) +tuple( p110)) )
#keyPairs.append( (('D',) +tuple( p110), ('pi',) +tuple( p111)) )
#keyPairs.append( (('D',) +tuple( p100), ('pi',) +tuple( p200)) )
### 110
#keyPairs.append( (('D',) +tuple( p110), ('pi',) +tuple( p000)) )
#keyPairs.append( (('D',) +tuple( p111), ('pi',) +tuple( p100)) )
#keyPairs.append( (('D',) +tuple( p000), ('pi',) +tuple( p110)) )
#keyPairs.append( (('D',) +tuple( p200), ('pi',) +tuple( p110)) )
#keyPairs.append( (('D',) +tuple( p100), ('pi',) +tuple( p111)) )
#keyPairs.append( (('D',) +tuple( p110), ('pi',) +tuple( p200)) )
#keyPairs.append( (('D',) +tuple( p100), ('pi',) +tuple( p100)) )
#keyPairs.append( (('D',) +tuple( p110), ('pi',) +tuple( p110)) )
### 111
#keyPairs.append( (('D',) +tuple( p111), ('pi',) +tuple( p000)) )
#keyPairs.append( (('D',) +tuple( p110), ('pi',) +tuple( p100)) )
#keyPairs.append( (('D',) +tuple( p100), ('pi',) +tuple( p110)) )
#keyPairs.append( (('D',) +tuple( p000), ('pi',) +tuple( p111)) )
#keyPairs.append( (('D',) +tuple( p111), ('pi',) +tuple( p200)) )
#
#builtKeys = []
#for nkey,mkey in keyPairs:
#  rdict = pd[ nkey].daughter_irreps()
#  for pkey in sorted( rdict.keys()):
#    r1 = pd[ nkey].copy_daughter_irrep( *rdict[ pkey])
#    r2 = pd[ mkey].copy_daughter_irrep(0,0) ## there can only be one
#    r1.compute_tensor_product( r2)
#    pd[ nkey +(pkey,) +mkey[1:]] = r1
#    builtKeys.append( nkey +(pkey,) +mkey[1:])
#    print nkey +(pkey,) +mkey[1:]
#    r1.list_irreps()
#
#printKeys = []
#printKeys.append(( '000_g1+', 0))
#printKeys.append(( '000_hh+', 0))
#printKeys.append(( '100_g10', 1))
#printKeys.append(( '100_g20', 1))
#printKeys.append(( '110_gg0', 2))
#printKeys.append(( '111_gg0', 2))
#
#norm2 = lambda x: np.inner(x,x)
#
#for pkey,p2 in printKeys:
#  for key in builtKeys:
#    if len( key) < 8:
#      continue
#    rdict = pd[ key].daughter_irreps()
#    if not( pkey in rdict):
#      continue
#    ict = [x.irrepTag for x in pd[ key].daughters[0].irreps
#      if norm2( x.refMom) == p2].count( pkey)
#    if ict > 0:
#      print ('%30s %10s %d' % (key,pkey,ict))

