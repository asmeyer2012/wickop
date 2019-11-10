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
a1x_100 = [rep_matrix_mom(pl100,x) for x in [r12v,r23v,ppv]]
a1x_110 = [rep_matrix_mom(pl110,x) for x in [r12v,r23v,ppv]]
a1x_111 = [rep_matrix_mom(pl111,x) for x in [r12v,r23v,ppv]]
a1x_210 = [rep_matrix_mom(pl210,x) for x in [r12v,r23v,ppv]]
a1x_211 = [rep_matrix_mom(pl211,x) for x in [r12v,r23v,ppv]]
a1x_321 = [rep_matrix_mom(pl321,x) for x in [r12v,r23v,ppv]]

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
rep_a1x_100 = rep_object(a1x_100,pl100,irrepTag='100_a1x')
rep_a1x_110 = rep_object(a1x_110,pl110,irrepTag='110_a1x')
rep_a1x_111 = rep_object(a1x_111,pl111,irrepTag='111_a1x')
rep_a1x_210 = rep_object(a1x_210,pl210,irrepTag='210_a1x')
rep_a1x_211 = rep_object(a1x_211,pl211,irrepTag='211_a1x')
rep_a1x_321 = rep_object(a1x_321,pl321,irrepTag='321_a1x')

### build nontrivial nonzero momentum irreps
## 100 momentum
rep_a2x_100 = extract_irrep_from_product(rep_a1x_100,rep_a2p_000,'100_a2x')
rep_eex_100 = extract_irrep_from_product(rep_a1x_100,rep_t1p_000,'100_eex')
rep_g1x_100 = extract_irrep_from_product(rep_g1p_000,rep_a1x_100,'100_g1x')
rep_g2x_100 = extract_irrep_from_product(rep_g2p_000,rep_a1x_100,'100_g2x')
rep_b1x_100 = extract_irrep_from_product(rep_a1x_100,rep_a1m_000,'100_b1x')
rep_b2x_100 = extract_irrep_from_product(rep_b1x_100,rep_a2p_000,'100_b2x')
## 110 momentum
rep_b1x_110 = extract_irrep_from_product(rep_a1x_110,rep_a1m_000,'110_b1x')
rep_ggx_110 = extract_irrep_from_product(rep_a1x_110,rep_g1p_000,'110_ggx')
rep_a2x_110 = extract_irrep_from_product(rep_a1x_110,rep_a2p_000,'110_a2x')
rep_b2x_110 = extract_irrep_from_product(rep_b1x_110,rep_a2p_000,'110_b2x')
### 111 momentum
r0 = rep_hhp_000.copy()
r1 = rep_a1x_111.copy()
r0.compute_tensor_product(r1)
rdict = r0.daughter_irreps()
rep_l1x_111 = r0.copy_daughter_irrep(*rdict['111_l1x'])
rep_l2x_111 = r0.copy_daughter_irrep(*rdict['111_l2x'])
rep_ggx_111 = r0.copy_daughter_irrep(*rdict['111_ggx'])
rep_eex_111 = extract_irrep_from_product(rep_a1x_111,rep_eep_000,'111_eex')
rep_a2x_111 = extract_irrep_from_product(rep_a1x_111,rep_a1m_000,'111_a2x')
## 210 momentum
r0 = rep_g1p_000.copy()
r1 = rep_a1x_210.copy()
r0.compute_tensor_product(r1)
rdict = r0.daughter_irreps()
rep_l1x_210 = r0.copy_daughter_irrep(*rdict['210_l1x'])
rep_l2x_210 = r0.copy_daughter_irrep(*rdict['210_l2x'])
rep_a2x_210 = extract_irrep_from_product(rep_a1x_210,rep_a1m_000,'210_a2x')
## 211 momentum
r0 = rep_g1p_000.copy()
r1 = rep_a1x_211.copy()
r0.compute_tensor_product(r1)
rdict = r0.daughter_irreps()
rep_l1x_211 = r0.copy_daughter_irrep(*rdict['211_l1x'])
rep_l2x_211 = r0.copy_daughter_irrep(*rdict['211_l2x'])
rep_a2x_211 = extract_irrep_from_product(rep_a1x_211,rep_a1m_000,'211_a2x')
## 321 momentum
rep_l1x_321 = extract_irrep_from_product(rep_a1x_321,rep_g1p_000,'321_l1x')

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
#a1x_100_full = build_from_generators(*a1x_100)
#a1x_110_full = build_from_generators(*a1x_110)
#a1x_111_full = build_from_generators(*a1x_111)
#a1x_210_full = build_from_generators(*a1x_210)
#a1x_211_full = build_from_generators(*a1x_211)
#a1x_321_full = build_from_generators(*a1x_321)
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

