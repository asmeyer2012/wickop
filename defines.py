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

smallNum = 1e-7
sr2 = np.sqrt(2.)
c1 = complex(1.,0.)
I = complex(0.,1.)

## Pauli sigma matrices
id2 = np.diag([c1,c1])
s3 = np.diag([c1,-c1])
s1 = np.reshape(np.array([0.,c1,c1,0.]),(2,2))
s2 = -I*np.dot(s3,s1)
## Gamma matrices
id4 = np.kron(id2,id2)
g0 = np.kron(s1,id2)
g1 = np.kron(s2,s1)
g2 = np.kron(s2,s2)
g3 = np.kron(s2,s3)
g5 = np.kron(s3,id2)

## rotation matrices acting on 2-spinors
r12 = np.array(\
[[ 1.-I, 0.],
 [ 0., 1.+I]] )/np.sqrt(2.)
r23 = np.array(\
[[ 1.,-I ],
 [-I , 1.]] )/np.sqrt(2.)
r21 = np.linalg.inv(r12)
r32 = np.linalg.inv(r23)
r13 = r12.dot(r23).dot(r21) ## mconj(r23,r12)
r31 = np.linalg.inv(r13)
pp = np.diag(-np.ones(2,dtype=complex))

## rotation matrices acting on 3-vectors
id3 = np.diag(np.ones(3))
r12v = np.array(\
[[ 0.,-1., 0.],
 [ 1., 0., 0.],
 [ 0., 0., 1.]])
r23v = np.array(\
[[ 1., 0., 0.],
 [ 0., 0.,-1.],
 [ 0., 1., 0.]])
r21v = np.linalg.inv(r12v)
r32v = np.linalg.inv(r23v)
r13v = r12v.dot(r23v).dot(r21v) ## mconj(r23v,r12v)
r31v = np.linalg.inv(r13v)
ppv = np.diag(-np.ones(3))

## representation names
#
rep_list_spinmom = []
## zero momentum, positive parity
rep_list_spinmom.append( '000_a1+' )
rep_list_spinmom.append( '000_a2+' )
rep_list_spinmom.append( '000_ee+' )
rep_list_spinmom.append( '000_t1+' )
rep_list_spinmom.append( '000_t2+' )
rep_list_spinmom.append( '000_g1+' )
rep_list_spinmom.append( '000_g2+' )
rep_list_spinmom.append( '000_hh+' )
## zero momentum, negative parity
rep_list_spinmom.append( '000_a1-' )
rep_list_spinmom.append( '000_a2-' )
rep_list_spinmom.append( '000_ee-' )
rep_list_spinmom.append( '000_t1-' )
rep_list_spinmom.append( '000_t2-' )
rep_list_spinmom.append( '000_g1-' )
rep_list_spinmom.append( '000_g2-' )
rep_list_spinmom.append( '000_hh-' )
p000slice = slice(0,16)
## (1,0,0) momentum
rep_list_spinmom.append( '100_a1x' )
rep_list_spinmom.append( '100_a2x' )
rep_list_spinmom.append( '100_b1x' )
rep_list_spinmom.append( '100_b2x' )
rep_list_spinmom.append( '100_eex' )
rep_list_spinmom.append( '100_g1x' )
rep_list_spinmom.append( '100_g2x' )
p100slice = slice(16,23)
## (1,1,0) momentum
rep_list_spinmom.append( '110_a1x' )
rep_list_spinmom.append( '110_a2x' )
rep_list_spinmom.append( '110_b1x' )
rep_list_spinmom.append( '110_b2x' )
rep_list_spinmom.append( '110_ggx' )
p110slice = slice(23,28)
## (1,1,1) momentum
rep_list_spinmom.append( '111_a1x' )
rep_list_spinmom.append( '111_a2x' )
rep_list_spinmom.append( '111_eex' )
rep_list_spinmom.append( '111_l1x' )
rep_list_spinmom.append( '111_l2x' )
rep_list_spinmom.append( '111_ggx' )
p111slice = slice(28,34)
## (2,1,0) momentum
rep_list_spinmom.append( '210_a1x' )
rep_list_spinmom.append( '210_a2x' )
rep_list_spinmom.append( '210_l1x' )
rep_list_spinmom.append( '210_l2x' )
p210slice = slice(34,38)
## (2,1,1) momentum
rep_list_spinmom.append( '211_a1x' )
rep_list_spinmom.append( '211_a2x' )
rep_list_spinmom.append( '211_l1x' )
rep_list_spinmom.append( '211_l2x' )
p211slice = slice(38,42)
## (3,2,1) momentum
rep_list_spinmom.append( '321_a1x' )
rep_list_spinmom.append( '321_l1x' )
p321slice = slice(42,44)

momSlice = [p000slice,p100slice,p110slice,p111slice,p210slice,p211slice,p321slice]
momClass = ['000','100','110','111','210','211','321']

## irrep dimensions
irrepDimension = {
'000_a1+': 1, '000_a2+': 1, '000_ee+': 2, '000_t1+': 3, '000_t2+': 3,
'000_g1+': 2, '000_g2+': 2, '000_hh+': 4,
'000_a1-': 1, '000_a2-': 1, '000_ee-': 2, '000_t1-': 3, '000_t2-': 3,
'000_g1-': 2, '000_g2-': 2, '000_hh-': 4,
'100_a1x': 6, '100_a2x': 6, '100_b1x': 6, '100_b2x': 6, '100_eex':12,
'100_g1x':12, '100_g2x':12,
'110_a1x':12, '110_a2x':12, '110_b1x':12, '110_b2x':12, '110_ggx':24,
'111_a1x': 8, '111_a2x': 8, '111_eex':16, '111_l1x': 8, '111_l2x': 8, '111_ggx':16,
'210_a1x':24, '210_a2x':24, '210_l1x':24, '210_l2x':24,
'211_a1x':24, '211_a2x':24, '211_l1x':24, '211_l2x':24,
'321_a1x':48, '321_l1x':48
}
LGDimension = {
'000_a1+': 1, '000_a2+': 1, '000_ee+': 2, '000_t1+': 3, '000_t2+': 3,
'000_g1+': 2, '000_g2+': 2, '000_hh+': 4,
'000_a1-': 1, '000_a2-': 1, '000_ee-': 2, '000_t1-': 3, '000_t2-': 3,
'000_g1-': 2, '000_g2-': 2, '000_hh-': 4,
'100_a1x': 1, '100_a2x': 1, '100_b1x': 1, '100_b2x': 1, '100_eex': 2,
'100_g1x': 2, '100_g2x': 2,
'110_a1x': 1, '110_a2x': 1, '110_b1x': 1, '110_b2x': 1, '110_ggx': 2,
'111_a1x': 1, '111_a2x': 1, '111_eex': 2, '111_l1x': 1, '111_l2x': 1, '111_ggx': 2,
'210_a1x': 1, '210_a2x': 1, '210_l1x': 1, '210_l2x': 1,
'211_a1x': 1, '211_a2x': 1, '211_l1x': 1, '211_l2x': 1,
'321_a1x': 1, '321_l1x': 1
}
irrepCleanName = {
'000_a1+': 'a1p_000', '000_a2+': 'a2p_000', '000_ee+': 'eep_000', '000_t1+': 't1p_000',
'000_t2+': 't2p_000', '000_g1+': 'g1p_000', '000_g2+': 'g2p_000', '000_hh+': 'hhp_000',
'000_a1-': 'a1m_000', '000_a2-': 'a2m_000', '000_ee-': 'eem_000', '000_t1-': 't1m_000',
'000_t2-': 't2m_000', '000_g1-': 'g1m_000', '000_g2-': 'g2m_000', '000_hh-': 'hhm_000',
'100_a1x': 'a1x_100', '100_a2x': 'a2x_100', '100_b1x': 'b1x_100', '100_b2x': 'b2x_100',
'100_eex': 'eex_100', '100_g1x': 'g1x_100', '100_g2x': 'g2x_100',
'110_a1x': 'a1x_110', '110_a2x': 'a2x_110', '110_b1x': 'b1x_110',
'110_b2x': 'b2x_110', '110_ggx': 'ggx_110',
'111_a1x': 'a1x_111', '111_a2x': 'a2x_111', '111_eex': 'eex_111',
'111_l1x': 'l1x_111', '111_l2x': 'l2x_111', '111_ggx': 'ggx_111',
'210_a1x': 'a1x_210', '210_a2x': 'a2x_210', '210_l1x': 'l1x_210', '210_l2x': 'l2x_210',
'211_a1x': 'a1x_211', '211_a2x': 'a2x_211', '211_l1x': 'l1x_211', '211_l2x': 'l2x_211',
'321_a1x': 'a1x_321', '321_l1x': 'l1x_321'
}
irrepCleanNameInverse = {}
for key in irrepCleanName.keys():
  irrepCleanNameInverse[ irrepCleanName[ key]] = key

## exceptions, so they are not caught by most exception routines
class DimensionError(Exception):
  pass

#class SimilarityError(Exception):
#  pass

class IterationError(Exception):
  pass

class GroupTheoryError(Exception):
  pass

