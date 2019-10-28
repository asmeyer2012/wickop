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
from manip_array import all_zero
from manip_momentum import classify_momentum_type

## indices depend on how build_from_generators is defined!
## determined by a script, copypasted

## p000 character table
## 1, +R, +R'.R, +-R.R, +-R.R'.R, -1, -R, -R'.R
p000char = np.kron( np.array([[1.,1.],[1.,-1.]]), np.array(
[[ 1.,  1., 1., 1., 1., 1.,  1., 1.],
 [ 1., -1., 1., 1.,-1., 1., -1., 1.],
 [ 2.,  0.,-1., 2., 0., 2.,  0.,-1.],
 [ 3.,  1., 0.,-1.,-1., 3.,  1., 0.],
 [ 3., -1., 0.,-1., 1., 3., -1., 0.],
 [ 2., sr2, 1., 0., 0.,-2.,-sr2,-1.],
 [ 2.,-sr2, 1., 0., 0.,-2., sr2,-1.],
 [ 4.,  0.,-1., 0., 0.,-4.,  0., 1.]] ) )
## octahedral group centralizer
p000cent = np.kron(np.array([1.,1.]), np.array([1.,6.,8.,6.,12.,1.,6.,8.]))
p000ord = int(sum(p000cent)) ## order of group
p000lg = range(p000ord)
p000coset = [0] ## only the identity

## p100 little group character table
## a = R12, -a3 = R21
## b = R13.R13.P
## 1, -1, +a-a3, +a3-a, +-a.a, +-a.b+-a3.b, +-b+-a.a.b
p100char = np.array(
[[ 1., 1.,  1.,  1., 1., 1., 1.],
 [ 1., 1., -1., -1., 1.,-1., 1.],
 [ 1., 1.,  1.,  1., 1.,-1.,-1.],
 [ 1., 1., -1., -1., 1., 1.,-1.],
 [ 2., 2.,  0.,  0.,-2., 0., 0.],
 [ 2.,-2., sr2,-sr2, 0., 0., 0.],
 [ 2.,-2.,-sr2, sr2, 0., 0., 0.]] )
p100cent = np.array([1.,1.,2.,2.,2.,4.,4.])
p100ord = int(sum(p100cent)) ## order of group
#p100lg = [0,33,3,6,36,39,17,20,76,70,77,73,64,67,63,66] ## little group indices
p100lg = [0,33,3,6,36,39,17,20,76,70,77,73,64,67,63,66] ## inverted
#p100coset = [15,1,2,5,4,0]
p100coset = [15,1,5,2,4,0] ## inverted

## p110 little group character table
## a = R21.R13.R13
## b = R12.R12.P
## 1, -1, +-a, +-a.b, +-b
p110char = np.array(
[[ 1., 1., 1., 1., 1.],
 [ 1., 1.,-1.,-1., 1.],
 [ 1., 1., 1.,-1.,-1.],
 [ 1., 1.,-1., 1.,-1.],
 [ 2.,-2., 0., 0., 0.]] )
p110cent = np.array([1.,1.,2.,2.,2.])
p110ord = int(sum(p110cent)) ## order of group
#p110lg = [0,33,22,28,73,77,68,65] ## little group indices
p110lg = [0,33,25,29,70,76,68,65] ## inverted
#p110coset = [17,12,7,3,13,10,5,2,6,4,1,0]
p110coset = [17,11,8,3,14,9,2,5,6,4,1,0] ## inverted

## p111 little group character table
## a = R13.R12
## b = R12.R13.R13.P -> supposed to be R21.R13.R13.P ?
## 1, -1, +a.a-a, +a-a.a, b-a.b+a.a.b, -b+a.b-a.a.b
p111char = np.array(
[[ 1., 1., 1., 1., 1., 1.],
 [ 1., 1., 1., 1.,-1.,-1.],
 [ 2., 2.,-1.,-1., 0., 0.],
 [ 1.,-1., 1.,-1., I ,-I ],
 [ 1.,-1., 1.,-1.,-I , I ],
 [ 2.,-2.,-1., 1., 0., 0.]] )
p111cent = np.array([1.,1.,2.,2.,3.,3.])
p111ord = int(sum(p111cent)) ## order of group
#p111lg = [0,33,41,47,8,14,71,72,77,74,73,80] ## little group indices
p111lg = [0,33,40,46,7,13,70,74,79,76,71,75] ## inverted
#p111coset = [23,7,11,2,9,1,4,0]
p111coset = [22,8,11,3,10,1,2,0] ## inverted

## p210 little group character table
## a = R12.R12.P
## 1, a, a.a, a.a.a
p210char = np.array(
[[ 1., 1., 1., 1.],
 [ 1.,-1., 1.,-1.],
 [ 1., I ,-1.,-I ],
 [ 1.,-I ,-1., I ]])
p210cent = np.array([1.,1.,1.,1.])
p210ord = int(sum(p210cent)) ## order of group
#p210lg = [0,68,33,65] ## little group indices
p210lg = [0,68,33,65] ## inverted
#p210coset = [17,24,27,16,25,12,7,3,13,10,23,21,5,2,11,8,6,9,14,22,15,4,1,0]
p210coset = [17,27,24,16,22,11,8,3,14,9,23,21,2,5,12,7,6,10,13,25,15,4,1,0] ## inverted

## p211 little group character table
## a = R12.R13.R13.P -> supposed to be R21.R13.R13.P ?
## 1, a, a.a, a.a.a
p211char = p210char
p211cent = np.array([1.,1.,1.,1.])
p211ord = int(sum(p211cent)) ## order of group
#p211lg = [0,77,33,73] ## little group indices
p211lg = [0,76,33,70] ## inverted
#p211coset = [25,15,16,22,24,9,23,13,11,5,12,4,7,1,10,21,2,8,27,14,17,6,3,0]
p211coset = [22,15,16,25,27,10,23,14,12,2,11,4,8,1,9,21,5,7,24,13,17,6,3,0] ## inverted

## p321 little group character table
## 1, -1
p321char = np.array(
[[ 1., 1.],
 [ 1.,-1.]])
p321cent = np.array([1.,1.])
p321ord = int(sum(p321cent)) ## order of group
#p321lg = [0,33] ## little group indices
p321lg = [0,33] ## inverted
#p321coset = [23,50,62,9,12,55,69,5,56,13,24,75,49,4,11,58,25,51,48,15,16,65,54,\
# 22,70,6,17,64,63,0,3,73,10,59,52,1,27,72,61,8,53,21,7,60,57,14,2,71]
p321coset = [23,53,61,10,11,56,69,2,55,14,27,72,49,4,12,57,22,51,48,15,16,65,54,\
 25,73,6,17,64,63,0,3,70,9,60,52,1,24,75,62,7,50,21,8,59,58,13,5,71] ## inverted

typeTabs = ['000','100','110','111','210','211','321']
charTabs = [p000char,p100char,p110char,p111char,p210char,p211char,p321char]
centTabs = [p000cent,p100cent,p110cent,p111cent,p210cent,p211cent,p321cent]
ordTabs = [p000ord,p100ord,p110ord,p111ord,p210ord,p211ord,p321ord]
lgTabs = [p000lg,p100lg,p110lg,p111lg,p210lg,p211lg,p321lg]
cosetTabs = [p000coset,p100coset,p110coset,p111coset,p210coset,p211coset,p321coset]

## from a full set of representation matrices, extract only those that belong to the little group
def get_littlegroup_matrices(pref,rep):
  tp = classify_momentum_type(pref)
  lgidx = lgTabs[typeTabs.index(tp)]
  return [rep[i] for i in lgidx]

## given a reference momentum, get the character table of corresponding little group
def get_littlegroup_character_table(pref):
  tp = classify_momentum_type(pref)
  return charTabs[typeTabs.index(tp)]

## given a reference momentum, get the centralizer of corresponding little group
def get_littlegroup_centralizer(pref):
  tp = classify_momentum_type(pref)
  return centTabs[typeTabs.index(tp)]

## given a reference momentum, get the centralizer of corresponding little group
def get_littlegroup_cosets(pref):
  tp = classify_momentum_type(pref)
  return cosetTabs[typeTabs.index(tp)]

### unitarity checks on character tables
#for pn,ch,ce,od in zip(momClass,charTabs,centTabs,ordTabs):
#  if not( all_zero( np.conj(ch).T.dot(ch) - np.diag(od/ce) )):
#    raise GroupTheoryError(\
#     "Little group character table for momentum "+pn\
#     +"does not satisfy Schur orthogonality of columns!")
#  if not( all_zero( ch.dot(np.diag(ce)).dot(np.conj(ch).T) - np.diag(np.ones(len(ce))*od) )):
#    raise GroupTheoryError(\
#     "Little group character table for momentum "+pn\
#     +"does not satisfy Schur orthogonality of rows!")

