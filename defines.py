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
rep_list_spinmom.append( '100_a1+' )
rep_list_spinmom.append( '100_a2+' )
rep_list_spinmom.append( '100_a1-' )
rep_list_spinmom.append( '100_a2-' )
rep_list_spinmom.append( '100_bb0' )
rep_list_spinmom.append( '100_g10' )
rep_list_spinmom.append( '100_g20' )
p100slice = slice(16,23)
## (1,1,0) momentum
rep_list_spinmom.append( '110_a1+' )
rep_list_spinmom.append( '110_a2+' )
rep_list_spinmom.append( '110_a1-' )
rep_list_spinmom.append( '110_a2-' )
rep_list_spinmom.append( '110_gg0' )
p110slice = slice(23,28)
## (1,1,1) momentum
rep_list_spinmom.append( '111_aa+' )
rep_list_spinmom.append( '111_aa-' )
rep_list_spinmom.append( '111_bb0' )
rep_list_spinmom.append( '111_ll+' )
rep_list_spinmom.append( '111_ll-' )
rep_list_spinmom.append( '111_gg0' )
p111slice = slice(28,34)
## (2,1,0) momentum
rep_list_spinmom.append( '210_aa+' )
rep_list_spinmom.append( '210_aa-' )
rep_list_spinmom.append( '210_ll+' )
rep_list_spinmom.append( '210_ll-' )
p210slice = slice(34,38)
## (2,1,1) momentum
rep_list_spinmom.append( '211_aa+' )
rep_list_spinmom.append( '211_aa-' )
rep_list_spinmom.append( '211_ll+' )
rep_list_spinmom.append( '211_ll-' )
p211slice = slice(38,42)
## (3,2,1) momentum
rep_list_spinmom.append( '321_aa0' )
rep_list_spinmom.append( '321_ll0' )
p321slice = slice(42,44)

momSlice = [p000slice,p100slice,p110slice,p111slice,p210slice,p211slice,p321slice]
momClass = ['000','100','110','111','210','211','321']

## irrep dimensions
irrepDimension = {
'000_a1+': 1, '000_a2+': 1, '000_ee+': 2, '000_t1+': 3, '000_t2+': 3,
'000_g1+': 2, '000_g2+': 2, '000_hh+': 4,
'000_a1-': 1, '000_a2-': 1, '000_ee-': 2, '000_t1-': 3, '000_t2-': 3,
'000_g1-': 2, '000_g2-': 2, '000_hh-': 4,
'100_a1+': 6, '100_a2+': 6, '100_a1-': 6, '100_a2-': 6, '100_bb0':12,
'100_g10':12, '100_g20':12,
'110_a1+':12, '110_a2+':12, '110_a1-':12, '110_a2-':12, '110_gg0':24,
'111_aa+': 8, '111_aa-': 8, '111_bb0':16, '111_ll+': 8, '111_ll-': 8, '111_gg0':16,
'210_aa+':24, '210_aa-':24, '210_ll+':24, '210_ll-':24,
'211_aa+':24, '211_aa-':24, '211_ll+':24, '211_ll-':24,
'321_aa0':48, '321_ll0':48
}
LGDimension = {
'000_a1+': 1, '000_a2+': 1, '000_ee+': 2, '000_t1+': 3, '000_t2+': 3,
'000_g1+': 2, '000_g2+': 2, '000_hh+': 4,
'000_a1-': 1, '000_a2-': 1, '000_ee-': 2, '000_t1-': 3, '000_t2-': 3,
'000_g1-': 2, '000_g2-': 2, '000_hh-': 4,
'100_a1+': 1, '100_a2+': 1, '100_a1-': 1, '100_a2-': 1, '100_bb0': 2,
'100_g10': 2, '100_g20': 2,
'110_a1+': 1, '110_a2+': 1, '110_a1-': 1, '110_a2-': 1, '110_gg0': 2,
'111_aa+': 1, '111_aa-': 1, '111_bb0': 2, '111_ll+': 1, '111_ll-': 1, '111_gg0': 2,
'210_aa+': 1, '210_aa-': 1, '210_ll+': 1, '210_ll-': 1,
'211_aa+': 1, '211_aa-': 1, '211_ll+': 1, '211_ll-': 1,
'321_aa0': 1, '321_ll0': 1
}
irrepCleanName = {
'000_a1+': 'a1p_000', '000_a2+': 'a2p_000', '000_ee+': 'eep_000', '000_t1+': 't1p_000',
'000_t2+': 't2p_000', '000_g1+': 'g1p_000', '000_g2+': 'g2p_000', '000_hh+': 'hhp_000',
'000_a1-': 'a1m_000', '000_a2-': 'a2m_000', '000_ee-': 'eem_000', '000_t1-': 't1m_000',
'000_t2-': 't2m_000', '000_g1-': 'g1m_000', '000_g2-': 'g2m_000', '000_hh-': 'hhm_000',
'100_a1+': 'a1p_100', '100_a2+': 'a2p_100', '100_a1-': 'a1m_100', '100_a2-': 'a2m_100',
'100_bb0': 'bb0_100', '100_g10': 'g10_100', '100_g20': 'g20_100',
'110_a1+': 'a1p_110', '110_a2+': 'a2p_110', '110_a1-': 'a1m_110',
'110_a2-': 'a2m_110', '110_gg0': 'gg0_110',
'111_aa+': 'aap_111', '111_aa-': 'aam_111', '111_bb0': 'bb0_111',
'111_ll+': 'llp_111', '111_ll-': 'llm_111', '111_gg0': 'gg0_111',
'210_aa+': 'aap_210', '210_aa-': 'aam_210', '210_ll+': 'llp_210', '210_ll-': 'llm_210',
'211_aa+': 'aap_211', '211_aa-': 'aam_211', '211_ll+': 'llp_211', '211_ll-': 'llm_211',
'321_aa0': 'aa0_321', '321_ll0': 'll0_321'
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

