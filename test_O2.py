import psi4
import uhf
import frankenstein.tools.io_utils as iou
from frankenstein import scf

"""
Nuclear Repulsion Energy =             27.9895383055867768
One-Electron Energy =                -260.1988993488373012
Two-Electron Energy =                  83.4417277313030468
Total Energy =                       -148.7676333119474634
"""

psi4.core.set_output_file('O2_21g.dat')

O2_string = """
0 3
O        
O 1 1.21 
"""



O2 = psi4.geometry(O2_string)
psi4.set_options({'reference': 'uhf'})
O2_e, O2_uhf = psi4.energy('scf/3-21g', return_wfn=True)


ma=9; mb=7;
h, V, enuc = iou.get_ao_ints(O2_string, '3-21g',symm_orth=True)
print(enuc)

umf = scf.UHF(h=h, V=V, e_nuc = enuc, nocc=9, noccb=7, opt='diis')
umf.kernel()
print(umf.e_tot)

from frankenstein import lattice, scf
lat = lattice.LATTICE(h=h,V=V,e_nuc=enuc, nocc=8)
mf = scf.RHF(lat)
mf.kernel()
print(mf.e_tot)
print(mf.fock)


#O2_uhf = uhf.UHF(h,V,ma,mb)
#O2_uhf._do_scf()


















