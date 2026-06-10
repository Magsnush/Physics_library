# Introduction:
This is a python library for numerically computing different cross sections at small x. Currently the code can compute inclusive DIS cross sections at LO, using the optical theorem or using a finite energy constraint, and it can compute inclusive DIS cross sections with LO lightcone wavefunctions and a rcBK-evolved target amplitude.

# Description of the library:
In the library there is one folder containing building blocks and one folder containing cross sections which are constructed from the building blocks. 

In the building blocks folder I have implemented lightcone wavefunctions at LO for both the standard optical theorem based cross section and the finite energy constrained cross section (see ref below). I have also implemented multiple multipole correlators in the MV-model. 

In the cross sections folder there are scripts for computing multiple different inclusive DIS cross sections e.g. optical theorem (OT) at LO, finite-energy constrained (FEC) at LO (eq. 24 in https://arxiv.org/abs/2601.07302), OT with BK evolved dipole, FEC with BK.

# Usage:
  1. Download the folder and install it using "pip install -e." such that you import functions properly.
  2. After downloading and installing you can now compute cross sections. An example usage would can be found in script "test_bk_integration.bk". The arguments you can parse exist in the file. The only necessary input is a bkfile computed in the style of https://github.com/hejajama/rcbkdipole/blob/master/data/proton/mv.dat. You can then run this from the command line as 
  python3 test_bk_integration.bk --bkfile SOME_BK_FILE.dat


