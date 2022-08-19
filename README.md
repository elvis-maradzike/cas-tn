# Complete-Active-Space Tensor-Network (CAS-TN) Ansatz Library

## Overview
For a user-defined tensor network ansatz for the electronic 
wavefunction, the CAS-TN ansatz library computes the 
optimized tensor network parameters and the corresponding ground state
 energy.

## Dependencies 
An installation of the ExaTN library. 
Visit https://github.com/ORNL-QCI/exatn for details on how 
to download and build the library.

## Building executables and running CAS-TN
To run CAS-TN, you will need to specify your C++ compiler, 
and the location of your exatn install (EXATN\_DIR) in the 
Makefile provided. 

Then: make  
This compiles the CAS-TN source code and 
generates an object file (src/particle\_ansatz.o)

This also generates executables in tests/ that can be run as 
tests for this library.

