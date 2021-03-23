/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation: Main header
REVISION: 2021/01/29

Copyright (C) 2020-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2020-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 <Describe what functionality is provided by this header>
**/

#include "exatn.hpp"

#include <iostream>
#include <vector>
#include <memory>

#ifndef CASTN_SIMULATION_HPP_
#define CASTN_SIMULATION_HPP_

namespace castn {

class Simulation {

public:

 static constexpr double DEFAULT_CONVERGENCE_THRESH = 1e-5;

 /** Basic configuration of a quantum many-body system. **/
 Simulation(std::size_t num_orbitals,      //in: number of active orbitals
            std::size_t num_particles,     //in: number of active particles
            std::size_t num_core_orbitals, //in: number of core orbitals (with core particles)
            std::size_t total_orbitals,    //in: total number of orbitals
            std::size_t total_particles):  //in: total number of particles
  num_orbitals_(num_orbitals),
  num_particles_(num_particles),
  num_core_orbitals_(num_core_orbitals),
  total_orbitals_(total_orbitals),
  total_particles_(total_particles),
  num_states_(1),
  convergence_thresh_(DEFAULT_CONVERGENCE_THRESH)
  {}

 /** Resets the wavefunction ansatz with the provided tensor network. **/
 void resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorNetwork> ansatz);

 /** Resets the wavefunction ansatz with the provided tensor network expansion. **/
 void resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorExpansion> ansatz);

 /** Resets the wavefunction ansatz by constructing an appropriate tensor network. **/
 void resetWaveFunctionAnsatz(exatn::NetworkBuilder & ansatz_builder);

 /** Resets the quantum Hamiltonian tensors. **/
 void resetHamiltonian(const std::vector<std::shared_ptr<exatn::Tensor>> & hamiltonian);

 /** Optimizes the wavefunction ansatz to minimize the energy trace.
     Returns TRUE upon convergence, FALSE otherwise. **/
 bool optimize(std::size_t num_states = 1,                              //in: number of the lowest quantum states to optimize the energy for
               double convergence_thresh = DEFAULT_CONVERGENCE_THRESH); //in: tensor convergence threshold (for maxabs)

protected:

 /** Clears the simulation state. **/
 void clear();

 /** Mark optimizable tensors. **/
 void markOptimizableTensors();

 /** Appends two layers of ordering projectors to the wavefunction ansatz. **/
 void appendOrderingProjectors();

 /** Constructs the energy trace optimization functional. **/
 void constructEnergyFunctional();

 /** Constructs derivatives of the energy functional. **/
 void constructEnergyDerivatives();

 /** Creates the initial guess for the wavefunction ansatz
     (initializes all tensors in the wavefunction ansatz). **/
 void initWavefunctionAnsatz();

 /** Evaluates the current value of the energy functional. **/
 double evaluateEnergyFunctional();

 /** Evaluates derivates of the energy functional with respect to all optimized tensors. **/
 void evaluateEnergyDerivatives();

 /** Updates all tensors being optimized in the wavefunction ansatz. **/
 void updateWavefunctionAnsatzTensors();

 //Data members:
 std::size_t num_orbitals_;      //number of active orbitals
 std::size_t num_particles_;     //number of active particles
 std::size_t num_core_orbitals_; //number of core orbitals
 std::size_t total_orbitals_;    //total number of orbitals
 std::size_t total_particles_;   //total number of particles

 std::shared_ptr<exatn::TensorExpansion> ket_ansatz_;      //wavefunction ansatz ket
 std::shared_ptr<exatn::TensorExpansion> bra_ansatz_;      //wavefunction ansatz bra
 std::vector<std::shared_ptr<exatn::Tensor>> hamiltonian_; //Hamiltonian tensors

 std::size_t num_states_;             //number of the lowest-energy states to find
 std::vector<double> state_energies_; //quantum state energies
 double convergence_thresh_;          //tensor convergence threshold (for maxabs)

 double maxAbsGrad_; // absolute value of largest element in gradient tensor
 double maxAbs_;     // absolute value of largest element in tensor factor in question

 double normVal_;       // wavefunction norm <Psi|Psi>

 std::shared_ptr<exatn::TensorExpansion> functional_; //energy trace functional
 std::shared_ptr<exatn::TensorExpansion> norm_; // norm 

 std::vector<std::tuple<std::string,                             //tensor name
                        std::shared_ptr<exatn::TensorExpansion>, //derivative tensor expansion
                        std::shared_ptr<exatn::Tensor>,           //derivative tensor
                        std::shared_ptr<exatn::TensorExpansion>, //derivative tensor expansion
                        std::shared_ptr<exatn::Tensor>           //derivative tensor
                       >> derivatives_; //derivatives of the energy functional


 struct Environment{
 std::shared_ptr<exatn::Tensor> tensor;     //tensor being optimized
 std::shared_ptr<exatn::Tensor> gradient;   //gradient w.r.t. the tensor
 //exatn::TensorExpansion gradient_expansion_e; //gradient tensor network expansion
 //std::shared_ptr<exatn::Tensor> gradient_n;   //gradient w.r.t. the tensor
 //exatn::TensorExpansion gradient_expansion_n; //gradient tensor network expansion
 };

 std::vector<Environment> environments_;

 int iter; // iteration index;

};


} //namespace castn

#endif //CASTN_SIMULATION_HPP_
