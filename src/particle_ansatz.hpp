/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation: Main header
REVISION: 2022/08/19

Copyright (C) 2020-2022 Dmitry I. Lyakh (Liakh, Elvis Maradzike
Copyright (C) 2020-2022 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 Declares a class ParticleAnsatz and member functions that 
 antisymmetrize a wavefunction expressed as a tensor network
 (expansion) in the particle number basis, and sets up 
 the optimization of such based on the minimization of the 
 Rayleigh functional <x|H|x>/<x|x>.
**/

#include "exatn.hpp"

#include <iostream>
#include <vector>
#include <memory>

#ifndef CASTN_PARTICLE_ANSATZ_HPP_
#define CASTN_PARTICLE_ANSATZ_HPP_

namespace castn {

class ParticleAnsatz {

public:

  static constexpr double DEFAULT_CONVERGENCE_THRESH = 1e-5;

 /** Basic configuration of a quantum many-body system. **/
  ParticleAnsatz(std::size_t num_orbitals,      //in: number of active orbitals
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

 //Data members:
 std::size_t num_orbitals_;      //number of active orbitals
 std::size_t num_particles_;     //number of active particles
 std::size_t num_core_orbitals_; //number of core orbitals
 std::size_t total_orbitals_;    //total number of orbitals
 std::size_t total_particles_;   //total number of particles

 std::shared_ptr<exatn::TensorExpansion> ket_ansatz_;      //wavefunction ansatz ket
 std::shared_ptr<exatn::TensorExpansion> bra_ansatz_;      //wavefunction ansatz bra
 std::vector<std::shared_ptr<exatn::Tensor>> hamiltonian_; //Hamiltonian tensors
 std::shared_ptr<exatn::TensorOperator> hamiltonian_operator_; // hamiltonian operator

 std::size_t num_states_;             //number of the lowest-energy states to find
 std::vector<double> state_energies_; //quantum state energies
 double convergence_thresh_;          //tensor convergence threshold (for maxabs)

 class FunctorInitDelta: public talsh::TensorFunctor<Identifiable>{
public:

  FunctorInitDelta() = default;
  virtual ~FunctorInitDelta() = default;
  virtual const std::string name() const override
  {
    return "TensorFunctorInitDelta";
  }

  virtual const std::string description() const override
  {
    return "Initializes tensor with Kronecker delta";
  }
 
  /** Packs data members into a byte packet. **/
  virtual void pack(BytePacket & packet) override;
  /** Unpacks data members from a byte packet. **/
  virtual void unpack(BytePacket & packet) override;
  /** Description **/
  virtual int apply(talsh::Tensor & local_tensor) override;

private:

};

 class FunctorInitOrdering: public talsh::TensorFunctor<Identifiable>{
public:
  FunctorInitOrdering() = default;
  virtual ~FunctorInitOrdering() = default;
  virtual const std::string name() const override
  {
    return "TensorFunctorInitOrdering";
  }

  virtual const std::string description() const override
  {
    return "Initializes tensor: 1.0 if  m < n, 0.0 otherwise";
  }
  /** Packs data members into a byte packet. **/
  virtual void pack(BytePacket & packet) override;
  /** Unpacks data members from a byte packet. **/
  virtual void unpack(BytePacket & packet) override;
  /** Description **/
  virtual int apply(talsh::Tensor & local_tensor) override;

private:

};

};

class  SpinSiteAnsatz: public ParticleAnsatz {
  
public:
  SpinSiteAnsatz(std::size_t total_orbitals, std::size_t total_particles): ParticleAnsatz(num_orbitals_, num_particles_, 
                               num_core_orbitals_, total_orbitals_, total_particles_){}

  /** Resets Hamiltonian operator. **/
  void resetHamiltonianOperator(std::shared_ptr<exatn::TensorOperator> hamiltonian);

  /** Optimizes the wavefunction ansatz to minimize the energy trace **/
  bool optimize(std::size_t num_states = 1,                              //in: number of the lowest quantum states to optimize the energy for
               double convergence_thresh = DEFAULT_CONVERGENCE_THRESH); //in: tensor convergence threshold (for maxabs)

protected:

private:

};

} //namespace castn

#endif //CASTN_PARTICLE_ANSATZ_HPP_
