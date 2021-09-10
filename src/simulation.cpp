/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation
REVISION: 2021/01/08

Copyright (C) 2020-2021 Dmitry I. Lyakh (Liakh), Elvis Maradzike
Copyright (C) 2020-2021 Oak Ridge National Laboratory (UT-Battelle)

**/

#include "simulation.hpp"
#include "exatn.hpp"
#include "talshxx.hpp"
#include <unordered_set>
#include "lapacke.h"
#include "cblas.h"

namespace castn {

void Simulation::clear()
{
 state_energies_.clear();
 derivatives_.clear();
 functional_.reset();
 return;
}


void Simulation::resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorNetwork> ansatz)
{
  clear();
  bra_ansatz_.reset();
  ket_ansatz_.reset();

  ket_ansatz_ = std::make_shared<exatn::TensorExpansion>();
  ket_ansatz_->appendComponent(ansatz,{1.0,0.0});
  return;
}


void Simulation::resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorExpansion> ansatz)
{
  clear();
  bra_ansatz_.reset();
  ket_ansatz_.reset();
  ket_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ansatz);
  return;
}


void Simulation::resetWaveFunctionAnsatz(exatn::NetworkBuilder & ansatz_builder)
{
 clear();
 //`Implement
 return;
}


void Simulation::resetHamiltonian(const std::vector<std::shared_ptr<exatn::Tensor>> & hamiltonian)
{
 clear();
 hamiltonian_ = hamiltonian;
 return;
}

void Simulation::markOptimizableTensors(){
  for (auto component = ket_ansatz_->begin(); component != ket_ansatz_->end(); ++component){
    component->network->markOptimizableAllTensors();
  }
}

bool Simulation::optimize(std::size_t num_states, double convergence_thresh){
  
  //marking optimizable tensors
  markOptimizableTensors();
  
  //appending ordering projectors
  appendOrderingProjectors();
 
  ket_ansatz_->rename("KetExpansion"); 
  bra_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ket_ansatz_);
  bra_ansatz_->rename("BraExpansion");
  bra_ansatz_->conjugate();
  
  //initializing optimizable tensors
  initWavefunctionAnsatz();

  // hamiltonian operator
  auto hamiltonian_operator = exatn::makeSharedTensorOperator("HamiltonianOperator");
    //(anti)symmetrization 
  auto success = hamiltonian_operator->appendSymmetrizeComponent(hamiltonian_[0],{0,1},{2,3}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  success = hamiltonian_operator->appendSymmetrizeComponent(hamiltonian_[1],{0},{1}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  // setting up and calling the optimizer in ../src/exatn/..
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(hamiltonian_operator,ket_ansatz_,convergence_thresh_);
  optimizer.resetLearningRate(0.5);
  optimizer.resetMicroIterations(1);
  optimizer.resetDebugLevel(2);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }

  return true;
}

/** Appends two layers of ordering projectors to the wavefunction ansatz. **/
void Simulation::appendOrderingProjectors(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  auto created = exatn::createTensor("Q", TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_, total_orbitals_, total_orbitals_, total_orbitals_}); assert(created);
  auto success = exatn::transformTensorSync("Q", std::shared_ptr<exatn::TensorMethod>{new exatn::numerics::FunctorInitProj()});
  
  auto appended = false;
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    auto & network = *(iter->network);
    unsigned int tensorCounter = 1 + network.getNumTensors();
    // applying layer 1
    for ( unsigned int i = 0; i < num_particles_-1; i=i+2){
      appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      tensorCounter++;
    }
    // applying layer 2
    for ( unsigned int i = 1; i < num_particles_-1; i=i+2){
      appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      tensorCounter++;
    }
  }

  //ket_ansatz_->printIt();
}


void Simulation::constructEnergyFunctional(){
}

void Simulation::constructEnergyDerivatives(){
}

void Simulation::initWavefunctionAnsatz(){
  // normalize wavefunction ansatz
  auto success = exatn::balanceNormalizeNorm2Sync(*ket_ansatz_,1.0,1.0,true);
}


double Simulation::evaluateEnergyFunctional(){
  double energy = 0.0;
  return energy;
}

void Simulation::evaluateEnergyDerivatives(){
}
  
void Simulation::updateWavefunctionAnsatzTensors(){
}
 
} //namespace castn
