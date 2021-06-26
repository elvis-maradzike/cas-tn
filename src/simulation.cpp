/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation
REVISION: 2021/01/08

Copyright (C) 2020-2021 Dmitry I. Lyakh (Liakh), Elvis Maradzike
Copyright (C) 2020-2021 Oak Ridge National Laboratory (UT-Battelle)

**/

#include "simulation.hpp"
#include "exatn.hpp"
#include "talshxx.hpp"
#include <unordered_set>

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
 
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  const auto TENS_SHAPE_0INDEX = exatn::TensorShape{};
  const auto TENS_SHAPE_2INDEX = exatn::TensorShape{total_orbitals_, total_orbitals_};
  const auto TENS_SHAPE_4INDEX = exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_};

  //hamiltonian tensors
  auto h1 = std::make_shared<exatn::Tensor>("H1", TENS_SHAPE_2INDEX);
  auto h2 = std::make_shared<exatn::Tensor>("H2", TENS_SHAPE_4INDEX);

  auto created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);
  
  auto initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

  hamiltonian_.push_back(h2);
  hamiltonian_.push_back(h1);

  auto ham = exatn::makeSharedTensorOperator("Hamiltonian");
  auto appended = false;
 
  //(anti)symmetrization 
  auto success = ham->appendSymmetrizeComponent(hamiltonian_[0],{0,1},{2,3}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  success = ham->appendSymmetrizeComponent(hamiltonian_[1],{0},{1}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);

  //marking optimizable tensors
  markOptimizableTensors();
  
  //appending ordering projectors
  appendOrderingProjectors();
 
  ket_ansatz_->rename("VectorExpansion"); 
  //bra ansatz
  bra_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ket_ansatz_);
  bra_ansatz_->rename(ket_ansatz_->getName()+"Bra");
  bra_ansatz_->conjugate();
  bra_ansatz_->printIt();
  
  //initializing optimizable tensors
  initWavefunctionAnsatz();

  // setting up and calling the optimizer in ../src/exatn/..
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(ham,ket_ansatz_,1e-4);
  optimizer.resetLearningRate(0.2);
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
  const auto TENS_SHAPE_2INDEX = exatn::TensorShape{total_orbitals_, total_orbitals_};
  const auto TENS_SHAPE_4INDEX = exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_};
  // array of elements to initialize tensors that comprise the ordering projectors
  std::vector<double> tmpData(total_orbitals_*total_orbitals_*total_orbitals_*total_orbitals_,0.0);
  for ( unsigned int i = 0; i < total_orbitals_; i++){
    for ( unsigned int j = 0; j < total_orbitals_; j++){
      for ( unsigned int k = 0; k < total_orbitals_; k++){
        for ( unsigned int l = 0; l < total_orbitals_; l++){
          if ( i < j){
          tmpData[i*total_orbitals_*total_orbitals_*total_orbitals_
                 +j*total_orbitals_*total_orbitals_
                 +k*total_orbitals_+l] = double((i==k)*(j==l));
          }else{
          tmpData[i*total_orbitals_*total_orbitals_*total_orbitals_
                 +j*total_orbitals_*total_orbitals_
                 +k*total_orbitals_+l] = 0.0;
          }
        }
      }
    }
  }
 
  //create tensors for ordering projectors
  auto created = exatn::createTensor("Q", TENS_ELEM_TYPE, TENS_SHAPE_4INDEX); assert(created);
  auto initialized = exatn::initTensorData("Q", tmpData); assert(initialized);
  
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

  ket_ansatz_->printIt();
}


void Simulation::constructEnergyFunctional(){
}

void Simulation::constructEnergyDerivatives(){
}

void Simulation::initWavefunctionAnsatz(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        auto initialized = exatn::initTensorRnd(tensor.getName()); assert(initialized);
        /*
        auto initialized = exatn::initTensorFile(tensor.getName(),"h4_tensor.txt"); assert(initialized);
        */
      } 
    }
  }
 
  // evaluate norm of wavefunction
  exatn::TensorExpansion bra_ket(*bra_ansatz_,*ket_ansatz_);
  bra_ket_ = std::make_shared<exatn::TensorExpansion>(bra_ket);

  auto created = exatn::createTensorSync("_InnerProduct", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
  auto inner_product = exatn::getTensor("_InnerProduct");
  auto initialized = exatn::initTensor(inner_product->getName(), 0.0); assert(initialized);
  auto evaluated = exatn::evaluateSync(*bra_ket_,inner_product); assert(evaluated);
  
  auto local_copy = exatn::getLocalTensor(inner_product->getName()); assert(local_copy);
  const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
  auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val = *body_ptr;
  body_ptr = nullptr;
 
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        // number of tensors in network less the number of ordering projectors
        int num_optimizable = network->network->getNumTensors() - (num_particles_-1);
        double factor = pow(sqrt(1.0/val), 1.0/double(num_optimizable));
        auto scaled = exatn::scaleTensor(tensor.getName(), factor); assert(scaled);
      } 
    }
  }

  auto destroyed = exatn::destroyTensorSync(inner_product->getName()); assert(destroyed);
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
