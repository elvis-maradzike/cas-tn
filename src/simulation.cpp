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
    component->network_->markOptimizableAllTensors();
  }
}

bool Simulation::optimize(std::size_t num_states, double convergence_thresh){
 
  // print out some optimization parameters 
  std::cout << "Starting the optimization procedure" << std::endl;
  std::cout << "Number of spin orbitals: " << total_orbitals_ << std::endl;
  std::cout << "Number of particles: " << num_particles_ << std::endl;
  std::cout << "Number of states: " << num_states_ << std::endl;
  std::cout << "Convergence threshold: " << convergence_thresh_ << std::endl;

  // hamiltonian
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  auto h1 = std::make_shared<exatn::Tensor>("H1",exatn::TensorShape{total_orbitals_, total_orbitals_});
  auto h2 = std::make_shared<exatn::Tensor>("H2",exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_});

  auto created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
  created =      exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);
  
  auto initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized =      exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

  hamiltonian_.push_back(h2);
  hamiltonian_.push_back(h1);

  // set up wavefunction and its optimization
  markOptimizableTensors();
  appendOrderingProjectors();
  constructEnergyFunctional();
  constructEnergyDerivatives();
  initWavefunctionAnsatz();
  evaluateEnergyFunctional();
  evaluateEnergyDerivatives();
  // updateWavefunctionAnsatzTensors();
  auto ham = exatn::makeSharedTensorOperator("Hamiltonian");
  
  // two-body
  for ( unsigned int ketIndex1 = 0; ketIndex1 < num_particles_; ketIndex1++){
    for ( unsigned int ketIndex2 = ketIndex1+1; ketIndex2 < num_particles_; ketIndex2++){
      for ( unsigned int braIndex1 = 0; braIndex1 < num_particles_; braIndex1++){    
        for ( unsigned int braIndex2 = braIndex1+1; braIndex2 < num_particles_; braIndex2++){    
          auto appended = ham->appendComponent(hamiltonian_[0],{{ketIndex1,0},{ketIndex2,1}},{{braIndex1,2},{braIndex2,3}},{0.5,0.0}); assert(appended);
        }
      }
    }
  }  

  // one-body
  for ( unsigned int ketIndex = 0; ketIndex < num_particles_; ketIndex++){
    for ( unsigned int braIndex = 0; braIndex < num_particles_; braIndex++){
      auto appended = ham->appendComponent(hamiltonian_[1],{{ketIndex,0}},{{braIndex,1}},{1.0,0.0}); assert(appended);
    std::cout << ketIndex * num_particles_+braIndex << std::endl;
    }
  }
  
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    bool created = false;
  }

  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(ham,ket_ansatz_,1e-2);
  optimizer.resetLearningRate(0.1);
  bool converged = optimizer.optimize();
  bool success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
  
  return true;
}

/** Appends two layers of ordering projectors to the wavefunction ansatz. **/
void Simulation::appendOrderingProjectors(){
  std::cout << "Appending Ordering Projectors..." << std::endl;
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
  auto created = exatn::createTensor("Q", exatn::TensorElementType::REAL64, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  auto initialized = exatn::initTensorData("Q", tmpData); assert(initialized);
  
  auto appended = false;
  unsigned int tensorCounter = 1+num_particles_;
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    // applying layer 1
    for ( unsigned int i = 0; i < num_particles_-1; i=i+2){
      appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      tensorCounter++;
    }
    // applying layer 2
    for ( unsigned int i = 1; i < num_particles_-1; i=i+2){
      appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
    //  network.printIt();
      tensorCounter++;
    }
  }

  ket_ansatz_->printIt();
 
}


void Simulation::constructEnergyFunctional(){

  bool created = false, success = false, destroyed = false;
  bra_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ket_ansatz_);
  bra_ansatz_->conjugate();
  bra_ansatz_->printIt();

  exatn::TensorOperator ham("Hamiltonian");
  auto appended = false;
  // hamiltonian tensors 
  for ( unsigned int ketIndex1 = 0; ketIndex1 < num_particles_; ketIndex1++){
    for ( unsigned int ketIndex2 = ketIndex1+1; ketIndex2 < num_particles_; ketIndex2++){
      for ( unsigned int braIndex1 = 0; braIndex1 < num_particles_; braIndex1++){    
        for ( unsigned int braIndex2 = braIndex1+1; braIndex2 < num_particles_; braIndex2++){    
          appended = ham.appendComponent(hamiltonian_[0],{{ketIndex1,0},{ketIndex2,1}},{{braIndex1,2},{braIndex2,3}},{0.5,0.0}); assert(appended);
        }
      }
    }
  }  

  for ( unsigned int ketIndex = 0; ketIndex < num_particles_; ketIndex++){
    for ( unsigned int braIndex = 0; braIndex < num_particles_; braIndex++){
      appended = ham.appendComponent(hamiltonian_[1],{{ketIndex,0}},{{braIndex,1}},{1.0,0.0}); assert(appended);
    }
  }

  // energy functional as closed product
  exatn::TensorExpansion closedProd((*bra_ansatz_),(*ket_ansatz_),ham);
  functional_ = std::make_shared<exatn::TensorExpansion>(closedProd);
  (*functional_).printIt();
}

void Simulation::constructEnergyDerivatives(){
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    // network.printIt();
    // loop through all optimizable tensors in tensor network
    bool created = false;
    for ( unsigned int i = 1; i < num_particles_+1; i++){ 
      // gettensor; indices for tensors in MPS start from #1
      std::cout << iter->network_->getTensor(i)->getName() << std::endl; 
      const std::string TENSOR_NAME = iter->network_->getTensor(i)->getName();
      const exatn::TensorShape shape = (*exatn::getTensor(TENSOR_NAME)).getShape();
      created = exatn::createTensor("DerivTensor_" + TENSOR_NAME,exatn::TensorElementType::REAL64,shape); assert(created);
      auto volume = (*exatn::getTensor("DerivTensor_" + TENSOR_NAME)).getVolume();
      std::cout << volume << std::endl;
      exatn::TensorExpansion tmpDerivExpansion((*functional_),TENSOR_NAME,true);
      tmpDerivExpansion.rename("DerivExpansion_"+TENSOR_NAME);
      auto derivExpansion = std::make_shared<exatn::TensorExpansion>(tmpDerivExpansion);
      auto derivative = std::make_shared<exatn::Tensor>((*exatn::getTensor("DerivTensor_" + TENSOR_NAME)));
      derivatives_.push_back(std::make_tuple(TENSOR_NAME,derivExpansion,derivative));
    }
  }
}

void Simulation::initWavefunctionAnsatz(){
  // initialize all tensors that comprise the wavefunction ansatz
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    // network.printIt();
    // loop through all optimizable tensors in network
    bool initialized = false;
    bool success = false;
    for ( unsigned int i = 1; i < num_particles_+1; i++){ 
      // gettensor; indices for optimizable tensors in MPS start from #1
      std::cout << iter->network_->getTensor(i)->getName() << std::endl; 
      const std::string TENSOR_NAME = iter->network_->getTensor(i)->getName();
      initialized = exatn::initTensorRnd(TENSOR_NAME); assert(initialized);
    }
  }

  // compute norm of MPS
  auto created = exatn::createTensorSync("AC_NetworkABCD",exatn::TensorElementType::REAL64, exatn::TensorShape{total_orbitals_, total_orbitals_, total_orbitals_,total_orbitals_}); assert(created);
  auto acNetworkABCD = exatn::getTensor("AC_NetworkABCD");
  bool evaluated = exatn::evaluateSync((*ket_ansatz_),acNetworkABCD); assert(evaluated);
  double norm = 0.;
  auto success = exatn::computeNorm2Sync("AC_NetworkABCD", norm); assert(success);
  std::cout << "Norm of MPS ansatz: " << norm << std::endl;

  // scale constituent tensors
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    // loop through all optimizable tensors in network
    bool initialized = false;
    bool success = false;
    for ( unsigned int i = 1; i < num_particles_+1; i++){ 
      // gettensor; indices for optimizable tensors in MPS start from #1
      std::cout << iter->network_->getTensor(i)->getName() << std::endl; 
      const std::string TENSOR_NAME = iter->network_->getTensor(i)->getName();
      success     = exatn::scaleTensor(TENSOR_NAME, pow((1.0/norm),1./double(num_particles_))); assert(success);
    }
  }
}

double Simulation::evaluateEnergyFunctional(){
  double energy = 0.0;
  // create accumulator rensor for the closed tensor expansion
  auto created = exatn::createTensorSync("AC0",exatn::TensorElementType::REAL64, exatn::TensorShape{}); assert(created);
  auto accumulator0 = exatn::getTensor("AC0");
  bool evaluated = exatn::evaluateSync((*functional_),accumulator0); assert(evaluated);
  // get value 
  auto talsh_tensor = exatn::getLocalTensor("AC0");
  const double * body_ptr;
  if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
    for ( int i = 0; i < talsh_tensor->getVolume(); i++){
      energy  = fabs(body_ptr[i]);
    }
  }
  body_ptr = nullptr;
  state_energies_.clear();
  for ( unsigned int i = 0; i < num_states_; ++i){
    state_energies_.push_back(energy);
  }

  std::cout << "Energy: " << energy << std::endl;
  auto destroyed = exatn::destroyTensorSync("AC0"); assert(destroyed);
  return energy;
}

void Simulation::evaluateEnergyDerivatives(){
  for (unsigned int i = 0; i < derivatives_.size(); i++){
    auto TENSOR_NAME = std::get<0>(derivatives_[i]);
    const exatn::TensorShape SHAPE = (*exatn::getTensor(TENSOR_NAME)).getShape();
    auto accumulator1 = exatn::getTensor("DerivTensor_" + TENSOR_NAME);
    auto accumulator1Name = (*accumulator1).getName();
    auto derivExpansion = std::get<1>(derivatives_[i]);
    auto derivExpansionName = (*derivExpansion).getName();
    std::cout << derivExpansionName << std::endl;
    std::cout << accumulator1Name << std::endl;
    auto evaluated = exatn::evaluateSync((*derivExpansion), accumulator1); assert(evaluated);
  }
}
  
void Simulation::updateWavefunctionAnsatzTensors(){
/*  exatn::TensorExpansion lagrangian(*functional_);
  std::unordered_set<std::string> tensor_names;

  for (auto iter = (*bra_ansatz_).begin(); iter != (*bra_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    for ( auto iter2 = 1; iter2 != num_particles_+1; iter2++){

      std::cout << iter->network_->getTensor(iter2)->getName() << std::endl;
      const std::string TENSOR_NAME = iter->network_->getTensor(iter2)->getName();
      const exatn::TensorShape TENSOR_SHAPE = (*exatn::getTensor(TENSOR_NAME)).getShape();
      const exatn::TensorSignature TENSOR_SIGNATURE = (*exatn::getTensor(TENSOR_NAME)).getSignature();
      std::cout << " tensor name is " << TENSOR_NAME  << std::endl;

      unsigned int rank = iter->network_->getTensor(iter2)->getRank();
      std::cout << "Rank of tensor " << TENSOR_NAME << " is " << rank << std::endl;

      auto tensor = exatn::getTensor(TENSOR_NAME);
           std::cout << " optimizable tensor name is " << TENSOR_NAME  << std::endl;

      auto res = tensor_names.emplace(TENSOR_NAME);

      auto gradTensor = exatn::getTensor("DerivTensor_" + TENSOR_NAME);
      auto derivExpansion = std::get<1>(derivatives_[iter2-1]);
      environments_.emplace_back(Environment{exatn::getTensor(TENSOR_NAME),
                                            std::make_shared<exatn::Tensor>("gradTensor_"+TENSOR_NAME,
                                                                     TENSOR_SHAPE,
                                                                     TENSOR_SIGNATURE),
                                             exatn::TensorExpansion(lagrangian,TENSOR_NAME,true)
                                           });
    }
  }

 //Optimization procedure:
  bool converged = (environments_.size() == 0);
  bool success = false, done = false;
  std::cout << " number to be optimized: " << environments_.size() << std::endl;

  for(auto & environment: environments_){
      double maxAbsInGrad = 0.;
      //Create the gradient tensor:
      done = exatn::createTensorSync(environment.gradient,environment.tensor->getElementType()); assert(done);
      //Initialize the gradient tensor to zero:
      done = exatn::initTensorSync(environment.gradient->getName(),0.0); assert(done);
      std::cout << " Before addition "  << environment.gradient->getName() << std::endl;
     // success = exatn::printTensorSync(environment.gradient->getName()); assert(success);
      success = exatn::printTensorSync(environment.tensor->getName()); assert(success);
      //Evaluate the gradient tensor expansion:
      done = exatn::evaluateSync(environment.gradient_expansion,environment.gradient); assert(done);
      //Update the optimizable tensor using the computed gradient (conjugated):
      std::string add_pattern;
      done = exatn::generate_addition_pattern(environment.tensor->getRank(),add_pattern,true,
                                     environment.tensor->getName(),environment.gradient->getName()); assert(done);
      std::cout << add_pattern << std::endl;
      done = exatn::addTensors(add_pattern,-0.1); assert(done);
      std::cout << " After addition: "  << environment.gradient->getName() << std::endl;
     // success = exatn::printTensorSync(environment.gradient->getName()); assert(success);
      success = exatn::printTensorSync(environment.tensor->getName()); assert(success);
      success = exatn::computeMaxAbsSync(environment.gradient->getName(), maxAbsInGrad); assert(success);
      std::cout << "Max Abs element in derivative w.r.t " << environment.gradient->getName() << " is " << maxAbsInGrad << std::endl;
      //Destroy the gradient tensor:
      done = exatn::destroyTensorSync(environment.gradient->getName()); assert(done);
  }*/

} 
 
} //namspace castn
