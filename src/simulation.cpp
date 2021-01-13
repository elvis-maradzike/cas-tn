/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation
REVISION: 2021/01/08

Copyright (C) 2020-2021 Dmitry I. Lyakh (Liakh), Elvis Maradzike
Copyright (C) 2020-2021 Oak Ridge National Laboratory (UT-Battelle)

**/

#include "simulation.hpp"
#include "exatn.hpp"
#include "talshxx.hpp"

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

bool Simulation::optimize(std::size_t num_states, double convergence_thresh){
  
  std::cout << "Starting the optimization procedure" << std::endl;
  std::cout << "Number of spin orbitals: " << total_orbitals_ << std::endl;
  std::cout << "Number of particles: " << num_particles_ << std::endl;
  std::cout << "Number of states: " << num_states_ << std::endl;
  std::cout << "Convergence threshold: " << convergence_thresh_ << std::endl;

  appendOrderingProjectors();
  constructEnergyFunctional();
  constructEnergyDerivatives();
  initWavefunctionAnsatz();
  evaluateEnergyFunctional();
  evaluateEnergyDerivatives();
  updateWavefunctionAnsatzTensors();
  return true;
}

/** Appends two layers of ordering projectors to the wavefunction ansatz. **/
void Simulation::appendOrderingProjectors(){
  std::cout << "Appending Ordering Projectors..." << std::endl;
  (*ket_ansatz_).printIt(); 
  bool isket = (*ket_ansatz_).isKet();
  std::cout << "isket?: " << isket << std::endl;

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
 
  // create tensors for ordering projectors
  for ( unsigned int i = 0; i < num_particles_-1; i++){
    const bool created = exatn::createTensor("Q" + std::to_string(i) + std::to_string(i+1), exatn::TensorElementType::REAL64, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  
    const bool initialized = exatn::initTensorData("Q" + std::to_string(i) + std::to_string(i+1), tmpData); assert(initialized);
  }
  
    
  //exatn::TensorExpansion e = (*ket_ansatz_);
  bool appended = false;
  unsigned int tensorCounter = 1+num_particles_;
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    network.printIt();
    // applying layer 1
    for ( unsigned int i = 0; i < num_particles_-1; i=i+2){
      std::cout << "i: " << i << std::endl;
      appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q" + std::to_string(i) + std::to_string(i+1)),{i,i+1}); assert(appended);
      network.printIt();
      tensorCounter++;
    }
    // applying layer 2
    for ( unsigned int i = 1; i < num_particles_-1; i=i+2){
      std::cout << "i: " << i << std::endl;
      appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q" + std::to_string(i) + std::to_string(i+1)),{i,i+1}); assert(appended);
      network.printIt();
      tensorCounter++;
    }
  }

  
}

void Simulation::constructEnergyFunctional(){

  bra_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ket_ansatz_);
  auto h1 = std::make_shared<exatn::Tensor>("H1",exatn::TensorShape{total_orbitals_,total_orbitals_});
  auto h2 = std::make_shared<exatn::Tensor>("H2",exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_});

  bool created = false;
  created = createTensor(h1, exatn::TensorElementType::REAL64); assert(created);
  created = createTensor(h2, exatn::TensorElementType::REAL64); assert(created);

  bool success = false;
  success = exatn::initTensorFile("H1","oei.txt"); assert(success);
  success = exatn::initTensorFile("H2","tei.txt"); assert(success);

  hamiltonian_.push_back(h1);
  hamiltonian_.push_back(h2);

  unsigned int i = 0;
  for ( unsigned int ketIndex1 = 0; ketIndex1 < num_particles_; ketIndex1++){
    for ( unsigned int ketIndex2 = ketIndex1+1; ketIndex2 < num_particles_; ketIndex2++){
      for ( unsigned int braIndex1 = 0; braIndex1 < num_particles_; braIndex1++){    
        for ( unsigned int braIndex2 = braIndex1+1; braIndex2 < num_particles_; braIndex2++){    
         const bool created = exatn::createTensor("H2_" + std::to_string(ketIndex1) + std::to_string(ketIndex2) + "_" + std::to_string(braIndex1) + std::to_string(braIndex2), exatn::TensorElementType::REAL64, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_});
         success = exatn::initTensorFile("H2_" + std::to_string(ketIndex1) + std::to_string(ketIndex2) + "_" + std::to_string(braIndex1) + std::to_string(braIndex2),"tei.txt"); assert(success);
        }
      }
    }
  }  
  for ( unsigned int ketIndex = 0; ketIndex < num_particles_; ketIndex++){
    for ( unsigned int braIndex = 0; braIndex < num_particles_; braIndex++){
      const bool created = exatn::createTensor("H1_" + std::to_string(ketIndex) + "_" + std::to_string(braIndex), exatn::TensorElementType::REAL64, exatn::TensorShape{total_orbitals_,total_orbitals_});
         success = exatn::initTensorFile("H1_" + std::to_string(ketIndex) + "_" + std::to_string(braIndex),"oei.txt"); assert(success);
    
    }
  }
  exatn::TensorOperator ham("Hamiltonian");
  auto appended = false;
  // hamiltonian tensors 
  for ( unsigned int ketIndex1 = 0; ketIndex1 < num_particles_; ketIndex1++){
    for ( unsigned int ketIndex2 = ketIndex1+1; ketIndex2 < num_particles_; ketIndex2++){
      for ( unsigned int braIndex1 = 0; braIndex1 < num_particles_; braIndex1++){    
        for ( unsigned int braIndex2 = braIndex1+1; braIndex2 < num_particles_; braIndex2++){    
          appended = ham.appendComponent(exatn::getTensor("H2_" + std::to_string(ketIndex1) + std::to_string(ketIndex2) + "_" + std::to_string(braIndex1) + std::to_string(braIndex2)),{{ketIndex1,0},{ketIndex2,1}},{{braIndex1,2},{braIndex2,3}},{1.0,0.0}); assert(appended);
        }
      }
    }
  }  
//  ham.printIt();

  for ( unsigned int ketIndex = 0; ketIndex < num_particles_; ketIndex++){
    for ( unsigned int braIndex = 0; braIndex < num_particles_; braIndex++){
      appended = ham.appendComponent(exatn::getTensor("H1_" + std::to_string(ketIndex) + "_" + std::to_string(braIndex)),{{ketIndex,0}},{{braIndex,1}},{1.0,0.0}); assert(appended);
    std::cout << ketIndex * num_particles_+braIndex << std::endl;
    }
  }
//  ham.printIt();

    (*bra_ansatz_).conjugate();
    (*bra_ansatz_).printIt();
  
    exatn::TensorExpansion tmp2((*bra_ansatz_),(*ket_ansatz_),ham);
    functional_ = std::make_shared<exatn::TensorExpansion>(tmp2);
    (*functional_).printIt();
  
}

void Simulation::constructEnergyDerivatives(){
  // exatn::TensorExpansion tmp((*functional_),"A",true);
  // tmp.rename("DerivativeA");
  // tmp.printIt();

  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    network.printIt();
    // loop through network
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
      auto gradExpansion = std::make_shared<exatn::TensorExpansion>(tmpDerivExpansion);
      auto grad = std::make_shared<exatn::Tensor>((*exatn::getTensor("DerivTensor_" + TENSOR_NAME)));
      derivatives_.push_back(std::make_tuple(TENSOR_NAME,gradExpansion,grad));
    }
  }
}

void Simulation::initWavefunctionAnsatz(){
  // initialize all tensors that comprise the wavefunction ansatz
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    network.printIt();
    // loop through network
    bool initialized = false;
    bool success = false;
    for ( unsigned int i = 1; i < num_particles_+1; i++){ 
      // gettensor; indices for tensors in MPS start from #1
      std::cout << iter->network_->getTensor(i)->getName() << std::endl; 
      const std::string TENSOR_NAME = iter->network_->getTensor(i)->getName();
      initialized = exatn::initTensorRnd(TENSOR_NAME); assert(initialized);
      // normalize tensor
      double norm = 0.;
      success = exatn::computeNorm2Sync(TENSOR_NAME, norm); assert(success);
      success = exatn::scaleTensor(TENSOR_NAME, 4.0/norm); assert(success);
      success = exatn::printTensorSync(TENSOR_NAME); assert(success);
    }
  }
}

double Simulation::evaluateEnergyFunctional(){
  double energy = 0.0;
  // create accumulator rensor for the closed tensor expansion
  bool created = false;
  created = exatn::createTensorSync("AC0",exatn::TensorElementType::REAL64, exatn::TensorShape{}); assert(created);
  auto accumulator0 = exatn::getTensor("AC0");
  // evaluate the closed tensor expansion
  bool evaluated = false;
  evaluated = exatn::evaluateSync((*functional_),accumulator0); assert(evaluated);
  // get value 
  auto talsh_tensor = exatn::getLocalTensor("AC0");
  const double * body_ptr;
  if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
    for ( int i = 0; i < talsh_tensor->getVolume(); i++){
      energy  = fabs(body_ptr[i]);
    }
  }
  body_ptr = nullptr;

  std::cout << "Energy: " << energy << std::endl;
  return energy;
}

void Simulation::evaluateEnergyDerivatives(){
  for (unsigned int i = 0; i < derivatives_.size(); i++){
    auto TENSOR_NAME = std::get<0>(derivatives_[i]);
    const exatn::TensorShape SHAPE = (*exatn::getTensor(TENSOR_NAME)).getShape();
  //  bool created = false;
  //  created = exatn::createTensorSync("AC1_" + TENSOR_NAME, exatn::TensorElementType::REAL64, SHAPE); assert(created);
    //auto accumulator1 = exatn::getTensor("AC1_" + TENSOR_NAME);
    auto accumulator1 = exatn::getTensor("DerivTensor_" + TENSOR_NAME);
    auto accumulator1Name = (*accumulator1).getName();
    std::cout << accumulator1Name << std::endl;
    bool evaluated = false;
    auto derivExpansion = std::get<1>(derivatives_[i]);
    auto derivExpansionName = (*derivExpansion).getName();
    std::cout << derivExpansionName << std::endl;
    evaluated = exatn::evaluateSync((*derivExpansion), accumulator1); assert(evaluated);
  }
}
  
void Simulation::updateWavefunctionAnsatzTensors(){
  // get wavefunction ansatz tensors 
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    network.printIt();
    // loop through network
    bool initialized = false;
    bool success = false;
    for ( unsigned int i = 1; i < num_particles_+1; i++){ 
      // gettensor; indices for tensors in MPS start from #1
      std::cout << iter->network_->getTensor(i)->getName() << std::endl; 
      const std::string TENSOR_NAME = iter->network_->getTensor(i)->getName();
      double maxAbsInGrad = 0.;
      success = exatn::computeMaxAbsSync("DerivTensor_" + TENSOR_NAME, maxAbsInGrad); assert(success);
      std::cout << "Max Abs element in derivative w.r.t " << TENSOR_NAME << " is " << maxAbsInGrad << std::endl;
    }
  }
  
  // get derivatives 
  

}
} //namspace castn
