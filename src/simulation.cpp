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
  created = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);
  
  auto initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

  hamiltonian_.push_back(h2);
  hamiltonian_.push_back(h1);

  // set up wavefunction and its optimization
  markOptimizableTensors();
  appendOrderingProjectors();
  constructEnergyFunctional();
  constructEnergyDerivatives();
  initWavefunctionAnsatz();

  double energyOld = 0.0, energyNew = 1.0, energyChange = 0.0, maxAbsGrad = 0.0;

  // overall gradient convergence and energy decrease
  int iter = 0;
  do {
    energyOld = energyNew;
    energyNew = evaluateEnergyFunctional();
    evaluateEnergyDerivatives();
    updateWavefunctionAnsatzTensors();
    iter++;
    energyChange = energyNew - energyOld;
    std::cout << "Energy Change: " << energyChange << std::endl;
  }while ((fabs(energyChange) > convergence_thresh_));
 
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
    iter->network;
    iter->coefficient;
    auto & network = *(iter->network);
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

  auto ham = exatn::makeSharedTensorOperator("Hamiltonian");
  auto appended = false;
  // hamiltonian tensors 
  success = ham->appendSymmetrizeComponent(hamiltonian_[0],{0,1},{2,3}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  success = ham->appendSymmetrizeComponent(hamiltonian_[1],{0},{1}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);

  // energy functional as closed product
  exatn::TensorExpansion psiHpsi(*bra_ansatz_,*ket_ansatz_,*ham);
  functional_ = std::make_shared<exatn::TensorExpansion>(psiHpsi);
  functional_->printIt();
  
  exatn::TensorExpansion psipsi(*bra_ansatz_,*ket_ansatz_);
  norm_ = std::make_shared<exatn::TensorExpansion>(psipsi);
  norm_->printIt();
}

void Simulation::constructEnergyDerivatives(){
  for (auto iter = ket_ansatz_->begin(); iter != ket_ansatz_->end(); ++iter){
    iter->network;
    iter->coefficient;
    auto & network = *(iter->network);
    // loop through all optimizable tensors in tensor network
    bool created = false;
    for ( unsigned int i = 1; i < num_particles_+1; i++){ 
      // gettensor; indices for tensors in MPS start from #1
      std::cout << iter->network->getTensor(i)->getName() << std::endl; 
      const std::string TENSOR_NAME = iter->network->getTensor(i)->getName();
      const exatn::TensorShape SHAPE = (*exatn::getTensor(TENSOR_NAME)).getShape();
      created = exatn::createTensor("DerivativeTensorE_"+TENSOR_NAME,exatn::TensorElementType::REAL64,SHAPE); assert(created);
      created = exatn::createTensor("DerivativeTensorN_"+TENSOR_NAME,exatn::TensorElementType::REAL64,SHAPE); assert(created);
      exatn::TensorExpansion tmp1(*functional_,TENSOR_NAME,true);
      exatn::TensorExpansion tmp2(*norm_,TENSOR_NAME,true);
      tmp1.rename("DerivativeExpansionE_"+TENSOR_NAME);
      tmp2.rename("DerivativeExpansionN_"+TENSOR_NAME);
      auto derivativeExpansionEnergyFunctional = std::make_shared<exatn::TensorExpansion>(tmp1);
      auto derivativeExpansionNormFunctional = std::make_shared<exatn::TensorExpansion>(tmp2);
      auto derivativeEnergyFunctional = std::make_shared<exatn::Tensor>((*exatn::getTensor("DerivativeTensorE_" + TENSOR_NAME)));
      auto derivativeNormFunctional = std::make_shared<exatn::Tensor>((*exatn::getTensor("DerivativeTensorN_" + TENSOR_NAME)));
      derivatives_.push_back(std::make_tuple(TENSOR_NAME,derivativeExpansionEnergyFunctional,derivativeEnergyFunctional, derivativeExpansionNormFunctional, derivativeNormFunctional));
    }
  }
}

void Simulation::initWavefunctionAnsatz(){
  // initialize all tensors that comprise the wavefunction ansatz
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    iter->network;
    iter->coefficient;
    auto & network = *(iter->network);
    // network.printIt();
    // loop through all optimizable tensors in network
    bool initialized = false;
    bool success = false;
    for ( unsigned int i = 1; i < num_particles_+1; i++){ 
      // gettensor; indices for optimizable tensors in MPS start from #1
      std::cout << iter->network->getTensor(i)->getName() << std::endl; 
      const std::string TENSOR_NAME = iter->network->getTensor(i)->getName();
      initialized = exatn::initTensorRnd(TENSOR_NAME); assert(initialized);
    }
  }
}

double Simulation::evaluateEnergyFunctional(){
  // create accumulator rensor for the closed tensor expansion
  auto created = exatn::createTensorSync("AC_PsiHPsi",exatn::TensorElementType::REAL64, exatn::TensorShape{}); assert(created);
  created = exatn::createTensorSync("AC_PsiPsi",exatn::TensorElementType::REAL64, exatn::TensorShape{}); assert(created);
  auto ac_PsiHPsi = exatn::getTensor("AC_PsiHPsi");
  auto ac_PsiPsi = exatn::getTensor("AC_PsiPsi");
  exatn::TensorExpansion psipsi(*bra_ansatz_,*ket_ansatz_);
  bool evaluated = exatn::evaluateSync((*functional_),ac_PsiHPsi); assert(evaluated);
  evaluated = exatn::evaluateSync(psipsi,ac_PsiPsi); assert(evaluated);

  // get value 
  double energy = 0.0;
  double val_psihpsi = 0.0;
  double val_psipsi = 0.0;
  auto talsh_tensor = exatn::getLocalTensor("AC_PsiHPsi");
  const double * body_ptr;
  if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
    for ( int i = 0; i < talsh_tensor->getVolume(); i++){
      val_psihpsi  = body_ptr[i];
    }
  }
  body_ptr = nullptr;
  talsh_tensor = exatn::getLocalTensor("AC_PsiPsi");
  if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
    for ( int i = 0; i < talsh_tensor->getVolume(); i++){
      val_psipsi  = body_ptr[i];
      normVal_ = body_ptr[i];
    }
  }
  body_ptr = nullptr;

  energy = val_psihpsi/val_psipsi;

  // store value
  state_energies_.clear();
  for ( unsigned int i = 0; i < num_states_; ++i){
    state_energies_.push_back(energy);
  }
  std::cout << "Energy: " << energy << " Eh" << std::endl;
  // destroy AC0
  auto destroyed = exatn::destroyTensorSync("AC_PsiHPsi"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("AC_PsiPsi"); assert(destroyed);

  return energy;
}

void Simulation::evaluateEnergyDerivatives(){
  //  derivative: <Psi|H|Psi>
  for (unsigned int i = 0; i < derivatives_.size(); i++){
    auto TENSOR_NAME = std::get<0>(derivatives_[i]);
    auto TENSOR = exatn::getTensor(TENSOR_NAME);
    const exatn::TensorShape TENSOR_SHAPE = (*exatn::getTensor(TENSOR_NAME)).getShape();
    const exatn::TensorSignature TENSOR_SIGNATURE = (*exatn::getTensor(TENSOR_NAME)).getSignature();
    unsigned int RANK = TENSOR->getRank();
    auto ac_deriv_e = exatn::getTensor("DerivativeTensorE_" + TENSOR_NAME);
    auto ac_deriv_e_name = (*ac_deriv_e).getName();
    auto derivative_expansion_e = std::get<1>(derivatives_[i]);
    auto derivative_expansion_name_e = (*derivative_expansion_e).getName();
    auto ac_deriv_n = exatn::getTensor("DerivativeTensorN_" + TENSOR_NAME);
    auto ac_deriv_n_name = (*ac_deriv_n).getName();
    auto derivative_expansion_n = std::get<3>(derivatives_[i]);
    auto derivative_expansion_name_n = (*derivative_expansion_n).getName();
    std::cout << derivative_expansion_name_n << std::endl;
    std::cout << ac_deriv_n_name << std::endl;
    auto evaluated = exatn::evaluateSync((*derivative_expansion_e), ac_deriv_e); assert(evaluated);
    evaluated = exatn::evaluateSync((*derivative_expansion_n), ac_deriv_n); assert(evaluated);
    // get maxAbsGrad and maxAbs
    auto grad = std::make_shared<exatn::Tensor>("Grad_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    auto created = exatn::createTensorSync(grad,TENSOR->getElementType()); assert(created);
    auto initialized = exatn::initTensorSync(grad->getName(),0.0); assert(initialized);
    std::string add_pattern;
    auto generated = exatn::generate_addition_pattern(RANK,add_pattern,true,grad->getName(), ac_deriv_e_name); assert(generated);
    auto added = exatn::addTensors(add_pattern,-1.0/normVal_); assert(added);
    add_pattern.clear();
    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,grad->getName(), ac_deriv_n_name); assert(generated);
    double energy_tmp = state_energies_[0];
    added = exatn::addTensors(add_pattern,1.0 * energy_tmp/(normVal_ * normVal_)); assert(added);
    auto success = exatn::computeMaxAbsSync(grad->getName(), maxAbsGrad_); assert(success);
    std::cout << "Max Abs element in derivative w.r.t " << grad->getName() << " is " << maxAbsGrad_ << std::endl;
    std::cout << " Optimizable tensor name is " << TENSOR_NAME  << std::endl;
    environments_.emplace_back(Environment{exatn::getTensor(TENSOR_NAME),
                 std::make_shared<exatn::Tensor>("Grad_"+TENSOR_NAME,TENSOR_SHAPE, TENSOR_SIGNATURE)
                 });
   
    std::cout << "Done evaluating derivative" << std::endl;   
  }
}
  
void Simulation::updateWavefunctionAnsatzTensors(){
  for (auto & environment: environments_){
    const std::string TENSOR_NAME = environment.tensor->getName();
    auto TENSOR = exatn::getTensor(TENSOR_NAME);
    const std::string GRAD_NAME = environment.gradient->getName();
    const exatn::TensorShape TENSOR_SHAPE = environment.tensor->getShape();
    const exatn::TensorSignature TENSOR_SIGNATURE = environment.tensor->getSignature();
    const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
    unsigned int RANK = environment.tensor->getRank();
    auto talsh_tensor = exatn::getLocalTensor(GRAD_NAME);
    const double * body_ptr;
    //if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
    auto VOL = talsh_tensor->getVolume();
   // }
    std::cout << TENSOR_NAME << VOL << std::endl;
    // ADAM
    // auxiliary vectors
    // squared gradient
    std::vector<double> g2_aux(VOL);
    // first order momenta
    std::vector<double> ov1_aux(VOL);
    // second order momenta
    std::vector<double> ov2_aux(VOL);
    // effective gradient
    std::vector<double> eg_aux(VOL);
  
    if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
      for ( int i = 0; i < talsh_tensor->getVolume(); i++){
        g2_aux[i] =  body_ptr[i] * body_ptr[i];
      }
    }
    body_ptr = nullptr;

    // squared gradients
    auto g2 = std::make_shared<exatn::Tensor>("G2_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    auto created = exatn::createTensorSync(g2,TENSOR->getElementType()); assert(created);
    auto initialized = exatn::initTensorDataSync(g2->getName(),g2_aux); assert(initialized);

    // effective gradients
    auto eg = std::make_shared<exatn::Tensor>("EG_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    created = exatn::createTensorSync(eg,TENSOR->getElementType()); assert(created);
    initialized = exatn::initTensor(eg->getName(), 0.0); assert(initialized);

    // first-order momenta
    auto ov1 = std::make_shared<exatn::Tensor>("OV1_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    created = exatn::createTensorSync(ov1,TENSOR->getElementType()); assert(created);
    auto v1p = std::make_shared<exatn::Tensor>("V1P_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    created = exatn::createTensorSync(v1p,TENSOR->getElementType()); assert(created);
    auto v1c = std::make_shared<exatn::Tensor>("V1C_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    created = exatn::createTensorSync(v1c,TENSOR->getElementType()); assert(created);

    // second-order momenta
    auto ov2 = std::make_shared<exatn::Tensor>("OV2_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    created = exatn::createTensorSync(ov2,TENSOR->getElementType()); assert(created);
    auto v2p = std::make_shared<exatn::Tensor>("V2P_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    created = exatn::createTensorSync(v2p,TENSOR->getElementType()); assert(created);
    auto v2c = std::make_shared<exatn::Tensor>("V2C_"+TENSOR_NAME, TENSOR_SHAPE, TENSOR_SIGNATURE);
    created = exatn::createTensorSync(v2c,TENSOR->getElementType()); assert(created);

    auto b1 = 0.9, b2 = 0.999, e  = 1.e-8, lr = 0.8;
   
    // compute first-order momenta
    initialized = exatn::initTensor(v1c->getName(), 0.0); assert(initialized);
    if (iter == 0){
      initialized = exatn::initTensor(eg->getName(), 0.0); assert(initialized);
      initialized = exatn::initTensor(v1p->getName(), 0.0); assert(initialized);
    }
    initialized = exatn::initTensor(ov1->getName(), 0.0); assert(initialized);
    std::string add_pattern;
    auto generated = exatn::generate_addition_pattern(RANK,add_pattern,true,v1c->getName(), v1p->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    auto added = exatn::addTensors(add_pattern, b1); assert(added);
    add_pattern.clear();
    std::cout << "Gradient name is :" << GRAD_NAME << std::endl;
    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,v1c->getName(), GRAD_NAME); assert(generated);
    std::cout << add_pattern << std::endl;
    added = exatn::addTensors(add_pattern, 1.-b1); assert(added);
    add_pattern.clear();
    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,ov1->getName(), v1c->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    auto factor1 = 1./(1.-pow(b1,double(iter+1)));
    added = exatn::addTensors(add_pattern, factor1); assert(added);
    add_pattern.clear();

    // compute second-order momenta
    initialized = exatn::initTensor(v2c->getName(), 0.0); assert(initialized);
    if (iter == 0){
      initialized = exatn::initTensor(v2p->getName(), 0.0); assert(initialized);
    }
    initialized = exatn::initTensor(ov2->getName(), 0.0); assert(initialized);

    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,v2c->getName(), v2p->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    added = exatn::addTensors(add_pattern, b2); assert(added);
    add_pattern.clear();
    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,v2c->getName(), g2->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    added = exatn::addTensors(add_pattern, 1.-b2); assert(added);
    add_pattern.clear();
    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,ov2->getName(), v2c->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    auto factor2 = 1./(1.-pow(b2,double(iter+1)));
    added = exatn::addTensors(add_pattern, factor2); assert(added);
    add_pattern.clear();

    // compute effective gradient
    talsh_tensor = exatn::getLocalTensor(ov1->getName());
    if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
      for ( int i = 0; i < talsh_tensor->getVolume(); i++){
        ov1_aux[i] = body_ptr[i];
      }
    }
    body_ptr = nullptr;
    talsh_tensor = exatn::getLocalTensor(ov2->getName());
    if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
      for ( int i = 0; i < talsh_tensor->getVolume(); i++){
        ov2_aux[i] = body_ptr[i];
      }
    }
    body_ptr = nullptr;

    for ( int i = 0; i < VOL; i++){
      eg_aux[i] = ov1_aux[i]/(sqrt(ov2_aux[i]) + e);
    }
    
    initialized = exatn::initTensorData(eg->getName(), eg_aux); assert(initialized);

    // update variable
    std::cout << " Before addition "  << TENSOR_NAME << std::endl;
    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,TENSOR_NAME, eg->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    added = exatn::addTensors(add_pattern,-lr); assert(added);
    add_pattern.clear();
   
    initialized = exatn::initTensorSync(v1p->getName(), 0.0); assert(initialized);
    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,v1p->getName(), v1c->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    added = exatn::addTensors(add_pattern, 1.0); assert(added);
    add_pattern.clear();
    
    initialized = exatn::initTensorSync(v2p->getName(), 0.0); assert(initialized);
    generated = exatn::generate_addition_pattern(RANK,add_pattern,true,v2p->getName(), v2c->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    added = exatn::addTensors(add_pattern, 1.0); assert(added);
    add_pattern.clear();

    auto destroyed = exatn::destroyTensorSync(environment.gradient->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(g2->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(eg->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(ov1->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(v1p->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(v1c->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(ov2->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(v2p->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(v2c->getName()); assert(destroyed);
  } 
  environments_.clear();
}
 
} //namespace castn
