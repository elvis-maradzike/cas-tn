#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>

using namespace std::chrono;

int main(int argc, char** argv){

  // active orbitals
  std::size_t nAO = 8;
  // active particles
  std::size_t nAP = 4;
  // core orbitals
  std::size_t nCO = 0;
  // total orbitals
  std::size_t nTO = 8;
  // total particles
  std::size_t nTP = 4;
  
  auto success = false, created = false, initialized = false, appended = false;
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;

  exatn::initialize();
  {
  // declare tensor network
  auto x = std::make_shared<exatn::Tensor>("X", exatn::TensorShape{nTO,nTO,nTO,nTO});
  auto tmp = std::make_shared<exatn::Tensor>("TMP", exatn::TensorShape{nTO,nTO,nTO,nTO});
  auto xtx = std::make_shared<exatn::Tensor>("XtX", exatn::TensorShape{1});
  auto xOld = std::make_shared<exatn::Tensor>("XOld", exatn::TensorShape{nTO,nTO,nTO,nTO});
  auto tn_abcd = std::make_shared<exatn::Tensor>("TN_ABCD", exatn::TensorShape{nTO,nTO,nTO,nTO});
  

  // create constituent tensors
  created = exatn::createTensor(x, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(tmp, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(xtx, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(xOld, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor("Grad", TENS_ELEM_TYPE,x->getShape()); assert(created);
  //auto accumulator1 = exatn::getTensor("GRAD");


  // hamiltonian
  auto h1 = std::make_shared<exatn::Tensor>("H1",exatn::TensorShape{nTO, nTO});
  auto h2 = std::make_shared<exatn::Tensor>("H2",exatn::TensorShape{nTO, nTO, nTO, nTO});
  auto network_abcd = exatn::makeSharedTensorNetwork(
                 "NetworkABCD", //tensor network name
                 "TN_ABCD(p,q,r,s)+=X(p,q,r,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"TN_ABCD",tn_abcd},{"X",x}
                 }
                );
  std::shared_ptr<exatn::TensorExpansion> expansion_abcd_ket;
  expansion_abcd_ket = std::make_shared<exatn::TensorExpansion>(); 
  expansion_abcd_ket->appendComponent(network_abcd,{1.0,0.0});

  created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);

  initialized = exatn::initTensorSync("XtX",0.0); assert(initialized);
  initialized = exatn::initTensorRnd("X"); assert(initialized);
  //initialized = exatn::initTensorFile("TMP","h4_tensor.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

  /*
  auto  talsh_tensor = exatn::getLocalTensor("TMP");
  const double * body_ptr;
  auto access_granted = false;

  std::vector<double> tmpDatax(nTO*nTO*nTO*nTO,0.0);
  for ( unsigned int i = 0; i < nTO; i++){
    for ( unsigned int j = 0; j < nTO; j++){
      for ( unsigned int k = 0; k < nTO; k++){
        for ( unsigned int l = 0; l < nTO; l++){
          int ijkl = i*nTO*nTO*nTO + j*nTO*nTO + k*nTO + l;
          if ( i < j && j < k && k < l){
            if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
            tmpDatax[ijkl] =  body_ptr[ijkl];
            }
          }else{
            if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
            tmpDatax[ijkl] = 0.0;
            }
          }
        }
      }
    }
  }
  
  body_ptr = nullptr;
  
  initialized = exatn::initTensorData("X", tmpDatax); assert(initialized);
  */
 // initialized = exatn::initTensorFile("X","h4_tensor.txt"); assert(initialized);
  
  success = exatn::printTensorSync("X"); assert(success);
  //success = exatn::printTensorSync("TMP"); assert(success);

  
  // array of elements to initialize tensors that comprise the ordering projectors
  std::vector<double> tmpData(nTO*nTO*nTO*nTO,0.0);
  for ( unsigned int i = 0; i < nTO; i++){
    for ( unsigned int j = 0; j < nTO; j++){
      for ( unsigned int k = 0; k < nTO; k++){
        for ( unsigned int l = 0; l < nTO; l++){
          if ( i < j){
          tmpData[i*nTO*nTO*nTO
                 +j*nTO*nTO
                 +k*nTO
                 +l] = double((i==k)*(j==l));
          }else{
          tmpData[i*nTO*nTO*nTO
                 +j*nTO*nTO
                 +k*nTO
                 +l] = 0.0;
          }
        }
      }
    }
  }
 
  //create tensors: ordering projectors
  created = exatn::createTensor("Q", TENS_ELEM_TYPE, exatn::TensorShape{nTO, nTO, nTO, nTO}); assert(created);
  initialized = exatn::initTensorData("Q", tmpData); assert(initialized);

  std::cout << "Appending Ordering Projectors..." << std::endl;
  unsigned int tensorCounter = 2;
  for (auto iter = expansion_abcd_ket->begin(); iter != expansion_abcd_ket->end(); ++iter){
    iter->network;
    iter->coefficient;
    auto & network = *(iter->network);
    appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{0,1}); assert(appended);
    tensorCounter++;
    appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{2,3}); assert(appended);
    tensorCounter++;
    appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{1,2}); assert(appended);
  } 
 
  // checking expansion
  expansion_abcd_ket->printIt();

  
  // hamiltonian
  auto ham = exatn::makeSharedTensorOperator("Hamiltonian");

  /*
  for ( unsigned int ketIndex1 = 0; ketIndex1 < nTP-1; ketIndex1++){
    for ( unsigned int ketIndex2 = ketIndex1+1; ketIndex2 < nTP; ketIndex2++){
      for ( unsigned int braIndex1 = 0; braIndex1 < nTP-1; braIndex1++){
        for ( unsigned int braIndex2 = braIndex1+1; braIndex2 < nTP; braIndex2++){
          auto appended = ham->appendComponent(h2,{{ketIndex1,0},{ketIndex2,1}},{{braIndex1,2},{braIndex2,3}},{1.0,0.0}); assert(appended);
        }
      }
    }
  }
  
  // one-body part
  for ( unsigned int ketIndex = 0; ketIndex < nTP; ketIndex++){
    for ( unsigned int braIndex = 0; braIndex < nTP; braIndex++){
      auto appended = ham->appendComponent(h1,{{ketIndex,0}},{{braIndex,1}},{1.0,0.0}); assert(appended);
    }
  }
  */

 
 bool success = ham->appendSymmetrizeComponent(h1,{0},{1},4,4,{1.0,0.0},true); assert(success);
 success = ham->appendSymmetrizeComponent(h2,{0,1},{2,3},4,4,{1.0,0.0},true); assert(success);
 

  ham->printIt();
 
  /* 
  // call ExaTN optimizer
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(ham,expansion_abcd_ket,1e-4);
  optimizer.resetLearningRate(0.5);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
    std::cout << "Optimization succeeded!" << std::endl;
    // print ordering projector after optimization
    // success = exatn::printTensorSync("Q"); assert(success);
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
  
  // print ordering projector before optimization
  //success = exatn::printTensorSync("Q"); assert(success);
  */
  
  
 
 // alternative optimization scheme
  // energy functional as closed product
  std::shared_ptr<exatn::TensorExpansion> expansion_abcd_bra;
  expansion_abcd_bra = std::make_shared<exatn::TensorExpansion>(*expansion_abcd_ket);
  expansion_abcd_bra->conjugate();
  //expansion_abcd_bra->printIt();
  
  exatn::TensorExpansion energy_functional(*expansion_abcd_bra,*expansion_abcd_ket,*ham);
  energy_functional.printIt();
  energy_functional.rename("EnergyFunctional");

  exatn::TensorExpansion denominator(*expansion_abcd_bra,*expansion_abcd_ket);
  //denominator.printIt();
  denominator.rename("Denominator");

  exatn::TensorExpansion derivNum(energy_functional,"X",true);
  derivNum.rename("DerivNum");
  //derivNum.printIt();
  exatn::TensorExpansion derivDen(denominator,"X",true);
  derivDen.rename("DerivDen");
  //derivDen.printIt();
 
 // create accumulator tensor for the closed tensor expansion
       created = exatn::createTensorSync("ACNum",TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
       created = exatn::createTensorSync("ACDen",TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
       created = exatn::createTensorSync("ACDerivNum",TENS_ELEM_TYPE, exatn::TensorShape{nTO,nTO,nTO,nTO}); assert(created);
       created = exatn::createTensorSync("ACDerivDen",TENS_ELEM_TYPE, exatn::TensorShape{nTO,nTO,nTO,nTO}); assert(created);
  auto acNum = exatn::getTensor("ACNum");
  auto acDen = exatn::getTensor("ACDen");
  auto acDerivNum = exatn::getTensor("ACDerivNum");
  auto acDerivDen = exatn::getTensor("ACDerivDen");

  initialized = exatn::initTensorSync("ACNum",0.0); assert(initialized);
  initialized = exatn::initTensorSync("ACDen",0.0); assert(initialized);
  initialized = exatn::initTensorSync("ACDerivNum",0.0); assert(initialized);
  initialized = exatn::initTensorSync("ACDerivDen",0.0); assert(initialized);

  auto evaluated = false;
  evaluated = exatn::evaluateSync(energy_functional,acNum); assert(evaluated);
  evaluated = exatn::evaluateSync(denominator,acDen); assert(evaluated);
  // get value
  double e0 = 0.;
  double norm = 0.; 
  auto talsh_tensor = exatn::getLocalTensor("ACNum");
  const double * body_ptr;
  if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
    for ( int i = 0; i < talsh_tensor->getVolume(); i++){
      e0 = body_ptr[i];
    }
  }
  body_ptr = nullptr;
  talsh_tensor = exatn::getLocalTensor("ACDen");
  success = exatn::printTensorSync("ACNum"); assert(success);
  success = exatn::printTensorSync("ACDen"); assert(success);
  if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
    for ( int i = 0; i < talsh_tensor->getVolume(); i++){
      norm = body_ptr[i];
    }
  }
  body_ptr = nullptr;

    double dum = 0.;
    double dum1 = 0.;
    success = exatn::computeNorm2Sync("X", dum); assert(success);
    success = exatn::contractTensors("XtX() += X(d,c,b,a) * X(a,b,c,d)", 1.0); assert(success);
    success = exatn::computeNorm2Sync("XtX", dum1); assert(success);
    printf("Norm of X is %6.6lf\n", dum);
    printf("Norm of XtX is %6.6lf\n", dum1);
  printf("e0: %6.6lf, norm: %6.6lf, e0/norm: %6.6lf\n", e0, norm, e0/norm);
 
  // evaluate gradients 
  double delta = 0.;
  double maxAbsGrad = 0.;
  double epsilon = 0.5;
 
  int iter = 0;  
  do {
    // energy functional
    evaluated = exatn::evaluateSync(energy_functional,acNum); assert(evaluated);
    evaluated = exatn::evaluateSync(denominator,acDen); assert(evaluated);
    talsh_tensor = exatn::getLocalTensor("ACNum");
    if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
      for ( int i = 0; i < talsh_tensor->getVolume(); i++){
        e0 = body_ptr[i];
      }
    }
    body_ptr = nullptr;
    talsh_tensor = exatn::getLocalTensor("ACDen");
    if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
      for ( int i = 0; i < talsh_tensor->getVolume(); i++){
        norm = body_ptr[i];
      }
    }
    body_ptr = nullptr;
    std::cout << iter << std::endl;
    printf("e0: %6.6lf, norm: %6.6lf, e0/norm: %6.6lf\n", e0, norm, e0/norm);
    
    // derivatives
    evaluated = exatn::evaluateSync(derivNum, acDerivNum); assert(evaluated);
    evaluated = exatn::evaluateSync(derivDen, acDerivDen); assert(evaluated);
    // compute gradient
    initialized = exatn::initTensor("Grad", 0.0); assert(initialized);
    success = exatn::addTensors("Grad(p,q,r,s) += ACDerivNum(p,q,r,s)", norm); assert(success);
    success = exatn::addTensors("Grad(p,q,r,s) += ACDerivDen(p,q,r,s)",-e0); assert(success);
    success = exatn::scaleTensor("Grad", 1.0/(norm * norm)); assert(success);
    // update wavefunction
    initialized = exatn::initTensor("XOld", 0.0); assert(initialized);
    success = exatn::addTensors("XOld(p,q,r,s) += X(p,q,r,s)",1.0); assert(success);
    initialized = exatn::initTensor("X", 0.0); assert(initialized);
    success = exatn::addTensors("X(p,q,r,s) += XOld(p,q,r,s)",1.0); assert(success);
    success = exatn::addTensors("X(p,q,r,s) += Grad(p,q,r,s)",-epsilon); assert(success);
    
    success = exatn::computeMaxAbsSync("Grad", maxAbsGrad); assert(success);

   std::cout << std::setw(1) << std::right << "  X:" <<  std::setw(4) << std::right 
   << iter << std::setw(15) << std::fixed << norm << std::setw(15) << std::fixed << e0/norm << std::setw(15) << std::fixed << maxAbsGrad << std::endl;  
    iter++;
  } while(fabs(maxAbsGrad) > 1e-4 );   

  // destroying tensors 
  auto destroyed = false;
  //destroyed = exatn::destroyTensor("AC0"); assert(destroyed);
  destroyed = exatn::destroyTensor("H1"); assert(destroyed);
  destroyed = exatn::destroyTensor("H2"); assert(destroyed);
  destroyed = exatn::destroyTensor("Q"); assert(destroyed);
  destroyed = exatn::destroyTensor("X"); assert(destroyed);
  //destroyed = exatn::destroyTensor("XOld"); assert(destroyed);

 }

 
  exatn::finalize();
 
  return 0;

}
