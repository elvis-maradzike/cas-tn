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
  auto abcd = std::make_shared<exatn::Tensor>("ABCD", exatn::TensorShape{nTO,nTO,nTO,nTO});
  auto tn_abcd = std::make_shared<exatn::Tensor>("TN_ABCD", exatn::TensorShape{nTO,nTO,nTO,nTO});
  
  auto network_abcd = exatn::makeSharedTensorNetwork(
                 "NetworkABCD", //tensor network name
                 "TN_ABCD(p,q,r,s)+=ABCD(p,q,r,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"TN_ABCD",tn_abcd},{"ABCD",abcd}
                 }
                );

  // create constituent tensors
  created = exatn::createTensor(abcd, TENS_ELEM_TYPE); assert(created);
  initialized = exatn::initTensorRnd("ABCD"); assert(initialized);

  std::shared_ptr<exatn::TensorExpansion> expansion_abcd;
  expansion_abcd = std::make_shared<exatn::TensorExpansion>(); 
  expansion_abcd->appendComponent(network_abcd,{1.0,0.0});

  // hamiltonian
  auto h1 = std::make_shared<exatn::Tensor>("H1",exatn::TensorShape{nTO, nTO});
  auto h2 = std::make_shared<exatn::Tensor>("H2",exatn::TensorShape{nTO, nTO, nTO, nTO});

  created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);

  initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

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

  // mark only network_abcd as optimizable
  for (auto component = expansion_abcd->begin(); component != expansion_abcd->end(); ++component){
    component->network_->markOptimizableAllTensors();
  }

  //create tensors: ordering projectors
  created = exatn::createTensor("Q", TENS_ELEM_TYPE, exatn::TensorShape{nTO, nTO, nTO, nTO}); assert(created);
  initialized = exatn::initTensorData("Q", tmpData); assert(initialized);

  std::cout << "Appending Ordering Projectors..." << std::endl;
  unsigned int tensorCounter = 2;
  for (auto iter = expansion_abcd->begin(); iter != expansion_abcd->end(); ++iter){
    iter->network_;
    iter->coefficient_;
    auto & network = *(iter->network_);
    appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{0,1}); assert(appended);
    tensorCounter++;
    appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{2,3}); assert(appended);
    tensorCounter++;
    appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{1,2}); assert(appended);
  } 
  // checking expansion
  expansion_abcd->printIt();

  // hamiltonian
  auto ham = exatn::makeSharedTensorOperator("Hamiltonian");
  // two-body part
  for ( unsigned int ketIndex1 = 0; ketIndex1 < nTP; ketIndex1++){
    for ( unsigned int ketIndex2 = ketIndex1+1; ketIndex2 < nTP; ketIndex2++){
      for ( unsigned int braIndex1 = 0; braIndex1 < nTP; braIndex1++){
        for ( unsigned int braIndex2 = braIndex1+1; braIndex2 < nTP; braIndex2++){
          auto appended = ham->appendComponent(h2,{{ketIndex1,0},{ketIndex2,1}},{{braIndex1,2},{braIndex2,3}},{0.5,0.0}); assert(appended);
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

  // print ordering projector before optimization
  success = exatn::printTensorSync("Q"); assert(success);

  // call ExaTN optimizer
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(ham,expansion_abcd,0.9);
  optimizer.resetLearningRate(0.5);
  bool converged = optimizer.optimize();
  bool success = exatn::sync(); assert(success);
  if(converged){
    std::cout << "Optimization succeeded!" << std::endl;
    // print ordering projector after optimization
    success = exatn::printTensorSync("Q"); assert(success);
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }

  // destroying tensors 
  auto destroyed = false;
  destroyed = exatn::destroyTensor("H1"); assert(destroyed);
  destroyed = exatn::destroyTensor("H2"); assert(destroyed);
  destroyed = exatn::destroyTensor("Q"); assert(destroyed);
  destroyed = exatn::destroyTensor("ABCD"); assert(destroyed);

 }

 
  exatn::finalize();
 
  return 0;

}
