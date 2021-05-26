#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../src/simulation.hpp"
using namespace castn;

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
  auto tn_abcd = std::make_shared<exatn::Tensor>("TN_ABCD", exatn::TensorShape{nTO,nTO,nTO,nTO});
  

  // create constituent tensors
  created = exatn::createTensor(x, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor("Grad", TENS_ELEM_TYPE,x->getShape()); assert(created);

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

  // vector: stores Hamiltonian tensors
  std::vector<std::shared_ptr<exatn::Tensor> > hamiltonian;

  // declare object from Simulation class
  Simulation myObject(nAO, nAP, nCO, nTO, nTP);
  myObject.resetWaveFunctionAnsatz(network_abcd);
  myObject.resetHamiltonian(hamiltonian);
  myObject.optimize(1,1e-5);
  }
 
  exatn::finalize();
 
  return 0;

}
