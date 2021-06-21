#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../src/simulation.hpp"

using namespace std::chrono;
using namespace castn;

int main(int argc, char** argv){

  // active orbitals
  std::size_t nao = 8;
  // active particles
  std::size_t nap = 4;
  // core orbitals
  std::size_t nco = 0;
  // total orbitals
  std::size_t nto = 8;
  // total particles
  std::size_t ntp = 4;
  
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;

  exatn::initialize();
  {
  // declare tensor network
  auto abcd = std::make_shared<exatn::Tensor>("ABCD", exatn::TensorShape{nto,nto,nto,nto});
  auto tn_abcd = std::make_shared<exatn::Tensor>("TN_ABCD", exatn::TensorShape{nto,nto,nto,nto});
  
  auto network_abcd = exatn::makeSharedTensorNetwork(
                 "NetworkABCD", //tensor network name
                 "TN_ABCD(p,q,r,s)+=ABCD(p,q,r,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"TN_ABCD",tn_abcd},
                  {"ABCD",abcd}
                 }
                );

  // create constituent tensors
  auto created = exatn::createTensor(tn_abcd, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(abcd, TENS_ELEM_TYPE); assert(created);

  std::vector<std::shared_ptr<exatn::Tensor> > hamiltonian;

  Simulation myObject(nao, nap, nco, nto, ntp);
  myObject.resetWaveFunctionAnsatz(network_abcd);
  myObject.resetHamiltonian(hamiltonian);
  myObject.optimize(1,1e-5);
  }

  exatn::finalize();
 
  return 0;

}
