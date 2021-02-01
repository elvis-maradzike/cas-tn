#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../src/simulation.hpp"

using namespace std::chrono;
using namespace castn;

int main(int argc, char** argv){

  std::size_t num_orbitals = 8;
  std::size_t num_particles = 4;
  std::size_t num_core_orbitals = 0;
  std::size_t total_orbitals = 8;
  std::size_t total_particles = 4;
  std::size_t np = total_orbitals, nq = total_orbitals;
  std::size_t nr = total_orbitals, ns = total_orbitals;
  std::size_t ni = total_orbitals, nk = total_orbitals;
  std::size_t nj = total_orbitals * total_orbitals;
  
  auto success = false, created = false, initialized = false, appended = false;
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;

  exatn::initialize();
  {
  // declare tensor network
  auto a = std::make_shared<exatn::Tensor>("A", exatn::TensorShape{np,ni});
  auto b = std::make_shared<exatn::Tensor>("B", exatn::TensorShape{ni,nq,nj});
  auto c = std::make_shared<exatn::Tensor>("C", exatn::TensorShape{nj,nr,nk});
  auto d = std::make_shared<exatn::Tensor>("D", exatn::TensorShape{nk,ns});
  auto abcd = std::make_shared<exatn::Tensor>("ABCD", exatn::TensorShape{np,nq,nr,ns});
  
  auto network_abcd = exatn::makeSharedTensorNetwork(
                 "NetworkABCD", //tensor network name
                 "ABCD(p,q,r,s)+=A(p,i)*B(i,q,j)*C(j,r,k)*D(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"ABCD",abcd},
                  {"A",a},
                  {"B",b},
                  {"C",c},
                  {"D",d}
                 }
                );

  // create constituent tensors
  created = exatn::createTensor(abcd, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(a, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d, TENS_ELEM_TYPE); assert(created);

  std::vector<std::shared_ptr<exatn::Tensor> > hamiltonian;

  // declare object from Simulation class
  Simulation myObject(num_orbitals, num_particles, num_core_orbitals, total_orbitals, total_particles);
  myObject.resetWaveFunctionAnsatz(network_abcd);
  myObject.resetHamiltonian(hamiltonian);
  myObject.optimize(1,1e-5);

  }

  exatn::finalize();
 
  return 0;

}
