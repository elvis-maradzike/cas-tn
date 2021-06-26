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
  // wavefunction ansatz parameters
  std::size_t np = nto, nq = nto, nr = nto, ns = nto;
  
  if ( argc != 4){ 
    printf("\n");
    printf("    usage: ./h4.x ni nj nk \n");
    printf("\n");
    exit(0);
  }

  // bond dimensions
  char * char1; 
  char * char2; 
  char * char3; 
  long long1 = strtol(argv[1], &char1, 10);
  long long2 = strtol(argv[2], &char2, 10);
  long long3 = strtol(argv[3], &char3, 10);
  std::size_t ni = long1;
  std::size_t nj = long2;
  std::size_t nk = long3;

  
  auto success = false, created = false, initialized = false, appended = false;
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;

  exatn::initialize();
  {
  // declare tensor network A-B-C-D
  auto a = std::make_shared<exatn::Tensor>("A", exatn::TensorShape{np,ni});
  auto b = std::make_shared<exatn::Tensor>("B", exatn::TensorShape{ni,nq,nj});
  auto c = std::make_shared<exatn::Tensor>("C", exatn::TensorShape{nj,nr,nk});
  auto d = std::make_shared<exatn::Tensor>("D", exatn::TensorShape{nk,ns});
  auto tn_abcd = std::make_shared<exatn::Tensor>("TN_ABCD", exatn::TensorShape{np,nq,nr,ns});
  
  auto network_abcd = exatn::makeSharedTensorNetwork(
                 "NetworkABCD", //tensor network name
                 "TN_ABCD(p,q,r,s)+=A(p,i)*B(i,q,j)*C(j,r,k)*D(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {"TN_ABCD",tn_abcd}, 
                  {"A",a},
                  {"B",b},
                  {"C",c},
                  {"D",d}
                 }
                );

  // create constituent tensors
  created = exatn::createTensor(tn_abcd, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(a, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d, TENS_ELEM_TYPE); assert(created);

  std::vector<std::shared_ptr<exatn::Tensor> > hamiltonian;

  // declare object from Simulation class
  Simulation myObject(nao, nap, nco, nto, ntp);
  myObject.resetWaveFunctionAnsatz(network_abcd);
  myObject.resetHamiltonian(hamiltonian);
  myObject.optimize(1,1e-5);

  }

  exatn::finalize();
 
  return 0;

}
