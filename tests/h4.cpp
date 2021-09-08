#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../src/simulation.hpp"
#include "cblas.h"
#include "lapacke.h"

using namespace std::chrono;
using namespace castn;

/* marks as optimizable all tensors that comprise a specified tensor network expansion */
void markOptimizableTensors(std::shared_ptr<exatn::TensorExpansion> ansatz){
  for (auto component = ansatz->begin(); component != ansatz->end(); ++component){
    component->network->markOptimizableAllTensors();
  }
}

/* marks as optimizable all tensors that comprise a specified tensor network */
void markOptimizableTensors(std::shared_ptr<exatn::TensorNetwork> ansatz){
  ansatz->markOptimizableAllTensors();
}

int main(int argc, char** argv){

  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  
  // active orbitals, active particles, core orbitals, total particles
  std::size_t nao = 8, nap = 4, nco = 0, nto = 8, ntp = 4;
  
  // wavefunction ansatz parameters
  const exatn::DimExtent np = nto, nq = nto, nr = nto, ns = nto;

  exatn::initialize();
  {
  // declare output tensor
  auto output_tensor = std::make_shared<exatn::Tensor>("Z0", exatn::TensorShape{np,nq,nr,ns});

  // builder for mps
  auto builder = exatn::getTensorNetworkBuilder("MPS");
  auto success = builder->setParameter("max_bond_dim", 64); assert(success);
  auto ansatz_net = exatn::makeSharedTensorNetwork("KetAnsatz", output_tensor, *builder);
  ansatz_net->printIt();

  //Allocate/initialize tensors in the tensor network ansatz:
  for(auto tens_conn = ansatz_net->begin(); tens_conn != ansatz_net->end(); ++tens_conn){
    if(tens_conn->first != 0){ 
      success = exatn::createTensor(tens_conn->second.getTensor(),TENS_ELEM_TYPE); assert(success);
      success = exatn::initTensorRnd(tens_conn->second.getName()); assert(success);
    }
  }

  // declare tensors for one and two-electron integrals
  auto h1 = exatn::makeSharedTensor("H1", exatn::TensorShape{nto,nto});
  auto h2 = exatn::makeSharedTensor("H2", exatn::TensorShape{nto,nto,nto,nto});
 
  // create tensors for one and two-electron integrals 
  auto created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);

  // initializing tensors for one and two-electron integrals
  auto initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

 //create hamiltonian operator from one and two-electron integrals
  std::vector<std::shared_ptr<exatn::Tensor> > hamiltonian;
  hamiltonian.push_back(h2);
  hamiltonian.push_back(h1);
 
  std::shared_ptr<exatn::TensorExpansion> ansatz;
  ansatz = std::make_shared<exatn::TensorExpansion>();
  ansatz->appendComponent(ansatz_net,{1.0,0.0});

  double convergence_thresh = 1e-5;
  // declare object from Simulation class
  Simulation optimizer(nao, nap, nco, nto, ntp);
  optimizer.resetWaveFunctionAnsatz(ansatz);
  optimizer.resetHamiltonian(hamiltonian);
  optimizer.optimize(1,convergence_thresh);

  }

  exatn::finalize();
 
  return 0;

}
