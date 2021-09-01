#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../src/simulation.hpp"
#include "mkl_lapacke.h"

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

/* applies ordering projectors to tensor networks that comprise a specified tensor expansion */
void appendOrderingProjectors(int num_particles, int total_orbitals, std::shared_ptr<exatn::TensorExpansion> ansatz){
  auto appended = false;
  for (auto iter = ansatz->begin(); iter != ansatz->end(); ++iter){
    auto & network = *(iter->network);
    unsigned int projectorTensorId = 1 + network.getNumTensors();
    // (applies) layer 1
    for ( unsigned int i = 0; i < num_particles-1; i=i+2){
      appended = network.appendTensorGate(projectorTensorId, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      projectorTensorId++;
    }
    // (applies) layer 2
    for ( unsigned int i = 1; i < num_particles-1; i=i+2){
      appended = network.appendTensorGate(projectorTensorId, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      projectorTensorId++;
    }
  }
}

/* applies ordering projectors to specified tensor network */
void appendOrderingProjectors(int num_particles, int total_orbitals, std::shared_ptr<exatn::TensorNetwork> ansatz){
  auto appended = false;
  unsigned int projectorTensorId = 1 + ansatz->getNumTensors();
  // (applies) layer 1
  for ( unsigned int i = 0; i < num_particles-1; i=i+2){
    appended = ansatz->appendTensorGate(projectorTensorId, exatn::getTensor("Q"),{i,i+1}); assert(appended);
    projectorTensorId++;
  }
  // (applies) layer 2
  for ( unsigned int i = 1; i < num_particles-1; i=i+2){
    appended = ansatz->appendTensorGate(projectorTensorId, exatn::getTensor("Q"),{i,i+1}); assert(appended);
    projectorTensorId++;
  }
}

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
  char * char1, * char2, * char3; 
  long long1 = strtol(argv[1], &char1, 10);
  long long2 = strtol(argv[2], &char2, 10);
  long long3 = strtol(argv[3], &char3, 10);

  std::size_t ni = long1;
  std::size_t nj = long2;
  std::size_t nk = long3;

  // tensor shapes  
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  const auto TENS_SHAPE_A = exatn::TensorShape{np,ni};
  const auto TENS_SHAPE_B = exatn::TensorShape{ni,nq,nj};
  const auto TENS_SHAPE_C = exatn::TensorShape{nj,nr,nk};
  const auto TENS_SHAPE_D = exatn::TensorShape{nk,ns};
  const auto TENS_SHAPE_ABCD = exatn::TensorShape{np,nq,nr,ns};
  
  // array of elements to initialize tensors that comprise the ordering projector
  std::vector<double> tmpData(nto*nto*nto*nto,0.0);
  for ( unsigned int i = 0; i < nto; i++){
    for ( unsigned int j = 0; j < nto; j++){
      for ( unsigned int k = 0; k < nto; k++){
        for ( unsigned int l = 0; l < nto; l++){
          if ( i < j){
          tmpData[i*nto*nto*nto + j*nto*nto + k*nto + l] = double((i==k)*(j==l));
          }else{
          tmpData[i*nto*nto*nto + j*nto*nto + k*nto + l] = 0.0;
          }
        }
      }
    }
  }

  exatn::initialize();
  {

  // number of tensor networks used in computing initial guess
  unsigned int N = 8;
  
  // declare tensor network A-B-C-D
  auto a = std::make_shared<exatn::Tensor>("A", TENS_SHAPE_A );
  auto b = std::make_shared<exatn::Tensor>("B", TENS_SHAPE_B );
  auto c = std::make_shared<exatn::Tensor>("C", TENS_SHAPE_C );
  auto d = std::make_shared<exatn::Tensor>("D", TENS_SHAPE_D );
  auto abcd = std::make_shared<exatn::Tensor>("ABCD", TENS_SHAPE_ABCD );
  
  // tensors used in generating guess
  auto a1 = std::make_shared<exatn::Tensor>("A1", TENS_SHAPE_A);
  auto b1 = std::make_shared<exatn::Tensor>("B1", TENS_SHAPE_B);
  auto c1 = std::make_shared<exatn::Tensor>("C1", TENS_SHAPE_C);
  auto d1 = std::make_shared<exatn::Tensor>("D1", TENS_SHAPE_D);
 
  auto a2 = std::make_shared<exatn::Tensor>("A2", TENS_SHAPE_A);
  auto b2 = std::make_shared<exatn::Tensor>("B2", TENS_SHAPE_B);
  auto c2 = std::make_shared<exatn::Tensor>("C2", TENS_SHAPE_C);
  auto d2 = std::make_shared<exatn::Tensor>("D2", TENS_SHAPE_D);
 
  auto a3 = std::make_shared<exatn::Tensor>("A3", TENS_SHAPE_A);
  auto b3 = std::make_shared<exatn::Tensor>("B3", TENS_SHAPE_B);
  auto c3 = std::make_shared<exatn::Tensor>("C3", TENS_SHAPE_C);
  auto d3 = std::make_shared<exatn::Tensor>("D3", TENS_SHAPE_D);
  
  auto a4 = std::make_shared<exatn::Tensor>("A4", TENS_SHAPE_A);
  auto b4 = std::make_shared<exatn::Tensor>("B4", TENS_SHAPE_B);
  auto c4 = std::make_shared<exatn::Tensor>("C4", TENS_SHAPE_C);
  auto d4 = std::make_shared<exatn::Tensor>("D4", TENS_SHAPE_D);
 
  auto a5 = std::make_shared<exatn::Tensor>("A5", TENS_SHAPE_A);
  auto b5 = std::make_shared<exatn::Tensor>("B5", TENS_SHAPE_B);
  auto c5 = std::make_shared<exatn::Tensor>("C5", TENS_SHAPE_C);
  auto d5 = std::make_shared<exatn::Tensor>("D5", TENS_SHAPE_D);
  
  auto a6 = std::make_shared<exatn::Tensor>("A6", TENS_SHAPE_A);
  auto b6 = std::make_shared<exatn::Tensor>("B6", TENS_SHAPE_B);
  auto c6 = std::make_shared<exatn::Tensor>("C6", TENS_SHAPE_C);
  auto d6 = std::make_shared<exatn::Tensor>("D6", TENS_SHAPE_D);
 
  auto a7 = std::make_shared<exatn::Tensor>("A7", TENS_SHAPE_A);
  auto b7 = std::make_shared<exatn::Tensor>("B7", TENS_SHAPE_B);
  auto c7 = std::make_shared<exatn::Tensor>("C7", TENS_SHAPE_C);
  auto d7 = std::make_shared<exatn::Tensor>("D7", TENS_SHAPE_D);
 
  auto a8 = std::make_shared<exatn::Tensor>("A8", TENS_SHAPE_A);
  auto b8 = std::make_shared<exatn::Tensor>("B8", TENS_SHAPE_B);
  auto c8 = std::make_shared<exatn::Tensor>("C8", TENS_SHAPE_C);
  auto d8 = std::make_shared<exatn::Tensor>("D8", TENS_SHAPE_D);
  
  // one and two-electron integrals
  auto h1 = std::make_shared<exatn::Tensor>("H1", exatn::TensorShape{nto,nto});
  auto h2 = std::make_shared<exatn::Tensor>("H2", exatn::TensorShape{nto,nto,nto,nto});
  
  // accumulator tensors for tensor networks used in building guess
  auto ac1 = std::make_shared<exatn::Tensor>("AC1", TENS_SHAPE_ABCD);
  auto ac2 = std::make_shared<exatn::Tensor>("AC2", TENS_SHAPE_ABCD);
  auto ac3 = std::make_shared<exatn::Tensor>("AC3", TENS_SHAPE_ABCD);
  auto ac4 = std::make_shared<exatn::Tensor>("AC4", TENS_SHAPE_ABCD);
  auto ac5 = std::make_shared<exatn::Tensor>("AC5", TENS_SHAPE_ABCD);
  auto ac6 = std::make_shared<exatn::Tensor>("AC6", TENS_SHAPE_ABCD);
  auto ac7 = std::make_shared<exatn::Tensor>("AC7", TENS_SHAPE_ABCD);
  auto ac8 = std::make_shared<exatn::Tensor>("AC8", TENS_SHAPE_ABCD);

  // accumulator for approximant
  auto ac_approximant = std::make_shared<exatn::Tensor>("_ac_approximant", TENS_SHAPE_ABCD);

  // tensor networks 
  auto network_1 = exatn::makeSharedTensorNetwork(
                 "Network_1", //tensor network name
                 "AC1(p,q,r,s)+=A1(p,i)*B1(i,q,j)*C1(j,r,k)*D1(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {ac1->getName(),ac1},
                  {a1->getName(),a1},
                  {b1->getName(),b1},
                  {c1->getName(),c1},
                  {d1->getName(),d1}
                 }
                );
  auto network_2 = exatn::makeSharedTensorNetwork(
                 "Network_2", //tensor network name
                 "AC2(p,q,r,s)+=A2(p,i)*B2(i,q,j)*C2(j,r,k)*D2(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {ac2->getName(),ac2},
                  {a2->getName(),a2},
                  {b2->getName(),b2},
                  {c2->getName(),c2},
                  {d2->getName(),d2}
                 }
                );
  auto network_3 = exatn::makeSharedTensorNetwork(
                 "Network_3", //tensor network name
                 "AC3(p,q,r,s)+=A3(p,i)*B3(i,q,j)*C3(j,r,k)*D3(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {ac3->getName(),ac3},
                  {a3->getName(),a3},
                  {b3->getName(),b3},
                  {c3->getName(),c3},
                  {d3->getName(),d3}
                 }
                );
  auto network_4 = exatn::makeSharedTensorNetwork(
                 "Network_4", //tensor network name
                 "AC4(p,q,r,s)+=A4(p,i)*B4(i,q,j)*C4(j,r,k)*D4(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {ac4->getName(),ac4},
                  {a4->getName(),a4},
                  {b4->getName(),b4},
                  {c4->getName(),c4},
                  {d4->getName(),d4}
                 }
                );
  auto network_5 = exatn::makeSharedTensorNetwork(
                 "Network_5", //tensor network name
                 "AC5(p,q,r,s)+=A5(p,i)*B5(i,q,j)*C5(j,r,k)*D5(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {ac5->getName(),ac5},
                  {a5->getName(),a5},
                  {b5->getName(),b5},
                  {c5->getName(),c5},
                  {d5->getName(),d5}
                 }
                );
  auto network_6 = exatn::makeSharedTensorNetwork(
                 "Network_6", //tensor network name
                 "AC6(p,q,r,s)+=A6(p,i)*B6(i,q,j)*C6(j,r,k)*D6(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {ac6->getName(),ac6},
                  {a6->getName(),a6},
                  {b6->getName(),b6},
                  {c6->getName(),c6},
                  {d6->getName(),d6}
                 }
                );
  auto network_7 = exatn::makeSharedTensorNetwork(
                 "Network_7", //tensor network name
                 "AC7(p,q,r,s)+=A7(p,i)*B7(i,q,j)*C7(j,r,k)*D7(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {ac7->getName(),ac7},
                  {a7->getName(),a7},
                  {b7->getName(),b7},
                  {c7->getName(),c7},
                  {d7->getName(),d7}
                 }
                );
  auto network_8 = exatn::makeSharedTensorNetwork(
                 "Network_8", //tensor network name
                 "AC8(p,q,r,s)+=A8(p,i)*B8(i,q,j)*C8(j,r,k)*D8(k,s)",
                 std::map<std::string,std::shared_ptr<exatn::Tensor>>{
                  {ac8->getName(),ac8},
                  {a8->getName(),a8},
                  {b8->getName(),b8},
                  {c8->getName(),c8},
                  {d8->getName(),d8}
                 }
                );
  
 // creating tensors
  auto created = exatn::createTensor(a1, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b1, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c1, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d1, TENS_ELEM_TYPE); assert(created);
 
  created = exatn::createTensor(a2, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b2, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c2, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d2, TENS_ELEM_TYPE); assert(created);
 
  created = exatn::createTensor(a3, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b3, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c3, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d3, TENS_ELEM_TYPE); assert(created);
 
  created = exatn::createTensor(a4, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b4, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c4, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d4, TENS_ELEM_TYPE); assert(created);
 
  created = exatn::createTensor(a5, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b5, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c5, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d5, TENS_ELEM_TYPE); assert(created);

  created = exatn::createTensor(a6, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b6, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c6, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d6, TENS_ELEM_TYPE); assert(created);

  created = exatn::createTensor(a7, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b7, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c7, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d7, TENS_ELEM_TYPE); assert(created);

  created = exatn::createTensor(a8, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b8, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c8, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d8, TENS_ELEM_TYPE); assert(created);
 
  created = exatn::createTensor(a, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(b, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(c, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(d, TENS_ELEM_TYPE); assert(created);

  created = exatn::createTensor(abcd, TENS_ELEM_TYPE); assert(created);
  
  created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);

  created = exatn::createTensor(ac1, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(ac2, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(ac3, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(ac4, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(ac5, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(ac6, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(ac7, TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensor(ac8, TENS_ELEM_TYPE); assert(created);

  created = exatn::createTensor(ac_approximant, TENS_ELEM_TYPE); assert(created);

  // initializing tensors
  auto initialized = exatn::initTensorRnd("A"); assert(initialized);
  initialized = exatn::initTensorRnd("B"); assert(initialized);
  initialized = exatn::initTensorRnd("C"); assert(initialized);
  initialized = exatn::initTensorRnd("D"); assert(initialized);
  
  initialized = exatn::initTensorRnd("A1"); assert(initialized);
  initialized = exatn::initTensorRnd("B1"); assert(initialized);
  initialized = exatn::initTensorRnd("C1"); assert(initialized);
  initialized = exatn::initTensorRnd("D1"); assert(initialized);
  initialized = exatn::initTensorRnd("A2"); assert(initialized);
  initialized = exatn::initTensorRnd("B2"); assert(initialized);
  initialized = exatn::initTensorRnd("C2"); assert(initialized);
  initialized = exatn::initTensorRnd("D2"); assert(initialized);
  initialized = exatn::initTensorRnd("A3"); assert(initialized);
  initialized = exatn::initTensorRnd("B3"); assert(initialized);
  initialized = exatn::initTensorRnd("C3"); assert(initialized);
  initialized = exatn::initTensorRnd("D3"); assert(initialized);
  initialized = exatn::initTensorRnd("A4"); assert(initialized);
  initialized = exatn::initTensorRnd("B4"); assert(initialized);
  initialized = exatn::initTensorRnd("C4"); assert(initialized);
  initialized = exatn::initTensorRnd("D4"); assert(initialized);
  initialized = exatn::initTensorRnd("A5"); assert(initialized);
  initialized = exatn::initTensorRnd("B5"); assert(initialized);
  initialized = exatn::initTensorRnd("C5"); assert(initialized);
  initialized = exatn::initTensorRnd("D5"); assert(initialized);
  initialized = exatn::initTensorRnd("A6"); assert(initialized);
  initialized = exatn::initTensorRnd("B6"); assert(initialized);
  initialized = exatn::initTensorRnd("C6"); assert(initialized);
  initialized = exatn::initTensorRnd("D6"); assert(initialized);
  initialized = exatn::initTensorRnd("A7"); assert(initialized);
  initialized = exatn::initTensorRnd("B7"); assert(initialized);
  initialized = exatn::initTensorRnd("C7"); assert(initialized);
  initialized = exatn::initTensorRnd("D7"); assert(initialized);
  initialized = exatn::initTensorRnd("A8"); assert(initialized);
  initialized = exatn::initTensorRnd("B8"); assert(initialized);
  initialized = exatn::initTensorRnd("C8"); assert(initialized);
  initialized = exatn::initTensorRnd("D8"); assert(initialized);

  initialized = exatn::initTensorRnd("ABCD"); assert(initialized);

  initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

  //creating and initializing tensor for ordering projectors
  created = exatn::createTensor("Q", TENS_ELEM_TYPE, exatn::TensorShape{nto,nto,nto,nto}); assert(created);
  initialized = exatn::initTensorData("Q", tmpData); assert(initialized);

  
  // mark as optimizable tensors in tensor networks
  markOptimizableTensors(network_1);
  markOptimizableTensors(network_2);
  markOptimizableTensors(network_3);
  markOptimizableTensors(network_4);
  markOptimizableTensors(network_5);
  markOptimizableTensors(network_6);
  markOptimizableTensors(network_7);
  markOptimizableTensors(network_8);

  // append ordering projectors to all tensor networks
  appendOrderingProjectors(ntp, nto, network_1);
  appendOrderingProjectors(ntp, nto, network_2);
  appendOrderingProjectors(ntp, nto, network_3);
  appendOrderingProjectors(ntp, nto, network_4);
  appendOrderingProjectors(ntp, nto, network_5);
  appendOrderingProjectors(ntp, nto, network_6);
  appendOrderingProjectors(ntp, nto, network_7);
  appendOrderingProjectors(ntp, nto, network_8);

  // declare and make tensor expansions, ket
  std::shared_ptr<exatn::TensorExpansion> ket_1;
  std::shared_ptr<exatn::TensorExpansion> ket_2;
  std::shared_ptr<exatn::TensorExpansion> ket_3;
  std::shared_ptr<exatn::TensorExpansion> ket_4;
  std::shared_ptr<exatn::TensorExpansion> ket_5;
  std::shared_ptr<exatn::TensorExpansion> ket_6;
  std::shared_ptr<exatn::TensorExpansion> ket_7;
  std::shared_ptr<exatn::TensorExpansion> ket_8;
  ket_1 = std::make_shared<exatn::TensorExpansion>();
  ket_2 = std::make_shared<exatn::TensorExpansion>();
  ket_3 = std::make_shared<exatn::TensorExpansion>();
  ket_4 = std::make_shared<exatn::TensorExpansion>();
  ket_5 = std::make_shared<exatn::TensorExpansion>();
  ket_6 = std::make_shared<exatn::TensorExpansion>();
  ket_7 = std::make_shared<exatn::TensorExpansion>();
  ket_8 = std::make_shared<exatn::TensorExpansion>();
  
  ket_1->appendComponent(network_1,{1.0,0.0});
  auto success = exatn::balanceNormalizeNorm2Sync(*ket_1,1.0,1.0,true); assert(success);
  ket_2->appendComponent(network_2,{1.0,0.0});
  success = exatn::balanceNormalizeNorm2Sync(*ket_2,1.0,1.0,true); assert(success);
  ket_3->appendComponent(network_3,{1.0,0.0});
  success = exatn::balanceNormalizeNorm2Sync(*ket_3,1.0,1.0,true); assert(success);
  ket_4->appendComponent(network_4,{1.0,0.0});
  success = exatn::balanceNormalizeNorm2Sync(*ket_4,1.0,1.0,true); assert(success);
  ket_5->appendComponent(network_5,{1.0,0.0});
  success = exatn::balanceNormalizeNorm2Sync(*ket_5,1.0,1.0,true); assert(success);
  ket_6->appendComponent(network_6,{1.0,0.0});
  success = exatn::balanceNormalizeNorm2Sync(*ket_6,1.0,1.0,true); assert(success);
  ket_7->appendComponent(network_7,{1.0,0.0});
  success = exatn::balanceNormalizeNorm2Sync(*ket_7,1.0,1.0,true); assert(success);
  ket_8->appendComponent(network_8,{1.0,0.0});
  success = exatn::balanceNormalizeNorm2Sync(*ket_8,1.0,1.0,true); assert(success);

  // make bra form of ket tensor expansions
  std::shared_ptr<exatn::TensorExpansion> bra_1;
  std::shared_ptr<exatn::TensorExpansion> bra_2;
  std::shared_ptr<exatn::TensorExpansion> bra_3;
  std::shared_ptr<exatn::TensorExpansion> bra_4;
  std::shared_ptr<exatn::TensorExpansion> bra_5;
  std::shared_ptr<exatn::TensorExpansion> bra_6;
  std::shared_ptr<exatn::TensorExpansion> bra_7;
  std::shared_ptr<exatn::TensorExpansion> bra_8;
  bra_1 = std::make_shared<exatn::TensorExpansion>(*ket_1);
  bra_2 = std::make_shared<exatn::TensorExpansion>(*ket_2);
  bra_3 = std::make_shared<exatn::TensorExpansion>(*ket_3);
  bra_4 = std::make_shared<exatn::TensorExpansion>(*ket_4);
  bra_5 = std::make_shared<exatn::TensorExpansion>(*ket_5);
  bra_6 = std::make_shared<exatn::TensorExpansion>(*ket_6);
  bra_7 = std::make_shared<exatn::TensorExpansion>(*ket_7);
  bra_8 = std::make_shared<exatn::TensorExpansion>(*ket_8);
  bra_1->conjugate();
  bra_2->conjugate();
  bra_3->conjugate();
  bra_4->conjugate();
  bra_5->conjugate();
  bra_6->conjugate();
  bra_7->conjugate();
  bra_8->conjugate();


  // make lists of tensor expansions above
  std::vector<std::shared_ptr<exatn::TensorExpansion>> listOfKetTensorExpansions;
  std::vector<std::shared_ptr<exatn::TensorExpansion>> listOfBraTensorExpansions;

  listOfKetTensorExpansions.push_back(ket_1);
  listOfKetTensorExpansions.push_back(ket_2);
  listOfKetTensorExpansions.push_back(ket_3);
  listOfKetTensorExpansions.push_back(ket_4);
  listOfKetTensorExpansions.push_back(ket_5);
  listOfKetTensorExpansions.push_back(ket_6);
  listOfKetTensorExpansions.push_back(ket_7);
  listOfKetTensorExpansions.push_back(ket_8);

  listOfBraTensorExpansions.push_back(bra_1);
  listOfBraTensorExpansions.push_back(bra_2);
  listOfBraTensorExpansions.push_back(bra_3);
  listOfBraTensorExpansions.push_back(bra_4);
  listOfBraTensorExpansions.push_back(bra_5);
  listOfBraTensorExpansions.push_back(bra_6);
  listOfBraTensorExpansions.push_back(bra_7);
  listOfBraTensorExpansions.push_back(bra_8);

  //create hamiltonian operator from one and two-electron integrals
  auto ham = exatn::makeSharedTensorOperator("Hamiltonian");
  auto appended = false;

  //(anti)symmetrization 
  success = ham->appendSymmetrizeComponent(h2,{0,1},{2,3}, ntp, ntp, {1.0,0.0},true); assert(success);
  success = ham->appendSymmetrizeComponent(h1,{0},{1}, ntp, ntp, {1.0,0.0},true); assert(success);

  // Hamiltonian matrix in basis of tensor network expansions above
  double H[N*N]; 
  //overlap matrix
  double S[N*N];
  
  /* H_ij = < ket_i| h | ket_j> and S_ij = < ket_i| ket_j> */
  for ( unsigned int i = 0; i < listOfBraTensorExpansions.size(); i++){
    for ( unsigned int j = 0; j < listOfKetTensorExpansions.size(); j++){

      exatn::TensorExpansion hamiltonianMatrixElement(*listOfBraTensorExpansions[i], *listOfKetTensorExpansions[j], *ham);
      exatn::TensorExpansion overlapMatrixElement(*listOfBraTensorExpansions[i], *listOfKetTensorExpansions[j]);
      auto created = exatn::createTensorSync("_ac_ham", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
      created = exatn::createTensorSync("_ac_overlap", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
      auto ac_ham = exatn::getTensor("_ac_ham");
      auto ac_overlap = exatn::getTensor("_ac_overlap");
  
      auto initialized = exatn::initTensor(ac_ham->getName(), 0.0); assert(initialized);
      initialized = exatn::initTensor(ac_ham->getName(), 0.0); assert(initialized);
      auto evaluated = exatn::evaluateSync(hamiltonianMatrixElement,ac_ham); assert(evaluated);
      evaluated = exatn::evaluateSync(overlapMatrixElement,ac_overlap); assert(evaluated);
      auto local_copy = exatn::getLocalTensor(ac_ham->getName());
      const exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
      auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
      H[i*8+j] =  *body_ptr;
      body_ptr = nullptr;

      local_copy = exatn::getLocalTensor(ac_overlap->getName());
      access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
      S[i*8+j] =  *body_ptr;

      auto destroyed = exatn::destroyTensorSync(ac_ham->getName()); assert(destroyed);
      destroyed = exatn::destroyTensorSync(ac_overlap->getName()); assert(destroyed);
    }
  }  

  std::cout << "Print out Hamiltonian matrix..." << std::endl;
  for ( unsigned int i = 0; i < listOfBraTensorExpansions.size(); i++){
    for ( unsigned int j = 0; j < listOfKetTensorExpansions.size(); j++){
      printf("%6.6lf  ", H[i*8+j]);
    }
    printf("\n");
  }
  std::cout << "Print out overlap matrix..." << std::endl;
  for ( unsigned int i = 0; i < listOfBraTensorExpansions.size(); i++){
    for ( unsigned int j = 0; j < listOfKetTensorExpansions.size(); j++){
      printf("%6.6lf  ", S[i*8+j]);
    }
    printf("\n");
  }
    
  MKL_INT n = N, lda = N, ldb = N, info;
  double w[N];
  info = LAPACKE_dsygv (LAPACK_ROW_MAJOR, 1, 'V', 'U', n, H, lda, S, ldb, w);
  if( info > 0 ) {
    printf( "The algorithm failed to compute eigenvalues.\n" );
    exit( 1 );
  }
  std::cout << "Eigenvalues..." << std::endl;
  for ( unsigned int i = 0; i < N; i++){
    printf("w[%i]: %6.6lf  \n", i, w[i]);
  }

  // build target
  std::shared_ptr<exatn::TensorExpansion> target;
  target = std::make_shared<exatn::TensorExpansion>();
  target->appendComponent(network_1,{H[0],0.0});
  target->appendComponent(network_2,{H[8],0.0});
  target->appendComponent(network_3,{H[16],0.0});
  target->appendComponent(network_4,{H[24],0.0});
  target->appendComponent(network_5,{H[32],0.0});
  target->appendComponent(network_6,{H[40],0.0});
  target->appendComponent(network_7,{H[48],0.0});
  target->appendComponent(network_8,{H[56],0.0});

  // set up approximant
  std::shared_ptr<exatn::TensorExpansion> approximant;
  approximant = std::make_shared<exatn::TensorExpansion>();
  auto network_for_approximant = exatn::makeTensorNetwork("NetworkForApproximant","_ac_approximant(p,q,r,s)=A(p,i)*B(i,q,j)*C(j,r,k)*D(k,s)");
  approximant->appendComponent(network_for_approximant,{1.0,0.0});
  markOptimizableTensors(approximant);
  approximant->conjugate();

  success = exatn::balanceNormalizeNorm2Sync(*target,1.0,1.0,true); assert(success);
  success = exatn::balanceNormalizeNorm2Sync(*approximant,1.0,1.0,true); assert(success);

  //run reconstruction 
  exatn::TensorNetworkReconstructor::resetDebugLevel(2);
  exatn::TensorNetworkReconstructor reconstructor(target,approximant,1e-4);
  //reconstructor.resetTolerance(1e-5); 

  //Run the reconstructor:
  success = exatn::sync(); assert(success);
  double residual_norm, fidelity;
  bool reconstructed = reconstructor.reconstruct(&residual_norm,&fidelity,true);
  success = exatn::sync(); assert(success);
  if(reconstructed){
    std::cout << "Reconstruction succeeded: Residual norm = " << residual_norm
            << "; Fidelity = " << fidelity << std::endl;
 }else{
  std::cout << "Reconstruction failed!" << std::endl; assert(false);
 }

  // get network of tensors (without) ordering projectors to build ansatz
  std::shared_ptr<exatn::TensorExpansion> ansatz;
  ansatz = std::make_shared<exatn::TensorExpansion>();
  ansatz->appendComponent(network_for_approximant,{1.0,0.0});
  markOptimizableTensors(ansatz);

  auto destroyed = exatn::destroyTensorSync(h1->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync(h2->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("A1"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("B1"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("C1"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("D1"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("A2"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("B2"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("C2"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("D2"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("A3"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("B3"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("C3"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("D3"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("A4"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("B4"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("C4"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("D4"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("A5"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("B5"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("C5"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("D5"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("A6"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("B6"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("C6"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("D6"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("A7"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("B7"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("C7"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("D7"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("A8"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("B8"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("C8"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("D8"); assert(destroyed);

  std::vector<std::shared_ptr<exatn::Tensor> > hamiltonian;

  // declare object from Simulation class
  Simulation myObject(nao, nap, nco, nto, ntp);
  myObject.resetWaveFunctionAnsatz(ansatz);
  myObject.resetHamiltonian(hamiltonian);
  myObject.optimize(1,1e-5);

  }

  exatn::finalize();
 
  return 0;

}
