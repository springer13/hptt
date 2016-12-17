#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <stdlib.h>

#include "ttc.h"

#include "sTranspose_210_384x2320x64.h"
#include "sTranspose_210_96x9216x96.h"
#include "sTranspose_3210_96x96x96x96.h"
#include "sTranspose_3021_96x96x96x96.h"
#include "sTranspose_43210_64x64x64x64x64.h"
#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/operations/operation_trans.h>


using namespace std;
using namespace hptc;


template <typename FloatType,
          GenNumType HEIGHT,
          GenNumType WIDTH>
class BenchmarkCreator {
public:
  using Deduced = DeducedFloatType<FloatType>;
  using OpBase = Operation<FloatType, ParamTrans>;

  BenchmarkCreator(vector<TensorIdx> input_size,
      const vector<TensorDim> &perm, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta) {
    // Prepare internal data member
    this->micro_size_ = 32 / sizeof(FloatType);
    this->macro_height_ = HEIGHT * this->micro_size_;
    this->macro_width_ = WIDTH * this->micro_size_;

    // Create tensor size objects
    TensorDim tensor_dim = static_cast<TensorDim>(input_size.size());
    TensorSize input_size_obj(input_size), output_size_obj(tensor_dim);
    for (TensorDim idx = 0; idx < tensor_dim; ++idx)
      output_size_obj[idx] = input_size_obj[perm[idx]];

    // Create raw data and initialize value
    this->input_data_len = this->output_data_len = accumulate(
        input_size.begin(), input_size.end(), 1, multiplies<TensorIdx>());
    this->input_data = new FloatType [this->input_data_len];
    this->output_data = new FloatType [this->output_data_len];
    this->reset_data();

    // Initialize tensor wrapper
    this->input_tensor = TensorWrapper<FloatType>(input_size_obj,
        this->input_data);
    this->output_tensor = TensorWrapper<FloatType>(output_size_obj,
        this->output_data);

    // Initialize transpose parameter
    this->param = make_shared<ParamTrans<FloatType>>(this->input_tensor,
        this->output_tensor, perm, alpha, beta);

    // Initialization computational graph
    this->build_graph();
  }


  ~BenchmarkCreator() {
    delete [] this->input_data;
    delete [] this->output_data;
  }


  void reset_data() {
#pragma omp parallel for
     for(int i=0;i < this->input_data_len ; ++i)
        input_data[i] = (((i+1)*13 % 10000) - 5000.) / 10000.;
#pragma omp parallel for
     for(int i=0;i < this->input_data_len ; ++i)
        output_data[i] = (((i+1)*13 % 10000) - 5000.) / 10000.;
//    constexpr TensorIdx inner_offset = sizeof(FloatType) / sizeof(Deduced);
//    // Reset input data
//    for (TensorIdx idx = 0; idx < this->input_data_len; ++idx) {
//      Deduced *reset_ptr = reinterpret_cast<Deduced *>(&this->input_data[idx]);
//      for (TensorIdx inner_idx = 0; inner_idx < inner_offset; ++inner_idx)
//        reset_ptr[inner_idx] = static_cast<Deduced>(idx);
//    }
//
//    // Reset output data
//    Deduced *reset_ptr = reinterpret_cast<Deduced *>(this->output_data);
//    fill(reset_ptr, reset_ptr + this->output_data_len * inner_offset,
//        static_cast<float>(-1));
  }


  shared_ptr<OpBase> build_graph() {
    using SingleFor = OpLoopFor<FloatType, ParamTrans, 1>;
    if (nullptr == this->input_data or nullptr == output_data)
      return nullptr;

    // Get dimension informaiton
    TensorDim tensor_dim = this->input_tensor.get_size().get_dim();

    // Create for loops
    shared_ptr<SingleFor> curr_op
        = static_pointer_cast<SingleFor>(this->operation);
    for (TensorDim idx = 0; idx < tensor_dim; ++idx) {
      // Compute parameters
      TensorIdx begin = 0, end, step;
      if (0 == idx) {
        end = this->input_tensor.get_size()[0];
        step = this->macro_height_;
      }
      else if (this->param->perm[0] == idx) {
        end = this->input_tensor.get_size()[idx];
        step = this->macro_width_;
      }
      else {
        end = this->input_tensor.get_size()[idx];
        step = 1;
      }

      // Create a new for loop
      shared_ptr<SingleFor> new_op = make_shared<SingleFor>(this->param,
          this->param->macro_loop_idx[idx], begin, end, step);

      // Connect new for loop with previous one
      if (nullptr == curr_op)
        this->operation = new_op;
      else
        curr_op->init_operation(new_op);
      curr_op = new_op;
    }

    // Create macro kernel
    curr_op->init_operation(
        make_shared<OpMacroTrans<FloatType, HEIGHT, WIDTH>>(this->param));

    return this->operation;
  }


  inline void exec() {
    this->operation->exec();
  }


  TensorWrapper<FloatType> &get_input_tensor() {
    return this->input_tensor;
  }


  TensorWrapper<FloatType> &get_output_tensor() {
    return this->output_tensor;
  }

//private:
  TensorIdx micro_size_, macro_height_, macro_width_;

  TensorIdx input_data_len, output_data_len;
  FloatType *input_data, *output_data;
  TensorWrapper<FloatType> input_tensor, output_tensor;
  shared_ptr<ParamTrans<FloatType>> param;
  shared_ptr<OpBase> operation;
};

void equal_(const float *A, const float*B, int total_size){
  int error = 0;
   const float *Atmp= A;
   const float *Btmp= B;
   for(int i=0;i < total_size ; ++i){
      double Aabs = (Atmp[i] < 0) ? -Atmp[i] : Atmp[i];
      double Babs = (Btmp[i] < 0) ? -Btmp[i] : Btmp[i];
      double max = (Aabs < Babs) ? Babs : Aabs;
      double diff = (Aabs - Babs);
      diff = (diff < 0) ? -diff : diff;
      if(diff > 0){
         double relError = (diff / max);
         if(relError > 4e-5){
            //printf("i: %d relError: %.8e\n",i,relError);
            error += 1;
         }
      }
   }
   if( error > 0 ) 
     printf("ERROR\n");
  else
     printf("SUCCESS\n");
}

int main(int argc, char *argv[]) {
  int numThreads = 1;
  if( getenv("OMP_NUM_THREADS") != NULL )
     numThreads = atoi(getenv("OMP_NUM_THREADS"));
  printf("numThreads: %d\n",numThreads);

  //###################### 4-dim ##########################################
  // Prepare data
  vector<TensorIdx> size{ 64, 2320, 384 };
  vector<TensorDim> perm{ 2, 1, 0 };
  float alpha = 2.1;
  float beta = 4.0;

  // Create transpose computational graph
  BenchmarkCreator<float, 2, 2> inst(size, perm, alpha, beta);
  //
  //create a copy of B
  int total_size = 384* 2320* 64;
  int largerThanL3 = 1024*1024*100/sizeof(double);
  float* B_copy, *B_ttc;
  double *trash1, *trash2;
  posix_memalign((void**) &trash1, 32, sizeof(double) * largerThanL3);
  posix_memalign((void**) &trash2, 32, sizeof(double) * largerThanL3);
  posix_memalign((void**) &B_copy, 32, sizeof(float) * total_size);
  posix_memalign((void**) &B_ttc, 32, sizeof(float) * total_size);
  int nRepeat = 5;
#pragma omp parallel for
  for(int i=0;i < total_size ; ++i)
  {
     B_copy[i] = inst.output_data[i];
     B_ttc[i] = inst.output_data[i];
  }

#pragma omp parallel for
  for(int i=0;i < largerThanL3; ++i)
  {
     trash1[i] = ((i+1)*13)%100000;
     trash2[i] = ((i+1)*13)%100000;
  }

  // Execute transpose
  { // original ttc
     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        sTranspose_210_384x2320x64<384,2320,64>( inst.input_data, B_ttc, alpha, beta, NULL, NULL);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC: 3-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  {  //ttc-c-paul
     int perm_[] = {2,1,0};
     int size_[] = {384,2320,64}; // ATTENTION: different order than in Tong's code
     int dim_ = 3;
     auto plan = createPlan(NULL, NULL, size_, perm_, dim_, numThreads);

     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        ttc_sTranspose(inst.input_data, B_copy, alpha, beta, plan);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (paul): 3-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  // Verification
  equal_(B_ttc, B_copy, total_size);

  {  //ttc-c-tong
     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        inst.exec();
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (tong): 3-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }

  // Verification
  equal_(inst.output_data, B_ttc, total_size);
  free(B_copy);
  free(B_ttc);

  //###################### 4-dim ##########################################
  // Prepare data
  vector<TensorIdx> size4{ 96, 96, 96, 96 };
  vector<TensorDim> perm4{ 3, 0, 2, 1 };

  // Create transpose computational graph
  BenchmarkCreator<float, 2, 2> inst4(size4, perm4, alpha, beta);
  posix_memalign((void**) &B_ttc , 32, sizeof(float) * inst4.input_data_len);
  posix_memalign((void**) &B_copy, 32, sizeof(float) * inst4.input_data_len);

#pragma omp parallel for
  for(int i=0;i < inst4.input_data_len; ++i){
     B_copy[i] = inst4.output_data[i];
     B_ttc[i] = inst4.output_data[i];
  }

  // Execute transpose
  { // original ttc
     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        //sTranspose_3210_96x96x96x96<96,96,96,96>(inst4.input_data, B_ttc, alpha, beta, NULL, NULL);
        sTranspose_3021_96x96x96x96<96,96,96,96>(inst4.input_data, B_ttc, alpha, beta, NULL, NULL);
        //sTranspose_210_96x9216x96<96,9216,96>(inst4.input_data, B_ttc, alpha, beta, NULL, NULL);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (orig): 4-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  {  // my TTC-C
     int perm_[] = {3,0,2,1};
     int size_[] = {96,96,96,96}; // ATTENTION: different order than in Tong's code
     int dim_ = 4;
     auto plan = createPlan(NULL, NULL, size_, perm_, dim_, numThreads);

     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        ttc_sTranspose(inst4.input_data, B_copy, alpha, beta, plan);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (paul): 4-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  printf("my: ");
  equal_(B_ttc, B_copy, total_size);

  {  // Tong's TTC-C
     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        inst4.exec();
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (tong): 4-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  } 
  // Verification
  printf("tong: ");
  equal_(inst4.output_data, B_copy, total_size);
  free(B_copy);
  free(B_ttc);
  exit(0);

  //###################### 5-dim ##########################################
  // Prepare data
  vector<TensorIdx> size5{ 64, 64, 64, 64, 64};
  vector<TensorDim> perm5{ 4, 3, 2, 1, 0 };

  // Create transpose computational graph
  BenchmarkCreator<float, 2, 2> inst5(size5, perm5, alpha, beta);
  posix_memalign((void**) &B_ttc , 32, sizeof(float) * inst5.input_data_len);
  posix_memalign((void**) &B_copy, 32, sizeof(float) * inst5.input_data_len);

#pragma omp parallel for
  for(int i=0;i < inst5.input_data_len; ++i){
     B_copy[i] = inst5.output_data[i];
     B_ttc[i] = inst5.output_data[i];
  }


  // Execute transpose
  { // original ttc
     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        sTranspose_43210_64x64x64x64x64<64,64,64,64,64>(inst5.input_data, B_ttc, alpha, beta, NULL, NULL);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (orig): 5-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  {  // my TTC-C
     int perm_[] = {4,3,2,1,0};
     int size_[] = {64,64,64,64,64}; // ATTENTION: different order than in Tong's code
     int dim_ = 5;
     auto plan = createPlan(NULL, NULL, size_, perm_, dim_, numThreads);

     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        ttc_sTranspose(inst5.input_data, B_copy, alpha, beta, plan);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (paul): 5-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  equal_(B_ttc, B_copy, total_size);

  {  // tong
     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        inst5.exec();
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (tong): 5-dim Elapsed time: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  printf("tong: ");
  equal_(inst5.output_data, B_copy, total_size);

  cout << "Transpose done!" << endl;
  return 0;
}
