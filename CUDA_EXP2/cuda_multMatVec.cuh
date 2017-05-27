#ifndef __CUDA_MULT_MAT_VEC_CUH__
#define __CUDA_MULT_MAT_VEC_CUH__

#include<stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <assert.h>

#define IN		const
#define OUT
#define INOUT
#define NEWOUT

typedef float TIMER_T;

const int REPEAT_COUNT = 1;
const unsigned ELEM_PER_VECTOR = 32;

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}

#define SELECTIVE
#define HOMEWORK
#define ADVANCED

/*
 함수명 해석 방법

  GlobalMemory:					Kernel에서 사용 중인 Parameter에 대해 모두 global memory를 사용하는 경우
  SharedMemory:					Kernel에서 사용 중인 parameter에 대해 하나 이상 shared memory를 사용하는 경우
  *ConstantMatrix:				Kernel에서 행렬 matA에 대해 constant memory를 사용하는 경우
   - Simple:					행렬 matA를 메모리 접근 형태에 대한 고려 없이 constantMatA로 단순 치환
   - Broadcast:	(구현 안됨)		행렬 matA에 대해 32개의 thread가 broadcast의 형태로 데이터를 접근하도록 고려하여 constantMatA로 치환
  
  Strided*:						과제를 하면서 이해할 것

  WithoutRegister:				내부에 Register의 사용 없이 직접 global memory로 접근하는 경우
 								WithoutRegister가 없는 경우는 일반적으로 내부에서 register 사용을 전제로 함
 
  _Vector:						thread가 하나의 vector 계산을 수행함
  _Element*:					thread가 하나의 element 계산을 수행함
   - 32ThreadsPerBlock:			하나의 thread block 당 32개의 thread 사용
   - 1024ThreadsPerBlock:		하나의 thread block 당 1024개의 thread 사용
  (1, 32)VectorWith32Threads:	32개의 thread로 1개 혹은 32개의 vector 계산을 수행함

  WithAtomic:					CUDA가 제공하는 atomicAdd()를 이용하여 계산에 사용
 */

__global__				void MultMatVec_GPU_GlobalMemoryWithoutRegister_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ SELECTIVE	void MultMatVec_GPU_GlobalMemory_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__				void MultMatVec_GPU_GlobalMemoryWithoutRegister_Element32ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__				void MultMatVec_GPU_GlobalMemory_Element32ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ SELECTIVE	void MultMatVec_GPU_GlobalMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ HOMEWORK		void MultMatVec_GPU_SimpleConstantMatrix_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__				void MultMatVec_GPU_BroadcastConstantMatrix_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ HOMEWORK		void MultMatVec_GPU_SimpleConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_GlobalMemoryWithAtomic_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ HOMEWORK		void MultMatVec_GPU_StridedGlobalMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_StridedConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__				void MultMatVec_GPU_Strided32VectorGlobalMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__				void MultMatVec_GPU_Strided32VectorConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__				void MultMatVec_GPU_Strided32VectorSharedMemoryConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ HOMEWORK		void MultMatVec_GPU_Strided32VectorSharedMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ ADVANCED		void MultMatVec_GPU_SharedMemoryWithAtomic_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_SharedMemoryWithAtomic_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ ADVANCED		void MultMatVec_GPU_VariableSharedMemoryConstantMatrix_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_VariableSharedMemoryWithAtomic_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_VariableSharedMemoryConstantMatrix_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_VariableSharedMemory_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ ADVANCED		void MultMatVec_GPU_SimpleConstantMatrixWithAtomic_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_SharedMemoryConstantMatrixWithAtomic_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ ADVANCED		void MultMatVec_GPU_SharedMemoryConstantMatrix_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_SharedMemoryConstantMatrix_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ ADVANCED		void MultMatVec_GPU_SharedMemory_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

__global__ ADVANCED		void MultMatVec_GPU_SharedMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );
__global__ ADVANCED		void MultMatVec_GPU_SharedMemoryConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX );

void GenerateConstantMatrix( IN float( *matA )[ ELEM_PER_VECTOR ] );

#endif