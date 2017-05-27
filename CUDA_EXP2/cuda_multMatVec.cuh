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
 �Լ��� �ؼ� ���

  GlobalMemory:					Kernel���� ��� ���� Parameter�� ���� ��� global memory�� ����ϴ� ���
  SharedMemory:					Kernel���� ��� ���� parameter�� ���� �ϳ� �̻� shared memory�� ����ϴ� ���
  *ConstantMatrix:				Kernel���� ��� matA�� ���� constant memory�� ����ϴ� ���
   - Simple:					��� matA�� �޸� ���� ���¿� ���� ��� ���� constantMatA�� �ܼ� ġȯ
   - Broadcast:	(���� �ȵ�)		��� matA�� ���� 32���� thread�� broadcast�� ���·� �����͸� �����ϵ��� ����Ͽ� constantMatA�� ġȯ
  
  Strided*:						������ �ϸ鼭 ������ ��

  WithoutRegister:				���ο� Register�� ��� ���� ���� global memory�� �����ϴ� ���
 								WithoutRegister�� ���� ���� �Ϲ������� ���ο��� register ����� ������ ��
 
  _Vector:						thread�� �ϳ��� vector ����� ������
  _Element*:					thread�� �ϳ��� element ����� ������
   - 32ThreadsPerBlock:			�ϳ��� thread block �� 32���� thread ���
   - 1024ThreadsPerBlock:		�ϳ��� thread block �� 1024���� thread ���
  (1, 32)VectorWith32Threads:	32���� thread�� 1�� Ȥ�� 32���� vector ����� ������

  WithAtomic:					CUDA�� �����ϴ� atomicAdd()�� �̿��Ͽ� ��꿡 ���
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