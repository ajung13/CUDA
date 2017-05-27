#include "cuda_multMatVec.cuh"


__global__ void MultMatVec_GPU_GlobalMemoryWithoutRegister_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		vecY[ tid * ELEM_PER_VECTOR + i] = 0.0f; // No use register
		for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			vecY[ tid * ELEM_PER_VECTOR + i ] += matA[ i ][ j ] * vecX[ tid * ELEM_PER_VECTOR + j ];
		}
	}
}

__global__ SELECTIVE void MultMatVec_GPU_GlobalMemory_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	float result;
	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		result = 0.0f;
		for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			result += matA[ i ][ j ] * vecX[ tid * ELEM_PER_VECTOR + j ];
		}
		vecY[ tid * ELEM_PER_VECTOR + i ] = result;
	}
}

__global__ void MultMatVec_GPU_GlobalMemoryWithoutRegister_Element32ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

	vecY[ tid ] = 0.0f;
	for( unsigned j = 0; j < blockDim.x; ++j )
	{
		vecY[ tid ] += matA[ threadIdx.x ][ j ] * vecX[ blockIdx.x * ELEM_PER_VECTOR + j ];
	}
}

__global__ void MultMatVec_GPU_GlobalMemory_Element32ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	float result = 0.0f;
	for( unsigned j = 0; j < blockDim.x; ++j )
	{
		result += matA[ threadIdx.x ][ j ] * vecX[ blockIdx.x * ELEM_PER_VECTOR + j ];
	}
	vecY[ blockIdx.x * blockDim.x + threadIdx.x ] = result;
}

__global__ SELECTIVE void MultMatVec_GPU_GlobalMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += matA[ eid ][ j ] * vecX[ vid * ELEM_PER_VECTOR + j ];
	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__global__ ADVANCED void MultMatVec_GPU_GlobalMemoryWithAtomic_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		vecY[ vid * ELEM_PER_VECTOR + i ] = 0.0f;
		atomicAdd( &vecY[ vid * ELEM_PER_VECTOR + i ], matA[ i ][ eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ] );
	}
}

__global__ HOMEWORK void MultMatVec_GPU_StridedGlobalMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned n = gridDim.x * blockDim.x;
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned stride = ( n / ELEM_PER_VECTOR );

	unsigned vid = tid % stride;
	unsigned eid = tid / stride;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += matA[ eid ][ j ] * vecX[ vid * ELEM_PER_VECTOR + j ];
	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__constant__ float constantMatA[ ELEM_PER_VECTOR ][ ELEM_PER_VECTOR ];
void GenerateConstantMatrix( IN float( *matA )[ ELEM_PER_VECTOR ] )
{
	cudaMemcpyToSymbol( constantMatA, matA, sizeof( float ) * ELEM_PER_VECTOR * ELEM_PER_VECTOR );
}

__global__ HOMEWORK void MultMatVec_GPU_SimpleConstantMatrix_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	float result;
	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		result = 0.0f;
		for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			result += constantMatA[ i ][ j ] * vecX[ tid * ELEM_PER_VECTOR + j ];
		}
		vecY[ tid * ELEM_PER_VECTOR + i ] = result;
	}
}

__global__ ADVANCED void MultMatVec_GPU_SimpleConstantMatrixWithAtomic_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		vecY[ vid * ELEM_PER_VECTOR + i ] = 0.0f;
		atomicAdd( &vecY[ vid * ELEM_PER_VECTOR + i ], constantMatA[ i ][ eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ] );
	}
}

__global__ HOMEWORK void MultMatVec_GPU_SimpleConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += constantMatA[ eid ][ j ] * vecX[ vid * ELEM_PER_VECTOR + j ];
	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__global__ ADVANCED void MultMatVec_GPU_SharedMemoryConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	__shared__ float sharedVecX[ 1024 ];

	sharedVecX[ threadIdx.x ] = vecX[ tid ];
	unsigned svid = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += constantMatA[ eid ][ j ] * sharedVecX[ svid + j ];

	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__global__ ADVANCED void MultMatVec_GPU_SharedMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	__shared__ float sharedVecX[ 1024 ];
	__shared__ float sharedMatA[ 1024 ];

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )matA )[ accessID + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	sharedVecX[ threadIdx.x ] = vecX[ tid ];
	unsigned svid = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += sharedMatA[ eid * ELEM_PER_VECTOR + j ] * sharedVecX[ svid + j ];

	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__global__ ADVANCED void MultMatVec_GPU_StridedConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	// BLOCK DIM: 128 (CC:3.5)
	unsigned n = gridDim.x * blockDim.x;
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned stride = ( n / ELEM_PER_VECTOR );
	
	unsigned vid = tid % stride;
	unsigned eid = tid / stride;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += constantMatA[ eid ][ j ] * vecX[ vid * ELEM_PER_VECTOR + j ];
	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__global__ void MultMatVec_GPU_Strided32VectorGlobalMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	// BLOCK DIM: 128 (CC:3.5)
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = ( tid % ELEM_PER_VECTOR ) + ( tid / 1024 * ELEM_PER_VECTOR ); // blockDim.x
	unsigned eid = ( tid % 1024 / ELEM_PER_VECTOR );

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += matA[ eid ][ j ] * vecX[ vid * ELEM_PER_VECTOR + j ];
	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__global__ void MultMatVec_GPU_Strided32VectorConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	// BLOCK DIM: 128 (CC:3.5)
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = ( tid % ELEM_PER_VECTOR ) + ( tid / 1024 * ELEM_PER_VECTOR ); // blockDim.x
	unsigned eid = ( tid % 1024 / ELEM_PER_VECTOR );

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += constantMatA[ eid ][ j ] * vecX[ vid * ELEM_PER_VECTOR + j ];
	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__global__ void MultMatVec_GPU_Strided32VectorSharedMemoryConstantMatrix_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	// BLOCK DIM: 128 (CC:3.5)
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = ( tid % ELEM_PER_VECTOR ) + ( tid / 1024 * ELEM_PER_VECTOR ); // blockDim.x
	unsigned eid = ( tid % 1024 / ELEM_PER_VECTOR );

	__shared__ float sharedVecX[ 1024 ];
	sharedVecX[ threadIdx.x ] = vecX[ tid ];
	unsigned svid = threadIdx.x % ELEM_PER_VECTOR * ELEM_PER_VECTOR;
	__syncthreads();

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += constantMatA[ eid ][ j ] * sharedVecX[ svid + j ];
	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__global__ HOMEWORK void MultMatVec_GPU_Strided32VectorSharedMemory_Element1024ThreadsPerBlock( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	// BLOCK DIM: 128 (CC:3.5)
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = ( tid % ELEM_PER_VECTOR ) + ( tid / 1024 * ELEM_PER_VECTOR ); // blockDim.x
	unsigned eid = ( tid % 1024 / ELEM_PER_VECTOR );

	__shared__ float sharedVecX[ 1024 ];
	__shared__ float sharedMatA[ 1024 ];

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )matA )[ accessID + i * ELEM_PER_VECTOR ];

	sharedVecX[ threadIdx.x ] = vecX[ tid ];
	unsigned svid = threadIdx.x % ELEM_PER_VECTOR * ELEM_PER_VECTOR;
	__syncthreads( );

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
	{
		result += sharedMatA[ eid * ELEM_PER_VECTOR + j ] * sharedVecX[ svid + j ];
	}
	vecY[ vid * ELEM_PER_VECTOR + eid ] = result;
}

// 이거 아직 실험해볼 코드가 구성이 안됨
__global__ void MultMatVec_GPU_BroadcastConstantMatrix_Vector( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	float result;
	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		result = 0.0f;
		for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			result += constantMatA[ i ][ j ] * vecX[ tid * ELEM_PER_VECTOR + j ];
		}
		vecY[ tid * ELEM_PER_VECTOR + i ] = result;
	}
}

//extern __shared__ unsigned char sharedBuffer[];
//__shared__ float sharedVecX[];
__global__ ADVANCED void MultMatVec_GPU_SharedMemoryWithAtomic_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	__shared__ float sharedMatA[ 1024 ];
	__shared__ float sharedVecY[ 128 / ELEM_PER_VECTOR ];

	// normally access
	// GTX 690, 29.x ms
	//for( unsigned i = 0; i < ratio; ++i )
	//	sharedMatA[ threadIdx.x * ratio + i ] = ( ( float* )matA )[ threadIdx.x * ratio + i ];
	//__syncthreads();

	// coalesed access
	// GTX 690, 25.x ms
	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * (ELEM_PER_VECTOR * ratio) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )matA )[ accessID + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ] = 0.0f;
		atomicAdd( &sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ], sharedMatA[ i * ELEM_PER_VECTOR + eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ] );
		vecY[ vid * ELEM_PER_VECTOR + i ] = sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ];

		//vecY[ vid * ELEM_PER_VECTOR + i ] = 0.0f;
		//atomicAdd( &vecY[ vid * ELEM_PER_VECTOR + i ], sharedMatA[ i * ELEM_PER_VECTOR + eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ] );
	}
}

__global__ ADVANCED void MultMatVec_GPU_SharedMemoryConstantMatrixWithAtomic_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	//__shared__ float sharedMatA[ 1024 ];
	__shared__ float sharedVecY[ 128 / ELEM_PER_VECTOR ];

	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ] = 0.0f;
		atomicAdd( &sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ], constantMatA[ i ][ eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ] );
		vecY[ vid * ELEM_PER_VECTOR + i ] = sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ];
	}
}

__global__ ADVANCED void MultMatVec_GPU_SharedMemoryConstantMatrix_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	__shared__ float sharedVecY[ 128 ];

	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		sharedVecY[ threadIdx.x ] = constantMatA[ i ][ eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ];

		if( threadIdx.x % ELEM_PER_VECTOR == 0 )
		{
			// shared vector index
			unsigned sviStart = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
			unsigned sviEnd = sviStart + ELEM_PER_VECTOR;
			for( unsigned j = sviStart + 1; j < sviEnd; ++j )
				sharedVecY[ sviStart ] += sharedVecY[ j ];
			vecY[ vid * ELEM_PER_VECTOR + i ] = sharedVecY[ sviStart ];
		}
	}
}

__shared__ float _sharedVecY[];
__shared__ float _sharedVecX[];
__global__ ADVANCED void MultMatVec_GPU_VariableSharedMemoryConstantMatrix_1VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix

	for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		_sharedVecY[ threadIdx.x ] = constantMatA[ i ][ eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ];

		if( threadIdx.x % ELEM_PER_VECTOR == 0 )
		{
			// shared vector index
			unsigned sviStart = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
			unsigned sviEnd = sviStart + ELEM_PER_VECTOR;
			for( unsigned j = sviStart + 1; j < sviEnd; ++j )
				_sharedVecY[ sviStart ] += _sharedVecY[ j ];
			vecY[ vid * ELEM_PER_VECTOR + i ] = _sharedVecY[ sviStart ];
		}
	}
}

__global__ ADVANCED void MultMatVec_GPU_SharedMemoryWithAtomic_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	__shared__ float sharedMatA[ 1024 ];
	__shared__ float sharedVecY[ 128 / ELEM_PER_VECTOR ];

	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )matA )[ accessID + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	unsigned vidMax = vid + ELEM_PER_VECTOR;
	for( ; vid < vidMax; ++vid )
	{
		for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
		{
			sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ] = 0.0f;
			atomicAdd( &sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ], sharedMatA[ i * ELEM_PER_VECTOR + eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ] );
			vecY[ vid * ELEM_PER_VECTOR + i ] = sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ];
		}
	}
}

__global__ ADVANCED void MultMatVec_GPU_VariableSharedMemoryWithAtomic_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	__shared__ float sharedMatA[ 1024 ];
	//__shared__ float sharedVecY[ blockDim.x / ELEM_PER_VECTOR ];

	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )matA )[ accessID + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	unsigned vidMax = vid + ELEM_PER_VECTOR;
	for( ; vid < vidMax; ++vid )
	{
		for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
		{
			_sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ] = 0.0f;
			atomicAdd( &_sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ], sharedMatA[ i * ELEM_PER_VECTOR + eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ] );
			vecY[ vid * ELEM_PER_VECTOR + i ] = _sharedVecY[ threadIdx.x / ELEM_PER_VECTOR ];
		}
	}
}

__global__ ADVANCED void MultMatVec_GPU_SharedMemoryConstantMatrix_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	__shared__ float sharedVecY[ 128 ];

	unsigned vidMax = vid + ELEM_PER_VECTOR;
	for( ; vid < vidMax; ++vid )
	{
		for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
		{
			sharedVecY[ threadIdx.x ] = constantMatA[ i ][ eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ];
			__syncthreads();

			if( threadIdx.x % ELEM_PER_VECTOR == 0 )
			{
				// shared vector index
				unsigned sviStart = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
				unsigned sviEnd = sviStart + ELEM_PER_VECTOR;
				for( unsigned j = sviStart + 1; j < sviEnd; ++j )
					sharedVecY[ sviStart ] += sharedVecY[ j ];
				vecY[ vid * ELEM_PER_VECTOR + i ] = sharedVecY[ sviStart ];
			}
		}
	}
}

__global__ ADVANCED void MultMatVec_GPU_VariableSharedMemoryConstantMatrix_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	//__shared__ float sharedVecY[ blockDim.x ];

	unsigned vidMax = vid + ELEM_PER_VECTOR;
	for( ; vid < vidMax; ++vid )
	{
		for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
		{
			_sharedVecY[ threadIdx.x ] = constantMatA[ i ][ eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ];

			if( threadIdx.x % ELEM_PER_VECTOR == 0 )
			{
				// shared vector index
				unsigned sviStart = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
				unsigned sviEnd = sviStart + ELEM_PER_VECTOR;
				for( unsigned j = sviStart + 1; j < sviEnd; ++j )
					_sharedVecY[ sviStart ] += _sharedVecY[ j ];
				vecY[ vid * ELEM_PER_VECTOR + i ] = _sharedVecY[ sviStart ];
			}
		}
	}
}

__global__ ADVANCED void MultMatVec_GPU_SharedMemory_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	__shared__ float sharedMatA[ 1024 ];
	__shared__ float sharedVecY[ 1024 ];

	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )matA )[ accessID + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	unsigned vidMax = vid + ELEM_PER_VECTOR;
	for( ; vid < vidMax; ++vid )
	{
		for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
		{
			sharedVecY[ threadIdx.x ] = sharedMatA[ i * ELEM_PER_VECTOR + eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ];

			if( threadIdx.x % ELEM_PER_VECTOR == 0 )
			{
				// shared vector index
				unsigned sviStart = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
				unsigned sviEnd = sviStart + ELEM_PER_VECTOR;
				for( unsigned j = sviStart + 1; j < sviEnd; ++j )
					sharedVecY[ sviStart ] += sharedVecY[ j ];
				vecY[ vid * ELEM_PER_VECTOR + i ] = sharedVecY[ sviStart ];
			}
		}
	}
}

__shared__ float sharedFloatBuffer[ ];
__global__ ADVANCED void MultMatVec_GPU_VariableSharedMemory_32VectorWith32Threads( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX )
{
	unsigned tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	float *sharedMatA = sharedFloatBuffer;
	float *sharedVecY = sharedFloatBuffer + 1024;

	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )matA )[ accessID + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	unsigned vidMax = vid + ELEM_PER_VECTOR;
	for( ; vid < vidMax; ++vid )
	{
		for( unsigned i = 0; i < ELEM_PER_VECTOR; ++i )
		{
			sharedVecY[ threadIdx.x ] = sharedMatA[ i * ELEM_PER_VECTOR + eid ] * vecX[ vid * ELEM_PER_VECTOR + eid ];

			if( threadIdx.x % ELEM_PER_VECTOR == 0 )
			{
				// shared vector index
				unsigned sviStart = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;
				unsigned sviEnd = sviStart + ELEM_PER_VECTOR;
				for( unsigned j = sviStart + 1; j < sviEnd; ++j )
					sharedVecY[ sviStart ] += sharedVecY[ j ];
				vecY[ vid * ELEM_PER_VECTOR + i ] = sharedVecY[ sviStart ];
			}
		}
	}
}