#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

#include "cuda_multMatVec.cuh"

typedef float TIMER_T;

#define USE_CPU_TIMER 1
#define USE_GPU_TIMER 1


#if USE_CPU_TIMER == 1
__int64 start, freq, end;
#define CHECK_TIME_START() { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }
#else
#define CHECK_TIME_START()
#define CHECK_TIME_END(a)
#endif


#if USE_GPU_TIMER == 1
cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

void create_device_timer()
{
	CUDA_CALL( cudaEventCreate( &cuda_timer_start ) );
	CUDA_CALL( cudaEventCreate( &cuda_timer_stop ) );
}

void destroy_device_timer()
{
	CUDA_CALL( cudaEventDestroy( cuda_timer_start ) );
	CUDA_CALL( cudaEventDestroy( cuda_timer_stop ) );
}

inline void start_device_timer()
{
	cudaEventRecord( cuda_timer_start, CUDA_STREAM_0 );
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord( cuda_timer_stop, CUDA_STREAM_0 );
	cudaEventSynchronize( cuda_timer_stop );

	cudaEventElapsedTime( &ms, cuda_timer_start, cuda_timer_stop );
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }
#else
#define CHECK_TIME_INIT_GPU()
#define CHECK_TIME_START_GPU()
#define CHECK_TIME_END_GPU(a)
#define CHECK_TIME_DEST_GPU()
#endif

__host__ void cuda_error_check( const char * prefix, const char * postfix )
{
	if( cudaPeekAtLastError() != cudaSuccess )
	{
		printf( "%s%s%s", prefix, cudaGetErrorString( cudaGetLastError() ), postfix );
		cudaDeviceReset();
		//wait_exit();
		exit( 1 );
	}
}

void MultMatVec_CPU_WithPrefetch( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndPrefetch( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling2( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling2( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_PrefetchAndUnrolling4_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling8( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling8_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling16( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling32( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );

inline float absIEEE754( float f)
{
	return ( float& )( ( int& )f &= 0x7fffffff );
}

float GetErrorRate( IN float* vecYcpuResult, IN float* vecYgpuResult, IN int numOfVectorElems )
{
	int cnt = 0;
	float epsilon = 0.000005f;
	for( int i = 0; i < numOfVectorElems; ++i )
	{
		if( absIEEE754( vecYcpuResult[ i ] - vecYgpuResult[ i ] ) > epsilon )
		{
			cnt++;
			//printf( "[%d][%d]: %f != %f\n", i / ELEM_PER_VECTOR, i % ELEM_PER_VECTOR, vecYcpuResult[ i ], vecYgpuResult[ i ] );
		}
	}

	//printf( " - Num of total elements: %d\n", numOfVectorElems );
	//printf( " - Num of error counts: %d\n", cnt );
	return float( cnt ) / numOfVectorElems * 100.f;
}

int main()
{
	float cpuTime, totalCPUtime;
	float gpuTime, totalGPUtime;

	float *vecX, *vecYcpuResult, *vecYgpuResult, ( *matA )[ ELEM_PER_VECTOR ];
	float *vecXcpu, *vecYcpu, ( *matAcpu )[ ELEM_PER_VECTOR ];
	float *vecXgpu, *vecYgpu, ( *matAgpu )[ ELEM_PER_VECTOR ];

	CHECK_TIME_INIT_GPU();

	FILE* fp = fopen( "gen.bin", "rb" );

	int numOfVectors, numOfVectorElems, numOfMatrixElems;
	fread( &numOfVectors, sizeof( int ), 1, fp );

	numOfVectorElems = numOfVectors * ELEM_PER_VECTOR;
	numOfMatrixElems = ELEM_PER_VECTOR * ELEM_PER_VECTOR;

	vecYcpuResult = new float[ numOfVectorElems ]();
	vecYgpuResult = new float[ numOfVectorElems ]();
	vecX = new float[ numOfVectorElems ]();
	matA = new float[ ELEM_PER_VECTOR ][ ELEM_PER_VECTOR ]();

	fread( vecX, sizeof( float ), numOfVectorElems, fp );
	fread( matA, sizeof( float ), numOfMatrixElems, fp );

	printf( "Finish CPU memory allocation and read datas from storage to host\n\n" );

#define CPU_FUNC_CALL(funcname) \
	totalCPUtime = 0; \
	vecYcpu = new float[ numOfVectorElems ](); \
	vecXcpu = new float[ numOfVectorElems ]();						memcpy( vecXcpu, vecX, sizeof( float ) * numOfVectorElems ); \
	matAcpu = new float[ ELEM_PER_VECTOR ][ ELEM_PER_VECTOR ]();	memcpy( matAcpu, matA, sizeof( float ) * numOfMatrixElems ); \
	for( int i = 0; i < REPEAT_COUNT; ++i ) \
	{ \
		CHECK_TIME_START(); \
		funcname( vecYcpu, matAcpu, vecXcpu, numOfVectors ); \
		CHECK_TIME_END( cpuTime ); \
		totalCPUtime += cpuTime; \
	} \
	printf( "Finish " #funcname " calculation\n" ); \
	printf( " - Elapsed time: %f\n\n", totalCPUtime / REPEAT_COUNT ); \
	memcpy( vecYcpuResult, vecYcpu, sizeof( float ) * numOfVectorElems ); \
	delete[] vecYcpu; \
	delete[] vecXcpu; \
	delete[] matAcpu;
#define CPU_FUNC_CALL__MACRO_END

	//CPU_FUNC_CALL( MultMatVec_CPU_WithPrefetch );
	//CPU_FUNC_CALL( MultMatVec_CPU_ );
	//CPU_FUNC_CALL( MultMatVec_CPU_AndPrefetch );
	//CPU_FUNC_CALL( MultMatVec_CPU_AndUnrolling2 );
	//CPU_FUNC_CALL( MultMatVec_CPU_AndUnrolling4 );
	//CPU_FUNC_CALL( MultMatVec_CPU_AndUnrolling4_DoubleUnrolling2 );
	//CPU_FUNC_CALL( MultMatVec_CPU_AndUnrolling4_DoubleUnrolling4 );
	CPU_FUNC_CALL( MultMatVec_CPU_PrefetchAndUnrolling4_DoubleUnrolling4 );
	//CPU_FUNC_CALL( MultMatVec_CPU_AndUnrolling16 );
	//CPU_FUNC_CALL( MultMatVec_CPU_AndUnrolling32 );


	size_t numThreads = ( 1 << 10 );
	size_t numBlocks = numOfVectors / numThreads;
	
	size_t _32Threads = ( 1 << 5 );
	size_t _32Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _32Threads;
	size_t _32Blocks_perVector = numOfVectors / _32Threads;
	size_t _32Blocks_perVector32threads = _32Blocks_perElement;

	size_t _64Threads = ( 1 << 6 );
	size_t _64Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _64Threads;
	size_t _64Blocks_perVector = numOfVectors / _64Threads;
	size_t _64Blocks_perVector32threads = _64Blocks_perElement;

	size_t _128Threads = ( 1 << 7 );
	size_t _128Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _128Threads;
	size_t _128Blocks_perVector = numOfVectors / _128Threads;
	size_t _128Blocks_perVector32threads = _128Blocks_perElement;

	size_t _256Threads = ( 1 << 8 );
	size_t _256Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _256Threads;
	size_t _256Blocks_perVector = numOfVectors / _256Threads;
	size_t _256Blocks_perVector32threads = _256Blocks_perElement;

	size_t _512Threads = ( 1 << 9 );
	size_t _512Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _512Threads;
	size_t _512Blocks_perVector = numOfVectors / _512Threads;
	size_t _512Blocks_perVector32threads = _512Blocks_perElement;

	size_t _1024Threads = ( 1 << 10 );
	size_t _1024Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _1024Threads;
	size_t _1024Blocks_perVector = numOfVectors / _1024Threads;
	size_t _1024Blocks_perVector32threads = _1024Blocks_perElement;

	GenerateConstantMatrix( matA );

	CUDA_CALL( cudaMalloc( &vecXgpu, numOfVectorElems * sizeof( float ) ) );
	CUDA_CALL( cudaMalloc( &vecYgpu, numOfVectorElems * sizeof( float ) ) );
	CUDA_CALL( cudaMalloc( &matAgpu, numOfMatrixElems * sizeof( float ) ) );
	printf( "Finish GPU memory allocation and copy datas from host to device\n\n" );
	CUDA_CALL( cudaMemcpy( vecXgpu, vecX, numOfVectorElems * sizeof( float ), cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy( matAgpu, matA, numOfMatrixElems * sizeof( float ), cudaMemcpyHostToDevice ) );

#define GPU_FUNC_CALL_SHARED(funcname, num_block, num_thread, size_shared) \
	totalGPUtime = 0; \
	CUDA_CALL( cudaMemset( vecYgpu, 0x00, numOfVectorElems * sizeof( float ) ) ); \
	for( int i = 0; i < REPEAT_COUNT; ++i ) \
	{ \
		CHECK_TIME_START_GPU(); \
		funcname <<< num_block, num_thread, size_shared >>> ( vecYgpu, matAgpu, vecXgpu ); \
		cuda_error_check( "ERROR: ", " when " #funcname "() was launched.\n" ); \
		CHECK_TIME_END_GPU( gpuTime ); \
		totalGPUtime += gpuTime; \
	} \
	printf( "Finish " #funcname "<<< %d, %d >>> calculation\n", num_block, num_thread ); \
	if( size_shared != 0 ) printf( " - Shared memory size: %d bytes\n", size_shared ); \
	printf( " - Elapsed time: %f\n", totalGPUtime / REPEAT_COUNT ); \
	CUDA_CALL( cudaMemcpy( vecYgpuResult, vecYgpu, numOfVectorElems * sizeof( float ), cudaMemcpyDeviceToHost ) ); \
	CUDA_CALL( cudaDeviceSynchronize() ); \
	printf( " - Error rate: %.2f%%\n\n", GetErrorRate( vecYcpuResult, vecYgpuResult, numOfVectorElems ) );
#define GPU_FUNC_CALL_SHARED__MACRO_END

#define GPU_FUNC_CALL(funcname, num_block, num_thread) GPU_FUNC_CALL_SHARED(funcname, num_block, num_thread, 0)

	GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemoryWithoutRegister_Vector,									_1024Blocks_perVector,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemory_Vector,													_1024Blocks_perVector,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_SimpleConstantMatrix_Vector,											_1024Blocks_perVector,			_1024Threads );

	GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemoryWithoutRegister_Element32ThreadsPerBlock,				_32Blocks_perElement,			_32Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemory_Element32ThreadsPerBlock,								_32Blocks_perElement,			_32Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemory_Element1024ThreadsPerBlock,								_1024Blocks_perElement,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_SimpleConstantMatrix_Element1024ThreadsPerBlock,						_1024Blocks_perElement,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemoryConstantMatrix_Element1024ThreadsPerBlock,				_1024Blocks_perElement,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemory_Element1024ThreadsPerBlock,								_1024Blocks_perElement,			_1024Threads );

	GPU_FUNC_CALL(			MultMatVec_GPU_StridedGlobalMemory_Element1024ThreadsPerBlock,						_1024Blocks_perElement,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_StridedConstantMatrix_Element1024ThreadsPerBlock,					_1024Blocks_perElement,			_1024Threads );

	GPU_FUNC_CALL(			MultMatVec_GPU_Strided32VectorGlobalMemory_Element1024ThreadsPerBlock,				_1024Blocks_perElement,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_Strided32VectorConstantMatrix_Element1024ThreadsPerBlock,			_1024Blocks_perElement,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_Strided32VectorSharedMemoryConstantMatrix_Element1024ThreadsPerBlock,_1024Blocks_perElement,			_1024Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_Strided32VectorSharedMemory_Element1024ThreadsPerBlock,				_1024Blocks_perElement,			_1024Threads );

	// Wrong case and modified version
	GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemoryWithAtomic_Vector,										_1024Blocks_perVector,			_1024Threads );	// 96.88% error
	GPU_FUNC_CALL(			MultMatVec_GPU_SimpleConstantMatrixWithAtomic_1VectorWith32Threads,					_1024Blocks_perVector,			_1024Threads ); // 96.88% error
	GPU_FUNC_CALL(			MultMatVec_GPU_SimpleConstantMatrixWithAtomic_1VectorWith32Threads,	/* Correct */	_128Blocks_perVector32threads,	_128Threads );

	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemoryWithAtomic_1VectorWith32Threads,							_128Blocks_perVector32threads,	_128Threads );
	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemoryConstantMatrixWithAtomic_1VectorWith32Threads,			_128Blocks_perVector32threads,	_128Threads );
	
	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemoryConstantMatrix_1VectorWith32Threads,						_128Blocks_perVector32threads,	_128Threads );
	// Verifying
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_1VectorWith32Threads,				_64Blocks_perVector32threads,	_64Threads,		64 * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_1VectorWith32Threads,				_128Blocks_perVector32threads,	_128Threads,	128 * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_1VectorWith32Threads,				_256Blocks_perVector32threads,	_256Threads,	256 * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_1VectorWith32Threads,				_512Blocks_perVector32threads,	_512Threads,	512 * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_1VectorWith32Threads,				_1024Blocks_perVector32threads,	_1024Threads,	1024 * sizeof( float ) );

	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemoryWithAtomic_32VectorWith32Threads,						_512Blocks_perVector,			_512Threads );
	// Verifying
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryWithAtomic_32VectorWith32Threads,				_64Blocks_perVector,			_64Threads,		64 / ELEM_PER_VECTOR * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryWithAtomic_32VectorWith32Threads,				_128Blocks_perVector,			_128Threads,	128 / ELEM_PER_VECTOR * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryWithAtomic_32VectorWith32Threads,				_256Blocks_perVector,			_256Threads,	256 / ELEM_PER_VECTOR * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryWithAtomic_32VectorWith32Threads,				_512Blocks_perVector,			_512Threads,	512 / ELEM_PER_VECTOR * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryWithAtomic_32VectorWith32Threads,				_1024Blocks_perVector,			_1024Threads,	1024 / ELEM_PER_VECTOR * sizeof( float ) );

	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemoryConstantMatrix_32VectorWith32Threads,					_128Blocks_perVector,			_128Threads );
	// Verifying
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_32VectorWith32Threads,			_64Blocks_perVector,			_64Threads,		64 * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_32VectorWith32Threads,			_128Blocks_perVector,			_128Threads,	128 * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_32VectorWith32Threads,			_256Blocks_perVector,			_256Threads,	256 * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_32VectorWith32Threads,			_512Blocks_perVector,			_512Threads,	512 * sizeof( float ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemoryConstantMatrix_32VectorWith32Threads,			_1024Blocks_perVector,			_1024Threads,	1024 * sizeof( float ) );

	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemory_32VectorWith32Threads,									_1024Blocks_perVector,			_1024Threads );
	// Verifying
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemory_32VectorWith32Threads,							_64Blocks_perVector,			_64Threads,		sizeof( float ) * ( 1024 + 64 ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemory_32VectorWith32Threads,							_128Blocks_perVector,			_128Threads,	sizeof( float ) * ( 1024 + 128 ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemory_32VectorWith32Threads,							_256Blocks_perVector,			_256Threads,	sizeof( float ) * ( 1024 + 256 ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemory_32VectorWith32Threads,							_512Blocks_perVector,			_512Threads,	sizeof( float ) * ( 1024 + 512 ) );
	GPU_FUNC_CALL_SHARED(	MultMatVec_GPU_VariableSharedMemory_32VectorWith32Threads,							_1024Blocks_perVector,			_1024Threads,	sizeof( float ) * ( 1024 + 1024 ) );

	CHECK_TIME_DEST_GPU();

	CUDA_CALL( cudaFree( vecXgpu ) );
	CUDA_CALL( cudaFree( vecYgpu ) );
	CUDA_CALL( cudaFree( matAgpu ) );

	int cnt = 0;
	float epsilon = 0.000005f;
	for( int i = 0; i < numOfVectorElems; ++i )
	{
		if( absIEEE754( vecYcpuResult[ i ] - vecYgpuResult[ i ] ) > epsilon )
		{
			cnt++;
			//printf( "[%d]: %f != %f\n", i, vecYcpuResult[ i ], vecYgpuResult[ i ] );
		}
	}
	printf( "Finish comparation\n" );
	//printf( " - Error rate: %.2f%%\n", float( cnt ) / numOfVectorElems * 100.f);

	delete[] vecX;
	delete[] vecYcpuResult;
	delete[] vecYgpuResult;
	delete[] matA;
}

void MultMatVec_CPU_WithPrefetch( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	for( int i = 0; i < n; ++i )
	{
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			vecY[ i * ELEM_PER_VECTOR + j ] = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; ++k )
			{
				vecY[ i * ELEM_PER_VECTOR + j ] += matA[ j ][ k ] * vecX[ i * ELEM_PER_VECTOR + k ];
			}
		}
	}
}

void MultMatVec_CPU_( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	for( int i = 0; i < n; ++i )
	{
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; ++k )
			{
				result += matA[ j ][ k ] * vecX[ i * ELEM_PER_VECTOR + k ];
			}
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}

void MultMatVec_CPU_AndPrefetch( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; ++k )
			{
				result += matA[ j ][ k ] * vecX[ i * ELEM_PER_VECTOR + k ];
			}
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}

void MultMatVec_CPU_AndUnrolling2( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; k += 2 )
			{
				result +=
					+ matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}

void MultMatVec_CPU_AndUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; k += 4 )
			{
				result +=
					+ matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}

void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling2( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; j += 2 )
		{
			float result[ 2 ] = { 0.0f, };
			for( int k = 0; k < ELEM_PER_VECTOR; k += 4 )
			{
				result[ 0 ] +=
					+ matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 1 ] +=
					+ matA[ j + 1 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 1 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 1 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 1 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j + 0 ] = result[ 0 ];
			vecY[ i * ELEM_PER_VECTOR + j + 1 ] = result[ 1 ];
		}
	}
}

void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; j += 4 )
		{
			float result[ 4 ] = { 0.0f, };
			for( int k = 0; k < ELEM_PER_VECTOR; k += 4 )
			{
				result[ 0 ] +=
					+ matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 1 ] += 
					+ matA[ j + 1][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 1][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 1][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 1][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 2 ] +=
					 +matA[ j + 2 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 2 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 2 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 2 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 3 ] +=
					+ matA[ j + 3 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 3 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 3 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 3 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j + 0 ] = result[ 0 ];
			vecY[ i * ELEM_PER_VECTOR + j + 1 ] = result[ 1 ];
			vecY[ i * ELEM_PER_VECTOR + j + 2 ] = result[ 2 ];
			vecY[ i * ELEM_PER_VECTOR + j + 3 ] = result[ 3 ];
		}
	}
}

void MultMatVec_CPU_PrefetchAndUnrolling4_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] && vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] &&
		vecX[ i * ELEM_PER_VECTOR ] && vecX[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] == 0.0;
		for( int j = 0; j < ELEM_PER_VECTOR; j += 4 )
		{
			float result[ 4 ] = { 0.0f, };
			matA[ j + 0 ][ HALF_ELEM_PER_VECTOR ] && matA[ j + 1 ][ HALF_ELEM_PER_VECTOR ] &&
			matA[ j + 2 ][ HALF_ELEM_PER_VECTOR ] && matA[ j + 3 ][ HALF_ELEM_PER_VECTOR ] == 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; k += 4 )
			{
				result[ 0 ] +=
					+matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 1 ] +=
					+matA[ j + 1 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 1 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 1 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 1 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 2 ] +=
					+matA[ j + 2 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 2 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 2 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 2 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 3 ] +=
					+matA[ j + 3 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 3 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 3 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 3 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j + 0 ] = result[ 0 ];
			vecY[ i * ELEM_PER_VECTOR + j + 1 ] = result[ 1 ];
			vecY[ i * ELEM_PER_VECTOR + j + 2 ] = result[ 2 ];
			vecY[ i * ELEM_PER_VECTOR + j + 3 ] = result[ 3 ];
		}
	}
}

void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling8( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; j += 8 )
		{
			float result[ 8 ] = { 0.0f, };
			for( int k = 0; k < ELEM_PER_VECTOR; k += 4 )
			{
				result[ 0 ] +=
					+ matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 1 ] +=
					+ matA[ j + 1 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 1 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 1 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 1 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 2 ] +=
					+ matA[ j + 2 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 2 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 2 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 2 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 3 ] +=
					+ matA[ j + 3 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 3 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 3 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 3 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 4 ] +=
					+ matA[ j + 4 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 4 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 4 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 4 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 5 ] +=
					+ matA[ j + 5 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 5 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 5 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 5 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 6 ] +=
					+ matA[ j + 6 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 6 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 6 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 6 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];

				result[ 7 ] +=
					+ matA[ j + 7 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 7 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 7 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 7 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j + 0 ] = result[ 0 ];
			vecY[ i * ELEM_PER_VECTOR + j + 1 ] = result[ 1 ];
			vecY[ i * ELEM_PER_VECTOR + j + 2 ] = result[ 2 ];
			vecY[ i * ELEM_PER_VECTOR + j + 3 ] = result[ 3 ];
			vecY[ i * ELEM_PER_VECTOR + j + 4 ] = result[ 4 ];
			vecY[ i * ELEM_PER_VECTOR + j + 5 ] = result[ 5 ];
			vecY[ i * ELEM_PER_VECTOR + j + 6 ] = result[ 6 ];
			vecY[ i * ELEM_PER_VECTOR + j + 7 ] = result[ 7 ];
		}
	}
}

void MultMatVec_CPU_AndUnrolling8( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; k += 8 )
			{
				result +=
					+ matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ]
					+ matA[ j ][ k + 4 ] * vecX[ i * ELEM_PER_VECTOR + k + 4 ]
					+ matA[ j ][ k + 5 ] * vecX[ i * ELEM_PER_VECTOR + k + 5 ]
					+ matA[ j ][ k + 6 ] * vecX[ i * ELEM_PER_VECTOR + k + 6 ]
					+ matA[ j ][ k + 7 ] * vecX[ i * ELEM_PER_VECTOR + k + 7 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}

void MultMatVec_CPU_AndUnrolling8_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; j += 4 )
		{
			float result[ 4 ] = { 0.0f, };
			for( int k = 0; k < ELEM_PER_VECTOR; k += 8 )
			{
				result[ 0 ] +=
					+ matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ]
					+ matA[ j ][ k + 4 ] * vecX[ i * ELEM_PER_VECTOR + k + 4 ]
					+ matA[ j ][ k + 5 ] * vecX[ i * ELEM_PER_VECTOR + k + 5 ]
					+ matA[ j ][ k + 6 ] * vecX[ i * ELEM_PER_VECTOR + k + 6 ]
					+ matA[ j ][ k + 7 ] * vecX[ i * ELEM_PER_VECTOR + k + 7 ];

				result[ 1 ] +=
					+ matA[ j + 1 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 1 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 1 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 1 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ]
					+ matA[ j + 1 ][ k + 4 ] * vecX[ i * ELEM_PER_VECTOR + k + 4 ]
					+ matA[ j + 1 ][ k + 5 ] * vecX[ i * ELEM_PER_VECTOR + k + 5 ]
					+ matA[ j + 1 ][ k + 6 ] * vecX[ i * ELEM_PER_VECTOR + k + 6 ]
					+ matA[ j + 1 ][ k + 7 ] * vecX[ i * ELEM_PER_VECTOR + k + 7 ];

				result[ 2 ] +=
					+ matA[ j + 2 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 2 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 2 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 2 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ]
					+ matA[ j + 2 ][ k + 4 ] * vecX[ i * ELEM_PER_VECTOR + k + 4 ]
					+ matA[ j + 2 ][ k + 5 ] * vecX[ i * ELEM_PER_VECTOR + k + 5 ]
					+ matA[ j + 2 ][ k + 6 ] * vecX[ i * ELEM_PER_VECTOR + k + 6 ]
					+ matA[ j + 2 ][ k + 7 ] * vecX[ i * ELEM_PER_VECTOR + k + 7 ];

				result[ 3 ] +=
					+ matA[ j + 3 ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j + 3 ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j + 3 ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j + 3 ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ]
					+ matA[ j + 3 ][ k + 4 ] * vecX[ i * ELEM_PER_VECTOR + k + 4 ]
					+ matA[ j + 3 ][ k + 5 ] * vecX[ i * ELEM_PER_VECTOR + k + 5 ]
					+ matA[ j + 3 ][ k + 6 ] * vecX[ i * ELEM_PER_VECTOR + k + 6 ]
					+ matA[ j + 3 ][ k + 7 ] * vecX[ i * ELEM_PER_VECTOR + k + 7 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j + 0 ] = result[ 0 ];
			vecY[ i * ELEM_PER_VECTOR + j + 1 ] = result[ 1 ];
			vecY[ i * ELEM_PER_VECTOR + j + 2 ] = result[ 2 ];
			vecY[ i * ELEM_PER_VECTOR + j + 3 ] = result[ 3 ];
		}
	}
}

void MultMatVec_CPU_AndUnrolling16( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; k += 16 )
			{
				result +=
					+ matA[ j ][ k + 0  ] * vecX[ i * ELEM_PER_VECTOR + k + 0  ]
					+ matA[ j ][ k + 1  ] * vecX[ i * ELEM_PER_VECTOR + k + 1  ]
					+ matA[ j ][ k + 2  ] * vecX[ i * ELEM_PER_VECTOR + k + 2  ]
					+ matA[ j ][ k + 3  ] * vecX[ i * ELEM_PER_VECTOR + k + 3  ]
					+ matA[ j ][ k + 4  ] * vecX[ i * ELEM_PER_VECTOR + k + 4  ]
					+ matA[ j ][ k + 5  ] * vecX[ i * ELEM_PER_VECTOR + k + 5  ]
					+ matA[ j ][ k + 6  ] * vecX[ i * ELEM_PER_VECTOR + k + 6  ]
					+ matA[ j ][ k + 7  ] * vecX[ i * ELEM_PER_VECTOR + k + 7  ]
					+ matA[ j ][ k + 8  ] * vecX[ i * ELEM_PER_VECTOR + k + 8  ]
					+ matA[ j ][ k + 9  ] * vecX[ i * ELEM_PER_VECTOR + k + 9  ]
					+ matA[ j ][ k + 10 ] * vecX[ i * ELEM_PER_VECTOR + k + 10 ]
					+ matA[ j ][ k + 11 ] * vecX[ i * ELEM_PER_VECTOR + k + 11 ]
					+ matA[ j ][ k + 12 ] * vecX[ i * ELEM_PER_VECTOR + k + 12 ]
					+ matA[ j ][ k + 13 ] * vecX[ i * ELEM_PER_VECTOR + k + 13 ]
					+ matA[ j ][ k + 14 ] * vecX[ i * ELEM_PER_VECTOR + k + 14 ]
					+ matA[ j ][ k + 15 ] * vecX[ i * ELEM_PER_VECTOR + k + 15 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}

void MultMatVec_CPU_AndUnrolling32( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			
				result +=
					+ matA[ j ][ 0  ] * vecX[ i * ELEM_PER_VECTOR + 0  ]
					+ matA[ j ][ 1  ] * vecX[ i * ELEM_PER_VECTOR + 1  ]
					+ matA[ j ][ 2  ] * vecX[ i * ELEM_PER_VECTOR + 2  ]
					+ matA[ j ][ 3  ] * vecX[ i * ELEM_PER_VECTOR + 3  ]
					+ matA[ j ][ 4  ] * vecX[ i * ELEM_PER_VECTOR + 4  ]
					+ matA[ j ][ 5  ] * vecX[ i * ELEM_PER_VECTOR + 5  ]
					+ matA[ j ][ 6  ] * vecX[ i * ELEM_PER_VECTOR + 6  ]
					+ matA[ j ][ 7  ] * vecX[ i * ELEM_PER_VECTOR + 7  ]
					+ matA[ j ][ 8  ] * vecX[ i * ELEM_PER_VECTOR + 8  ]
					+ matA[ j ][ 9  ] * vecX[ i * ELEM_PER_VECTOR + 9  ]
					+ matA[ j ][ 10 ] * vecX[ i * ELEM_PER_VECTOR + 10 ]
					+ matA[ j ][ 11 ] * vecX[ i * ELEM_PER_VECTOR + 11 ]
					+ matA[ j ][ 12 ] * vecX[ i * ELEM_PER_VECTOR + 12 ]
					+ matA[ j ][ 13 ] * vecX[ i * ELEM_PER_VECTOR + 13 ]
					+ matA[ j ][ 14 ] * vecX[ i * ELEM_PER_VECTOR + 14 ]
					+ matA[ j ][ 15 ] * vecX[ i * ELEM_PER_VECTOR + 15 ]
					+ matA[ j ][ 16 ] * vecX[ i * ELEM_PER_VECTOR + 16 ]
					+ matA[ j ][ 17 ] * vecX[ i * ELEM_PER_VECTOR + 17 ]
					+ matA[ j ][ 18 ] * vecX[ i * ELEM_PER_VECTOR + 18 ]
					+ matA[ j ][ 19 ] * vecX[ i * ELEM_PER_VECTOR + 19 ]
					+ matA[ j ][ 20 ] * vecX[ i * ELEM_PER_VECTOR + 20 ]
					+ matA[ j ][ 21 ] * vecX[ i * ELEM_PER_VECTOR + 21 ]
					+ matA[ j ][ 22 ] * vecX[ i * ELEM_PER_VECTOR + 22 ]
					+ matA[ j ][ 23 ] * vecX[ i * ELEM_PER_VECTOR + 23 ]
					+ matA[ j ][ 24 ] * vecX[ i * ELEM_PER_VECTOR + 24 ]
					+ matA[ j ][ 25 ] * vecX[ i * ELEM_PER_VECTOR + 25 ]
					+ matA[ j ][ 26 ] * vecX[ i * ELEM_PER_VECTOR + 26 ]
					+ matA[ j ][ 27 ] * vecX[ i * ELEM_PER_VECTOR + 27 ]
					+ matA[ j ][ 28 ] * vecX[ i * ELEM_PER_VECTOR + 28 ]
					+ matA[ j ][ 29 ] * vecX[ i * ELEM_PER_VECTOR + 29 ]
					+ matA[ j ][ 30 ] * vecX[ i * ELEM_PER_VECTOR + 30 ]
					+ matA[ j ][ 31 ] * vecX[ i * ELEM_PER_VECTOR + 31 ];
			
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}