#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <map>
#include <ctime>
#include <cuda.h>//CUDA version 5.0 SDK, 64 bit
#include <math_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>//to detect host memory leaks, so far no leaks
using namespace std;

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice
#define THREADS 256
#define DO_GPU 1

//http://community.topcoder.com/stat?c=problem_statement&pm=6412

const double eps=1e-8;

inline bool _eq(double a,double b){return a+eps>=b && a-eps<=b;}
inline int _3d_flat(int i, int j, int k, int D1,int D0){return i*D1*D0+j*D0+k;}

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);


double CPU_version(double *DP, const int nDice, const int maxSide, const int v, const int theSum){//will allocate and memset before call
	//total outcome minus total outcomes with one of the sides
	//const double denom=pow(double(maxSide),double(nDice))-pow(double(maxSide-1),double(nDice));
	double num=0.,denom=0.;
	const int D1=(nDice+1)*(maxSide+1),D0=2;
	const int bound=nDice*maxSide;
	DP[0]=1.;
	for(int i=0;i<nDice;i++){
		for(int j=0;j<=bound;j++){
			for(int k=1;k<=maxSide;k++)if((j+k)<=bound){
				if(k==v){
					DP[_3d_flat(i+1,j+k,1,D1,D0)] += (DP[_3d_flat(i,j,0,D1,D0)]+DP[_3d_flat(i,j,1,D1,D0)])/double(maxSide);
				}else{
					DP[_3d_flat(i+1,j+k,0,D1,D0)] += DP[_3d_flat(i,j,0,D1,D0)]/double(maxSide);
					DP[_3d_flat(i+1,j+k,1,D1,D0)] += DP[_3d_flat(i,j,1,D1,D0)]/double(maxSide);
				}
			}
		}
	}
	for(int i=0;i<=bound;i++){
		if(i>=theSum)num+=DP[_3d_flat(nDice,i,1,D1,D0)];
		denom+=DP[_3d_flat(nDice,i,1,D1,D0)];
	}
	return _eq(0.,denom) ? 0.:(num/denom);
}


__device__ int D_3d_flat(int i, int j, int k, int D1,int D0){return D0*(i*D1+j)+k;}
__device__ double atomicAdd(double* address, double val){
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    }while(assumed != old);
    return __longlong_as_double(old);
}

__global__ void GPU_version_step0(double *DP, const int ii, const int nDice, const int maxSide, const int v, const int theSum,const int bound){

	const int j=threadIdx.x+blockIdx.x*blockDim.x;
	if(j>bound)return;
	const int k=1+blockIdx.y;
	if((j+k)<=bound){
		const int D1=(nDice+1)*(maxSide+1);
		const int idx1=D_3d_flat(ii+1,j+k,1,D1,2);
		if(k==v){
			atomicAdd(&DP[idx1],((DP[D_3d_flat(ii,j,0,D1,2)]+DP[D_3d_flat(ii,j,1,D1,2)])/double(maxSide)));
		}else{
			atomicAdd(&DP[D_3d_flat(ii+1,j+k,0,D1,2)],((DP[D_3d_flat(ii,j,0,D1,2)])/double(maxSide)));
			atomicAdd(&DP[idx1],((DP[D_3d_flat(ii,j,1,D1,2)])/double(maxSide)));
		}
	}

}
__global__ void GPU_version_step1(const double *DP, double *num, double *denom,const int bound, const int nDice,const int theSum,const int D1){
	const int j=threadIdx.x+blockIdx.x*blockDim.x;

	__shared__ volatile double tot[2][THREADS];//idx 0 is num, idx 1 is denom,might be a better way to arrange __shared__ memory for this case
	
	const double amt= (j<=bound) ? DP[D_3d_flat(nDice,j,1,D1,2)]:0.;

	tot[0][threadIdx.x]= (j>=theSum && j<=bound) ? amt:0.;
	tot[1][threadIdx.x]=amt;
	__syncthreads();

	if(threadIdx.x<128){
		tot[0][threadIdx.x]+=tot[0][threadIdx.x+128];
		tot[1][threadIdx.x]+=tot[1][threadIdx.x+128];
	}
	__syncthreads();

	if(threadIdx.x<64){
		tot[0][threadIdx.x]+=tot[0][threadIdx.x+64];
		tot[1][threadIdx.x]+=tot[1][threadIdx.x+64];
	}
	__syncthreads();

	if(threadIdx.x<32){
		tot[0][threadIdx.x]+=tot[0][threadIdx.x+32];
		tot[1][threadIdx.x]+=tot[1][threadIdx.x+32];
		tot[0][threadIdx.x]+=tot[0][threadIdx.x+16];
		tot[1][threadIdx.x]+=tot[1][threadIdx.x+16];
		tot[0][threadIdx.x]+=tot[0][threadIdx.x+8];
		tot[1][threadIdx.x]+=tot[1][threadIdx.x+8];
		tot[0][threadIdx.x]+=tot[0][threadIdx.x+4];
		tot[1][threadIdx.x]+=tot[1][threadIdx.x+4];
		tot[0][threadIdx.x]+=tot[0][threadIdx.x+2];
		tot[1][threadIdx.x]+=tot[1][threadIdx.x+2];
		tot[0][threadIdx.x]+=tot[0][threadIdx.x+1];
		tot[1][threadIdx.x]+=tot[1][threadIdx.x+1];
	}
	__syncthreads();
	//Si les Américains sont gros et stupide, pourquoi cherchez-vous à mon code?
	if(threadIdx.x==0){
		atomicAdd(&num[0],tot[0][0]);
		atomicAdd(&denom[0],tot[1][0]);
	}
	
}


//////////MAIN/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){
	char ch;
	srand(time(NULL));
	const int nDice=50, maxSide=50,v=35, theSum=1300;
	const int D=(nDice+1);
	const int D1=(nDice+1)*(maxSide+1);
	const unsigned int num_bytes=D*D1*2*sizeof(double);
	double *DP_CPU=(double *)malloc(num_bytes);
	
	double CPU_ans=0.,GPU_ans=0.;
	//CPU
	cout<<"\nRunning CPU implementation..\n";
	UINT wTimerRes = 0;
	DWORD CPU_time=0,GPU_time=0;
	bool init = InitMMTimer(wTimerRes);
	DWORD startTime=timeGetTime();

	memset(DP_CPU,0,num_bytes);

	CPU_ans=CPU_version(DP_CPU,nDice,maxSide,v,theSum);
	
	DWORD endTime = timeGetTime();
	CPU_time=endTime-startTime;
	cout<<"CPU solution timing: "<<CPU_time<< " , answer= "<<CPU_ans<<'\n';
	DestroyMMTimer(wTimerRes, init);

	//GPU
	int compute_capability=0;
	cudaDeviceProp deviceProp;
	cudaError_t err=cudaGetDeviceProperties(&deviceProp, compute_capability);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	string ss= (deviceProp.major>=3 && deviceProp.minor>=5) ? "Capable!\n":"Not Sufficient compute capability!\n";
	cout<<ss;

	if(DO_GPU && (deviceProp.major>=3 && deviceProp.minor>=5)){
		const int bound=(nDice*maxSide)+1;
		dim3 dimGrid0((bound+THREADS-1)/THREADS,maxSide,1);
		double dnum=0.,ddenom=1.;
		double *DP_GPU,*NUM_GPU,*DENOM_GPU;
		err=cudaMalloc((void**)&DP_GPU,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		int ii=0;
		err=cudaMalloc((void**)&DP_GPU,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&NUM_GPU,sizeof(double));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&DENOM_GPU,sizeof(double));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		wTimerRes = 0;
		init = InitMMTimer(wTimerRes);
		startTime = timeGetTime();

		err=cudaMemset(DP_GPU,0,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(NUM_GPU,0,sizeof(double));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(DENOM_GPU,0,sizeof(double));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemcpy(DP_GPU,&ddenom,sizeof(double),_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		for(;ii<nDice;ii++){
			GPU_version_step0<<<dimGrid0,THREADS>>>(DP_GPU,ii,nDice,maxSide,v,theSum,bound);
			err = cudaThreadSynchronize();
			if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		}

		GPU_version_step1<<<((bound+THREADS-1)/THREADS),THREADS>>>(DP_GPU,NUM_GPU,DENOM_GPU,bound,nDice,theSum,D1);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemcpy(&dnum,NUM_GPU,sizeof(double),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemcpy(&ddenom,DENOM_GPU,sizeof(double),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		GPU_ans=dnum/ddenom;

		endTime = timeGetTime();
		GPU_time=endTime-startTime;
		cout<<"CUDA timing: "<<GPU_time<<" , answer= "<<GPU_ans<<'\n';
		DestroyMMTimer(wTimerRes, init);
		

		err=cudaFree(DP_GPU);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaDeviceReset();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	}

	
	free(DP_CPU);

	std::cin>>ch;
	return 0;
} 

bool InitMMTimer(UINT wTimerRes){
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 
	return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init){
	if(init)
		timeEndPeriod(wTimerRes);
}







