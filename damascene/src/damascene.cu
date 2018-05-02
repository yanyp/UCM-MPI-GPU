// vim: ts=4 syntax=cpp comments=

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_image.h>
#include <fcntl.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>
#include "texton.h"
#include "convert.h"
#include "intervening.h"
#include "lanczos.h"
#include "stencilMVM.h"

#include "localcues.h"
#include "combine.h"
#include "nonmax.h"
#include "spectralPb.h"
#include "globalPb.h"
#include "skeleton.h"
#include "log.h"
#include "exception.h"

#define __TIMER_SPECFIC

#define TEXTON64 2
#define TEXTON32 1

float* loadArray(char* filename, uint& width, uint& height) {
  FILE* fp;
  fp = fopen(filename, "r");
  int dim;
  fread(&dim, sizeof(int), 1, fp);
  assert(dim == 2);
  fread(&width, sizeof(int), 1, fp);
  fread(&height, sizeof(int), 1, fp);
  float* buffer = (float*)malloc(sizeof(float) * width * height);
  int counter = 0;
  for(int col = 0; col < width; col++) {
    for(int row = 0; row < height; row++) {
      float element;
      fread(&element, sizeof(float), 1, fp);
      counter++;
      buffer[row * width + col] = element;
    }
  }
 /*  for(int row = 0; row < height; row++) { */
/*     for(int col = 0; col < width; col++) { */
/*       printf("%f ", buffer[row*width + col]); */
/*     } */
/*     printf("\n"); */
/*   } */
  return buffer;
}

void writeTextImage(const char* filename, uint width, uint height, float* image) {
  FILE* fp = fopen(filename, "w");
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      fprintf(fp, "%f ", image[row * width + col]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void writeFile(char* file, int width, int height, int* input)
{
    int fd;
    float* pb = (float*)malloc(sizeof(float)*width*height);
    for(int i = 0; i < width * height; i++) {
      pb[i] = (float)input[i];
    }
    fd = open(file, O_CREAT|O_WRONLY, 0666);
    write(fd, &width, sizeof(int));
    write(fd, &height, sizeof(int));
    write(fd, pb, width*height*sizeof(float));
    close(fd);
}

void writeFile(char* file, int width, int height, float* pb)
{
    int fd;

    fd = open(file, O_CREAT|O_WRONLY, 0666);
    write(fd, &width, sizeof(int));
    write(fd, &height, sizeof(int));
    write(fd, pb, width*height*sizeof(float));
    close(fd);
}

void writeGradients(char* file, int width, int height, int pitchInFloats, int norients, int scales, float* pb)
{
    int fd;

    fd = open(file, O_CREAT|O_WRONLY, 0666);
    write(fd, &width, sizeof(int));
    write(fd, &height, sizeof(int));
    write(fd, &norients, sizeof(int));
    write(fd, &scales, sizeof(int));
    for(int scale = 0; scale < scales; scale++) {
      for(int orient = 0; orient < norients; orient++) {
        float* currentPointer = &pb[pitchInFloats * orient + pitchInFloats * scale * norients];
        write(fd, currentPointer, width*height*sizeof(float));
      }
    }
    close(fd);
}

void writeArray(char* file, int ndim, int* dim, float* input) {
  int fd;
  fd = open(file, O_CREAT|O_WRONLY|O_TRUNC, 0666);
  int size = 1;
  for(int i = 0; i < ndim; i++) {
    size *= dim[i];
  }
  write(fd, &ndim, sizeof(int));
  write(fd, dim, sizeof(int) * ndim);
  write(fd, input, sizeof(float) * size);
  close(fd);
}

void transpose(int width, int height, float* input, float* output) {
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      output[col * height + row] = input[row * width + col];
      // output[row * width + col] = input[row * width + col];
    }
  }                                         
}

void checkInputValue(int& nEigNum, float& fEigTolerance, int& nTextonChoice)
{
	if (nEigNum > 25)
	{
		//printf("\nException: Do not support for more than 25 eigen vectors.\n");
		log_error("Exception: Do not support for more than 25 eigen vectors");
		nEigNum = 25;
	}
	if (nEigNum < 2)
	{
		//printf("\nException: Do not support for less than 2 eigen vectors.\n");
		log_error("Exception: Do not support for less than 2 eigen vectors");
		nEigNum = 9;
	}
	if  (fEigTolerance < 1e-5)
	{
		//printf("\nException: Do not support for accuracy below 1e-5.\n");
		log_error("Exception: Do not support for accuracy below 1e-5");
		fEigTolerance = 1e-4;
	}
	if  (fEigTolerance > 1e-1)
	{
		//printf("\nException: Do not support for accuracy above 1e-1.\n");
		log_error("Exception: Do not support for accuracy above 1e-1");
		fEigTolerance = 1e-3;
	}
	if (nTextonChoice > 2 || nTextonChoice < 1)
	{
		//printf("\nException: Only support choice 1 (32 bins) and choice 2 (64 bins)\n");
		log_error("Exception: Only support choice 1 (32 bins) and choice 2 (64 bins)");
	}
}

void parsingCommand(int argc, char** argv, int& nEigNum, float& fEigTolerance, int& nTextonChoice)
{
	if (argc < 3)
	{
		nEigNum = 9;
		fEigTolerance = 1e-3;
		nTextonChoice = TEXTON32;
		return;
	}
	if (argc < 4)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = 1e-3;
		nTextonChoice = TEXTON32;
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}
	if (argc < 5)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = atof(argv[3]);
		nTextonChoice = TEXTON32;
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}

	if (argc < 6)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = atof(argv[3]);
		nTextonChoice = atoi(argv[4]);
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}

}

/*
int main(int argc, char** argv) {
  char* filename = argv[1];
  unsigned int* data;
  uint width;
  uint height;
  sdkLoadPPM4ub(filename, (unsigned char**) &data, &width, &height);

  float* hostGPb;
  float* hostGPbAllConcat;
  srand(time(NULL));
  computeGPb(rand() % 2, width, height, data, &hostGPb, &hostGPbAllConcat);

  float *p;
  char *savename;
  p = (float*) malloc(width * height * sizeof(int));
  savename = (char*) malloc(255 * sizeof(char));
  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        // p[i*width + j] = hostGPbAllConcat[k*height*width + i*width + j];
        p[i*width + j] = hostGPbAllConcat[k*height*width + j*height + i];
      }
    }
    sprintf(savename, "%s_%d.pgm", filename, k);
    sdkSavePGM(savename, p, width, height);
  }
  sprintf(savename, "%s.pgm", filename);
  sdkSavePGM(savename, hostGPb, width, height);
  free(p);
  free(savename);

  free(hostGPb);
  free(hostGPbAllConcat);
}
*/

int getCudaDeviceCount() {
  cuInit(0);
  int cudaDeviceCount;
  cudaGetDeviceCount(&cudaDeviceCount);
  return cudaDeviceCount;
}

void computeGPb(uint rank, uint width, uint height, unsigned int* data, float** hostGPb, float** hostGPbAllConcat) {
  char file_name[20];
  sprintf(file_name, "damascene_%d.log", getpid());
  FILE *fp = fopen(file_name, "a");
  log_set_fp(fp);
  char* env_v = getenv("VERBOSE");
  if (env_v == NULL || strcmp(env_v, "0") == 0) {
    log_set_quiet(1);
  }

  cuInit(0);
  int cudaDeviceCount;
  cudaGetDeviceCount(&cudaDeviceCount);
  int cudaDevice = 0;
  struct cudaDeviceProp dp;
  cudaDevice = rank % cudaDeviceCount;
  cudaGetDeviceProperties(&dp, cudaDevice);
  //printf("Using cuda device %i: %s\n", cudaDevice, dp.name);
  log_info("Using cuda device %i: %s", cudaDevice, dp.name);
  cudaSetDevice(cudaDevice);

/*
  if (argc < 2) {
	printf("\nUsage: damascene input_image.ppm eigenvector_num eigenvector_tolerance texton_choice");
	printf("\nInput image should be in ppm format");
	printf("\nThe number of eigenvectors is from 2 to 25");
	printf("\nThe eigenvector tolerance is from 1e-2 to 1e-5");
	printf("\nFor the texton choice parameter, 1 for 32 bins, 2 for 64 bins\n");
    exit(1);
  }



  char* filename = argv[1];
  char outputPGMfilename[1000];
  char outputthinPGMfilename[1000];
  char outputPBfilename[1000];
  char outputthinPBfilename[1000];
  char outputgpbAllfilename[1000];
  printf("Processing: %s, output in ", filename);
  char* period = strrchr(filename, '.');
  if (period == 0) {
    period = strrchr(filename, 0);
  }
  strncpy(outputPGMfilename, filename, period - filename);
  sprintf(&outputPGMfilename[0] + (period - filename) , "Pb.pgm");
  strncpy(outputthinPGMfilename, filename, period - filename);
  sprintf(&outputthinPGMfilename[0] + (period - filename) , "Pbthin.pgm");
  
  strncpy(outputPBfilename, filename, period - filename);
  sprintf(&outputPBfilename[0] + (period - filename), ".pb");
  strncpy(outputthinPBfilename, filename, period - filename);
  sprintf(&outputthinPBfilename[0] + (period - filename), ".thin.pb");
  
  printf("%s and %s\n", outputPGMfilename, outputPBfilename);
  strncpy(outputgpbAllfilename, filename, period - filename);
  sprintf(&outputgpbAllfilename[0] + (period - filename), "GpbAll.ary");
*/
  int nEigNum = 9;
  float fEigTolerance = 1e-3;
  int nTextonChoice = TEXTON32;
/*
  parsingCommand(argc, argv, nEigNum, fEigTolerance, nTextonChoice);
  printf("\nEig %d Tol %f Texton %d\n", nEigNum, fEigTolerance, nTextonChoice);
*/
  
  uint imageSize = sizeof(uint) * width * height;
  uint* devRgbU;
  cudaMalloc((void**) &devRgbU, imageSize);
  cudaMemcpy(devRgbU, data, imageSize, cudaMemcpyHostToDevice);
  int nPixels = width * height;
  //printf("Image found: %i x %i pixels\n", width, height);
  log_info("Image found: %i x %i pixels", width, height);
  assert(width > 0);
  assert(height > 0);
  StopWatchInterface *timer=NULL;
#ifdef __TIMER_SPECFIC
  StopWatchInterface *timer_specific=NULL;
#endif

  size_t totalMemory, availableMemory;
  cuMemGetInfo(&availableMemory,&totalMemory );
  //printf("Available %zu bytes on GPU\n", availableMemory);
  log_info("Available %zu bytes on GPU", availableMemory);

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
 
#ifdef __TIMER_SPECFIC
  sdkCreateTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  float* devGreyscale;
  rgbUtoGreyF(width, height, devRgbU, &devGreyscale);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< rgbUtoGrayF | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< rgbUtoGrayF | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

//   float* hostG = (float*)malloc(sizeof(float) * nPixels); 
//   customCheckCudaErrors(cudaMemcpy(hostG, devGreyscale, height*width*sizeof(float),cudaMemcpyDeviceToHost));
//   cutSavePGMf("grey.pgm", hostG, width, height);
//   free(hostG);

  int* devTextons;
  findTextons(width, height, devGreyscale, &devTextons, nTextonChoice);
/*   int* hostTextons = (int*)malloc(sizeof(int)*width*height); */
/*   cudaMemcpy(hostTextons, devTextons, sizeof(int)*width*height, cudaMemcpyDeviceToHost); */
/*   writeFile("textons.pb", width, height, hostTextons); */

/*   float* hostFTextons = loadArray("goodTextons.dat", width, height); */
/*   printf("Host textons found %i width, %i height\n", width, height); */
/*   int * hostTextons = (int*)malloc(sizeof(float)*width*height); */
/*   for(int i = 0; i < width * height; i++) { */
/*     hostTextons[i] = (float)hostFTextons[i]; */
/*   } */
/*   int* devTextons; */
/*   cudaMalloc((void**)&devTextons, sizeof(int) * width * height); */
/*   cudaMemcpy(devTextons, hostTextons, sizeof(int) * width * height, cudaMemcpyHostToDevice); */
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< texton | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< texton | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  float* devL;
  float* devA;
  float* devB;
  rgbUtoLab3F(width, height, 2.5, devRgbU, &devL, &devA, &devB);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< rgbUtoLab3F | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< rgbUtoLab3F | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  normalizeLab(width, height, devL, devA, devB);
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< normalizeLab | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< normalizeLab | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  int border = 30;
  int borderWidth = width + 2 * border;
  int borderHeight = height + 2 * border;
  float* devLMirrored;
  mirrorImage(width, height, border, devL, &devLMirrored);
/*   float* hostLMirrored = (float*)malloc(borderWidth * borderHeight * sizeof(float)); */
/*   cudaMemcpy(hostLMirrored, devLMirrored, borderWidth * borderHeight * sizeof(float), cudaMemcpyDeviceToHost); */
/*   writeFile("L.pb", borderWidth, borderHeight, hostLMirrored); */
 
  cudaThreadSynchronize();
  cudaFree(devRgbU);
  cudaFree(devGreyscale);
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< mirrorImage | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< mirrorImage | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  float* devBg;
  float* devCga;
  float* devCgb;
  float* devTg;
  int matrixPitchInFloats;
 
 StopWatchInterface *localcuestimer=NULL; 
 sdkCreateTimer(&localcuestimer);
 sdkStartTimer(&localcuestimer);

  localCues(width, height, devL, devA, devB, devTextons, &devBg, &devCga, &devCgb, &devTg, &matrixPitchInFloats, nTextonChoice);

  sdkStopTimer(&localcuestimer);
  //printf("localcues time: %f seconds\n", sdkGetTimerValue(&localcuestimer)/1000.0);
  log_info("localcues time: %f seconds", sdkGetTimerValue(&localcuestimer)/1000.0);
  sdkDeleteTimer(&localcuestimer);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< localcues | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< localcues | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
   //float* hostG = (float*)malloc(sizeof(float) * nPixels); 
   //customCheckCudaErrors(cudaMemcpy(hostG, devBg, height*width*sizeof(float),cudaMemcpyDeviceToHost));
   //cutSavePGMf("Bg.pgm", hostG, width, height);
   //free(hostG);

  cudaFree(devTextons);
  cudaFree(devL);
  cudaFree(devA);
  cudaFree(devB);
  
/*   int size = matrixPitchInFloats * 8 * 3 * sizeof(float); */
/*   float* hostBg = (float*)malloc(size); */
/*   float* hostCga = (float*)malloc(size); */
/*   float* hostCgb = (float*)malloc(size); */
/*   float* hostTg = (float*)malloc(size); */
/*   cudaMemcpy(hostBg, devBg, size, cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(hostCga, devCga, size, cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(hostCgb, devCgb, size, cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(hostTg, devTg, size, cudaMemcpyDeviceToHost); */
/*   writeGradients("bg.gra", width, height, matrixPitchInFloats, 8, 3, hostBg); */
/*   writeGradients("cga.gra", width, height, matrixPitchInFloats, 8, 3, hostCga); */
/*   writeGradients("cgb.gra", width, height, matrixPitchInFloats, 8, 3, hostCgb); */
/*   writeGradients("tg.gra", width, height, matrixPitchInFloats, 8, 3, hostTg); */
  float* devMPbO;
  float *devCombinedGradient;
  combine(width, height, matrixPitchInFloats, devBg, devCga, devCgb, devTg, &devMPbO, &devCombinedGradient, nTextonChoice);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< combine | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< combine | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  customCheckCudaErrors(cudaFree(devBg));
  customCheckCudaErrors(cudaFree(devCga));
  customCheckCudaErrors(cudaFree(devCgb));
  customCheckCudaErrors(cudaFree(devTg));

  float* devMPb;
  cudaMalloc((void**)&devMPb, sizeof(float) * nPixels);
  nonMaxSuppression(width, height, devMPbO, matrixPitchInFloats, devMPb);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< nonmaxsupression | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< nonmaxsupression | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  
  //int devMatrixPitch = matrixPitchInFloats * sizeof(float);
  int radius = 5;
  //int radius = 10;

  Stencil theStencil(radius, width, height, matrixPitchInFloats);
  int nDimension = theStencil.getStencilArea();
  float* devMatrix;
  intervene(theStencil, devMPb, &devMatrix);
  //printf("Intervening contour completed\n");
  log_info("Intervening contour completed");
 
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< intervene | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< intervene | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif

  float* eigenvalues;
  float* devEigenvectors;
  //int nEigNum = 17;
  generalizedEigensolve(theStencil, devMatrix, matrixPitchInFloats, nEigNum, &eigenvalues, &devEigenvectors, fEigTolerance);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< generalizedEigensolve | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< generalizedEigensolve | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  float* devSPb = 0;
  size_t devSPb_pitch = 0;
  customCheckCudaErrors(cudaMallocPitch((void**)&devSPb, &devSPb_pitch, nPixels *  sizeof(float), 8));
  cudaMemset(devSPb, 0, matrixPitchInFloats * sizeof(float) * 8);

  spectralPb(eigenvalues, devEigenvectors, width, height, nEigNum, devSPb, matrixPitchInFloats);

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< spectralPb | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< spectralPb | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
  float* devGPb = 0;
  customCheckCudaErrors(cudaMalloc((void**)&devGPb, sizeof(float) * nPixels));
  float* devGPball = 0;
  customCheckCudaErrors(cudaMalloc((void**)&devGPball, sizeof(float) * matrixPitchInFloats * 8));
  //StartCalcGPb(nPixels, matrixPitchInFloats, 8, devbg1, devbg2, devbg3, devcga1, devcga2, devcga3, devcgb1, devcgb2, devcgb3, devtg1, devtg2, devtg3, devSPb, devMPb, devGPball, devGPb);
  StartCalcGPb(nPixels, matrixPitchInFloats, 8, devCombinedGradient, devSPb, devMPb, devGPball, devGPb);
 
#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< StartCalcGpb | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< StartCalcGpb | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkResetTimer(&timer_specific);
  sdkStartTimer(&timer_specific);
#endif
/*
  float* devGPb_thin = 0;
  customCheckCudaErrors(cudaMalloc((void**)&devGPb_thin, nPixels * sizeof(float) ));
  PostProcess(width, height, width, devGPb, devMPb, devGPb_thin); //note: 3rd param width is the actual pitch of the image
*/
  NormalizeGpbAll(nPixels, 8, matrixPitchInFloats, devGPball);
  
  cudaThreadSynchronize();
  sdkStopTimer(&timer);
  //printf("CUDA Status : %s\n", cudaGetErrorString(cudaGetLastError()));
  log_info("CUDA Status : %s", cudaGetErrorString(cudaGetLastError()));

#ifdef __TIMER_SPECFIC
  sdkStopTimer(&timer_specific);
  //printf(">+< PostProcess | %f | ms\n", sdkGetTimerValue(&timer_specific));
  log_info(">+< PostProcess | %f | ms", sdkGetTimerValue(&timer_specific));
  sdkDeleteTimer(&timer_specific);
#endif
  //printf(">+< Computation time: | %f | seconds\n", sdkGetTimerValue(&timer)/1000.0);
  log_info(">+< Computation time: | %f | seconds", sdkGetTimerValue(&timer)/1000.0);
  sdkDeleteTimer(&timer);
  *hostGPb = (float*)malloc(sizeof(float)*nPixels);
  memset(*hostGPb, 0, sizeof(float) * nPixels);
  cudaMemcpy(*hostGPb, devGPb, sizeof(float)*nPixels, cudaMemcpyDeviceToHost);
/*
  sdkSavePGM(outputPGMfilename, hostGPb, width, height);
  writeFile(outputPBfilename, width, height, hostGPb);
*/
  /* thin image */
/*
  float* hostGPb_thin = (float*)malloc(sizeof(float)*nPixels);
  memset(hostGPb_thin, 0, sizeof(float) * nPixels);
  cudaMemcpy(hostGPb_thin, devGPb_thin, sizeof(float)*nPixels, cudaMemcpyDeviceToHost);
  sdkSavePGM(outputthinPGMfilename, hostGPb_thin, width, height);
  writeFile(outputthinPBfilename, width, height, hostGPb);
  free(hostGPb_thin);
*/
  /* end thin image */

  float* hostGPbAll = (float*)malloc(sizeof(float) * matrixPitchInFloats * 8);
  cudaMemcpy(hostGPbAll, devGPball, sizeof(float) * matrixPitchInFloats * 8, cudaMemcpyDeviceToHost);

  //int oriMap[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int oriMap[] = {4, 5, 6, 7, 0, 1, 2, 3};
  //int oriMap[] = {3, 2, 1, 0, 7, 6, 5, 4};
  *hostGPbAllConcat = (float*)malloc(sizeof(float) * width * height * 8);
  for(int i = 0; i < 8; i++) {
    transpose(width, height, hostGPbAll + matrixPitchInFloats * oriMap[i], *hostGPbAllConcat + width * height * i);
  }
  free(hostGPbAll);
  /*
   *int dim[3];
   *dim[0] = 8; 
   *dim[1] = width;
   *dim[2] = height;
   *writeArray(outputgpbAllfilename, 3, dim, hostGPbAllConcat);
   */

  /*
  for(int orientation = 0; orientation < 8; orientation++) {
    sprintf(orientationIndicator, "_%i_Pb.pgm", orientation);
    cutSavePGMf(outputPGMAllfilename, hostGPbAll + matrixPitchInFloats * orientation, width, height);
  }
  */

/*
  free(hostGPbAllConcat);
*/

/*   filename = "polynesiaPb.txt"; */
/*   writeTextImage(filename, width, height, hostGPb);  */
/*   int getNEigs = 9; */
/*   FILE* fp; */
/*   fp = fopen("eigenVectors.txt", "w"); */
/* 	//Print out the eigenvectors */
/*   for (int j = 0; j < nPixels; j++) { */
/*     for (int i = 0; i < getNEigs; i++) { */
/*       fprintf(fp, "%f ", eigenvectors[i*nPixels+j]); */
/*     } */
/*     fprintf(fp, "\n"); */
/*   } */
/*   fclose(fp); */

/*   fp = fopen("eigenValues.txt", "w"); */
/* 	for (int i = 0; i < getNEigs; i++) { */
/* 		fprintf(fp, "%e\n", eigenvalues[i]); */
/* 	} */
/* 	fclose(fp); */

/*  customCheckCudaErrors(cudaFree(devBgcombined));
  customCheckCudaErrors(cudaFree(devCgacombined));
  customCheckCudaErrors(cudaFree(devCgbcombined));
  customCheckCudaErrors(cudaFree(devTgcombined));*/

  customCheckCudaErrors(cudaFree(devEigenvectors));
  customCheckCudaErrors(cudaFree(devCombinedGradient));
  customCheckCudaErrors(cudaFree(devSPb));
  customCheckCudaErrors(cudaFree(devGPb));
/*
  customCheckCudaErrors(cudaFree(devGPb_thin));
*/
  customCheckCudaErrors(cudaFree(devGPball));

  fclose(fp);

  cudaDeviceReset();
}
