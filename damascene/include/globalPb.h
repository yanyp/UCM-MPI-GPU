#ifndef GLOBALPB
#define GLOBALPB

#define IMUL(a, b) __mul24(a, b)


/**
 * Given all the local cues and the spectral pb, calculate the weighted sum among them and normalize the result.
 * Assuming all the arrays are in GPU, each with size p_nMatrixPitch*p_nOrient
 * @param p_nPixels: Number of pixels in the image
 * @param p_nMatrixPitch: The pitch of each matrix
 * @param p_nOrient: Number of orientations 
 * @param devCombinedGradient: The combined gradient matrix 
 * @param devspb: The spb matrix
 * @param devmpb: The mpb matrix
 * @param devGpball : The final Gpb output, with size p_nPixels * p_nOrient
 * @param devResult: The result matrix, with size p_nPixels * 1 
 * 
 **/

void StartCalcGPb(int p_nPixels, int p_nMatrixPitch, int p_nOrient,
                  float* devCombinedGradient, float* devspb, float* devmpb,
                  float* devGpball, float* devResult);


/**
 * Given an image, calculates the GPb
 * @param rank: Rank of the process calling the function, used to select which GPU to run under
 * @param width: Width of the image in pixels
 * @param height: Height of the image in pixels
 * @param data: An unsigned int representation of the image, of size width * height * sizeof(unsigned int) ,
 *                4 because 1 for red, 1 for blue, 1 for green, and 1 padded with zero
 * @param hostGPb: Reference to the nonmax suppressed GPb matrix, of size width * height
 * @param hostGPbAllConcat: Reference to the whole GPb matrix, of size width * height * 8, 8 because we have 8 orientations
 *
 **/
void computeGPb(unsigned int rank, unsigned int width, unsigned int height, unsigned int* data, float** hostGPb, float** hostGPbAllConcat);


/**
 * Returns the number of CUDA devices on the machine
 *
 **/
int getCudaDeviceCount();

#endif

