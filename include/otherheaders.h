#ifndef OTRHDRS_H_
#define OTRHDRS_H_

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <queue>

#include "UCM_CONSTANTS.h"
#include "detmpb.h"
#include "log.h"

#include "mpi.h"

template <typename T>
std::vector<T> operator+(std::vector<T>& a, const std::vector<T>& b) {
    assert(a.size() == b.size());
    std::vector<T> result;
    result.reserve(a.size());
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result), std::plus<T>());
    return result;
}

template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b) {
    assert(a.size() == b.size());
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T>());
    return a;
}

/*  sort pair<int,int> based on first value
typedef std::pair<int,int> mypair;
bool comparatorPair(const mypair& l, const mypair& r) {
    return l.first < r.first;
}
*/

void mex_pb_parts_final_selected(cv::Mat*& plhs0, cv::Mat***& plhs_bg, cv::Mat***& plhs_cga, cv::Mat***& plhs_cgb,
                                 cv::Mat***& plhs_tg, const cv::Mat& channels0, const cv::Mat& channels1,
                                 const cv::Mat& channels2);

void multiscalePb(cv::Mat*& mPb_nmax, cv::Mat*& mPb_nmax_rsz, detmpb& detmpb_obj, cv::Mat& imd, cv::Size orig_size,
                  int nChan, std::string outFile, double rsz = 1.0);
void det_mPb(detmpb& returnVals, cv::Mat& im);
void GlobalPb(std::string imgFile, std::string outFile, std::vector<cv::Mat>& gPb, double rsz = 1.0);
void GlobalPbWithNmax(std::string imgFile, std::string outFile, std::vector<cv::Mat>& gPb, double rsz = 1.0);
void GlobalPbCuda(cv::Mat& imagePatch, std::string outFile, std::vector<cv::Mat>& gPbOrient, cv::Mat& gPbNonmax,
                  unsigned int nOrient, unsigned int rank, double rsz = 1.0);
cv::Mat* nonmax_channels(std::vector<cv::Mat*> mPb_all, double nonmax_ori_tol = PI / 8);
cv::Mat* fitparab(cv::Mat* z, double ra, double rb, double theta, cv::Mat* filt);
void nonmax_oriented(cv::Mat*& nmax, cv::Mat* a, cv::Mat* ind, double nonmax_ori_tol = PI / 8);
void savgol_border(cv::Mat* b, cv::Mat* a, cv::Mat* z, double ra, double rb, double theta);
void spectralPb(std::vector<cv::Mat*>& sPb, cv::Mat* mPb_rsz, cv::Size orig_size, std::string outFile2, int nvec = 17);
void buildW(std::vector<double>& valD, std::vector<double>& val, std::vector<int>& I, std::vector<int>& J, cv::Mat& l1,
            cv::Mat& l2);
cv::Mat oeFilter(double sigma, int support = 3, double theta = 0.0, int deriv = 0, int hil = 0, int vis = 0);

/*
 * Given a image and parameters for maximum piece size, returns a vector of patches from the image
 * each of which can fit inside a square of piece_size x piece_size
 * Also divide the ground truth files for each image, and writes them to disk
 */
std::vector<cv::Mat> Img2Pieces(std::string basePrefix, cv::Mat& bigImg, int overlap, int piece_size);

/*
 * Combines the gPb orient matrices from the child processes and returns them as a vector of matrices
 */
void Pieces2Gpb(std::vector<std::vector<cv::Mat>>& patchGpbOrient, std::vector<cv::Mat>& patchGpbNonmax,
                std::vector<cv::Mat>& gPbOrient, cv::Mat& gPb_nmax,
                std::string pieceFilePat, int overlap, int rsz, int rows, int cols, int nOrient);
int bwlabelManu(cv::Mat& labels, cv::Mat& BW, int numNeighbor);
void meshgrid(cv::Mat& XMat, cv::Mat& YMat, int startX, int endX, int startY, int endY);
void CreateSuperpixelGraph(std::vector<std::unordered_set<int>>& superpixelGraph, cv::Mat& labels,
                           std::vector<std::pair<int, int>>& bdry, int nLabels, int labelOffset, int* adjTRow,
                           int* adjBRow, int* adjLCol, int* adjRCol, int up, int down, int left, int right, int aTL,
                           int aTR, int aBL, int aBR, int tl, int tr, int bl, int br, int M, int N);
int PopulateUniqueVector(std::vector<int>& uniqueVec, int* origArr, int origLength);
int ReadGTFile(std::vector<int>& gtSuperpixels, std::vector<int>& gtClassLabels, std::string gtFileName,
               cv::Mat& labels, std::vector<std::pair<int, int>>* gtCoords);
void OpenCVFileWriter(const cv::Mat& matrixToBeWritten, std::string fileName, std::string matrixName);
void OpenCVImageWriter(const cv::Mat& matrixToBeWritten, std::string imageName);
void im2col(cv::Mat& Out, cv::Mat& In, int radius, int check = -1);

void ReadGTFile(std::string gtFileName, std::vector<int>* gtClassLabels, std::vector<std::pair<int, int>>* gtCoords);

/*
 * Merges the components in dataVector into a larger matrix targetMat. Requires that targetMat be valid, initialized
 * memory prior to calling this function.
 */
void Pieces2Mat(std::string pieceFilePat, std::string mark, int overlap, int piece_size, cv::Mat& targetMat,
                std::vector<cv::Mat> dataVector);

/*
 * Globally get the training patches and save into files.
 */
std::string padStrZeros(int n_zero, std::string old_string);
std::string getTrainingPatches(std::string basePrefix, cv::Mat& bigImg, int rows, int cols);

/*
 * Sends an opencv matrix to process with rank destId
 * within the MPI_COMM_WORLD
 */
void MpiSendMat(cv::Mat& image, int destId);

/*
 * Recieves an opencv matrix sent from the process with rank sourceId
 * within the MPI_COMM_WORLD
 */
cv::Mat MpiRecvMat(int sourceId);

/*
 * Sends a vector of opencv matrices to process with rank destId
 * within the MPI_COMM_WORLD
 */
void MpiSendMatVec(std::vector<cv::Mat>& images, int destId);

/*
 * Recieves a vector of opencv matrices sent from the process with rank sourceId
 * within the MPI_COMM_WORLD
 */
std::vector<cv::Mat> MpiRecvMatVec(int sourceId);

/*
 * Returns the MPI_Datatype for the passed OpenCV type
 * This is required to correctly pass information around between processes since there is no
 * fixed interpretation of the buffer backing an opencv matrix.
 *
 * Use this to query the correct type and cast the matrix buffer appropriately prior to sending it using
 * MPI_Send
 *
 * @see Link (https://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html) with a map
 *      of type, nChannels to the backing data type
 *
 * @see Link (http://beige.ucs.indiana.edu/I590/node100.html) for MPI datatpyes
 *
 * @param cvType The integer representing
 * @param nChannels Number of channels in the image
 * @return The corresponding MPI_datatype for the given OpenCV matrix .type() attribute
 */
MPI_Datatype MpiTypeFor(int cvType, int nChannels);

#endif
