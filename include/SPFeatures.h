#ifndef SRC_SPFEATURES_H_
#define SRC_SPFEATURES_H_

#include <cv.h>
#include <highgui.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include "UCM_CONSTANTS.h"
#include "head.h"
#include "ml.h"
#include "otherheaders.h"

class SPFeatures {
   private:
    void genLabelBucketsAtScaleK(std::vector<std::vector<std::pair<int, int>>>& scaleKBuckets,
                                 cv::Mat& labelMatAtScaleK);

   public:
    cv::Mat& imageRef;
    cv::Mat& labelMat;
    ///cv::Mat& midLabelMat;
    ///cv::Mat& coarseLabelMat;
    cv::Mat allDescrs;
    int binStepSize;
    int nBins;
    int nBaseLabels;///, nMidLabels, nCoarseLabels;
    bool colorImageFlag;
    int nFeatures, nClasses;
    std::vector<std::vector<std::pair<int, int>>> baseLabelBuckets;
    ///std::vector<std::vector<std::pair<int, int>>> midLabelBuckets;
    ///std::vector<std::vector<std::pair<int, int>>> coarseLabelBuckets;
    cv::Mat grayScaleImage;

    int nAuxFeatures;
    cv::Mat auxFeatures;

    CvDTree* DT = NULL;

    SPFeatures(int binStep, int _nBaseLabels, int _nClasses, bool _colorImageFlag,
               cv::Mat& _imageRef, cv::Mat& labelMat, std::string _pieceFilePath);
    const cv::Mat& genAuxFeatures();
    void genIntensityHistograms();
    void genCornerDescrs(int blockSize = 2, int apertureSize = 3, double k = 0.04, int threshold = 60);
    void genAverageFeatures();
    void DTFeatures(const std::vector<int>& gtSuperPixels, const std::vector<int>& gtClassLabels, int labelOffset);
    void addBiasOne();

    cv::Mat makefilter(int sup, int sigma, int tau);
    void makeSfilters(std::vector<cv::Mat>* F, int NF, int SUP);
    void genTextonFeatures(int centerSize);
    cv::Mat genTextonMap(cv::Mat& textonImage, int clusterCount, bool isBroadcasted);   // modified by Yupeng, 04/23
    std::string pieceFilePath;
    void genSuperpixelFeatures();
    void genSpCenterFeatures();
    void genRecPatchFeatures(std::vector<std::pair<int, int>>* centerCoords, cv::Mat& recDescrs);
    void genSeedCenterFeatures(std::vector<std::pair<int, int>>* gtCoords, cv::Mat& trainDescrs);
    int clusterCount;
    void addBiasOne(cv::Mat& anyDescrs);

    void genRecPatchFeatures(cv::Mat& rgbPatch, cv::Mat& recDescrs, cv::Mat& textonDescrs, int n);
    void genSeedCenterFeatures(std::vector<cv::String>* filenames, std::vector<int>& gtClassLabels, cv::Mat& trainDescrs);

    virtual ~SPFeatures();
};

#endif
