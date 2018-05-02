#ifndef SRC_GLSVM_H_
#define SRC_GLSVM_H_

#include <cv.h>
#include <highgui.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <limits>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "otherheaders.h"
// #include "head.h"
#include "UCM_CONSTANTS.h"

class GLSVM {
   public:
    const std::vector<std::unordered_set<int>>& superpixelGraph;
    const cv::Mat& allDescrs;
    const cv::Mat& aUTopRowFeatures;
    const cv::Mat& aUBotRowFeatures;
    const cv::Mat& aULeftColFeatures;
    const cv::Mat& aURightColFeatures;
    const cv::Mat& aTLFeatures;
    const cv::Mat& aTRFeatures;
    const cv::Mat& aBLFeatures;
    const cv::Mat& aBRFeatures;
    const std::vector<int>& nodeDegree;

    int* adjUTRow;
    int* adjUBRow;
    int* adjULCol;
    int* adjURCol;

    int adjUTRowLen;
    int adjUBRowLen;
    int adjULColLen;
    int adjURColLen;
    int aTL;
    int aTR;
    int aBL;
    int aBR;

    int* adjUTRowDeg;
    int* adjUBRowDeg;
    int* adjULColDeg;
    int* adjURColDeg;
    int adjTLDeg;
    int adjTRDeg;
    int adjBLDeg;
    int adjBRDeg;

    int nBaseLabels;
    int labelOffset;
    int nFeatures;

    double tau;

    cv::Mat A;

    GLSVM(const std::vector<std::unordered_set<int> >& _superpixelGraph, cv::Mat& _allDescrs,
          cv::Mat& _aUTopRowFeatures, cv::Mat& _aUBotRowFeatures, cv::Mat& _aULeftColFeatures,
          cv::Mat& _aURightColFeatures, cv::Mat& _aTLFeatures, cv::Mat& _aTRFeatures, cv::Mat& _aBLFeatures,
          cv::Mat& _aBRFeatures, int* _adjUTRow, int* _adjUBRow, int* _adjULCol, int* _adjURCol, int adjUTRowLen,
          int adjUBRowLen, int adjULColLen, int adjURColLen, int aTL, int aTR, int aBL, int aBR,
          std::vector<int>& _nodeDegree, int* adjUTRowDeg, int* adjUBRowDeg, int* adjULColDeg, int* adjURColDeg,
          int adjTLDeg, int adjTRDeg, int adjBLDeg, int adjBRDeg, int _nBaseLabels, int _labelOffset, int _nFeat,
          double _tau);
    virtual ~GLSVM();

    int GenerateA();

    void FinalSolverAtRootNode(cv::Mat& weightVectors, const cv::Mat& allGTFeatures, const cv::Mat& summedA,
                               double lambdaS, double lambdaH, double epsZ, int nClasses, int totalNGTPoints,
                               int* allGTClassLabels, double convergenceThreshold) const;

   private:
    double GenerateCijandWij(cv::Mat& cIJ, cv::Mat& xI, cv::Mat& xJ, int I, int J, int jSelection, int jIndex);
};

#endif
