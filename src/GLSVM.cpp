#include "GLSVM.h"
#include "log.h"

using namespace std;

GLSVM::GLSVM(const vector<unordered_set<int> >& _superpixelGraph, cv::Mat& _allDescrs,
             cv::Mat& _aUTopRowFeatures, cv::Mat& _aUBotRowFeatures, cv::Mat& _aULeftColFeatures,
             cv::Mat& _aURightColFeatures, cv::Mat& _aTLFeatures, cv::Mat& _aTRFeatures, cv::Mat& _aBLFeatures,
             cv::Mat& _aBRFeatures, int* _adjUTRow, int* _adjUBRow, int* _adjULCol, int* _adjURCol, int _adjUTRowLen,
             int _adjUBRowLen, int _adjULColLen, int _adjURColLen, int _aTL, int _aTR, int _aBL, int _aBR,
             vector<int>& _nodeDegree, int* _adjUTRowDeg, int* _adjUBRowDeg, int* _adjULColDeg, int* _adjURColDeg,
             int _adjTLDeg, int _adjTRDeg, int _adjBLDeg, int _adjBRDeg, int _nBaseLabels, int _labelOffset, int _nFeat,
             double _tau)
    : superpixelGraph(_superpixelGraph),
      allDescrs(_allDescrs),
      aUTopRowFeatures(_aUTopRowFeatures),
      aUBotRowFeatures(_aUBotRowFeatures),
      aULeftColFeatures(_aULeftColFeatures),
      aURightColFeatures(_aURightColFeatures),
      aTLFeatures(_aTLFeatures),
      aTRFeatures(_aTRFeatures),
      aBLFeatures(_aBLFeatures),
      aBRFeatures(_aBRFeatures),
      nodeDegree(_nodeDegree) {
    /// TODO: Auto-generated constructor stub

    adjUTRow = _adjUTRow;
    adjUBRow = _adjUBRow;
    adjULCol = _adjULCol;
    adjURCol = _adjURCol;

    adjUTRowLen = _adjUTRowLen;
    adjUBRowLen = _adjUBRowLen;
    adjULColLen = _adjULColLen;
    adjURColLen = _adjURColLen;
    aTL = _aTL;
    aTR = _aTR;
    aBL = _aBL;
    aBR = _aBR;

    adjUTRowDeg = _adjUTRowDeg;
    adjUBRowDeg = _adjUBRowDeg;
    adjULColDeg = _adjULColDeg;
    adjURColDeg = _adjURColDeg;
    adjTLDeg = _adjTLDeg;
    adjTRDeg = _adjTRDeg;
    adjBLDeg = _adjBLDeg;
    adjBRDeg = _adjBRDeg;

    nBaseLabels = _nBaseLabels;
    labelOffset = _labelOffset;
    nFeatures = _nFeat;

    tau = _tau;

    A = cv::Mat::zeros(nFeatures, nFeatures, CV_64F);

    for (auto it = nodeDegree.begin(); it != nodeDegree.end(); it++) {
        if (*it <= 0) {
            log_debug("nodeDegree: %d", *it);
        }
    }
    log_debug("nodeDegree at 5: %d", nodeDegree.at(5));
}

GLSVM::~GLSVM() {}

double GLSVM::GenerateCijandWij(cv::Mat& cIJ, cv::Mat& xI, cv::Mat& xJ, int I, int J, int jSelection, int jIndex) {
    // cout << "called " << endl;

    if (nodeDegree.at(I) <= 0) {
        log_debug("Degree is %d at %d", I, J);
    }
    assert((I >= 0) && (I < nBaseLabels) && (jSelection != -1) && (jIndex != -1) && (jSelection >= 0) &&
           (jSelection <= 8) && (nodeDegree.at(I) > 0));

    cv::Mat xIMxJ_GL;

    switch (jSelection) {
        case 0:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(nodeDegree.at(jIndex));
            break;
        case 1:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(adjUTRowDeg[jIndex]);
            break;
        case 2:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(adjUBRowDeg[jIndex]);
            break;
        case 3:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(adjULColDeg[jIndex]);
            break;
        case 4:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(adjURColDeg[jIndex]);
            break;
        case 5:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(adjTLDeg);
            break;
        case 6:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(adjTRDeg);
            break;
        case 7:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(adjBLDeg);
            break;
        case 8:
            xIMxJ_GL = xI / sqrt(nodeDegree.at(I)) - xJ / sqrt(adjBRDeg);
            break;
        default:
            throw jSelection;
            break;
    }

    cIJ = xIMxJ_GL.t() * xIMxJ_GL;

    double norm_xIMxJ = cv::norm(xI - xJ);
    double wIJ = exp(-norm_xIMxJ / (2.0 * pow(tau, 2)));

    if (I == jIndex) {
        log_debug("norm_xIMxJ: %lf jSelection: %d, I: %d, J: %d", norm_xIMxJ, jSelection, I, J);
    }

    return wIJ;
}

int GLSVM::GenerateA() {
    // cv::Mat A(nFeatures, nFeatures, CV_64F);

    int cnt = 0;
    for (auto itI = superpixelGraph.begin(); itI != superpixelGraph.end(); itI++) {
        int I = cnt;
        if (nodeDegree.at(I) <= 0) {
            log_debug("I: %d, nodeDegree: %d", I, nodeDegree.at(itI - superpixelGraph.begin()));
        }

        cv::Mat xI = allDescrs.row(I);
        for (auto itJ = itI->begin(); itJ != itI->end(); itJ++) {
            cv::Mat xJ;
            int J = *itJ;
            int localJ = J - labelOffset;
            int* p = NULL;
            int jSelection = -1, jIndex = -1;

            /// FIXME: replace asserts with a raised exception
            try {
                if (localJ >= 0 && localJ < nBaseLabels) {
                    xJ = allDescrs.row(localJ);
                    jSelection = 0;
                    jIndex = localJ;
                }
                else if ((p = find(adjUTRow, adjUTRow + adjUTRowLen, J)) != adjUTRow + adjUTRowLen) {
                    jSelection = 1;
                    jIndex = p - adjUTRow;
                    xJ = aUTopRowFeatures.row(jIndex);
                    assert(jIndex >= 0 && jIndex < adjUTRowLen);
                }
                else if ((p = find(adjUBRow, adjUBRow + adjUBRowLen, J)) != adjUBRow + adjUBRowLen) {
                    jSelection = 2;
                    jIndex = p - adjUBRow;
                    xJ = aUBotRowFeatures.row(jIndex);
                    assert(jIndex >= 0 && jIndex < adjUBRowLen);
                }
                else if ((p = find(adjULCol, adjULCol + adjULColLen, J)) != adjULCol + adjULColLen) {
                    jSelection = 3;
                    jIndex = p - adjULCol;
                    xJ = aULeftColFeatures.row(jIndex);
                    assert(jIndex >= 0 && jIndex < adjULColLen);
                }
                else if ((p = find(adjURCol, adjURCol + adjURColLen, J)) != adjURCol + adjURColLen) {
                    jSelection = 4;
                    jIndex = p - adjURCol;
                    xJ = aURightColFeatures.row(jIndex);
                    assert(jIndex >= 0 && jIndex < adjURColLen);
                }
                else if (J == aTL) {
                    xJ = aTLFeatures;
                    jSelection = 5;
                    jIndex = -2;
                }
                else if (J == aTR) {
                    xJ = aTRFeatures;
                    jSelection = 6;
                    jIndex = -2;
                }
                else if (J == aBL) {
                    xJ = aBLFeatures;
                    jSelection = 7;
                    jIndex = -2;
                }
                /// TODO: @Yupeng - replace the training feature in patch-based way
                else if (J == aBR) {
                    xJ = aBRFeatures;
                    jSelection = 8;
                    jIndex = -2;
                }
                else {
                    /// TODO: @Yupeng - Investigate if there is a better way to raise an exception than `throw J`
                    throw J;
                }
            }
            catch (int e) {
                log_error("An exception occurred for I: %d, J: %d", I, J);
                assert(!xJ.empty());
                return -1;
            }

            cv::Mat cIJ;
            double wIJ;

            try {
                wIJ = this->GenerateCijandWij(cIJ, xI, xJ, I, J, jSelection, jIndex);
            }
            catch (int e) {
                log_error("An exception occurred for invalid jSelection: %d", e);
                return -1;
            }

            A += wIJ * cIJ;

            /// FIXME: Raise an exception when wIJ == nan
            if (std::isnan(wIJ)) {
                log_error("wIJ is nan for I: %d, J: %d, jSelection: %d, jIndex: %d, nBaseLabels: %d, tau: %lf", I,
                          J, jSelection, jIndex, nBaseLabels, tau);
                assert(!std::isnan(wIJ));
            }
        }
        cnt++;
    }

    /// FIXME: replace with a suitable exception
    assert(cnt == nBaseLabels);

    log_debug("nBaseLabels: %d, cnt: %d", nBaseLabels, cnt);
    return 0;
}

void GLSVM::FinalSolverAtRootNode(cv::Mat& weightVectors, const cv::Mat& allGTFeatures, const cv::Mat& summedA,
                                  double lambdaS, double lambdaH, double epsZ, int nClasses, int totalNGTPoints,
                                  int* allGTClassLabels, double convergenceThreshold) const {
    cv::Mat RMat = cv::Mat::eye(nFeatures, nFeatures, CV_64F);      /// TODO: augment all features by 1
    RMat.at<double>(0, 0) = 0;

    /* modified by Yupeng */
    OpenCVFileWriter(summedA, "summedA_node_inside.yml", "summedA");
    OpenCVFileWriter(RMat, "RMat.yml", "RMat");

    // cv::Mat gtFeatures = summedA.t() * summedA;      /// TODO: correct this

    // cv::FileStorage fslabels("./labeledData/GLSVM_wVec_node0.yml", cv::FileStorage::WRITE);
    // cv::FileStorage fslabels1("./labeledData/GLSVM_wMat_node0.yml", cv::FileStorage::WRITE);
    // cv::FileStorage fslabels2("./labeledData/GLSVM_yVec_node0.yml", cv::FileStorage::WRITE);

    cv::Mat allGTLabels(1, totalNGTPoints, CV_32S, allGTClassLabels);
    // fslabels2 << "allGTLabels" << allGTLabels;

    weightVectors = cv::Mat::zeros(nFeatures, nClasses, CV_64F);

    for (int k = 0; k < nClasses; k++) {
        cv::Mat yVec = cv::Mat::ones(1, totalNGTPoints, CV_64F);
        for (int i = 0; i < totalNGTPoints; i++) {
            if (allGTClassLabels[i] != k + 1) {
                yVec.at<double>(0, i) = -1;
            }
        }

        // fslabels2 << "yVec_" + to_string(k) << yVec;

        cv::Mat wVec = cv::Mat::zeros(nFeatures, 1, CV_64F);
        cv::Mat wVecPrev = wVec.clone();

        int iter = 0;
        bool converged = 0;
        double error = 10;      /// FIXME: @Yupeng - should be initialized with large value

        log_debug("nFeatures: %d, wVec.rows: %d, wVec.cols: %d, allGTFeatures.rows: %d, allGTFeatures.cols: %d",
                  nFeatures, wVec.rows, wVec.cols, allGTFeatures.rows, allGTFeatures.cols);

        while (!converged) {
            cv::Mat temp = cv::abs(1 - wVec.t() * allGTFeatures.t());

            cv::Mat zVec = cv::max(epsZ, yVec.mul(temp));

            // OpenCVFileWriter(zVec, "zVec_node0.xml", "zvec");

            cv::Mat zMat = cv::repeat(zVec, nFeatures, 1);

            cv::Mat GTMat = (1.0 / (2 * zMat)).mul(allGTFeatures.t()) * allGTFeatures;

            // OpenCVFileWriter(GTMat, "GTMat_node0.xml", "GTMat");

            cv::Mat AMat = RMat + lambdaH * GTMat + 2 * lambdaS * summedA;

            // OpenCVFileWriter(AMat, "AMat_node0.xml", "AMat");

            cv::Mat dVec = lambdaH * ((((1.0 + zVec) / (2 * zVec)).mul(yVec)) * allGTFeatures).t();

            // OpenCVFileWriter(dVec, "dVec_node0.xml", "dVec");

            cv::solve(AMat, dVec, wVec);

            // OpenCVFileWriter(wVec, "wVec_node0.xml", "wVec");

            // FIXME: @Yupeng - move out error
            error = cv::norm(wVec - wVecPrev);
            /* modified by Yupeng */
            if (iter % 20 == 0) {
                log_debug("iter: %d, error: %f", iter, error);
                OpenCVFileWriter(zVec, "zVec_node_" + to_string(iter) + ".yml", "zvec");
                OpenCVFileWriter(GTMat, "GTMat_node_" + to_string(iter) + ".yml", "GTMat");
                OpenCVFileWriter(AMat, "AMat_node_" + to_string(iter) + ".yml", "AMat");
                OpenCVFileWriter(dVec, "dVec_node_" + to_string(iter) + ".yml", "dVec");
                OpenCVFileWriter(wVec, "wVec_node_" + to_string(iter) + ".yml", "wVec");
            }

            if ((error < convergenceThreshold) || iter >= 100) {
                converged = 1;
            }
            /* else is not needed for the line below but can be helpful if printout of the difference at
               convergence is required */
            else {
                wVecPrev = wVec;
            }

            iter++;
        }
        log_debug("class k: %d, iter: %d, error: %f", k + 1, iter, error);
        // fslabels << "class_" + to_string(k) << wVec;
        wVec.copyTo(weightVectors.col(k));  /* weightVectors.col(k) = wVec + 0 didnt work */
    }

    // fslabels1 << "Full_W_columnwise" << weightVectors;

    // fslabels.release();
    // fslabels1.release();
    // fslabels2.release();
}
