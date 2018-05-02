#ifndef SRC_UCMGENERATOR_H_
#define SRC_UCMGENERATOR_H_

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
#include "otherheaders.h"

class UCM_generator {
   public:
    cv::Mat ucm2, ucm, labels;               /* labels mean small labels */
    std::vector<std::pair<int, int>> bdry;  /* bdry means small bdry */
    std::string pieceFilePath;
    int rows, cols;
    int nLabels;

    cv::Mat midLabels;
    std::vector<std::pair<int, int> > midBdry;
    int nMidLabels;

    cv::Mat coarseLabels;
    std::vector<std::pair<int, int> > coarseBdry;
    int nCoarseLabels;

    UCM_generator();
    UCM_generator(int grid_rank_rows, int grid_rank_cols, std::string pieceFilePath);
    virtual ~UCM_generator();

    void generateUCM2(std::vector<cv::Mat>& gPb_orient, std::string fmt);
    void generateUCM();

    void generateBdryAtScaleK(double scaleK, std::vector<std::pair<int, int> >& boundary);
    int generateLabelsAtScaleK(double scaleK, cv::Mat& labelMat);

    void generateBdryAtScaleK(double scaleK, int option);
    int generateLabelsAtScaleK(double scaleK, int option);
};

#endif
