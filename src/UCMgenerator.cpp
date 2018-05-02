#include <UCMgenerator.h>
#include "log.h"

using namespace std;

UCM_generator::UCM_generator() {
    /// TODO: Auto-generated constructor stub
    rows = 0;
    cols = 0;
}

UCM_generator::UCM_generator(int grid_rank_rows, int grid_rank_cols, string _pieceFilePath) {
    rows = grid_rank_rows;
    cols = grid_rank_cols;
    ucm = cv::Mat::zeros(rows, cols, CV_64FC1);
    pieceFilePath = _pieceFilePath;
}

UCM_generator::~UCM_generator() {
    // TODO: Auto-generated destructor stub
}

void UCM_generator::generateUCM2(vector<cv::Mat>& gPb_orient, string fmt) {
    // FIXME: @Yupeng - may be removed??? --- "After dividing"
    string gPbMatrixName = "gPb_orient_eachAngle_AfterDiv";
    for (int i = 0; i < gPb_orient.size(); i++) {
        string fileName = "imagedata/gPbCJTorient_afterdivide_i_" + to_string(i) + ".yml";
        cv::FileStorage file1(fileName, cv::FileStorage::WRITE);
        file1 << gPbMatrixName << gPb_orient.at(i);
        file1.release();
    }

    // TODO: @Yupeng - only compute ucm2 if the file does not exist (need to optimize ucm below)
    /*
    string ucmBigPartFileName = pieceFilePath + "_ucm2.yml";
    ifstream ifile(ucmBigPartFileName);
    if (!ifile) {
    */
    ucm2 = Contours2Ucm(pieceFilePath, gPb_orient, fmt);
    /*
        string ucmMatrixName = "ucm2";
        cv::FileStorage fs(ucmBigPartFileName, cv::FileStorage::WRITE);
        fs << ucmMatrixName << ucm2;
        fs.release();
    }
    else {
        cv::FileStorage fs(ucmBigPartFileName, cv::FileStorage::READ);
        fs["ucm2"] >> ucm2;
        fs.release();
    }
    */
}

void UCM_generator::generateUCM() {
    for (int i = 0; i < ucm.rows; i++)
        for (int j = 0; j < ucm.cols; j++) {
            double ucm2Val = ucm2.at<double>(2 + i * 2, 2 + j * 2);
            /*
            if (std::isnan(ucm2Val)) {
                cout << "1111 nan exists!!!!!!!!!!!!!!!!! grid_rank =  " << grid_rank << endl;
            }
            */
            ucm.at<double>(i, j) = ucm2Val;
            /*
            if (ucm2Val >= scaleK) {
                bdry.push_back(make_pair(i,j));
            }
            */
        }
}

void UCM_generator::generateBdryAtScaleK(double scaleK, int option) {
    switch (option) {
        case 1:
            generateBdryAtScaleK(scaleK, bdry);
            break;
        case 2:
            generateBdryAtScaleK(scaleK, midBdry);
            break;
        case 3:
            generateBdryAtScaleK(scaleK, coarseBdry);
            break;
        case 4:
            assert(option == 1 || option == 2 || option == 3);
            break;
    }
}

int UCM_generator::generateLabelsAtScaleK(double scaleK, int option) {
    int numOfLabelsAtScaleK = -1;

    switch (option) {
        case 1:
            numOfLabelsAtScaleK = generateLabelsAtScaleK(scaleK, labels);
            nLabels = numOfLabelsAtScaleK;
            break;
        case 2:
            numOfLabelsAtScaleK = generateLabelsAtScaleK(scaleK, midLabels);
            nMidLabels = numOfLabelsAtScaleK;
            break;
        case 3:
            numOfLabelsAtScaleK = generateLabelsAtScaleK(scaleK, coarseLabels);
            nCoarseLabels = numOfLabelsAtScaleK;
            break;
        case 4:
            assert(option == 1 || option == 2 || option == 3);
            break;
    }

    assert(numOfLabelsAtScaleK > 0);

    return numOfLabelsAtScaleK;
}

void UCM_generator::generateBdryAtScaleK(double scaleK, vector<pair<int, int> >& boundary) {
    for (int i = 0; i < ucm.rows; i++) {
        for (int j = 0; j < ucm.cols; j++) {
            if (ucm.at<double>(i, j) >= scaleK) {
                boundary.push_back(make_pair(i, j));
            }
        }
    }
}

int UCM_generator::generateLabelsAtScaleK(double scaleK, cv::Mat& labelMat) {
    cv::Mat labels2;
    cv::Mat BW = cv::Mat::zeros(ucm2.rows, ucm2.cols, CV_32S);

    for (int i = 0; i < ucm2.rows; i++)
        for (int j = 0; j < ucm2.cols; j++) {
            /*
            if (std::isnan(ucm2.at<double>(i,j))) {
                cout << "2222 nan exists!!!!!!!!!!!!!!!!!! grid_rank = " << grid_rank << endl;
            }
            */
            if (ucm2.at<double>(i, j) <= scaleK) BW.at<int>(i, j) = 1;
        }

    /// TODO: @Yupeng debug
    int nLabels = bwlabelManu(labels2, BW, 8);
    log_debug("BW Size | rows: %d, cols: %d", BW.rows, BW.cols);
    log_debug("labels2 | rows: %d, cols: %d", labels2.rows, labels2.cols);
    BW.release();

    labelMat = cv::Mat::zeros((labels2.rows - 1) / 2, (labels2.cols - 1) / 2, CV_32S);
    for (int i = 0; i < labelMat.rows; i++) {
        for (int j = 0; j < labelMat.cols; j++) {
            labelMat.at<int>(i, j) = labels2.at<int>(2 * i + 1, 2 * j + 1) - 1;
        }
    }

    /*
    string labelFileName = pieceFilePath + "_labels.xml";
    string labelMatrixName = "labels";
    cv::FileStorage fslabels(labelFileName, cv::FileStorage::WRITE);
    fslabels << labelMatrixName << labels;
    fslabels.release();
    */

    return nLabels;
}
