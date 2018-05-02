#include <queue>

#include "head.h"

using namespace std;
using namespace cv;

vector<RegionpropsObject> regionprops(Mat labels, string option) {
    /* only positive numbers are considered as "region" */
    int nLabels = 0;
    vector<RegionpropsObject> regions;

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            nLabels = (labels.at<int>(i, j) > nLabels) ? labels.at<int>(i, j) : nLabels;
        }
    }
    for (int i = 0; i < nLabels; i++) {
        RegionpropsObject object; /* object is also a vector */
        regions.push_back(object);
    }

    if (option == "PixelList") {
        for (int i = 0; i < labels.rows; i++) {
            for (int j = 0; j < labels.cols; j++) {
                if (labels.at<int>(i, j) > 0) {
                    coordinates pair;
                    pair.row = i;
                    pair.col = j;
                    /* index starts from zero */
                    regions.at(labels.at<int>(i, j) - 1).PixelList.push_back(pair);
                }
            }
        }
    }
    return regions;
}

Mat bwmorph(Mat& binaryImg, string option, int numNeighbor) {
    int checkround = 0;
    if (option == "clean") {
        Mat binaryImg_tmp = binaryImg.clone();
        while (++checkround) {  /// TODO: "inf" each time (default)
            bool isStable = true;
            bool isIsolated;
            for (int i = 0; i < binaryImg.rows; i++) {
                for (int j = 0; j < binaryImg.cols; j++) {
                    if (binaryImg.at<unsigned char>(i, j) == 1) {
                        isIsolated = true;
                        if (i - 1 >= 0) {
                            isIsolated &= (binaryImg.at<unsigned char>(i - 1, j) == 0);
                        }
                        if (j - 1 >= 0) {
                            isIsolated &= (binaryImg.at<unsigned char>(i, j - 1) == 0);
                        }
                        if (j + 1 < binaryImg.cols) {
                            isIsolated &= (binaryImg.at<unsigned char>(i, j + 1) == 0);
                        }
                        if (i + 1 < binaryImg.rows) {
                            isIsolated &= (binaryImg.at<unsigned char>(i + 1, j) == 0);
                        }
                        if (numNeighbor == 8) {
                            if (i - 1 >= 0 & j - 1 >= 0) {
                                isIsolated &= (binaryImg.at<unsigned char>(i - 1, j - 1) == 0);
                            }
                            if (i - 1 >= 0 & j + 1 < binaryImg.cols) {
                                isIsolated &= (binaryImg.at<unsigned char>(i - 1, j + 1) == 0);
                            }
                            if (i + 1 < binaryImg.rows & j - 1 >= 0) {
                                isIsolated &= (binaryImg.at<unsigned char>(i + 1, j - 1) == 0);
                            }
                            if (i + 1 < binaryImg.rows & j + 1 < binaryImg.cols) {
                                isIsolated &= (binaryImg.at<unsigned char>(i + 1, j + 1) == 0);
                            }
                        }
                        if (isIsolated == true) {
                            binaryImg_tmp.at<unsigned char>(i, j) = 0;
                            isStable = false;
                        }
                    }
                }
            }
            binaryImg = binaryImg_tmp.clone();
            if (isStable == true) {
                break;
            }
        }
    }
    return binaryImg;
}

void bwlabelCore(int i, int j, int numRegion, Mat& labels, int numNeighbor) {
    labels.at<int>(i, j) = numRegion;
    int row[8] = {i - 1, i, i, i + 1, i - 1, i - 1, i + 1, i + 1};
    int col[8] = {j, j - 1, j + 1, j, j - 1, j + 1, j - 1, j + 1};

    int rows = labels.rows;
    int cols = labels.cols;

    for (int k = 0; k < numNeighbor; k++) {
        if ((row[k] > -1) && (row[k] < rows) && (col[k] > -1) && (col[k] < cols)) {
            if (labels.at<int>(row[k], col[k]) == 1) {
                bwlabelCore(row[k], col[k], numRegion, labels, numNeighbor);
            }
        }
    }
}

Mat bwlabelYupeng(Mat& bw, int numNeighbor) {
    Mat labels = bw.clone();

    int nLabels = 1;  /// FIXME: changed by Manu from 1 to 0
    for (int j = 0; j < labels.cols; j++) {
        for (int i = 0; i < labels.rows; i++) {
            if (labels.at<int>(i, j) == 1) {
                nLabels += 1;
                bwlabelCore(i, j, nLabels, labels, numNeighbor); /* pass by reference */
            }
        }
    }

    for (int m = 0; m < labels.rows; m++) {
        for (int n = 0; n < labels.cols; n++) {
            if (labels.at<int>(m, n) > 0) {
                labels.at<int>(m, n) = labels.at<int>(m, n) - 1; /* start from index 1 */
            }
        }
    }
    return labels;
}

int bwlabelYupeng(Mat& labels, Mat& bw, int numNeighbor) {
    labels = bw.clone();

    int nLabels = 1;
    for (int j = 0; j < labels.cols; j++) {
        for (int i = 0; i < labels.rows; i++) {
            if (labels.at<int>(i, j) == 1) {
                nLabels += 1;
                bwlabelCore(i, j, nLabels, labels, numNeighbor); /* pass by reference */
            }
        }
    }

    for (int m = 0; m < labels.rows; m++) {
        for (int n = 0; n < labels.cols; n++) {
            /* start from index 1 */
            if (labels.at<int>(m, n) > 0) labels.at<int>(m, n) = labels.at<int>(m, n) - 1;
        }
    }
    return nLabels;
}

bool checkValidity(Mat& bw, Mat& labels, int i, int j, int m, int n, int nLabels) {
    if ((i >= 0) && (i < m) && (j >= 0) && (j < n))
        if (bw.at<int>(i, j) == 1 && labels.at<int>(i, j) == 0) {
            labels.at<int>(i, j) = nLabels;
            return true;
        }

    return false;
}

int bwlabelManu(Mat& labels, Mat& bw, int numNeighbor) {
    int m = bw.rows;
    int n = bw.cols;

    labels = Mat::zeros(m, n, CV_32S);

    int nLabels = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (bw.at<int>(i, j) == 1 && labels.at<int>(i, j) == 0) {
                queue<pair<int, int> > pixelQ;
                pixelQ.push(make_pair(i, j));
                nLabels += 1;
                labels.at<int>(i, j) = nLabels;

                while (!pixelQ.empty()) {
                    pair<int, int> p = pixelQ.front();
                    pixelQ.pop();
                    int I = p.first;
                    int J = p.second;

                    /* check neighbors */
                    if (checkValidity(bw, labels, I, J - 1, m, n, nLabels)) {
                        pixelQ.push(make_pair(I, J - 1));
                    }
                    if (checkValidity(bw, labels, I, J + 1, m, n, nLabels)) {
                        pixelQ.push(make_pair(I, J + 1));
                    }
                    if (checkValidity(bw, labels, I - 1, J - 1, m, n, nLabels)) {
                        pixelQ.push(make_pair(I - 1, J - 1));
                    }
                    if (checkValidity(bw, labels, I - 1, J, m, n, nLabels)) {
                        pixelQ.push(make_pair(I - 1, J));
                    }
                    if (checkValidity(bw, labels, I - 1, J + 1, m, n, nLabels)) {
                        pixelQ.push(make_pair(I - 1, J + 1));
                    }
                    if (checkValidity(bw, labels, I + 1, J - 1, m, n, nLabels)) {
                        pixelQ.push(make_pair(I + 1, J - 1));
                    }
                    if (checkValidity(bw, labels, I + 1, J, m, n, nLabels)) {
                        pixelQ.push(make_pair(I + 1, J));
                    }
                    if (checkValidity(bw, labels, I + 1, J + 1, m, n, nLabels)) {
                        pixelQ.push(make_pair(I + 1, J + 1));
                    }
                }
            }
        }
    }
    /// TODO: nLabels+1 was changed to nLabels -> check
    return nLabels;
}

Mat bwlabel(Mat& bw, int numNeighbor) {
    Mat labels;
    bwlabelManu(labels, bw, numNeighbor);
    return labels;
}

/*
Mat bwlabel(Mat& bw, int numNeighbor) {
    int m = bw.rows;
    int n = bw.cols;

    Mat labels = Mat::zeros(m, n, CV_32S);

    int nLabels = 0;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            if (checkValidity(bw, labels, i, j, m, n)) {
                queue<pair<int, int> > pixelQ;
                pixelQ.push(make_pair(i, j));
                nLabels += 1;

                while (!pixelQ.empty()) {
                    pair<int, int> p = pixelQ.front();
                    pixelQ.pop();
                    int I = p.first;
                    int J = p.second;

                    labels.at<int>(I, J) = nLabels;

                    // check neighbours
                    if (checkValidity(bw, labels, I, J - 1, m, n)) {
                        pixelQ.push(make_pair(I, J - 1));
                    }
                    if (checkValidity(bw, labels, I, J + 1, m, n)) {
                        pixelQ.push(make_pair(I, J + 1));
                    }
                    if (checkValidity(bw, labels, I - 1, J - 1, m, n)) {
                        pixelQ.push(make_pair(I - 1, J - 1));
                    }
                    if (checkValidity(bw, labels, I - 1, J, m, n)) {
                        pixelQ.push(make_pair(I - 1, J));
                    }
                    if (checkValidity(bw, labels, I - 1, J + 1, m, n)) {
                        pixelQ.push(make_pair(I - 1, J + 1));
                    }
                    if (checkValidity(bw, labels, I + 1, J - 1, m, n)) {
                        pixelQ.push(make_pair(I + 1, J - 1));
                    if (checkValidity(bw, labels, I + 1, J, m, n)) {
                        pixelQ.push(make_pair(I + 1, J));
                    }
                    if (checkValidity(bw, labels, I + 1, J + 1, m, n)) {
                        pixelQ.push(make_pair(I + 1, J + 1));
                    }

                }
            }

        }

    return labels;

}
*/

void meshgrid(Mat& XMat, Mat& YMat, int startX, int endX, int startY, int endY) {
    int n = endX - startX + 1;
    int m = endY - startY + 1;
    int* XVec = new int[n];
    int* YVec = new int[m];

    for (int i = 0; i < n; i++) {
        XVec[i] = i + startX;
    }

    for (int j = 0; j < m; j++) {
        YVec[j] = j + startY;
    }

    Mat X(1, n, CV_32S, (char*)XVec);
    Mat Y(m, 1, CV_32S, (char*)YVec);

    repeat(X, m, 1, XMat);
    repeat(Y, 1, n, YMat);
}

void im2col(Mat& Out, Mat& In, int radius, int check) {
    int sz = 2 * radius + 1;
    int m = In.rows;
    int n = In.cols;

    int Mout = m - 2 * radius;
    int Nout = n - 2 * radius;
    // int cnt = 0;
    Out.create(sz * sz, Mout * Nout, CV_32S);
    for (int i = 0; i < Mout; i++)
        for (int j = 0; j < Nout; j++) {
            cv::Mat subMat1 = In.colRange(j, j + sz).rowRange(i, i + sz);
            cv::Mat subMat;
            subMat1.copyTo(subMat);

            // cout << subMat.rows << "\t" << m << "\t" << sz << "\t" << subMat.cols << "\t" << n << "\t" << sz << endl;
            assert(subMat.rows == sz && subMat.cols == sz);

            subMat = subMat.reshape(0, sz * sz);
            // cout << i << "\t" << j << endl;
            cv::Mat Out1 = Out.col(i * Nout + j);
            // cout << subMat.rows << "\t" << Out1.rows << "\t" << subMat.cols << "\t" << Out1.cols << endl;
            assert(subMat.rows == Out1.rows && subMat.cols == Out1.cols);
            subMat.copyTo(Out.col(i * Nout + j));
            // cv::hconcat(Out, subMat, Out);
        }

    /// FIXME: change it to a functor
    /*
    if (check != -1) {
        assert(Out.cols == check);
    }
    */
}
