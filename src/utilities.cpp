#include <cv.h>
#include <cstring>
#include <highgui.h>
#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <unordered_set>
#include <utility>
#include <vector>
#include "head.h"
#include "otherheaders.h"

using namespace cv;
using namespace std;

/// TODO: @Yupeng - move the definition in "other headers.h"

void OpenCVImageWriter(const cv::Mat& matrixToBeWritten, string imageName) {
    double minXi, maxXi;
    double alpha, beta;
    Mat destIm;
    minMaxLoc(matrixToBeWritten, &minXi, &maxXi);

    alpha = 255.0 / (maxXi - minXi);
    beta = -minXi * 255.0 / (maxXi - minXi);
    matrixToBeWritten.convertTo(destIm, CV_8U, alpha, beta);
    imwrite(imageName, destIm);
}

void OpenCVFileWriter(const cv::Mat& matrixToBeWritten, string fileName, string matrixName) {
    cv::FileStorage fslabels(fileName, cv::FileStorage::WRITE);
    fslabels << matrixName << matrixToBeWritten;
    fslabels.release();
}

int ReadGTFile(vector<int>& gtSuperpixels, vector<int>& gtClassLabels, string gtFileName,
               cv::Mat& labels, vector<pair<int, int>>* gtCoords) {
    gtSuperpixels.clear();
    gtClassLabels.clear();

    // cout << "read GT file name = " << gtFileName << endl;
    assert(gtSuperpixels.empty() && gtClassLabels.empty());

    ifstream infile(gtFileName);
    int c, r, y;
    char d1, d2;
    if (infile.is_open()) {
        while ((infile >> c >> d1 >> r >> d2 >> y) && (d1 == ',') && (d2 == ',')) {
            // cout << c << " "<< r << " " << y << endl;
            gtSuperpixels.push_back(labels.at<int>(r, c));  /// TODO: check r,c order or x,y
            gtClassLabels.push_back(y);
            /* added for patch-based method */
            gtCoords->push_back(make_pair(r, c));
        }
    }
    infile.close();

    assert(gtSuperpixels.size() == gtClassLabels.size());

    return gtSuperpixels.size();
}

/* overload */
void ReadGTFile(string gtFileName, queue<int>* gtClassLabels, queue<pair<int, int>>* gtCoords) {
    ifstream infile(gtFileName);
    int c, r, y;
    char d1, d2;
    if (infile.is_open())
        while ((infile >> c >> d1 >> r >> d2 >> y) && (d1 == ',') && (d2 == ',')) {
            gtCoords->push(make_pair(r, c));
            gtClassLabels->push(y);
        }
    infile.close();
}

int PopulateUniqueVector(vector<int>& uniqueVec, int* origArr, int origLength) {
    uniqueVec.clear();
    int newLength = 0;
    for (int i = 0; i < origLength; i++) {
        int j = 0;
        for (j = 0; j < newLength; j++) {
            if (uniqueVec.at(j) == origArr[i]) {
                break;
            }
        }
        if (j == newLength) {
            uniqueVec.push_back(origArr[i]);
            newLength++;
        }
    }
    /* check */
    assert(uniqueVec.size() == newLength);
    return newLength;
}

void CreateSuperpixelGraph(vector<unordered_set<int>>& superpixelGraph, cv::Mat& labels,
                           vector<pair<int, int>>& bdry, int nLabels, int labelOffset, int* adjTRow,
                           int* adjBRow, int* adjLCol, int* adjRCol, int up, int down, int left, int right, int aTL,
                           int aTR, int aBL, int aBR, int tl, int tr, int bl, int br, int M, int N) {
    superpixelGraph.clear();
    superpixelGraph.resize(nLabels);
    int nodeLabel = labelOffset;

    /*
    for (auto it = superpixelGraph.begin(); it != superpixelGraph.end(); it++) {
        it->insert(nodeLabel++);
    }
    for (auto it = superpixelGraph.begin(); it != superpixelGraph.end(); it++) {
        it->push_back(nodeLabel++);
    }

    bool flagArray[nLabels][nLabels];
    for (int cntI = 0; cntI < nLabels; cntI++) {
        for (int cntJ = 0; cntJ < nLabels; cntJ++) {
            flagArray[cntI][cntJ] = false;
        }
    }

    for (int i = 0; i < bdry.size(); i++) {
        int I = bdry.at(i).first;
        int J = bdry.at(i).second;
        int labelIJ = labels.at<int>(I,J);

        int neighbor[8];

        neighbor[0] = ((J!=0) ? labels.at<int>(I,J-1) : ((left!=MPI_PROC_NULL)?adjLCol[I]:-1));
        neighbor[1] = ((J!=N-1) ? labels.at<int>(I,J+1) : ((right!=MPI_PROC_NULL)?adjRCol[I]:-1));
        neighbor[2] = ((I!=0 && J!=0) ? labels.at<int>(I-1,J-1) : ((tl!=MPI_PROC_NULL)?aTL:-1));
        neighbor[3] = ((I!=0) ? labels.at<int>(I-1,J) : ((up!=MPI_PROC_NULL)?adjTRow[J]:-1));
        neighbor[4] = ((I!=0 && J!=N-1) ? labels.at<int>(I-1,J+1) : ((tr!=MPI_PROC_NULL)?aTR:-1));
        neighbor[5] = ((I!=M-1 && J!=0) ? labels.at<int>(I+1,J-1) : ((bl!=MPI_PROC_NULL)?aBL:-1));
        neighbor[6] = ((I!=M-1) ? labels.at<int>(I+1,J) : ((down!=MPI_PROC_NULL)?adjBRow[J]:-1));
        neighbor[7] = ((I!=M-1 && J!=N-1) ? labels.at<int>(I+1,J+1) : ((br!=MPI_PROC_NULL)?aBR:-1));

        for (int j = 0; j < 8; j++) {
            if (labelIJ != neighbor[j] && neighbor[j] != -1) {
                superpixelGraph.at(labelIJ).insert(neighbor[j]);
            }
        }
    */
    for (int i = 0; i < bdry.size(); i++) {
        int I = bdry.at(i).first;
        int J = bdry.at(i).second;
        int labelIJ = labels.at<int>(I, J);
        int localLabel = labelIJ - labelOffset;

        assert(I >= 0 && I < M && J >= 0 && J < N);
        // cout << I << "\t " << J << endl;
        // cout << "past alert" << endl;

        int neighbor[8];

        neighbor[0] = ((J != 0) ? labels.at<int>(I, J - 1) : ((left != MPI_PROC_NULL) ? adjLCol[I] : -1));
        neighbor[1] = ((J != N - 1) ? labels.at<int>(I, J + 1) : ((right != MPI_PROC_NULL) ? adjRCol[I] : -1));
        neighbor[2] = ((I != 0 && J != 0) ? labels.at<int>(I - 1, J - 1) : ((tl != MPI_PROC_NULL) ? aTL : -1));
        neighbor[3] = ((I != 0) ? labels.at<int>(I - 1, J) : ((up != MPI_PROC_NULL) ? adjTRow[J] : -1));
        neighbor[4] = ((I != 0 && J != N - 1) ? labels.at<int>(I - 1, J + 1) : ((tr != MPI_PROC_NULL) ? aTR : -1));
        neighbor[5] = ((I != M - 1 && J != 0) ? labels.at<int>(I + 1, J - 1) : ((bl != MPI_PROC_NULL) ? aBL : -1));
        neighbor[6] = ((I != M - 1) ? labels.at<int>(I + 1, J) : ((down != MPI_PROC_NULL) ? adjBRow[J] : -1));
        neighbor[7] = ((I != M - 1 && J != N - 1) ? labels.at<int>(I + 1, J + 1) : ((br != MPI_PROC_NULL) ? aBR : -1));

        for (int j = 0; j < 8; j++)
            if (labelIJ != neighbor[j] && neighbor[j] != -1) {
                superpixelGraph.at(localLabel).insert(neighbor[j]);
            }
        /*
        for (int j = 0; j < 8; j++) {
            if (labelIJ != neighbor[j] && neighbor[j] != -1 && !flagArray[localLabel][neighbor[j]]) {
                superpixelGraph.at(localLabel).push_back(neighbor[j]);
                flagArray[localLabel][neighbor[j]] = true;
            }
        }
        */
    }
}

void Pieces2Gpb(vector<vector<cv::Mat>>& patchGpbOrient, vector<cv::Mat>& patchGpbNonmax, vector<cv::Mat>& gPbOrient,
                cv::Mat& gPbNmax, string outFilePrefix, int overlap, int rsz, int rows, int cols, int nOrient) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto clockStart = chrono::high_resolution_clock::now();
    log_info("Combining gPb subpatches within patch %d", rank);
    int sizes[3] = {rows, cols, nOrient};   /// TODO: bcast or not but pass
    int ndims = 3;
    // Mat maskAll = Mat::zeros(rows, cols, CV_64F);
    // Mat maskAll = Mat::zeros(rows, cols, CV_32S);

    /* size of parts */
    int txb = ceil((float)cols / rsz);
    int tyb = ceil((float)rows / rsz);
    cv::Mat temp;
    for(int i = 0; i < rsz; ++i) {
        for(int j = 0; j < rsz; ++j) {
            // Top left corner
            int xi = max(0, i * txb - overlap);
            int yi = max(0, j * tyb - overlap);

            // Bottom right corner
            int xe = min((i + 1) * txb + overlap - 1, cols - 1);
            int ye = min((j + 1) * tyb + overlap - 1, rows - 1);

            // Define region of interest
            Rect currentRegion(xi, yi, xe - xi + 1, ye - yi + 1);
            patchGpbNonmax[j * rsz + i].copyTo(gPbNmax(currentRegion));
            for(int orient = 0; orient < nOrient; ++orient) {
                patchGpbOrient[j * rsz + i][orient].copyTo(gPbOrient[orient](currentRegion));
            }
        }
    }
    //for (int sy = 0; sy <= floor((rows - 1) / tyb); sy++) {
        //for (int sx = 0; sx <= floor((cols - 1) / txb); sx++) {
            //int xi = max(0, sx * txb - overlap);
            //int xe = min((sx + 1) * txb + overlap - 1, cols - 1);

            //int yi = max(0, sy * tyb - overlap);
            //int ye = min((sy + 1) * tyb + overlap - 1, rows - 1);

            //int m = ye - yi + 1;
            //int n = xe - xi + 1;

            //// Mat maskPiece = Mat::ones(m,n,CV_32S);
            //Mat maskPiece = Mat::ones(m, n, CV_64F);
            ///// TODO: check for order efficiency
            //Mat auxMask = maskAll.colRange(xi, xe + 1).rowRange(yi, ye + 1);
            //// cout << "m of auxMask = " << auxMask.rows << "m = " << maskPiece.rows << endl;
            //// cout << "n of auxMask = " << auxMask.cols << "n = " << maskPiece.cols << endl;

            //// Mat auxMask = maskAll(Rect(xi,yi,n,m));
            //auxMask += maskPiece;

            //string outFile = outFilePrefix + "_" + to_string(sy) + "_" + to_string(sx);
            //string gPbPartFile = outFile + "_gPb.yml";
            //string nmaxPartFile = outFile + "_gPb_nmax.yml";
            //for (int i = 0; i < nOrient; i++) {
                //string gPbMatrixName = "gPb_" + to_string(i);
                //// cout << gPbMatrixName << endl;

                //[> read partFile <]
                //FileStorage fs(gPbPartFile, cv::FileStorage::READ);
                //Mat gPbPiece;
                //fs[gPbMatrixName] >> gPbPiece;
                //fs.release();

                //// imwrite(outFile + gPbMatrixName + ".png", gPbPiece);

                //// OpenCVFileWriter(gPbPiece, "labeledData/gPb_piece"+to_string(i)+".yml", "gPb_piece");

                //Mat auxGpbOrient = gPbOrient.at(i).colRange(xi, xe + 1).rowRange(yi, ye + 1);
                //// cout << "m of auxMask = " << auxMask.rows << "m = " << gPbPiece.rows << endl;
                //// cout << "n of auxMask = " << auxMask.cols << "n = " << gPbPiece.cols << endl;

                //// log_debug("I reached before addition");
                //// log_debug("m = %d\t%d", auxGpbOrient.rows, gPbPiece.rows);
                //// log_debug("m = %d\t%d", auxGpbOrient.cols, gPbPiece.cols);
                //auxGpbOrient += gPbPiece;
                //// log_debug("after addition");
                //// OpenCVFileWriter(gPbOrient.at(i), "labeledData/gPb_piece_aa"+to_string(i)+".yml", "gPb_piece_aa");
            //}

            //Mat gPbNmaxPiece;
            //FileStorage nmaxFile(nmaxPartFile, FileStorage::READ);
            //nmaxFile["gPb_nmax"] >> gPbNmaxPiece;
            ///* Copy from a smaller image to a larger one using a ROI, xi, yi specify cooridinate of
               //top-left pixel, additional arguments are the x, y offsets respectively */
            //gPbNmaxPiece.copyTo(gPbNmax(Rect(xi, yi, gPbNmaxPiece.cols, gPbNmaxPiece.rows)));
        //}
    //}

    /*
    Mat destMask(rows, cols, CV_64FC1);
    string imageMaskName = outFilePrefix+"_mask_.png";
    double alpha, beta, minXi, maxXi;
    minMaxLoc(maskAll, &minXi, &maxXi);
    cout << "minXi = " << minXi << " maxXi = " << maxXi << endl;
    cout << "mask val at 1 1 = " << maskAll.at<double>(1,1) << endl;
    alpha = 255.0/(maxXi-minXi);
    beta = -minXi*255.0/(maxXi-minXi);
    maskAll.convertTo(destMask, CV_8U, alpha, beta);
    // imwrite(imageMaskName, maskAll);
    imwrite(imageMaskName, destMask);
    */

    // OpenCVFileWriter(maskAll, outFilePrefix + "maskall.yml", "maskall");
    // OpenCVFileWriter(gPbOrient.at(0), outFilePrefix + "_gPbBeforeMask_" + to_string(0) + ".yml", "gPbBeforeMask");

    /*
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (maskAll.at<double>(i,j) == 0) {
                cout << "zero at i = " << i << " j = " << j << endl;
            }
        }
    }

    for (int i = 0; i < nOrient; i++) {
        gPbOrient.at(i) /= maskAll;
        divide(gPbOrient.at(i), maskAll, gPbOrient.at(i));
    }
    */

    /*
    for (int o = 0; o < nOrient; o++)
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                bool flag = (o == 0 && i >= 75 && i < 100 && j >= 75 && j < 100);
                if (flag) {
                    // log_debug("%lf\t%lf\t", gPbOrient.at(o).at<double>(i, j), maskAll.at<double>(i, j));
                }
                gPbOrient.at(o).at<double>(i, j) = gPbOrient.at(o).at<double>(i, j) / (int)maskAll.at<double>(i, j);
                if (flag) {
                    // log_debug("%lf\t%lf\t", gPbOrient.at(o).at<double>(i, j), maskAll.at<double>(i, j));
                }
            }
    */
    // OpenCVFileWriter(gPbOrient.at(0), outFilePrefix + "_gPbAfterMask_" + to_string(0) + ".yml", "gPbAfterMask");

    /*
    bool flag = false;
    for (int k = 0; k < nOrient; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (std::isnan(gPbOrient.at(k).at<double>(i, j)) && !flag) {
                    flag = true;
                    log_error("gPbOrient = nan at i = %d, j = %d, k = %d", i, j, k);
                }
            }
        }
    }
    */

    /* write to file */
    /*
    string gPbFileName = outFilePrefix + "_gPb.xml";
    FileStorage fs(gPbFileName, cv::FileStorage::WRITE);
    for (int i = 0; i < nOrient; i++) {
        string gPbMatrixName = "gPb_orient_" + to_string(i);
        fs << gPbMatrixName << gPbOrient.at(i);

        if (i == 0) {
            Mat destIm(rows, cols, CV_64FC1);
            string imageName = outFilePrefix + "_gPb_" + to_string(i) + ".png";
            double alpha, beta, minXi, maxXi;
            minMaxLoc(gPbOrient.at(i), &minXi, &maxXi);
            alpha = 255.0 / (maxXi - minXi);
            beta = -minXi * 255.0 / (maxXi - minXi);
            gPbOrient.at(i).convertTo(destIm, CV_8U, alpha, beta);
            imwrite(imageName, destIm);
        }
    }
    fs.release();
    */

    auto clockStop = chrono::high_resolution_clock::now();
    chrono::duration<double> clockElapsed = clockStop - clockStart;
    log_info("Combined gPb subpatches within patch %d [%.3f sec]", rank, clockElapsed.count());
}

MPI_Datatype MpiTypeFor(int cvType, int nChannels)
{
    // Each channel occupies 8 bits, so to get the type index for the CV_XXC1 representation
    // Subtract 8 * number of excess channels
    cvType -= (nChannels - 1) * 8;

    // Maps single channel CV type to corresponding MPI data type
    MPI_Datatype bufType[] = {
        MPI_UNSIGNED_CHAR,  /* CV_8UC*  */
        MPI_CHAR,           /* CV_8SC*  */
        MPI_UNSIGNED_SHORT, /* CV_16UC* */
        MPI_SHORT,          /* CV_16SC* */
        MPI_INT,            /* CV_32SC* */
        MPI_FLOAT,          /* CV_32FC* */
        MPI_DOUBLE          /* CV_64FC*  */
    };

    // Now cvType should in [0, 6] (see links in header comments)
    return bufType[cvType];
}

void MpiSendMat(cv::Mat& image, int destId) {
    int dims[4] = {image.rows, image.cols, image.channels(), image.type()};

    // First send the destination process the dimensions of the image
    MPI_Send(dims, 4, MPI_INT, destId, 0, MPI_COMM_WORLD);
    if(!image.isContinuous()) {
        image = image.clone();
    }

    // Extract the type information so that send interprets image buffer correctly
    MPI_Datatype bufType = MpiTypeFor(dims[3], dims[2]);

    // Now send the data buffer of uchars that is the Matrix
    MPI_Send(image.data, dims[0] * dims[1] * dims[2], bufType, destId, 1, MPI_COMM_WORLD);
}

cv::Mat MpiRecvMat(int sourceId) {
    int dims[4];
    MPI_Status status;
    cv::Mat image;

    MPI_Recv(dims, 4, MPI_INT, sourceId, 0, MPI_COMM_WORLD, &status);

    // Extract the type information so that send interprets image buffer correctly
    MPI_Datatype bufType = MpiTypeFor(dims[3], dims[2]);

    image = cv::Mat(dims[0], dims[1], dims[3]);

    MPI_Recv(image.data, dims[0] * dims[1] * dims[2], bufType, sourceId, 1, MPI_COMM_WORLD, &status);
    return image;
}

void MpiSendMatVec(vector<cv::Mat>& images, int destId) {
    int dims[4];
    int count = images.size();
    cv::Mat image;
    MPI_Send(&count, 1, MPI_INT, destId, 0, MPI_COMM_WORLD);

    for(int i = 0; i < count; ++i) {
        image = images[i];
        dims[0] = image.rows;
        dims[1] = image.cols;
        dims[2] = image.channels();
        dims[3] = image.type();

        // First send the destination process the dimensions of the image
        MPI_Send(dims, 4, MPI_INT, destId, 2 * i + 1, MPI_COMM_WORLD);
        if(!image.isContinuous()) {
            image = image.clone();
        }

        MPI_Datatype bufType = MpiTypeFor(dims[3], dims[2]);
        // Now send the data buffer of uchars that is the Matrix
        MPI_Send(image.data, dims[0] * dims[1] * dims[2], bufType, destId, 2 * i + 2, MPI_COMM_WORLD);
    }
}

vector<cv::Mat> MpiRecvMatVec(int sourceId) {
    int dims[4];
    int count;
    MPI_Status status;
    cv::Mat image;
    vector<cv::Mat> images;

    MPI_Recv(&count, 1, MPI_INT, sourceId, 0, MPI_COMM_WORLD, &status);

    for(int i = 0; i < count; ++i) {
        MPI_Recv(dims, 4, MPI_INT, sourceId, 2 * i + 1, MPI_COMM_WORLD, &status);
        image = cv::Mat(dims[0], dims[1], dims[3]);

        MPI_Datatype bufType = MpiTypeFor(dims[3], dims[2]);

        MPI_Recv(image.data, dims[0] * dims[1] * dims[2],
                 bufType, sourceId, 2 * i + 2, MPI_COMM_WORLD, &status);
        images.push_back(image);
    }

    return images;
}

vector<cv::Mat> Img2Pieces(string basePrefix, cv::Mat& bigImg, int overlap, int nPieces) {
    int nChan =  bigImg.channels();
    Size origSize = bigImg.size();
    int rows = origSize.height;
    int cols = origSize.width;
    int nPixels = rows * cols;
    vector<cv::Mat> patches;

    /* pieces in each dimension */
    //int rsz = ceil(sqrt((float)nPixels / (float)nPieces));    // removed by Yupeng
    int rsz = sqrt(nPieces);
    log_debug("nPixels: %d, nPieces: %d, rsz: %d", nPixels, nPieces, rsz);

    /* size of parts */
    int txb = ceil((float)cols / rsz);
    int tyb = ceil((float)rows / rsz);

    log_debug("txb: %d, tyb: %d", txb, tyb);

    /* read the groundtruth file */
    string gtFileName = basePrefix + "_GT.txt";
    queue<int> gtClassLabels;
    queue<pair<int, int>> gtCoords;
    ReadGTFile(gtFileName, &gtClassLabels, &gtCoords);

    for (int sy = 0; sy < rsz; sy++) {
        for (int sx = 0; sx < rsz; sx++) {
            int xi = max(0, sx * txb - overlap);
            int xe = min((sx + 1) * txb + overlap - 1, cols - 1);

            int yi = max(0, sy * tyb - overlap);
            int ye = min((sy + 1) * tyb + overlap - 1, rows - 1);

            Point topLeft(xi, yi);
            Point bottomRight(xe + 1, ye + 1);
            Rect R(topLeft, bottomRight);

            Mat pieceImg = bigImg(R);
            patches.push_back(pieceImg);
            string prefix = basePrefix + "_" + to_string(sy) + "_" + to_string(sx);

            /// TODO: @Yupeng - divided the groundtruth file
            ofstream ofile;
            string gtTextName = prefix + "_GT.txt";
            ofile.open(gtTextName.c_str());
            int pointCount = gtClassLabels.size();
            // cout << "xi = " << xi << "\txe = " << xe << "\tyi = " << yi << "\tye = " << ye << endl;
            for (int k = 0; k < pointCount; k++) {
                pair<int, int> curr_point = gtCoords.front();
                gtCoords.pop();
                int currLabel = gtClassLabels.front();
                gtClassLabels.pop();
                // gtCoords(<row, col>)
                if (curr_point.first >= yi && curr_point.first <= ye && curr_point.second >= xi &&
                    curr_point.second <= xe) {
                    curr_point.first -= yi;
                    curr_point.second -= xi;
                    // output: first column #, then row #
                    ofile << curr_point.second << ',' << curr_point.first << ',' << currLabel << '\n';
                    // cout << "first = " << curr_point.first << "\tsecond = " << curr_point.second
                    //      << "\tlabel = " << currLabel << endl;

                    /// FIXME: @Yupeng - test the true position
                    /*
                    if(currLabel == 1) {
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[0] = 0;
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[1] = 0;
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[2] = 255;
                    }
                    else if(currLabel == 2) {
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[0] = 255;
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[1] = 255;
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[2] = 255;
                    }
                    else if(currLabel == 3) {
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[0] = 0;
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[1] = 255;
                            pieceImg.at<cv::Vec3b>(curr_point.first, curr_point.second)[2] = 0;
                    }*/
                }
                else {
                    gtCoords.push(curr_point);
                    gtClassLabels.push(currLabel);
                }
            }
            ofile.close();

            // string imageName = prefix + ".png";
            // imwrite(imageName, pieceImg);
        }
    }
    log_info("basePrefix: %s\tremaining #coords: %lu", basePrefix.c_str(), gtCoords.size());

    return patches;
}

/// FIXME: @Yupeng - assume no overlapping!
void Pieces2Mat(string pieceFilePath, string mark, int overlap, int nPieces, cv::Mat& targetMat,
                std::vector<cv::Mat> dataVector) {
    cv::Size origSize = targetMat.size();
    int rows = origSize.height;
    int cols = origSize.width;
    int nPixels = rows * cols;
    /* pieces in each dimension */
    // int rsz = ceil(sqrt((float)nPixels / (float)nPieces));   // removed by Yupeng
    int rsz = sqrt(nPieces);
    /* size of parts */
    int txb = ceil((float)cols / rsz);
    int tyb = ceil((float)rows / rsz);

    for (int sy = 0; sy < rsz; sy++) {
        for (int sx = 0; sx < rsz; sx++) {
            int xi = max(0, sx * txb - overlap);
            int xe = min((sx + 1) * txb + overlap - 1, cols - 1);
            int yi = max(0, sy * tyb - overlap);
            int ye = min((sy + 1) * tyb + overlap - 1, rows - 1);

            Point topLeft(xi, yi);
            Point bottomRight(xe + 1, ye + 1);
            Rect R(topLeft, bottomRight);

            // string prefix = pieceFilePath + "_" + to_string(sy) + "_" + to_string(sx);
            /* Read Yml File */
            // cv::FileStorage fs(prefix + "_classlabels.yml", cv::FileStorage::READ);
            // cv::Mat classLabels = classLabelMatVector[sy * rsz + sx];
            cv::Mat data = dataVector[sy * rsz + sx];
            data.copyTo(targetMat(R));
            // fs["classLabels"] >> classLabels;
            // classLabels.copyTo(targetLabelMap(R));
            // fs.release();
            /* Read Png file */
            // cv::Mat imagePiece = cv::imread(prefix + "_classlabels.png");
            // classLabels.copyTo(targetBigImg(R));
            /* Read Yml File */
            // fs = cv::FileStorage(prefix + "_ucm.yml", cv::FileStorage::READ);
            // cv::Mat ucmMap = ucmDataVector[sy * rsz + sx];
            // fs["ucm"] >> ucmMap;
            // ucmMap.copyTo(targetUcmMap(R));
            // fs.release();
        }
    }

    /* Save the results */
    /*
     * cv::FileStorage fs(pieceFilePath + mark + "_classlabels.yml", cv::FileStorage::WRITE);
     * fs << "classLabels" << targetLabelMap;
     * fs.release();
     * log_info("Class Label Map is written");
     * cv::imwrite(pieceFilePath + mark + "_classlabels.png", targetBigImg);
     * log_info("Class Image Map is written");
     * fs = cv::FileStorage(pieceFilePath + mark + "_ucm.yml", cv::FileStorage::WRITE);
     * fs << "ucmMap" << targetUcmMap;
     * fs.release();
     * log_info("UCM Map is written");
     */
}

// crop the training patches globally
std::string padStrZeros(int n_zero, std::string old_string) {
    return std::string(n_zero - old_string.length(), '0') + old_string;
}

std::string getTrainingPatches(std::string basePrefix, cv::Mat& bigImg, int rows, int cols) {
    Size bigOrigSize = bigImg.size();
    assert(rows == bigOrigSize.height);
    assert(cols == bigOrigSize.width);
    std::string imagePath = basePrefix.substr(0, basePrefix.find_last_of("/") + 1) + "gtImages/";

    // gtCoords save in the format of (r, c)
    string gtFileName = basePrefix + "_GT.txt";
    queue<int> gtClassLabels;
    queue<pair<int, int>> gtCoords;
    ReadGTFile(gtFileName, &gtClassLabels, &gtCoords);

    int patchSize = 64;
    Size rgbPatchSize(patchSize, patchSize);
    int cnt = 0;
    while(!gtCoords.empty()) {
        pair<int, int> curr_point = gtCoords.front();
        gtCoords.pop();
        int currLabel = gtClassLabels.front();
        gtClassLabels.pop();
        
        int north = min(patchSize / 2, curr_point.first - 0);
        int south = min(patchSize / 2, rows - 1 - curr_point.first);
        int west = min(patchSize / 2, curr_point.second - 0);
        int east = min(patchSize / 2, cols - 1 - curr_point.second);

        cv::Rect r(curr_point.second - west, curr_point.first - north, west + east + 1, north + south + 1);
        cv::Mat rgbPatch;
        bigImg(r).copyTo(rgbPatch);
        cv::resize(rgbPatch, rgbPatch, rgbPatchSize);
        std::string patchFileName = imagePath + "Class_" + padStrZeros(1, std::to_string(currLabel))
            +"_Seed_" + padStrZeros(3, std::to_string(cnt)) + ".jpg";
        imwrite(patchFileName, rgbPatch);
        cnt++;
    }
    return imagePath;
}
