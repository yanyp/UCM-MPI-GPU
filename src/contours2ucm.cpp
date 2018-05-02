#include <chrono>
#include <cstring>

#include "head.h"
#include "otherheaders.h"

using namespace std;
using namespace cv;

void Imsave(string filename, Mat &matrix) {
    Mat img = Matrix2Image(matrix);
    imwrite(filename, img);
}

void ImsaveScale(string filename, Mat &matrix) {
    double minmin, maxmax;
    minMaxLoc(matrix, &minmin, &maxmax);
    // cout << "minmin = " << minmin << '\t' << "maxmax = " << maxmax << endl;
    Mat matrixPrint = matrix.clone();
    matrix.convertTo(matrixPrint, CV_8U, (int)255 / maxmax);
    Imsave(filename, matrixPrint);
}

superContour4COutput SuperContour4C(Mat &pb) {
    int tx = pb.rows;
    int ty = pb.cols;
    /* MATLAB
    V = min(pb(1:end-1,:), pb(2:end,:));
    H = min(pb(:,1:end-1), pb(:,2:end));
    */
    Mat V(tx - 1, ty, CV_64F);
    for (int i = 0; i < V.rows; i++) {
        for (int j = 0; j < V.cols; j++) {
            V.at<double>(i, j) = min(pb.at<double>(i, j), pb.at<double>(i + 1, j));
        }
    }
    Mat H(tx, ty - 1, CV_64F);
    for (int i = 0; i < H.rows; i++) {
        for (int j = 0; j < H.cols; j++) {
            H.at<double>(i, j) = min(pb.at<double>(i, j), pb.at<double>(i, j + 1));
        }
    }

    /* MATLAB
    [tx, ty] = size(pb);
    pb2 = zeros(2*tx, 2*ty);
    pb2(1:2:end, 1:2:end) = pb;
    pb2(1:2:end, 2:2:end-2) = H;
    pb2(2:2:end-2, 1:2:end) = V;
    pb2(end,:) = pb2(end-1, :);
    pb2(:,end) = max(pb2(:,end), pb2(:,end-1));
    */
    superContour4COutput res;
    Mat pb2 = Mat::zeros(2 * tx, 2 * ty, CV_64F);
    for (int i = 0; i < pb2.rows; i = i + 2) {
        for (int j = 0; j < pb2.cols; j = j + 2) {
            pb2.at<double>(i, j) = pb.at<double>(i / 2, j / 2);
            if (j != pb2.cols - 2) {
                pb2.at<double>(i, j + 1) = H.at<double>(i / 2, j / 2);
            }
            if (i != pb2.rows - 2) {
                pb2.at<double>(i + 1, j) = V.at<double>(i / 2, j / 2);
            }
        }
    }
    for (int j = 0; j < pb2.cols; j++) {
        pb2.at<double>(pb2.rows - 1, j) = pb2.at<double>(pb2.rows - 2, j);
    }
    for (int i = 0; i < pb2.rows; i++) {
        pb2.at<double>(i, pb2.cols - 1) = max(pb2.at<double>(i, pb2.cols - 1), pb2.at<double>(i, pb2.cols - 2));
    }

    /* outout */
    res.pb2 = pb2;
    res.V = V;
    res.H = H;
    return res;
}

Mat CleanWatersheds(Mat &wsClean) {
    /* MATLAB
    wsClean = ws;   // input is ws
    c = bwmorph(wsClean == 0, 'clean', inf);
    */
    int rows = wsClean.rows;
    int cols = wsClean.cols;
    Mat wsClean_tmp = Mat::zeros(rows, cols, CV_8U);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (wsClean.at<double>(i, j) == 0) {
                wsClean_tmp.at<unsigned char>(i, j) = 1;
            }
        }
    }
    log_debug("At bwmorph");   /* TODO temp, delete later*/
    Mat c = bwmorph(wsClean_tmp, "clean", 8);   /* pass by reference */

    /* MATLAB
    artifacts = ( c==0 & wsClean==0 );
    R = regionprops(bwlabel(artifacts), 'PixelList');
    */
    Mat artifacts = Mat::zeros(rows, cols, CV_32S);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            /* since _tmp matrix is the wsClean==0 mark */
            if (c.at<unsigned char>(i, j) == 0 & wsClean.at<double>(i, j) == 0) {
                artifacts.at<int>(i, j) = 1;
            }
        }
    }
    log_debug("At regionprops");   /* TODO temp, delete later*/
    vector<RegionpropsObject> R = regionprops(bwlabel(artifacts, 8), "PixelList");

    log_debug("Entering regionprops loop");   /* TODO temp, delete later*/
    int xc, yc;
    for (int r = 0; r < R.size(); r++) {
        /* MATLAB
        xc = R(r).PixelList(1,2);
        yc = R(r).PixelList(1,1);
        */
        /* only use PixelList(1,?) because of isolated regions */
        xc = R.at(r).PixelList.begin()->row;
        /* no offset "-1" here - (1,2) for row and (1,1) for col */
        yc = R.at(r).PixelList.begin()->col;
        // cout << "xc = " << xc << "\tyc = " << yc << endl;
        double vec[4];
        vec[0] = max(wsClean.at<double>(xc - 2, yc - 1), wsClean.at<double>(xc - 1, yc - 2));
        vec[1] = max(wsClean.at<double>(xc + 2, yc - 1), wsClean.at<double>(xc + 1, yc - 2));
        vec[2] = max(wsClean.at<double>(xc + 2, yc + 1), wsClean.at<double>(xc + 1, yc + 2));
        vec[3] = max(wsClean.at<double>(xc - 2, yc + 1), wsClean.at<double>(xc - 1, yc + 2));

        /* MATLAB
        [nd,id] = min(vec);
        */
        double nd = vec[0];
        int id = 0;
        for (int i = 1; i < 4; i++) {
            if (vec[i] < nd) {
                id = i;
                nd = vec[i];
            }
        }

        switch (id) {
            case 0:
                if (wsClean.at<double>(xc - 2, yc - 1) < wsClean.at<double>(xc - 1, yc - 2)) {
                    wsClean.at<double>(xc, yc - 1) = 0;
                    wsClean.at<double>(xc - 1, yc) = vec[0];
                } else {
                    wsClean.at<double>(xc, yc - 1) = vec[0];
                    wsClean.at<double>(xc - 1, yc) = 0;
                }
                wsClean.at<double>(xc - 1, yc - 1) = vec[0];
                break;
            case 1:
                if (wsClean.at<double>(xc + 2, yc - 1) < wsClean.at<double>(xc + 1, yc - 2)) {
                    wsClean.at<double>(xc, yc - 1) = 0;
                    wsClean.at<double>(xc + 1, yc) = vec[1];
                } else {
                    wsClean.at<double>(xc, yc - 1) = vec[1];
                    wsClean.at<double>(xc + 1, yc) = 0;
                }
                wsClean.at<double>(xc + 1, yc - 1) = vec[1];
                break;
            case 2:
                if (wsClean.at<double>(xc + 2, yc + 1) < wsClean.at<double>(xc + 1, yc + 2)) {
                    wsClean.at<double>(xc, yc + 1) = 0;
                    wsClean.at<double>(xc + 1, yc) = vec[2];
                } else {
                    wsClean.at<double>(xc, yc + 1) = vec[2];
                    wsClean.at<double>(xc + 1, yc) = 0;
                }
                wsClean.at<double>(xc + 1, yc + 1) = vec[2];
                break;
            case 3:
                if (wsClean.at<double>(xc - 2, yc + 1) < wsClean.at<double>(xc - 1, yc + 2)) {
                    wsClean.at<double>(xc, yc + 1) = 0;
                    wsClean.at<double>(xc - 1, yc) = vec[3];
                } else {
                    wsClean.at<double>(xc, yc + 1) = vec[3];
                    wsClean.at<double>(xc - 1, yc) = 0;
                }
                wsClean.at<double>(xc - 1, yc + 1) = vec[3];
                break;
        }
    }
    log_debug("Done with regionprops loop");   /* TODO temp, delete later*/
    return wsClean;
}

Mat NormalizeOutput(Mat pb) {
    /* MATLAB
    [tx, ty] = size(pb);
    beta = [-2.7487; 11.1189];
    pbNorm = pb(:);
    x = [ones(size(pbNorm)) pbNorm]';
    pbNorm = 1 ./ (1 + (exp(-x'*beta)));
    pbNorm = (pbNorm - 0.0602) / (1 - 0.0602);
    pbNorm=min(1,max(0,pbNorm));
    pbNorm = reshape(pbNorm, [tx ty]);
    */
    int tx = pb.rows;
    int ty = pb.cols;
    Mat beta(2, 1, CV_64F);
    beta.at<double>(0, 0) = -2.7487;
    beta.at<double>(1, 0) = 11.1189;
    Mat pbNorm(tx * ty, 1, CV_64F);
    for (int i = 0; i < tx; i++) {
        for (int j = 0; j < ty; j++) {
            pbNorm.at<double>(i + j * tx, 0) = pb.at<double>(i, j);     /* col first */
        }
    }
    OpenCVFileWriter(pbNorm, "imagedata/pb_norm_initial.yml", "pb_norm");
    Mat x(tx * ty, 1 + 1, CV_64F);      /* no need to implement transpose */
    for (int i = 0; i < tx * ty; i++) {
        x.at<double>(i, 0) = 1;
        x.at<double>(i, 1) = pbNorm.at<double>(i, 0);
    }
    Mat comp = x * beta;    /* (tx*ty,1) */
    log_debug("Inside NormalizeOutPut: attempt to write imagedata");   /* TODO temp, delete later*/
    OpenCVFileWriter(x, "imagedata/x.yml", "x");
    OpenCVFileWriter(comp, "imagedata/comp.yml", "comp");
    for (int i = 0; i < tx * ty; i++) {
        pbNorm.at<double>(i, 0) = 1.0 / (1.0 + std::exp(-comp.at<double>(i, 0)));
        if (i < 100) {
            // log_debug("%d\tpbNorm: %lf", i, pbNorm.at<double>(i, 0));
        }

        pbNorm.at<double>(i, 0) = (pbNorm.at<double>(i, 0) - 0.0602) / (1.0 - 0.0602);
        if (i < 100) {
            // log_debug("%d\tpbNorm: %lf", i, pbNorm.at<double>(i, 0));
        }

        pbNorm.at<double>(i, 0) = std::min(1.0, std::max(0.0, pbNorm.at<double>(i, 0)));
        if (i < 100) {
            // log_debug("%d\tpbNorm: %lf", i, pbNorm.at<double>(i, 0));
        }
    }
    Mat pbNormFinal(tx, ty, CV_64F);
    for (int i = 0; i < tx; i++) {
        for (int j = 0; j < ty; j++) {
            pbNormFinal.at<double>(i, j) = pbNorm.at<double>(i + j * tx);
        }
    }
    OpenCVFileWriter(pbNormFinal, "imagedata/pb_norm.yml", "pb_norm");
    OpenCVFileWriter(pbNormFinal, "imagedata/pb_norm_final.yml", "pb_norm_final");
    log_debug("End of NormalizeOutPut: attempt to write imagedata");   /* TODO temp, delete later*/
    return pbNormFinal;
}

Mat CreateFinestPartition(std::string pieceFilePat, vector<Mat> &pbOriented) {
    /* MATLAB
    pb = max(pbOriented,[],3);
    */
    Mat pb = pbOriented.at(0).clone();
    int pb_rows = pb.rows;
    int pb_cols = pb.cols;

    log_debug("Before pb loop");   /* TODO temp, delete later*/
    for (int k = 1; k < pbOriented.size(); k++) {
        for (int i = 0; i < pb_rows; i++) {
            for (int j = 0; j < pb_cols; j++) {
                if (pb.at<double>(i, j) < pbOriented.at(k).at<double>(i, j)) {
                    pb.at<double>(i, j) = pbOriented.at(k).at<double>(i, j);
                }
            }
        }
    }
    double pbMax, pbMin;
    minMaxLoc(pb, &pbMin, &pbMax);
    // cout << pbMax << "\t" << pbMin << endl;
    double alpha = (255 - 0) / (pbMax - pbMin);
    double beta = -(255 - 0) * pbMin / (pbMax - pbMin);
    // log_debug("alpha: %lf, beta: %lf", alpha, beta);

    Size orig_size = pb.size();
    Mat pbDest(orig_size, CV_8UC1);
    pb.convertTo(pbDest, CV_8UC1, alpha, beta);     /// TODO: different from "imwrite" in MATLAB

    /* MATLAB
    ws = watershed(pb);
    */
    Mat ws;
    auto clockStart = chrono::high_resolution_clock::now();
    log_info("Starting watershed...");
    ws = watershed_yupeng(pbDest, 8);
    auto clockStop = chrono::high_resolution_clock::now();
    chrono::duration<double> clockElapsed = clockStop - clockStart;
    log_info("Watershed completed [%.3f sec]", clockElapsed.count());
    /*
    ws -> CV_32S
    pb.release();

    ofstream fout("ws.txt");
    if(fout.is_open()) {
        for(int i = 0; i < ws.rows; i++) {
            for(int j = 0; j < ws.cols; j++) {
                fout << ws.at<int>(i,j) << ",";
            }
            fout << "\n";
        }
        fout.close();
    }*/

    /* MATLAB
    ws_bw = (ws == 0);
    */
    log_debug("Before wsBw loop");   /* TODO temp, delete later*/
    Mat wsBw(ws.rows, ws.cols, CV_64F);
    for (int i = 0; i < ws.rows; i++) {
        for (int j = 0; j < ws.cols; j++) {
            if (ws.at<int>(i, j) != 0) {
                wsBw.at<double>(i, j) = 0;
            } else {
                wsBw.at<double>(i, j) = 1;
            }
        }
    }
    // ImsaveScale(pieceFilePat + "_01_ws_bw.png", wsBw);

    /* MATLAB
    contours = fit_contour(double(ws_bw));
    */
    clockStart = chrono::high_resolution_clock::now();
    log_info("Starting contour fitting...");
    FitContourOutput contours = fit_contour(wsBw);
    clockStop = chrono::high_resolution_clock::now();
    clockElapsed = clockStop - clockStart;
    log_info("Contour fitting completed [%.3f sec]", clockElapsed.count());

    int contourEdgeNum = contours.edge_x_coords.size();
    Mat angles(contourEdgeNum, 1, CV_32F);
    /* 2D coordinates */
    Mat v1(2, 1, CV_32S);
    Mat v2(2, 1, CV_32S);
    for (int e = 0; e < contourEdgeNum; e++) {
        if (contours.is_completion.at<int>(e, 0) == 1) {
            continue;
        }
        v1.at<int>(0, 0) = contours.vertices.at<int>(contours.edges.at<int>(e, 0), 0);
        v1.at<int>(1, 0) = contours.vertices.at<int>(contours.edges.at<int>(e, 0), 1);
        v2.at<int>(0, 0) = contours.vertices.at<int>(contours.edges.at<int>(e, 1), 0);
        v2.at<int>(1, 0) = contours.vertices.at<int>(contours.edges.at<int>(e, 1), 1);

        float ang = M_PI / 2;   /// TODO: In theory it should not be 90
        if (v1.at<int>(1, 0) != v2.at<int>(1, 0)) {
            ang = atan((v1.at<int>(0, 0) - v2.at<int>(0, 0)) * 1.0 / (v1.at<int>(1, 0) - v2.at<int>(1, 0)));
        }
        angles.at<float>(e, 0) = ang * 180 / M_PI;
    }

    Mat orient = Mat::zeros(contourEdgeNum, 1, CV_32S);
    for (int i = 0; i < contourEdgeNum; i++) {
        if (angles.at<float>(i, 0) < -78.75 | angles.at<float>(i, 0) >= 78.75) {
            orient.at<int>(i, 0) = 1;
        } else if (angles.at<float>(i, 0) < 78.75 & angles.at<float>(i, 0) >= 56.25) {
            orient.at<int>(i, 0) = 2;
        } else if (angles.at<float>(i, 0) < 56.25 & angles.at<float>(i, 0) >= 33.75) {
            orient.at<int>(i, 0) = 3;
        } else if (angles.at<float>(i, 0) < 33.75 & angles.at<float>(i, 0) >= 11.25) {
            orient.at<int>(i, 0) = 4;
        } else if (angles.at<float>(i, 0) < 11.25 & angles.at<float>(i, 0) >= -11.25) {
            orient.at<int>(i, 0) = 5;
        } else if (angles.at<float>(i, 0) < -11.25 & angles.at<float>(i, 0) >= -33.75) {
            orient.at<int>(i, 0) = 6;
        } else if (angles.at<float>(i, 0) < -33.75 & angles.at<float>(i, 0) >= -56.25) {
            orient.at<int>(i, 0) = 7;
        } else if (angles.at<float>(i, 0) < -56.25 & angles.at<float>(i, 0) >= -78.75) {
            orient.at<int>(i, 0) = 8;
        }
        orient.at<int>(i, 0) = orient.at<int>(i, 0) - 1;    /* index */
    }

    Mat wsWt = Mat::zeros(wsBw.rows, wsBw.cols, CV_64F);
    // wsBw.release();
    int rowInd, colInd, orient_e;
    double value1, value2, pbValue;
    for (int e = 0; e < contourEdgeNum; e++) {
        if (contours.is_completion.at<int>(e, 0) == 1) continue;
        orient_e = orient.at<int>(e, 0);
        for (int p = 0; p < contours.edge_x_coords.at(e).rows; p++) {
            rowInd = contours.edge_x_coords.at(e).at<int>(p, 0);
            colInd = contours.edge_y_coords.at(e).at<int>(p, 0);
            value1 = pbOriented.at(orient_e).at<double>(rowInd, colInd);
            value2 = wsWt.at<double>(rowInd, colInd);
            wsWt.at<double>(rowInd, colInd) = max(value1, value2);
        }
        v1.at<int>(0, 0) = contours.vertices.at<int>(contours.edges.at<int>(e, 0), 0);
        v1.at<int>(1, 0) = contours.vertices.at<int>(contours.edges.at<int>(e, 0), 1);
        v2.at<int>(0, 0) = contours.vertices.at<int>(contours.edges.at<int>(e, 1), 0);
        v2.at<int>(1, 0) = contours.vertices.at<int>(contours.edges.at<int>(e, 1), 1);

        pbValue = pbOriented.at(orient_e).at<double>(v1.at<int>(0, 0), v1.at<int>(1, 0));
        wsWt.at<double>(v1.at<int>(0, 0), v1.at<int>(1, 0)) =
            max(pbValue, wsWt.at<double>(v1.at<int>(0, 0), v1.at<int>(1, 0)));
        pbValue = pbOriented.at(orient_e).at<double>(v2.at<int>(0, 0), v2.at<int>(1, 0));
        wsWt.at<double>(v2.at<int>(0, 0), v2.at<int>(1, 0)) =
            max(pbValue, wsWt.at<double>(v2.at<int>(0, 0), v2.at<int>(1, 0)));
    }

    // ImsaveScale(pieceFilePat + "_02_ws_wt.png", wsWt);
    return wsWt;
}

Mat Contours2Ucm(std::string pieceFilePat, vector<Mat> &pb_oriented, string fmt) {
    /* create finest partition and transfer contour strength */
    Mat wsWt = CreateFinestPartition(pieceFilePat, pb_oriented);
    log_info("Phase 1: Create Finest Partition is done");

    /* MATLAB
    % prepare pb for ucm
    ws_wt2 = double(SuperContour4C(ws_wt));
    ws_wt2 = CleanWatersheds(ws_wt2);
    labels2 = bwlabel(ws_wt2 == 0, 8);
    labels = labels2(2:2:end, 2:2:end) - 1; % labels begin at 0 in mex file.
    ws_wt2(end+1, :) = ws_wt2(end, :);
    ws_wt2(:, end+1) = ws_wt2(:, end);  */
    superContour4COutput superContour = SuperContour4C(wsWt);
    // wsWt.release();
    log_info("Phase 2.1: Super Contour computation is done");

    Mat wsWt2Temp = superContour.pb2.clone();   /* return double type */
    wsWt2Temp = CleanWatersheds(wsWt2Temp);     /* return double type, should be closed */
    log_info("Phase 2.2: Clean Watershed is done");

    log_debug("Before wsWt loop");   /* TODO temp, delete later*/
    int rows = wsWt2Temp.rows;
    int cols = wsWt2Temp.cols;
    Mat wsWt2Flag = Mat::zeros(rows, cols, CV_32S);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (wsWt2Temp.at<double>(i, j) == 0) wsWt2Flag.at<int>(i, j) = 1;
        }
    }
    log_debug("Before bwlabel");   /* TODO temp, delete later*/
    Mat labels2 = bwlabel(wsWt2Flag, 8);
    // wsWt2Flag.release();

    log_debug("Before labels2 loop");   /* TODO temp, delete later*/
    Mat labels = Mat::zeros(rows / 2, cols / 2, CV_32S);
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            labels.at<int>(i, j) = labels2.at<int>(2 * i + 1, 2 * j + 1) - 1;
        }
    }
    // labels2.release();

    log_info("Rows: %d, Cols: %d", rows, cols);

    Mat wsWt2(rows + 1, cols + 1, CV_64F);
    int rowReal, colReal;
    for (int i = 0; i < rows + 1; i++) {
        rowReal = (i == rows) ? rows - 1 : i;
        for (int j = 0; j < cols + 1; j++) {
            colReal = (j == cols) ? cols - 1 : j;
            wsWt2.at<double>(i, j) = wsWt2Temp.at<double>(rowReal, colReal);
        }
    }
    // OpenCVFileWriter(wsWt2, pieceFilePat + "ws_wt2.yml", "ws_wt2");
    // wsWt2Temp.release();
    // ImsaveScale(pieceFilePat + "_03_ws_wt2.png", wsWt2);
    log_info("Phase 2.3: Pb for UCM is done");

    log_debug("Entering UcmMeanPb");   /* TODO temp, delete later*/
    /* compute ucm with mean pb */
    Mat superUcm = UcmMeanPb(wsWt2, labels);
    // OpenCVFileWriter(superUcm, pieceFilePat + "super_ucm.yml", "superucm");
    // ImsaveScale(pieceFilePat + "_04_super_ucm.png", superUcm);
    /* output */
    log_debug("Entering NormalizeOutput");   /* TODO temp, delete later*/
    superUcm = NormalizeOutput(superUcm);
    // OpenCVFileWriter(superUcm, pieceFilePat + "normalized_super_ucm.yml", "superucm");
    // ImsaveScale(pieceFilePat + "_05_super_ucm_normalize.png", superUcm);
    log_info("Phase 3.1: Super UCM computation is done");

    Mat ucm;
    if (fmt == "doubleSize") {
        ucm = superUcm.clone();
    } else {
        ucm = Mat::zeros((superUcm.rows + 1) / 2, (superUcm.cols + 1) / 2, CV_64F);
        for (int i = 2; i < superUcm.rows; i = i + 2) {
            for (int j = 2; j < superUcm.cols; j = j + 2) {
                ucm.at<double>((i - 2) / 2, (j - 2) / 2) = superUcm.at<double>(i, j);
            }
        }
    }
    log_info("Phase 3.2: UCM is done");
    return ucm;
    // check: return pbOriented.at(0);
}
