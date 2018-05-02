#include "head.h"

using namespace cv;
using namespace std;

Mat Matrix2Image(Mat& Pb_8U) {
    Mat channel[3] = {Pb_8U, Pb_8U, Pb_8U};
    Mat image;
    merge(channel, 3, image);
    return image;
}

vector<coordinates> find_8U(Mat& matrix, int level) {
    vector<coordinates> loc;
    coordinates loc_point;
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            if (matrix.at<unsigned char>(i, j) == level) {
                loc_point.row = i;
                loc_point.col = j;
                loc.push_back(loc_point);
            }
        }
    }
    return loc;
}

Mat watershed_yupeng(Mat& A, int conn) {
    int s[2] = {A.rows, A.cols};
    Mat L = Mat::zeros(s[0], s[1], CV_32S) - 1;
    /*
    for(int i = 0; i < 50; i++) {
        for(int j = 0; j < 20; j++) {
            cout << L.at<int>(i,j) << "\t";
        }
        cout << endl;
    }
    */

    int couleur = 1;

    /* Get min and max levels of the input matrix */
    double minmin, maxmax;
    minMaxLoc(A, &minmin, &maxmax);
    int min_A = (int)minmin;
    int max_A = (int)maxmax;

    /* For each levels in the input image */
    vector<coordinates> XY_A;
    int marker;
    for (int level = min_A; level <= max_A; level++) {
        /* Get pixels coordinates in the current level */
        XY_A = find_8U(A, level);

        /* Examines each pixel of the current level */
        marker = 0;
        while (marker < XY_A.size()) {
            int x = XY_A.at(marker).row;
            int y = XY_A.at(marker).col;
            marker++;

            /* Pixel state: -1 (unknown), 0 (watershed pixel), n>0 (pixel into region #n) */
            int etat = -1;

            /* Pick up pixel's neighbors, with 8-connectivity */
            int val;
            if (conn == 8) {
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        if (i != 0 || j != 0) {
                            if ((x + i >= 0) && (x + i < s[0]) && (y + j >= 0) && (y + j < s[1])) {
                                /* Update pixel's state according to the neighbor */
                                val = L.at<int>(x + i, y + j);
                                /* if the neighbor pixel already has a state */
                                if (val > 0) {
                                    /* if the current pixel has no old state */
                                    if (etat == -1) {
                                        /* The current pixel takes the state of the neighbor */
                                        etat = val;
                                    }
                                    else {
                                        /* If the old state of the pixel is different from the neighbor's state */
                                        if (val != etat) {
                                            /* It becomes a watershed pixel */
                                            etat = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* Update label matrix according pixel's state */
            if (etat >= 0) {
                /* The pixel belongs to an existing region or is a watershed pixel */
                L.at<int>(x, y) = etat;
            }
            else {
                /* The pixel belongs to a new region */
                L.at<int>(x, y) = couleur;
                couleur = couleur + 1;
            }
        }
    }
    return L;
}
