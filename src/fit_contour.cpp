#include "head.h"
#include "logger.h"
using namespace cv;
using namespace std;

FitContourOutput fit_contour(Mat& nmax) {
    /* MATLAB
    % extract contours
    [skel, labels, is_v, is_e, assign, vertices, edges, ...
    v_left, v_right, e_left, e_right, c_left, c_right, ...
    edge_equiv_ids, is_compl, e_x_coords, e_y_coords] = ...
    MexContourSides(nmax, true);

    Mat x_coord(1,1,CV_32S);
    x_coord.at<int>(0,0) = contours_init.e_x_coords(i,0);
    des(nmax,true);
    */
    FitContourOutput contours;
    log_debug("Before MexContourSides");   /* TODO temp, delete later*/
    MexContourSides(contours, nmax);
    log_debug("After MexContourSides");   /* TODO temp, delete later*/
    return contours;
}
