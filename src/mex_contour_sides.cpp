#include "collections/array_list.hh"
#include "collections/pointers/auto_collection.hh"
#include "io/streams/cout.hh"
#include "io/streams/iomanip.hh"
#include "io/streams/ios.hh"
#include "lang/array.hh"
#include "lang/exceptions/ex_index_out_of_bounds.hh"
#include "lang/exceptions/exception.hh"
#include "lang/pointers/auto_ptr.hh"
#include "math/libraries/lib_image.hh"
#include "math/math.hh"
#include "math/matrices/matrix.hh"

#include "concurrent/threads/thread.hh"

#include <time.h>

#include "head.h"
// #include <mex.h>
#include "log.h"

using namespace cv;
// using namespace std;


using collections::pointers::auto_collection;
using collections::array_list;
using io::streams::cout;
using io::streams::ios;
using io::streams::iomanip::setiosflags;
using io::streams::iomanip::setw;
using lang::array;
using lang::exceptions::exception;
using lang::exceptions::ex_index_out_of_bounds;
using lang::pointers::auto_ptr;
using math::libraries::lib_image;
using math::matrices::matrix;

using concurrent::threads::thread;

/**********************************
 * Matlab matrix conversion routines.
 **********************************/

/*
 * Convert an mxArray to a matrix.
 */

/*
matrix<> ToMatrix(const Mat a) {
    unsigned long mRows = static_cast<unsigned long>(a.rows);
    unsigned long nCols = static_cast<unsigned long>(a.cols);
    double *data = (double*) a.data;
    matrix<> m(mRows, nCols);
    for (unsigned long r = 0; r < mRows; r++) {
        for (unsigned long c = 0; c < nCols; c++) {
           m(r,c) = data[(c*mRows) + r];
        }
    }
    for(unsigned long r = 0; r < mRows; r++) {
        for(unsigned long c = 0; c < nCols; c++) {
     	   m(r,c) = a.at<double>(r,c);
        }
    }
    return m;
}
*/
matrix<> ToMatrix(const Mat& a);

/*
 * Convert a 2D matrix to an mxArray.
 */
Mat ToOpenCvIntMatMatrix(const matrix<>& m) {
    unsigned long mRows = m.size(0);
    unsigned long nCols = m.size(1);
    Mat a(static_cast<int>(mRows), static_cast<int>(nCols), CV_32S);
    int* data = (int*)a.data;
    for (unsigned long r = 0; r < mRows; r++) {
        for (unsigned long c = 0; c < nCols; c++) {
            data[(r * nCols) + c] = m(r, c);
        }
    }
    /*
    for(unsigned long r = 0; r < mRows; r++) {
        for(unsigned long c = 0; c < nCols; c++) {
            a.at<int>(r,c) = m(r,c);
        }
    }
    */
    return a;
}

/*
 * Convert an array to an mxArray.
 * */
Mat ToOpenCvIntMatArray(const array<double>& m) {
    unsigned long mRows = m.size();
    unsigned long nCols = 1;
    Mat a(static_cast<int>(mRows), static_cast<int>(nCols), CV_32S);
    /*
    int *data = (int*) a.data;
    for (unsigned long r = 0; r < mRows; r++) {
        for (unsigned long c = 0; c < nCols; c++) {
            data[(c*mRows) + r] = m[r];
        }
    }
    */
    for (unsigned long r = 0; r < mRows; r++) {
        for (unsigned long c = 0; c < nCols; c++) {
            a.at<int>(r, c) = m[r];
        }
    }
    return a;
}

/*
 * Matlab interface.
 */
// void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
void MexContourSides(FitContourOutput& contours_init, Mat& nmax) {
    try {
        /* get nonmax-suppressed image */
        // matrix<> im = ToMatrix(prhs[0]);
        log_debug("Before ToMatrix");   /* TODO temp, delete later*/
        matrix<> im = ToMatrix(nmax);
        /* compute discrete contours */
        /*
        cout << setiosflags(ios::left);
        cout << setw(40) << "compute contours ";
        cout.flush();
        */
        clock_t start_time = clock();
        clock_t time = start_time;
        matrix<> im_skel = im;
        // lib_image::skeletonize_2D(im);
        log_debug("After tomatrix, before connected_components()");   /* TODO temp, delete later*/
        matrix<unsigned long> labels = lib_image::connected_components(im_skel);
        lib_image::contour_set contours(labels);
    log_debug("After connected_components, Before subdivide_local");   /* TODO temp, delete later*/
        contours.subdivide_local();
    log_debug("After subdivide_local(), Before subdivide_global");   /* TODO temp, delete later*/
        contours.subdivide_global();
    log_debug("After subdivide_global, Before completion_cdt");   /* TODO temp, delete later*/
        contours.completion_cdt();
        // cout << "[" << (double(clock() - time) / CLOCKS_PER_SEC) << " sec]\n";
        // cout.flush();
        // log_info("compute contours [%lf sec]", (double(clock() - time) / CLOCKS_PER_SEC));

        /* get sizes of regions */
        /*
        cout << setw(40) << "returning results ";
        cout.flush();
        */
        time = clock();
        /* extract vertex/edge map */
        matrix<bool> is_v(labels.dimensions());
        matrix<bool> is_e(labels.dimensions());
        log_debug("Before loop to extract V-E map");   /* TODO temp, delete later*/
        matrix<unsigned long> assign(labels.dimensions());
        for (unsigned long n = 0; n < labels.size(); n++) {
            is_v[n] = contours.is_vertex(n);
            is_e[n] = contours.is_edge(n);
            if (is_v[n]) {
                assign[n] = contours.vertex_id(n);
            }
            else if (is_e[n]) {
                assign[n] = contours.edge_id(n);
            }
        }
        /* extract vertex coordinates */
        unsigned long n_vertices = contours.vertices_size();
        matrix<unsigned long> vertex_coords(n_vertices, 2);
        log_debug("V-E map done, before loop to extract coords");   /* TODO temp, delete later*/
        for (unsigned long n = 0; n < n_vertices; n++) {
            vertex_coords(n, 0) = contours.vertex(n).x;
            vertex_coords(n, 1) = contours.vertex(n).y;
        }
        /* extract edge endpoints */
        unsigned long n_edges = contours.edges_size();
        // debug - cout << n_vertices << '\t' << n_edges << endl;
        log_debug("Coords done, Extracting edge endpoints");   /* TODO temp, delete later*/
        matrix<unsigned long> edge_endpoints(n_edges, 2);
        for (unsigned long n = 0; n < n_edges; n++) {
            edge_endpoints(n, 0) = contours.edge(n).vertex_start->id;
            edge_endpoints(n, 1) = contours.edge(n).vertex_end->id;
        }
        /* extract edge coordinates */
        // mxArray* mx_e_x_coords = mxCreateCellMatrix(static_cast<int>(n_edges), 1);
        // mxArray* mx_e_y_coords = mxCreateCellMatrix(static_cast<int>(n_edges), 1);
        vector<cv::Mat> mx_e_x_coords(static_cast<int>(n_edges));
        vector<cv::Mat> mx_e_y_coords(static_cast<int>(n_edges));
        log_debug("Extracting edge coordinates");   /* TODO temp, delete later*/
        for (unsigned long n = 0; n < n_edges; n++) {
            array<double> e_x(contours.edge(n).x_coords);
            array<double> e_y(contours.edge(n).y_coords);
            // cout << n << '\t' << e_x.size() << '\t' << e_y.size() << '\t' << contours.edge(n).is_completion << endl;
            mx_e_x_coords.at(n) = ToOpenCvIntMatArray(e_x);
            mx_e_y_coords.at(n) = ToOpenCvIntMatArray(e_y);
        }
        /* extract completion flags */
        matrix<bool> is_compl(n_edges, 1);
        log_debug("Before is_completion loop");   /* TODO temp, delete later*/
        for (unsigned long n = 0; n < n_edges; n++) is_compl[n] = contours.edge(n).is_completion;

        // return all
        log_debug("Converting vertices, edges, is_completion to opencv mats");   /* TODO temp, delete later*/
        contours_init.vertices = ToOpenCvIntMatMatrix(matrix<>(vertex_coords));
        contours_init.edges = ToOpenCvIntMatMatrix(matrix<>(edge_endpoints));
        contours_init.is_completion = ToOpenCvIntMatMatrix(matrix<>(is_compl));
        contours_init.edge_x_coords = mx_e_x_coords;
        contours_init.edge_y_coords = mx_e_y_coords;

        /*
        cout << "[" << (double(clock() - time) / CLOCKS_PER_SEC) << " sec]\n";
        cout.flush();
        */
        // log_info("returning results [%lf sec]", (double(clock() - time) / CLOCKS_PER_SEC));
    }
    catch (ex_index_out_of_bounds& e) {
        log_error("Index: %lu | %s", e.index(), e.what());
    }
    catch (exception& e) {
        log_error("%s", e.what());
    }
}
