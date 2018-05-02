#include <cv.h>
#include <highgui.h>
#include "log.h"

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

using namespace cv;

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

/*
 * Convert an mxArray to a matrix.
 */
matrix<> ToMatrix(const Mat& a) {
    unsigned long mrows = static_cast<unsigned long>(a.rows);
    unsigned long ncols = static_cast<unsigned long>(a.cols);
    // double *data = NULL;
    double* data = (double*) a.data;
    matrix<> m(mrows, ncols);
    for (unsigned long r = 0; r < mrows; r++) {
        // data = (double*) (a.ptr<double>(r));   /// TODO: check template for double
        for (unsigned long c = 0; c < ncols; c++) {
          // m(r,c) = data[c];
          m(r, c) = data[r * ncols + c];
        }
    }
    return m;
}

/*
 * Convert a 2D matrix to an mxArray.
 */
Mat* to_openCVMat(const matrix<>& m) {
    unsigned long mrows = m.size(0);
    unsigned long ncols = m.size(1);
    /// TODO: check if CV_64FC1 means double
    Mat* a = new Mat(static_cast<int>(mrows), static_cast<int>(ncols), CV_64FC1);
    // double *data = NULL;
    double* data = (double*) a->data;
    for (unsigned long r = 0; r < mrows; r++) {
      // data = (double*) (a->ptr<double>(r));    /// TODO: check if I need to use ptr<double>
        for (unsigned long c = 0; c < ncols; c++) {
          // data[c] = m(r,c);
          data[(r * ncols) + c] = m(r, c);
        }
    }
    return a;
}

void mex_pb_parts_final_selected(Mat*& plhs0, Mat***& plhs_bg, Mat***& plhs_cga, Mat***& plhs_cgb, Mat***& plhs_tg,
                                 const Mat& channels0, const Mat& channels1, const Mat& channels2) {
    try {
        /* parameters - binning and smoothing */
        unsigned long n_ori = 8;                       /* number of orientations */
        unsigned long num_L_bins = 25;                 /* # bins for bg */
        unsigned long num_a_bins = 25;                 /* # bins for cg_a */
        unsigned long num_b_bins = 25;                 /* # bins for cg_b */
        double bg_smooth_sigma = 0.1;                  /* bg histogram smoothing sigma */
        double cg_smooth_sigma = 0.05;                 /* cg histogram smoothing sigma */
        unsigned long border = 30;                     /* border pixels */
        double sigma_tg_filt_sm = 2.0;                 /* sigma for small tg filters */
        double sigma_tg_filt_lg = math::sqrt(2) * 2.0; /* sigma for large tg filters */
        /* parameters - radii */
        unsigned long n_bg = 3;
        unsigned long n_cg = 3;
        unsigned long n_tg = 3;
        unsigned long r_bg[] = {3, 5, 10};
        unsigned long r_cg[] = {5, 10, 20};
        unsigned long r_tg[] = {5, 10, 20};

        /* compute bg histogram smoothing kernel */
        matrix<> bg_smooth_kernel = lib_image::gaussian(bg_smooth_sigma * num_L_bins);
        matrix<> cga_smooth_kernel = lib_image::gaussian(cg_smooth_sigma * num_a_bins);
        matrix<> cgb_smooth_kernel = lib_image::gaussian(cg_smooth_sigma * num_b_bins);

        /* get image */
        matrix<> L = ToMatrix(channels0);
        matrix<> a = ToMatrix(channels1);
        matrix<> b = ToMatrix(channels2);

        /* mirror border */
        L = lib_image::border_mirror_2D(L, border);
        a = lib_image::border_mirror_2D(a, border);
        b = lib_image::border_mirror_2D(b, border);

        /* convert to grayscale */
        clock_t start_time = clock();
        clock_t time = start_time;
        log_info("Converting RGB to grayscale");
        matrix<> gray = lib_image::grayscale(L, a, b);
        log_info("Conversion complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);

        /* gamma correct */
        lib_image::rgb_gamma_correct(L, a, b, 2.5);

        /* convert to Lab */
        log_info("Converting RGB to Lab");
        time = clock();
        lib_image::rgb_to_lab(L, a, b);
        lib_image::lab_normalize(L, a, b);
        log_info("Conversion complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);

        /* quantize color channels  */
        log_info("Quantizing color channels");
        time = clock();
        matrix<unsigned long> Lq = lib_image::quantize_values(L, num_L_bins);
        matrix<unsigned long> aq = lib_image::quantize_values(a, num_a_bins);
        matrix<unsigned long> bq = lib_image::quantize_values(b, num_b_bins);
        log_info("Quantization complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);

        /* compute texton filter set */
        log_info("Computing filter set for textons");
        time = clock();
        auto_collection<matrix<>, array_list<matrix<> > > filters_small =
            lib_image::texton_filters(n_ori, sigma_tg_filt_sm);
        auto_collection<matrix<>, array_list<matrix<> > > filters_large =
            lib_image::texton_filters(n_ori, sigma_tg_filt_lg);
        array_list<matrix<> > filters;
        filters.add(*filters_small);
        filters.add(*filters_large);
        log_info("Computation complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);

        /* compute textons */
        log_info("Computing textons");
        time = clock();
        auto_collection<matrix<>, array_list<matrix<> > > textons;
        matrix<unsigned long> t_assign = lib_image::textons(gray, filters, textons, 64);
        t_assign = matrix<unsigned long>(
            lib_image::border_mirror_2D(lib_image::border_trim_2D(matrix<>(t_assign), border), border));
        log_info("Computation complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);

        /* return textons */
        plhs0 = to_openCVMat(lib_image::border_trim_2D(matrix<>(t_assign), border));

        /* compute bg at each radius */
        plhs_bg = new Mat**[n_bg];
        for (unsigned long rnum = 0; rnum < n_bg; rnum++) {
            /* compute bg */
            log_info("Computing bg | r = %lu", r_bg[rnum]);
            time = clock();
            auto_collection<matrix<>, array_list<matrix<> > > bgs =
                lib_image::hist_gradient_2D(Lq, r_bg[rnum], n_ori, bg_smooth_kernel);
            log_info("Computation complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);
            /* return bg */
            plhs_bg[rnum] = new Mat*[n_ori];

            for (unsigned long n = 0; n < n_ori; n++) {
                plhs_bg[rnum][n] = to_openCVMat(lib_image::border_trim_2D((*bgs)[n], border));
            }
        }

        /* compute cga at each radius */
        plhs_cga = new Mat**[n_cg];
        for (unsigned long rnum = 0; rnum < n_cg; rnum++) {
            /* compute cga */
            log_info("Computing cg_a | r = %lu", r_cg[rnum]);
            time = clock();
            auto_collection<matrix<>, array_list<matrix<> > > cgs_a =
                lib_image::hist_gradient_2D(aq, r_cg[rnum], n_ori, cga_smooth_kernel);
            log_info("Computation complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);
            /* return cga */
            plhs_cga[rnum] = new Mat*[n_ori];

            for (unsigned long n = 0; n < n_ori; n++) {
                plhs_cga[rnum][n] = to_openCVMat(lib_image::border_trim_2D((*cgs_a)[n], border));
            }
        }

        /* compute cgb at each radius */
        plhs_cgb = new Mat**[n_cg];
        for (unsigned long rnum = 0; rnum < n_cg; rnum++) {
            /* compute cgb */
            log_info("Computing cg_b | r = %lu", r_cg[rnum]);
            time = clock();
            auto_collection<matrix<>, array_list<matrix<> > > cgs_b =
                lib_image::hist_gradient_2D(bq, r_cg[rnum], n_ori, cgb_smooth_kernel);
            log_info("Computation complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);
            /* return cgb */
            plhs_cgb[rnum] = new Mat*[n_ori];

            for (unsigned long n = 0; n < n_ori; n++) {
                plhs_cgb[rnum][n] = to_openCVMat(lib_image::border_trim_2D((*cgs_b)[n], border));
            }
        }

        /* compute tg at each radius */
        plhs_tg = new Mat**[n_tg];
        for (unsigned long rnum = 0; rnum < n_tg; rnum++) {
            /* compute tg */
            log_info("Computing tg | r = %lu", r_tg[rnum]);
            time = clock();
            auto_collection<matrix<>, array_list<matrix<> > > tgs =
                lib_image::hist_gradient_2D(t_assign, r_tg[rnum], n_ori);
            log_info("Computation complete: %f sec", double(clock() - time) / CLOCKS_PER_SEC);
            /* return tg */
            plhs_tg[rnum] = new Mat*[n_ori];

            for (unsigned long n = 0; n < n_ori; n++) {
                plhs_tg[rnum][n] = to_openCVMat(lib_image::border_trim_2D((*tgs)[n], border));
            }
        }
    }
    catch (ex_index_out_of_bounds& e) {
        log_error("Index: %lu | %s", e.index(), e.what());
    }
    catch (exception& e) {
        log_error("%s", e.what());
    }
}
