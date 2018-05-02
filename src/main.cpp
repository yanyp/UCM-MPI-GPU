#include <cv.h>
#include <highgui.h>
#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <ctime>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sstream>

#include <globalPb.h>
#include <sys/file.h>
#include <unistd.h>

#include "GLSVM.h"
#include "SPFeatures.h"
#include "UCMgenerator.h"
#include "head.h"
#include "otherheaders.h"

#include <thread>
#include <chrono>

using namespace cv;
using namespace std;

struct para {
    int nClasses;
    /* OpenMPI */
    int nPieces;    // modified by Yupeng, 01/21
    int nBigPieces; // modified by Yupeng, 01/21
    int overlap;
    int nOrient;
    int bigOverlap;
    /* UCM */
    double scaleK;
    /* GLSVM */
    int binStep;
    double tau;
    double lambdaS;
    double lambdaH;
    double epsZ;
    double convergenceThreshold;
    /* Modeling Method */
    string mode;
    string seed;
    bool broadcast;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if (argc != 7) {    // modified by Yupeng, 01/21
        log_fatal("Only detect %d arguments, please pass 7 arguments", argc);
        return -1;
    }

    double t1, t2;
    t1 = MPI_Wtime();

    int nprocs, rank;
    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* read the very big image */
    String biggestImgFN = argv[1];
    /* modified by Yupeng */
    int bigRsz;

    struct para option;
    option.overlap = 0;
    option.nOrient = 8;
    option.bigOverlap = 0;
    option.scaleK = atof(argv[5]);
    option.binStep = 4;
    option.tau = 1;
    option.lambdaS = 0;
    option.lambdaH = 1;
    option.epsZ = 0.001;
    option.convergenceThreshold = 0.00001;

    option.nClasses = atoi(argv[2]);
    option.nPieces = atoi(argv[3]);
    option.nBigPieces = atoi(argv[4]);
    option.mode = "P";
    option.seed = "P";
    option.broadcast = false; ///(atoi(argv[9]) == 1); // modified by Yupeng, 04/23

    /* modified by Yupeng */
    String bigPieceFilePath = biggestImgFN.substr(0, biggestImgFN.find_last_of("."));
    String fileTemp = bigPieceFilePath.substr(bigPieceFilePath.find_last_of("/") + 1);
    bigPieceFilePath += "_" + string(argv[3]) + "_" + string(argv[4]) + "/" + fileTemp;

    /// TODO: Later to be fixed for non-zero overlaps.
    /*
     * Currently overlap between small pieces
     * to compute gPb is set to zero.
     */

    cv::Mat biggestImg;
    cv::Mat bigImg;
    cv::Mat currentImage;
    vector<cv::Mat> patches;
    /* only rank 0 reads the biggest 10k by 10k image */

    if (rank == 0) {
        log_info("Broadcast mode is set to %d", option.broadcast);
        time_t nowd1, nowd2;
        time(&nowd1);

        log_info("Number of processes: %d", nprocs);
        log_info("fileTemp: %s", fileTemp.c_str()); /* modified by Yupeng */
        log_info("bigPieceFilePath: %s", bigPieceFilePath.c_str()); /* modified by Yupeng */

        biggestImg = imread(biggestImgFN);
        Size bigOrigSize = biggestImg.size();
        int rows = bigOrigSize.height;
        int cols = bigOrigSize.width;
        int nPixels = rows * cols;

        log_info("Image: %s, Dimensions: %d x %d, nPixels: %d", biggestImgFN.c_str(), cols, rows, nPixels);

        // modified by Yupeng, 04/23
        getTrainingPatches(bigPieceFilePath, biggestImg, rows, cols);

        patches = Img2Pieces(bigPieceFilePath, biggestImg, option.bigOverlap, option.nBigPieces);
        bigRsz = (int) sqrt(patches.size());
        assert(bigRsz * bigRsz == option.nBigPieces);    // added by Yupeng, 01/21
        time(&nowd2);
        log_debug("bigRsz: %d, patches.size(): %d", bigRsz, patches.size());
        log_debug("Time for first set division: %ld", nowd2 - nowd1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&bigRsz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        bigImg = patches[0];

        /* patches.size is always going to be bigRsz * bigRsz by virtue of our division scheme */
        for(int i = 1; i < patches.size(); ++i) {
            log_info("Sending image %d -> %d (%d x %d)", rank, i, patches[i].rows, patches[i].cols);
            MpiSendMat(patches[i], i);
        }
    }
    else if(rank < bigRsz * bigRsz) {
        // Accept results of first division
        bigImg = MpiRecvMat(0);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /*
     * There will be as many processes available as there are small sized pieces for
     * eigenvector computation (gPb computation)
     */
    int* rsz = new int[nprocs];

    int sub_rsz = 0;
    int nBigIm = bigRsz * bigRsz;
    // int* rows = new int [nBigIm];
    // int* cols = new int [nBigIm];
    int* rows = new int[nprocs];
    int* cols = new int[nprocs];
    int bigPartRow = 0, bigPartCol = 0;

    /*
     * Only processes which have rank less than nBigIm participate to further
     * subdivide the image into pieces. Note that nBigIm < nprocs
     */
    if (rank < nBigIm) {
        time_t nowd3, nowd4;
        time(&nowd3);

        int big_sy = rank / bigRsz;
        int big_sx = fmod(rank, bigRsz);
        String bigImgFileName = bigPieceFilePath + "_" + to_string(big_sy) + "_" + to_string(big_sx) + ".png";
        string pieceFilePath = bigPieceFilePath + "_" + to_string(big_sy) + "_" + to_string(big_sx);

        // Mat bigImg = imread(bigImgFileName);
        Size bigPartSize = bigImg.size();
        bigPartRow = bigPartSize.height;
        bigPartCol = bigPartSize.width;

        patches = Img2Pieces(pieceFilePath, bigImg, option.overlap, option.nPieces / option.nBigPieces);    // modified by Yupeng, 01/21
        sub_rsz = (int) sqrt(patches.size());
        currentImage = patches[0];

        time(&nowd4);
        log_info("Time for second set division: %lu", nowd4 - nowd3);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&sub_rsz, 1, MPI_INT, rsz, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&bigPartRow, 1, MPI_INT, rows, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&bigPartCol, 1, MPI_INT, cols, 1, MPI_INT, MPI_COMM_WORLD);
    /// TODO: the above gather should only be done for a subset of processors bigRsz^2

    // Assumption: Number of subdivisions in any patch is the same as that in the root processes's patch
    int subDivPerPatch = rsz[0] * rsz[0];   // moidified by Yupeng, 01/21
    //int subDivPerPatch = rsz[0];
    log_info("rank: %d, subDivPerPatch: %d, rsz[0]: %d", rank, subDivPerPatch, rsz[0]);

    // Ensure everyone has their copy of the subdivided image
    if(rank < nBigIm) {
        log_debug("patches.size(): %d", patches.size());
        currentImage = patches[0];
        if(subDivPerPatch > 1) {
            for(int i = 1; i < patches.size(); ++i) {
                int destId = nBigIm + rank * (subDivPerPatch - 1) + (i - 1);
                log_info("Sending image %d -> %d (%d x %d)", rank, destId, patches[i].rows, patches[i].cols);
                MpiSendMat(patches[i], destId);
            }
        }
    }
    else {
        if(subDivPerPatch > 1) {
            int parentId = (rank - nBigIm) /(subDivPerPatch - 1);
            currentImage = MpiRecvMat(parentId);
        }
    }

    int nVals = nBigIm + 1;
    int* cum_rsz = new int[nVals];      /// TODO: must be allocated at all nodes?

    if (rank == 0) {
        time_t now5, now6;
        time(&now5);

        cum_rsz[0] = 0;
        // cum_rsz[0] = rsz[0];
        for (int i = 1; i < nVals; i++) {
            cum_rsz[i] = cum_rsz[i - 1] + pow(rsz[i - 1], 2);
        }

        time(&now6);
        double seconds2 = difftime(now6, now5);
        // log_debug("rsz table %d took %ld sec", rank, seconds2);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(cum_rsz, nVals, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Containers for process local gpbOrient and gPbNonmax
    // To be merged into parent process later
    vector<cv::Mat> gPbOrient;
    cv::Mat gPbNonmax;
    if (rank < cum_rsz[nVals - 1]) {
        time_t now1, now2;
        time(&now1);

        log_debug("rank: %d, cum_rsz[nVals-1]: %d", rank, cum_rsz[nVals - 1]);
        int* low_pr = lower_bound(cum_rsz, cum_rsz + nVals, rank);
        // int low = *low_pr;
        int low = low_pr - cum_rsz;
        if (cum_rsz[low] > rank) {
            low = low - 1;
        }
        int bigsy = low / bigRsz;
        int bigsx = fmod(low, bigRsz);

        int local_rank = rank - cum_rsz[low];
        log_debug("low: %d, local_rank: %d, rsz[low]: %d", low, local_rank, rsz[low]);
        int sy = local_rank / rsz[low];
        int sx = fmod(local_rank, rsz[low]);

        String outFile = bigPieceFilePath + "_" + to_string(bigsy) + "_" + to_string(bigsx) + "_" +
                         to_string(sy) + "_" + to_string(sx);

        int cudaDeviceCount = getCudaDeviceCount();
        int targetDevice = rank % cudaDeviceCount;
        char hostname[256];
        gethostname(hostname, 255);
        char file[300];
        sprintf(file, "./ucm-mpi_%s_%d.lock", hostname, targetDevice);
        int fd = open(file, O_CREAT | O_EXCL, 0644);
        while (fd == -1) {
            sleep(1);
            fd = open(file, O_CREAT | O_EXCL, 0644);
        }
        log_debug("Acquired mutex: %s", file);
        GlobalPbCuda(currentImage, outFile, gPbOrient, gPbNonmax, option.nOrient, rank);
        int can_close = close(fd);
        if (can_close == -1) {
            log_warn("Unable to close mutex: %s", file);
        }
        int can_remove = remove(file);
        if (can_remove != 0) {
            log_warn("Unable to remove mutex: %s", file);
        }

        // }
        time(&now2);
        double seconds = difftime(now2, now1);
        log_info("Rank: %d, gPb time taken: %f", rank, seconds);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    vector<vector<cv::Mat>> patchGpbOrient;
    vector<cv::Mat> patchGpbNonmax;

    // Transfer non max and orient seperately since they share message tags / identifications
    if(rank >= nBigIm) {
        if(subDivPerPatch > 1) {
            int parentId = (rank - nBigIm) / (subDivPerPatch - 1);
            log_info("Sending gPbNonmax matrices (%d -> %d)", rank, parentId);
            MpiSendMat(gPbNonmax, parentId);
        }
    }
    else {
        patchGpbNonmax.push_back(gPbNonmax);
        if(subDivPerPatch > 1) {
            for(int i = 1; i < subDivPerPatch; ++i) {
                int childId = nBigIm + rank * (subDivPerPatch - 1) + (i - 1);
                log_info("Receiving gPbNonmax matrices (%d -> %d)", childId, rank);
                patchGpbNonmax.push_back(MpiRecvMat(childId));
            }
        }
    }

    // Transfer oriented matrices now that tag ids are available
    if(rank >= nBigIm) {
        if(subDivPerPatch > 1) {
            int parentId = (rank - nBigIm) / (subDivPerPatch - 1);
            log_info("Sending gPbOrient matrices (%d -> %d)", rank, parentId);
            MpiSendMatVec(gPbOrient, parentId);
        }
    }
    else {
        patchGpbOrient.push_back(gPbOrient);
        if(subDivPerPatch > 1) {
            for(int i = 1; i < subDivPerPatch; ++i) {
                int childId = nBigIm + rank * (subDivPerPatch - 1) + (i - 1);
                log_info("Receiving gPbOrient matrices (%d -> %d)", childId, rank);
                patchGpbOrient.push_back(MpiRecvMatVec(childId));
            }
        }
    }

    /// TODO: Provide detailed comments here and crosscheck X and Y dimensions
    /*
     * The code below creates a creates a new communicator. This can be confusing based on the documentation
     * you read since X and Y directions are not clearly specified.
     */

    MPI_Comm grid_comm;
    int dimSizes[2];
    int wrapAround[2];
    int reorder = 1;
    dimSizes[0] = bigRsz;
    dimSizes[1] = bigRsz;
    wrapAround[0] = 0;
    wrapAround[1] = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dimSizes, wrapAround, reorder, &grid_comm);

    /// TODO: @Yupeng - should I move the two lines below inside the 'if'?
    int gridRank;
    int coordinates[2];

    vector<cv::Mat> classLabelMatVector(bigRsz * bigRsz);
    vector<cv::Mat> ucmDataVector(bigRsz * bigRsz);
    vector<cv::Mat> fineBdryVector(bigRsz * bigRsz);
 
    /*
     * The code below works within the new communicator which is grid_comm
     * Note that gridRank < nBigIm
     */
    /* combine small patches into big patches and make the graphs */

    // MPI_DEBUG: Yupeng added
    if (grid_comm == MPI_COMM_NULL) {
        int MAX_SLEEP_CYCLES = atof(argv[6]), sleep_cycles = 0;
        int wakeup_flag;
        MPI_Status status_curr;
        while (sleep_cycles < MAX_SLEEP_CYCLES) {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &wakeup_flag, &status_curr);
            if (wakeup_flag == 1) {
                log_info("rank (%d) wakes up with sleep cycles (%d)", rank, sleep_cycles);
                break;
            } else {
            std::this_thread::sleep_for(std::chrono::seconds(1));
         }
            sleep_cycles++;
        }
    }
    else {

        MPI_Comm_rank(grid_comm, &gridRank);
        MPI_Cart_coords(grid_comm, gridRank, 2, coordinates);

        log_info("rank: %d, gridRank: %d", rank, gridRank);

        time_t now3, now4, now34, now5; // Yupeng added now5, 01/25
        time(&now3);

        int bigsy = gridRank / bigRsz;
        int bigsx = fmod(gridRank, bigRsz);

        if (bigsx == coordinates[1]) {
            log_debug("big sx equality holds");
        } else {
            log_debug("big sx not equal");
        }

        if (bigsy == coordinates[0]) {
            log_debug("big sy equality holds");
        } else {
            log_debug("big sy not equals");
        }

        if (bigsx == coordinates[0]) {
            log_debug("opposite equals");
        } else {
            log_debug("no opposite");
        }

        String pieceFilePath = bigPieceFilePath + "_" + to_string(bigsy) + "_" + to_string(bigsx);
        log_debug("gridRank: %d, pieceFilePath: %s", gridRank, pieceFilePath.c_str());

        vector<cv::Mat> gPbOrientMerged;
        gPbOrientMerged.reserve(option.nOrient);
        for (int i = 0; i < option.nOrient; i++) {
            cv::Mat gPb_orient_piece = cv::Mat::zeros(rows[gridRank], cols[gridRank], CV_64FC1);
            gPbOrientMerged.push_back(gPb_orient_piece);
        }

        cv::Mat gPbNonmaxMerged(gPbOrientMerged[0].rows, gPbOrientMerged[0].cols, CV_64FC1);

        Mat temp;
        double pbMin, pbMax, alpha, beta;

        // if (!ifile) {
        Pieces2Gpb(patchGpbOrient, patchGpbNonmax, gPbOrientMerged, gPbNonmaxMerged, pieceFilePath, option.overlap,
                   rsz[gridRank], rows[gridRank], cols[gridRank], option.nOrient);

        // Write non max gpb to disk for visualization
        OpenCVImageWriter(gPbNonmaxMerged, pieceFilePath + "_gpb_nmax.png");
        OpenCVFileWriter(gPbNonmaxMerged, pieceFilePath + "_gpb_nmax.yml", "gpb_nonmax");

        // Write oriented gpb to disk for visualization
        for(int o = 0; o < option.nOrient; ++o) {
            OpenCVImageWriter(gPbOrientMerged[o], pieceFilePath + "_gpb_orient_" + to_string(o) + ".png");
            // modified by Yupeng, 01/21
            // OpenCVFileWriter(gPbOrientMerged[o], pieceFilePath + "_gpb_orient_" + to_string(o) + ".yml", "gpb_orient_" + to_string(o));
        }

        MPI_Barrier(grid_comm);

        UCM_generator ucm_data(rows[gridRank], cols[gridRank], pieceFilePath);
        String fmt = "doubleSize";
        ucm_data.generateUCM2(gPbOrientMerged, fmt);
        ucm_data.generateUCM();

        time(&now34);
        double seconds34 = difftime(now34, now3);
        log_info("gridRank: %d, real UCM combine time: %f", gridRank, seconds34);
        
        /* Yupeng added, Feb 19th */
        double tAfterUCM = MPI_Wtime();
        log_info("After UCM: %f", tAfterUCM - t1);

        // added by Yupeng
        OpenCVFileWriter(ucm_data.ucm2, pieceFilePath + "_ucm2.yml", "ucm2");
        OpenCVImageWriter(ucm_data.ucm, pieceFilePath + "_ucm.png");
        log_info("UCM piece data has been written by gridRank (%d)", gridRank);

        ucm_data.generateBdryAtScaleK(option.scaleK, 1);
        int nBaseLabels = ucm_data.generateLabelsAtScaleK(option.scaleK, 1);

        log_info("nBaseLabels: %d", nBaseLabels);
        // OpenCVFileWriter(ucm_data.labels, pieceFilePath + "_labelsFine.yml", "fineLabels");
        log_info("ucm_data.labelsSize: %d x %d", ucm_data.labels.cols, ucm_data.labels.rows);

        /* show the boundary */
        cv::Mat finerBdry = cv::Mat::zeros(rows[gridRank], cols[gridRank], CV_8U);
        finerBdry.setTo(cv::Scalar(255), ucm_data.ucm > option.scaleK);
        String imageNameBdry = pieceFilePath + "_bdryFiner.png";
        OpenCVImageWriter(finerBdry, imageNameBdry);

        /* ==================== Mid and Coarse scales ====================== */

        /* ================================================================= */

        MPI_Barrier(grid_comm);

        // String imgFile = pieceFilePath + ".png";
        // cv::Mat im = cv::imread(imgFile);   /* reads as color image */

        /// TODO: @Yupeng - change it to a text file and import
        // int binStep = 4;
        // int nClasses = 3;
        bool colorImageFlag = 1;

        cv::Mat& localLabels = ucm_data.labels;     /* Note: localLabels mean baseLabels */
        vector<pair<int, int>>& bdry = ucm_data.bdry;

        SPFeatures DES(option.binStep, nBaseLabels, option.nClasses, colorImageFlag,
                        bigImg, localLabels, pieceFilePath);    /// remove mid and coarse labels
        if (option.mode == "P") {
            DES.genSpCenterFeatures();
        } else {
            DES.genSuperpixelFeatures();
        }

        ///log_debug("After genAuxFeatures");
        time(&now4);

        double seconds1 = difftime(now4, now3);
        log_info("gridRank: %d, feature extraction time: %f", gridRank, seconds1); // modified by Yupeng, 01/25

        MPI_Barrier(grid_comm);
        log_debug("After MPI Barrier");

        int* allNLabels = new int[dimSizes[0] * dimSizes[1]];
        MPI_Allgather(&nBaseLabels, 1, MPI_INT, allNLabels, 1, MPI_INT, grid_comm);
        log_debug("After Allgather");
        MPI_Barrier(grid_comm);
        log_debug("After MPI Barrier");

        /* each node just maintains the totalNumberOfLabels across all nodes */
        int totalNumberOfLabels = 0;
        for (int i = 0; i < dimSizes[0] * dimSizes[1]; i++) {
            totalNumberOfLabels += allNLabels[i];
        }

        /* each node first finds its offset and then adds its offset */
        int labelOffset = 0;
        for (int i = 0; i < gridRank; i++) {
            labelOffset += allNLabels[i];
        }

        cv::Mat labels = localLabels + labelOffset;

        MPI_Barrier(grid_comm);
        log_debug("After MPI Barrier");

        Mat tRowLabels = labels.rowRange(0, 1);
        Mat bRowLabels = labels.rowRange(labels.rows - 1, labels.rows);
        Mat lColLabels = labels.colRange(0, 1);
        Mat rColLabels = labels.colRange(labels.cols - 1, labels.cols);

        int* tRowData = (int*)tRowLabels.data;
        int* bRowData = (int*)bRowLabels.data;
        int* lColData = (int*)lColLabels.data;
        int* rColData = (int*)rColLabels.data;

        log_info("gridRank: %d, tRowLabels.rows: %d, tRowLabels.cols: %d", cols[gridRank], tRowLabels.rows, tRowLabels.cols);

        int* adjTRow = new int[cols[gridRank]];
        int* adjBRow = new int[cols[gridRank]];
        int* adjLCol = new int[rows[gridRank]];
        int* adjRCol = new int[rows[gridRank]];

        int up, down, left, right;
        MPI_Cart_shift(grid_comm, 1, 1, &left, &right);
        log_debug("After MPI Cart shift | L/R");
        MPI_Cart_shift(grid_comm, 0, 1, &up, &down);
        log_debug("After MPI Cart shift | U/D");

        if (up != MPI_PROC_NULL) {
            MPI_Send(tRowData, cols[gridRank], MPI_INT, up, 0, grid_comm);
            log_debug("After MPI Send | U");
        }
        if (down != MPI_PROC_NULL) {
            MPI_Send(bRowData, cols[gridRank], MPI_INT, down, 1, grid_comm);
            log_debug("After MPI Send | D");
        }
        if (left != MPI_PROC_NULL) {
            MPI_Send(lColData, rows[gridRank], MPI_INT, left, 2, grid_comm);
            log_debug("After MPI Send | L");
        }
        if (right != MPI_PROC_NULL) {
            MPI_Send(rColData, rows[gridRank], MPI_INT, right, 3, grid_comm);
            log_debug("After MPI Send | R");
        }

        if (up != MPI_PROC_NULL) {
            MPI_Recv(adjTRow, cols[gridRank], MPI_INT, up, 1, grid_comm, &status);
            log_debug("After MPI Receive | U");
        }
        if (down != MPI_PROC_NULL) {
            MPI_Recv(adjBRow, cols[gridRank], MPI_INT, down, 0, grid_comm, &status);
            log_debug("After MPI Receive | D");
        }
        if (left != MPI_PROC_NULL) {
            MPI_Recv(adjLCol, rows[gridRank], MPI_INT, left, 3, grid_comm, &status);
            log_debug("After MPI Receive | L");
        }
        if (right != MPI_PROC_NULL) {
            MPI_Recv(adjRCol, rows[gridRank], MPI_INT, right, 2, grid_comm, &status);
            log_debug("After MPI Receive | R");
        }

        int tl = MPI_PROC_NULL, tr = MPI_PROC_NULL, bl = MPI_PROC_NULL, br = MPI_PROC_NULL;

        if (coordinates[0] > 0 && coordinates[1] > 0) {
            int topleft[2] = {coordinates[0] - 1, coordinates[1] - 1};
            MPI_Cart_rank(grid_comm, topleft, &tl);
            log_debug("After MPI Cart rank | UL");
        }

        if (coordinates[0] < dimSizes[0] - 1 && coordinates[1] > 0) {
            int topright[2] = {coordinates[0] + 1, coordinates[1] - 1};
            MPI_Cart_rank(grid_comm, topright, &tr);
            log_debug("After MPI Cart rank | UR");
        }
        if (coordinates[0] > 0 && coordinates[1] < dimSizes[1] - 1) {
            int bottomleft[2] = {coordinates[0] - 1, coordinates[1] + 1};
            MPI_Cart_rank(grid_comm, bottomleft, &bl);
            log_debug("After MPI Cart rank | DL");
        }
        if (coordinates[0] < dimSizes[0] - 1 && coordinates[1] < dimSizes[1] - 1) {
            int bottomright[2] = {coordinates[0] + 1, coordinates[1] + 1};
            MPI_Cart_rank(grid_comm, bottomright, &br);
            log_debug("After MPI Cart rank | DR");
        }

        if (tl != MPI_PROC_NULL) {
            MPI_Send(tRowData, 1, MPI_INT, tl, 4, grid_comm);
            log_debug("After MPI Send | UL");
        }
        if (tr != MPI_PROC_NULL) {
            MPI_Send(tRowData + cols[gridRank] - 1, 1, MPI_INT, tr, 5, grid_comm);
            log_debug("After MPI Send | UR");
        }
        if (bl != MPI_PROC_NULL) {
            MPI_Send(bRowData, 1, MPI_INT, bl, 6, grid_comm);
            log_debug("After MPI Send | DL");
        }
        if (br != MPI_PROC_NULL) {
            MPI_Send(bRowData + cols[gridRank] - 1, 1, MPI_INT, br, 7, grid_comm);
            log_debug("After MPI Send | DR");
        }

        int aTL, aTR, aBL, aBR;
        if (tl != MPI_PROC_NULL) {
            MPI_Recv(&aTL, 1, MPI_INT, tl, 7, grid_comm, &status);
            log_debug("After MPI Receive | UL");
        }
        if (tr != MPI_PROC_NULL) {
            MPI_Recv(&aTR, 1, MPI_INT, tr, 6, grid_comm, &status);
            log_debug("After MPI Receive | UR");
        }
        if (bl != MPI_PROC_NULL) {
            MPI_Recv(&aBL, 1, MPI_INT, bl, 5, grid_comm, &status);
            log_debug("After MPI Receive | DL");
        }
        if (br != MPI_PROC_NULL) {
            MPI_Recv(&aBR, 1, MPI_INT, br, 4, grid_comm, &status);
            log_debug("After MPI Receive | DR");
        }

        /* create superpixel graph */
        vector<unordered_set<int>> superpixelGraph;
        // vector<vector<int> > superpixelGraph;
        CreateSuperpixelGraph(superpixelGraph, labels, bdry, nBaseLabels, labelOffset, adjTRow, adjBRow, adjLCol,
                              adjRCol, up, down, left, right, aTL, aTR, aBL, aBR, tl, tr, bl, br, rows[gridRank],
                              cols[gridRank]);
        log_info("Superpixel-graph generated at rank %d", gridRank);

        vector<int> nodeDegree;
        nodeDegree.resize(nBaseLabels);

        int cnt = 0;
        for (auto itI = superpixelGraph.begin(); itI != superpixelGraph.end(); itI++) {
            nodeDegree.at(cnt) = itI->size();
            if (nodeDegree.at(cnt) <= 0) {
                log_info("cnt: %d, itI.size: %lu, nodeDegree[cnt]: %d", cnt, itI->size(), nodeDegree.at(cnt));
            }
            cnt++;
        }
        assert(cnt == nBaseLabels);

        MPI_Barrier(grid_comm);
        log_debug("After MPI Barrier");

        /* =============== Ground truth data =============================================== */
        // modified by Yupeng, 04/23
        cv::Mat trainDescrs;
        vector<cv::Mat> gtImages;
        vector<int> gtSuperpixels, gtClassLabels;
        vector<pair<int, int>> gtCoords;
        int nGTPoints;

        if (!option.broadcast) {
            string gtFileName = pieceFilePath + "_GT.txt";
            log_info("Reading GT points from file: %s", gtFileName.c_str());
            nGTPoints = ReadGTFile(gtSuperpixels, gtClassLabels, gtFileName, labels, &gtCoords);
            log_debug("nGTPoints per node: %d", nGTPoints);

            // string seed = "arbitrary";
            trainDescrs = cv::Mat::zeros(gtCoords.size(), DES.nFeatures - DES.clusterCount, CV_64F);
            log_debug("DES.nFeatures: %d", DES.nFeatures);
            if (option.mode == "P" && option.seed == "P") {
                log_info("Generating GT features with patch-based method in arbitrary locations (Broadcast False)");
                log_debug("pieceFilePath: %s", pieceFilePath.c_str());
                DES.genSeedCenterFeatures(&gtCoords, trainDescrs);
                OpenCVFileWriter(trainDescrs, pieceFilePath + "_trainDescrs.yml", "trainDescrs");
            }
        } else {
            vector<cv::String> fn;
            std::string gtpath = bigPieceFilePath.substr(0, bigPieceFilePath.find_last_of("/") + 1) + "gtImages/";
            cv::glob(gtpath + "*.jpg", fn, false);
            trainDescrs = cv::Mat::zeros(static_cast<int>(fn.size()), DES.nFeatures - DES.clusterCount, CV_64F);
            if (option.mode == "P" && option.seed == "P") {
                log_info("Generating GT features with patch-based method in arbitrary locations (Broadcast True)");
                DES.genSeedCenterFeatures(&fn, gtClassLabels, trainDescrs);
                OpenCVFileWriter(trainDescrs, pieceFilePath + "_trainDescrs.yml", "trainDescrs");   // TODO: reduce to one file later
            }

            nGTPoints = trainDescrs.rows;
            gtSuperpixels = vector<int>(nGTPoints, 1);
        }

        int* gtSP_ptr = gtSuperpixels.data();
        int* gtCL_ptr = gtClassLabels.data();

        /* Number of all ground truth points at each node computed only at node 0 */
        int* allNGTPoints = NULL;
        if (gridRank == 0) {
            allNGTPoints = new int[dimSizes[0] * dimSizes[1]];
        }
        MPI_Gather(&nGTPoints, 1, MPI_INT, allNGTPoints, 1, MPI_INT, 0, grid_comm);
        log_debug("After MPI Allgather");
        MPI_Barrier(grid_comm);
        log_debug("After MPI Barrier");

        /* Total number of GT points across all nodes computed only at node 0 */
        int totalNGTPoints = 0;
        if (gridRank == 0) {
            for (int i = 0; i < dimSizes[0] * dimSizes[1]; i++) {
                totalNGTPoints += allNGTPoints[i];
                log_debug("totalNGTPoints: %d", totalNGTPoints);
            }
            log_debug("totalNGTPoints: %d", totalNGTPoints);
        }

        /// TODO: check for displacements in the two arrays below
        int* displacements = NULL;
        int* allGTPoints = NULL;
        int* allGTClassLabels = NULL;
        if (gridRank == 0) {
            displacements = new int[dimSizes[0] * dimSizes[1]];
            displacements[0] = 0;
            for (int i = 1; i < dimSizes[0] * dimSizes[1]; i++) {
                displacements[i] = displacements[i - 1] + allNGTPoints[i - 1];
            }

            allGTPoints = new int[totalNGTPoints];
            allGTClassLabels = new int[totalNGTPoints];
        }
        /// TODO: check for displacements here
        // Commented by Yupeng, 04/23 ---> allGTPoints are not used, but allGTClassLabels are used
        MPI_Barrier(grid_comm);
        log_debug("After MPI Barrier");
        MPI_Gatherv(gtSP_ptr, nGTPoints, MPI_INT, allGTPoints, allNGTPoints, displacements, MPI_INT, 0, grid_comm);
        log_debug("After MPI Gather");
        MPI_Gatherv(gtCL_ptr, nGTPoints, MPI_INT, allGTClassLabels, allNGTPoints, displacements, MPI_INT, 0, grid_comm);
        log_debug("After MPI Gather");

        /// TODO: check above strides and disps
        MPI_Barrier(grid_comm);
        log_debug("After MPI Barrier");

        /* ================= End of Ground truth data =============================================== */

        /// TODO: @Yupeng-test the binary-feature effects
        if (option.mode != "P") {
            /// FIXME @Yupeng - Probably not necessary
            // DES.DTFeatures(gtSuperpixels, gtClassLabels, labelOffset);
        }

        DES.addBiasOne();
        int nFeat = DES.nFeatures;

        /* ================= send GT Features ================= */
        cv::Mat gtFeatures;
        if (option.mode == "P" && option.seed == "P") {
            log_debug("Using patch-based method + arbitrary location training features...");
            trainDescrs.copyTo(gtFeatures);
            DES.addBiasOne(gtFeatures);
            log_info("gtFeatures.rows: %d, gtFeatures.cols: %d", gtFeatures.rows, gtFeatures.cols);
            // trainDescrs.copyTo(allGTFeatures.colRange(1/* + DES.nClasses*/, nFeat));
        } else {
            log_debug("Using original method to extract seed features from allDescrs...");
            gtFeatures = cv::Mat(0, nFeat, CV_64F);
            for (auto it = gtSuperpixels.begin(); it != gtSuperpixels.end(); it++)
                gtFeatures.push_back(DES.allDescrs.row(*it - labelOffset));
        }

        log_info("Send GT Features | labelOffset: %d", labelOffset);
        // OpenCVFileWriter(gtFeatures, pieceFilePath + "_gtF.yml", "gtFeatures");

        assert(nGTPoints == gtFeatures.rows);

        // check if they are continuous
        // log_info("gridRank: %d, gtFeatures.isContinuous: %d", gridRank, gtFeatures.isContinuous());
        // assert(gtFeatures.isContinuous());

        // Ensure that gtFeatures has continuous memory
        if(!gtFeatures.isContinuous()) {
            gtFeatures = gtFeatures.clone();
        }

        double* gtFeaturesPtr = (double*)gtFeatures.data;
        double* allGtFeaturesPtr = NULL;

        int* allNGTFeatures = NULL;
        int* featureDisp = NULL;
        if (gridRank == 0) {
            allNGTFeatures = new int[totalNGTPoints];
            for (int i = 0; i < dimSizes[0] * dimSizes[1]; i++) {
                allNGTFeatures[i] = allNGTPoints[i] * nFeat;
            }

            featureDisp = new int[totalNGTPoints];
            featureDisp[0] = 0;
            for (int i = 1; i < dimSizes[0] * dimSizes[1]; i++) {
                featureDisp[i] = featureDisp[i - 1] + allNGTFeatures[i - 1];
            }

            allGtFeaturesPtr = new double[totalNGTPoints * nFeat];
        }
        /// TODO: check feature disp

        MPI_Gatherv(gtFeaturesPtr, nGTPoints * nFeat, MPI_DOUBLE, allGtFeaturesPtr, allNGTFeatures, featureDisp,
                    MPI_DOUBLE, 0, grid_comm);

        MPI_Barrier(grid_comm);

        cv::Mat allGTFeatures;
        if (gridRank == 0) {
            allGTFeatures = cv::Mat(totalNGTPoints, nFeat, CV_64F, allGtFeaturesPtr);
            /// TODO @Yupeng - extract seed feature from arbitrary location + patch-based way
            log_debug("allGTFeatures.rows: %d, allGTFeatures.cols: %d", allGTFeatures.rows, allGTFeatures.cols);
            log_debug("totalNGTPoints: %d, nFeat: %d", totalNGTPoints, nFeat);
            // OpenCVFileWriter(allGTFeatures, pieceFilePath + "_all_GTF.yml", "allGTF");
        }

        /* ===================================================== */

        vector<int> uniqueTRow, uniqueBRow, uniqueLCol, uniqueRCol;

        int uTRowLen = PopulateUniqueVector(uniqueTRow, tRowData, cols[gridRank]);
        int uBRowLen = PopulateUniqueVector(uniqueBRow, bRowData, cols[gridRank]);
        int uLColLen = PopulateUniqueVector(uniqueLCol, lColData, rows[gridRank]);
        int uRColLen = PopulateUniqueVector(uniqueRCol, rColData, rows[gridRank]);

        int* uTRowDeg = new int[uTRowLen];
        int* uBRowDeg = new int[uBRowLen];
        int* uLColDeg = new int[uLColLen];
        int* uRColDeg = new int[uRColLen];

        cv::Mat uTopRowFeatures(0, nFeat, CV_64F);
        for (auto it = uniqueTRow.begin(); it != uniqueTRow.end(); it++) {
            uTopRowFeatures.push_back(DES.allDescrs.row(*it - labelOffset));
            uTRowDeg[it - uniqueTRow.begin()] = nodeDegree.at(*it - labelOffset);
        }

        cv::Mat uBotRowFeatures(0, nFeat, CV_64F);
        for (auto it = uniqueBRow.begin(); it != uniqueBRow.end(); it++) {
            uBotRowFeatures.push_back(DES.allDescrs.row(*it - labelOffset));
            uBRowDeg[it - uniqueBRow.begin()] = nodeDegree.at(*it - labelOffset);
        }

        cv::Mat uLeftColFeatures(0, nFeat, CV_64F);
        for (auto it = uniqueLCol.begin(); it != uniqueLCol.end(); it++) {
            uLeftColFeatures.push_back(DES.allDescrs.row(*it - labelOffset));
            uLColDeg[it - uniqueLCol.begin()] = nodeDegree.at(*it - labelOffset);
        }

        cv::Mat uRightColFeatures(0, nFeat, CV_64F);
        for (auto it = uniqueRCol.begin(); it != uniqueRCol.end(); it++) {
            uRightColFeatures.push_back(DES.allDescrs.row(*it - labelOffset));
            uRColDeg[it - uniqueRCol.begin()] = nodeDegree.at(*it - labelOffset);
        }

        /* push diag features for transfer in the following order: TL TR BL BR */
        cv::Mat tlFeatures(0, nFeat, CV_64F);
        tlFeatures.push_back(DES.allDescrs.row(*tRowData - labelOffset));
        cv::Mat trFeatures(0, nFeat, CV_64F);
        trFeatures.push_back(DES.allDescrs.row(*(tRowData + cols[gridRank] - 1) - labelOffset));
        cv::Mat blFeatures(0, nFeat, CV_64F);
        blFeatures.push_back(DES.allDescrs.row(*bRowData - labelOffset));
        cv::Mat brFeatures(0, nFeat, CV_64F);
        brFeatures.push_back(DES.allDescrs.row(*(bRowData + cols[gridRank] - 1) - labelOffset));

        int tlDeg, trDeg, blDeg, brDeg;
        tlDeg = nodeDegree.at(*tRowData - labelOffset);
        trDeg = nodeDegree.at(*(tRowData + cols[gridRank] - 1) - labelOffset);
        blDeg = nodeDegree.at(*bRowData - labelOffset);
        brDeg = nodeDegree.at(*(bRowData + cols[gridRank] - 1) - labelOffset);

        /* check continuity */
        log_info("uTopRow.isContinuous: %d ", uTopRowFeatures.isContinuous());
        log_info("uBotRow.isContinuous: %d", uBotRowFeatures.isContinuous());
        log_info("uLeftCol.isContinuous: %d", uLeftColFeatures.isContinuous());
        log_info("uRightCol.isContinuous: %d", uRightColFeatures.isContinuous());
        log_info("tlFeatures.isContinuous: %d", tlFeatures.isContinuous());
        log_info("trFeatures.isContinuous: %d", trFeatures.isContinuous());
        log_info("blFeatures.isContinuous: %d", blFeatures.isContinuous());
        log_info("brFeatures.isContinuous: %d", brFeatures.isContinuous());

        MPI_Barrier(grid_comm);

        /* send the unique superpixel order */
        int* uTRowData = (int*)uniqueTRow.data();
        int* uBRowData = (int*)uniqueBRow.data();
        int* uLColData = (int*)uniqueLCol.data();
        int* uRColData = (int*)uniqueRCol.data();

        if (up != MPI_PROC_NULL) {
            MPI_Send(uTRowData, uTRowLen, MPI_INT, up, 10, grid_comm);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Send(uBRowData, uBRowLen, MPI_INT, down, 11, grid_comm);
        }
        if (left != MPI_PROC_NULL) {
            MPI_Send(uLColData, uLColLen, MPI_INT, left, 12, grid_comm);
        }
        if (right != MPI_PROC_NULL) {
            MPI_Send(uRColData, uRColLen, MPI_INT, right, 13, grid_comm);
        }

        int* adjUTRow = new int[cols[gridRank]];
        int* adjUBRow = new int[cols[gridRank]];
        int* adjULCol = new int[rows[gridRank]];
        int* adjURCol = new int[rows[gridRank]];

        int adjUTRowLen = 0;
        if (up != MPI_PROC_NULL) {
            MPI_Recv(adjUTRow, cols[gridRank], MPI_INT, up, 11, grid_comm, &status);
            if (status.MPI_SOURCE == up) {
                log_info("Rank matches");
            }
            if (status.MPI_TAG == 11) {
                log_info("Message tag matched");
            }
            MPI_Get_count(&status, MPI_INT, &adjUTRowLen);
            log_debug("gridRank: %d, adjUTRowLen: %d", cols[gridRank], adjUTRowLen);
        }
        int adjUBRowLen = 0;
        if (down != MPI_PROC_NULL) {
            MPI_Recv(adjUBRow, cols[gridRank], MPI_INT, down, 10, grid_comm, &status);
            if (status.MPI_SOURCE == down) {
                log_debug("Rank matches");
            }
            if (status.MPI_TAG == 10) {
                log_debug("Message tag matched");
            }
            MPI_Get_count(&status, MPI_INT, &adjUBRowLen);
            log_debug("gridRank: %d, adjUBRowLen: %d", cols[gridRank], adjUBRowLen);
        }
        int adjULColLen = 0;
        if (left != MPI_PROC_NULL) {
            MPI_Recv(adjULCol, rows[gridRank], MPI_INT, left, 13, grid_comm, &status);
            if (status.MPI_SOURCE == left) {
                log_debug("Rank matches");
            }
            if (status.MPI_TAG == 13) {
                log_debug("Message tag matched");
            }
            MPI_Get_count(&status, MPI_INT, &adjULColLen);
            log_debug("gridRank: %d, adjULColLen: %d", cols[gridRank], adjULColLen);
        }
        int adjURColLen = 0;
        if (right != MPI_PROC_NULL) {
            MPI_Recv(adjURCol, rows[gridRank], MPI_INT, right, 12, grid_comm, &status);
            if (status.MPI_SOURCE == up) {
                log_debug("Rank matches");
            }
            if (status.MPI_TAG == 11) {
                log_debug("Message tag matched");
            }
            MPI_Get_count(&status, MPI_INT, &adjURColLen);
            log_debug("gridRank: %d, adjURColLen: %d", cols[gridRank], adjURColLen);
        }

        MPI_Barrier(grid_comm);
        
        /* Yupeng added, Feb 19th */
        double tBeforeGLSVM = MPI_Wtime();
        log_info("Before GLSVM: %f", tBeforeGLSVM - t1);

        /* now transfer the features */
        double* uTRF = (double*)uTopRowFeatures.data;
        double* uBRF = (double*)uBotRowFeatures.data;
        double* uLCF = (double*)uLeftColFeatures.data;
        double* uRCF = (double*)uRightColFeatures.data;
        double* tlF = (double*)tlFeatures.data;
        double* trF = (double*)trFeatures.data;
        double* blF = (double*)blFeatures.data;
        double* brF = (double*)brFeatures.data;

        log_info("Transferring features | sending length: %d", uTRowLen * nFeat);
        log_info("Transferring features | actual size: %d", uTopRowFeatures.rows * uTopRowFeatures.cols);

        double* adjUTRF = new double[adjUTRowLen * nFeat];
        double* adjUBRF = new double[adjUBRowLen * nFeat];
        double* adjULCF = new double[adjULColLen * nFeat];
        double* adjURCF = new double[adjURColLen * nFeat];
        double* adjTLF = new double[nFeat];
        double* adjTRF = new double[nFeat];
        double* adjBLF = new double[nFeat];
        double* adjBRF = new double[nFeat];

        if (up != MPI_PROC_NULL) {
            MPI_Sendrecv(uTRF, uTRowLen * nFeat, MPI_DOUBLE, up, 20, adjUTRF, adjUTRowLen * nFeat, MPI_DOUBLE, up, 21,
                         grid_comm, &status);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Sendrecv(uBRF, uBRowLen * nFeat, MPI_DOUBLE, down, 21, adjUBRF, adjUBRowLen * nFeat, MPI_DOUBLE, down,
                         20, grid_comm, &status);
        }
        if (left != MPI_PROC_NULL) {
            MPI_Sendrecv(uLCF, uLColLen * nFeat, MPI_DOUBLE, left, 22, adjULCF, adjULColLen * nFeat, MPI_DOUBLE, left,
                         23, grid_comm, &status);
        }
        if (right != MPI_PROC_NULL) {
            MPI_Sendrecv(uRCF, uRColLen * nFeat, MPI_DOUBLE, right, 23, adjURCF, adjURColLen * nFeat, MPI_DOUBLE, right,
                         22, grid_comm, &status);
        }
        if (tl != MPI_PROC_NULL) {
            MPI_Sendrecv(tlF, nFeat, MPI_DOUBLE, tl, 24, adjTLF, nFeat, MPI_DOUBLE, tl, 27, grid_comm, &status);
        }
        if (tr != MPI_PROC_NULL) {
            MPI_Sendrecv(trF, nFeat, MPI_DOUBLE, tr, 25, adjTRF, nFeat, MPI_DOUBLE, tr, 26, grid_comm, &status);
        }
        if (bl != MPI_PROC_NULL) {
            MPI_Sendrecv(blF, nFeat, MPI_DOUBLE, bl, 26, adjBLF, nFeat, MPI_DOUBLE, bl, 25, grid_comm, &status);
        }
        if (br != MPI_PROC_NULL) {
            MPI_Sendrecv(brF, nFeat, MPI_DOUBLE, br, 27, adjBRF, nFeat, MPI_DOUBLE, br, 24, grid_comm, &status);
        }

        cv::Mat aUTopRowFeatures(adjUTRowLen, nFeat, CV_64F, adjUTRF);
        cv::Mat aUBotRowFeatures(adjUBRowLen, nFeat, CV_64F, adjUBRF);
        cv::Mat aULeftColFeatures(adjULColLen, nFeat, CV_64F, adjULCF);
        cv::Mat aURightColFeatures(adjURColLen, nFeat, CV_64F, adjURCF);
        cv::Mat aTLFeatures(1, nFeat, CV_64F, adjTLF);
        cv::Mat aTRFeatures(1, nFeat, CV_64F, adjTRF);
        cv::Mat aBLFeatures(1, nFeat, CV_64F, adjBLF);
        cv::Mat aBRFeatures(1, nFeat, CV_64F, adjBRF);

        /* send and receive boundary SP degrees */
        int* adjUTRowDeg = new int[adjUTRowLen];
        int* adjUBRowDeg = new int[adjUBRowLen];
        int* adjULColDeg = new int[adjULColLen];
        int* adjURColDeg = new int[adjURColLen];
        int adjTLDeg, adjTRDeg, adjBLDeg, adjBRDeg;

        if (up != MPI_PROC_NULL) {
            MPI_Sendrecv(uTRowDeg, uTRowLen, MPI_INT, up, 28, adjUTRowDeg, adjUTRowLen, MPI_INT, up, 29, grid_comm,
                         &status);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Sendrecv(uBRowDeg, uBRowLen, MPI_INT, down, 29, adjUBRowDeg, adjUBRowLen, MPI_INT, down, 28, grid_comm,
                         &status);
        }
        if (left != MPI_PROC_NULL) {
            MPI_Sendrecv(uLColDeg, uLColLen, MPI_INT, left, 30, adjULColDeg, adjULColLen, MPI_INT, left, 31, grid_comm,
                         &status);
        }
        if (right != MPI_PROC_NULL) {
            MPI_Sendrecv(uRColDeg, uRColLen, MPI_INT, right, 31, adjURColDeg, adjURColLen, MPI_INT, right, 30,
                         grid_comm, &status);
        }
        if (tl != MPI_PROC_NULL) {
            MPI_Sendrecv(&tlDeg, 1, MPI_INT, tl, 32, &adjTLDeg, 1, MPI_INT, tl, 35, grid_comm, &status);
        }
        if (tr != MPI_PROC_NULL) {
            MPI_Sendrecv(&trDeg, 1, MPI_INT, tr, 33, &adjTRDeg, 1, MPI_INT, tr, 34, grid_comm, &status);
        }
        if (bl != MPI_PROC_NULL) {
            MPI_Sendrecv(&blDeg, 1, MPI_INT, bl, 34, &adjBLDeg, 1, MPI_INT, bl, 33, grid_comm, &status);
        }
        if (br != MPI_PROC_NULL) {
            MPI_Sendrecv(&brDeg, 1, MPI_INT, br, 35, &adjBRDeg, 1, MPI_INT, br, 32, grid_comm, &status);
        }

        MPI_Barrier(grid_comm);

        /* ======================= local Graph Solver ============================================ */
        // double tau = 1.0;
        /// TODO: @Yupeng - input the superpixel feature in patch-based way
        // OpenCVFileWriter(DES.allDescrs, pieceFilePath + "_F_beforeGLSVM.yml", "AllFeatures");
        GLSVM localGraphSolver(superpixelGraph, DES.allDescrs, aUTopRowFeatures, aUBotRowFeatures, aULeftColFeatures,
                               aURightColFeatures, aTLFeatures, aTRFeatures, aBLFeatures, aBRFeatures, adjUTRow,
                               adjUBRow, adjULCol, adjURCol, adjUTRowLen, adjUBRowLen, adjULColLen, adjURColLen, aTL,
                               aTR, aBL, aBR, nodeDegree, adjUTRowDeg, adjUBRowDeg, adjULColDeg, adjURColDeg, adjTLDeg,
                               adjTRDeg, adjBLDeg, adjBRDeg, nBaseLabels, labelOffset, nFeat, option.tau);
        localGraphSolver.GenerateA();

        cv::Mat& localA = localGraphSolver.A;
        // OpenCVFileWriter(localA, pieceFilePath + "_localA.yml", "localA");
        double* localA_ptr = (double*)localA.data;
        double* summedA_ptr = NULL;
        if (gridRank == 0) summedA_ptr = new double[nFeat * nFeat];
        MPI_Reduce(localA_ptr, summedA_ptr, nFeat * nFeat, MPI_DOUBLE, MPI_SUM, 0, grid_comm);
        MPI_Barrier(grid_comm);
        cv::Mat summedA;
        if (gridRank == 0) {
            ///summedA = cv::Mat(nFeat, nFeat, CV_64F, summedA_ptr);    // change back!!!!! Yupeng
            summedA = cv::Mat::zeros(nFeat, nFeat, CV_64F); /* moidified by Yupeng - completely SVM */
        }
        /// TODO: @Yupeng - should I use create here? No because it doesn't take pointer data
        /// TODO: @Yupeng - do I need char* typecast?
        /* ================================================================================= */

        /* ================= Now call the global function of the solver on root node only =========== */
        cv::Mat weightVectors;
        double* p_weightVectors;
        if (gridRank == 0) {
            /* modified by Yupeng */
            OpenCVFileWriter(summedA, "summedA_node_outside.yml", "summedA");
            /// TODO: @Yupeng - compute weight vector for each class
            localGraphSolver.FinalSolverAtRootNode(weightVectors, allGTFeatures, summedA, option.lambdaS,
                                                   option.lambdaH, option.epsZ, option.nClasses, totalNGTPoints,
                                                   allGTClassLabels, option.convergenceThreshold);
        /* ========================================================================================== */

            log_debug("weightVectors.isContinuous: %d", weightVectors.isContinuous());
            assert(weightVectors.isContinuous());
            assert(weightVectors.cols == option.nClasses && weightVectors.rows == nFeat);
            p_weightVectors = (double*)weightVectors.data;
        }
        else {
            p_weightVectors = new double[option.nClasses * nFeat];
        }

        // modified by Yupeng, 01/25
        time(&now5);
        log_info("gridRank: %d, GLSVM time: %f", gridRank, difftime(now5, now4));
        
        
        /* Now Broadcast the weight vector to all nodes */
        MPI_Barrier(grid_comm);
        MPI_Bcast(p_weightVectors, option.nClasses * nFeat, MPI_DOUBLE, 0, grid_comm);
        MPI_Barrier(grid_comm);

        if (gridRank != 0) {
            weightVectors = cv::Mat(nFeat, option.nClasses, CV_64F, (char*)p_weightVectors);
        }
        /// TODO: @Yupeng - ^^ Is char* typecast necessary here?

        /* Now compute the inner product between features and weights */
        assert(DES.allDescrs.cols == nFeat);
        assert(DES.allDescrs.rows == nBaseLabels);
        cv::Mat localClassLabelMat = DES.allDescrs * weightVectors;

        /// FIXME: @Yupeng - find the potential problem here, index not consistent with later part
        cv::Mat localClassLabelVec = cv::Mat::zeros(nBaseLabels, 1, CV_32S);
        for (int i = 1; i < nBaseLabels + 1; i++) {
            double maxVal = localClassLabelMat.at<double>(i - 1, 0);
            int maxValInd = 0;
            for (int j = 1; j < option.nClasses; j++) {
                double newMaxVal = localClassLabelMat.at<double>(i - 1, j);
                if (newMaxVal > maxVal) {
                    maxVal = newMaxVal;
                    maxValInd = j;
                }
            }
            /// FIXME: @Yupeng - find the potential problem here, index not consistent with later part
            localClassLabelVec.at<int>(i - 1) = maxValInd + 1;
            /*
             * ^^ + 1 is done above because classes are labeled from 1 and not 0.
             * And class label of zero is assigned to boundary pixels which do not belong to any superpixel.
             */
        }

        /* Organize the labels in the original image format */
        cv::Mat classLabelMat = cv::Mat::zeros(rows[gridRank], cols[gridRank], CV_32S);
        for (int i = 0; i < rows[gridRank]; i++) {
            for (int j = 0; j < cols[gridRank]; j++) {
                classLabelMat.at<int>(i, j) = localClassLabelVec.at<int>(localLabels.at<int>(i, j));
            }
        }

        // Merge class label data at root
        if(rank == 0) {
            classLabelMatVector[0] = classLabelMat;
            for(int i = 1; i < classLabelMatVector.size(); ++i) {
                classLabelMatVector[i] = MpiRecvMat(i);
            }
        }
        else if(rank < nBigIm) {
            // Send data to root process to combine
            MpiSendMat(classLabelMat, 0);
        }

        // Merge UCM data at root
        if(rank == 0) {
            ucmDataVector[0] = ucm_data.ucm;
            for(int i = 1; i < ucmDataVector.size(); ++i) {
                ucmDataVector[i] = MpiRecvMat(i);
            }
        }
        else if(rank < nBigIm) {
            // Send data to root process to combine
            MpiSendMat(ucm_data.ucm, 0);
        }

        // Get fine boundary data at root
        if(rank == 0) {
            fineBdryVector[0] = finerBdry;
            for(int i = 1; i < fineBdryVector.size(); ++i) {
                fineBdryVector[i] = MpiRecvMat(i);
            }
        }
        else if(rank < nBigIm) {
            // Send data to root process to combine
            MpiSendMat(finerBdry, 0);
        }


        // MPI_DEBUG: Yupeng added
        MPI_Barrier(grid_comm);
        if (option.nPieces != option.nBigPieces) {
            option.nPieces = atoi(argv[3]);
            option.nBigPieces = atoi(argv[4]);
            for (int i = 0; i < nprocs; i++) {
                if (rank != i) {
                    MPI_Send(&rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }
            log_info("rank (%d) & gridRank (%d) requests to wake up idle processes", rank, gridRank);
        } else {
            log_info("no need to wake up idle processes since all of them are working");
        }
    }

    t2 = MPI_Wtime();
    log_info("MPI_Wtime: %f", t2 - t1); // Yupeng modified, 01/25
    /// TODO: @Yupeng - compute classification accuracies based on ground-truth
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        /// TODO: @Yupeng - save the information below (compute multiple times)
        Size bigOrigSize = biggestImg.size();
        int rows = bigOrigSize.height;
        int cols = bigOrigSize.width;

        cv::Mat labelMap(rows, cols, CV_32S);
        cv::Mat ucmMap(rows, cols, CV_64F);
        cv::Mat fineBdryMap(rows, cols, CV_64F);
        cv::Mat midBdryMap(rows, cols, CV_64F);
        cv::Mat coarseBdryMap(rows, cols, CV_64F);

        string mark;
        mark = to_string(option.scaleK);
        // Merge ucm pieces into the big picture
        Pieces2Mat(bigPieceFilePath, mark, option.bigOverlap, option.nBigPieces, ucmMap, ucmDataVector);

        // Merge class label pieces into a big picture
        Pieces2Mat(bigPieceFilePath, mark, option.bigOverlap, option.nBigPieces, labelMap, classLabelMatVector);

        // Merge respective boundaries into a big picture
        Pieces2Mat(bigPieceFilePath, mark, option.bigOverlap, option.nBigPieces, fineBdryMap, fineBdryVector);

        vector<cv::Mat> labelsPerClass(option.nClasses);
        for(int i = 0; i < option.nClasses; ++i) {
            labelsPerClass[i] = cv::Mat::zeros(labelMap.rows, labelMap.cols, CV_32S);
        }

        for(int i = 0; i < labelMap.rows; ++i) {
            for(int j = 0; j < labelMap.cols; ++j) {
                labelsPerClass[labelMap.at<int>(i, j) - 1].at<int>(i, j) = 1;
            }
        }

        for(int i = 0; i < option.nClasses; ++i) {
            OpenCVImageWriter(labelsPerClass[i],
                    bigPieceFilePath + "_" +
                    mark +
                    "_classlabels_" + to_string(i + 1) + ".png"
            );
            OpenCVFileWriter(labelsPerClass[i],
                    bigPieceFilePath + "_" +
                    mark +
                    "_classlabels_" + to_string(i + 1) + ".yml",
                    "classlabels_" + to_string(i + 1)
            );
        }
        OpenCVImageWriter(ucmMap, bigPieceFilePath + "_" + mark + "_ucm.png");
        OpenCVFileWriter(ucmMap, bigPieceFilePath + "_" + mark + "_ucm.yml", "ucmMap");

        OpenCVImageWriter(labelMap, bigPieceFilePath + "_" + mark + "_classlabels.png");
        OpenCVFileWriter(labelMap, bigPieceFilePath + "_" + mark + "_classlabels.yml", "labelMap");

        OpenCVImageWriter(fineBdryMap, bigPieceFilePath + "_" + mark + "_bdry_fine.png");
        OpenCVFileWriter(fineBdryMap, bigPieceFilePath + "_" + mark + "_bdry_fine.yml", "boundary_fine");

        log_debug("Beginning to compute the classification accuracy...");
        int countPixels = 0;
        for (int i = 1; i <= option.nClasses; i++) {
            cv::Mat labels_map;
            string fname = bigPieceFilePath + "_labels_map_class_" + to_string(i) + ".yml";
            ifstream groundTruthFile(fname);
            // Gracefully exit if any of the ground truth files are missing
            if(!groundTruthFile.good()) {
                log_warn("Ground truth missing for class %d, unable to locate: %s", i, fname.c_str());
                MPI_Finalize();
                return 0;
            }
            cv::FileStorage fs(bigPieceFilePath + "_labels_map_class_" + to_string(i) + ".yml",
                               cv::FileStorage::READ);
            fs["labels_map"] >> labels_map;
            log_debug("labels_map | rows: %d, cols: %d", labels_map.rows, labels_map.cols);
            for (int r = 0; r < labels_map.rows; r++) {
                for (int c = 0; c < labels_map.cols; c++) {
                    if ((labelMap.at<int>(r, c) == i) && (floor(labels_map.at<float>(r, c) + 0.5) == 1)) {
                        countPixels++;
                    }
                }
            }
            fs.release();
        }
        log_info("Correct/Incorrect number of pixels: %d/%d", countPixels, rows * cols - countPixels);
        log_info("Pixel-based error: %f%%", (1 - 1.0 * countPixels / (rows * cols)) * 100);
    }
    MPI_Finalize();
    return 0;
}
