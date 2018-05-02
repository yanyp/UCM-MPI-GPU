# UCM

This is a C++ implementation of the [UCM](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf) algorithm. Note that it requires an Nvidia GPU.

## Compilation

For it to run, you need CMake (>= 2.8), GCC (>= 4.x), MPICH (= 3.x), OpenCV (= 2.x), [CUDA](https://developer.nvidia.com/cuda-downloads) (>= 7.5), and [ACML](http://developer.amd.com/tools-and-sdks/archive/acml-downloads-resources/) (>= 5.3)

CMake will detect all the libraries provided they are installed at their default paths. If that's not the case, you can use some of these CMake switches:

- ACML_INSTALL_DIR

```bash
cmake -DACML_INSTALL_DIR=$HOME/tech/acml5.3.1/gfortran64
```

- MPI_INSTALL_DIR

```bash
cmake -DMPI_INSTALL_DIR=$HOME/tech/mpi
```

- OPENCV_INSTALL_DIR

```bash
cmake -DOPENCV_INSTALL_DIR=$HOME/tech/opencv
```

- CUDA_ARCH

```bash
cmake -DCUDA_ARCH=sm_35
```

To compile, run:

```bash
cmake .
make -j20   # spawns 20 parallel jobs
```

## Execution

```bash
mpirun -n 4 ./bin/ucm-mpi ./labeledData/rio1.jpg 3 600 600 0.07 0.08 0.10
```

Let us go over the arguments:

- -n 4 : use 4 MPI processes. Note that the value of ```n``` has to be at least ```ceil(sqrt((<image_width> * <image_height>) / (largePatchSize * largePatchSize))) * ceil(sqrt((largePatchSize * largePatchSize) / (smallPatchSize * smallPatchSize)))```
- ./bin/ucmmpi : location of the executable
- ./labeledData/rio1.jpg : location of the input image
- 3 : 3 class labels. For ```rio1.jpg```, these 3 class labels denote slum, urban, and forest.
- 600 : size of the small patch, over which gPb is computed. Also referred to as ```smallPatchSize```
- 600 : size of the large patch, over which UCM is computed. Also referred to as ```largePatchSize```
- 0.07 : fine level threshold for the UCM
- 0.08 : mid level threshold for the UCM
- 0.10 : coarse level threshold for the UCM

## Labeled data

Each input image is to have a labeled ground truth file and a texton codebook. Optionally, you can provide ```n``` other labeled files to evaluate the pixel-level classification error:

- Ground truth : A few pixels that are classified as class_1, class_2, etc. This is the input for the SVM that classifies the superpixels later on. The file has to be named ```<filename>_GT.txt```
- texton codebook : A 13 x 32 OpenCV matrix, denoting the 32 13-dimensional textons as a result of k-means over the texton responses in MATLAB. The file has to be named ```<filename>_texton_codebook.yml```
- class_n matrix : An OpenCV matrix of the same dimensions as the image, filled with 0s (pixel does not belong to class_n) and 1s (pixel belongs to class_n). The files have to be named ```<filename>_labels_map_class_<class_no>.yml```
