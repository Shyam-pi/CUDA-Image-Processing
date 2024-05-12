# CUDA-Image-Processing

This repo consists of various image processing pipelines for Parallel Computing using CUDA. By harnessing the power of CUDA, it aims to accelerate various image processing tasks, such as gaussian blurring, edge detection etc., by parallelizing computations on GPU architectures. [repo](https://github.com/thomasplantin/cuda-image-processing/tree/master)

### Test Cases

To run the available test cases, run the following in your terminal:
```chmod u+x ex_batch.sh``` // To change permissions

Followed by running the following in your terminal:
```ex_single.sh```

This will run all of our filters on the image "images/lena_rgb.png" and will be saved in 'outputs' folder in root.

### Repository structure

main.cu simply parses arguments and calls the necessary filters
stb_image/ contains the image library used.
image.h is a wrapper for the image library.
filters/ contains each filter implemented in a header file
filters/convolve.h is called from every convolution filter.
Other filters have their own kernels.
expected_output/ stores the expected_output of each filter
images/lena_rgb.png is the input image for all the test cases.

### Manual Running

Note: You must have the ability to run CUDA files on your end in order to render any of the work in this repository. For more information about CUDA, please visit this link: https://developer.nvidia.com/about-cuda

Before any filters can be applied, the `main.cu` file must be compiled. To do that, open your terminal and run the following command from the root directory of this project:

```nvcc main.cu```

You can ignore any warnings that are printed to the console. A file named `a.out` should now be stored in the root directory.

To apply a filter to an image, please follow the next steps:
* Import an image of your choice in the `images` directory, or just use one of the images already there.
* From the root directory, run `a.out` with the following arguments (see filter arguments in the table below):

```./a.out path_to_image_input path_to_image_output filter_arg```

### Table: Filters and their arguments
|      Filter     |  Filter Arg |
|:---------------:|:-----------:|
| Horizontal Flip | hflip       |
| Vertical Flip   | vflip       |
| Sharpening      | sharpen     |
| Box Blurring    | blur        |
| Grayscale       | gray        |
| Grayscale-wtd   | grayweight  |
| Gaussian Blur   | gauss       |
| Edge Detection  | edge        |
