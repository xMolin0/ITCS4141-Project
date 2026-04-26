/*
============================================================
CUDA Image Filtering — Readable Version

Same functionality as your original program,
but rewritten to be easier to study and understand.

Filters included:
1. LPF blur (divisor 6)
2. LPF blur (divisor 9)
3. LPF blur (divisor 10)
4. LPF blur (divisor 16)
5. LPF blur (divisor 32)
6. HPF sharpen (filter 1)
7. HPF sharpen (filter 2)
8. HPF sharpen (filter 3)
9. Median Filter
10. Min Pixel Filter
11. Max Pixel Filter

Core CUDA idea:
    1 thread = 1 pixel
============================================================
*/

#include <iostream>
#include <cuda_runtime.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;


/*
============================================================
FILTER MASKS
============================================================
*/

/* Low-pass filters (blur) */

int lpf_filter_6[3][3] = {
    {0, 1, 0},
    {1, 2, 1},
    {0, 1, 0}
};

int lpf_filter_9[3][3] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1}
};

int lpf_filter_10[3][3] = {
    {1, 1, 1},
    {1, 2, 1},
    {1, 1, 1}
};

int lpf_filter_16[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

int lpf_filter_32[3][3] = {
    {1, 4, 1},
    {4, 12, 4},
    {1, 4, 1}
};

/* High-pass filters (sharpen) */

int hpf_filter_1[3][3] = {
    {0, -1, 0},
    {-1, 5, -1},
    {0, -1, 0}
};

int hpf_filter_2[3][3] = {
    {-1, -1, -1},
    {-1, 9, -1},
    {-1, -1, -1}
};

int hpf_filter_3[3][3] = {
    {1, -2, 1},
    {-2, 5, -2},
    {1, -2, 1}
};


/*
============================================================
CUDA ERROR CHECKER
============================================================
*/

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        cout << "CUDA Error: "                                \
             << cudaGetErrorString(err)                       \
             << " at line " << __LINE__ << endl;              \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)


/*
============================================================
KERNEL 1: CONVOLUTION FILTER

Used for:
- Blur
- Sharpen
- Edge enhancement

Each thread:
    computes ONE output pixel

It reads the 3x3 neighborhood,
multiplies by filter weights,
divides by divisor,
then clamps result to [0,255].
============================================================
*/

__global__ void convolutionKernel(
    unsigned char* src,
    unsigned char* dst,
    size_t pitch,
    int width,
    int height,
    int* filter,
    int divisor
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Skip border pixels */

    if (x < 1 || x >= width - 1 ||
        y < 1 || y >= height - 1)
        return;

    int blue = 0;
    int green = 0;
    int red = 0;

    /*
    Visit all 9 neighboring pixels
    */

    for (int dy = -1; dy <= 1; dy++)
    {
        uchar3* row = (uchar3*)(src + (y + dy) * pitch);

        for (int dx = -1; dx <= 1; dx++)
        {
            uchar3 pixel = row[x + dx];

            int weight = filter[(dy + 1) * 3 + (dx + 1)];

            blue += pixel.x * weight;
            green += pixel.y * weight;
            red += pixel.z * weight;
        }
    }

    /*
    Divide + clamp
    */

    blue = max(0, min(255, blue / divisor));
    green = max(0, min(255, green / divisor));
    red = max(0, min(255, red / divisor));

    uchar3 result;
    result.x = blue;
    result.y = green;
    result.z = red;

    uchar3* outRow = (uchar3*)(dst + y * pitch);
    outRow[x] = result;
}


/*
============================================================
MEDIAN FILTER HELPER

Sort 9 values using insertion sort
(very fast for tiny arrays)
============================================================
*/

__device__ void sort9(unsigned char arr[9])
{
    for (int i = 1; i < 9; i++)
    {
        unsigned char key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }

        arr[j + 1] = key;
    }
}


/*
============================================================
KERNEL 2: MEDIAN FILTER

Used for:
- salt and pepper noise removal

Instead of averaging:
    sort values
    take middle value
============================================================
*/

__global__ void medianKernel(
    unsigned char* src,
    unsigned char* dst,
    size_t pitch,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 ||
        y < 1 || y >= height - 1)
        return;

    unsigned char blue[9];
    unsigned char green[9];
    unsigned char red[9];

    int index = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        uchar3* row = (uchar3*)(src + (y + dy) * pitch);

        for (int dx = -1; dx <= 1; dx++)
        {
            uchar3 pixel = row[x + dx];

            blue[index] = pixel.x;
            green[index] = pixel.y;
            red[index] = pixel.z;

            index++;
        }
    }

    sort9(blue);
    sort9(green);
    sort9(red);

    uchar3 result;
    result.x = blue[4];
    result.y = green[4];
    result.z = red[4];

    uchar3* outRow = (uchar3*)(dst + y * pitch);
    outRow[x] = result;
}


/*
============================================================
KERNEL 3: MIN PIXEL FILTER

Take smallest value from 3x3 area
============================================================
*/

__global__ void minKernel(
    unsigned char* src,
    unsigned char* dst,
    size_t pitch,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 ||
        y < 1 || y >= height - 1)
        return;

    unsigned char minB = 255;
    unsigned char minG = 255;
    unsigned char minR = 255;

    for (int dy = -1; dy <= 1; dy++)
    {
        uchar3* row = (uchar3*)(src + (y + dy) * pitch);

        for (int dx = -1; dx <= 1; dx++)
        {
            uchar3 pixel = row[x + dx];

            if (pixel.x < minB) minB = pixel.x;
            if (pixel.y < minG) minG = pixel.y;
            if (pixel.z < minR) minR = pixel.z;
        }
    }

    uchar3 result = {minB, minG, minR};

    uchar3* outRow = (uchar3*)(dst + y * pitch);
    outRow[x] = result;
}


/*
============================================================
KERNEL 4: MAX PIXEL FILTER

Take largest value from 3x3 area
============================================================
*/

__global__ void maxKernel(
    unsigned char* src,
    unsigned char* dst,
    size_t pitch,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 ||
        y < 1 || y >= height - 1)
        return;

    unsigned char maxB = 0;
    unsigned char maxG = 0;
    unsigned char maxR = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        uchar3* row = (uchar3*)(src + (y + dy) * pitch);

        for (int dx = -1; dx <= 1; dx++)
        {
            uchar3 pixel = row[x + dx];

            if (pixel.x > maxB) maxB = pixel.x;
            if (pixel.y > maxG) maxG = pixel.y;
            if (pixel.z > maxR) maxR = pixel.z;
        }
    }

    uchar3 result = {maxB, maxG, maxR};

    uchar3* outRow = (uchar3*)(dst + y * pitch);
    outRow[x] = result;
}


/*
============================================================
MAIN
============================================================
*/

int main()
{
    /*
    Load image
    */

    Mat image = imread("../data/lena.jpg");

    if (image.empty())
    {
        cout << "Could not load image\n";
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t rowBytes = width * 3;

    /*
    Allocate GPU memory
    */

    unsigned char* d_src;
    unsigned char* d_dst;
    size_t pitch;

    CUDA_CHECK(cudaMallocPitch(
        &d_src,
        &pitch,
        rowBytes,
        height
    ));

    CUDA_CHECK(cudaMallocPitch(
        &d_dst,
        &pitch,
        rowBytes,
        height
    ));

    /*
    Copy CPU -> GPU
    */

    CUDA_CHECK(cudaMemcpy2D(
        d_src,
        pitch,
        image.ptr(),
        image.step,
        rowBytes,
        height,
        cudaMemcpyHostToDevice
    ));

    /*
    Thread layout
    */

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + 15) / 16,
        (height + 15) / 16
    );

    /*
    Menu
    */

    cout << "\n===== CUDA Image Filters =====\n";
    cout << "1. LPF divisor 6\n";
    cout << "2. LPF divisor 9\n";
    cout << "3. LPF divisor 10\n";
    cout << "4. LPF divisor 16\n";
    cout << "5. LPF divisor 32\n";
    cout << "6. HPF sharpen 1\n";
    cout << "7. HPF sharpen 2\n";
    cout << "8. HPF sharpen 3\n";
    cout << "9. Median Filter\n";
    cout << "10. Min Pixel Filter\n";
    cout << "11. Max Pixel Filter\n";
    cout << "Choose: ";

    int choice;
    cin >> choice;

    /*
    Select filter
    */

    int (*selectedMask)[3] = nullptr;

    switch (choice)
    {
        case 1: selectedMask = lpf_filter_6; break;
        case 2: selectedMask = lpf_filter_9; break;
        case 3: selectedMask = lpf_filter_10; break;
        case 4: selectedMask = lpf_filter_16; break;
        case 5: selectedMask = lpf_filter_32; break;
        case 6: selectedMask = hpf_filter_1; break;
        case 7: selectedMask = hpf_filter_2; break;
        case 8: selectedMask = hpf_filter_3; break;
    }

    /*
    Run selected filter
    */

    if (selectedMask != nullptr)
    {
        int flat[9];
        int divisor = 0;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                flat[i * 3 + j] = selectedMask[i][j];
                divisor += selectedMask[i][j];
            }
        }

        if (divisor == 0)
            divisor = 1;

        int* d_filter;

        CUDA_CHECK(cudaMalloc(&d_filter, 9 * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(
            d_filter,
            flat,
            9 * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        convolutionKernel<<<gridSize, blockSize>>>(
            d_src,
            d_dst,
            pitch,
            width,
            height,
            d_filter,
            divisor
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_filter);
    }
    else if (choice == 9)
    {
        medianKernel<<<gridSize, blockSize>>>(
            d_src, d_dst, pitch, width, height
        );
    }
    else if (choice == 10)
    {
        minKernel<<<gridSize, blockSize>>>(
            d_src, d_dst, pitch, width, height
        );
    }
    else
    {
        maxKernel<<<gridSize, blockSize>>>(
            d_src, d_dst, pitch, width, height
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /*
    Copy GPU -> CPU
    */

    Mat result(height, width, CV_8UC3);

    CUDA_CHECK(cudaMemcpy2D(
        result.ptr(),
        result.step,
        d_dst,
        pitch,
        rowBytes,
        height,
        cudaMemcpyDeviceToHost
    ));

    /*
    Show result
    */

    imshow("Original", image);
    imshow("Filtered", result);

    waitKey();

    /*
    Cleanup
    */

    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}