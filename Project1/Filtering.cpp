#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
using namespace cv;


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
    {1,  4, 1},
    {4, 12, 4},
    {1,  4, 1}
};

// High-pass (sharpening) filters
int hpf_filter_1[3][3] = {
    { 0, -1,  0},
    {-1,  5, -1},
    { 0, -1,  0}
};

int hpf_filter_2[3][3] = {
    {-1, -1, -1},
    {-1,  9, -1},
    {-1, -1, -1}
};

int hpf_filter_3[3][3] = {
    { 1, -2,  1},
    {-2,  5, -2},
    { 1, -2,  1}
};

/**
 * applyConvolutionFilter - Apply a 3x3 convolution filter to an RGB image
 * @param src - Source image
 * @param filter - 3x3 filter mask
 * @param divisor - Normalization divisor (sum of filter weights)
 * @return - Filtered image
 */
Mat applyConvolutionFilter(Mat src, int filter[3][3]) {
    Mat dst = Mat::zeros(src.size(), src.type());
    int rows = src.rows;
    int cols = src.cols;


    int divisor = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            divisor += filter[i][j];
        }
    }

    // Avoid divide-by-zero (edge detection filters)
    if (divisor == 0) divisor = 1;

    cout << "Using divisor: " << divisor << endl;

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                int sum = 0;

                for (int a = -1; a <= 1; a++) {
                    for (int b = -1; b <= 1; b++) {
                        sum += src.at<Vec3b>(i + a, j + b)[c] *
                                filter[a + 1][b + 1];
                    }
                }

                sum = sum / divisor;

                if (sum < 0) sum = 0;
                if (sum > 255) sum = 255;

                dst.at<Vec3b>(i, j)[c] = (uchar)sum;
            }
        }
    }
    

    return dst;
}

/**
 * applyMedianFilter - Apply a 3x3 median filter to an RGB image
 * Collects the 9 neighborhood values for each channel, sorts them,
 * and writes back the middle value.
 */
Mat applyMedianFilter(Mat src) {
    Mat dst = src.clone();
    int rows = src.rows;
    int cols = src.cols;

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                vector<int> values;
                values.reserve(9);

                for (int a = -1; a <= 1; a++) {
                    for (int b = -1; b <= 1; b++) {
                        values.push_back(src.at<Vec3b>(i + a, j + b)[c]);
                    }
                }

                sort(values.begin(), values.end());
                dst.at<Vec3b>(i, j)[c] = static_cast<uchar>(values[4]);
            }
        }
    }

    return dst;
}



/**
 * applyLowPixelFilter - Apply a 3x3 low-pixel filter to an RGB image.
 * Replaces each center pixel with the minimum value from its neighborhood.
 */
Mat applyLowPixelFilter(Mat src) {
    Mat dst = src.clone();
    int rows = src.rows;
    int cols = src.cols;

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                vector<int> values;
                values.reserve(9);

                for (int a = -1; a <= 1; a++) {
                    for (int b = -1; b <= 1; b++) {
                        values.push_back(src.at<Vec3b>(i + a, j + b)[c]);
                    }
                }

                dst.at<Vec3b>(i, j)[c] = static_cast<uchar>(*min_element(values.begin(), values.end()));
            }
        }
    }

    return dst;
}

/**
 * applyHighPixelFilter - Apply a 3x3 high-pixel filter to an RGB image.
 * Replaces each center pixel with the maximum value from its neighborhood.
 */
Mat applyHighPixelFilter(Mat src) {
    Mat dst = src.clone();
    int rows = src.rows;
    int cols = src.cols;

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                vector<int> values;
                values.reserve(9);

                for (int a = -1; a <= 1; a++) {
                    for (int b = -1; b <= 1; b++) {
                        values.push_back(src.at<Vec3b>(i + a, j + b)[c]);
                    }
                }

                dst.at<Vec3b>(i, j)[c] = static_cast<uchar>(*max_element(values.begin(), values.end()));
            }
        }
    }

    return dst;
}



int main( int argc, char** argv )
{
    String imageName("../data/lena.jpg"); // by default
    if (argc > 1)
    {
        imageName = argv[1];
    }
    
    Mat image = imread( imageName );
    if (image.empty()) {
        cout << "Could not load image: " << imageName << endl;
        return -1;
    }
    
    cout << "===== Image Filtering Tool =====" << endl;
    cout << "Image loaded: " << imageName << endl;
    cout << "Image size: " << image.rows << " x " << image.cols << endl << endl;
    
    cout << "Select an algorithm:" << endl;
    cout << "1. Low-pass (lpf_filter_6)" << endl;
    cout << "2. Low-pass (lpf_filter_9)" << endl;
    cout << "3. Low-pass (lpf_filter_10)" << endl;
    cout << "4. Low-pass (lpf_filter_16)" << endl;
    cout << "5. Low-pass (lpf_filter_32)" << endl;
    cout << "6. High-pass (hpf_filter_1)" << endl;
    cout << "7. High-pass (hpf_filter_2)" << endl;
    cout << "8. High-pass (hpf_filter_3)" << endl;
    cout << "9. Median filter" << endl;
    cout << "10. Min filter" << endl;
    cout << "11. Max filter" << endl;
    cout << "Enter choice: ";
    int choice;
    cin >> choice;

    Mat filtered;
    auto filterStart = chrono::high_resolution_clock::now();

    switch (choice) {
        case 1:
            filtered = applyConvolutionFilter(image, lpf_filter_6);
            break;
        case 2:
            filtered = applyConvolutionFilter(image, lpf_filter_9);
            break;
        case 3:
            filtered = applyConvolutionFilter(image, lpf_filter_10);
            break;
        case 4:
            filtered = applyConvolutionFilter(image, lpf_filter_16);
            break;
        case 5:
            filtered = applyConvolutionFilter(image, lpf_filter_32);
            break;
        case 6:
            filtered = applyConvolutionFilter(image, hpf_filter_1);
            break;
        case 7:
            filtered = applyConvolutionFilter(image, hpf_filter_2);
            break;
        case 8:
            filtered = applyConvolutionFilter(image, hpf_filter_3);
            break;
        case 9:
            filtered = applyMedianFilter(image);
            break;
        case 10:
            filtered = applyLowPixelFilter(image);
            break;
        case 11:
            filtered = applyHighPixelFilter(image);
            break;
        default:
            cout << "Invalid choice. Using default blur.\n";
            filtered = applyConvolutionFilter(image, lpf_filter_9);
    }

    auto filterEnd = chrono::high_resolution_clock::now();
    auto filterMs = chrono::duration_cast<chrono::duration<double, std::milli>>(filterEnd - filterStart).count();
    cout << "Filter time: " << filterMs << " ms" << endl;
    
    // Display original and filtered images
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Filtered Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    imshow("Filtered Image", filtered);
    
    cout << "Press any key to close windows..." << endl;
    waitKey();
    
    return 0;
}
