#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define a

int main(int argc, char *argv[])
{
    if (argc < 6) {
        cout << "Please specify the desired origin, destination, threshold numbers and brightness change amount." << endl;
        return 0;
    }
    string img_path;
    img_path = argv[1];
    // img_path = "./Test_Images/01_7680x4320.jpg";
    int temp_threshold_1, temp_threshold_2, temp_brigtness_change;
    try {
        temp_threshold_1 = atoi(argv[3]);
        temp_threshold_2 = atoi(argv[4]);
        temp_brigtness_change = atoi(argv[5]);
    } catch (int e) {
        cout << "Please specify a valid number for threshold and brightness change amount." << endl;
        return 0;
    }
    const int threshold_1 = temp_threshold_1, threshold_2 = temp_threshold_2;
    const int brigtness_change = temp_brigtness_change;

    Mat input_img = imread(img_path, IMREAD_COLOR);

    int channels = input_img.channels();
    int img_size = input_img.rows * input_img.cols * channels;
    int row_byte_offset = input_img.cols * channels;

    int row_len = input_img.cols;

    // Apply sobel filter.
    uint8_t *tmp = new uint8_t[input_img.rows * input_img.cols];
    uint8_t *result = new uint8_t[img_size];
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < input_img.rows; i++)
        for (int j = 0; j < input_img.cols; j++)
            tmp[i * row_len + j] = (((int) input_img.data[i * row_byte_offset + j * channels] + 
                                            input_img.data[i * row_byte_offset + j * channels + 1] +
                                            input_img.data[i * row_byte_offset + j * channels + 2] +
                                            3 * brigtness_change)) / 3;
    for (int i = 0; i < input_img.rows * input_img.cols; i++)
    {
        int res_x = 0;
        int res_y = 0;
        bool left_bound = i % input_img.cols != 0;
        bool right_bound = i % input_img.cols != input_img.cols - 1;
        if (i - row_len >= 0)
        {
            res_y += ((int) tmp[i - row_len]) << 1;
            if (left_bound)
            {
                res_x -= tmp[i - row_len - 1];
                res_y += tmp[i - row_len - 1];
            }
            if (right_bound)
            {
                res_x += tmp[i - row_len + 1];
                res_y += tmp[i - row_len + 1];
            }
        }
        if (i + row_len < img_size)
        {
            res_y -= ((int) tmp[i + row_len]) << 1;
            if (left_bound)
            {
                res_x -= tmp[i + row_len - 1];
                res_y -= tmp[i + row_len - 1];
            }
            if (right_bound)
            {
                res_x += tmp[i + row_len + 1];
                res_y -= tmp[i + row_len + 1];
            }
        }
        if (left_bound)
            res_x -= ((int) tmp[i - 1]) << 1;
        if (right_bound)
            res_x += ((int) tmp[i + 1]) << 1;
        res_x = abs(res_x);
        res_y = abs(res_y);
        result[i] = (res_x + res_y < 256 ? res_x + res_y : 255);
        if (result[i] <= threshold_1)
            result[i] = 0;
        if (result[i] >= threshold_2)
            result[i] = 255;
    }
    Mat output_img(input_img.rows, input_img.cols, CV_8UC1);
    auto t2 = chrono::high_resolution_clock::now();
    memcpy(output_img.data, result, input_img.rows * input_img.cols);
    delete tmp;
    delete result;

    long long microseconds = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    cout << "Execution Time: " << microseconds << " microseconds" << endl;

    imwrite(argv[2], output_img);
}