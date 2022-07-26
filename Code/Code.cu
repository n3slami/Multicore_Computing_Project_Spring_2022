#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define MAX_GRID_SIZE 1024
#define BLOCK_SIZE 1024
#define TILE_HEIGHT 8
#define get_mean_3(img,ind) ((((int) img[ind]) + img[ind + 1] + img[ind + 2]) / 3)

__global__ void change_brightness(uint8_t *img, int width, int height, int channels, int offset)
{
    const int img_byte_size = width * height * channels;
    const int unrolling = 4;
    const int stride = gridDim.x * blockDim.x * unrolling;
    for (int base_index = blockIdx.x * blockDim.x * unrolling; base_index < img_byte_size; base_index += stride)
    {
        int ptr = base_index + threadIdx.x * unrolling, val;
        int overflow, underflow;
        
        if (ptr + 3 < img_byte_size)
        {
            val = img[ptr];
            val += offset;
            overflow = val > 255;
            underflow = val < 0;
            val &= (underflow - 1);
            val |= (-overflow);
            img[ptr] = val & 255;


            val = img[ptr + 1];
            val += offset;
            overflow = val > 255;
            underflow = val < 0;
            val &= (underflow - 1);
            val |= (-overflow);
            img[ptr + 1] = val & 255;
            
            val = img[ptr + 2];
            val += offset;
            overflow = val > 255;
            underflow = val < 0;
            val &= (underflow - 1);
            val |= (-overflow);
            img[ptr + 2] = val & 255;

            val = img[ptr + 3];
            val += offset;
            overflow = val > 255;
            underflow = val < 0;
            val &= (underflow - 1);
            val |= (-overflow);
            img[ptr + 3] = val & 255;
        }
        else
        {
            for (int i = 0; i < img_byte_size - ptr; i++)
            {
                val = img[ptr + i];
                val += offset;
                overflow = val > 255;
                underflow = val < 0;
                val &= (underflow - 1);
                val |= (-overflow);
                img[ptr + i] = val & 255;
            }
        }
    }
}

__global__ void tiled_sobel(uint8_t *out, uint8_t *img, int width, int height, int channels, uint8_t t1, uint8_t t2)
{   // Tiling is used to minimize the amount of reads to the memory hierarchy.
    const int img_byte_size = width * height * channels;
    const int row_size = width * channels;
    const int tile_stride = gridDim.x;
    const int tile_cols = BLOCK_SIZE / TILE_HEIGHT;
    const int tiles_per_row = (width + tile_cols - 1) / tile_cols;
    const int total_tiles = ((height + TILE_HEIGHT - 3) / TILE_HEIGHT) * tiles_per_row;
    __shared__ uint8_t smem[TILE_HEIGHT + 2][tile_cols + 2];

    for (int tile_index = blockIdx.x; tile_index < total_tiles; tile_index += tile_stride)
    {   // Loop unrolling wouldn't really help much here, since this is very compute intensive.
        int y_base = (tile_index / tiles_per_row) * TILE_HEIGHT + 1;
        int x_base = (tile_index % tiles_per_row) * tile_cols;
        int ptr = y_base * row_size + x_base * channels;
        int ind_x = threadIdx.x + 1, ind_y = threadIdx.y + 1;
        int actual_tile_width = min(tile_cols, width - x_base);
        int actual_tile_height = min(TILE_HEIGHT, height - 1 - y_base);
        int location;

        // Load the middle smem part with no divergence.
        int should_calc = threadIdx.x < actual_tile_width && threadIdx.y < actual_tile_height;
        smem[ind_y][ind_x] = (-should_calc) & get_mean_3(img, threadIdx.y * row_size + threadIdx.x * channels + ptr);

        // Handle the corners of the conv with minimal divergence
        bool should_handle_corners = (threadIdx.y == 0 || threadIdx.y == TILE_HEIGHT - 1)
                                    || (threadIdx.x == 0 || threadIdx.x == tile_cols - 1);
        int corner_x = (threadIdx.x == 0) * (-1) + (threadIdx.x == tile_cols - 1) * tile_cols;
        int corner_y = (threadIdx.y == 0) * (-1) + (threadIdx.y == TILE_HEIGHT - 1) * TILE_HEIGHT;
        corner_x += (corner_x == 0) * threadIdx.x;
        corner_y += (corner_y == 0) * threadIdx.y;
        if (should_handle_corners)
        {
            location = corner_y * row_size + corner_x * channels + ptr;
            bool should_zero_out = (x_base + corner_x < 0 || x_base + corner_x >= width)
                            || (location >= img_byte_size || location < 0);
            smem[corner_y + 1][corner_x + 1] = (should_zero_out ? 0 : get_mean_3(img, location));
        }
        should_handle_corners = (threadIdx.y == 0 || threadIdx.y == TILE_HEIGHT - 1)
                                    && (threadIdx.x == 0 || threadIdx.x == tile_cols - 1);
        if (should_handle_corners)
        {
            corner_x = (threadIdx.x == 0) * (-1) + (threadIdx.x == tile_cols - 1) * tile_cols;
            corner_y = (threadIdx.y == 0) * (-1) + (threadIdx.y == TILE_HEIGHT - 1) * TILE_HEIGHT;
            location = corner_y * row_size + threadIdx.x * channels + ptr;
            smem[corner_y + 1][threadIdx.x + 1] = get_mean_3(img, location);
            location = threadIdx.y * row_size + corner_x * channels + ptr;
            smem[threadIdx.y + 1][corner_x + 1] = get_mean_3(img, location);
        }
        
        // Synchronize the block so that smem can be used safely.
        __syncthreads();

        // Apply filter without bank conflicts.
        int res_x = 0, res_y = 0;
        res_x -= smem[ind_y - 1][ind_x - 1];                // Top-Left
        res_y += smem[ind_y - 1][ind_x - 1]; 

        res_y += ((int) smem[ind_y - 1][ind_x]) << 1;       // Top-Middle
        
        res_x += smem[ind_y - 1][ind_x + 1];                // Top-Right
        res_y += smem[ind_y - 1][ind_x + 1];
        
        res_x -= ((int) smem[ind_y][ind_x - 1]) << 1;       // Middle-Left

        res_x += ((int) smem[ind_y][ind_x + 1]) << 1;       // Middle-Right

        res_x -= smem[ind_y + 1][ind_x - 1];                // Bottom-Left
        res_y -= smem[ind_y + 1][ind_x - 1];

        res_y -= ((int) smem[ind_y + 1][ind_x]) << 1;       // Bottom-Middle

        res_x += smem[ind_y + 1][ind_x + 1];                // Bottom-Right
        res_y -= smem[ind_y + 1][ind_x + 1];

        // Get the absolute value of the gradient components in a branchless fashion.
        int cond_x = res_x < 0;
        int cond_y = res_y < 0;
        res_x = (res_x ^ (-cond_x)) + cond_x; // Calculate the appropriate result using twos complement.
        res_y = (res_y ^ (-cond_y)) + cond_y;

        // Clamp the result into the [0, 255] range in a branchless fashion.
        int res = res_x + res_y;
        uint8_t set_all = res > 255;
        if (should_calc)
        {
            uint8_t writeback = (-set_all) | ((uint8_t) res);
            uint8_t check_t1 = writeback <= t1;     // Threshold the output in a branchless fashion.
            uint8_t check_t2 = writeback >= t2;
            writeback &= (check_t1 - 1);
            writeback |= (-check_t2);
            out[(y_base - 1 + threadIdx.y) * width + x_base + threadIdx.x] = writeback;
        }
        __syncthreads();
    }
}

// __global__ void sobel(uint8_t *out, uint8_t *img, int width, int height, int channels)
// {
//     const int stride = gridDim.x * BLOCK_SIZE;
//     const int img_size = width * height * channels;
//     extern __shared__ uint8_t cache_data[];
    
//     const int row_byte_offset = width * channels;
//     const int stopping_bound = img_size - row_byte_offset;
//     int load_base = blockIdx.x * BLOCK_SIZE + row_byte_offset;
//     int load_cache_stride = row_byte_offset + 2 * channels;
//     while (load_base < stopping_bound)
//     {
//         if (load_base + threadIdx.x < stopping_bound)
//         {
//             int ind = threadIdx.x + channels;
//             // Load data into shared memory for conv.
//             if (threadIdx.x < 3)       // Minimally divergent behaviour.
//             {       // Corner case: doesn't even deal with bank conflicts!
//                 cache_data[threadIdx.x] = img[load_base - channels + threadIdx.x - row_byte_offset];
//                 cache_data[BLOCK_SIZE + threadIdx.x + channels] =
//                         img[load_base + threadIdx.x + BLOCK_SIZE - row_byte_offset];
//             }
//             cache_data[ind] = img[load_base + threadIdx.x - row_byte_offset];
//             if (threadIdx.x < 3)       // Minimally divergent behaviour.
//             {       // Corner case: doesn't even deal with bank conflicts!
//                 cache_data[load_cache_stride + threadIdx.x] = img[load_base - channels + threadIdx.x];
//                 cache_data[load_cache_stride + BLOCK_SIZE + threadIdx.x + channels] =
//                         img[load_base + threadIdx.x + BLOCK_SIZE];
//             }
//             cache_data[load_cache_stride + ind] = img[load_base + threadIdx.x];
//             if (threadIdx.x < 3)       // Minimally divergent behaviour.
//             {       // Corner case: doesn't even deal with bank conflicts!
//                 cache_data[load_cache_stride * 2 + threadIdx.x] =
//                         img[load_base - channels + threadIdx.x + row_byte_offset];
//                 cache_data[load_cache_stride * 2 + BLOCK_SIZE + threadIdx.x + channels] =
//                         img[load_base + threadIdx.x + BLOCK_SIZE + row_byte_offset];
//             }
//             cache_data[load_cache_stride * 2 + ind] = img[load_base + threadIdx.x + row_byte_offset];

//             // Calculate the result. Has no bank conflicts!
//             int res_x = 0;
//             int res_y = (((int) cache_data[ind]) - cache_data[ind + load_cache_stride * 2]) << 1;
//             if (((load_base + threadIdx.x) / channels) % width != 0)
//             {
//                 res_y += ((int) cache_data[ind - channels]) - cache_data[ind + load_cache_stride * 2 - channels];
//                 res_x -= ((int) cache_data[ind + load_cache_stride - channels]) << 1;
//                 res_x -= ((int) cache_data[ind - channels]) + cache_data[ind + load_cache_stride - channels];
//             }
//             if (((load_base + threadIdx.x) / channels) % width != width - 1)
//             {
//                 res_y += ((int) cache_data[ind + channels]) - cache_data[ind + load_cache_stride * 2 + channels];
//                 res_x += ((int) cache_data[ind + load_cache_stride + channels]) << 1;
//                 res_x += ((int) cache_data[ind + channels]) + cache_data[ind + load_cache_stride * 2 + channels];
//             }
//             // Get the absolute value of the gradient components in a branchless fashion.
//             int cond_x = res_x < 0;
//             int cond_y = res_y < 0;
//             res_x = (res_x ^ cond_x) + cond_x;  // Calculate the appropriate result using twos complement.
//             res_y = (res_y ^ cond_y) + cond_y;

//             // Clamp the result into the [0, 255] range in a branchless fashion.
//             int res = res_x + res_y;
//             uint8_t set_all = res > 255;
//             out[load_base + threadIdx.x - row_byte_offset] = (-set_all) | ((uint8_t) res);
//         }
//         load_base += stride;
//     }
// }

int main(int argc, char *argv[])
{
    if (argc < 7) {
        cout << "Please specify the desired origin and destination (brightness and Sobel) paths." << endl;
        return 0;
    }
    cudaError_t cudaerr;
    string img_path;
    img_path = argv[1];
    // img_path = "./Test_Images/01_7680x4320.jpg";
    // cin >> img_path;
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

    uint8_t *sys_img;
    uint8_t *result;
    int img_size = input_img.rows * input_img.cols * input_img.channels();
    int padding = input_img.cols * input_img.channels();

    int padded_size = img_size + padding * (1 + input_img.channels());
    cudaMalloc(&sys_img, padded_size);
    cudaMemcpy(sys_img + padding, input_img.data, img_size, cudaMemcpyHostToDevice);

    auto t1 = chrono::high_resolution_clock::now();
    int grid_size = min((img_size + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRID_SIZE);
    change_brightness<<<grid_size, BLOCK_SIZE>>>(sys_img, input_img.cols,
                                input_img.rows + 1 + input_img.channels(), input_img.channels(), brigtness_change);
    
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        cerr << "Brightness change failed with error \"" << cudaGetErrorString(cudaerr) << "\"." << endl;
        return 1;
    }
    auto res = chrono::high_resolution_clock::now() - t1;
    
    cudaMemcpy(input_img.data, sys_img, img_size, cudaMemcpyDeviceToHost);

    cudaMemset(sys_img, 0, padding);
    cudaMemset(sys_img + padding + img_size, 0, padding * input_img.channels());
    cudaMalloc(&result, input_img.rows * input_img.cols);

    t1 = chrono::high_resolution_clock::now();
    grid_size = min((input_img.rows * input_img.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRID_SIZE);
    dim3 block_size;
    block_size.x = BLOCK_SIZE / TILE_HEIGHT;
    block_size.y = TILE_HEIGHT;
    tiled_sobel<<<grid_size, block_size>>>(result, sys_img,
                    input_img.cols, input_img.rows + 2, input_img.channels(), threshold_1, threshold_2);
    Mat output_img(input_img.rows, input_img.cols, CV_8UC1);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        cerr << "Sobel launch failed with error \"" << cudaGetErrorString(cudaerr) << "\"." << endl;
        return 1;
    }
    res += chrono::high_resolution_clock::now() - t1;
    cudaMemcpy(output_img.data, result, input_img.rows * input_img.cols, cudaMemcpyDeviceToHost);
    cudaFree(sys_img);
    cudaFree(result);

    long long microseconds = chrono::duration_cast<chrono::microseconds>(res).count();
    cout << "Execution Time: " << microseconds << " microseconds" << endl;

    imwrite(argv[2], input_img);
    // imwrite("./Result_Images/output_cuda_brightness.png", input_img);
    imwrite(argv[3], output_img);
    // imwrite("./Result_Images/output_cuda.png", output_img);
}
