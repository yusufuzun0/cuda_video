#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <algorithm>

__device__ unsigned char my_min(unsigned char a, unsigned char b)
{
	return a < b ? a : b;
}

__global__ void ornekKernel(uchar3* src, uchar3* dst, int cols, int rows, const int factor)
{
	int x = blockIdx.x * (blockDim.x + threadIdx.x);
	int y = blockIdx.y * (blockDim.y + threadIdx.y);

	if (x < cols && y < rows)
	{
		int index = y * cols + x;

		dst[index].x = my_min(src[index].x * factor, (unsigned char)255);
		dst[index].y = my_min(src[index].y * factor, (unsigned char)255);
		dst[index].z = my_min(src[index].z * factor, (unsigned char)255);
		
	}
}



int main()
{
	cv::VideoCapture cap(0);

	if (!cap.isOpened())
	{
		std::cerr << "Kamera acilmadi" << std::endl;
		return -1;
	}

	while (true)
	{
		cv::Mat frame;
		cap >> frame;

		cudaError_t err;
		uchar3* devSrc;
		uchar3* devDst;


		int cols = frame.cols;
		int rows = frame.rows;


		err = cudaMalloc((void**)&devSrc, cols * rows * sizeof(uchar3));

		if (err != cudaSuccess)
		{
			std::cerr << "Hata: " << cudaGetErrorString(err) << std::endl;
			break;
		}


		err = cudaMalloc((void**)&devDst, cols * rows * sizeof(uchar3));

		if (err != cudaSuccess)
		{
			std::cerr << "Hata: " << cudaGetErrorString(err) << std::endl;
			break;
		}

		cudaMemcpy(devSrc, frame.data, cols * rows * sizeof(uchar3), cudaMemcpyHostToDevice);

		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,(rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

		ornekKernel <<< numBlocks, threadsPerBlock >> > (devSrc, devDst, cols, rows, 10);

		cv::Mat output(rows, cols, CV_8UC3);

		cudaMemcpy(output.data, devDst, cols * rows * sizeof(uchar3), cudaMemcpyDeviceToHost);

		cv::imshow("Görüntü", output);

		cudaFree(devSrc);
		cudaFree(devDst);


		char k = cv::waitKey(30);
		if (k == 27) // ASCII kodu 27 => ESC tuþu
		{
			break;
		}

	}
		return 0;

}