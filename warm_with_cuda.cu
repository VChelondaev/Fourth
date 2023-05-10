#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include <cub/cub.cuh>


__global__
void calculateMatrix(double* arr, double* arr_new, size_t size)
{
        //вычисляем индекс элемента
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i * size + j > size * size) return;
    
    if(!((j == 0 || i == 0 || j == size - 1 || i == size - 1))){        //Проверка на границы массива
                    int n = i * N + j;
                    arr_new[n] = 0.25 * (arr[n - 1] + arr[n + 1] + arr[(i - 1) * N + j] + arr[(i + 1) * N + j]);
                    }
}


// Вычисляем матрицу ошибок
__global__
void getErrorMatrix(double* arr, double* arr_new, double* output_arr, size_t size) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;     
	
	if (idx > size * size) return;
	
	output_arr[idx] = std::abs(arr_new[idx] - arr[idx]);

}


int main(int argc, char** argv) {

	double accuracy = 0.000001;
	int N, ITER_MAX;
	accuracy = atof(argv[1]);
	N = atoi(argv[2]);
	ITER_MAX = atoi(argv[3]);

	double* arr = (double*)calloc(N * N, sizeof(double));
	double* arr_new = (double*)calloc(N * N, sizeof(double));
	double step = 10.0 / (N-1);
	

	arr[0] = 10;
	arr[N-1] = 20;
	arr[N * (N - 1)] = 20;
	arr[N * N -1] = 30;


	for (int i = 1; i < N; i++) {
		arr[i] = arr[0] + step * i;
		arr[N * (N - 1) + i] = arr[N - 1] + step * i;
		arr[(N * i)] = arr[0] + step * i;
		arr[N - 1 + i * N] = arr[N - 1] + step * i;
	}

	memcpy(arr_new, arr, N * N * sizeof(double));

    cudaSetDevice(3);   //Выбор 3 девайса

        double* device_arr_Ptr, *device_arr_new_Ptr, *device_Error, *error_arr, *temp_Storage = NULL;
	size_t temp_Storage_Size = 0;
        int size = N * N;

	cudaMalloc((void**)(&device_arr_Ptr), sizeof(double) * size);    //  Выделил память на видеокарте
	cudaMalloc((void**)(&device_arr_new_Ptr), sizeof(double) * size);
	cudaMalloc((void**)&device_Error, sizeof(double));                                       
	cudaMalloc((void**)&error_arr, sizeof(double) * size);        //Выделение памяти под матрицу ошибок 
	
	cudaMemcpy(device_arr_Ptr, arr, sizeof(double) * size, cudaMemcpyHostToDevice);       //Скопировал матрицы на видеокарту
	cudaMemcpy(device_arr_new_Ptr, arr_new, sizeof(double) * size, cudaMemcpyHostToDevice);

	// Для работы функции редукции, ей необходимо выделить память. При  первом запуске функция возвращает количество памяти, которое ей надо выделить.
	// Для этого так же нужно, чтобы указатель на массив был NULL
	cub::DeviceReduce::Max(temp_Storage, temp_Storage_Size, error_arr, device_Error, size);
	cudaMalloc((void**)&temp_Storage, temp_Storage_Size);

	int iter = 0;
	double error = 1.0;
	
	bool isGraphCreated = false;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	
	size_t threads = std::min(N, 1024);
	
	unsigned int blocks = size / threads;

	dim3 blockDim(threads / 16, threads / 16);
    	dim3 gridDim(blocks * 16, blocks * 16);

	clock_t start = clock();

	std::cout << N << " " << accuracy <<  " " << std::endl;
	for (; ((iter < ITER_MAX) && (error > accuracy)); iter++) {	
		if (isGraphCreated) {
			cudaGraphLaunch(instance, stream);    //Запуск графа
			
			cudaMemcpy(error, deviceError, sizeof(double), cudaMemcpyDeviceToHost, stream);    //копирует ошибку на хост

			cudaStreamSynchronize(stream);    //Ждём, когда закончится выполнение потока stream

			iter += 100;
		}
		else {
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);   //Начинаем захватывать граф
			for(size_t i = 0; i < 50; i++)
			{
				calculateMatrix<<<gridDim, blockDim, 0, stream>>>(device_arr_Ptr, device_arr_new_Ptr, N);   //Запускаем ядро для рассчета функций
				calculateMatrix<<<gridDim, blockDim, 0, stream>>>(device_arr_new_Ptr, device_arr_Ptr, N);
			}
			// Расчитываем ошибку каждую сотую итерацию
			getErrorMatrix<<<threads * blocks * blocks, threads,  0, stream>>>(device_arr_Ptr, device_arr_new_Ptr, error_arr, N);
			cub::DeviceReduce::Max(temp_Storage, temp_Storage_Size, error_arr, device_Error, size);   //находит максимальную ошибку
	
			cudaStreamEndCapture(stream, &graph);   //Заканчиваем захватывать граф
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);  //Инициализация графа
			isGraphCreated = true;
  		}
	}

	clock_t end = clock();

	std::cout << "Time: " << 1.0 * (end - start) / CLOCKS_PER_SEC << std::endl;
	std::cout << "Iter: " << iter << " Error: " << error << std::endl;

	cudaFree(device_arr_Ptr);
	cudaFree(device_arr_new_Ptr);
	cudaFree(error_arr);
	cudaFree(temp_Storage);

	delete[] arr;
	delete[] arr_new;

	return 0;
}
