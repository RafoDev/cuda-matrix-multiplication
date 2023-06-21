# CUDA Matrix Multiplication

Se utiliza CUDA C/C++ para implementar dos kernels de multiplicación de matrices:

- Multiplicación de matrices (convencional)
- Multiplicación de matrices + Memoria compartida

## Input

El input para ambos kernels es una matríz generada aleatoriamente, se utiliza la función `generateRandomMatrix`: 

```c++
// include/utils.hpp

void generateRandomMatrix(vector<float> &matrix, int N)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dis(0.0, 9.0);
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			matrix[i * N + j] = round(dis(gen));
}
```

Esta función genera numeros de coma flotante del 0 al 9 y se redondean. Esto se hizo para verificar las matrices resultantes, sin embargo, puede ser alterado para emitir flotantes no redondeados y en cualquier rango.

## Output

El output de los kernels es una matríz producto de la multiplicación, se puede imprimir añadiéndo el argumento `-DEBUG` al ejecutar.

```bash
matrixMulKernel 4 -DEBUG
```

## Multiplicación de matrices (convencional)

Este kernel realiza una multiplicación de matrices en paralelo.

```c++
__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width)
{
	// Se calculan las coordenadas globales de cada hilo en la matríz P
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
	// Se verifica que el hilo esté dentro de los límites válidos de la matriz P
    if ((Row < Width) && (Col < Width))
    {
		// Pvalue es una variable temporal para almacenar el resultado parcial de la multiplicación realizada por cada hilo
        float Pvalue = 0;
        // Itera a través del width de las matrices para realizar la multiplicación 
		for (int k = 0; k < Width; ++k)
        {
			// Se realiza la multiplicación de matrices y se acumulan los resultados
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }
		// Se actualiza la matríz de salida P con el resultado parcial calculado por el hijo
        P[Row * Width + Col] += Pvalue;
    }
}
```

## Multiplicación de matrices + Memoria compartida

Este kernel realiza una multiplicación de matrices en paralelo utilizando la memoria compartida de cada bloque.

```c++
__global__ void MatrixMulKernelShared(float *M, float *N, float *P, int width)
{
	// Se definen dos matrices en memoria compartida del tamaño del bloque (TILE_WIDTH)
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	// Obtenemos los índices de bloque y de hilo
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	// Se calculan las coordenadas globales de la matriz P correspondientes al hilo actual
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

	// Pvalue es una variable temporal para almacenar el resultado parcial de la multiplicación realizada por cada hilo

    float Pvalue = 0;

	// Se inicia un bucle para dividir la matríz en bloques más pequeños
    for (int ph = 0; ph < width / TILE_WIDTH; ph++)
    {
		// Se cargan los bloques de las matriz M y N en memoria compartida Mds Nds
        Mds[ty][tx] = M[Row * width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + Col];

		// Se sincronizan los hilos del bloque antes de continuar, caso contrario no habrían datos para multiplicar
        __syncthreads();

		// Bucle para realizar la multiplicación de matrices en bloques cargados en memoria compartida
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
			// Se realiza la multiplicación de matrices y se acumulan los resultados
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }	
		// Se sincronizan los hilos del bloque para asegurar que todos los hilos hayan terminado de multiplicar antes de iterar nuevamente
        __syncthreads();
    }
    P[Row * width + Col] = Pvalue;
}
```

### Lanzamiento

El lanzamiento debe configurado de esta manera:

```c++ 
int blockDim = TILE_WIDTH;
int gridDim = ceil(width / blockDim);

dim3 dimGrid(gridDim, gridDim, 1);
dim3 dimBlock(blockDim, blockDim, 1);

MatrixMulKernelShared<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
```

El motivo de esta configuración es asegurar que todos los elementos de la matríz P sean procesados correctamente. 

## Comparación de rendimiento

Para hacer la comprobación, al compilar el proyecto se copiará un script de bash llamado `test.sh` el cual ejecutará varias veces los programas con distintos inputs. De esa manera se obtienen los tiempos en milisegundos de ambas implementaciones para todos los casos de prueba.

<img src="https://i.postimg.cc/HncLV2bP/mat-cuda.png" alt="Original">


Ejemplo del desenfoque aplicado:

<div style="display: flex;">


<img src="https://i.postimg.cc/9FLW0Z2z/blurred-racoon.png" alt="Blurred" style="width: 300px; height:300px;">

</div>



## Compilación y uso

Para clonar el proyecto:

```bash
git clone https://github.com/RafoDev/cuda-filters.git
```
Para compilarlo:

```bash
cmake -B build
cd build
make
```
Se obtendrán tres binarios:

- getDeviceProps
- matrixMulKernel
- matrixMulKernelShared

Para utilizarlos:

```bash
./getDeviceProps
./matrixMulKernel 128 -DEBUG
./matrixMulKernelShared 128 
```

