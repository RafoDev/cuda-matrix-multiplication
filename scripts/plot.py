import matplotlib.pyplot as plt

sizes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
miliseconds = [0.034816, 0.01328, 0.021504, 0.168384, 0.669312, 4.70298, 40.7422, 304.995, 2297.64, 18481]
miliseconds_shared = [0.013088, 0.013248, 0.018464, 0.148448, 0.56928, 3.57853, 30.3747, 196.14, 1664.62, 13357.6]

plt.plot(sizes, miliseconds, label='matrixMulKernel')
plt.plot(sizes, miliseconds_shared, label='matrixMulKernelShared')

plt.xlabel('Tamaño de la matriz (2^2n)')
plt.ylabel('Tiempo (ms)')
plt.title('Tiempo de ejecución')
plt.legend()

plt.show()