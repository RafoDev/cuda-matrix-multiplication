#include <vector>
#include <iostream>
#include <random>

using namespace std;

void generateRandomMatrix(vector<float> &matrix, int N)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dis(0.0, 9.0);
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			matrix[i * N + j] = round(dis(gen));
}

void printMatrix(vector<float> &matrix, int N)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			cout << matrix[i * N + j] << " ";
		cout << '\n';
	}
}