#ifndef UTILS_H
#define UTILS_H

/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * General utility functions for generating random numbers.
 */

#define DEBUG false
#define TEST false

#define toDigit(c) c - '0'

#include <iostream>
#include <errno.h>
#include <random>
#include <time.h>
#include <vector>
#include <iterator>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>

#include "../time/Time.h"

#define PI 3.14159265

#define TESTB(v,p) ((v) & (1<<(p)))
#define SETB(X) (1 << X)

#define MIN(A,B) (A < B ? A : B)
#define MAX(A,B) (A > B ? A : B)

#define BOOL(B) (B == 0 ? "false" : "true")

#define FOLDER "data/"

namespace vz{
	static void pause(){ std::cout << "\nPress enter to continue ..." << std::endl; while (getchar() != '\n'); }
	static void error(std::string error){
		std::cout << "RunTime Error Encountered:("<<error<<")\nPress enter to continue ..." << std::endl; while (getchar() != '\n');
		exit(1);
	}
}

template<class T>
class Utils{
public:
	Utils(){
		srand(time(0)*rand());
		this->seed = PI*(rand() % INT_MAX);
		this->generator.seed(this->seed);
	};
	~Utils(){};

	//Random Number Generators//
	T uni(T max);
	T uni(T min, T max);
	T rnum(unsigned int low, unsigned int high);
	void setSeed(unsigned int);
	void randDataToFile(unsigned int d, unsigned int n, unsigned int max);
	void randDataToFile(std::string file, unsigned int d, unsigned int n, unsigned int max);
	void randDataToFile(std::string file, unsigned int d, unsigned int n, unsigned int min,unsigned int max);
	void shuffleArray(T *&arr, unsigned int n);

	//String Tokenize//
	std::vector<std::string> split(std::string str, std::string delimiter);
	void print_array(const T *arr, unsigned int limit);

protected:
	unsigned int seed;
	std::string delim = ",";	
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution{ 0.0, 1.0 };
};

/*

Method and Construstor implementations

*/

static uint64_t cells(arr2D arr){
	return arr.first*arr.second;
}

static void print(arr2D arr){
	std::cout << "("<<arr.first<<","<<arr.second<<")" << std::endl;
}

/***********************/
/*	  Utils Class	   */
/***********************/

template<class T>
T Utils<T>::uni(T max){
	return this->distribution(generator)*max;
}

template<class T>
T Utils<T>::uni(T min, T max){
	return this->distribution(generator)*max + min;
}

template<class T>
T Utils<T>::rnum(unsigned int low, unsigned high){
	srand(time(0)*rand()); return PI*(rand() % high + low);
}

template<class T>
void Utils<T>::setSeed(unsigned int seed){
	this->seed = seed;
	this->generator.seed(this->seed);
}

template<class T>
void Utils<T>::shuffleArray(T *&arr,unsigned int n){
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(arr, (arr+n), std::default_random_engine(seed));
}

template<class T>
void Utils<T>::print_array(const T *arr,unsigned int limit){
	for(int i =0;i<limit;i++) std::cout<<arr[i] << " ";
	std::cout<<std::endl;
}

#endif
