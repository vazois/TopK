#ifndef TIME_H
#define TIME_H

/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * Utility functions for measuring execution time.
 */

#include<ratio>
#include<ctime>
#include<chrono>
#include <iostream>
#include<string>

#include<ratio>
#include<climits>
#include<chrono>
#include<ctime>

typedef std::pair<uint64_t, uint64_t> arr2D;

typedef std::chrono::duration<double, std::ratio<1, 1000000000>> nsecs;
typedef std::chrono::duration<double, std::ratio<1,1000>> msecs;
typedef std::chrono::duration<double, std::ratio<1,1> > secs;
typedef std::chrono::duration<double, std::ratio<60, 1> > mins;
typedef std::chrono::duration<double, std::ratio<3600, 1> > hours;
typedef std::chrono::duration<double, std::ratio<60 * 60 * 24, 1> > days;
typedef std::chrono::duration<double, std::ratio<60 * 60 * 24 * 30, 1> > months;

template<class M>
class Time{
public:
	Time(){
		this->lap_rdtsc = 0;
	};
	~Time(){};
	void start();
	void reset();
	double lap(std::string);
	double lap();

	void rdtsc_start();
	uint64_t rdtsc_stop();
	uint64_t rdtsc_lap();
	void rdtsc_clear(){ this->lap_rdtsc = 0; }
	void rdtsc_print(){ std::cout << "\n { rdtsc }:" << this->lap_rdtsc << std::endl; }

private:
	std::chrono::high_resolution_clock::time_point tp1;
	std::chrono::high_resolution_clock::time_point tp2;
	uint64_t start_rdtsc;
	uint64_t end_rdtsc;
	uint64_t lap_rdtsc;
};

template<class M>
void Time<M>::start(){
	this->tp1 = std::chrono::high_resolution_clock::now();
}

template<class M>
void Time<M>::reset(){
	this->tp1 = std::chrono::high_resolution_clock::now();
}

template<class M>
double Time<M>::lap(){
	return this->lap("");
}

template<class M>
double Time<M>::lap(std::string comment){
	this->tp2 = std::chrono::high_resolution_clock::now();
	//M time_span = std::chrono::duration_cast<M>(this->tp2 - this->tp1);
	M time_span = this->tp2 - this->tp1;

	double tt = time_span.count();
	if ( comment != "" ) std::cout << "Elapsed Time ( " << comment << " ): " << tt << std::endl;
	this->tp1 = std::chrono::high_resolution_clock::now();

	return tt;
}

template<class M>
void Time<M>::rdtsc_start()
{
	uint32_t hi, lo;

	__asm volatile
	    ("rdtsc" : "=a" (lo), "=d" (hi));

	this->start_rdtsc = ((uint64_t)hi << 32) | lo;
}

template<class M>
uint64_t Time<M>::rdtsc_stop()
{
	uint32_t hi, lo;

	__asm volatile
	    ("rdtsc" : "=a" (lo), "=d" (hi));

	this->end_rdtsc = ((uint64_t)hi << 32) | lo;
	return this->end_rdtsc - this->start_rdtsc;
}

template<class M>
uint64_t Time<M>::rdtsc_lap()
{
	uint32_t hi, lo;

	__asm volatile
	    ("rdtsc" : "=a" (lo), "=d" (hi));

	this->end_rdtsc = ((uint64_t)hi << 32) | lo;
	this->lap_rdtsc += this->end_rdtsc - this->start_rdtsc;
	return this->end_rdtsc - this->start_rdtsc;
}

#ifdef _WIN32
#include <Windows.h>
static double get_wall_time(){
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)){
		//  Handle error
		return 0;
	}
	if (!QueryPerformanceCounter(&time)){
		//  Handle error
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}
static double get_cpu_time(){
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0){
		//  Returns total user time.
		//  Can be tweaked to include kernel times as well.
		return
			(double)(d.dwLowDateTime |
			((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else{
		//  Handle error
		return 0;
	}
}

//  Posix/Linux
#else
#include <sys/time.h>
static double get_wall_time(){
	struct timeval time;
	if (gettimeofday(&time, NULL)){
		//  Handle error
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
static double get_cpu_time(){
	return (double)clock() / CLOCKS_PER_SEC;
}
#endif



#endif
