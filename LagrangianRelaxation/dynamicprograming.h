#pragma once
#include <limits.h>
#include <algorithm>

#include "instance.h"

UpperSolution dynamicProgramming(const Instance & instance) {
	
	long subproblems_number = (long) pow(2, instance.n);
	
	std::vector<long> wT;
	std::vector<long> last_job;
	std::vector<long> pi;
	
	for (long k = 0; k <= subproblems_number; k++) wT.push_back(0);
	for (long k = 0; k <= subproblems_number; k++) last_job.push_back(0);
	for (long i = 0; i < instance.n; i++) pi.push_back(0);

	for (long k = 1; k < subproblems_number; k++) {
		long Psum = 0;
		for (long i = 0; i < instance.n; i++) {
			if ((1 << i) & k)
				Psum += instance.p[i];
		}
		
		long minwT = LONG_MAX;
		long job = -1;
		for (long i = 0; i < instance.n; i++) {
			if ((1 << i) & k) {
				long tmp = std::max(0L, Psum - instance.d[i]) * instance.w[i] + wT[k^(1<<i)];
				if (tmp < minwT) {
					minwT = tmp;
					job = i;
				}
			}
		}
		wT[k] = minwT;
		last_job[k] = job;
	}
	long idx = subproblems_number -1 ;
	long job = -1;
	long pos = instance.n - 1;
	for (long i = 0; i < instance.n; i++) {
		job = last_job[idx];
		pi[pos--] = job;
		idx = idx ^ (1 << job);
	}

	return UpperSolution(wT[subproblems_number-1], pi);
}