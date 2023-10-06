#pragma once
#include <iostream>
#include <algorithm>
#include <utility>
#include "instance.h"
UpperSolution greedy(const Instance& inst) {

    std::vector<long> pi;
    long best = -1;
    std::vector<std::pair<long, long>> Dpi;
    for (long i = 0; i < inst.n; ++i)
        Dpi.push_back(std::pair<double, long>((double)inst.w[i]/(double)inst.d[i], i));
    std::sort(Dpi.begin(), Dpi.end(),
        [](const std::pair<double, long>& a, const std::pair<double, long>& b) -> bool
        {
            return a.first < b.first;
        });
    pi.push_back(Dpi[0].second);
    for (long i = 1; i < inst.n ; ++i) {
        pi.insert(pi.begin(), Dpi[i].second);
        long localBest = evaluateSolution(inst, pi);
        long idxBest = 0;
        for (long k = 0; k < pi.size()-1; k++) {
            std::swap(pi[k], pi[k + 1]);
            long local = evaluateSolution(inst, pi);
            if (local <= localBest) {
                localBest = local;
                idxBest = k;
            }
        }
        std::swap(pi[idxBest], pi [pi.size() - 1]);
    }
    best = evaluateSolution(inst, pi);
    return UpperSolution(best, pi);
}
