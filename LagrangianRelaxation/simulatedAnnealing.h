#pragma once
#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>
#include "utils.h"
#include "instance.h"
#include "lowerProblemBasic.h"

void simulatedAnnealing(
    const Instance & inst,
    double& WBest, std::vector<double> & u, std::vector<long> & S, std::vector<double> & trajectory, std::vector<long> & UB, std::vector<double> & alphas,
    long seed, double coolingRate = 0.99, double startingTemperature = 1000.0, long epochSize = 5, long haltingLimit = 50, double A = 2.0
) {
    auto clock = startClock();
    std::mt19937 mtg(seed);
    double T = startingTemperature;
    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];
    u.clear();
    for (long t = 0; t <= H; t++)
        u.push_back(0);
    std::vector<long> g;
    for (long t = 0; t <= H; t++)
        g.push_back(0);
    S.clear();
    for (long i = 0; i < inst.n; i++)
        S.push_back(0);
    double W;
    lowerProblemBasic(inst, u, W, g, S);
    long Wplus = getUBByHeuristic(inst, S);
    UB.clear();
    UB.push_back(Wplus);    
    WBest = W;
    trajectory.clear();
    trajectory.push_back(WBest);
    long it = 1;
    std::vector<double> diff;
    for (long t = 0; t <= H; t++)
        diff.push_back(0);    
    while (T > 0 && (haltingLimit > 0 && it <= haltingLimit) || (haltingLimit < 0 && getClockMicro(clock) < -haltingLimit)) {
        double A = 2.0;
        auto oldU = u;
        double diff2 = (mtg() % 1000001) / 1000000.0 * A;
        for (long t = 1; t <= H; t++) {
            double diff = (mtg() % 1000001) / 1000000.0 * A;
            u[t] += diff2 * (g[t] - 1);
            if (u[t] < 0)
                u[t] = 0;
        }
        double Wk;
        lowerProblemBasic(inst, u, Wk, g, S);
        long Wt = getUBByHeuristic(inst, S);
        if (Wt < Wplus)
            Wplus = Wt;        
        if (Wk < W && (mtg() % 1000001) / 1000000.0 > exp((W - Wk) / T)) {
            Wk = W;
            u = oldU;
        }
        W = Wk;
        WBest = std::max(W, WBest);
        if (it % epochSize == 0)
            T *= coolingRate;
        it++;
        trajectory.push_back(WBest);
        UB.push_back(Wplus);
        alphas.push_back(T);        
    }                        
}
