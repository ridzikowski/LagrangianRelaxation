#pragma once
#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>
#include "utils.h"
#include "instance.h"
#include "lowerProblemExtensionFast.h"

void subgradientHarmonicExtensionFast(
    const Instance & inst, long m,
    double& WBest, std::vector<double> & v, std::vector<long> & S, std::vector<double> & trajectory, std::vector<long> & UB, std::vector<double> & alphas,
    bool isSqrt = false, long haltingLimit = 50
) {
    auto clock = startClock();
    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];

    std::vector<long> B;
    B.push_back(0);

    std::vector<long> tk;
    v.clear();
    long prev = 1;
    for (long k = 0; k <= m; k++) {
        v.push_back(0);
        long next = H * k / m;
        tk.push_back(next);
        if (k > 0) {
            for (long kk = prev; kk <= next; ++kk)
                B.push_back(k);
        }
        prev = next + 1;
    }

    std::vector<double> vBest;
    vBest = v;

    std::vector<double> h;
    for (long k = 0; k <= m; k++)
        h.push_back(0);

    S.clear();
    for (long i = 0; i < inst.n; i++)
        S.push_back(0);

    lowerProblemExtensionFast(inst, tk, v, WBest, h, S, B);
    long Wplus = getUBByHeuristic(inst, S);

    UB.clear();
    UB.push_back(Wplus);

    long it = 1;
    double C = 1.0;

    trajectory.clear();
    trajectory.push_back(WBest);

    double alpha = C / (isSqrt ? sqrt(it) : it);
    alphas.clear();
    alphas.push_back(alpha);

    while ((haltingLimit > 0 && it <= haltingLimit) || (haltingLimit < 0 && getClockMicro(clock) > -haltingLimit)) {

        for (long k = 1; k <= m; k++) {
            v[k] = v[k] + alpha * (h[k] - 1);
            if (v[k] < 0)
                v[k] = 0;
        }

        double W;
        lowerProblemExtensionFast(inst, tk, v, W, h, S, B);

        if (W > WBest) {
            WBest = W;
            vBest = v;
        }

        long Wt = getUBByHeuristic(inst, S);

        if (Wt < Wplus)
            Wplus = Wt;
        it++;
        alpha = C / (isSqrt ? sqrt(it) : it);

        trajectory.push_back(WBest);
        UB.push_back(Wplus);
        alphas.push_back(alpha);
    }
}

void subgradientGammaFormulaExtensionFast(
    const Instance & inst, long m,
    double& WBest, std::vector<double> & v, std::vector<long> & S, std::vector<double> & trajectory, std::vector<long> & UB, std::vector<double> & alphas, long haltingLimit = 50
) {
    auto clock = startClock();
    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];

    std::vector<long> B;
    B.push_back(0);

    std::vector<long> tk;
    v.clear();
    long prev = 1;
    for (long k = 0; k <= m; k++) {
        v.push_back(0);
        long next = H * k / m;
        tk.push_back(next);
        if (k > 0) {
            for (long kk = prev; kk <= next; ++kk)
                B.push_back(k);
        }
        prev = next + 1;
    }

    std::vector<double> vBest;        
    vBest = v;

    std::vector<double> h;
    for (long k = 0; k <= m; k++)
        h.push_back(0);

    S.clear();
    for (long i = 0; i < inst.n; i++)
        S.push_back(0);

    lowerProblemExtensionFast(inst, tk, v, WBest, h, S, B);
    long Wplus = getUBByHeuristic(inst, S);
    UB.clear();
    UB.push_back(Wplus);
    long it = 1;
    trajectory.clear();
    trajectory.push_back(WBest);
    double gamma = 2.0;
    double hSum = 0.0;
    for (long k = 1; k <= m; k++)
        hSum += (h[k] - 1) * (h[k] - 1);
    double alpha = gamma * (Wplus - WBest) / hSum;
    long critic = 0;
    alphas.clear();
    alphas.push_back(alpha);
    while ((haltingLimit > 0 && it <= haltingLimit) || (haltingLimit < 0 && getClockMicro(clock) < -haltingLimit)) {
        for (long k = 1; k <= m; k++) {
            v[k] = v[k] + alpha * (h[k] - 1);
            if (v[k] < 0)
                v[k] = 0;
        }

        double W;
        lowerProblemExtensionFast(inst, tk, v, W, h, S, B);
        long Wt = getUBByHeuristic(inst, S);
        if (Wt < Wplus)
            Wplus = Wt;
        critic++;
        if (W > WBest) {
            WBest = W;
            vBest = v;
            critic = 0;
        }
        hSum = 0.0;
        for (long k = 1; k <= m; k++)
            hSum += (h[k] - 1) * (h[k] - 1);
        alpha = gamma * (Wplus - W) / hSum;
        it++;
        if (critic == 5) {
            gamma *= 0.95;
            critic = 0;
        }
        trajectory.push_back(WBest);
        UB.push_back(Wplus);
        alphas.push_back(alpha);                
    }
}

void subgradientGammaRandomExtensionFast(
    const Instance & inst, long m,
    double& WBest, std::vector<double> & v, std::vector<long> & S, std::vector<double> & trajectory, std::vector<long> & UB, std::vector<double> & alphas,
    long seed, long haltingLimit = 50
) {
    auto clock = startClock();
    std::mt19937 mtg(seed);
    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];

    std::vector<long> B;
    B.push_back(0);

    std::vector<long> tk;
    v.clear();
    long prev = 1;
    for (long k = 0; k <= m; k++) {
        v.push_back(0);
        long next = H * k / m;
        tk.push_back(next);
        if (k > 0) {
            for (long kk = prev; kk <= next; ++kk)
                B.push_back(k);
        }
        prev = next + 1;
    }
    
    std::vector<double> vBest;        
    vBest = v;

    std::vector<double> h;
    for (long k = 0; k <= m; k++)
        h.push_back(0);

    S.clear();
    for (long i = 0; i < inst.n; i++)
        S.push_back(0);

    lowerProblemExtensionFast(inst, tk, v, WBest, h, S, B);
    long Wplus = getUBByHeuristic(inst, S);
    UB.clear();
    UB.push_back(Wplus);
    long it = 1;
    trajectory.clear();
    trajectory.push_back(WBest);
    double gamma = 2.0;
    double hSum = 0.0;
    for (long k = 1; k <= m; k++)
        hSum += (h[k] - 1) * (h[k] - 1);
    double alpha = gamma * (Wplus - WBest) / hSum;
    alphas.clear();
    alphas.push_back(alpha);
    while ((haltingLimit > 0 && it <= haltingLimit) || (haltingLimit < 0 && getClockMicro(clock) < -haltingLimit)) {
        for (long k = 1; k <= m; k++) {
            v[k] = v[k] + alpha * (h[k] - 1);
            if (v[k] < 0)
                v[k] = 0;
        }

        double W;
        lowerProblemExtensionFast(inst, tk, v, W, h, S, B);
        long Wt = getUBByHeuristic(inst, S);
        if (Wt < Wplus)
            Wplus = Wt;
        if (W > WBest) {
            WBest = W;
            vBest = v;
        }
        hSum = 0.0;
        for (long k = 1; k <= m; k++)
            hSum += (h[k] - 1) * (h[k] - 1);
        gamma = (mtg() % 11) / 100.0 + 0.95;
        alpha = gamma * (Wplus - W) / hSum;
        it++;
        trajectory.push_back(WBest);
        UB.push_back(Wplus);
        alphas.push_back(alpha);                
    }                        
}
