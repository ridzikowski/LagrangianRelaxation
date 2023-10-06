#pragma once
#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>
#include "utils.h"
#include "instance.h"
#include "lowerProblemBasic.h"

void subgradientHarmonic(
    const Instance & inst,
    double& WBest, std::vector<double> & u, std::vector<long> & S, std::vector<double> & trajectory, std::vector<long> & UB, std::vector<double> & alphas,
    bool isSqrt = false, long haltingLimit = 50
) {
    auto clock = startClock();
    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];
    u.clear();
    for (long t = 0; t <= H; t++)
        u.push_back(0);
    std::vector<double> uBest;        
    uBest = u;
    std::vector<long> g;
    for (long t = 0; t <= H; t++)
        g.push_back(0);
    S.clear();
    for (long i = 0; i < inst.n; i++)
        S.push_back(0);
    lowerProblemBasic(inst, u, WBest, g, S);
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
    while ((haltingLimit > 0 && it <= haltingLimit) || (haltingLimit < 0 && getClockMicro(clock) < -haltingLimit)) {
        for (long t = 1; t <= H; t++) {
            u[t] = u[t] + alpha * (g[t] - 1);
            if (u[t] < 0)
                u[t] = 0;
        }

        double W;
        lowerProblemBasic(inst, u, W, g, S);
        if (W > WBest) {
            WBest = W;
            uBest = u;
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

void subgradientGammaFormula(
    const Instance & inst,
    double& WBest, std::vector<double> & u, std::vector<long> & S, std::vector<double> & trajectory, std::vector<long> & UB, std::vector<double> & alphas,
    long haltingLimit = 50
) {
    auto clock = startClock();
    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];
    u.clear();
    for (long t = 0; t <= H; t++)
        u.push_back(0);
    std::vector<double> uBest;        
    uBest = u;
    std::vector<long> g;
    for (long t = 0; t <= H; t++)
        g.push_back(0);
    S.clear();
    for (long i = 0; i < inst.n; i++)
        S.push_back(0);
    lowerProblemBasic(inst, u, WBest, g, S);
    long Wplus = getUBByHeuristic(inst, S);
    UB.clear();
    UB.push_back(Wplus);
    long it = 1;
    trajectory.clear();
    trajectory.push_back(WBest);
    double gamma = 2.0;
    double gSum = 0.0;
    for (long t = 1; t <= H; t++)
        gSum += (g[t] - 1) * (g[t] - 1);
    double alpha = gamma * (Wplus - WBest) / gSum;
    long critic = 0;
    alphas.clear();
    alphas.push_back(alpha);
    while ((haltingLimit > 0 && it <= haltingLimit) || (haltingLimit < 0 && getClockMicro(clock) < -haltingLimit)) {
        for (long t = 1; t <= H; t++) {
            u[t] = u[t] + alpha * (g[t] - 1);
            if (u[t] < 0)
                u[t] = 0;
        }

        double W;
        lowerProblemBasic(inst, u, W, g, S);
        long Wt = getUBByHeuristic(inst, S);
        if (Wt < Wplus)
            Wplus = Wt;
        critic++;
        if (W > WBest) {
            WBest = W;
            uBest = u;
            critic = 0;
        }
        gSum = 0.0;
        for (long t = 1; t <= H; t++)
            gSum += (g[t] - 1) * (g[t] - 1);
        alpha = gamma * (Wplus - W) / gSum;
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

void subgradientGammaRandom(
    const Instance & inst,
    double& WBest, std::vector<double> & u, std::vector<long> & S, std::vector<double> & trajectory, std::vector<long> & UB, std::vector<double> & alphas,
    long seed, long haltingLimit = 50
) {
    auto clock = startClock();
    std::mt19937 mtg(seed);
    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];
    u.clear();
    for (long t = 0; t <= H; t++)
        u.push_back(0);
    std::vector<double> uBest;        
    uBest = u;
    std::vector<long> g;
    for (long t = 0; t <= H; t++)
        g.push_back(0);
    S.clear();
    for (long i = 0; i < inst.n; i++)
        S.push_back(0);
    lowerProblemBasic(inst, u, WBest, g, S);
    long Wplus = getUBByHeuristic(inst, S);
    UB.clear();
    UB.push_back(Wplus);
    long it = 1;
    trajectory.clear();
    trajectory.push_back(WBest);
    double gamma = 2.0;
    double gSum = 0.0;
    for (long t = 1; t <= H; t++)
        gSum += (g[t] - 1) * (g[t] - 1);
    double alpha = gamma * (Wplus - WBest) / gSum;
    alphas.clear();
    alphas.push_back(alpha);
    while ((haltingLimit > 0 && it <= haltingLimit) || (haltingLimit < 0 && getClockMicro(clock) < -haltingLimit)) {
        for (long t = 1; t <= H; t++) {
            u[t] = u[t] + alpha * (g[t] - 1);
            if (u[t] < 0)
                u[t] = 0;
        }

        double W;
        lowerProblemBasic(inst, u, W, g, S);
        long Wt = getUBByHeuristic(inst, S);
        if (Wt < Wplus)
            Wplus = Wt;
        if (W > WBest) {
            WBest = W;
            uBest = u;
        }
        gSum = 0.0;
        for (long t = 1; t <= H; t++)
            gSum += (g[t] - 1) * (g[t] - 1);
        gamma = (mtg() % 11) / 100.0 + 0.95;
        alpha = gamma * (Wplus - W) / gSum;
        it++;
        trajectory.push_back(WBest);
        UB.push_back(Wplus);
        alphas.push_back(alpha);                
    }                        
}

void subgradientGammaFormulaScaled(long b, const Instance & inst,
    double& WBest, std::vector<double> & u, std::vector<long> & S, std::vector<double> & trajectory, std::vector<long> & UB, std::vector<double> & alphas,
    long haltingLimit = 50
) {
    Instance * inst_ = inst.scaleArbitrary(b);
    long H_ = 0;
    for (long i = 0; i < inst_->n; i++)
        H_ += inst_->p[i];

    double WBest_;
    std::vector<double> u_;
    for (long t = 0; t <= H_; t++)
        u_.push_back(0);
    std::vector<long> S_;
    for (long i = 0; i < inst_->n; i++)
        S_.push_back(0);            

    subgradientGammaFormula(*inst_, WBest_, u_, S_, trajectory, UB, alphas, haltingLimit);

    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];
    u.clear();
    for (long t = 0; t <= H; t++)
        u.push_back(0);
    for (long k = 1; k <= H_; ++k) {
        for (long t = (k - 1) * b; t <= k * b; ++t) {
            u[t] = u_[k];
        }
    }

    std::vector<long> g;
    for (long t = 0; t <= H; t++)
        g.push_back(0);
    S.clear();
    for (long i = 0; i < inst.n; i++)
        S.push_back(0);

    lowerProblemBasic(inst, u, WBest, g, S);
    UB.push_back(getUBByHeuristic(inst, S));        
}
