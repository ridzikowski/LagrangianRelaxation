#pragma once
#include <iostream>
#include <algorithm>
#include <utility>
#include "instance.h"

void lowerProblemBasic(const Instance& inst, std::vector<double>& u, double& W, std::vector<long>& g, std::vector<long>& S) {
    long H = 0;
    for (long i = 0; i < inst.n; i++)
        H += inst.p[i];
    W = 0;
    std::vector<double> U;
    U.push_back(0);
    for (long t = 1; t <= H; ++t)
        U.push_back(U[t - 1] + u[t]);

    for (long t = 1; t <= H; ++t)
        g[t] = 0;

    for (long i = 0; i < inst.n; ++i) {
        long Si = 0;
        double Vi = std::max(0L, inst.p[i] - inst.d[i]) * inst.w[i] + U[inst.p[i]] - U[0];
        for (long t = 1; t <= H - inst.p[i]; ++t)
        {
            double tmpV = std::max(0L, t + inst.p[i] - inst.d[i]) * inst.w[i] + U[t + inst.p[i]] - U[t];
            if (tmpV < Vi) {
                Vi = tmpV;
                Si = t;
            }
        }
        for (long t = Si + 1; t <= Si + inst.p[i]; t++)
            g[t]++;
        W += Vi;
        S[i] = Si;
    }

    W -= U[H];
}
