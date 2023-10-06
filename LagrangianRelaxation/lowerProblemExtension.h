#pragma once
#include <iostream>
#include <algorithm>
#include <utility>
#include "utils.h"
#include "instance.h"

void lowerProblemExtension(const Instance& inst, std::vector<long>& tk, std::vector<double>& v, double& W, std::vector<double>& h, std::vector<long>& S) {
    W = 0.0;
    long m = v.size() - 1;

    for (long k = 1; k <= m; k++)
        h[k] = 0;

    for (long i = 0; i < inst.n; ++i) {
        bool first = true;
        double Mi = 0;
        long Si;
        for (long j = -1; j <= m; j++) { 
            long t;

            if (j == -1)
                t = inst.d[i];
            else
                t = tk[j];

            if (t + inst.p[i] <= tk[m]) {

                double tempM = std::max(0L, t + inst.p[i] - inst.d[i]) * inst.w[i];

                for (long k = 1; k <= m; ++k) {

                    double hki = lengthOfIntervalIntersection(t, t + inst.p[i], tk[k - 1], tk[k]) / ((double) (tk[k] - tk[k - 1]));
                    
                    tempM += v[k] * hki;
                }

                if (first || tempM < Mi) {
                    first = false;
                    Mi = tempM;
                    Si = t;
                }
            }

            if (j == -1)
                t = inst.d[i] - inst.p[i];
            else
                t = tk[j] - inst.p[i];

            if (t >= 0) {
                double tempM = std::max(0L, t + inst.p[i] - inst.d[i]) * inst.w[i];

                for (long k = 1; k <= m; ++k) {
                    double hki = lengthOfIntervalIntersection(t, t + inst.p[i], tk[k - 1], tk[k]) / ((double) (tk[k] - tk[k - 1]));
                    tempM += v[k] * hki;
                }

                if (first || tempM < Mi) {
                    first = false;
                    Mi = tempM;
                    Si = t;
                }
            }
        }

        W += Mi;
        S[i] = Si;

        for (long k = 1; k <= m; ++k)
            h[k] += lengthOfIntervalIntersection(Si, Si + inst.p[i], tk[k - 1], tk[k]) / ((double) (tk[k] - tk[k - 1]));
    }

    double V = 0;
    for (long k = 1; k <= m; ++k) {
        V += v[k];
    }

    W -= V;
}
