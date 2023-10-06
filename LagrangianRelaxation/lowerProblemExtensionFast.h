#pragma once
#include <iostream>
#include <algorithm>
#include <utility>
#include "utils.h"
#include "instance.h"

void lowerProblemExtensionFast(const Instance& inst, std::vector<long>& tk, std::vector<double>& v, double& W, std::vector<double>& h, std::vector<long>& S, std::vector<long> & B) {
    W = 0.0;
    long m = v.size() - 1;

    for (long k = 1; k <= m; k++)
        h[k] = 0;

    std::vector<std::vector<double>> A;
    for (long x = 0; x <= m; ++x) {
        A.push_back(std::vector<double>());
        double sum = 0;
        for (long y = 0; y <= m; ++y) {
            A[x].push_back(0.0);
            if (y < x || x == 0) continue;
            sum += v[y] * 1;
            A[x][y] = sum;
        }
    }
  
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

                long startBucket = B[t + 1];
                long endBucket = B[t + inst.p[i]];

                if (endBucket == startBucket) {
                    double hki = lengthOfIntervalIntersection(t, t + inst.p[i], tk[startBucket - 1], tk[startBucket]) / ((double) (tk[startBucket] - tk[startBucket - 1]));
                    tempM += v[startBucket] * hki;
                }
                else if (endBucket == startBucket + 1) {
                    double hkiStart = lengthOfIntervalIntersection(t, t + inst.p[i], tk[startBucket - 1], tk[startBucket]) / ((double) (tk[startBucket] - tk[startBucket - 1]));
                    double hkiEnd = lengthOfIntervalIntersection(t, t + inst.p[i], tk[endBucket - 1], tk[endBucket]) / ((double) (tk[endBucket] - tk[endBucket - 1]));
                    tempM += v[startBucket] * hkiStart;
                    tempM += v[endBucket] * hkiEnd;
                }
                else {
                    double hkiStart = lengthOfIntervalIntersection(t, t + inst.p[i], tk[startBucket - 1], tk[startBucket]) / ((double) (tk[startBucket] - tk[startBucket - 1]));
                    double hkiEnd = lengthOfIntervalIntersection(t, t + inst.p[i], tk[endBucket - 1], tk[endBucket]) / ((double) (tk[endBucket] - tk[endBucket - 1]));
                    tempM += v[startBucket] * hkiStart;
                    tempM += A[startBucket + 1][endBucket - 1];
                    tempM += v[endBucket] * hkiEnd;
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

                long startBucket = B[t + 1];
                long endBucket = B[t + inst.p[i]];

                if (endBucket == startBucket) {
                    double hki = lengthOfIntervalIntersection(t, t + inst.p[i], tk[startBucket - 1], tk[startBucket]) / ((double) (tk[startBucket] - tk[startBucket - 1]));
                    tempM += v[startBucket] * hki;               
                }
                else if (endBucket == startBucket + 1) {
                    double hkiStart = lengthOfIntervalIntersection(t, t + inst.p[i], tk[startBucket - 1], tk[startBucket]) / ((double) (tk[startBucket] - tk[startBucket - 1]));
                    double hkiEnd = lengthOfIntervalIntersection(t, t + inst.p[i], tk[endBucket - 1], tk[endBucket]) / ((double) (tk[endBucket] - tk[endBucket - 1]));
                    tempM += v[startBucket] * hkiStart;
                    tempM += v[endBucket] * hkiEnd;               
                }
                else {
                    double hkiStart = lengthOfIntervalIntersection(t, t + inst.p[i], tk[startBucket - 1], tk[startBucket]) / ((double) (tk[startBucket] - tk[startBucket - 1]));
                    double hkiEnd = lengthOfIntervalIntersection(t, t + inst.p[i], tk[endBucket - 1], tk[endBucket]) / ((double) (tk[endBucket] - tk[endBucket - 1]));
                    tempM += v[startBucket] * hkiStart;
                    tempM += A[startBucket + 1][endBucket - 1];                
                    tempM += v[endBucket] * hkiEnd;
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
