#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
#include <algorithm>
#include "utils.h"

typedef std::pair<long,std::vector<long>> UpperSolution;

class LowerSolution {
    public:

    LowerSolution(long W, std::vector<long> & g, std::vector<long> & S): W(W), g(g), S(S) {}

    long W;
    std::vector<long> g;
    std::vector<long> S;
};

class Instance {

    public:

    std::vector<long> p;
    std::vector<long> w;
    std::vector<long> d;

    long n;

    static Instance * createRandomInstance(long n, long seed, float TF = 0.6, float RDD = 0.4) {
        std::mt19937 mtg(seed);
        Instance * inst = new Instance();
        inst-> n = n;
        long P = 0;
        for (long i = 0; i < n; ++i) {
            inst->p.push_back((mtg() % 10) + 1);
            P += inst->p[i];
        }
        for (long i = 0; i < n; ++i)
            inst->w.push_back((mtg() % 10) + 1);
        long low = floor(P * (1 - TF - RDD / 2));
        long up =  floor(P * (1 - TF + RDD / 2));
        for (long i = 0; i < n; ++i)
            inst->d.push_back(mtg() % (up - low + 1) + low);
        return inst;
    }     

    Instance * scale(long a) const {
        Instance* inst = new Instance();
        inst->n = this->n;

        for (long i = 0; i < this->n; ++i) {
            inst->p.push_back(this->p[i] / a);
            inst->d.push_back(this->d[i] / a);
            inst->w = this->w;
        }

        return inst;
    }

    Instance * scaleArbitrary(long b) const {
        Instance* inst = new Instance();
        inst->n = this->n;

        for (long i = 0; i < this->n; ++i) {
            inst->p.push_back(floorInts(this->p[i], b));
            inst->d.push_back(ceilInts(this->d[i], b));
            inst->w = this->w;
        }

        return inst;
    }

    Instance * scaleAndReturn(long b) const {
        Instance* inst = new Instance();
        inst->n = this->n;

        for (long i = 0; i < this->n; ++i) {
            inst->p.push_back(floorInts(this->p[i], b) * b);
            inst->d.push_back(floorInts(this->d[i], b) * b);
            inst->w = this->w;
        }

        return inst;
    }

    Instance * scaleAndReturn2(long b) const {
        Instance* inst = new Instance();
        inst->n = this->n;

        for (long i = 0; i < this->n; ++i) {
            inst->p.push_back(floorInts(this->p[i], b) * b);
            inst->d.push_back(ceilInts(this->d[i], b) * b);
            inst->w = this->w;
        }

        return inst;
    }        

    void printInstance() {
        std::cout << "p: ";
        for (long i = 0; i < n; ++i)
            std::cout << p[i] << " ";
        std::cout << std::endl;

        std::cout << "w: ";
        for (long i = 0; i < n; ++i)
            std::cout << w[i] << " ";
        std::cout << std::endl;

        std::cout << "d: ";
        for (long i = 0; i < n; ++i)
            std::cout << d[i] << " ";
        std::cout << std::endl;
    }
};

long evaluateSolution(const Instance & inst, std::vector<long> pi) {
    long wiTi = 0;
    long C = 0;
    for (long i = 0; i < pi.size(); ++i) {
        C = C + inst.p[pi[i]];
        wiTi += inst.w[pi[i]] * std::max(0L, C - inst.d[pi[i]]);
    }
    return wiTi;
}

long getUBByHeuristic(const Instance & inst, const std::vector<long> & S) {
    std::vector<std::pair<long,long>> Spi;
    for (long i = 0; i < inst.n; ++i)
        Spi.push_back(std::pair<long,long>(S[i], i));

    sort(Spi.begin(), Spi.end(), 
    [](const std::pair<long,long> & a, const std::pair<long,long> & b) -> bool
    { 
        return a.first < b.first; 
    });

    std::vector<long> pi;
    for (long i = 0; i < inst.n; ++i)
        pi.push_back(Spi[i].second);
    return evaluateSolution(inst, pi);
}
