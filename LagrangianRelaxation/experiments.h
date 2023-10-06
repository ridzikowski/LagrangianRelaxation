#include "utils.h"
#include "subgradient.h"
#include "subgradientExtensionFast.h"
#include "simulatedAnnealing.h"
#include "greedy.h"

void largeScaleTests() {

    std::ofstream outputFile("output/largeScaleTesting.csv");

    long count = 0;
    long Ns[] = { 30, 40, 50 };
    for (long n : Ns) {
        long limit = -100 * n;
        for (int seed = 0; seed < 50; ++seed) {
            Instance * inst = Instance::createRandomInstance(n, seed);
            if (count % 2 == 0) {
                for (int i = 0; i < n; ++i) {
                    inst->p[i] *= 16;
                    inst->d[i] *= 16;
                }            
            }
            long H = 0;
            for (long i = 0; i < inst->n; i++)
                H += inst->p[i];
            long m = std::max(H / 50, 5L);
            long b = 3;

            long greedyUB = greedy(*inst).first;

            std::vector<double> lbSubHarmonic;
            std::vector<long> ubSubHarmonic;

            {
                auto clock = startClock();
                double WBest;
                std::vector<double> u;
                std::vector<long> S;
                std::vector<double> alphas;
                subgradientHarmonic(*inst, WBest, u, S, lbSubHarmonic, ubSubHarmonic, alphas, true, limit);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
            }

            std::vector<double> lbSubHarmonic2;
            std::vector<long> ubSubHarmonic2;

            {
                auto clock = startClock();
                double WBest;
                std::vector<double> v;
                std::vector<long> S;
                std::vector<double> alphas;
                subgradientHarmonicExtensionFast(*inst, m, WBest, v, S, lbSubHarmonic2, ubSubHarmonic2, alphas, true, limit);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
            }                    

            std::vector<double> lbSubHarmonic3;
            std::vector<long> ubSubHarmonic3;
            
            {
                auto clock = startClock();
                double WBest; std::vector<double> u; std::vector<long> S; std::vector<double> trajectory; std::vector<long> UB; std::vector<double> alphas;
                Instance * inst_ = inst->scaleArbitrary(b);
                long H_ = 0;
                for (long i = 0; i < inst_->n; i++)
                    H_ += inst_->p[i];

                double WBest_;
                std::vector<double> u_;
                std::vector<long> S_;

                subgradientHarmonic(*inst_, WBest_, u_, S_, trajectory, UB, alphas, true, limit);

                long H = 0;
                for (long i = 0; i < inst->n; i++)
                    H += inst->p[i];
                double ratio = H_ / (double) H;

                u.clear();
                for (long t = 0; t <= H; t++)
                    u.push_back(u_[t * ratio]);

                std::vector<long> g;
                for (long t = 0; t <= H; t++)
                    g.push_back(0);
                S.clear();
                for (long i = 0; i < inst->n; i++)
                    S.push_back(0);

                lowerProblemBasic(*inst, u, WBest, g, S);
                UB.push_back(getUBByHeuristic(*inst, S));
                lbSubHarmonic3.push_back(WBest);
                ubSubHarmonic3.push_back(UB[UB.size() - 1]);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;

                delete inst_;
            }

            std::vector<double> lbSubGamma;
            std::vector<long> ubSubGamma;

            {
                auto clock = startClock();
                double WBest;
                std::vector<double> u;
                std::vector<long> S;
                std::vector<double> alphas;
                subgradientGammaFormula(*inst, WBest, u, S, lbSubGamma, ubSubGamma, alphas, limit);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
            }

            std::vector<double> lbSubGamma2;
            std::vector<long> ubSubGamma2;

            {
                auto clock = startClock();
                double WBest;
                std::vector<double> v;
                std::vector<long> S;
                std::vector<double> alphas;
                subgradientGammaFormulaExtensionFast(*inst, m, WBest, v, S, lbSubGamma2, ubSubGamma2, alphas, limit);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
            }

            std::vector<double> lbSubGamma3;
            std::vector<long> ubSubGamma3;
            
            {
                auto clock = startClock();
                double WBest; std::vector<double> u; std::vector<long> S; std::vector<double> trajectory; std::vector<long> UB; std::vector<double> alphas;
                Instance * inst_ = inst->scaleArbitrary(b);
                long H_ = 0;
                for (long i = 0; i < inst_->n; i++)
                    H_ += inst_->p[i];

                double WBest_;
                std::vector<double> u_;
                std::vector<long> S_;

                subgradientGammaFormula(*inst_, WBest_, u_, S_, trajectory, UB, alphas, limit);

                long H = 0;
                for (long i = 0; i < inst->n; i++)
                    H += inst->p[i];
                double ratio = H_ / (double) H;

                u.clear();
                for (long t = 0; t <= H; t++)
                    u.push_back(u_[t * ratio]);

                std::vector<long> g;
                for (long t = 0; t <= H; t++)
                    g.push_back(0);
                S.clear();
                for (long i = 0; i < inst->n; i++)
                    S.push_back(0);

                lowerProblemBasic(*inst, u, WBest, g, S);
                UB.push_back(getUBByHeuristic(*inst, S));
                lbSubGamma3.push_back(WBest);
                ubSubGamma3.push_back(UB[UB.size() - 1]);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;

                delete inst_;
            }

            std::vector<double> lbSubRandom;
            std::vector<long> ubSubRandom;

            {
                auto clock = startClock();
                double WBest;
                std::vector<double> u;
                std::vector<long> S;
                std::vector<double> alphas;
                subgradientGammaRandom(*inst, WBest, u, S, lbSubRandom, ubSubRandom, alphas, 1, limit);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
            }

            std::vector<double> lbSubRandom2;
            std::vector<long> ubSubRandom2;

            {
                auto clock = startClock();
                double WBest;
                std::vector<double> v;
                std::vector<long> S;
                std::vector<double> alphas;
                subgradientGammaRandomExtensionFast(*inst, m, WBest, v, S, lbSubRandom2, ubSubRandom2, alphas, 1, limit);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
            }

            std::vector<double> lbSubRandom3;
            std::vector<long> ubSubRandom3;
            
            {
                auto clock = startClock();
                double WBest; std::vector<double> u; std::vector<long> S; std::vector<double> trajectory; std::vector<long> UB; std::vector<double> alphas;
                Instance * inst_ = inst->scaleArbitrary(b);
                long H_ = 0;
                for (long i = 0; i < inst_->n; i++)
                    H_ += inst_->p[i];

                double WBest_;
                std::vector<double> u_;
                std::vector<long> S_;

                subgradientGammaRandom(*inst_, WBest_, u_, S_, trajectory, UB, alphas, 1, limit);

                long H = 0;
                for (long i = 0; i < inst->n; i++)
                    H += inst->p[i];
                double ratio = H_ / (double) H;

                u.clear();
                for (long t = 0; t <= H; t++)
                    u.push_back(u_[t * ratio]);

                std::vector<long> g;
                for (long t = 0; t <= H; t++)
                    g.push_back(0);
                S.clear();
                for (long i = 0; i < inst->n; i++)
                    S.push_back(0);

                lowerProblemBasic(*inst, u, WBest, g, S);
                UB.push_back(getUBByHeuristic(*inst, S));
                lbSubRandom3.push_back(WBest);
                ubSubRandom3.push_back(UB[UB.size() - 1]);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;

                delete inst_;
            }

            std::vector<double> lbSA;
            std::vector<long> ubSA;

            {
                auto clock = startClock();
                double WBest;
                std::vector<double> u;
                std::vector<long> S;
                std::vector<double> alphas;
                simulatedAnnealing(*inst, WBest, u, S, lbSA, ubSA, alphas, 1, 0.95, 1000.0, 1, limit);
                std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
            }

            outputFile << n << " " << seed << " " << (count % 2)
            << " " << lbSubHarmonic[lbSubHarmonic.size() - 1] << " " << ubSubHarmonic[ubSubHarmonic.size() - 1]
            << " " << lbSubHarmonic2[lbSubHarmonic2.size() - 1] << " " << ubSubHarmonic2[ubSubHarmonic2.size() - 1]
            << " " << lbSubHarmonic3[lbSubHarmonic3.size() - 1] << " " << ubSubHarmonic3[ubSubHarmonic3.size() - 1]
            << " " << lbSubGamma[lbSubGamma.size() - 1] << " " << ubSubGamma[ubSubGamma.size() - 1]
            << " " << lbSubGamma2[lbSubGamma2.size() - 1] << " " << ubSubGamma2[ubSubGamma2.size() - 1]
            << " " << lbSubGamma3[lbSubGamma3.size() - 1] << " " << ubSubGamma3[ubSubGamma3.size() - 1]
            << " " << lbSubRandom[lbSubRandom.size() - 1] << " " << ubSubRandom[ubSubRandom.size() - 1]
            << " " << lbSubRandom2[lbSubRandom2.size() - 1] << " " << ubSubRandom2[ubSubRandom2.size() - 1]
            << " " << lbSubRandom3[lbSubRandom3.size() - 1] << " " << ubSubRandom3[ubSubRandom3.size() - 1]
            << " " << lbSA[lbSA.size() - 1] << " " << ubSA[ubSA.size() - 1]
            << " " << greedyUB
            << std::endl;

            count++;
        }
    }
}

void arbitraryTestingIterationsAndSpeed(long seed) {
    long n = 30;
    long limit = 300;

    std::ofstream outputFile("output/arbitraryScalingItSp.csv");

    long greedyUB = 0;
    Instance * inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }
    greedyUB = greedy(*inst).first;

    for (int limit = 0; limit < 100; limit++) {
        outputFile << limit;

        long bs[] = {1, 5, 10, 15};

        for (long b : bs) {
            double lb = 0;
            long ub = 0;
            
            auto clock = startClock();
            double WBest; std::vector<double> u; std::vector<long> S; std::vector<double> trajectory; std::vector<long> UB; std::vector<double> alphas;
            Instance * inst_ = inst->scaleArbitrary(b);
            long H_ = 0;
            for (long i = 0; i < inst_->n; i++)
                H_ += inst_->p[i];

            double WBest_;
            std::vector<double> u_;
            std::vector<long> S_;

            subgradientGammaFormula(*inst_, WBest_, u_, S_, trajectory, UB, alphas, limit);

            long H = 0;
            for (long i = 0; i < inst->n; i++)
                H += inst->p[i];
            double ratio = H_ / (double) H;

            u.clear();
            for (long t = 0; t <= H; t++)
                u.push_back(u_[t * ratio]);

            std::vector<long> g;
            for (long t = 0; t <= H; t++)
                g.push_back(0);
            S.clear();
            for (long i = 0; i < inst->n; i++)
                S.push_back(0);

            lowerProblemBasic(*inst, u, WBest, g, S);
            UB.push_back(getUBByHeuristic(*inst, S));
            lb = WBest;
            ub = UB[UB.size() - 1];
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;

            delete inst_;

            outputFile << " " << lb << " " << ub;
        }

        outputFile << " " << greedyUB;
        outputFile << std::endl;        
    }
}

void arbitraryTestingTime(long seed) {
    long n = 30;

    std::ofstream outputFile("output/arbitraryScalingTime.csv");

    long greedyUB = 0;
    Instance * inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }
    greedyUB = greedy(*inst).first;

    for (int limit = -100; limit > -10000; limit -= 10) {
        outputFile << -limit;

        long bs[] = {1, 5, 10, 15};

        for (long b : bs) {
            double lb = 0;
            long ub = 0;

            auto clock = startClock();
            double WBest; std::vector<double> u; std::vector<long> S; std::vector<double> trajectory; std::vector<long> UB; std::vector<double> alphas;
            Instance * inst_ = inst->scaleArbitrary(b);
            long H_ = 0;
            for (long i = 0; i < inst_->n; i++)
                H_ += inst_->p[i];

            double WBest_;
            std::vector<double> u_;
            std::vector<long> S_;

            subgradientGammaFormula(*inst_, WBest_, u_, S_, trajectory, UB, alphas, limit);

            long H = 0;
            for (long i = 0; i < inst->n; i++)
                H += inst->p[i];
            double ratio = H_ / (double) H;

            u.clear();
            for (long t = 0; t <= H; t++)
                u.push_back(u_[t * ratio]);

            std::vector<long> g;
            for (long t = 0; t <= H; t++)
                g.push_back(0);
            S.clear();
            for (long i = 0; i < inst->n; i++)
                S.push_back(0);

            lowerProblemBasic(*inst, u, WBest, g, S);
            UB.push_back(getUBByHeuristic(*inst, S));
            lb = WBest;
            ub = UB[UB.size() - 1];
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;

            delete inst_;

            outputFile << " " << lb << " " << ub;
        }

        outputFile << " " << greedyUB;
        outputFile << std::endl;        
    }                                 
}

void gcdTestingTime(long seed) {
    long n = 30;

    std::ofstream outputFile("output/gcdScalingTime.csv");

    long greedyUB = 0;
    Instance * inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }
    greedyUB = greedy(*inst).first;

    for (int limit = -100; limit > -10000; limit -= 10) {
        std::vector<double> lb1;
        std::vector<long> ub1;

        Instance * inst = Instance::createRandomInstance(n, seed);
        for (int i = 0; i < n; ++i) {
            inst->p[i] *= 1;
            inst->d[i] *= 1;
        }

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaFormula(*inst, WBest, u, S, lb1, ub1, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lb4;
        std::vector<long> ub4;

        inst = Instance::createRandomInstance(n, seed);
        for (int i = 0; i < n; ++i) {
            inst->p[i] *= 4;
            inst->d[i] *= 4;
        }

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaFormula(*inst, WBest, u, S, lb4, ub4, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lb8;
        std::vector<long> ub8;

        inst = Instance::createRandomInstance(n, seed);
        for (int i = 0; i < n; ++i) {
            inst->p[i] *= 8;
            inst->d[i] *= 8;
        }

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaFormula(*inst, WBest, u, S, lb8, ub8, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lb16;
        std::vector<long> ub16;

        inst = Instance::createRandomInstance(n, seed);
        for (int i = 0; i < n; ++i) {
            inst->p[i] *= 16;
            inst->d[i] *= 16;
        }

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaFormula(*inst, WBest, u, S, lb16, ub16, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        outputFile << -limit;
        outputFile << " " << lb1[lb1.size() - 1] * 16 << " " << ub1[ub1.size() - 1] * 16;
        outputFile << " " << lb4[lb4.size() - 1] * 4 << " " << ub4[ub4.size() - 1] * 4;
        outputFile << " " << lb8[lb8.size() - 1] * 2 << " " << ub8[ub8.size() - 1] * 2;
        outputFile << " " << lb16[lb16.size() - 1] * 1 << " " << ub16[ub16.size() - 1] * 1;
        outputFile << " " << greedyUB;
        outputFile << std::endl;
    }
}

void gcdTestingIterationsAndSpeed(long seed) {
    long n = 30;
    long limit = 300;

    std::ofstream outputFile("output/gcdScalingItSp.csv");

    long greedyUB = 0;
    Instance * inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }
    greedyUB = greedy(*inst).first;

    std::vector<double> lb1;
    std::vector<long> ub1;

    inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 1;
        inst->d[i] *= 1;
    }

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaFormula(*inst, WBest, u, S, lb1, ub1, alphas, limit);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lb4;
    std::vector<long> ub4;

    inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 4;
        inst->d[i] *= 4;
    }

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaFormula(*inst, WBest, u, S, lb4, ub4, alphas, limit);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lb8;
    std::vector<long> ub8;

    inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 8;
        inst->d[i] *= 8;
    }

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaFormula(*inst, WBest, u, S, lb8, ub8, alphas, limit);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lb16;
    std::vector<long> ub16;

    inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaFormula(*inst, WBest, u, S, lb16, ub16, alphas, limit);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    for (int i = 0; i < limit; ++i) {
        outputFile << i;
        outputFile << " " << lb1[i] * 16 << " " << ub1[i] * 16;
        outputFile << " " << lb4[i] * 4 << " " << ub4[i] * 4;
        outputFile << " " << lb8[i] * 2 << " " << ub8[i] * 2;
        outputFile << " " << lb16[i] * 1 << " " << ub16[i] * 1;
        outputFile << " " << greedyUB;
        outputFile << std::endl;
    }
}

void aggregationTestingTime(long seed) {
    long n = 30;

    std::ofstream outputFile("output/aggregationTime.csv");

    long greedyUB = 0;
    Instance * inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }
    greedyUB = greedy(*inst).first;

    long H = 0;
    for (long i = 0; i < inst->n; ++i)
        H += inst->p[i];
    std::cout << "H: " << H << std::endl;

    for (int limit = -100; limit > -10000; limit -= 10) {
        std::vector<double> lbOrg;
        std::vector<long> ubOrg;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaFormula(*inst, WBest, u, S, lbOrg, ubOrg, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lbA;
        std::vector<long> ubA;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            long m = 250;
            subgradientGammaFormulaExtensionFast(*inst, m, WBest, u, S, lbA, ubA, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lbB;
        std::vector<long> ubB;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            long m = 50;
            subgradientGammaFormulaExtensionFast(*inst, m, WBest, u, S, lbB, ubB, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lbC;
        std::vector<long> ubC;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            long m = 10;
            subgradientGammaFormulaExtensionFast(*inst, m, WBest, u, S, lbC, ubC, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }        

        outputFile << -limit;
        outputFile << " " << lbOrg[lbOrg.size() - 1] << " " << ubOrg[ubOrg.size() - 1];
        outputFile << " " << lbA[lbA.size() - 1] << " " << ubA[ubA.size() - 1];
        outputFile << " " << lbB[lbB.size() - 1] << " " << ubB[ubB.size() - 1];
        outputFile << " " << lbC[lbC.size() - 1] << " " << ubC[ubC.size() - 1];
        outputFile << " " << greedyUB;
        outputFile << std::endl;
    }
}

void aggregationTestingIterationsAndSpeed(long seed) {
    long n = 30;
    long limit = 300;

    std::ofstream outputFile("output/aggregationItSp.csv");

    long greedyUB = 0;
    Instance * inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }
    greedyUB = greedy(*inst).first;

    std::vector<double> lbOrg;
    std::vector<long> ubOrg;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaFormula(*inst, WBest, u, S, lbOrg, ubOrg, alphas, limit);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lbA;
    std::vector<long> ubA;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        long m = 250;
        subgradientGammaFormulaExtensionFast(*inst, m, WBest, u, S, lbA, ubA, alphas, limit);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lbB;
    std::vector<long> ubB;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        long m = 50;
        subgradientGammaFormulaExtensionFast(*inst, m, WBest, u, S, lbB, ubB, alphas, limit);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lbC;
    std::vector<long> ubC;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        long m = 10;
        subgradientGammaFormulaExtensionFast(*inst, m, WBest, u, S, lbC, ubC, alphas, limit);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }        

    for (int i = 0; i < limit; ++i) {
        outputFile << i;
        outputFile << " " << lbOrg[i] << " " << ubOrg[i];
        outputFile << " " << lbA[i] << " " << ubA[i];
        outputFile << " " << lbB[i] << " " << ubB[i];
        outputFile << " " << lbC[i] << " " << ubC[i];
        outputFile << " " << greedyUB;
        outputFile << std::endl;
    }
}

void illustrativeTestingTime(long seed) {
    long n = 30;
    long m = 50;    

    std::ofstream outputFile("output/illustrativeTime.csv");

    long greedyUB = 0;
    Instance * inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }
    greedyUB = greedy(*inst).first;

    for (long limit = -100; limit > -10000; limit -= 10) {

        std::vector<double> lbSubHarmonic;
        std::vector<long> ubSubHarmonic;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientHarmonic(*inst, WBest, u, S, lbSubHarmonic, ubSubHarmonic, alphas, true, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lbSubHarmonic2;
        std::vector<long> ubSubHarmonic2;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> v;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientHarmonicExtensionFast(*inst, m, WBest, v, S, lbSubHarmonic2, ubSubHarmonic2, alphas, true, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lbSubGamma;
        std::vector<long> ubSubGamma;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaFormula(*inst, WBest, u, S, lbSubGamma, ubSubGamma, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lbSubGamma2;
        std::vector<long> ubSubGamma2;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> v;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaFormulaExtensionFast(*inst, m, WBest, v, S, lbSubGamma2, ubSubGamma2, alphas, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lbSubRandom;
        std::vector<long> ubSubRandom;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaRandom(*inst, WBest, u, S, lbSubRandom, ubSubRandom, alphas, 1, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        std::vector<double> lbSubRandom2;
        std::vector<long> ubSubRandom2;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> v;
            std::vector<long> S;
            std::vector<double> alphas;
            subgradientGammaRandomExtensionFast(*inst, m, WBest, v, S, lbSubRandom2, ubSubRandom2, alphas, 1, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }    

        std::vector<double> lbSA;
        std::vector<long> ubSA;

        {
            auto clock = startClock();
            double WBest;
            std::vector<double> u;
            std::vector<long> S;
            std::vector<double> alphas;
            simulatedAnnealing(*inst, WBest, u, S, lbSA, ubSA, alphas, 1, 0.95, 1000.0, 1, limit);
            std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
        }

        outputFile << -limit;
        outputFile << " " << lbSubHarmonic[lbSubHarmonic.size() - 1] << " " << ubSubHarmonic[ubSubHarmonic.size() - 1];
        outputFile << " " << lbSubHarmonic2[lbSubHarmonic2.size() - 1] << " " << ubSubHarmonic2[ubSubHarmonic2.size() - 1];
        outputFile << " " << lbSubGamma[lbSubGamma.size() - 1] << " " << ubSubGamma[ubSubGamma.size() - 1];
        outputFile << " " << lbSubGamma2[lbSubGamma2.size() - 1] << " " << ubSubGamma2[ubSubGamma2.size() - 1];
        outputFile << " " << lbSubRandom[lbSubRandom.size() - 1] << " " << ubSubRandom[ubSubRandom.size() - 1];
        outputFile << " " << lbSubRandom2[lbSubRandom2.size() - 1] << " " << ubSubRandom2[ubSubRandom2.size() - 1];
        outputFile << " " << lbSA[lbSA.size() - 1] << " " << ubSA[ubSA.size() - 1];
        outputFile << " " << greedyUB;
        outputFile << std::endl;
    }    
}

void illustrativeTestingIterationsAndSpeed(long seed) {
    long n = 30;
    long it = 300;
    long m = 50;

    long greedyUB = 0;
    Instance * inst = Instance::createRandomInstance(n, seed);
    for (int i = 0; i < n; ++i) {
        inst->p[i] *= 16;
        inst->d[i] *= 16;
    }
    greedyUB = greedy(*inst).first;

    std::ofstream outputFile("output/illustrativeItSp.csv");

    long H = 0;
    for (long i = 0; i < inst->n; ++i)
        H += inst->p[i];
    std::cout << "H: " << H << std::endl;

    std::vector<double> lbSubHarmonic;
    std::vector<long> ubSubHarmonic;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientHarmonic(*inst, WBest, u, S, lbSubHarmonic, ubSubHarmonic, alphas, true, it);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lbSubHarmonic2;
    std::vector<long> ubSubHarmonic2;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> v;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientHarmonicExtensionFast(*inst, m, WBest, v, S, lbSubHarmonic2, ubSubHarmonic2, alphas, true, it);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lbSubGamma;
    std::vector<long> ubSubGamma;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaFormula(*inst, WBest, u, S, lbSubGamma, ubSubGamma, alphas, it);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lbSubGamma2;
    std::vector<long> ubSubGamma2;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> v;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaFormulaExtensionFast(*inst, m, WBest, v, S, lbSubGamma2, ubSubGamma2, alphas, it);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lbSubRandom;
    std::vector<long> ubSubRandom;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaRandom(*inst, WBest, u, S, lbSubRandom, ubSubRandom, alphas, 1, it);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    std::vector<double> lbSubRandom2;
    std::vector<long> ubSubRandom2;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> v;
        std::vector<long> S;
        std::vector<double> alphas;
        subgradientGammaRandomExtensionFast(*inst, m, WBest, v, S, lbSubRandom2, ubSubRandom2, alphas, 1, it);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }    

    std::vector<double> lbSA;
    std::vector<long> ubSA;

    {
        auto clock = startClock();
        double WBest;
        std::vector<double> u;
        std::vector<long> S;
        std::vector<double> alphas;
        simulatedAnnealing(*inst, WBest, u, S, lbSA, ubSA, alphas, 1, 0.95, 1000.0, 1, it);
        std::cout << "Time: " << getClockMicro(clock) / 1000000.0 << std::endl;
    }

    for (int i = 0; i < it; ++i) {
        outputFile << i
        << " " << lbSubHarmonic[i] << " " << ubSubHarmonic[i]
        << " " << lbSubHarmonic2[i] << " " << ubSubHarmonic2[i]
        << " " << lbSubGamma[i] << " " << ubSubGamma[i]
        << " " << lbSubGamma2[i] << " " << ubSubGamma2[i]
        << " " << lbSubRandom[i] << " " << ubSubRandom[i]
        << " " << lbSubRandom2[i] << " " << ubSubRandom2[i]
        << " " << lbSA[i] << " " << ubSA[i]
        << " " << greedyUB
        << std::endl;
    }
}
