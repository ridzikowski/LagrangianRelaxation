#pragma once
#include <chrono>
#include <bits/stdc++.h>

inline std::chrono::time_point<std::chrono::steady_clock> startClock() {
    return std::chrono::steady_clock::now();
}

inline long getClockMicro(std::chrono::time_point<std::chrono::steady_clock> & start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
}

long gcd(long a, long b)
{
    if (a == 0)
        return b;
    return gcd(b % a, a);
}
 
long findGCD(const std::vector<long> & arr)
{
    long result = arr[0];
    for (long i = 1; i < arr.size(); i++) {
        result = gcd(arr[i], result);
        if(result == 1)
            return 1;
    }
    return result;
}

long floorInts(long x, long y) {
    return x / y;
}

long ceilInts(long x, long y) {
    return x % y == 0 ? x / y : (x + y) / y;
}

long lengthOfIntervalIntersection(long firstStart, long firstEnd, long secondStart, long secondEnd) {
    if (firstStart > secondStart)
        return lengthOfIntervalIntersection(secondStart, secondEnd, firstStart, firstEnd);

    if (firstEnd > secondStart)
        return std::min(firstEnd, secondEnd) - secondStart;
    else
        return 0;
}
