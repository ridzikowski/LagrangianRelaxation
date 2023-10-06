#include <iostream>
#include "experiments.h"
#include "dynamicprograming.h"

int main()
{
   long seed = 15;

   gcdTestingTime(seed);
   gcdTestingIterationsAndSpeed(seed);

   aggregationTestingTime(seed);
   aggregationTestingIterationsAndSpeed(seed);

   arbitraryTestingTime(seed);
   arbitraryTestingIterationsAndSpeed(seed);

   illustrativeTestingTime(seed);
   illustrativeTestingIterationsAndSpeed(seed);

   largeScaleTests();
}
