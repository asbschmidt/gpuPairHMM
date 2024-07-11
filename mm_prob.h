#ifndef CONTEXT_H
#define CONTEXT_H

#include <cmath> // std::isinf
#include <algorithm> // std::min

#define MAX_QUAL 254
#define MAX_JACOBIAN_TOLERANCE 8.0
#define JACOBIAN_LOG_TABLE_STEP 0.0001
#define JACOBIAN_LOG_TABLE_INV_STEP (1.0 / JACOBIAN_LOG_TABLE_STEP)
#define JACOBIAN_LOG_TABLE_SIZE ((int) (MAX_JACOBIAN_TOLERANCE / JACOBIAN_LOG_TABLE_STEP) + 1)

template<class NUMBER>
class ContextBase
{
  public:
    static NUMBER ph2pr[128];
    static NUMBER INITIAL_CONSTANT;
    static NUMBER LOG10_INITIAL_CONSTANT;
    static NUMBER RESULT_THRESHOLD; 

    static bool staticMembersInitializedFlag;
    static NUMBER jacobianLogTable[JACOBIAN_LOG_TABLE_SIZE];
    static NUMBER matchToMatchProb[((MAX_QUAL + 1) * (MAX_QUAL + 2)) >> 1];


    static void initializeJacobianLogTable()
    {
      for (int k = 0; k < JACOBIAN_LOG_TABLE_SIZE; k++) {
        // Casting k to a double is acceptable considering that a loss of precision is expected in numerical analysis.
        // The same goes for the final cast o NUMBER which may result in cutting precision in half.
        jacobianLogTable[k] = (NUMBER)(log10(1.0 + pow(10.0, -((double) k) * JACOBIAN_LOG_TABLE_STEP)));
      }
    }

//Called during computation - use single precision where possible
    static int fastRound(NUMBER d) {
      // The cast to NUMBER is to allow trading off precision for speed.
      return (d > ((NUMBER)0.0)) ? (int) (d + ((NUMBER)0.5)) : (int) (d - ((NUMBER)0.5));
    }
    //Called during computation - use single precision where possible
    static NUMBER approximateLog10SumLog10(NUMBER small, NUMBER big) {
      // make sure small is really the smaller value
      if (small > big) {
        NUMBER t = big;
        big = small;
        small = t;
      }

      if (std::isinf(small) || std::isinf(big))
        return big;

      NUMBER diff = big - small;
      // The cast to NUMBER is to allow trading off precision for speed.
      if (diff >= ((NUMBER)MAX_JACOBIAN_TOLERANCE))
        return big;

      // OK, so |y-x| < tol: we use the following identity then:
      // we need to compute log10(10^x + 10^y)
      // By Jacobian logarithm identity, this is equal to
      // max(x,y) + log10(1+10^-abs(x-y))
      // we compute the second term as a table lookup with integer quantization
      // we have pre-stored correction for 0,0.1,0.2,... 10.0

      // The cast to NUMBER is to allow trading off precision for speed.
      int ind = fastRound((NUMBER)(diff * ((NUMBER)JACOBIAN_LOG_TABLE_INV_STEP))); // hard rounding
      return big + jacobianLogTable[ind];
    }

    inline double set_mm_prob(int insQual, int delQual) {
    int minQual = delQual;
    int maxQual = insQual;
    if (insQual <= delQual) {
      minQual = insQual;
      maxQual = delQual;
    }

    return MAX_QUAL < maxQual ?
        1.0 - POW(10.0, approximateLog10SumLog10(-0.1 * minQual, -0.1 * maxQual)) :
        matchToMatchProb[((maxQual * (maxQual + 1)) >> 1) + minQual];
  }
}