/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/25.
*/

#ifndef FAISS_V_FAISSVASSERT_H
#define FAISS_V_FAISSVASSERT_H

#include "FaissVException.h"


#define FAISSV_THROW_MSG(MSG)                                            \
  do {                                                                  \
    throw faiss_v::FaissVException(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
  } while (false)

#endif //FAISS_V_FAISSVASSERT_H
