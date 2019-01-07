/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/25.
*/

#ifndef FAISS_V_FAISSVASSERT_H
#define FAISS_V_FAISSVASSERT_H

#include "FaissVException.h"

/// Assertions

#define FAISSV_ASSERT(X)                                                            \
  do {                                                                             \
    if (!(X)) {                                                                   \
      fprintf(stderr, "Faiss assertion '%s' failed in %s "                         \
               "at %s:%d\n",                                                       \
               #X, __PRETTY_FUNCTION__, __FILE__, __LINE__);                       \
      abort();                                                                     \
    }                                                                              \
  } while (false)


#define FAISSV_ASSERT_MSG(X, MSG)                                                   \
  do {                                                                             \
    if (!(X)) {                                                                   \
      fprintf(stderr, "Faiss assertion '%s' failed in %s "                         \
               "at %s:%d; details: " MSG "\n",                                     \
               #X, __PRETTY_FUNCTION__, __FILE__, __LINE__);                       \
      abort();                                                                     \
    }                                                                              \
  } while (false)

#define FAISSV_ASSERT_FMT(X, FMT, ...)                                             \
  do {                                                                            \
    if (!(X)) {                                                                   \
      fprintf(stderr, "Faiss assertion '%s' failed in %s "                        \
               "at %s:%d; details: " FMT "\n",                                    \
               #X, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);         \
      abort();                                                                    \
    }                                                                             \
  } while (false)

/// Exceptions for returning user errors

#define FAISSV_THROW_MSG(MSG)                                                     \
do {                                                                              \
  throw faiss_v::FaissVException(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__);   \
  abort();                                                                        \
} while (false)

#define FAISSV_THROW_FMT(FMT, ...)                                                 \
  do {                                                                            \
    std::string __s;                                                              \
    int __size = snprintf(nullptr, 0, FMT, __VA_ARGS__);                          \
    __s.resize(__size + 1);                                                       \
    snprintf(&__s[0], __s.size(), FMT, __VA_ARGS__);                              \
    throw faiss_v::FaissVException(__s, __PRETTY_FUNCTION__, __FILE__, __LINE__);    \
  } while (false)

/// Exceptions thrown upon a conditional failure

#define FAISSV_THROW_IF_NOT_MSG(X, MSG)                                            \
  do {                                                                            \
    if (!(X)) {                                                                   \
      FAISSV_THROW_FMT("Error: '%s' failed: " MSG, #X);                            \
    }                                                                             \
  } while (false)

#define FAISSV_THROW_IF_NOT_FMT(X, FMT, ...)                                       \
  do {                                                                            \
    if (!(X)) {                                                                   \
      FAISSV_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__);               \
    }                                                                             \
  } while (false)

#endif //FAISS_V_FAISSVASSERT_H
