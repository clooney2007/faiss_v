/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/25.
*/

#ifndef FAISS_V_FAISSVEXCEPTION_H
#define FAISS_V_FAISSVEXCEPTION_H

#include <exception>
#include <string>

namespace faiss_v {
    /// Base class for Faiss exceptions
    class FaissVException : public std::exception {
    public:
        explicit FaissVException(const std::string& msg);

        FaissVException(const std::string& msg,
                       const char* funcName,
                       const char* file,
                       int line);

        /// from std::exception
        const char* what() const noexcept override;

        std::string msg;
    };
}

#endif //FAISS_V_FAISSVEXCEPTION_H
