#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <stdlib.h>
#include <stdint.h>
#include <helper_cuda.h>
#include "log.h"

// throw C++ exception using machine codes [https://stackoverflow.com/a/24488262]
extern "C" void *__cxa_allocate_exception(size_t thrown_size);
extern "C" void __cxa_throw(void *thrown_exception, void* *tinfo, void (*dest) (void *));
extern "C" void *_ZTIl;    // typeinfo of long

#define CUDA_ERROR 1

template<typename T>
void customCheck(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    log_error("CUDA error at %s:%d code=%d(%s) \"%s\"", file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    cudaDeviceReset();

    int64_t *p = (int64_t*) __cxa_allocate_exception(8);
    *p = CUDA_ERROR;
    __cxa_throw(p, &_ZTIl, 0);
  }
}

#define customCheckCudaErrors(val)  customCheck((val), #val, __FILE__, __LINE__)

#endif
