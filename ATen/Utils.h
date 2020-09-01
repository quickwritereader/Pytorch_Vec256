#pragma once 
#include <c10/core/StorageImpl.h>
#include <c10/core/UndefinedTensorImpl.h>

#include <c10/core/ScalarType.h> 
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <sstream>
#include <typeinfo>
#include <numeric>
#include <memory>

#define AT_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete

namespace at {

CAFFE2_API inline int _crash_if_asan(int arg){
    volatile char x[3];
    x[arg] = 0;
    return x[0];
}


template <size_t N>
std::array<int64_t, N> check_intlist(ArrayRef<int64_t> list, const char * name, int pos) {
  if (list.empty()) {
    // TODO: is this necessary?  We used to treat nullptr-vs-not in IntList differently
    // with strides as a way of faking optional.
    list = {};
  }
  auto res = std::array<int64_t, N>();
  if (list.size() == 1 && N > 1) {
    res.fill(list[0]);
    return res;
  }
  if (list.size() != N) {
    AT_ERROR("Expected a list of ", N, " ints but got ", list.size(), " for argument #", pos, " '", name, "'");
  }
  std::copy_n(list.begin(), N, res.begin());
  return res;
}

inline int64_t sum_intlist(ArrayRef<int64_t> list) {
  return std::accumulate(list.begin(), list.end(), 0ll);
}

inline int64_t prod_intlist(ArrayRef<int64_t> list) {
  return std::accumulate(list.begin(), list.end(), 1ll, std::multiplies<int64_t>());
}
  
} // at
