#include <c10/util/thread_name.h>

#include <algorithm>

#ifndef __GLIBC_PREREQ
#define __GLIBC_PREREQ(x, y) 0
#endif

#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 12) && !defined(__APPLE__) && \
    !defined(__ANDROID__)
#define C10_HAS_PTHREAD_SETNAME_NP
#endif

#ifdef C10_HAS_PTHREAD_SETNAME_NP
#include <pthread.h>
#endif

namespace c10 {

void setThreadName(std::string name) {
#ifdef C10_HAS_PTHREAD_SETNAME_NP
  constexpr size_t kMaxThreadName = 15;
  name.resize(std::min(name.size(), kMaxThreadName));

  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

} // namespace c10
