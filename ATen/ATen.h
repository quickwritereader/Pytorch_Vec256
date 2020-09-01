#pragma once

#if !defined(_MSC_VER) && __cplusplus < 201402L
#error C++14 or later compatible compiler is required to use ATen.
#endif

#include <c10/core/Allocator.h>
#include <c10/core/Layout.h> 
#include <c10/core/Storage.h> 
#include <c10/util/Exception.h> 