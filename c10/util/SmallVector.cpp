//===- llvm/ADT/SmallVector.cpp - 'Normally small' vectors ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallVector class.
//
//===----------------------------------------------------------------------===//

// ATen: modified from llvm::SmallVector.
// replaced report_bad_alloc_error with std::bad_alloc

#include <c10/util/SmallVector.h>

namespace c10 {

/// grow_pod - This is an implementation of the grow() method which only works
/// on POD-like datatypes and is out of line to reduce code duplication.
void SmallVectorBase::grow_pod(
    void* FirstEl,
    size_t MinSizeInBytes,
    size_t TSize) {
  size_t CurSizeBytes = size_in_bytes();
  size_t NewCapacityInBytes = 2 * capacity_in_bytes() + TSize; // Always grow.
  if (NewCapacityInBytes < MinSizeInBytes)
    NewCapacityInBytes = MinSizeInBytes;

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* NewElts;
  if (BeginX == FirstEl) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    NewElts = malloc(NewCapacityInBytes);
    if (NewElts == nullptr)
      throw std::bad_alloc();

    // Copy the elements over.  No need to run dtors on PODs.
    memcpy(NewElts, this->BeginX, CurSizeBytes);
  } else {
    // If this wasn't grown from the inline copy, grow the allocated space.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    NewElts = realloc(this->BeginX, NewCapacityInBytes);
    if (NewElts == nullptr)
      throw std::bad_alloc();
  }

  this->EndX = (char*)NewElts + CurSizeBytes;
  this->BeginX = NewElts;
  this->CapacityX = (char*)this->BeginX + NewCapacityInBytes;
}

} // namespace c10
