#pragma once

#include <ATen/ATen.h>
//#include <ATen/native/DispatchStub.h>

#include <cfenv>
#include <limits>
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace at {
namespace native { 
 
            template <typename T>
            inline void checkZeroPoint(const std::string& fn_name, int64_t zero_point) {
                TORCH_CHECK(
                    zero_point <= std::numeric_limits<T>::max(),
                    fn_name,
                    " zero_point ",
                    zero_point,
                    " is out of range.");
                TORCH_CHECK(
                    zero_point >= std::numeric_limits<T>::min(),
                    fn_name,
                    " zero_point ",
                    zero_point,
                    " is out of range.");
            }


#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
            template <class T>
            inline float Round(const float x) {
                return ::nearbyintf(x);
            }
            inline double Round(const double x) {
                return ::nearbyint(x);
            }
#else
            template <class T>
            inline T Round(const T x) {
                return std::nearbyint(x);
            }
#endif

            template <typename T>
            inline T quantize_val(double scale, int64_t zero_point, float value) {
                // std::nearbyint results in nearest integer value according to the current
                // rounding mode and the default rounding mode is rounds to even in half-way
                // cases in most popular processor architectures like x86 and ARM. This is
                // typically faster than an alternatives like std::round that rounds half-way
                // cases away from zero, and can be consistent with SIMD implementations for
                // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
                // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
                int64_t qvalue;
                constexpr int64_t qmin = std::numeric_limits<typename T::underlying>::min();
                constexpr int64_t qmax = std::numeric_limits<typename T::underlying>::max();
                float inv_scale = 1.0f / static_cast<float>(scale);
                qvalue = static_cast<int64_t>(zero_point + Round(value * inv_scale));
                qvalue = std::max<int64_t>(qvalue, qmin);
                qvalue = std::min<int64_t>(qvalue, qmax);
                return static_cast<T>(qvalue);
            }

           inline uint8_t quantize_val_arm(
                const float scale,
                const int32_t zero_point,
                const float value) {
                const int32_t qmin = std::numeric_limits<uint8_t>::min();
                const int32_t qmax = std::numeric_limits<uint8_t>::max();
                auto r = zero_point + static_cast<int32_t>(Round(value / scale));
                r = std::max(r, qmin);
                r = std::min(r, qmax);
                return static_cast<uint8_t>(r);
            }

            template <typename T, int precision=8>
            inline void quantize_vec(
                double scale,
                int64_t zero_point,
                const float* src,
                T* dst,
                size_t count) {
                checkZeroPoint<typename T::underlying>("quantize_vec", zero_point);
                for (int64_t i = 0; i < count; ++i) {
                    dst[i] = quantize_val<T>(scale, zero_point, src[i]);
                }
            }

            template <typename T>
            inline float dequantize_val(double scale, int64_t zero_point, T value) {
                // We need to convert the qint8 value to float to ensure the subtraction
                // subexpression returns a float
                return (static_cast<float>(value.val_) - zero_point) * scale;
            }


            template <typename SRC_T, typename DST_T>
           inline  DST_T requantize_val(
                double src_scale,
                int64_t src_zero_point,
                double dst_scale,
                int64_t dst_zero_point,
                SRC_T src) {
                const auto dq = dequantize_val<SRC_T>(src_scale, src_zero_point, src);
                return quantize_val<DST_T>(dst_scale, dst_zero_point, dq);
            }

            template <typename DST_T>
            inline DST_T requantize_from_int(double multiplier, int64_t zero_point, int64_t src) {
                int64_t quantize_down =
                    zero_point + std::lrintf(src * static_cast<float>(multiplier));
                int32_t min = std::numeric_limits<typename DST_T::underlying>::min();
                int32_t max = std::numeric_limits<typename DST_T::underlying>::max();
                return static_cast<DST_T>(
                    std::min<int64_t>(std::max<int64_t>(quantize_down, min), max));
            }
 

        } // namespace native
    } // namespace at

