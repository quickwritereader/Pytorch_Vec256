#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <gtest/gtest.h>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <complex>
#include <algorithm>
#include <math.h>
#include <float.h>

template <typename T>
using VecType = typename at::vec::Vectorized<T>;

// template <typename T>
// std::ostream& operator<<(std::ostream& stream, const VecType<T>& vec) {
//     T buf[VecType<T>::size()];
//     vec.store(buf);
//     stream << "vec[";
//     for (int i = 0; i != VecType<T>::size(); i++) {
//         if (i != 0) {
//             stream << ", ";
//         }
//         stream << buf[i];
//     }
//     stream << "]";
//     return stream;
// }

VecType<float> opp(VecType<float> g1, VecType<float> g2) {

    VecType<float> g3(8.f);
    auto z = g1 & g2;
    z = z + g3 + g2;
    return z;
}

VecType<double> opp(VecType<double> g1, VecType<double> g2) {

    VecType<double> g3(8.f);
    auto z = g1 & g2;
    z = z + g3 + g2;
    return z;
}

VecType<float> opp3(VecType<float> g1) {

    return g1.sin() + g1.sin();
}


VecType<double> opp3(VecType<double> g1) {

    return g1.sin() + g1.sin();
}

VecType<float> forex(VecType<float> g1,VecType<float> g2) {
    VecType<float> z{0};
    for(int i=0;i<2;i++){
        z = z.sin() + opp(g1.abs(), g2);
        z = opp(z, g1.abs());
    }
    return z;
}

#include <iostream>
int main() {
 #if !defined(__s390x__) || !defined(CPU_CAPABILITY_ZVECTOR)
std::cout<<"normal  "<<std::endl;
 #endif
 #if defined(__s390x__) && defined(CPU_CAPABILITY_ZVECTOR)
std::cout<<" defined(__s390x__) && defined(CPU_CAPABILITY_ZVECTOR)  "<<std::endl;
 #endif
    VecType<float> g1(-3.f);
    VecType<float> g2(5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f);
    auto z = forex(g1,g2);
    std::cout << z << std::endl;

    VecType<double> gd1(3.0);
    VecType<double> gd2(-5.0);

    auto r1 = VecType<int>::arange(1.0, 2);
    auto r2 = VecType<float>::arange(1.0, 2);
    std::cout << "aranged: " << r1.angle() << " " << r2.angle() << std::endl;
    auto zd = opp(gd1, gd2.neg());
    std::cout << zd << std::endl;
    auto r3 = VecType<double>(1.0, 0.0, 3.0, 0.0);
    std::cout << r3.zero_mask() << std::endl;
    auto r4 = VecType<float>(1., 0., 3., 0., 4., 0., 0., 0.);
    std::cout << r4 << std::endl;
    std::cout << r4.zero_mask() << std::endl;

    auto r5 = VecType<float>(1., 2., 3., 4., 5., 6., 7., 8.);
    std::cout << r5.reciprocal() << std::endl;
    std::cout << r5.sqrt() << std::endl;
    auto r6 = VecType<float>(1., 3.14 / 2, 3., 4., 5., 6., 7., 8.);
    auto rr = r6.sin() + r6.sin().tanh().tanh();
    std::cout << rr << std::endl;
    return 1;
}