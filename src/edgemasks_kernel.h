#ifndef VAPOURSYNTH_EDGEMASKS_KERNEL_H
#define VAPOURSYNTH_EDGEMASKS_KERNEL_H

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include "edgemasks.h"

// This header is compiled for several instruction sets and linked into one binary.
// Its functions have internal linkage so that the linker cannot merge code built for different ones.

namespace edgemasks {

// What the operator formulas compute with, such as int, float or a SIMD vector of either.
template<typename T>
concept Operand = std::copyable<T> && requires(const T& a) {
    { a + a } -> std::convertible_to<T>;
    { a - a } -> std::convertible_to<T>;
    { 2 * a } -> std::convertible_to<T>;
};

template<Pixel pixel_t>
using Scalar = std::conditional_t<std::integral<pixel_t>, int, float>;

template<Operand T, std::size_t size>
using Window = std::array<std::array<T, size>, size>;

template<Operand T>
struct Gradient {
    T x;
    T y;
};

template<Pixel pixel_t>
struct WindowRows {
    const pixel_t* prev2;
    const pixel_t* prev1;
    const pixel_t* center;
    const pixel_t* next1;
    const pixel_t* next2;
};

// Reflects index into [0, size) without repeating the edge element.
static constexpr int mirror(int index, int size) noexcept {
    if (index < 0)
        return -index;

    return index < size ? index : size - 2 - (index - size);
}

template<Pixel pixel_t>
static WindowRows<pixel_t> windowRows(const Plane<pixel_t>& plane, int y) noexcept {
    const auto row = [&](int index) noexcept { return plane.src + mirror(index, plane.height) * plane.srcStride; };
    return { row(y - 2), row(y - 1), row(y), row(y + 1), row(y + 2) };
}

template<int size, Pixel pixel_t, std::invocable<const pixel_t*> RowOf>
static auto makeWindow(const WindowRows<pixel_t>& rows, RowOf rowOf) noexcept {
    if constexpr (size == 3)
        return std::array{ rowOf(rows.prev1), rowOf(rows.center), rowOf(rows.next1) };
    else
        return std::array{ rowOf(rows.prev2), rowOf(rows.prev1), rowOf(rows.center), rowOf(rows.next1), rowOf(rows.next2) };
}

template<Operand T>
static T maxAbs(const T& g1, const T& g2, const T& g3, const T& g4) noexcept {
    using std::abs;
    using std::max;
    return max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4)));
}

template<Operator op, Operand T>
static auto response(const Window<T, 3>& window) noexcept {
    const auto& [row0, row1, row2] = window;
    const auto& [a00, a01, a02] = row0;
    const auto& [a10, a11, a12] = row1;
    const auto& [a20, a21, a22] = row2;

    if constexpr (op == Operator::Tritical) {
        return Gradient<T>{ a10 - a12, a01 - a21 };
    } else if constexpr (op == Operator::Cross) {
        return Gradient<T>{ a00 - a22, a02 - a20 };
    } else if constexpr (op == Operator::Prewitt) {
        return Gradient<T>{ a00 + a10 + a20 - a02 - a12 - a22, a00 + a01 + a02 - a20 - a21 - a22 };
    } else if constexpr (op == Operator::Sobel) {
        return Gradient<T>{ a00 + 2 * a10 + a20 - a02 - 2 * a12 - a22, a00 + 2 * a01 + a02 - a20 - 2 * a21 - a22 };
    } else if constexpr (op == Operator::Scharr) {
        return Gradient<T>{ 3 * (a00 + a20) + 10 * a10 - 3 * (a02 + a22) - 10 * a12, 3 * (a00 + a02) + 10 * a01 - 3 * (a20 + a22) - 10 * a21 };
    } else if constexpr (op == Operator::RScharr) {
        return Gradient<T>{ 47 * (a00 + a20) + 162 * a10 - 47 * (a02 + a22) - 162 * a12, 47 * (a00 + a02) + 162 * a01 - 47 * (a20 + a22) - 162 * a21 };
    } else if constexpr (op == Operator::Kroon) {
        return Gradient<T>{ 17 * (a00 + a20) + 61 * a10 - 17 * (a02 + a22) - 61 * a12, 17 * (a00 + a02) + 61 * a01 - 17 * (a20 + a22) - 61 * a21 };
    } else if constexpr (op == Operator::Robinson3) {
        return edgemasks::maxAbs(a02 + a01 + a00 - a20 - a21 - a22,
                                 a01 + a00 + a10 - a21 - a22 - a12,
                                 a00 + a10 + a20 - a22 - a12 - a02,
                                 a10 + a20 + a21 - a12 - a02 - a01);
    } else if constexpr (op == Operator::Robinson5) {
        return edgemasks::maxAbs(a02 + 2 * a01 + a00 - a20 - 2 * a21 - a22,
                                 a01 + 2 * a00 + a10 - a21 - 2 * a22 - a12,
                                 a00 + 2 * a10 + a20 - a22 - 2 * a12 - a02,
                                 a10 + 2 * a20 + a21 - a12 - 2 * a02 - a01);
    } else if constexpr (op == Operator::Kirsch) {
        const T g1 = 5 * (a02 + a01 + a00) - 3 * (a10 + a20 + a21 + a22 + a12);
        const T g2 = 5 * (a01 + a00 + a10) - 3 * (a20 + a21 + a22 + a12 + a02);
        const T g3 = 5 * (a00 + a10 + a20) - 3 * (a21 + a22 + a12 + a02 + a01);
        const T g4 = 5 * (a10 + a20 + a21) - 3 * (a22 + a12 + a02 + a01 + a00);
        const T g5 = 5 * (a20 + a21 + a22) - 3 * (a12 + a02 + a01 + a00 + a10);
        const T g6 = 5 * (a21 + a22 + a12) - 3 * (a02 + a01 + a00 + a10 + a20);
        const T g7 = 5 * (a22 + a12 + a02) - 3 * (a01 + a00 + a10 + a20 + a21);
        const T g8 = 5 * (a12 + a02 + a01) - 3 * (a00 + a10 + a20 + a21 + a22);
        using std::max;
        return max(edgemasks::maxAbs(g1, g2, g3, g4), edgemasks::maxAbs(g5, g6, g7, g8));
    }
}

template<Operator op, Operand T>
static auto response(const Window<T, 5>& window) noexcept {
    const auto& [row0, row1, row2, row3, row4] = window;
    const auto& [a00, a01, a02, a03, a04] = row0;
    const auto& [a10, a11, a12, a13, a14] = row1;
    const auto& [a20, a21, a22, a23, a24] = row2;
    const auto& [a30, a31, a32, a33, a34] = row3;
    const auto& [a40, a41, a42, a43, a44] = row4;

    if constexpr (op == Operator::ExPrewitt) {
        return Gradient<T>{ 2 * (a00 + a10 + a20 + a30 + a40) + a01 + a11 + a21 + a31 + a41 - a03 - a13 - a23 - a33 - a43 - 2 * (a04 + a14 + a24 + a34 + a44),
                            2 * (a00 + a01 + a02 + a03 + a04) + a10 + a11 + a12 + a13 + a14 - a30 - a31 - a32 - a33 - a34 - 2 * (a40 + a41 + a42 + a43 + a44) };
    } else if constexpr (op == Operator::ExSobel) {
        return Gradient<T>{ 2 * (a00 + a10 + a30 + a40) + 4 * a20 + a01 + a11 + 2 * a21 + a31 + a41
                                - a03 - a13 - 2 * a23 - a33 - a43 - 2 * (a04 + a14 + a34 + a44) - 4 * a24,
                            2 * (a00 + a01 + a03 + a04) + 4 * a02 + a10 + a11 + 2 * a12 + a13 + a14
                                - a30 - a31 - 2 * a32 - a33 - a34 - 2 * (a40 + a41 + a43 + a44) - 4 * a42 };
    } else if constexpr (op == Operator::FDoG) {
        return Gradient<T>{ a00 + a01 + a40 + a41 + 2 * (a10 + a11 + a30 + a31) + 3 * (a20 + a21)
                                - a03 - a04 - a43 - a44 - 2 * (a13 + a14 + a33 + a34) - 3 * (a23 + a24),
                            a00 + a10 + a04 + a14 + 2 * (a01 + a11 + a03 + a13) + 3 * (a02 + a12)
                                - a30 - a40 - a34 - a44 - 2 * (a31 + a41 + a33 + a43) - 3 * (a32 + a42) };
    } else if constexpr (op == Operator::ExKirsch) {
        const T g1 = 9 * (a14 + a04 + a03 + a02 + a01 + a00 + a10) - 7 * (a20 + a30 + a40 + a41 + a42 + a43 + a44 + a34 + a24)
            + 5 * (a13 + a12 + a11) - 3 * (a21 + a31 + a32 + a33 + a23);
        const T g2 = 9 * (a03 + a02 + a01 + a00 + a10 + a20 + a30) - 7 * (a40 + a41 + a42 + a43 + a44 + a34 + a24 + a14 + a04)
            + 5 * (a12 + a11 + a21) - 3 * (a31 + a32 + a33 + a23 + a13);
        const T g3 = 9 * (a01 + a00 + a10 + a20 + a30 + a40 + a41) - 7 * (a42 + a43 + a44 + a34 + a24 + a14 + a04 + a03 + a02)
            + 5 * (a11 + a21 + a31) - 3 * (a32 + a33 + a23 + a13 + a12);
        const T g4 = 9 * (a10 + a20 + a30 + a40 + a41 + a42 + a43) - 7 * (a44 + a34 + a24 + a14 + a04 + a03 + a02 + a01 + a00)
            + 5 * (a21 + a31 + a32) - 3 * (a33 + a23 + a13 + a12 + a11);
        const T g5 = 9 * (a30 + a40 + a41 + a42 + a43 + a44 + a34) - 7 * (a24 + a14 + a04 + a03 + a02 + a01 + a00 + a10 + a20)
            + 5 * (a31 + a32 + a33) - 3 * (a23 + a13 + a12 + a11 + a21);
        const T g6 = 9 * (a41 + a42 + a43 + a44 + a34 + a24 + a14) - 7 * (a04 + a03 + a02 + a01 + a00 + a10 + a20 + a30 + a40)
            + 5 * (a32 + a33 + a23) - 3 * (a13 + a12 + a11 + a21 + a31);
        const T g7 = 9 * (a43 + a44 + a34 + a24 + a14 + a04 + a03) - 7 * (a02 + a01 + a00 + a10 + a20 + a30 + a40 + a41 + a42)
            + 5 * (a33 + a23 + a13) - 3 * (a12 + a11 + a21 + a31 + a32);
        const T g8 = 9 * (a34 + a24 + a14 + a04 + a03 + a02 + a01) - 7 * (a00 + a10 + a20 + a30 + a40 + a41 + a42 + a43 + a44)
            + 5 * (a23 + a13 + a12) - 3 * (a11 + a21 + a31 + a32 + a33);
        using std::max;
        return max(edgemasks::maxAbs(g1, g2, g3, g4), edgemasks::maxAbs(g5, g6, g7, g8));
    }
}

template<Operand T>
static auto toFloat(const T& value) noexcept {
    if constexpr (std::is_arithmetic_v<T>)
        return static_cast<float>(value);
    else if constexpr (requires { to_float(value); })
        return to_float(value);
    else
        return value;
}

template<Operand T>
static auto magnitude(const Gradient<T>& gradient) noexcept {
    using std::sqrt;
    const auto gx = edgemasks::toFloat(gradient.x);
    const auto gy = edgemasks::toFloat(gradient.y);
    return sqrt(gx * gx + gy * gy);
}

template<Operand T>
static auto magnitude(const T& maximum) noexcept {
    return edgemasks::toFloat(maximum);
}

}  // namespace edgemasks

#endif  // VAPOURSYNTH_EDGEMASKS_KERNEL_H
