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

template<typename T>
using Scalar = std::conditional_t<std::integral<T>, int, float>;

template<Operand T, std::size_t size>
using Window = std::array<std::array<T, size>, size>;

template<Operand T>
struct Gradient {
    T x;
    T y;
};

template<typename T>
struct WindowRows {
    const T* prev2;
    const T* prev1;
    const T* center;
    const T* next1;
    const T* next2;
};

// Reflects index into [0, size) without repeating the edge element.
static constexpr int mirror(int index, int size) noexcept {
    if (index < 0)
        return -index;

    return index < size ? index : size - 2 - (index - size);
}

template<int size, typename T, std::invocable<const T*> RowOf>
static auto makeWindow(const WindowRows<T>& rows, RowOf rowOf) noexcept {
    if constexpr (size == 3)
        return std::array{ rowOf(rows.prev1), rowOf(rows.center), rowOf(rows.next1) };
    else
        return std::array{ rowOf(rows.prev2), rowOf(rows.prev1), rowOf(rows.center), rowOf(rows.next1), rowOf(rows.next2) };
}

template<Operand T>
struct RingSums {
    std::array<T, 8> triples;  // Element i sums the ring elements i, i + 1 and i + 2.
    T total;
};

template<Operand T>
static RingSums<T> ringSums(const T& r0, const T& r1, const T& r2, const T& r3, const T& r4, const T& r5, const T& r6, const T& r7) noexcept {
    const T pair0 = r0 + r1;
    const T pair2 = r2 + r3;
    const T pair4 = r4 + r5;
    const T pair6 = r6 + r7;
    return { { pair0 + r2, r1 + pair2, pair2 + r4, r3 + pair4, pair4 + r6, r5 + pair6, pair6 + r0, r7 + pair0 }, pair0 + pair2 + pair4 + pair6 };
}

// Largest |8 * value - center|.
template<Operand T>
static T maxDeviation(const std::array<T, 8>& values, const T& center) noexcept {
    using std::max;
    using std::min;
    const auto& [v0, v1, v2, v3, v4, v5, v6, v7] = values;
    const T highest = max(max(max(v0, v1), max(v2, v3)), max(max(v4, v5), max(v6, v7)));
    const T lowest = min(min(min(v0, v1), min(v2, v3)), min(min(v4, v5), min(v6, v7)));
    return max(8 * highest - center, center - 8 * lowest);
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
    } else if constexpr (op == Operator::Kirsch) {
        // Direction = 5 * arc - 3 * (ring - arc) = 8 * arc - 3 * ring.
        const auto [arcs, ring] = edgemasks::ringSums(a00, a01, a02, a12, a22, a21, a20, a10);
        return edgemasks::maxDeviation(arcs, 3 * ring);
    } else {
        const T p = a00 - a22;
        const T q = a01 - a21;
        const T r = a02 - a20;
        const T s = a10 - a12;

        if constexpr (op == Operator::Prewitt) {
            return Gradient<T>{ p - r + s, p + r + q };
        } else if constexpr (op == Operator::Sobel) {
            return Gradient<T>{ p - r + 2 * s, p + r + 2 * q };
        } else if constexpr (op == Operator::Scharr) {
            return Gradient<T>{ 3 * (p - r) + 10 * s, 3 * (p + r) + 10 * q };
        } else if constexpr (op == Operator::RScharr) {
            return Gradient<T>{ 47 * (p - r) + 162 * s, 47 * (p + r) + 162 * q };
        } else if constexpr (op == Operator::Kroon) {
            return Gradient<T>{ 17 * (p - r) + 61 * s, 17 * (p + r) + 61 * q };
        } else {
            const T pq = p + q;
            const T sr = s - r;
            const T g1 = pq + r;
            const T g2 = pq + s;
            const T g3 = sr + p;
            const T g4 = sr - q;

            using std::abs;
            using std::max;

            if constexpr (op == Operator::Robinson3) {
                return max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4)));
            } else {
                // The directions are g1 + g3, g1 - g3, g2 + g4 and g2 - g4, and max(|x + y|, |x - y|) = |x| + |y|.
                return max(abs(g1) + abs(g3), abs(g2) + abs(g4));
            }
        }
    }
}

// Only ExKirsch, as the other 5x5 operators are separable.
template<Operator op, Operand T>
static T response(const Window<T, 5>& window) noexcept {
    const auto& [row0, row1, row2, row3, row4] = window;
    const auto& [a00, a01, a02, a03, a04] = row0;
    const auto& [a10, a11, a12, a13, a14] = row1;
    const auto& [a20, a21, a22, a23, a24] = row2;
    const auto& [a30, a31, a32, a33, a34] = row3;
    const auto& [a40, a41, a42, a43, a44] = row4;

    // Direction = 9 * outerArc - 7 * (outer - outerArc) + 5 * innerArc - 3 * (inner - innerArc) = 8 * (2 * outerArc + innerArc) - 7 * outer - 3 * inner.
    const auto outer = edgemasks::ringSums(a00 + a10, a01 + a02, a03 + a04, a14 + a24, a34 + a44, a42 + a43, a40 + a41, a20 + a30);
    const auto& [t0, t1, t2, t3, t4, t5, t6, t7] = outer.triples;
    // Finishing these before the inner ring keeps fewer vectors live. Clang 21 leaves a NEON loop scalar when it needs more than 32 at once.
    const std::array<T, 8> outerArcs{ t0 + a14, t1 + a34, t2 + a43, t3 + a41, t4 + a30, t5 + a10, t6 + a01, t7 + a03 };
    const auto inner = edgemasks::ringSums(a11, a12, a13, a23, a33, a32, a31, a21);
    const auto arc = [&](std::size_t index) noexcept { return 2 * outerArcs[index] + inner.triples[index]; };
    const std::array<T, 8> arcs{ arc(0), arc(1), arc(2), arc(3), arc(4), arc(5), arc(6), arc(7) };
    return edgemasks::maxDeviation(arcs, 7 * outer.total + 3 * inner.total);
}

// These operators are one 1-D kernel along each axis: gx differentiates across smoothed columns and gy smooths across differentiated columns.
consteval bool isSeparable(Operator op) noexcept {
    return op == Operator::ExPrewitt || op == Operator::ExSobel || op == Operator::FDoG;
}

template<Operator op, Operand T>
static T smooth(const std::array<T, 5>& a) noexcept {
    if constexpr (op == Operator::ExPrewitt)
        return a[0] + a[1] + a[2] + a[3] + a[4];
    else if constexpr (op == Operator::ExSobel)
        return a[0] + a[1] + 2 * a[2] + a[3] + a[4];
    else
        return a[0] + a[4] + 2 * (a[1] + a[3]) + 3 * a[2];
}

template<Operator op, Operand T>
static T differentiate(const std::array<T, 5>& a) noexcept {
    if constexpr (op == Operator::FDoG)
        return a[0] + a[1] - a[3] - a[4];
    else
        return 2 * (a[0] - a[4]) + a[1] - a[3];
}

static constexpr int truncatei(float value) noexcept {
    return static_cast<int>(value);
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

}  // namespace edgemasks

#endif  // VAPOURSYNTH_EDGEMASKS_KERNEL_H
