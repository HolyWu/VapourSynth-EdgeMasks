#ifndef VAPOURSYNTH_EDGEMASKS_SIMD_H
#define VAPOURSYNTH_EDGEMASKS_SIMD_H

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <utility>

#include "edgemasks.h"
#include "edgemasks_kernel.h"

namespace edgemasks {

// load reads and store writes exactly as many pixels as the vector has lanes.
template<typename Isa, typename pixel_t>
concept VectorIo = requires(const pixel_t* src, pixel_t* dst) {
    { Isa::load(src) } -> Operand;
    { decltype(Isa::load(src))::size() } -> std::convertible_to<int>;
    Isa::store(Isa::load(src), dst);
};

// One instruction set: load and store for every pixel type.
// Not checked: its permute, and the vector math functions found by argument-dependent lookup.
template<typename Isa>
concept SimdIsa = VectorIo<Isa, std::uint8_t> && VectorIo<Isa, std::uint16_t> && VectorIo<Isa, float>;

// Lane i of the result is lane mirror(i + shift, lanes) of v.
template<SimdIsa Isa, int shift, Operand V>
static V mirrorShift(const V& v) noexcept {
    return [&]<int... lane>(std::integer_sequence<int, lane...>) noexcept {
        return Isa::template permute<mirror(lane + shift, V::size())...>(v);
    }(std::make_integer_sequence<int, V::size()>());
}

// Flattened because GCC otherwise leaves the larger formulas out of line, up to 20% slower at 1080p, and fuses Tritical's float squares differently.
template<SimdIsa Isa, Operator op, Pixel pixel_t, std::invocable<const pixel_t*> RowOf>
[[gnu::flatten]] static void filterVector(const WindowRows<pixel_t>& rows, RowOf rowOf, pixel_t* dst, float scale, float peak) noexcept {
    const auto scaled = edgemasks::magnitude(edgemasks::response<op>(makeWindow<windowSize(op)>(rows, rowOf))) * scale;

    if constexpr (std::integral<pixel_t>)
        Isa::store(truncatei(min(scaled + 0.5f, peak)), dst);
    else
        Isa::store(scaled, dst);
}

template<SimdIsa Isa, Pixel pixel_t, Operator op>
static void filterPlane(const Plane<pixel_t>& plane) noexcept {
    using V = decltype(Isa::load(plane.src));
    using Columns = std::array<V, windowSize(op)>;

    constexpr int lanes{ V::size() };
    const auto [src, dst, srcStride, dstStride, width, height, scale, peak] = plane;

    const int regularPart{ (width - 1) / lanes * lanes };
    // When width % lanes == 1, the tail also rewrites column width - 2, which the previous vector computes from padding for 5x5.
    const int tailStart{ std::min(regularPart, width - 2) };

    const auto laneIs = [&](int column) noexcept {
        const V laneIndices = []<int... lane>(std::integer_sequence<int, lane...>) noexcept { return V{ lane... }; }(std::make_integer_sequence<int, lanes>());

        return laneIndices == V{ static_cast<Scalar<pixel_t>>(column - tailStart) };
    };

    const auto isLast = laneIs(width - 1);
    const auto isSecondLast = laneIs(width - 2);

    const auto load = [](const pixel_t* row, int x) noexcept { return Isa::load(row + x); };

    const auto mirrorRight1 = [&](const V& left1, const V& center) noexcept { return select(isLast, left1, edgemasks::mirrorShift<Isa, 1>(center)); };

    const auto mirrorRight2 = [&](const V& left2, const V& center) noexcept {
        return select(isLast, left2, select(isSecondLast, center, edgemasks::mirrorShift<Isa, 2>(center)));
    };

    const auto firstColumns = [&](const pixel_t* row) noexcept {
        const V center = load(row, 0);
        const V left1 = edgemasks::mirrorShift<Isa, -1>(center);
        const V right1 = width > lanes ? load(row, 1) : mirrorRight1(left1, center);

        if constexpr (windowSize(op) == 3) {
            return Columns{ left1, center, right1 };
        } else {
            const V left2 = edgemasks::mirrorShift<Isa, -2>(center);
            const V right2 = width > lanes ? load(row, 2) : mirrorRight2(left2, center);
            return Columns{ left2, left1, center, right1, right2 };
        }
    };

    const auto regularColumns = [&](const pixel_t* row, int x) noexcept {
        if constexpr (windowSize(op) == 3)
            return Columns{ load(row, x - 1), load(row, x), load(row, x + 1) };
        else
            return Columns{ load(row, x - 2), load(row, x - 1), load(row, x), load(row, x + 1), load(row, x + 2) };
    };

    const auto tailColumns = [&](const pixel_t* row) noexcept {
        const V left1 = load(row, tailStart - 1);
        const V center = load(row, tailStart);
        const V right1 = mirrorRight1(left1, center);

        if constexpr (windowSize(op) == 3) {
            return Columns{ left1, center, right1 };
        } else {
            const V left2 = load(row, tailStart - 2);
            return Columns{ left2, left1, center, right1, mirrorRight2(left2, center) };
        }
    };

    for (int y = 0; y < height; y++) {
        const auto rows = windowRows(plane, y);
        pixel_t* const dstRow = dst + y * dstStride;

        filterVector<Isa, op>(rows, firstColumns, dstRow, scale, peak);

        for (int x = lanes; x < regularPart; x += lanes)
            filterVector<Isa, op>(rows, [&](const pixel_t* row) noexcept { return regularColumns(row, x); }, dstRow + x, scale, peak);

        if (regularPart >= lanes)
            filterVector<Isa, op>(rows, tailColumns, dstRow + tailStart, scale, peak);
    }
}

template<SimdIsa Isa, Pixel pixel_t>
static constexpr FilterTable<pixel_t> simdFilters = withAllOperators([](auto... ops) {
    return FilterTable<pixel_t>{ filterPlane<Isa, pixel_t, decltype(ops)::value>... };
});

}  // namespace edgemasks

#endif  // VAPOURSYNTH_EDGEMASKS_SIMD_H
