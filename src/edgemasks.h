#ifndef VAPOURSYNTH_EDGEMASKS_H
#define VAPOURSYNTH_EDGEMASKS_H

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace edgemasks {

enum class Operator { Tritical, Cross, Prewitt, Sobel, Scharr, RScharr, Kroon, Robinson3, Robinson5, Kirsch, ExPrewitt, ExSobel, FDoG, ExKirsch, Count };

// In enumerator order.
inline constexpr std::array operatorNames{
    "Tritical", "Cross", "Prewitt", "Sobel", "Scharr", "RScharr", "Kroon", "Robinson3", "Robinson5", "Kirsch", "ExPrewitt", "ExSobel", "FDoG", "ExKirsch"
};

static_assert(operatorNames.size() == static_cast<std::size_t>(Operator::Count));

constexpr const char* name(Operator op) noexcept {
    return operatorNames[static_cast<std::size_t>(op)];
}

constexpr int windowSize(Operator op) noexcept {
    return op >= Operator::ExPrewitt ? 5 : 3;
}

// The sum of the operator's positive weights, which bounds |response| per unit of pixel value.
static constexpr int gain(Operator op) noexcept {
    constexpr std::array gains{ 1, 1, 3, 4, 16, 256, 95, 3, 4, 15, 15, 18, 18, 78 };
    static_assert(gains.size() == static_cast<std::size_t>(Operator::Count));
    return gains[static_cast<std::size_t>(op)];
}

// Bounded by peak, so a sample above it can wrap the lanes. That spoils the mask around the sample, but the output stays within peak.
static constexpr bool fitsNarrow(Operator op, float peak) noexcept {
    return gain(op) * peak <= std::numeric_limits<std::int16_t>::max();
}

template<typename T>
concept Pixel = std::same_as<T, std::uint8_t> || std::same_as<T, std::uint16_t> || std::same_as<T, float>;

// Width and height are at least the operator's window size.
// Strides count pixels and are padded to a whole SIMD vector, because vector loads and stores run past width into that padding.
template<Pixel pixel_t>
struct Plane {
    const pixel_t* src;
    pixel_t* dst;
    std::ptrdiff_t srcStride;
    std::ptrdiff_t dstStride;
    int width;
    int height;
    float scale;
    float peak;
};

template<Pixel pixel_t>
using PlaneFilter = void (*)(const Plane<pixel_t>& plane);

// Indexed by Operator.
template<Pixel pixel_t>
using FilterTable = std::array<PlaneFilter<pixel_t>, operatorNames.size()>;

template<Pixel pixel_t>
const FilterTable<pixel_t>& filtersSse4() noexcept;

template<Pixel pixel_t>
const FilterTable<pixel_t>& filtersAvx2() noexcept;

template<Pixel pixel_t>
const FilterTable<pixel_t>& filtersAvx512() noexcept;

// Calls fn with every Operator, each as its own std::integral_constant.
template<typename Fn>
static constexpr auto withAllOperators(Fn fn) {
    return [&]<std::size_t... index>(std::index_sequence<index...>) {
        return fn(std::integral_constant<Operator, static_cast<Operator>(index)>{}...);
    }(std::make_index_sequence<operatorNames.size()>());
}

}  // namespace edgemasks

#endif  // VAPOURSYNTH_EDGEMASKS_H
