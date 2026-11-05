#ifndef VAPOURSYNTH_EDGEMASKS_H
#define VAPOURSYNTH_EDGEMASKS_H

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
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
using PlaneFilter = void (*)(const Plane<pixel_t>& plane) noexcept;

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
