#define VCL_NAMESPACE vcl_avx512

#include <array>
#include <cstdint>

#include "vectorclass/vectorclass.h"

#include "edgemasks.h"
#include "edgemasks_simd.h"

using namespace VCL_NAMESPACE;

namespace edgemasks {

namespace {

struct Avx512 : VclIsa<Vec32s, Vec32us, Vec16i, Vec16f> {
    using VclIsa::load;
    using VclIsa::store;

    static Vec32s load(const std::uint8_t* p) noexcept { return Vec32s{}.load_32uc(p); }
    static Vec16i load(const std::uint16_t* p) noexcept { return Vec16i{}.load_16us(p); }

    // These split within each 128-bit lane, which the two-vector stores undo.
    static std::array<Vec16i, 2> squaredNorm(const Vec32s x, const Vec32s y) noexcept {
        const __m512i low{ _mm512_unpacklo_epi16(x, y) };
        const __m512i high{ _mm512_unpackhi_epi16(x, y) };
        return { _mm512_madd_epi16(low, low), _mm512_madd_epi16(high, high) };
    }

    static std::array<Vec16i, 2> widen(const Vec32s v) noexcept { return { _mm512_unpacklo_epi16(v, zero_si512()), _mm512_unpackhi_epi16(v, zero_si512()) }; }

    static void store(const Vec32s v, std::uint8_t* p) noexcept { compress_saturated(Vec32us{ v }).store(p); }

    static void store(const Vec16i low, const Vec16i high, std::uint8_t* p) noexcept { store(Vec32s{ _mm512_packs_epi32(low, high) }, p); }
    static void store(const Vec16i low, const Vec16i high, std::uint16_t* p) noexcept { store(Vec32s{ _mm512_packus_epi32(low, high) }, p); }

    static void store(const Vec16i v, std::uint8_t* p) noexcept { Vec16uc{ _mm512_cvtusepi32_epi8(v) }.store(p); }
    static void store(const Vec16i v, std::uint16_t* p) noexcept { compress(v).store(p); }
};

}  // namespace

template<Pixel pixel_t>
const FilterTable<pixel_t>& filtersAvx512() noexcept {
    return simdFilters<Avx512, pixel_t>;
}

template const FilterTable<std::uint8_t>& filtersAvx512<std::uint8_t>() noexcept;
template const FilterTable<std::uint16_t>& filtersAvx512<std::uint16_t>() noexcept;
template const FilterTable<float>& filtersAvx512<float>() noexcept;

}  // namespace edgemasks
