#define VCL_NAMESPACE vcl_avx2

#include <array>
#include <cstdint>

#include "vectorclass/vectorclass.h"

#include "edgemasks.h"
#include "edgemasks_simd.h"

using namespace VCL_NAMESPACE;

namespace edgemasks {

namespace {

struct Avx2 : VclIsa<Vec16s, Vec16us, Vec8i, Vec8f> {
    using VclIsa::load;
    using VclIsa::store;

    static Vec16s load(const std::uint8_t* p) noexcept { return Vec16s{}.load_16uc(p); }
    static Vec8i load(const std::uint16_t* p) noexcept { return Vec8i{}.load_8us(p); }

    // These split within each 128-bit lane, which the two-vector stores undo.
    static std::array<Vec8i, 2> squaredNorm(const Vec16s x, const Vec16s y) noexcept {
        const __m256i low{ _mm256_unpacklo_epi16(x, y) };
        const __m256i high{ _mm256_unpackhi_epi16(x, y) };
        return { _mm256_madd_epi16(low, low), _mm256_madd_epi16(high, high) };
    }

    static std::array<Vec8i, 2> widen(const Vec16s v) noexcept { return { _mm256_unpacklo_epi16(v, zero_si256()), _mm256_unpackhi_epi16(v, zero_si256()) }; }

    static void store(const Vec16s v, std::uint8_t* p) noexcept { compress_saturated_s2u(v.get_low(), v.get_high()).store(p); }

    static void store(const Vec8i low, const Vec8i high, std::uint8_t* p) noexcept { store(Vec16s{ _mm256_packs_epi32(low, high) }, p); }
    static void store(const Vec8i low, const Vec8i high, std::uint16_t* p) noexcept { store(Vec16s{ _mm256_packus_epi32(low, high) }, p); }

    static void store(const Vec8i v, std::uint8_t* p) noexcept { compress_saturated_s2u(compress_saturated(v), zero_si128()).storel(p); }
    static void store(const Vec8i v, std::uint16_t* p) noexcept { compress_saturated_s2u(v.get_low(), v.get_high()).store(p); }
};

}  // namespace

template<Pixel pixel_t>
const FilterTable<pixel_t>& filtersAvx2() noexcept {
    return simdFilters<Avx2, pixel_t>;
}

template const FilterTable<std::uint8_t>& filtersAvx2<std::uint8_t>() noexcept;
template const FilterTable<std::uint16_t>& filtersAvx2<std::uint16_t>() noexcept;
template const FilterTable<float>& filtersAvx2<float>() noexcept;

}  // namespace edgemasks
