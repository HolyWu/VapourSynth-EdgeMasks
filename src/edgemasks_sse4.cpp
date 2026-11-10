#define VCL_NAMESPACE vcl_sse4

#include <array>
#include <cstdint>

#include "vectorclass/vectorclass.h"

#include "edgemasks.h"
#include "edgemasks_simd.h"

using namespace VCL_NAMESPACE;

namespace edgemasks {

namespace {

struct Sse4 : VclIsa<Vec8s, Vec8us, Vec4i, Vec4f> {
    using VclIsa::load;
    using VclIsa::store;

    static Vec8s load(const std::uint8_t* p) noexcept { return Vec8s{}.load_8uc(p); }
    static Vec4i load(const std::uint16_t* p) noexcept { return Vec4i{}.load_4us(p); }

    static std::array<Vec4i, 2> squaredNorm(const Vec8s x, const Vec8s y) noexcept {
        const __m128i low{ _mm_unpacklo_epi16(x, y) };
        const __m128i high{ _mm_unpackhi_epi16(x, y) };
        return { _mm_madd_epi16(low, low), _mm_madd_epi16(high, high) };
    }

    static std::array<Vec4i, 2> widen(const Vec8s v) noexcept { return { _mm_unpacklo_epi16(v, zero_si128()), _mm_unpackhi_epi16(v, zero_si128()) }; }

    static void store(const Vec8s v, std::uint8_t* p) noexcept { compress_saturated_s2u(v, zero_si128()).storel(p); }

    static void store(const Vec4i low, const Vec4i high, std::uint8_t* p) noexcept { store(Vec8s{ _mm_packs_epi32(low, high) }, p); }
    static void store(const Vec4i low, const Vec4i high, std::uint16_t* p) noexcept { store(Vec8s{ _mm_packus_epi32(low, high) }, p); }

    static void store(const Vec4i v, std::uint8_t* p) noexcept { compress_saturated_s2u(compress_saturated(v, zero_si128()), zero_si128()).store_si32(p); }
    static void store(const Vec4i v, std::uint16_t* p) noexcept { compress_saturated_s2u(v, zero_si128()).storel(p); }
};

}  // namespace

template<Pixel pixel_t>
const FilterTable<pixel_t>& filtersSse4() noexcept {
    return simdFilters<Sse4, pixel_t>;
}

template const FilterTable<std::uint8_t>& filtersSse4<std::uint8_t>() noexcept;
template const FilterTable<std::uint16_t>& filtersSse4<std::uint16_t>() noexcept;
template const FilterTable<float>& filtersSse4<float>() noexcept;

}  // namespace edgemasks
