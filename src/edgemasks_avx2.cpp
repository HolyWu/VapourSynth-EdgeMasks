#define VCL_NAMESPACE vcl_avx2
#include <cstdint>

#include "vectorclass/vectorclass.h"

#include "edgemasks.h"
#include "edgemasks_kernel.h"
#include "edgemasks_simd.h"

using namespace VCL_NAMESPACE;

namespace edgemasks {

namespace {

struct Avx2 {
    using IntVector = Vec8i;
    using FloatVector = Vec8f;

    template<int... lane, Operand V>
    static V permute(const V& v) noexcept {
        return permute8<lane...>(v);
    }

    static IntVector load(const std::uint8_t* p) noexcept { return IntVector{}.load_8uc(p); }
    static IntVector load(const std::uint16_t* p) noexcept { return IntVector{}.load_8us(p); }
    static FloatVector load(const float* p) noexcept { return FloatVector{}.load(p); }

    static void store(const IntVector v, std::uint8_t* p) noexcept {
        compress_saturated_s2u(compress_saturated(v, zero_si256()), zero_si256()).get_low().storel(p);
    }

    static void store(const IntVector v, std::uint16_t* p) noexcept { compress_saturated_s2u(v, zero_si256()).get_low().store(p); }
    static void store(const FloatVector v, float* p) noexcept { v.store(p); }
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
