#define VCL_NAMESPACE vcl_avx512
#include <cstdint>

#include "vectorclass/vectorclass.h"

#include "edgemasks.h"
#include "edgemasks_kernel.h"
#include "edgemasks_simd.h"

using namespace VCL_NAMESPACE;

namespace edgemasks {

namespace {

struct Avx512 {
    using IntVector = Vec16i;
    using FloatVector = Vec16f;

    template<int... lane, Operand V>
    static V permute(const V& v) noexcept {
        return permute16<lane...>(v);
    }

    static IntVector load(const std::uint8_t* p) noexcept { return IntVector{}.load_16uc(p); }
    static IntVector load(const std::uint16_t* p) noexcept { return IntVector{}.load_16us(p); }
    static FloatVector load(const float* p) noexcept { return FloatVector{}.load(p); }

    static void store(const IntVector v, std::uint8_t* p) noexcept {
        compress_saturated_s2u(compress_saturated(v, zero_si512()), zero_si512()).get_low().get_low().store(p);
    }

    static void store(const IntVector v, std::uint16_t* p) noexcept { compress_saturated_s2u(v, zero_si512()).get_low().store(p); }
    static void store(const FloatVector v, float* p) noexcept { v.store(p); }
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
