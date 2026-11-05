#define VCL_NAMESPACE vcl_sse4
#include <cstdint>

#include "vectorclass/vectorclass.h"

#include "edgemasks.h"
#include "edgemasks_kernel.h"
#include "edgemasks_simd.h"

using namespace VCL_NAMESPACE;

namespace edgemasks {

namespace {

struct Sse4 {
    using IntVector = Vec4i;
    using FloatVector = Vec4f;

    template<int... lane, Operand V>
    static V permute(const V& v) noexcept {
        return permute4<lane...>(v);
    }

    static IntVector load(const std::uint8_t* p) noexcept { return IntVector{}.load_4uc(p); }
    static IntVector load(const std::uint16_t* p) noexcept { return IntVector{}.load_4us(p); }
    static FloatVector load(const float* p) noexcept { return FloatVector{}.load(p); }

    static void store(const IntVector v, std::uint8_t* p) noexcept {
        compress_saturated_s2u(compress_saturated(v, zero_si128()), zero_si128()).store_si32(p);
    }

    static void store(const IntVector v, std::uint16_t* p) noexcept { compress_saturated_s2u(v, zero_si128()).storel(p); }
    static void store(const FloatVector v, float* p) noexcept { v.store(p); }
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
