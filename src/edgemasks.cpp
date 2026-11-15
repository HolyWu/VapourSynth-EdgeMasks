/*
MIT License

Copyright (c) 2025 HolyWu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#ifdef EDGEMASKS_X86
#include <pmmintrin.h>
#include <xmmintrin.h>

#include "vectorclass/instrset.h"
#endif

#include "edgemasks.h"
#include "edgemasks_kernel.h"
#include "edgemasks_params.h"
#include "edgemasks_simd.h"

namespace edgemasks {

namespace {

// In the order of the opt argument's values, as parseOpt casts the integer straight to Opt.
enum class Opt { Auto, C, Sse4, Avx2, Avx512 };

template<Pixel pixel_t>
struct EdgeMasksData {
    FilterParams params;
    PlaneFilter<pixel_t> filter;
};

// Plain values, for the compiler to vectorize.
struct ScalarIsa {
    using Narrow = void;

    template<typename T>
    static auto load(const T* p) noexcept {
        return static_cast<Scalar<T>>(*p);
    }

    template<typename T, typename U>
    static void store(T value, U* p) noexcept {
        *p = static_cast<U>(value);
    }
};

template<Pixel pixel_t>
constexpr FilterTable<pixel_t> filtersC = withAllOperators([](auto... ops) {
    return FilterTable<pixel_t>{ filterPlane<ScalarIsa, pixel_t, Scalar<pixel_t>, decltype(ops)::value>... };
});

#ifdef EDGEMASKS_X86
// Flushes denormals to zero while it lives. Vectors that reach past a row's end compute on the stride padding,
// where a stale denormal costs Intel CPUs a microcode assist per instruction.
class FlushDenormals {
public:
    FlushDenormals() noexcept { _mm_setcsr(saved | _MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON); }
    FlushDenormals(const FlushDenormals&) = delete;
    FlushDenormals& operator=(const FlushDenormals&) = delete;
    FlushDenormals(FlushDenormals&&) = delete;
    FlushDenormals& operator=(FlushDenormals&&) = delete;
    ~FlushDenormals() { _mm_setcsr(saved); }

private:
    const unsigned saved{ _mm_getcsr() };
};

// instrset_detect() must not run concurrently, so every pixel type shares this one call.
int detectedInstrset() noexcept {
    static const int iset{ instrset_detect() };
    return iset;
}

// VapourSynth pads strides to 64 bytes only if AVX-512 was enabled when it loaded, and macOS enables it per thread on first use.
bool stridesFitAvx512(VSCore* core, const VSAPI* vsapi) noexcept {
    static const bool fits{ [=] {
        VSVideoFormat format{};
        vsapi->queryVideoFormat(&format, cfGray, stInteger, 8, 0, 0, core);
        const FramePtr probe{ vsapi->newVideoFrame(&format, 1, 1, nullptr, core), FrameDeleter{ vsapi } };
        return vsapi->getStride(probe.get(), 0) >= 64;
    }() };
    return fits;
}
#endif

template<Pixel pixel_t>
const FilterTable<pixel_t>& selectFilters(Opt opt, [[maybe_unused]] VSCore* core, [[maybe_unused]] const VSAPI* vsapi) {
#ifdef EDGEMASKS_X86
    constexpr int instrsetSse4{ 6 };
    constexpr int instrsetAvx2{ 8 };
    constexpr int instrsetAvx512{ 10 };
    constexpr std::array optInstrsets{ 0, 0, instrsetSse4, instrsetAvx2, instrsetAvx512 };

    const int iset{ detectedInstrset() };
    int target{ opt == Opt::Auto ? iset : optInstrsets[static_cast<std::size_t>(opt)] };

    if (target >= instrsetAvx512 && !stridesFitAvx512(core, vsapi))
        target = instrsetAvx2;

    if (iset < target)
        throw std::invalid_argument{ "opt selects an instruction set that the CPU does not support" };

    if (target >= instrsetAvx512)
        return filtersAvx512<pixel_t>();

    if (target >= instrsetAvx2)
        return filtersAvx2<pixel_t>();

    if (target >= instrsetSse4)
        return filtersSse4<pixel_t>();
#else
    if (opt > Opt::C)
        throw std::invalid_argument{ "opt selects an instruction set that the CPU does not support" };
#endif

    return filtersC<pixel_t>;
}

template<Pixel pixel_t>
void filterFrame(const EdgeMasksData<pixel_t>& d, const VSFrame* src, VSFrame* dst, const VSAPI* vsapi) {
    constexpr std::ptrdiff_t pixelSize{ sizeof(pixel_t) };
#ifdef EDGEMASKS_X86
    const FlushDenormals flushDenormals{};
#endif

    for (int plane = 0; plane < d.params.vi->format.numPlanes; plane++) {
        if (d.params.process[plane]) {
            d.filter({ .src = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(src, plane)),
                       .dst = reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, plane)),
                       .srcStride = vsapi->getStride(src, plane) / pixelSize,
                       .dstStride = vsapi->getStride(dst, plane) / pixelSize,
                       .width = vsapi->getFrameWidth(src, plane),
                       .height = vsapi->getFrameHeight(src, plane),
                       .scale = d.params.scale[plane],
                       .peak = d.params.peak });
        }
    }
}

template<Pixel pixel_t>
const VSFrame* VS_CC edgemasksGetFrame(int n, int activationReason, void* instanceData, void**, VSFrameContext* frameCtx, VSCore* core,
                                       const VSAPI* vsapi) noexcept {
    const auto& d = *static_cast<const EdgeMasksData<pixel_t>*>(instanceData);
    const auto& node = d.params.node;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, node.get(), frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const FramePtr src{ vsapi->getFrameFilter(n, node.get(), frameCtx), FrameDeleter{ vsapi } };
        NewFramePtr dst{ newSharingFrame(d.params, src.get(), core, vsapi) };

        try {
            filterFrame(d, src.get(), dst.get(), vsapi);
        } catch (const std::bad_alloc&) {
            vsapi->setFilterError((std::string{ name(d.params.op) } + ": out of memory").c_str(), frameCtx);
            return nullptr;
        }

        vsapi->mapSetInt(vsapi->getFramePropertiesRW(dst.get()), "_Range", VSC_RANGE_FULL, maReplace);
        return dst.release();
    }

    return nullptr;
}

PerPlane<bool> parsePlanes(const VSMap* in, const VSVideoFormat& format, const VSAPI* vsapi) {
    const int count{ vsapi->mapNumElements(in, "planes") };

    PerPlane<bool> process{};
    process.fill(count <= 0);

    for (int i = 0; i < count; i++) {
        const int n{ vsapi->mapGetIntSaturated(in, "planes", i, nullptr) };

        if (n < 0 || n >= format.numPlanes)
            throw std::invalid_argument{ "plane index out of range" };

        if (process[n])
            throw std::invalid_argument{ "plane specified twice" };

        process[n] = true;
    }

    return process;
}

PerPlane<float> parseScale(const VSMap* in, const VSVideoFormat& format, const VSAPI* vsapi) {
    if (vsapi->mapNumElements(in, "scale") > format.numPlanes)
        throw std::invalid_argument{ "scale has more values specified than there are planes" };

    PerPlane<float> scale{};

    for (int plane = 0; plane < format.numPlanes; plane++) {
        int err{};
        scale[plane] = vsapi->mapGetFloatSaturated(in, "scale", plane, &err);

        if (err)
            scale[plane] = plane == 0 ? 1.0f : scale[plane - 1];
        else if (!std::isfinite(scale[plane]) || scale[plane] <= 0.0f)
            throw std::invalid_argument{ "scale must be finite and greater than 0.0" };
    }

    return scale;
}

void checkDimensions(const VSVideoInfo& vi, PerPlane<bool> process, int minimum) {
    const VSVideoFormat& format{ vi.format };

    for (int plane = 0; plane < format.numPlanes; plane++) {
        if (process[plane]) {
            const auto [planeWidth, planeHeight] = planeDimensions(vi, plane);

            if (planeWidth < minimum)
                throw std::invalid_argument{ "plane " + std::to_string(plane) + "'s width must be greater than or equal to " + std::to_string(minimum) };

            if (planeHeight < minimum)
                throw std::invalid_argument{ "plane " + std::to_string(plane) + "'s height must be greater than or equal to " + std::to_string(minimum) };
        }
    }
}

Opt parseOpt(const VSMap* in, const VSAPI* vsapi) {
    int err{};
    const int opt{ vsapi->mapGetIntSaturated(in, "opt", 0, &err) };

    if (opt < 0 || opt > static_cast<int>(Opt::Avx512))
        throw std::invalid_argument{ "opt must be 0, 1, 2, 3, or 4" };

    return static_cast<Opt>(opt);
}

constexpr float scaleDivisor(Operator op) noexcept {
    switch (op) {
    case Operator::Scharr:
        return 3.0f;
    case Operator::RScharr:
        return 47.0f;
    case Operator::Kroon:
        return 17.0f;
    case Operator::FDoG:
        return 2.0f;
    default:
        return 1.0f;
    }
}

template<Pixel pixel_t>
void startFilter(FilterParams params, Opt opt, VSMap* out, VSCore* core, const VSAPI* vsapi) {
    const PlaneFilter<pixel_t> filter{ selectFilters<pixel_t>(opt, core, vsapi)[static_cast<std::size_t>(params.op)] };
    createNode(std::make_unique<EdgeMasksData<pixel_t>>(std::move(params), filter), edgemasksGetFrame<pixel_t>, 0, out, core, vsapi);
}

void createFilter(Operator op, const VSMap* in, VSMap* out, VSCore* core, const VSAPI* vsapi) noexcept {
    try {
        NodePtr node{ vsapi->mapGetNode(in, "clip", 0, nullptr), NodeDeleter{ vsapi } };
        const VSVideoInfo* const vi = vsapi->getVideoInfo(node.get());
        const VSVideoFormat& format{ vi->format };

        const bool gpu{ vsapi->getNodeResidency(node.get()) == nrGPU };

        if (!vsh::isConstantVideoFormat(vi) ||
            (format.sampleType == stInteger && format.bitsPerSample > 16) ||
            (format.sampleType == stFloat && format.bitsPerSample != 32 && !(gpu && format.bitsPerSample == 16)))
            throw std::invalid_argument{ "only 8-16 bit integer and 32 bit float, or 16 bit float on the GPU, with constant format and dimensions supported" };

        const auto process = parsePlanes(in, format, vsapi);
        auto scale = parseScale(in, format, vsapi);

        checkDimensions(*vi, process, windowSize(op));

        const Opt opt{ parseOpt(in, vsapi) };

        for (float& value : scale)
            value /= scaleDivisor(op);

        const float peak{ format.sampleType == stInteger ? static_cast<float>((1u << format.bitsPerSample) - 1u) : 0.0f };
        FilterParams params{ .node = std::move(node), .vi = vi, .process = process, .scale = scale, .peak = peak, .op = op };

        if (gpu) {
            createVulkanFilter(std::move(params), out, core, vsapi);
            return;
        }

        switch (format.bytesPerSample) {
        case 1:
            startFilter<std::uint8_t>(std::move(params), opt, out, core, vsapi);
            break;
        case 2:
            startFilter<std::uint16_t>(std::move(params), opt, out, core, vsapi);
            break;
        default:
            startFilter<float>(std::move(params), opt, out, core, vsapi);
        }
    } catch (const std::exception& error) {
        vsapi->mapSetError(out, (std::string{ name(op) } + ": " + error.what()).c_str());
    }
}

template<Operator op>
void VS_CC edgemasksCreate(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) noexcept {
    createFilter(op, in, out, core, vsapi);
}

}  // namespace

}  // namespace edgemasks

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) noexcept {
    using namespace edgemasks;

    vspapi->configPlugin("com.holywu.edgemasks",
                         "edgemasks",
                         "Creates an edge mask using various operators",
                         VS_MAKE_VERSION(EDGEMASKS_VERSION_MAJOR, EDGEMASKS_VERSION_MINOR),
                         VAPOURSYNTH_API_VERSION,
                         0,
                         plugin);

    withAllOperators([&](auto... ops) {
        (vspapi->registerFunction(name(decltype(ops)::value),
                                  "clip:vnode:all;planes:int[]:opt;scale:float[]:opt;opt:int:opt;",
                                  "clip:vnode:all;",
                                  edgemasksCreate<decltype(ops)::value>,
                                  nullptr,
                                  plugin), ...);
    });
}
