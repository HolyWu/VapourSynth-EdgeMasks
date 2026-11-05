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

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#ifdef EDGEMASKS_X86
#include "vectorclass/instrset.h"
#endif

#include "edgemasks.h"
#include "edgemasks_kernel.h"

namespace edgemasks {

namespace {

// In the order of the opt argument's values, as parseOpt casts the integer straight to Opt.
enum class Opt { Auto, C, Sse4, Avx2, Avx512 };

template<typename T>
using PerPlane = std::array<T, 3>;

struct NodeDeleter {
    const VSAPI* vsapi;
    void operator()(VSNode* node) const noexcept { vsapi->freeNode(node); }
};

struct FrameDeleter {
    const VSAPI* vsapi;
    void operator()(const VSFrame* frame) const noexcept { vsapi->freeFrame(frame); }
};

using NodePtr = std::unique_ptr<VSNode, NodeDeleter>;
using FramePtr = std::unique_ptr<const VSFrame, FrameDeleter>;

struct FilterParams {
    NodePtr node;
    const VSVideoInfo* vi;
    PerPlane<bool> process;
    PerPlane<float> scale;
    float peak;
};

template<Pixel pixel_t>
struct EdgeMasksData {
    FilterParams params;
    PlaneFilter<pixel_t> filter;
};

// Out of line and flattened so that its loop holds nothing but the formulas, the shape that compilers vectorize.
template<Pixel pixel_t, Operator op, std::invocable<int> ColumnOf>
[[gnu::noinline, gnu::flatten]] void filterColumns(const Plane<pixel_t>& plane, int y, int begin, int end, ColumnOf columnOf) noexcept {
    using Columns = std::array<Scalar<pixel_t>, windowSize(op)>;

    const auto [src, dst, srcStride, dstStride, width, height, scale, peak] = plane;
    const auto rows = windowRows(plane, y);
    pixel_t* const dstRow = dst + y * dstStride;

    for (int x = begin; x < end; x++) {
        const auto rowOf = [&](const pixel_t* row) noexcept {
            if constexpr (windowSize(op) == 3)
                return Columns{ row[columnOf(x - 1)], row[x], row[columnOf(x + 1)] };
            else
                return Columns{ row[columnOf(x - 2)], row[columnOf(x - 1)], row[x], row[columnOf(x + 1)], row[columnOf(x + 2)] };
        };

        const float scaled{ magnitude(response<op>(makeWindow<windowSize(op)>(rows, rowOf))) * scale };

        if constexpr (std::integral<pixel_t>)
            dstRow[x] = static_cast<pixel_t>(std::min(scaled + 0.5f, peak));
        else
            dstRow[x] = scaled;
    }
}

template<Pixel pixel_t, Operator op>
void filterC(const Plane<pixel_t>& plane) noexcept {
    constexpr int radius{ windowSize(op) / 2 };

    const auto mirrored = [width = plane.width](int x) noexcept { return mirror(x, width); };

    for (int y = 0; y < plane.height; y++) {
        filterColumns<pixel_t, op>(plane, y, 0, radius, mirrored);
        filterColumns<pixel_t, op>(plane, y, radius, plane.width - radius, std::identity{});
        filterColumns<pixel_t, op>(plane, y, plane.width - radius, plane.width, mirrored);
    }
}

template<Pixel pixel_t>
constexpr FilterTable<pixel_t> filtersC = withAllOperators([](auto... ops) { return FilterTable<pixel_t>{ filterC<pixel_t, decltype(ops)::value>... }; });

#ifdef EDGEMASKS_X86
// instrset_detect() must not run concurrently, so every pixel type shares this one call.
int detectedInstrset() noexcept {
    static const int iset{ instrset_detect() };
    return iset;
}
#endif

template<Pixel pixel_t>
const FilterTable<pixel_t>& selectFilters([[maybe_unused]] Opt opt) noexcept {
#ifdef EDGEMASKS_X86
    constexpr int instrsetSse4{ 6 };
    constexpr int instrsetAvx2{ 8 };
    constexpr int instrsetAvx512{ 10 };

    const int iset{ detectedInstrset() };

    if ((opt == Opt::Auto && iset >= instrsetAvx512) || opt == Opt::Avx512)
        return filtersAvx512<pixel_t>();

    if ((opt == Opt::Auto && iset >= instrsetAvx2) || opt == Opt::Avx2)
        return filtersAvx2<pixel_t>();

    if ((opt == Opt::Auto && iset >= instrsetSse4) || opt == Opt::Sse4)
        return filtersSse4<pixel_t>();
#endif

    return filtersC<pixel_t>;
}

template<Pixel pixel_t>
void filterFrame(const EdgeMasksData<pixel_t>& d, const VSFrame* src, VSFrame* dst, const VSAPI* vsapi) noexcept {
    constexpr std::ptrdiff_t pixelSize{ sizeof(pixel_t) };

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
        PerPlane<const VSFrame*> planeSrc{ process[0] ? nullptr : src.get(), process[1] ? nullptr : src.get(), process[2] ? nullptr : src.get() };
        constexpr std::array planes{ 0, 1, 2 };
        VSFrame* const dst = vsapi->newVideoFrame2(&vi->format, vi->width, vi->height, planeSrc.data(), planes.data(), src.get(), core);

        filterFrame(d, src.get(), dst, vsapi);

        vsapi->mapSetInt(vsapi->getFramePropertiesRW(dst), "_Range", VSC_RANGE_FULL, maReplace);
        return dst;
    }

    return nullptr;
}

template<Pixel pixel_t>
void VS_CC edgemasksFree(void* instanceData, VSCore*, const VSAPI*) noexcept {
    const std::unique_ptr<EdgeMasksData<pixel_t>> d{ static_cast<EdgeMasksData<pixel_t>*>(instanceData) };
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
            const int planeWidth{ vi.width >> (plane > 0 ? format.subSamplingW : 0) };
            const int planeHeight{ vi.height >> (plane > 0 ? format.subSamplingH : 0) };

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
void startFilter(FilterParams params, Operator op, Opt opt, VSMap* out, VSCore* core, const VSAPI* vsapi) {
    const PlaneFilter<pixel_t> filter{ selectFilters<pixel_t>(opt)[static_cast<std::size_t>(op)] };
    auto d = std::make_unique<EdgeMasksData<pixel_t>>(std::move(params), filter);
    const VSFilterDependency dependency{ d->params.node.get(), rpStrictSpatial };

    vsapi->createVideoFilter(out, name(op), d->params.vi, edgemasksGetFrame<pixel_t>, edgemasksFree<pixel_t>, fmParallel, &dependency, 1, d.get(), core);
    // The core owns d now and releases it through edgemasksFree. It can only refuse an invalid vi, and this vi is the input clip's.
    std::ignore = d.release();
}

void createFilter(Operator op, const VSMap* in, VSMap* out, VSCore* core, const VSAPI* vsapi) noexcept {
    try {
        NodePtr node{ vsapi->mapGetNode(in, "clip", 0, nullptr), NodeDeleter{ vsapi } };
        const VSVideoInfo* const vi = vsapi->getVideoInfo(node.get());
        const VSVideoFormat& format{ vi->format };

        if (!vsh::isConstantVideoFormat(vi) ||
            (format.sampleType == stInteger && format.bitsPerSample > 16) ||
            (format.sampleType == stFloat && format.bitsPerSample != 32))
            throw std::invalid_argument{ "only 8-16 bit integer and 32 bit float with constant format and dimensions supported" };

        const auto process = parsePlanes(in, format, vsapi);
        auto scale = parseScale(in, format, vsapi);

        checkDimensions(*vi, process, windowSize(op));

        const Opt opt{ parseOpt(in, vsapi) };

        for (float& value : scale)
            value /= scaleDivisor(op);

        const float peak{ format.sampleType == stInteger ? static_cast<float>((1u << format.bitsPerSample) - 1u) : 0.0f };
        FilterParams params{ .node = std::move(node), .vi = vi, .process = process, .scale = scale, .peak = peak };

        switch (format.bytesPerSample) {
        case 1:
            startFilter<std::uint8_t>(std::move(params), op, opt, out, core, vsapi);
            break;
        case 2:
            startFilter<std::uint16_t>(std::move(params), op, opt, out, core, vsapi);
            break;
        default:
            startFilter<float>(std::move(params), op, opt, out, core, vsapi);
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
                                  "clip:vnode;planes:int[]:opt;scale:float[]:opt;opt:int:opt;",
                                  "clip:vnode;",
                                  edgemasksCreate<decltype(ops)::value>,
                                  nullptr,
                                  plugin), ...);
    });
}
