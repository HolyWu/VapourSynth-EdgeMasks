#ifndef VAPOURSYNTH_EDGEMASKS_SIMD_H
#define VAPOURSYNTH_EDGEMASKS_SIMD_H

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "edgemasks.h"
#include "edgemasks_kernel.h"

namespace edgemasks {

// load reads and store writes exactly as many elements as the vector has lanes.
template<typename Isa, typename work_t>
concept WorkIo = requires(const work_t* src, work_t* dst) {
    { Isa::load(src) } -> Operand;
    Isa::store(Isa::load(src), dst);
};

// One instruction set, or plain values for the compiler to vectorize. Isa::Narrow is its vector of 16-bit lanes, or void if it has none.
template<typename Isa>
concept KernelIsa = WorkIo<Isa, std::int32_t> && WorkIo<Isa, float> && requires { typename Isa::Narrow; };

// The members that the vectorclass instruction sets spell alike. Each instruction set's own vectorclass namespace keeps their code apart.
template<typename NarrowVector, typename NarrowUnsignedVector, typename WideVector, typename FloatVector>
struct VclIsa {
    using Narrow = NarrowVector;
    static constexpr bool u8StoresSaturate{ true };

    static Narrow load(const std::int16_t* p) noexcept { return Narrow{}.load(p); }
    static WideVector load(const std::int32_t* p) noexcept { return WideVector{}.load(p); }
    static FloatVector load(const float* p) noexcept { return FloatVector{}.load(p); }

    static Narrow minUnsigned(const Narrow a, const Narrow b) noexcept { return min(NarrowUnsignedVector{ a }, NarrowUnsignedVector{ b }); }

    static void store(const Narrow v, std::int16_t* p) noexcept { v.store(p); }
    static void store(const WideVector v, std::int32_t* p) noexcept { v.store(p); }
    static void store(const Narrow v, std::uint16_t* p) noexcept { v.store(p); }
    static void store(const FloatVector v, float* p) noexcept { v.store(p); }
};

template<KernelIsa Isa, typename work_t>
using Vector = decltype(Isa::load(static_cast<const work_t*>(nullptr)));

template<KernelIsa Isa, typename work_t>
inline constexpr int laneCount{ [] {
    if constexpr (std::is_arithmetic_v<Vector<Isa, work_t>>)
        return 1;
    else
        return Vector<Isa, work_t>::size();
}() };

// Rounded up to whole vectors, with a margin on both sides that is itself a whole number of cache lines.
// Isa keeps the member functions of each instruction set apart, as this header's functions are.
template<KernelIsa Isa, typename T>
class LineBuffers {
public:
    LineBuffers(int width, int count)
        : length{ margin + (width + lanes - 1) / lanes * lanes + margin }, storage{ std::make_unique_for_overwrite<T[]>(length * count + margin) } {
        void* aligned = storage.get();
        std::size_t space{ static_cast<std::size_t>(length * count + margin) * sizeof(T) };
        std::align(alignment, static_cast<std::size_t>(length * count) * sizeof(T), aligned, space);
        first = static_cast<T*>(aligned) + margin;
    }

    T* line(int index) const noexcept { return first + index * length; }

private:
    static constexpr int lanes{ laneCount<Isa, T> };
    static constexpr std::size_t alignment{ 64 };
    static constexpr std::ptrdiff_t margin{ alignment / sizeof(T) };

    std::ptrdiff_t length;
    std::unique_ptr<T[]> storage;
    T* first;
};

// Reflects the columns outside [0, width) without repeating the edge column.
template<int radius, typename T>
static void mirrorMargins(T* line, int width) noexcept {
    for (int x = 1; x <= radius; x++) {
        line[-x] = line[x];
        line[width - 1 + x] = line[width - 1 - x];
    }
}

// What a kernel loads from the frame: work_t when that has the pixel's size, else the pixel, which the load widens.
template<typename work_t, Pixel pixel_t>
using FrameSample = std::conditional_t<sizeof(work_t) == sizeof(pixel_t), work_t, pixel_t>;

// Widening a whole row once beats widening every window tap, unless there are as few taps as in Tritical and Cross.
// Separable operators also measured faster reading 16-bit pixels from the frame, but slower reading 8-bit ones.
template<typename work_t, Pixel pixel_t, Operator op>
inline constexpr bool readsFrame{ sizeof(work_t) == sizeof(pixel_t) ||
                                  (sizeof(work_t) == 2 * sizeof(pixel_t) &&
                                   (op == Operator::Tritical || op == Operator::Cross || (isSeparable(op) && sizeof(pixel_t) == 2))) };

template<typename work_t, Pixel pixel_t>
static WindowRows<FrameSample<work_t, pixel_t>> frameRows(const Plane<pixel_t>& plane, int y) noexcept {
    const auto row = [&](int offset) noexcept {
        return reinterpret_cast<const FrameSample<work_t, pixel_t>*>(plane.src + mirror(y + offset, plane.height) * plane.srcStride);
    };

    return { row(-2), row(-1), row(0), row(1), row(2) };
}

// A ring of the rows in the window, widened to work_t and with mirrored margins.
// When the kernel reads the frame, the rows hold only the columns that the vectors reaching past the plane's sides need.
template<KernelIsa Isa, Pixel pixel_t, typename work_t, Operator op>
class RowRing {
public:
    static constexpr int interiorBegin{ std::max(laneCount<Isa, work_t>, windowSize(op) / 2) };

    explicit RowRing(const Plane<pixel_t>& plane) : plane{ &plane }, lines{ plane.width, size } {
        for (int y = 0; y < radius; y++)
            bufferRow(y);
    }

    // Must be called for each row in turn, because it buffers the row that enters the window.
    WindowRows<work_t> rows(int y) noexcept {
        if (y + radius < plane->height)
            bufferRow(y + radius);

        const auto row = [&](int offset) noexcept { return lines.line(mirror(y + offset, plane->height) % size); };
        return { row(-2), row(-1), row(0), row(1), row(2) };
    }

    // From this column on, a vector reaches beyond the row's end.
    int interiorEnd() const noexcept { return (plane->width - radius) / lanes * lanes; }

private:
    static constexpr int size{ windowSize(op) };
    static constexpr int radius{ size / 2 };
    static constexpr int lanes{ laneCount<Isa, work_t> };

    void bufferRow(int y) noexcept {
        const pixel_t* const source = plane->src + y * plane->srcStride;
        work_t* const line = lines.line(y % size);
        // A local, because a loop that rereads plane->width does not vectorize.
        const int width{ plane->width };

        if constexpr (readsFrame<work_t, pixel_t, op>) {
            const auto copyVectors = [&](int begin, int end) noexcept {
                for (int x = begin; x < end; x += lanes)
                    Isa::store(Isa::load(reinterpret_cast<const FrameSample<work_t, pixel_t>*>(source) + x), line + x);
            };

            copyVectors(0, std::min(width, interiorBegin + radius));
            copyVectors(std::max((interiorEnd() - radius) / lanes * lanes, 0), width);
        } else {
            for (int x = 0; x < width; x++)
                line[x] = source[x];
        }

        mirrorMargins<radius>(line, width);
    }

    const Plane<pixel_t>* plane;
    LineBuffers<Isa, work_t> lines;
};

// An Isa whose stores to u8 saturate declares u8StoresSaturate true, which spares u8 the clamp to peak.
template<typename Isa, Pixel pixel_t>
inline constexpr bool storeSaturates{ std::same_as<pixel_t, std::uint8_t> && requires { requires Isa::u8StoresSaturate; } };

template<KernelIsa Isa, Pixel pixel_t>
struct Output {
    float scale;
    float peak;

    auto quantize(const auto& value) const noexcept {
        using std::min;

        if constexpr (storeSaturates<Isa, pixel_t>)
            // From this scale up every nonzero magnitude saturates too, and the conversion cannot overflow.
            return truncatei(value * (scale < 256.0f ? scale : 256.0f) + 0.5f);
        else
            return truncatei(min(value * scale + 0.5f, peak));
    }

    template<Operand V>
    void operator()(const Gradient<V>& gradient, pixel_t* dst) const noexcept {
        if constexpr (std::same_as<V, typename Isa::Narrow>) {
            const auto [low, high] = Isa::squaredNorm(gradient.x, gradient.y);
            Isa::store(quantize(sqrt(to_float(low))), quantize(sqrt(to_float(high))), dst);
        } else {
            (*this)(magnitude(gradient), dst);
        }
    }

    template<Operand V>
    void operator()(const V& maximum, pixel_t* dst) const noexcept {
        if constexpr (std::floating_point<pixel_t>) {
            Isa::store(maximum * scale, dst);
        } else if constexpr (std::same_as<V, typename Isa::Narrow>) {
            const auto [low, high] = Isa::widen(maximum);
            Isa::store(quantize(to_float(low)), quantize(to_float(high)), dst);
        } else {
            Isa::store(quantize(toFloat(maximum)), dst);
        }
    }
};

// For scale 1, a maximum needs no float stage.
template<KernelIsa Isa, Pixel pixel_t>
struct UnitScaleOutput {
    float peak;

    template<Operand V>
    void operator()(const V& maximum, pixel_t* dst) const noexcept {
        using std::min;

        // A wrapped lane is negative, which only an unsigned min keeps within peak.
        if constexpr (std::same_as<V, typename Isa::Narrow> && std::same_as<pixel_t, std::uint16_t>)
            Isa::store(Isa::minUnsigned(maximum, V(static_cast<int>(peak))), dst);
        else if constexpr (std::integral<pixel_t> && !storeSaturates<Isa, pixel_t>)
            Isa::store(min(maximum, V(static_cast<int>(peak))), dst);
        else
            Isa::store(maximum, dst);
    }
};

template<KernelIsa Isa, int size, typename T>
static auto loadColumns(const T* center) noexcept {
    return [&]<int... column>(std::integer_sequence<int, column...>) noexcept {
        return std::array{ Isa::load(center + column - size / 2)... };
    }(std::make_integer_sequence<int, size>());
}

// On Intel Core CPUs, a load soon after a store to the same physical address modulo 1 MiB stalls, and frames of one size often lie whole MiB apart.
// Then the next vector's loads are the last to overlap a store, so storing after them keeps a kernel reading the frame from running up to 12 times slower.
// Kernels that read only the ring, such as Sobel and RScharr, also measured 2-6% faster this way.
// Scalar code stays as it is for the sake of the vectorizers, and integer ExKirsch, which reads no frame, would only lose vector registers.
template<KernelIsa Isa, typename work_t, Pixel pixel_t, Operator op>
inline constexpr bool storesAfterNextLoads{ std::is_class_v<Vector<Isa, work_t>> && (op != Operator::ExKirsch || readsFrame<work_t, pixel_t, op>) };

// The loops over columns are out of line and flattened so that each holds nothing but the formulas.
// GCC otherwise splits the loop over rows at mirror's branches and leaves the copy that runs most rows unvectorized, as if it were cold.
// The rows get copied because the stores could alias a caller's copy, which would keep compilers from vectorizing.
template<KernelIsa Isa, typename work_t, Operator op, typename sample_t, Pixel pixel_t, typename OutputFn>
[[gnu::noinline, gnu::flatten]] static void filterColumns(const WindowRows<sample_t>& windowRows, int begin, int end, pixel_t* dst,
                                                          const OutputFn output) noexcept {
    constexpr int lanes{ laneCount<Isa, work_t> };
    static constexpr int size{ windowSize(op) };

    const WindowRows<sample_t> rows{ windowRows };

    // response stays out of the lambda, because Clang's flatten inlines only the calls written in this body.
    const auto windowAt = [&](int x) noexcept {
        const auto rowOf = [x](const sample_t* row) noexcept { return loadColumns<Isa, size>(row + x); };
        return makeWindow<size>(rows, rowOf);
    };

    if constexpr (storesAfterNextLoads<Isa, work_t, pixel_t, op>) {
        if (begin >= end)
            return;

        auto pending = response<op>(windowAt(begin));
        int x{ begin + lanes };

        for (; x < end; x += lanes) {
            const auto next = response<op>(windowAt(x));
            output(pending, dst + x - lanes);
            pending = next;
        }

        output(pending, dst + x - lanes);
    } else {
        for (int x = begin; x < end; x += lanes)
            output(response<op>(windowAt(x)), dst + x);
    }
}

template<KernelIsa Isa, Pixel pixel_t, typename work_t, Operator op>
static void filterDirect(const Plane<pixel_t>& plane) {
    using V = Vector<Isa, work_t>;

    const auto filterRows = [&](auto output) {
        using Ring = RowRing<Isa, pixel_t, work_t, op>;

        Ring ring{ plane };

        for (int y = 0; y < plane.height; y++) {
            const auto rows = ring.rows(y);
            pixel_t* const dst = plane.dst + y * plane.dstStride;

            if constexpr (readsFrame<work_t, pixel_t, op>) {
                const int interiorEnd{ ring.interiorEnd() };
                filterColumns<Isa, work_t, op>(rows, 0, Ring::interiorBegin, dst, output);
                filterColumns<Isa, work_t, op>(frameRows<work_t>(plane, y), Ring::interiorBegin, interiorEnd, dst, output);
                filterColumns<Isa, work_t, op>(rows, std::max(Ring::interiorBegin, interiorEnd), plane.width, dst, output);
            } else {
                filterColumns<Isa, work_t, op>(rows, 0, plane.width, dst, output);
            }
        }
    };

    if constexpr (std::same_as<decltype(response<op>(std::declval<Window<V, windowSize(op)>>())), V>) {
        if (plane.scale == 1.0f) {
            filterRows(UnitScaleOutput<Isa, pixel_t>{ plane.peak });
            return;
        }
    }

    filterRows(Output<Isa, pixel_t>{ plane.scale, plane.peak });
}

template<KernelIsa Isa, typename work_t, Operator op, typename sample_t>
[[gnu::noinline, gnu::flatten]] static void prefilterColumns(const WindowRows<sample_t>& windowRows, int width, work_t* smoothed,
                                                             work_t* differentiated) noexcept {
    constexpr int lanes{ laneCount<Isa, work_t> };

    const WindowRows<sample_t> rows{ windowRows };

    for (int x = 0; x < width; x += lanes) {
        const auto column = makeWindow<windowSize(op)>(rows, [x](const sample_t* row) noexcept { return Isa::load(row + x); });
        Isa::store(smooth<op>(column), smoothed + x);
        Isa::store(differentiate<op>(column), differentiated + x);
    }
}

template<KernelIsa Isa, typename work_t, Operator op, Pixel pixel_t>
[[gnu::noinline, gnu::flatten]] static void gradientColumns(const work_t* smoothed, const work_t* differentiated, int width, pixel_t* dst,
                                                            const Output<Isa, pixel_t> output) noexcept {
    using V = Vector<Isa, work_t>;

    constexpr int lanes{ laneCount<Isa, work_t> };
    constexpr int size{ windowSize(op) };

    for (int x = 0; x < width; x += lanes)
        output(Gradient<V>{ differentiate<op>(loadColumns<Isa, size>(smoothed + x)), smooth<op>(loadColumns<Isa, size>(differentiated + x)) }, dst + x);
}

template<KernelIsa Isa, Pixel pixel_t, typename work_t, Operator op>
static void filterSeparable(const Plane<pixel_t>& plane) {
    constexpr int radius{ windowSize(op) / 2 };

    const LineBuffers<Isa, work_t> lines{ plane.width, 2 };
    work_t* const smoothed = lines.line(0);
    work_t* const differentiated = lines.line(1);
    const Output<Isa, pixel_t> output{ plane.scale, plane.peak };

    const auto filterRow = [&](const auto& rows, int y) noexcept {
        prefilterColumns<Isa, work_t, op>(rows, plane.width, smoothed, differentiated);
        mirrorMargins<radius>(smoothed, plane.width);
        mirrorMargins<radius>(differentiated, plane.width);
        gradientColumns<Isa, work_t, op>(smoothed, differentiated, plane.width, plane.dst + y * plane.dstStride, output);
    };

    if constexpr (readsFrame<work_t, pixel_t, op>) {
        for (int y = 0; y < plane.height; y++)
            filterRow(frameRows<work_t>(plane, y), y);
    } else {
        RowRing<Isa, pixel_t, work_t, op> ring{ plane };

        for (int y = 0; y < plane.height; y++)
            filterRow(ring.rows(y), y);
    }
}

template<KernelIsa Isa, Pixel pixel_t, typename work_t, Operator op>
static void filterPlane(const Plane<pixel_t>& plane) {
    if constexpr (isSeparable(op))
        filterSeparable<Isa, pixel_t, work_t, op>(plane);
    else
        filterDirect<Isa, pixel_t, work_t, op>(plane);
}

// Integer clips compute in 16-bit lanes when their responses fit them. VapourSynth stores 8 bit samples in 1 byte and 9 to 16 bit samples in 2.
template<KernelIsa Isa, Pixel pixel_t, Operator op>
static void filterSimd(const Plane<pixel_t>& plane) {
    constexpr float lowestPeak{ sizeof(pixel_t) == 1 ? 255.0f : 511.0f };
    constexpr float highestPeak{ sizeof(pixel_t) == 1 ? 255.0f : 65535.0f };

    if constexpr (std::floating_point<pixel_t>)
        filterPlane<Isa, pixel_t, float, op>(plane);
    else if constexpr (fitsNarrow(op, highestPeak))
        filterPlane<Isa, pixel_t, std::int16_t, op>(plane);
    else if constexpr (fitsNarrow(op, lowestPeak))
        fitsNarrow(op, plane.peak) ? filterPlane<Isa, pixel_t, std::int16_t, op>(plane) : filterPlane<Isa, pixel_t, std::int32_t, op>(plane);
    else
        filterPlane<Isa, pixel_t, std::int32_t, op>(plane);
}

template<KernelIsa Isa, Pixel pixel_t>
static constexpr FilterTable<pixel_t> simdFilters = withAllOperators([](auto... ops) {
    return FilterTable<pixel_t>{ filterSimd<Isa, pixel_t, decltype(ops)::value>... };
});

}  // namespace edgemasks

#endif  // VAPOURSYNTH_EDGEMASKS_SIMD_H
