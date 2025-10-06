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

#include <cmath>

#include <algorithm>
#include <memory>

#include "edgemasks.h"

#ifdef EDGEMASKS_X86
template<typename pixel_t, int Operator, bool euclidean>
extern void filterSSE4(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;

template<typename pixel_t, int Operator, bool euclidean>
extern void filterAVX2(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;

template<typename pixel_t, int Operator, bool euclidean>
extern void filterAVX512(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
#endif

template<typename pixel_t, int Operator, bool euclidean>
static void filterC(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    using scalar_t = std::conditional_t<std::is_integral_v<pixel_t>, int, float>;

    for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const ptrdiff_t stride = vsapi->getStride(src, plane) / d->vi->format.bytesPerSample;
            auto srcp0 = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(src, plane));
            auto dstp = reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, plane));

            pixel_t a00, a01, a02, a03, a04;
            pixel_t a10, a11, a12, a13, a14;
            pixel_t a20, a21, a22, a23, a24;
            pixel_t a30, a31, a32, a33, a34;
            pixel_t a40, a41, a42, a43, a44;

            auto detect = [&]() noexcept {
                scalar_t gx, gy;
                float g;

                if constexpr (Operator == Tritical) {
                    gx = a10 - a12;
                    gy = a01 - a21;
                } else if constexpr (Operator == Cross) {
                    gx = a00 - a22;
                    gy = a02 - a20;
                } else if constexpr (Operator == Prewitt) {
                    gx = a00 + a10 + a20 - a02 - a12 - a22;
                    gy = a00 + a01 + a02 - a20 - a21 - a22;
                } else if constexpr (Operator == Sobel) {
                    gx = a00 + 2 * a10 + a20 - a02 - 2 * a12 - a22;
                    gy = a00 + 2 * a01 + a02 - a20 - 2 * a21 - a22;
                } else if constexpr (Operator == Scharr) {
                    gx = 3 * (a00 + a20) + 10 * a10 - 3 * (a02 + a22) - 10 * a12;
                    gy = 3 * (a00 + a02) + 10 * a01 - 3 * (a20 + a22) - 10 * a21;
                } else if constexpr (Operator == RScharr) {
                    gx = 47 * (a00 + a20) + 162 * a10 - 47 * (a02 + a22) - 162 * a12;
                    gy = 47 * (a00 + a02) + 162 * a01 - 47 * (a20 + a22) - 162 * a21;
                } else if constexpr (Operator == Kroon) {
                    gx = 17 * (a00 + a20) + 61 * a10 - 17 * (a02 + a22) - 61 * a12;
                    gy = 17 * (a00 + a02) + 61 * a01 - 17 * (a20 + a22) - 61 * a21;
                } else if constexpr (Operator == Robinson3) {
                    const scalar_t g1 = a02 + a01 + a00 - a20 - a21 - a22;
                    const scalar_t g2 = a01 + a00 + a10 - a21 - a22 - a12;
                    const scalar_t g3 = a00 + a10 + a20 - a22 - a12 - a02;
                    const scalar_t g4 = a10 + a20 + a21 - a12 - a02 - a01;
                    g = std::max({ std::abs(g1), std::abs(g2), std::abs(g3), std::abs(g4) });
                } else if constexpr (Operator == Robinson5) {
                    const scalar_t g1 = a02 + 2 * a01 + a00 - a20 - 2 * a21 - a22;
                    const scalar_t g2 = a01 + 2 * a00 + a10 - a21 - 2 * a22 - a12;
                    const scalar_t g3 = a00 + 2 * a10 + a20 - a22 - 2 * a12 - a02;
                    const scalar_t g4 = a10 + 2 * a20 + a21 - a12 - 2 * a02 - a01;
                    g = std::max({ std::abs(g1), std::abs(g2), std::abs(g3), std::abs(g4) });
                } else if constexpr (Operator == Kirsch) {
                    const scalar_t g1 = 5 * (a02 + a01 + a00) - 3 * (a10 + a20 + a21 + a22 + a12);
                    const scalar_t g2 = 5 * (a01 + a00 + a10) - 3 * (a20 + a21 + a22 + a12 + a02);
                    const scalar_t g3 = 5 * (a00 + a10 + a20) - 3 * (a21 + a22 + a12 + a02 + a01);
                    const scalar_t g4 = 5 * (a10 + a20 + a21) - 3 * (a22 + a12 + a02 + a01 + a00);
                    const scalar_t g5 = 5 * (a20 + a21 + a22) - 3 * (a12 + a02 + a01 + a00 + a10);
                    const scalar_t g6 = 5 * (a21 + a22 + a12) - 3 * (a02 + a01 + a00 + a10 + a20);
                    const scalar_t g7 = 5 * (a22 + a12 + a02) - 3 * (a01 + a00 + a10 + a20 + a21);
                    const scalar_t g8 = 5 * (a12 + a02 + a01) - 3 * (a00 + a10 + a20 + a21 + a22);
                    g = std::max({ std::abs(g1), std::abs(g2), std::abs(g3), std::abs(g4), std::abs(g5), std::abs(g6), std::abs(g7), std::abs(g8) });
                } else if constexpr (Operator == ExPrewitt) {
                    gx = 2 * (a00 + a10 + a20 + a30 + a40) + a01 + a11 + a21 + a31 + a41
                        - a03 - a13 - a23 - a33 - a43 - 2 * (a04 + a14 + a24 + a34 + a44);
                    gy = 2 * (a00 + a01 + a02 + a03 + a04) + a10 + a11 + a12 + a13 + a14
                        - a30 - a31 - a32 - a33 - a34 - 2 * (a40 + a41 + a42 + a43 + a44);
                } else if constexpr (Operator == ExSobel) {
                    gx = 2 * (a00 + a10 + a30 + a40) + 4 * a20 + a01 + a11 + 2 * a21 + a31 + a41
                        - a03 - a13 - 2 * a23 - a33 - a43 - 2 * (a04 + a14 + a34 + a44) - 4 * a24;
                    gy = 2 * (a00 + a01 + a03 + a04) + 4 * a02 + a10 + a11 + 2 * a12 + a13 + a14
                        - a30 - a31 - 2 * a32 - a33 - a34 - 2 * (a40 + a41 + a43 + a44) - 4 * a42;
                } else if constexpr (Operator == FDoG) {
                    gx = a00 + a01 + a40 + a41 + 2 * (a10 + a11 + a30 + a31) + 3 * (a20 + a21)
                        - a03 - a04 - a43 - a44 - 2 * (a13 + a14 + a33 + a34) - 3 * (a23 + a24);
                    gy = a00 + a10 + a04 + a14 + 2 * (a01 + a11 + a03 + a13) + 3 * (a02 + a12)
                        - a30 - a40 - a34 - a44 - 2 * (a31 + a41 + a33 + a43) - 3 * (a32 + a42);
                } else if constexpr (Operator == ExKirsch) {
                    const scalar_t g1 = 9 * (a14 + a04 + a03 + a02 + a01 + a00 + a10) - 7 * (a20 + a30 + a40 + a41 + a42 + a43 + a44 + a34 + a24)
                        + 5 * (a13 + a12 + a11) - 3 * (a21 + a31 + a32 + a33 + a23);
                    const scalar_t g2 = 9 * (a03 + a02 + a01 + a00 + a10 + a20 + a30) - 7 * (a40 + a41 + a42 + a43 + a44 + a34 + a24 + a14 + a04)
                        + 5 * (a12 + a11 + a21) - 3 * (a31 + a32 + a33 + a23 + a13);
                    const scalar_t g3 = 9 * (a01 + a00 + a10 + a20 + a30 + a40 + a41) - 7 * (a42 + a43 + a44 + a34 + a24 + a14 + a04 + a03 + a02)
                        + 5 * (a11 + a21 + a31) - 3 * (a32 + a33 + a23 + a13 + a12);
                    const scalar_t g4 = 9 * (a10 + a20 + a30 + a40 + a41 + a42 + a43) - 7 * (a44 + a34 + a24 + a14 + a04 + a03 + a02 + a01 + a00)
                        + 5 * (a21 + a31 + a32) - 3 * (a33 + a23 + a13 + a12 + a11);
                    const scalar_t g5 = 9 * (a30 + a40 + a41 + a42 + a43 + a44 + a34) - 7 * (a24 + a14 + a04 + a03 + a02 + a01 + a00 + a10 + a20)
                        + 5 * (a31 + a32 + a33) - 3 * (a23 + a13 + a12 + a11 + a21);
                    const scalar_t g6 = 9 * (a41 + a42 + a43 + a44 + a34 + a24 + a14) - 7 * (a04 + a03 + a02 + a01 + a00 + a10 + a20 + a30 + a40)
                        + 5 * (a32 + a33 + a23) - 3 * (a13 + a12 + a11 + a21 + a31);
                    const scalar_t g7 = 9 * (a43 + a44 + a34 + a24 + a14 + a04 + a03) - 7 * (a02 + a01 + a00 + a10 + a20 + a30 + a40 + a41 + a42)
                        + 5 * (a33 + a23 + a13) - 3 * (a12 + a11 + a21 + a31 + a32);
                    const scalar_t g8 = 9 * (a34 + a24 + a14 + a04 + a03 + a02 + a01) - 7 * (a00 + a10 + a20 + a30 + a40 + a41 + a42 + a43 + a44)
                        + 5 * (a23 + a13 + a12) - 3 * (a11 + a21 + a31 + a32 + a33);
                    g = std::max({ std::abs(g1), std::abs(g2), std::abs(g3), std::abs(g4), std::abs(g5), std::abs(g6), std::abs(g7), std::abs(g8) });
                }

                if constexpr (euclidean)
                    g = std::sqrt(static_cast<float>(gx) * gx + static_cast<float>(gy) * gy);

                g *= d->scale;

                if constexpr (std::is_integral_v<pixel_t>)
                    return std::min(static_cast<int>(g + 0.5f), d->peak);
                else
                    return g;
            };

            for (int y = 0; y < height; y++) {
                auto prev1 = (y == 0) ? srcp0 + stride : srcp0 - stride;
                auto next1 = (y == height - 1) ? srcp0 - stride : srcp0 + stride;

                if (d->matrix == 3) {
                    int x = 0;
                    a00 = prev1[x + 1]; a01 = prev1[x]; a02 = prev1[x + 1];
                    a10 = srcp0[x + 1]; a11 = srcp0[x]; a12 = srcp0[x + 1];
                    a20 = next1[x + 1]; a21 = next1[x]; a22 = next1[x + 1];
                    dstp[x] = detect();

                    for (x = 1; x < width - 1; x++) {
                        a00 = prev1[x - 1]; a01 = prev1[x]; a02 = prev1[x + 1];
                        a10 = srcp0[x - 1]; a11 = srcp0[x]; a12 = srcp0[x + 1];
                        a20 = next1[x - 1]; a21 = next1[x]; a22 = next1[x + 1];
                        dstp[x] = detect();
                    }

                    x = width - 1;
                    a00 = prev1[x - 1]; a01 = prev1[x]; a02 = prev1[x - 1];
                    a10 = srcp0[x - 1]; a11 = srcp0[x]; a12 = srcp0[x - 1];
                    a20 = next1[x - 1]; a21 = next1[x]; a22 = next1[x - 1];
                    dstp[x] = detect();
                } else {
                    auto prev2 = (y == 0) ? srcp0 + stride * 2 : (y == 1 ? srcp0 : srcp0 - stride * 2);
                    auto next2 = (y == height - 1) ? srcp0 - stride * 2 : (y == height - 2 ? srcp0 : srcp0 + stride * 2);

                    int x = 0;
                    a00 = prev2[x + 2]; a01 = prev2[x + 1]; a02 = prev2[x]; a03 = prev2[x + 1]; a04 = prev2[x + 2];
                    a10 = prev1[x + 2]; a11 = prev1[x + 1]; a12 = prev1[x]; a13 = prev1[x + 1]; a14 = prev1[x + 2];
                    a20 = srcp0[x + 2]; a21 = srcp0[x + 1]; a22 = srcp0[x]; a23 = srcp0[x + 1]; a24 = srcp0[x + 2];
                    a30 = next1[x + 2]; a31 = next1[x + 1]; a32 = next1[x]; a33 = next1[x + 1]; a34 = next1[x + 2];
                    a40 = next2[x + 2]; a41 = next2[x + 1]; a42 = next2[x]; a43 = next2[x + 1]; a44 = next2[x + 2];
                    dstp[x] = detect();

                    x = 1;
                    a00 = prev2[x]; a01 = prev2[x - 1]; a02 = prev2[x]; a03 = prev2[x + 1]; a04 = prev2[x + 2];
                    a10 = prev1[x]; a11 = prev1[x - 1]; a12 = prev1[x]; a13 = prev1[x + 1]; a14 = prev1[x + 2];
                    a20 = srcp0[x]; a21 = srcp0[x - 1]; a22 = srcp0[x]; a23 = srcp0[x + 1]; a24 = srcp0[x + 2];
                    a30 = next1[x]; a31 = next1[x - 1]; a32 = next1[x]; a33 = next1[x + 1]; a34 = next1[x + 2];
                    a40 = next2[x]; a41 = next2[x - 1]; a42 = next2[x]; a43 = next2[x + 1]; a44 = next2[x + 2];
                    dstp[x] = detect();

                    for (x = 2; x < width - 2; x++) {
                        a00 = prev2[x - 2]; a01 = prev2[x - 1]; a02 = prev2[x]; a03 = prev2[x + 1]; a04 = prev2[x + 2];
                        a10 = prev1[x - 2]; a11 = prev1[x - 1]; a12 = prev1[x]; a13 = prev1[x + 1]; a14 = prev1[x + 2];
                        a20 = srcp0[x - 2]; a21 = srcp0[x - 1]; a22 = srcp0[x]; a23 = srcp0[x + 1]; a24 = srcp0[x + 2];
                        a30 = next1[x - 2]; a31 = next1[x - 1]; a32 = next1[x]; a33 = next1[x + 1]; a34 = next1[x + 2];
                        a40 = next2[x - 2]; a41 = next2[x - 1]; a42 = next2[x]; a43 = next2[x + 1]; a44 = next2[x + 2];
                        dstp[x] = detect();
                    }

                    x = width - 2;
                    a00 = prev2[x - 2]; a01 = prev2[x - 1]; a02 = prev2[x]; a03 = prev2[x + 1]; a04 = prev2[x];
                    a10 = prev1[x - 2]; a11 = prev1[x - 1]; a12 = prev1[x]; a13 = prev1[x + 1]; a14 = prev1[x];
                    a20 = srcp0[x - 2]; a21 = srcp0[x - 1]; a22 = srcp0[x]; a23 = srcp0[x + 1]; a24 = srcp0[x];
                    a30 = next1[x - 2]; a31 = next1[x - 1]; a32 = next1[x]; a33 = next1[x + 1]; a34 = next1[x];
                    a40 = next2[x - 2]; a41 = next2[x - 1]; a42 = next2[x]; a43 = next2[x + 1]; a44 = next2[x];
                    dstp[x] = detect();

                    x = width - 1;
                    a00 = prev2[x - 2]; a01 = prev2[x - 1]; a02 = prev2[x]; a03 = prev2[x - 1]; a04 = prev2[x - 2];
                    a10 = prev1[x - 2]; a11 = prev1[x - 1]; a12 = prev1[x]; a13 = prev1[x - 1]; a14 = prev1[x - 2];
                    a20 = srcp0[x - 2]; a21 = srcp0[x - 1]; a22 = srcp0[x]; a23 = srcp0[x - 1]; a24 = srcp0[x - 2];
                    a30 = next1[x - 2]; a31 = next1[x - 1]; a32 = next1[x]; a33 = next1[x - 1]; a34 = next1[x - 2];
                    a40 = next2[x - 2]; a41 = next2[x - 1]; a42 = next2[x]; a43 = next2[x - 1]; a44 = next2[x - 2];
                    dstp[x] = detect();
                }

                srcp0 += stride;
                dstp += stride;
            }
        }
    }
}

template<typename pixel_t>
static auto selectC(const std::string& filterName) noexcept {
    if (filterName == "Tritical")
        return filterC<pixel_t, Tritical, true>;
    else if (filterName == "Cross")
        return filterC<pixel_t, Cross, true>;
    else if (filterName == "Prewitt")
        return filterC<pixel_t, Prewitt, true>;
    else if (filterName == "Sobel")
        return filterC<pixel_t, Sobel, true>;
    else if (filterName == "Scharr")
        return filterC<pixel_t, Scharr, true>;
    else if (filterName == "RScharr")
        return filterC<pixel_t, RScharr, true>;
    else if (filterName == "Kroon")
        return filterC<pixel_t, Kroon, true>;
    else if (filterName == "Robinson3")
        return filterC<pixel_t, Robinson3, false>;
    else if (filterName == "Robinson5")
        return filterC<pixel_t, Robinson5, false>;
    else if (filterName == "Kirsch")
        return filterC<pixel_t, Kirsch, false>;
    else if (filterName == "ExPrewitt")
        return filterC<pixel_t, ExPrewitt, true>;
    else if (filterName == "ExSobel")
        return filterC<pixel_t, ExSobel, true>;
    else if (filterName == "FDoG")
        return filterC<pixel_t, FDoG, true>;
    else
        return filterC<pixel_t, ExKirsch, false>;
}

template<typename pixel_t>
static auto selectSSE4(const std::string& filterName) noexcept {
    if (filterName == "Tritical")
        return filterSSE4<pixel_t, Tritical, true>;
    else if (filterName == "Cross")
        return filterSSE4<pixel_t, Cross, true>;
    else if (filterName == "Prewitt")
        return filterSSE4<pixel_t, Prewitt, true>;
    else if (filterName == "Sobel")
        return filterSSE4<pixel_t, Sobel, true>;
    else if (filterName == "Scharr")
        return filterSSE4<pixel_t, Scharr, true>;
    else if (filterName == "RScharr")
        return filterSSE4<pixel_t, RScharr, true>;
    else if (filterName == "Kroon")
        return filterSSE4<pixel_t, Kroon, true>;
    else if (filterName == "Robinson3")
        return filterSSE4<pixel_t, Robinson3, false>;
    else if (filterName == "Robinson5")
        return filterSSE4<pixel_t, Robinson5, false>;
    else if (filterName == "Kirsch")
        return filterSSE4<pixel_t, Kirsch, false>;
    else if (filterName == "ExPrewitt")
        return filterSSE4<pixel_t, ExPrewitt, true>;
    else if (filterName == "ExSobel")
        return filterSSE4<pixel_t, ExSobel, true>;
    else if (filterName == "FDoG")
        return filterSSE4<pixel_t, FDoG, true>;
    else
        return filterSSE4<pixel_t, ExKirsch, false>;
}

template<typename pixel_t>
static auto selectAVX2(const std::string& filterName) noexcept {
    if (filterName == "Tritical")
        return filterAVX2<pixel_t, Tritical, true>;
    else if (filterName == "Cross")
        return filterAVX2<pixel_t, Cross, true>;
    else if (filterName == "Prewitt")
        return filterAVX2<pixel_t, Prewitt, true>;
    else if (filterName == "Sobel")
        return filterAVX2<pixel_t, Sobel, true>;
    else if (filterName == "Scharr")
        return filterAVX2<pixel_t, Scharr, true>;
    else if (filterName == "RScharr")
        return filterAVX2<pixel_t, RScharr, true>;
    else if (filterName == "Kroon")
        return filterAVX2<pixel_t, Kroon, true>;
    else if (filterName == "Robinson3")
        return filterAVX2<pixel_t, Robinson3, false>;
    else if (filterName == "Robinson5")
        return filterAVX2<pixel_t, Robinson5, false>;
    else if (filterName == "Kirsch")
        return filterAVX2<pixel_t, Kirsch, false>;
    else if (filterName == "ExPrewitt")
        return filterAVX2<pixel_t, ExPrewitt, true>;
    else if (filterName == "ExSobel")
        return filterAVX2<pixel_t, ExSobel, true>;
    else if (filterName == "FDoG")
        return filterAVX2<pixel_t, FDoG, true>;
    else
        return filterAVX2<pixel_t, ExKirsch, false>;
}

template<typename pixel_t>
static auto selectAVX512(const std::string& filterName) noexcept {
    if (filterName == "Tritical")
        return filterAVX512<pixel_t, Tritical, true>;
    else if (filterName == "Cross")
        return filterAVX512<pixel_t, Cross, true>;
    else if (filterName == "Prewitt")
        return filterAVX512<pixel_t, Prewitt, true>;
    else if (filterName == "Sobel")
        return filterAVX512<pixel_t, Sobel, true>;
    else if (filterName == "Scharr")
        return filterAVX512<pixel_t, Scharr, true>;
    else if (filterName == "RScharr")
        return filterAVX512<pixel_t, RScharr, true>;
    else if (filterName == "Kroon")
        return filterAVX512<pixel_t, Kroon, true>;
    else if (filterName == "Robinson3")
        return filterAVX512<pixel_t, Robinson3, false>;
    else if (filterName == "Robinson5")
        return filterAVX512<pixel_t, Robinson5, false>;
    else if (filterName == "Kirsch")
        return filterAVX512<pixel_t, Kirsch, false>;
    else if (filterName == "ExPrewitt")
        return filterAVX512<pixel_t, ExPrewitt, true>;
    else if (filterName == "ExSobel")
        return filterAVX512<pixel_t, ExSobel, true>;
    else if (filterName == "FDoG")
        return filterAVX512<pixel_t, FDoG, true>;
    else
        return filterAVX512<pixel_t, ExKirsch, false>;
}

static const VSFrame* VS_CC edgemasksGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx,
                                              VSCore* core, const VSAPI* vsapi) {
    auto d = static_cast<const EdgeMasksData*>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame* fr[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] = { 0, 1, 2 };
        VSFrame* dst = vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        d->filter(src, dst, d, vsapi);

        vsapi->mapSetInt(vsapi->getFramePropertiesRW(dst), "_ColorRange", 0, maReplace);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC edgemasksFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d = static_cast<EdgeMasksData*>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC edgemasksCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<EdgeMasksData>();

    d->filterName = static_cast<const char*>(userData);

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);
        int err;

        if (!vsh::isConstantVideoFormat(d->vi) ||
            (d->vi->format.sampleType == stInteger && d->vi->format.bitsPerSample > 16) ||
            (d->vi->format.sampleType == stFloat && d->vi->format.bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported";

        const int m = vsapi->mapNumElements(in, "planes");

        for (int i = 0; i < 3; i++)
            d->process[i] = (m <= 0);

        for (int i = 0; i < m; i++) {
            const int n = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);

            if (n < 0 || n >= d->vi->format.numPlanes)
                throw "plane index out of range";

            if (d->process[n])
                throw "plane specified twice";

            d->process[n] = true;
        }

        d->scale = vsapi->mapGetFloatSaturated(in, "scale", 0, &err);
        if (err)
            d->scale = 1.0f;

        const int opt = vsapi->mapGetIntSaturated(in, "opt", 0, &err);

        if (d->filterName == "ExPrewitt" || d->filterName == "ExSobel" || d->filterName == "FDoG" || d->filterName == "ExKirsch")
            d->matrix = 5;
        else
            d->matrix = 3;

        auto frame = vsapi->getFrame(0, d->node, nullptr, 0);
        for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
            if (d->process[plane]) {
                if (vsapi->getFrameWidth(frame, plane) < d->matrix) {
                    vsapi->freeFrame(frame);
                    throw "plane's width must be greater than or equal to " + std::to_string(d->matrix);
                }

                if (vsapi->getFrameHeight(frame, plane) < d->matrix) {
                    vsapi->freeFrame(frame);
                    throw "plane's height must be greater than or equal to " + std::to_string(d->matrix);
                }
            }
        }
        vsapi->freeFrame(frame);

        if (d->scale <= 0.0f)
            throw "scale must be greater than 0.0";

        if (opt < 0 || opt > 4)
            throw "opt must be 0, 1, 2, 3, or 4";

        {
#ifdef EDGEMASKS_X86
            const int iset = instrset_detect();
#endif

            if (d->vi->format.bytesPerSample == 1) {
                d->filter = selectC<uint8_t>(d->filterName);

#ifdef EDGEMASKS_X86
                if ((opt == 0 && iset >= 10) || opt == 4)
                    d->filter = selectAVX512<uint8_t>(d->filterName);
                else if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = selectAVX2<uint8_t>(d->filterName);
                else if ((opt == 0 && iset >= 5) || opt == 2)
                    d->filter = selectSSE4<uint8_t>(d->filterName);
#endif
            } else if (d->vi->format.bytesPerSample == 2) {
                d->filter = selectC<uint16_t>(d->filterName);

#ifdef EDGEMASKS_X86
                if ((opt == 0 && iset >= 10) || opt == 4)
                    d->filter = selectAVX512<uint16_t>(d->filterName);
                else if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = selectAVX2<uint16_t>(d->filterName);
                else if ((opt == 0 && iset >= 5) || opt == 2)
                    d->filter = selectSSE4<uint16_t>(d->filterName);
#endif
            } else {
                d->filter = selectC<float>(d->filterName);

#ifdef EDGEMASKS_X86
                if ((opt == 0 && iset >= 10) || opt == 4)
                    d->filter = selectAVX512<float>(d->filterName);
                else if ((opt == 0 && iset >= 8) || opt == 3)
                    d->filter = selectAVX2<float>(d->filterName);
                else if ((opt == 0 && iset >= 5) || opt == 2)
                    d->filter = selectSSE4<float>(d->filterName);
#endif
            }
        }

        if (d->vi->format.sampleType == stInteger)
            d->peak = (1 << d->vi->format.bitsPerSample) - 1;
    } catch (const std::string& error) {
        vsapi->mapSetError(out, (d->filterName + ": " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    VSFilterDependency deps[] = { {d->node, rpStrictSpatial} };
    vsapi->createVideoFilter(out, d->filterName.c_str(), d->vi, edgemasksGetFrame, edgemasksFree, fmParallel, deps, 1, d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.edgemasks",
                         "edgemasks",
                         "Creates an edge mask using various operators",
                         VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION,
                         0,
                         plugin);

    const char* operators[] = {
        "Tritical", "Cross", "Prewitt", "Sobel", "Scharr", "RScharr", "Kroon", "Robinson3", "Robinson5", "Kirsch", "ExPrewitt", "ExSobel", "FDoG", "ExKirsch"
    };

    for (int i = 0; i < 14; i++)
        vspapi->registerFunction(operators[i],
                                 "clip:vnode;planes:int[]:opt;scale:float:opt;opt:int:opt;",
                                 "clip:vnode;",
                                 edgemasksCreate,
                                 const_cast<char*>(operators[i]),
                                 plugin);
}
