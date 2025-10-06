#ifdef EDGEMASKS_X86
#define INSTRSET 10
#include "edgemasks.h"

template<typename pixel_t, int Operator, bool euclidean>
void filterAVX512(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    using vector_t = std::conditional_t<std::is_integral_v<pixel_t>, Vec16i, Vec16f>;

    auto load = [](const pixel_t* srcp) noexcept {
        if constexpr (std::is_same_v<pixel_t, uint8_t>)
            return vector_t().load_16uc(srcp);
        else if constexpr (std::is_same_v<pixel_t, uint16_t>)
            return vector_t().load_16us(srcp);
        else
            return vector_t().load(srcp);
    };

    auto store = [&](const vector_t& srcp, pixel_t* dstp) noexcept {
        if constexpr (std::is_same_v<pixel_t, uint8_t>) {
            const auto result = compress_saturated_s2u(compress_saturated(srcp, zero_si512()), zero_si512()).get_low().get_low();
            result.store_nt(dstp);
        } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
            const auto result = compress_saturated_s2u(srcp, zero_si512()).get_low();
            min(result, d->peak).store_nt(dstp);
        } else {
            srcp.store_nt(dstp);
        }
    };

    for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const ptrdiff_t stride = vsapi->getStride(src, plane) / d->vi->format.bytesPerSample;
            auto srcp0 = reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(src, plane));
            auto dstp = reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, plane));

            vector_t a00, a01, a02, a03, a04;
            vector_t a10, a11, a12, a13, a14;
            vector_t a20, a21, a22, a23, a24;
            vector_t a30, a31, a32, a33, a34;
            vector_t a40, a41, a42, a43, a44;

            const int regularPart = (width - 1) & ~(vector_t().size() - 1);

            auto detect = [&]() noexcept {
                vector_t gx, gy, g;
                Vec16f gxF, gyF, gF;

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
                    const vector_t g1 = a02 + a01 + a00 - a20 - a21 - a22;
                    const vector_t g2 = a01 + a00 + a10 - a21 - a22 - a12;
                    const vector_t g3 = a00 + a10 + a20 - a22 - a12 - a02;
                    const vector_t g4 = a10 + a20 + a21 - a12 - a02 - a01;
                    g = max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4)));
                } else if constexpr (Operator == Robinson5) {
                    const vector_t g1 = a02 + 2 * a01 + a00 - a20 - 2 * a21 - a22;
                    const vector_t g2 = a01 + 2 * a00 + a10 - a21 - 2 * a22 - a12;
                    const vector_t g3 = a00 + 2 * a10 + a20 - a22 - 2 * a12 - a02;
                    const vector_t g4 = a10 + 2 * a20 + a21 - a12 - 2 * a02 - a01;
                    g = max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4)));
                } else if constexpr (Operator == Kirsch) {
                    const vector_t g1 = 5 * (a02 + a01 + a00) - 3 * (a10 + a20 + a21 + a22 + a12);
                    const vector_t g2 = 5 * (a01 + a00 + a10) - 3 * (a20 + a21 + a22 + a12 + a02);
                    const vector_t g3 = 5 * (a00 + a10 + a20) - 3 * (a21 + a22 + a12 + a02 + a01);
                    const vector_t g4 = 5 * (a10 + a20 + a21) - 3 * (a22 + a12 + a02 + a01 + a00);
                    const vector_t g5 = 5 * (a20 + a21 + a22) - 3 * (a12 + a02 + a01 + a00 + a10);
                    const vector_t g6 = 5 * (a21 + a22 + a12) - 3 * (a02 + a01 + a00 + a10 + a20);
                    const vector_t g7 = 5 * (a22 + a12 + a02) - 3 * (a01 + a00 + a10 + a20 + a21);
                    const vector_t g8 = 5 * (a12 + a02 + a01) - 3 * (a00 + a10 + a20 + a21 + a22);
                    g = max(max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4))), max(max(abs(g5), abs(g6)), max(abs(g7), abs(g8))));
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
                    const vector_t g1 = 9 * (a14 + a04 + a03 + a02 + a01 + a00 + a10) - 7 * (a20 + a30 + a40 + a41 + a42 + a43 + a44 + a34 + a24)
                        + 5 * (a13 + a12 + a11) - 3 * (a21 + a31 + a32 + a33 + a23);
                    const vector_t g2 = 9 * (a03 + a02 + a01 + a00 + a10 + a20 + a30) - 7 * (a40 + a41 + a42 + a43 + a44 + a34 + a24 + a14 + a04)
                        + 5 * (a12 + a11 + a21) - 3 * (a31 + a32 + a33 + a23 + a13);
                    const vector_t g3 = 9 * (a01 + a00 + a10 + a20 + a30 + a40 + a41) - 7 * (a42 + a43 + a44 + a34 + a24 + a14 + a04 + a03 + a02)
                        + 5 * (a11 + a21 + a31) - 3 * (a32 + a33 + a23 + a13 + a12);
                    const vector_t g4 = 9 * (a10 + a20 + a30 + a40 + a41 + a42 + a43) - 7 * (a44 + a34 + a24 + a14 + a04 + a03 + a02 + a01 + a00)
                        + 5 * (a21 + a31 + a32) - 3 * (a33 + a23 + a13 + a12 + a11);
                    const vector_t g5 = 9 * (a30 + a40 + a41 + a42 + a43 + a44 + a34) - 7 * (a24 + a14 + a04 + a03 + a02 + a01 + a00 + a10 + a20)
                        + 5 * (a31 + a32 + a33) - 3 * (a23 + a13 + a12 + a11 + a21);
                    const vector_t g6 = 9 * (a41 + a42 + a43 + a44 + a34 + a24 + a14) - 7 * (a04 + a03 + a02 + a01 + a00 + a10 + a20 + a30 + a40)
                        + 5 * (a32 + a33 + a23) - 3 * (a13 + a12 + a11 + a21 + a31);
                    const vector_t g7 = 9 * (a43 + a44 + a34 + a24 + a14 + a04 + a03) - 7 * (a02 + a01 + a00 + a10 + a20 + a30 + a40 + a41 + a42)
                        + 5 * (a33 + a23 + a13) - 3 * (a12 + a11 + a21 + a31 + a32);
                    const vector_t g8 = 9 * (a34 + a24 + a14 + a04 + a03 + a02 + a01) - 7 * (a00 + a10 + a20 + a30 + a40 + a41 + a42 + a43 + a44)
                        + 5 * (a23 + a13 + a12) - 3 * (a11 + a21 + a31 + a32 + a33);
                    g = max(max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4))), max(max(abs(g5), abs(g6)), max(abs(g7), abs(g8))));
                }

                if constexpr (std::is_integral_v<pixel_t>) {
                    if constexpr (euclidean) {
                        gxF = to_float(gx);
                        gyF = to_float(gy);
                    } else {
                        gF = to_float(g);
                    }
                } else {
                    if constexpr (euclidean) {
                        gxF = gx;
                        gyF = gy;
                    } else {
                        gF = g;
                    }
                }

                if constexpr (euclidean)
                    gF = sqrt(gxF * gxF + gyF * gyF);

                gF *= d->scale;

                if constexpr (std::is_integral_v<pixel_t>)
                    return truncatei(gF + 0.5f);
                else
                    return gF;
            };

            for (int y = 0; y < height; y++) {
                auto prev1 = (y == 0) ? srcp0 + stride : srcp0 - stride;
                auto next1 = (y == height - 1) ? srcp0 - stride : srcp0 + stride;

                if (d->matrix == 3) {
                    a01 = load(prev1);
                    a11 = load(srcp0);
                    a21 = load(next1);

                    a00 = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(a01);
                    a10 = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(a11);
                    a20 = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(a21);

                    if (width > vector_t().size()) {
                        a02 = load(prev1 + 1);
                        a12 = load(srcp0 + 1);
                        a22 = load(next1 + 1);
                    } else {
                        a02 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a01);
                        a12 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a11);
                        a22 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a21);
                    }

                    store(detect(), dstp);

                    for (int x = vector_t().size(); x < regularPart; x += vector_t().size()) {
                        a00 = load(prev1 + x - 1); a01 = load(prev1 + x); a02 = load(prev1 + x + 1);
                        a10 = load(srcp0 + x - 1); a11 = load(srcp0 + x); a12 = load(srcp0 + x + 1);
                        a20 = load(next1 + x - 1); a21 = load(next1 + x); a22 = load(next1 + x + 1);

                        store(detect(), dstp + x);
                    }

                    if (regularPart >= vector_t().size()) {
                        a00 = load(prev1 + regularPart - 1); a01 = load(prev1 + regularPart);
                        a10 = load(srcp0 + regularPart - 1); a11 = load(srcp0 + regularPart);
                        a20 = load(next1 + regularPart - 1); a21 = load(next1 + regularPart);

                        a02 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a01);
                        a12 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a11);
                        a22 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a21);

                        store(detect(), dstp + regularPart);
                    }
                } else {
                    auto prev2 = (y == 0) ? srcp0 + stride * 2 : (y == 1 ? srcp0 : srcp0 - stride * 2);
                    auto next2 = (y == height - 1) ? srcp0 - stride * 2 : (y == height - 2 ? srcp0 : srcp0 + stride * 2);

                    a02 = load(prev2);
                    a12 = load(prev1);
                    a22 = load(srcp0);
                    a32 = load(next1);
                    a42 = load(next2);

                    a00 = permute16<2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13>(a02);
                    a10 = permute16<2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13>(a12);
                    a20 = permute16<2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13>(a22);
                    a30 = permute16<2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13>(a32);
                    a40 = permute16<2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13>(a42);

                    a01 = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(a02);
                    a11 = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(a12);
                    a21 = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(a22);
                    a31 = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(a32);
                    a41 = permute16<1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>(a42);

                    if (width > vector_t().size()) {
                        a03 = load(prev2 + 1); a04 = load(prev2 + 2);
                        a13 = load(prev1 + 1); a14 = load(prev1 + 2);
                        a23 = load(srcp0 + 1); a24 = load(srcp0 + 2);
                        a33 = load(next1 + 1); a34 = load(next1 + 2);
                        a43 = load(next2 + 1); a44 = load(next2 + 2);
                    } else {
                        a03 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a02);
                        a13 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a12);
                        a23 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a22);
                        a33 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a32);
                        a43 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a42);

                        a04 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a02);
                        a14 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a12);
                        a24 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a22);
                        a34 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a32);
                        a44 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a42);
                    }

                    store(detect(), dstp);

                    for (int x = vector_t().size(); x < regularPart; x += vector_t().size()) {
                        a00 = load(prev2 + x - 2); a01 = load(prev2 + x - 1); a02 = load(prev2 + x); a03 = load(prev2 + x + 1); a04 = load(prev2 + x + 2);
                        a10 = load(prev1 + x - 2); a11 = load(prev1 + x - 1); a12 = load(prev1 + x); a13 = load(prev1 + x + 1); a14 = load(prev1 + x + 2);
                        a20 = load(srcp0 + x - 2); a21 = load(srcp0 + x - 1); a22 = load(srcp0 + x); a23 = load(srcp0 + x + 1); a24 = load(srcp0 + x + 2);
                        a30 = load(next1 + x - 2); a31 = load(next1 + x - 1); a32 = load(next1 + x); a33 = load(next1 + x + 1); a34 = load(next1 + x + 2);
                        a40 = load(next2 + x - 2); a41 = load(next2 + x - 1); a42 = load(next2 + x); a43 = load(next2 + x + 1); a44 = load(next2 + x + 2);

                        store(detect(), dstp + x);
                    }

                    if (regularPart >= vector_t().size()) {
                        a00 = load(prev2 + regularPart - 2); a01 = load(prev2 + regularPart - 1); a02 = load(prev2 + regularPart);
                        a10 = load(prev1 + regularPart - 2); a11 = load(prev1 + regularPart - 1); a12 = load(prev1 + regularPart);
                        a20 = load(srcp0 + regularPart - 2); a21 = load(srcp0 + regularPart - 1); a22 = load(srcp0 + regularPart);
                        a30 = load(next1 + regularPart - 2); a31 = load(next1 + regularPart - 1); a32 = load(next1 + regularPart);
                        a40 = load(next2 + regularPart - 2); a41 = load(next2 + regularPart - 1); a42 = load(next2 + regularPart);

                        a03 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a02);
                        a13 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a12);
                        a23 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a22);
                        a33 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a32);
                        a43 = permute16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14>(a42);

                        a04 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a02);
                        a14 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a12);
                        a24 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a22);
                        a34 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a32);
                        a44 = permute16<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13>(a42);

                        store(detect(), dstp + regularPart);
                    }
                }

                srcp0 += stride;
                dstp += stride;
            }
        }
    }
}

template void filterAVX512<uint8_t, Tritical, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, Cross, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, Prewitt, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, Sobel, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, Scharr, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, RScharr, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, Kroon, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, Robinson3, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, Robinson5, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, Kirsch, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, ExPrewitt, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, ExSobel, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, FDoG, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint8_t, ExKirsch, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;

template void filterAVX512<uint16_t, Tritical, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, Cross, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, Prewitt, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, Sobel, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, Scharr, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, RScharr, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, Kroon, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, Robinson3, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, Robinson5, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, Kirsch, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, ExPrewitt, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, ExSobel, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, FDoG, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<uint16_t, ExKirsch, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;

template void filterAVX512<float, Tritical, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, Cross, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, Prewitt, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, Sobel, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, Scharr, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, RScharr, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, Kroon, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, Robinson3, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, Robinson5, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, Kirsch, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, ExPrewitt, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, ExSobel, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, FDoG, true>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
template void filterAVX512<float, ExKirsch, false>(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
#endif
