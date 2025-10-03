#pragma once

#include <string>
#include <type_traits>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#ifdef EDGEMASKS_X86
#include "vectorclass/vectorclass.h"
#endif

struct EdgeMasksData final {
    VSNode* node;
    const VSVideoInfo* vi;
    bool process[3];
    float scale;
    int matrix, peak;
    std::string filterName;
    void (*filter)(const VSFrame* src, VSFrame* dst, const EdgeMasksData* VS_RESTRICT d, const VSAPI* vsapi) noexcept;
};

enum Operator {
    Prewitt,
    Sobel,
    Scharr,
    RScharr,
    Kroon,
    Robinson3,
    Robinson5,
    Kirsch,
    ExPrewitt,
    ExSobel,
    FDoG,
    ExKirsch
};
