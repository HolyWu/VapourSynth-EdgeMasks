#ifndef VAPOURSYNTH_EDGEMASKS_PARAMS_H
#define VAPOURSYNTH_EDGEMASKS_PARAMS_H

#include <array>
#include <memory>
#include <tuple>

#include <VapourSynth4.h>

#include "edgemasks.h"

namespace edgemasks {

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
using NewFramePtr = std::unique_ptr<VSFrame, FrameDeleter>;

template<typename Data>
void VS_CC freeInstance(void* instanceData, VSCore*, const VSAPI*) noexcept {
    const std::unique_ptr<Data> d{ static_cast<Data*>(instanceData) };
}

template<typename Data>
void createNode(std::unique_ptr<Data> d, VSFilterGetFrame getFrame, int flags, VSMap* out, VSCore* core, const VSAPI* vsapi) {
    const VSFilterDependency dependency{ d->params.node.get(), rpStrictSpatial };
    vsapi->createVideoFilterEx(out, name(d->params.op), d->params.vi, getFrame, freeInstance<Data>, fmParallel, flags, &dependency, 1, d.get(), core);
    if (!vsapi->mapGetError(out))
        std::ignore = d.release();
}

struct FilterParams {
    NodePtr node;
    const VSVideoInfo* vi;
    PerPlane<bool> process;
    PerPlane<float> scale;
    float peak;
    Operator op;
};

inline NewFramePtr newSharingFrame(const FilterParams& params, const VSFrame* src, VSCore* core, const VSAPI* vsapi) noexcept {
    PerPlane<const VSFrame*> planeSrc{ params.process[0] ? nullptr : src, params.process[1] ? nullptr : src, params.process[2] ? nullptr : src };
    constexpr std::array planes{ 0, 1, 2 };
    return { vsapi->newVideoFrame2(&params.vi->format, params.vi->width, params.vi->height, planeSrc.data(), planes.data(), src, core), FrameDeleter{ vsapi } };
}

constexpr std::array<int, 2> planeDimensions(const VSVideoInfo& vi, int plane) noexcept {
    return { vi.width >> (plane > 0 ? vi.format.subSamplingW : 0), vi.height >> (plane > 0 ? vi.format.subSamplingH : 0) };
}

void createVulkanFilter(FilterParams params, VSMap* out, VSCore* core, const VSAPI* vsapi);

}  // namespace edgemasks

#endif  // VAPOURSYNTH_EDGEMASKS_PARAMS_H
