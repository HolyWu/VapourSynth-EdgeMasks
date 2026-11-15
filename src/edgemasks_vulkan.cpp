#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSVulkan4.h>

#include "edgemasks.h"
#include "edgemasks_params.h"

namespace edgemasks {

namespace {

// Invocations per work group, each filtering two pixels of a row.
constexpr std::uint32_t groupWidth{ 16 };
constexpr std::uint32_t groupHeight{ 8 };

// The formulas are those of edgemasks_kernel.h. Each invocation filters the pixels x and x + 1 as the two lanes of V.
constexpr std::string_view kernelSource{ R"glsl(
// The strides count pixel pairs.
layout(push_constant) uniform Plane {
    int width;
    int height;
    int srcStride;
    int dstStride;
    float scale;
    float peak;
};

layout(set = 0, binding = 0, std430) readonly buffer Src { PAIR src[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Dst { PAIR dst[]; };

// Reflects index into [0, size) without repeating the edge element.
int mirror(int index, int size) {
    return index < 0 ? -index : index < size ? index : 2 * size - 2 - index;
}

#if defined(HALF) && (OP == Kirsch || OP == ExKirsch)
// Every operator's weights sum to zero, so it sees the same edges in the differences from one sample of the window.
// Half precision resolves those far better than the samples, whose sums would otherwise cancel down to rounding errors.
// The other operators subtract samples before they sum any, so there the origin would only add a rounding.
#define ORIGIN
S origin;
#define LOAD(index) (src[index] - origin)

// The median of three samples, as one outlier among them would leave every tap of both pixels far from the origin. Left neighbours stand in past the row's end.
// Kirsch takes x and x + 1 above and x below, as only rows y - 1 and y + 1 hold samples in both pixels' rings.
// ExKirsch takes x to x + 2, which measured more accurate for its 5x5 rings.
S originAt(int x, int y, bool edge) {
#if OP == Kirsch
    const int first = (edge ? mirror(y - 1, height) : y - 1) * srcStride + (x >> 1);
    const int third = (edge ? mirror(y + 1, height) : y + 1) * srcStride + (x >> 1);
#else
    const int first = y * srcStride + (x >> 1);
    const int third = !edge || width - x > 2 ? first + 1 : first - 1;
#endif
    const S a = src[first].x;
    const S b = !edge || width - x > 1 ? src[first].y : src[first - 1].y;
    const S c = src[third].x;
    return max(min(a, b), min(max(a, b), c));
}
#else
#define LOAD(index) V(src[index])
#endif

// The pairs of row y that hold columns x - 2 to x + 3, as far as the row has them.
struct Pairs {
    V l, c, r;
};

Pairs load(int x, int y, bool edge) {
    const int center = (edge ? mirror(y, height) : y) * srcStride + (x >> 1);
    return Pairs(LOAD(center - int(!edge || x > 0)), LOAD(center), LOAD(center + int(!edge || width - x > 2)));
}

// Columns x - 2 to x + 2 for both pixels. The columns beyond the row's ends mirror onto loaded ones.
V[5] taps(int x, Pairs p, bool edge) {
    if (!edge)
        return V[5](p.l, V(p.l.y, p.c.x), p.c, V(p.c.y, p.r.x), p.r);

    const int remaining = width - x;
    const V left = x > 0 ? p.l : V(p.r.x, p.c.y);

    if (remaining == 1) {
        p.c.y = p.l.y;
        p.r = V(p.l.x);
    } else if (remaining == 2) {
        p.r = V(p.c.x, p.l.y);
    } else if (remaining == 3) {
        p.r.y = p.c.y;
    }

    return V[5](left, V(left.y, p.c.x), p.c, V(p.c.y, p.r.x), p.r);
}

#ifdef INTEGER
void store(int index, vec2 value) {
    dst[index] = PAIR(uvec2(min(value * scale + 0.5, peak)));
}

void storeMagnitude(int index, V gx, V gy) {
    const vec2 x = vec2(gx);
    const vec2 y = vec2(gy);
    store(index, sqrt(x * x + y * y));
}

void storeMaximum(int index, V maximum) {
    store(index, vec2(maximum));
}
#else
void storeMagnitude(int index, V gx, V gy) {
#ifdef HALF
    // Half precision squares overflow from 256 and underflow below 0.008, which this form avoids by squaring a ratio.
    const V x = abs(gx);
    const V y = abs(gy);
    const V longer = max(x, y);
    // Scaling by a power of two keeps nonzero divisors in [2^-14, 2^14], where half division is accurate, so the guard only spares flat pixels a division by 0.
    const V lift = mix(mix(V(1), V(1024), lessThan(longer, V(0.0625))), V(0.0009765625), greaterThanEqual(longer, V(16384)));
    const V ratio = min(x, y) * lift / max(longer * lift, S(0.00006103515625));
    dst[index] = longer * S(scale) * sqrt(S(1) + ratio * ratio);
#else
    dst[index] = sqrt(gx * gx + gy * gy) * scale;
#endif
}

void storeMaximum(int index, V maximum) {
    dst[index] = maximum * S(scale);
}
#endif

struct RingSums {
    V triples[8];  // Element i sums the ring elements i, i + 1 and i + 2.
    V total;
};

RingSums ringSums(V r0, V r1, V r2, V r3, V r4, V r5, V r6, V r7) {
    const V pair0 = r0 + r1;
    const V pair2 = r2 + r3;
    const V pair4 = r4 + r5;
    const V pair6 = r6 + r7;
    return RingSums(V[8](pair0 + r2, r1 + pair2, pair2 + r4, r3 + pair4, pair4 + r6, r5 + pair6, pair6 + r0, r7 + pair0), pair0 + pair2 + pair4 + pair6);
}

// Largest |8 * value - center|.
V maxDeviation(V v[8], V center) {
    const V highest = max(max(max(v[0], v[1]), max(v[2], v[3])), max(max(v[4], v[5]), max(v[6], v[7])));
    const V lowest = min(min(min(v[0], v[1]), min(v[2], v[3])), min(min(v[4], v[5]), min(v[6], v[7])));
    return max(S(8) * highest - center, center - S(8) * lowest);
}

// precise keeps drivers from reassociating the sums, which in half precision would mix the column differences back into samples and round at their size.
// Float samples go without, as the contractions it forbids measured 5-12% faster for planes that fit in cache.
#ifdef HALF
#define PRECISE precise
#else
#define PRECISE
#endif

V smoothed(V a[5]) {
#if OP == ExPrewitt
    PRECISE V result = a[0] + a[1] + a[2] + a[3] + a[4];
#elif OP == ExSobel
    PRECISE V result = a[0] + a[1] + S(2) * a[2] + a[3] + a[4];
#else
    PRECISE V result = a[0] + a[4] + S(2) * (a[1] + a[3]) + S(3) * a[2];
#endif
    return result;
}

V differentiated(V a[5]) {
#if OP == FDoG
    PRECISE V result = (a[0] - a[3]) + (a[1] - a[4]);
#else
    PRECISE V result = S(2) * (a[0] - a[4]) + (a[1] - a[3]);
#endif
    return result;
}

void filterPair(int x, int y, bool edge) {
    const int index = y * dstStride + (x >> 1);

#ifdef ORIGIN
    origin = originAt(x, y, edge);
#endif

#if OP < ExPrewitt
    const V row0[5] = taps(x, load(x, y - 1, edge), edge);
    const V row1[5] = taps(x, load(x, y, edge), edge);
    const V row2[5] = taps(x, load(x, y + 1, edge), edge);
    const V a00 = row0[1], a01 = row0[2], a02 = row0[3];
    const V a10 = row1[1], a12 = row1[3];
    const V a20 = row2[1], a21 = row2[2], a22 = row2[3];

#if OP == Tritical
    storeMagnitude(index, a10 - a12, a01 - a21);
#elif OP == Cross
    storeMagnitude(index, a00 - a22, a02 - a20);
#elif OP == Kirsch
    // Direction = 5 * arc - 3 * (ring - arc) = 8 * arc - 3 * ring.
    const RingSums ring = ringSums(a00, a01, a02, a12, a22, a21, a20, a10);
    storeMaximum(index, maxDeviation(ring.triples, S(3) * ring.total));
#else
    const V p = a00 - a22;
    const V q = a01 - a21;
    const V r = a02 - a20;
    const V s = a10 - a12;

#if OP == Prewitt
    storeMagnitude(index, p - r + s, p + r + q);
#elif OP == Sobel
    storeMagnitude(index, p - r + S(2) * s, p + r + S(2) * q);
#elif OP == Scharr
    storeMagnitude(index, S(3) * (p - r) + S(10) * s, S(3) * (p + r) + S(10) * q);
#elif OP == RScharr
    storeMagnitude(index, S(47) * (p - r) + S(162) * s, S(47) * (p + r) + S(162) * q);
#elif OP == Kroon
    storeMagnitude(index, S(17) * (p - r) + S(61) * s, S(17) * (p + r) + S(61) * q);
#else
    const V pq = p + q;
    const V sr = s - r;
    const V g1 = pq + r;
    const V g2 = pq + s;
    const V g3 = sr + p;
    const V g4 = sr - q;

#if OP == Robinson3
    storeMaximum(index, max(max(abs(g1), abs(g2)), max(abs(g3), abs(g4))));
#else
    // The directions are g1 + g3, g1 - g3, g2 + g4 and g2 - g4, and max(|x + y|, |x - y|) = |x| + |y|.
    storeMaximum(index, max(abs(g1) + abs(g3), abs(g2) + abs(g4)));
#endif
#endif
#endif
#elif OP == ExKirsch
    V a[5][5];

    for (int i = 0; i < 5; i++)
        a[i] = taps(x, load(x, y + i - 2, edge), edge);

    // Direction = 9 * outerArc - 7 * (outer - outerArc) + 5 * innerArc - 3 * (inner - innerArc) = 8 * (2 * outerArc + innerArc) - 7 * outer - 3 * inner.
    const RingSums outer = ringSums(a[0][0] + a[1][0], a[0][1] + a[0][2], a[0][3] + a[0][4], a[1][4] + a[2][4], a[3][4] + a[4][4], a[4][2] + a[4][3],
                                    a[4][0] + a[4][1], a[2][0] + a[3][0]);
    const V outerArcs[8] = V[8](outer.triples[0] + a[1][4], outer.triples[1] + a[3][4], outer.triples[2] + a[4][3], outer.triples[3] + a[4][1],
                                outer.triples[4] + a[3][0], outer.triples[5] + a[1][0], outer.triples[6] + a[0][1], outer.triples[7] + a[0][3]);
    const RingSums inner = ringSums(a[1][1], a[1][2], a[1][3], a[2][3], a[3][3], a[3][2], a[3][1], a[2][1]);
    V arcs[8];

    for (int i = 0; i < 8; i++)
        arcs[i] = S(2) * outerArcs[i] + inner.triples[i];

    storeMaximum(index, maxDeviation(arcs, S(7) * outer.total + S(3) * inner.total));
#else
    V l[5];
    V c[5];
    V r[5];

    for (int i = 0; i < 5; i++) {
        const Pairs p = load(x, y + i - 2, edge);
        l[i] = p.l;
        c[i] = p.c;
        r[i] = p.r;
    }

    // gx differentiates across smoothed columns and gy smooths across differentiated columns.
    // The columns mirror after they are filtered, which spares filtering the ones that both pixels share twice.
    const V gy = smoothed(taps(x, Pairs(differentiated(l), differentiated(c), differentiated(r)), edge));

#if defined(HALF) && OP == FDoG
    // Half precision would round smoothed columns at the size of their samples, so there gx smooths across differentiated rows.
    V rows[5];

    for (int i = 0; i < 5; i++)
        rows[i] = differentiated(taps(x, Pairs(l[i], c[i], r[i]), edge));

    storeMagnitude(index, smoothed(rows), gy);
#elif defined(HALF)
    // Half precision smooths each column's difference from the center pair's column of the same parity, which mirroring keeps.
    // FDoG differentiates rows instead, as only that cancels its stripes of period 3.
    V lc[5];
    V rc[5];

    for (int i = 0; i < 5; i++) {
        lc[i] = l[i] - c[i];
        rc[i] = r[i] - c[i];
    }

    V left = smoothed(lc);

    // There c.y lies in the row's padding.
    if (width - x == 1)
        left.y = S(0);

    storeMagnitude(index, differentiated(taps(x, Pairs(left, V(0), smoothed(rc)), edge)), gy);
#else
    storeMagnitude(index, differentiated(taps(x, Pairs(smoothed(l), smoothed(c), smoothed(r)), edge)), gy);
#endif
#endif
}

void main() {
    const int x = int(gl_GlobalInvocationID.x) << 1;
    const int y = int(gl_GlobalInvocationID.y);

    if (x >= width || y >= height)
        return;

    // edge is constant in each call, so the invocations away from the plane's borders run a copy of filterPair without mirroring or the fix-ups of taps().
    if (x == 0 || width - x <= 3 || y < 2 || height - y <= 2)
        filterPair(x, y, true);
    else
        filterPair(x, y, false);
}
)glsl" };

std::string shaderSource(Operator op, const VSVideoFormat& format) {
    const bool integer{ format.sampleType == stInteger };

    // VapourSynth requires 8 and 16 bit storage buffer access of every device.
    std::string source{ "#version 460\n#extension GL_EXT_shader_8bit_storage : require\n#extension GL_EXT_shader_16bit_storage : require\n" };

    if (!integer && format.bytesPerSample == 2)
        source += "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
                  "#define PAIR f16vec2\n#define V f16vec2\n#define S float16_t\n#define HALF\n";
    else if (!integer)
        source += "#define PAIR vec2\n#define V vec2\n#define S float\n";
    else
        source += std::string{ "#define PAIR " } + (format.bytesPerSample == 2 ? "u16vec2" : "u8vec2") + "\n#define V ivec2\n#define S int\n#define INTEGER\n";

    for (std::size_t i = 0; i < operatorNames.size(); i++)
        source += std::string{ "#define " } + operatorNames[i] + " " + std::to_string(i) + "\n";

    source += "layout(local_size_x = " + std::to_string(groupWidth) + ", local_size_y = " + std::to_string(groupHeight) + ") in;\n";
    return (source + "#define OP " + name(op) + "\n").append(kernelSource);
}

struct PlaneConstants {
    std::int32_t width;
    std::int32_t height;
    std::int32_t srcStride;
    std::int32_t dstStride;
    float scale;
    float peak;
};

constexpr int errorSize{ 512 };
using ErrorMessage = std::array<char, errorSize>;

class VulkanData {
public:
    FilterParams params;
    const VSVULKANAPI* vkapi{};
    const VSVulkanFunctions* vk{};
    VkDevice device{};
    VkDescriptorSetLayout setLayout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline pipeline{};
    VSGPUExecPool* pool{};

    VulkanData(FilterParams filterParams, VSCore* core, const VSAPI* vsapi);
    VulkanData(const VulkanData&) = delete;
    VulkanData& operator=(const VulkanData&) = delete;
    VulkanData(VulkanData&&) = delete;
    VulkanData& operator=(VulkanData&&) = delete;

    // Submissions use the pipeline, so a drain that cannot establish their completion leaks everything, as the core's GPU filters do.
    // The pool leaks too and otherwise goes last, because its device reference keeps the device alive under what is left.
    ~VulkanData() {
        ErrorMessage error{};

        if (!vk || (pool && !vsGPUDrainSafeToDestroy(vkapi->gpuExecPoolWaitIdle(pool, error.data(), errorSize))))
            return;

        vk->vkDestroyPipeline(device, pipeline, nullptr);
        vk->vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vk->vkDestroyDescriptorSetLayout(device, setLayout, nullptr);

        if (pool)
            vkapi->freeGPUExecPool(pool);
    }

private:
    // The public constructor delegates to this one, so that the destructor releases what it created before it threw.
    VulkanData(FilterParams filterParams, const VSVULKANAPI* vulkanApi) noexcept : params{ std::move(filterParams) }, vkapi{ vulkanApi } {}
};

struct ShaderDeleter {
    const VSVULKANAPI* vkapi;
    void operator()(VSGPUShader* shader) const noexcept { vkapi->freeGPUShader(shader); }
};

// The float widths whose denormal mode the device ties to that of 16 bit, 16 included, or none if it cannot preserve the denormals of them all.
std::vector<std::uint32_t> denormPreserveWidths(const VkPhysicalDeviceVulkan12Properties& properties) {
    const bool tied32{ properties.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE };
    const bool tied64{ properties.denormBehaviorIndependence != VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL };

    if (!properties.shaderDenormPreserveFloat16 || (tied32 && !properties.shaderDenormPreserveFloat32) || (tied64 && !properties.shaderDenormPreserveFloat64))
        return {};

    std::vector<std::uint32_t> widths{ 16 };

    if (tied32)
        widths.push_back(32);

    if (tied64)
        widths.push_back(64);

    return widths;
}

// The SPIR-V module with the DenormPreserve execution mode for each of widths, which glslang can declare for one width only.
std::vector<std::uint32_t> withDenormPreserve(std::span<const std::uint32_t> code, std::span<const std::uint32_t> widths) {
    constexpr std::size_t headerWords{ 5 };
    constexpr std::uint32_t opEntryPoint{ 15 };
    constexpr std::uint32_t opExecutionMode{ 16 };
    constexpr std::uint32_t opCapability{ 17 };
    constexpr std::uint32_t modeDenormPreserve{ 4459 };
    constexpr std::uint32_t capabilityDenormPreserve{ 4464 };

    std::vector<std::uint32_t> words(code.begin(), code.begin() + headerWords);
    words.insert(words.end(), { 2u << 16 | opCapability, capabilityDenormPreserve });

    // The logical layout puts execution modes after the entry points, and a compute shader from glslang has one.
    for (std::size_t i = headerWords; i < code.size(); i += code[i] >> 16) {
        const std::span instruction{ code.subspan(i, code[i] >> 16) };
        words.insert(words.end(), instruction.begin(), instruction.end());

        if ((code[i] & 0xFFFF) == opEntryPoint)
            for (const std::uint32_t width : widths)
                words.insert(words.end(), { 4u << 16 | opExecutionMode, code[i + 2], modeDenormPreserve, width });
    }

    return words;
}

std::array<std::uint32_t, 2> groupCounts(int width, int height) noexcept {
    constexpr std::uint32_t pixelsWide{ 2 * groupWidth };
    return { (static_cast<std::uint32_t>(width) + pixelsWide - 1) / pixelsWide, (static_cast<std::uint32_t>(height) + groupHeight - 1) / groupHeight };
}

VulkanData::VulkanData(FilterParams filterParams, VSCore* core, const VSAPI* vsapi) : VulkanData{ std::move(filterParams), vsapi->getVulkanAPI() } {
    const VSVideoFormat& format{ params.vi->format };
    ErrorMessage error{};
    VSVulkanCoreHandles handles{};

    if (vkapi->getVulkanHandles(core, &handles, error.data(), errorSize))
        throw std::runtime_error{ error.data() };

    vk = vkapi->getVulkanFunctions(core, error.data(), errorSize);

    if (!vk)
        throw std::runtime_error{ error.data() };

    device = handles.device;

    VkPhysicalDeviceVulkan12Properties properties12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
    VkPhysicalDeviceProperties2 properties{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &properties12 };
    vk->vkGetPhysicalDeviceProperties2(handles.physicalDevice, &properties);
    const VkPhysicalDeviceLimits& limits{ properties.properties.limits };
    // The shader binds whole planes, and GPU frames have the strides of CPU frames of the same format and width.
    const FramePtr strideProbe{ vsapi->newVideoFrame(&format, params.vi->width, 1 << format.subSamplingH, nullptr, core), FrameDeleter{ vsapi } };

    for (int plane = 0; plane < format.numPlanes; plane++) {
        if (!params.process[plane])
            continue;

        const auto [planeWidth, planeHeight] = planeDimensions(*params.vi, plane);
        const auto [groupsX, groupsY] = groupCounts(planeWidth, planeHeight);
        const VkDeviceSize planeSize{ static_cast<VkDeviceSize>(vsapi->getStride(strideProbe.get(), plane)) * static_cast<VkDeviceSize>(planeHeight) };

        if (groupsX > limits.maxComputeWorkGroupCount[0] || groupsY > limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error{ "plane is larger than the Vulkan device's maximum work group count covers" };

        if (planeSize > limits.maxStorageBufferRange)
            throw std::runtime_error{ "plane is larger than the Vulkan device's storage buffer range" };
    }

    // Half precision takes the differences between dark samples into the subnormal range, which a device may flush unless the shader declares otherwise.
    std::vector<std::uint32_t> denormWidths;

    if (format.sampleType == stFloat && format.bytesPerSample == 2) {
        VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        VkPhysicalDeviceFeatures2 features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features12 };
        vk->vkGetPhysicalDeviceFeatures2(handles.physicalDevice, &features);

        if (!features12.shaderFloat16)
            throw std::runtime_error{ "16 bit float needs the shaderFloat16 feature, which the Vulkan device lacks" };

        denormWidths = denormPreserveWidths(properties12);

        // The shader multiplies by the scale, which the operator's divisor is part of, in half precision. There it has to be a normal number.
        for (int plane = 0; plane < format.numPlanes; plane++) {
            const float scale{ params.scale[plane] };

            if (params.process[plane] && (scale < 0x1p-14f || scale > 65504.0f))
                throw std::runtime_error{ "scale is out of range for 16 bit float" };
        }
    }

    constexpr auto storageBuffer = [](std::uint32_t binding) noexcept {
        return VkDescriptorSetLayoutBinding{
            .binding = binding, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };
    };

    const std::array bindings{ storageBuffer(0), storageBuffer(1) };
    const VkDescriptorSetLayoutCreateInfo setLayoutInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                         .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
                                                         .bindingCount = static_cast<std::uint32_t>(bindings.size()),
                                                         .pBindings = bindings.data() };

    // Created into locals, as a failed create leaves its output undefined for the destructor.
    VkDescriptorSetLayout newSetLayout{};

    if (vk->vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &newSetLayout) != VK_SUCCESS)
        throw std::runtime_error{ "failed to create the descriptor set layout" };

    setLayout = newSetLayout;

    const VkPushConstantRange pushConstants{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(PlaneConstants) };
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                         .setLayoutCount = 1,
                                                         .pSetLayouts = &setLayout,
                                                         .pushConstantRangeCount = 1,
                                                         .pPushConstantRanges = &pushConstants };

    VkPipelineLayout newPipelineLayout{};

    if (vk->vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &newPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error{ "failed to create the pipeline layout" };

    pipelineLayout = newPipelineLayout;

    const std::unique_ptr<VSGPUShader, ShaderDeleter> shader{
        vkapi->compileGPUShader(core, slGLSL, shaderSource(params.op, format).c_str(), error.data(), errorSize), ShaderDeleter{ vkapi }
    };

    if (!shader)
        throw std::runtime_error{ error.data() };

    // The pipeline takes the code straight from the stage, which needs no shader module.
    VkShaderModuleCreateInfo moduleInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    moduleInfo.pCode = vkapi->getGPUShaderCode(shader.get(), &moduleInfo.codeSize);
    std::vector<std::uint32_t> patchedCode;

    if (!denormWidths.empty()) {
        patchedCode = withDenormPreserve({ moduleInfo.pCode, moduleInfo.codeSize / sizeof(std::uint32_t) }, denormWidths);
        moduleInfo.pCode = patchedCode.data();
        moduleInfo.codeSize = patchedCode.size() * sizeof(std::uint32_t);
    }

    const VkComputePipelineCreateInfo pipelineInfo{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                    .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                               .pNext = &moduleInfo,
                                                               .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                               .pName = "main" },
                                                    .layout = pipelineLayout };

    if (vk->vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error{ "failed to create the compute pipeline" };

    pool = vkapi->createGPUExecPool(core, vqCompute, error.data(), errorSize);

    if (!pool)
        throw std::runtime_error{ error.data() };
}

void recordPlane(const VulkanData& d, VkCommandBuffer cmd, const VSFrame* src, const VSFrame* dst, int plane, const VSAPI* vsapi) noexcept {
    const std::ptrdiff_t pairSize{ 2 * d.params.vi->format.bytesPerSample };
    const PlaneConstants constants{ .width = vsapi->getFrameWidth(src, plane),
                                    .height = vsapi->getFrameHeight(src, plane),
                                    .srcStride = static_cast<std::int32_t>(vsapi->getStride(src, plane) / pairSize),
                                    .dstStride = static_cast<std::int32_t>(vsapi->getStride(dst, plane) / pairSize),
                                    .scale = d.params.scale[plane],
                                    .peak = d.params.peak };

    VSVulkanPlaneInfo srcPlane{};
    VSVulkanPlaneInfo dstPlane{};
    d.vkapi->getGPUPlane(src, plane, &srcPlane);
    d.vkapi->getGPUPlane(dst, plane, &dstPlane);

    const std::array buffers{ VkDescriptorBufferInfo{ .buffer = srcPlane.buffer, .range = VK_WHOLE_SIZE },
                              VkDescriptorBufferInfo{ .buffer = dstPlane.buffer, .range = VK_WHOLE_SIZE } };

    // One write per binding, as GPU-AV does not follow a write's descriptors on into the next binding.
    const auto write = [&](std::uint32_t binding) noexcept {
        return VkWriteDescriptorSet{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                     .dstBinding = binding,
                                     .descriptorCount = 1,
                                     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     .pBufferInfo = &buffers[binding] };
    };

    const std::array writes{ write(0), write(1) };
    d.vk->vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipelineLayout, 0, static_cast<std::uint32_t>(writes.size()), writes.data());

    const VkPushConstantsInfo pushInfo{ .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
                                        .layout = d.pipelineLayout,
                                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        .size = sizeof(constants),
                                        .pValues = &constants };
    d.vk->vkCmdPushConstants2(cmd, &pushInfo);

    const auto [groupsX, groupsY] = groupCounts(constants.width, constants.height);
    d.vk->vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

const VSFrame* VS_CC vulkanGetFrame(int n, int activationReason, void* instanceData, void**, VSFrameContext* frameCtx, VSCore* core,
                                    const VSAPI* vsapi) noexcept {
    const auto& d = *static_cast<const VulkanData*>(instanceData);
    const auto& [node, vi, process, scale, peak, op] = d.params;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, node.get(), frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const FramePtr src{ vsapi->getFrameFilter(n, node.get(), frameCtx), FrameDeleter{ vsapi } };
        const bool sharesPlanes{ std::any_of(process.begin(), process.begin() + vi->format.numPlanes, std::logical_not{}) };
        // newVideoFrame2 takes the residency from the planes it shares, so it cannot make a GPU frame without one.
        NewFramePtr dst{ sharesPlanes ? newSharingFrame(d.params, src.get(), core, vsapi)
                                      : NewFramePtr{ d.vkapi->newGPUVideoFrame(&vi->format, vi->width, vi->height, src.get(), core), FrameDeleter{ vsapi } } };

        ErrorMessage error{};
        VSGPUExecContext* const context = d.vkapi->gpuExecAcquire(d.pool, error.data(), errorSize);

        if (!context) {
            vsapi->setFilterError((std::string{ name(op) } + ": " + error.data()).c_str(), frameCtx);
            return nullptr;
        }

        d.vkapi->gpuExecReadsFrame(context, src.get());

        const VkCommandBuffer cmd{ d.vkapi->gpuExecCommandBuffer(context) };
        d.vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);

        for (int plane = 0; plane < vi->format.numPlanes; plane++) {
            if (process[plane]) {
                d.vkapi->gpuExecWritesPlane(context, dst.get(), plane);
                recordPlane(d, cmd, src.get(), dst.get(), plane, vsapi);
            }
        }

        if (d.vkapi->gpuExecSubmit(context, nullptr, error.data(), errorSize)) {
            vsapi->setFilterError((std::string{ name(op) } + ": " + error.data()).c_str(), frameCtx);
            return nullptr;
        }

        vsapi->mapSetInt(vsapi->getFramePropertiesRW(dst.get()), "_Range", VSC_RANGE_FULL, maReplace);
        return dst.release();
    }

    return nullptr;
}

}  // namespace

void createVulkanFilter(FilterParams params, VSMap* out, VSCore* core, const VSAPI* vsapi) {
    createNode(std::make_unique<VulkanData>(std::move(params), core, vsapi), vulkanGetFrame, ffGPUOutput, out, core, vsapi);
}

}  // namespace edgemasks
