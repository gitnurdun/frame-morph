#include "cuda_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <set>
#include <stdexcept>
#include <tuple>

namespace fm {
namespace {
void check(cudaError_t e, const char *where) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}

struct EventRecorder {
    struct Sample {
        std::string name;
        cudaEvent_t start = nullptr, end = nullptr;
    };
    std::vector<Sample> samples;
    GpuProfile result;
    ~EventRecorder() {
        for (auto &s : samples) {
            if (s.start)
                cudaEventDestroy(s.start);
            if (s.end)
                cudaEventDestroy(s.end);
        }
    }
    size_t begin(const char *name) {
        samples.push_back({name});
        auto &s = samples.back();
        check(cudaEventCreate(&s.start), "create profiling event");
        check(cudaEventCreate(&s.end), "create profiling event");
        check(cudaEventRecord(s.start), "record profiling start");
        return samples.size() - 1;
    }
    void finish(size_t id) {
        cudaEventRecord(samples[id].end);
    }
    void resolve() {
        for (auto &s : samples) {
            check(cudaEventSynchronize(s.end), "finish profiling");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, s.start, s.end), "read profiling event");
            result.stages[s.name] += ms;
        }
    }
};
thread_local EventRecorder *activeRecorder = nullptr;
struct Interval {
    EventRecorder *recorder = activeRecorder;
    size_t id = 0;
    explicit Interval(const char *name) {
        if (recorder)
            id = recorder->begin(name);
    }
    ~Interval() {
        if (recorder)
            recorder->finish(id);
    }
};
struct RecorderScope {
    EventRecorder *previous;
    explicit RecorderScope(EventRecorder *next) : previous(activeRecorder) {
        activeRecorder = next;
    }
    ~RecorderScope() {
        activeRecorder = previous;
    }
};
template <class T> struct DeviceBuffer {
    T *data = nullptr;
    size_t capacity = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() {
        if (data)
            cudaFree(data);
    }
    void resize(size_t n) {
        if (n <= capacity)
            return;
        if (data)
            check(cudaFree(data), "release CUDA buffer");
        data = nullptr;
        capacity = 0;
        check(cudaMalloc(reinterpret_cast<void **>(&data), n * sizeof(T)),
              "allocate CUDA buffer (reduce preview size if memory is exhausted)");
        capacity = n;
    }
    void upload(const T *src, size_t n) {
        resize(n);
        if (n) {
            Interval timing("upload");
            if (activeRecorder)
                activeRecorder->result.uploadedBytes += n * sizeof(T);
            check(cudaMemcpy(data, src, n * sizeof(T), cudaMemcpyHostToDevice),
                  "upload CUDA buffer");
        }
    }
};
__device__ float linear(float c) {
    c /= 255.f;
    return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}
__device__ unsigned char srgb(float c) {
    c = fminf(1.f, fmaxf(0.f, c));
    float s = c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
    return (unsigned char)(s * 255.f + 0.5f);
}
__device__ float labf(float x) {
    return x > 0.00885645f ? cbrtf(x) : 7.787037f * x + 16.f / 116.f;
}
__device__ float3 lab(const unsigned char *bgr) {
    float r = linear(bgr[2]), g = linear(bgr[1]), b = linear(bgr[0]);
    float x = labf((.4124564f * r + .3575761f * g + .1804375f * b) / .95047f);
    float y = labf(.2126729f * r + .7151522f * g + .0721750f * b);
    float z = labf((.0193339f * r + .1191920f * g + .9503041f * b) / 1.08883f);
    return make_float3(116 * y - 16, 500 * (x - y), 200 * (y - z));
}
__global__ void initLabels(const unsigned char *image, int *parent, int n, int seed,
                           float tolerance) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float3 a = lab(image + 3 * i), b = lab(image + 3 * seed);
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    parent[i] = d <= tolerance * tolerance ? i : -1;
}
__device__ int root(int *p, int i) {
    int r = atomicAdd(p + i, 0);
    while (r >= 0 && r != i) {
        int next = atomicAdd(p + r, 0);
        if (next >= 0)
            atomicMin(p + i, next);
        i = r;
        r = next;
    }
    return r;
}
__device__ void unite(int *p, int a, int b) {
    while (true) {
        a = root(p, a);
        b = root(p, b);
        if (a < 0 || b < 0 || a == b)
            return;
        int hi = max(a, b), lo = min(a, b);
        int old = atomicCAS(p + hi, hi, lo);
        if (old == hi)
            return;
        a = old;
        b = lo;
    }
}
__global__ void connect(int *p, int w, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h || p[i] < 0)
        return;
    int x = i % w, y = i / w;
    if (x + 1 < w && p[i + 1] >= 0)
        unite(p, i, i + 1);
    if (y + 1 < h) {
        if (p[i + w] >= 0)
            unite(p, i, i + w);
        if (x > 0 && p[i + w - 1] >= 0)
            unite(p, i, i + w - 1);
        if (x + 1 < w && p[i + w + 1] >= 0)
            unite(p, i, i + w + 1);
    }
}
__global__ void extract(int *p, unsigned char *mask, int n, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        mask[i] = (p[i] >= 0 && root(p, i) == root(p, seed)) ? 255 : 0;
}
__global__ void background(const unsigned char *a, const unsigned char *b, float4 *frame, int n,
                           float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    frame[i] = make_float4(linear(a[3 * i + 2]) * (1 - t) + linear(b[3 * i + 2]) * t,
                           linear(a[3 * i + 1]) * (1 - t) + linear(b[3 * i + 1]) * t,
                           linear(a[3 * i]) * (1 - t) + linear(b[3 * i]) * t, 1);
}
__global__ void interpolate(const GpuPoint *a, const GpuPoint *b, GpuPoint *out, int n, float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = {a[i].x * (1 - t) + b[i].x * t, a[i].y * (1 - t) + b[i].y * t};
}
// Only edges spanning this 16-pixel row band can cross a sample's ray.
// Unlike testing every contour edge per pixel, this scales well for long paths and glyphs.
__device__ bool inside(float x, float y, const GpuPoint *points, const GpuContour *edges,
                       const int *offsets, const int *refs, int band) {
    bool result = false;
    for (int i = offsets[band]; i < offsets[band + 1]; ++i) {
        auto edge = edges[refs[i]];
        GpuPoint a = points[edge.begin], b = points[edge.count];
        if ((a.y > y) != (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x)
            result = !result;
    }
    return result;
}
__global__ void raster(float4 *group, float *coverage, const GpuPoint *points,
                       const GpuContour *edges, const int *offsets, const int *refs, int w, int h,
                       int ox, int oy, int bw, int bh, float4 color) {
    int x = ox + blockIdx.x * blockDim.x + threadIdx.x,
        y = oy + blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || y < 0 || x >= w || y >= h || x >= ox + bw || y >= oy + bh)
        return;
    float cov = 0;
    for (int sy = 0; sy < 4; ++sy)
        for (int sx = 0; sx < 4; ++sx)
            cov += inside(x + .125f + .25f * sx, y + .125f + .25f * sy, points, edges, offsets,
                          refs, y / 16)
                       ? .0625f
                       : 0;
    cov *= color.w;
    if (cov <= 0)
        return;
    int i = y * w + x;
    float4 v = group[i];
    v.x += color.x * cov;
    v.y += color.y * cov;
    v.z += color.z * cov;
    v.w += cov;
    group[i] = v;
    coverage[i] = fmaxf(coverage[i], cov);
}

__global__ void clearRegion(float4 *group, float *coverage, int width, int ox, int oy, int rw,
                            int rh) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= rw * rh)
        return;
    int i = (oy + local / rw) * width + ox + local % rw;
    group[i] = make_float4(0, 0, 0, 0);
    coverage[i] = 0;
}
cv::Rect drawRegion(const GpuDraw &draw, int width, int height) {
    cv::Rect canvas(0, 0, width, height);
    if (draw.triangles.empty())
        return draw.bounds & canvas;
    float left = float(width), top = float(height), right = 0, bottom = 0;
    for (size_t i = 0; i < draw.from.size(); ++i) {
        float x = draw.from[i].x * (1 - draw.t) + draw.to[i].x * draw.t,
              y = draw.from[i].y * (1 - draw.t) + draw.to[i].y * draw.t;
        if (!std::isfinite(x) || !std::isfinite(y))
            continue;
        left = std::min(left, x);
        top = std::min(top, y);
        right = std::max(right, x);
        bottom = std::max(bottom, y);
    }
    int x0 = std::max(0, int(std::floor(left)) - 2), y0 = std::max(0, int(std::floor(top)) - 2),
        x1 = std::min(width, int(std::ceil(right)) + 2),
        y1 = std::min(height, int(std::ceil(bottom)) + 2);
    return x1 > x0 && y1 > y0 ? cv::Rect(x0, y0, x1 - x0, y1 - y0) : cv::Rect{};
}
__device__ float encodeCoverage(float c) {
    c = fminf(1.f, fmaxf(0.f, c));
    return c <= .0031308f ? 12.92f * c : 1.055f * powf(c, 1.f / 2.4f) - .055f;
}
__global__ void composite(float4 *frame, const float4 *group, const float *coverage, int width,
                          int ox, int oy, int rw, int rh) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= rw * rh)
        return;
    int i = (oy + local / rw) * width + ox + local % rw;
    float4 g = group[i];
    if (g.w <= 0)
        return;
    float a = fminf(1.f, coverage[i]);
    float4 f = frame[i];
    float3 color = make_float3(g.x / g.w, g.y / g.w, g.z / g.w);
    if (a >= 1.f) {
        frame[i] = make_float4(color.x, color.y, color.z, 1);
        return;
    }
    // Match the authored PNG/CPU preview's sRGB coverage compositing. Colors and texture
    // interpolation remain linear-light; alpha coverage must not change at playback start.

    frame[i] =
        make_float4(linear(255 * (encodeCoverage(f.x) * (1 - a) + encodeCoverage(color.x) * a)),
                    linear(255 * (encodeCoverage(f.y) * (1 - a) + encodeCoverage(color.y) * a)),
                    linear(255 * (encodeCoverage(f.z) * (1 - a) + encodeCoverage(color.z) * a)), 1);
}
__device__ float4 texel(const unsigned char *texture, int w, int h, int x, int y) {
    if (x < 0 || y < 0 || x >= w || y >= h)
        return make_float4(0, 0, 0, 0);
    const unsigned char *p = texture + 4 * (y * w + x);
    float a = p[3] / 255.f;
    return make_float4(linear(p[2]) * a, linear(p[1]) * a, linear(p[0]) * a, a);
}
__device__ float4 sample(const unsigned char *texture, int w, int h, float u, float v) {
    float x = u * w - .5f, y = v * h - .5f;
    int ix = floorf(x), iy = floorf(y);
    float fx = x - ix, fy = y - iy;
    float4 out = make_float4(0, 0, 0, 0);
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            auto c = texel(texture, w, h, ix + i, iy + j);
            float weight = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
            out.x += c.x * weight;
            out.y += c.y * weight;
            out.z += c.z * weight;
            out.w += c.w * weight;
        }
    return out;
}
struct TextureView {
    const unsigned char *data[16]{};
    int width[16]{},height[16]{},count=1;
    bool filtered=false;
};
// Area-average encoded RGB weighted by alpha. Transparent RGB never contaminates edges.
__global__ void prefilterTexture(const unsigned char *src,int sw,int sh,unsigned char *dst,int dw,int dh){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=dw||y>=dh)return;
    float l=float(x)*sw/dw,r=float(x+1)*sw/dw,t=float(y)*sh/dh,b=float(y+1)*sh/dh;
    float total=(r-l)*(b-t),alpha=0;float3 color=make_float3(0,0,0);
    for(int yy=int(floorf(t));yy<min(sh,int(ceilf(b)));++yy)
        for(int xx=int(floorf(l));xx<min(sw,int(ceilf(r)));++xx){
            float weight=(fminf(r,xx+1.f)-fmaxf(l,float(xx)))*(fminf(b,yy+1.f)-fmaxf(t,float(yy)));
            auto p=src+4*(yy*sw+xx);float a=p[3]*weight;alpha+=a;
            color.x+=p[0]*a;color.y+=p[1]*a;color.z+=p[2]*a;
        }
    auto out=dst+4*(y*dw+x);
    out[0]=alpha>0?(unsigned char)(color.x/alpha+.5f):0;
    out[1]=alpha>0?(unsigned char)(color.y/alpha+.5f):0;
    out[2]=alpha>0?(unsigned char)(color.z/alpha+.5f):0;
    out[3]=(unsigned char)(alpha/total+.5f);
}
__device__ float4 filteredSample(const TextureView &texture,float u,float v,float dux,float dvx,float duy,float dvy){
    if(!texture.filtered)return sample(texture.data[0],texture.width[0],texture.height[0],u,v);
    float xFoot=hypotf(dux*texture.width[0],dvx*texture.height[0]);
    float yFoot=hypotf(duy*texture.width[0],dvy*texture.height[0]);
    float lod=fmaxf(0.f,log2f(fmaxf(1.f,fminf(xFoot,yFoot))));
    lod=fminf(lod,float(texture.count-1));int lo=int(floorf(lod)),hi=min(lo+1,texture.count-1);float blend=lod-lo;
    float scale=exp2f(lod);
    int nx=min(8,max(1,int(ceilf(xFoot/scale-.0001f)))),ny=min(8,max(1,int(ceilf(yFoot/scale-.0001f))));
    float4 result=make_float4(0,0,0,0);
    for(int j=0;j<ny;++j)for(int i=0;i<nx;++i){
        float ox=(i+.5f)/nx-.5f,oy=(j+.5f)/ny-.5f;
        float uu=u+dux*ox+duy*oy,vv=v+dvx*ox+dvy*oy;
        auto a=sample(texture.data[lo],texture.width[lo],texture.height[lo],uu,vv);
        auto b=sample(texture.data[hi],texture.width[hi],texture.height[hi],uu,vv);
        result.x+=a.x+(b.x-a.x)*blend;result.y+=a.y+(b.y-a.y)*blend;
        result.z+=a.z+(b.z-a.z)*blend;result.w+=a.w+(b.w-a.w)*blend;
    }
    float weight=1.f/(nx*ny);result.x*=weight;result.y*=weight;result.z*=weight;result.w*=weight;return result;
}
__global__ void sprite(float4 *group, float *coverage, TextureView a, TextureView b, int w, int h, int ox, int oy, int rw,
                       int rh, float cx, float cy, float ew, float eh, float angle, float t,
                       float opacity, float opacityA, float opacityB, float4 clip) {
    int x = ox + blockIdx.x * blockDim.x + threadIdx.x,
        y = oy + blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || y < 0 || x >= w || y >= h || x >= ox + rw || y >= oy + rh)
        return;
    float dx = x + .5f - cx, dy = y + .5f - cy, co = cosf(angle), si = sinf(angle);
    float u = (co * dx + si * dy) / ew + .5f, v = (-si * dx + co * dy) / eh + .5f;
    float marginU=a.count>1||b.count>1?.5f/ew:0,marginV=a.count>1||b.count>1?.5f/eh:0;
    if(u < -marginU || v < -marginV || u > 1+marginU || v > 1+marginV)return;
    float reveal = 1;
    if (clip.x > 0)
        reveal = fminf(reveal, fmaxf(0.f, fminf(1.f, (u - clip.x) * ew + .5f)));
    if (clip.y > 0)
        reveal = fminf(reveal, fmaxf(0.f, fminf(1.f, (v - clip.y) * eh + .5f)));
    if (clip.z < 1)
        reveal = fminf(reveal, fmaxf(0.f, fminf(1.f, (clip.z - u) * ew + .5f)));
    if (clip.w < 1)
        reveal = fminf(reveal, fmaxf(0.f, fminf(1.f, (clip.w - v) * eh + .5f)));
    opacity *= reveal;
    float4 ca=filteredSample(a,u,v,co/ew,-si/eh,si/ew,co/eh),
           cb=filteredSample(b,u,v,co/ew,-si/eh,si/ew,co/eh);
    float wa = (1 - t) * opacityA, wb = t * opacityB;
    float alpha = (ca.w * wa + cb.w * wb) * opacity;
    if (alpha <= 0)
        return;
    int i = y * w + x;
    auto g = group[i];
    g.x += (ca.x * wa + cb.x * wb) * opacity;
    g.y += (ca.y * wa + cb.y * wb) * opacity;
    g.z += (ca.z * wa + cb.z * wb) * opacity;
    g.w += alpha;
    group[i] = g;
    coverage[i] = fmaxf(coverage[i], alpha);
}
__global__ void output(const float4 *frame, unsigned char *image, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float4 c = frame[i];
    image[3 * i] = srgb(c.z);
    image[3 * i + 1] = srgb(c.y);
    image[3 * i + 2] = srgb(c.x);
}
__global__ void meshRaster(float4 *group, float *coverage, const GpuPoint *positions,
                           const GpuPoint *uv, const int *triangles, const int *offsets,
                           const int *references, const unsigned char *texture, int tw, int th,
                           int w, int h, int tilesX, float opacity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    int tile = (y / 16) * tilesX + x / 16;
    float4 accumulated = make_float4(0, 0, 0, 0);
    for (int sy = 0; sy < 2; ++sy)
        for (int sx = 0; sx < 2; ++sx) {
            float px = x + .25f + .5f * sx, py = y + .25f + .5f * sy;
            bool hit = false;
            float uu = 0, vv = 0;
            for (int ref = offsets[tile]; ref < offsets[tile + 1]; ++ref) {
                int tri = references[ref] * 3, ia = triangles[tri], ib = triangles[tri + 1],
                    ic = triangles[tri + 2];
                auto a = positions[ia], b = positions[ib], c = positions[ic];
                float denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
                if (fabsf(denominator) < 1e-8f)
                    continue;
                float wa = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denominator,
                      wb = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denominator,
                      wc = 1 - wa - wb;
                if (wa >= -1e-5f && wb >= -1e-5f && wc >= -1e-5f) {
                    uu = uv[ia].x * wa + uv[ib].x * wb + uv[ic].x * wc;
                    vv = uv[ia].y * wa + uv[ib].y * wb + uv[ic].y * wc;
                    hit = true;
                }
            }
            if (hit) {
                auto color = sample(texture, tw, th, uu, vv);
                accumulated.x += color.x * .25f;
                accumulated.y += color.y * .25f;
                accumulated.z += color.z * .25f;
                accumulated.w += color.w * .25f;
            }
        }
    accumulated.x *= opacity;
    accumulated.y *= opacity;
    accumulated.z *= opacity;
    accumulated.w *= opacity;
    int i = y * w + x;
    auto g = group[i];
    g.x += accumulated.x;
    g.y += accumulated.y;
    g.z += accumulated.z;
    g.w += accumulated.w;
    group[i] = g;
    coverage[i] = fmaxf(coverage[i], accumulated.w);
}
__global__ void transparentOutput(const float4 *group, const float *coverage, unsigned char *rgba,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    auto c = group[i];
    float a = coverage[i];
    rgba[4 * i] = c.w > 0 ? srgb(c.z / c.w) : 0;
    rgba[4 * i + 1] = c.w > 0 ? srgb(c.y / c.w) : 0;
    rgba[4 * i + 2] = c.w > 0 ? srgb(c.x / c.w) : 0;
    rgba[4 * i + 3] = (unsigned char)(fminf(1.f, a) * 255 + .5f);
}
} // namespace
struct CudaBackend::Impl {
    bool profiling = false;
    bool regionCompositing = true;
    GpuProfile lastProfile;

    DeviceBuffer<unsigned char> imageA, imageB, result, selection;
    DeviceBuffer<int> parent;
    DeviceBuffer<float4> frame, group;
    DeviceBuffer<float> coverage;
    DeviceBuffer<GpuPoint> a, b, points;
    DeviceBuffer<GpuContour> curves;
    DeviceBuffer<unsigned char> textureA, textureB;
    DeviceBuffer<GpuPoint> uv;
    DeviceBuffer<int> triangles, offsets, references;
    using InputKey = std::tuple<uintptr_t, int, int, int, size_t>;
    struct ResidentInput {
        cv::Mat owner;
        DeviceBuffer<unsigned char> gpu;
        size_t cost = 0;
        uint64_t used = 0;
        std::vector<std::unique_ptr<DeviceBuffer<unsigned char>>> mips;
        bool mipsReady=false;
    };
    std::map<InputKey, std::unique_ptr<ResidentInput>> residentInputs;
    std::set<InputKey> pinnedInputs;
    bool cacheInputs = false,filterTextures=true;
    std::vector<std::unique_ptr<DeviceBuffer<unsigned char>>> streamedMipsA,streamedMipsB;
    size_t cacheBudget = 128u * 1024u * 1024u, cacheCost = 0, residentBytes = 0;
    size_t cacheHits = 0, cacheMisses = 0;
    uint64_t inputClock = 0;
    unsigned char *input(const cv::Mat &image, DeviceBuffer<unsigned char> &fallback) {
        auto stream = [&]() {
            auto packed = image.isContinuous() ? image : image.clone();
            fallback.upload(packed.data, packed.total() * packed.elemSize());
            return fallback.data;
        };
        if (!cacheInputs)
            return stream();
        InputKey key{reinterpret_cast<uintptr_t>(image.data), image.rows, image.cols, image.type(),
                     image.step[0]};
        if (auto found = residentInputs.find(key); found != residentInputs.end()) {
            found->second->used = ++inputClock;
            pinnedInputs.insert(key);
            ++cacheHits;
            return found->second->gpu.data;
        }
        ++cacheMisses;
        size_t bytes = image.total() * image.elemSize();
        size_t cost = std::max(bytes, size_t(image.datalimit - image.datastart));
        // Holding the original Mat prevents address reuse. A small ROI of a huge parent must
        // count its parent's retained allocation; external-memory Mats cannot be retained safely.
        if (!image.u || cost > cacheBudget)
            return stream();
        while (cacheCost + cost > cacheBudget || residentInputs.size() >= 128) {
            auto oldest = residentInputs.end();
            for (auto it = residentInputs.begin(); it != residentInputs.end(); ++it)
                if (!pinnedInputs.count(it->first) &&
                    (oldest == residentInputs.end() || it->second->used < oldest->second->used))
                    oldest = it;
            // Never evict a pointer already handed to a kernel in this frame. Streaming is safe
            // when two textures plus backgrounds cannot fit, even with a tiny test budget.
            if (oldest == residentInputs.end())
                return stream();
            cacheCost -= oldest->second->cost;
            residentBytes -= oldest->second->gpu.capacity;
            for(auto &mip:oldest->second->mips)residentBytes-=mip->capacity;
            residentInputs.erase(oldest);
        }
        auto entry = std::make_unique<ResidentInput>();
        entry->owner = image;
        entry->cost = cost;
        entry->used = ++inputClock;
        auto packed = image.isContinuous() ? image : image.clone();
        entry->gpu.upload(packed.data, bytes);
        auto pointer = entry->gpu.data;
        cacheCost += cost;
        residentBytes += entry->gpu.capacity;
        residentInputs.emplace(key, std::move(entry));
        pinnedInputs.insert(key);
        return pointer;
    }
    TextureView texture(const cv::Mat &image,DeviceBuffer<unsigned char> &fallback,
                        std::vector<std::unique_ptr<DeviceBuffer<unsigned char>>> &streamed,bool minified){
        TextureView view;view.data[0]=input(image,fallback);view.width[0]=image.cols;view.height[0]=image.rows;
        if(!filterTextures||!minified)return view;
        size_t extra=0;int w=image.cols,h=image.rows,levels=0;
        while((w>1||h>1)&&levels<15){w=(w+1)/2;h=(h+1)/2;extra+=size_t(w)*h*4;++levels;}
        InputKey key{reinterpret_cast<uintptr_t>(image.data),image.rows,image.cols,image.type(),image.step[0]};
        auto found=residentInputs.find(key);
        ResidentInput *entry=cacheInputs&&found!=residentInputs.end()?found->second.get():nullptr;
        size_t existing=0;if(entry)for(auto &mip:entry->mips)existing+=mip->capacity;
        bool retained=entry&&(entry->mipsReady||cacheCost+extra-existing<=cacheBudget);
        auto &mips=retained?entry->mips:streamed;
        bool build=!retained||!entry->mipsReady;
        if(build){
            Interval timing("texture_prefilter");
            w=image.cols;h=image.rows;
            for(int i=0;i<levels;++i){
                int nw=(w+1)/2,nh=(h+1)/2;
                if(i>=int(mips.size()))mips.push_back(std::make_unique<DeviceBuffer<unsigned char>>());
                auto before=mips[i]->capacity;mips[i]->resize(size_t(nw)*nh*4);
                if(retained){size_t added=mips[i]->capacity-before;residentBytes+=added;cacheCost+=added;entry->cost+=added;}
                auto src=i?mips[i-1]->data:view.data[0];
                prefilterTexture<<<dim3((nw+15)/16,(nh+15)/16),dim3(16,16)>>>(src,w,h,mips[i]->data,nw,nh);
                w=nw;h=nh;
            }
            check(cudaGetLastError(),"prefilter minified texture");
            if(retained)entry->mipsReady=true;
        }
        w=image.cols;h=image.rows;
        for(int i=0;i<levels;++i){w=(w+1)/2;h=(h+1)/2;view.data[i+1]=mips[i]->data;view.width[i+1]=w;view.height[i+1]=h;}
        view.count=levels+1;view.filtered=true;return view;
    }
    void drawMesh(const GpuDraw &d, int w, int h) {
        if (d.from.size() != d.to.size() || d.from.size() != d.meshUV.size() ||
            d.textureA.type() != CV_8UC4)
            throw std::runtime_error("Invalid transformation mesh");
        int tx = (w + 15) / 16, ty = (h + 15) / 16;
        std::vector<std::vector<int>> bins(tx * ty);
        std::vector<GpuPoint> interpolated(d.from.size());
        for (size_t i = 0; i < d.from.size(); ++i)
            interpolated[i] = {d.from[i].x * (1 - d.t) + d.to[i].x * d.t,
                               d.from[i].y * (1 - d.t) + d.to[i].y * d.t};
        size_t referenceCount = 0;
        for (size_t i = 0; i + 2 < d.triangles.size(); i += 3) {
            auto a = interpolated.at(d.triangles[i]), b = interpolated.at(d.triangles[i + 1]),
                 c = interpolated.at(d.triangles[i + 2]);
            if (!std::isfinite(a.x + a.y + b.x + b.y + c.x + c.y))
                continue;
            float minx = std::min({a.x, b.x, c.x}), maxx = std::max({a.x, b.x, c.x}),
                  miny = std::min({a.y, b.y, c.y}), maxy = std::max({a.y, b.y, c.y});
            if (maxx < 0 || maxy < 0 || minx >= w || miny >= h)
                continue;
            int left = std::clamp(int(std::floor(minx / 16)), 0, tx - 1),
                right = std::clamp(int(std::floor(maxx / 16)), 0, tx - 1),
                top = std::clamp(int(std::floor(miny / 16)), 0, ty - 1),
                bottom = std::clamp(int(std::floor(maxy / 16)), 0, ty - 1);
            referenceCount += (right - left + 1) * (bottom - top + 1);
            if (referenceCount > 4000000)
                throw std::runtime_error(
                    "Transform stretches the mesh too far; reduce the mapping or domain");
            for (int y = top; y <= bottom; ++y)
                for (int x = left; x <= right; ++x)
                    bins[y * tx + x].push_back(i / 3);
        }
        std::vector<int> starts(bins.size() + 1), refs;
        refs.reserve(referenceCount);
        for (size_t i = 0; i < bins.size(); ++i) {
            starts[i] = refs.size();
            refs.insert(refs.end(), bins[i].begin(), bins[i].end());
        }
        starts.back() = refs.size();
        a.upload(d.from.data(), d.from.size());
        b.upload(d.to.data(), d.to.size());
        points.resize(d.from.size());
        uv.upload(d.meshUV.data(), d.meshUV.size());
        triangles.upload(d.triangles.data(), d.triangles.size());
        offsets.upload(starts.data(), starts.size());
        references.upload(refs.data(), refs.size());
        auto texture = input(d.textureA, textureA);
        Interval timing("mesh_raster");
        interpolate<<<(d.from.size() + 255) / 256, 256>>>(a.data, b.data, points.data,
                                                          d.from.size(), d.t);
        meshRaster<<<dim3(tx, ty), dim3(16, 16)>>>(
            group.data, coverage.data, points.data, uv.data, triangles.data, offsets.data,
            references.data, texture, d.textureA.cols, d.textureA.rows, w, h, tx, d.color[3]);
    }
};
CudaBackend::CudaBackend() : p_(std::make_unique<Impl>()) {
    check(cudaSetDevice(0), "initialize CUDA device");
}
CudaBackend::~CudaBackend() = default;
void CudaBackend::enableInputCaching(bool enabled, size_t budgetBytes) {
    p_->cacheInputs = enabled;
    p_->cacheBudget = budgetBytes;
    p_->residentInputs.clear();
    p_->pinnedInputs.clear();
    p_->cacheCost = 0;
    p_->residentBytes = 0;
}
size_t CudaBackend::inputCacheBytes() const {
    return p_->residentBytes;
}
void CudaBackend::enableTextureFiltering(bool value){p_->filterTextures=value;}
void CudaBackend::enableRegionCompositing(bool value) {
    p_->regionCompositing = value;
}
void CudaBackend::enableProfiling(bool value) {
    p_->profiling = value;
}
GpuProfile CudaBackend::lastProfile() const {
    return p_->lastProfile;
}

std::string CudaBackend::deviceDescription() {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "query CUDA device");
    return std::string(prop.name) + " | " + std::to_string(prop.totalGlobalMem / (1024 * 1024)) +
           " MiB | CUDA " + std::to_string(CUDART_VERSION / 1000) + "." +
           std::to_string((CUDART_VERSION % 1000) / 10);
}
cv::Mat CudaBackend::select(const cv::Mat &image, cv::Point seed, float tolerance) {
    if (image.type() != CV_8UC3 || !image.isContinuous())
        throw std::runtime_error("Selection requires a contiguous BGR image");
    if (seed.x < 0 || seed.y < 0 || seed.x >= image.cols || seed.y >= image.rows)
        throw std::runtime_error("Selection seed outside slide");
    int n = image.rows * image.cols, s = seed.y * image.cols + seed.x;
    p_->imageA.upload(image.data, 3 * n);
    p_->parent.resize(n);
    p_->selection.resize(n);
    initLabels<<<(n + 255) / 256, 256>>>(p_->imageA.data, p_->parent.data, n, s, tolerance);
    connect<<<(n + 255) / 256, 256>>>(p_->parent.data, image.cols, image.rows);
    extract<<<(n + 255) / 256, 256>>>(p_->parent.data, p_->selection.data, n, s);
    check(cudaGetLastError(), "CUDA connected component selection");
    cv::Mat out(image.size(), CV_8UC1);
    check(cudaMemcpy(out.data, p_->selection.data, n, cudaMemcpyDeviceToHost),
          "download selection");
    return out;
}
cv::Mat CudaBackend::render(const cv::Mat &a, const cv::Mat &b, float t,
                            const std::vector<GpuDraw> &draws) {
    if (a.size() != b.size() || a.type() != CV_8UC3 || b.type() != CV_8UC3)
        throw std::runtime_error("Invalid renderer backgrounds");
    auto began = std::chrono::steady_clock::now();
    EventRecorder recorder;
    RecorderScope recording(p_->profiling ? &recorder : nullptr);
    int n = a.rows * a.cols;
    int blocks = (n + 255) / 256;
    p_->pinnedInputs.clear();
    p_->cacheHits = 0;
    p_->cacheMisses = 0;
    auto backgroundA = p_->input(a, p_->imageA), backgroundB = p_->input(b, p_->imageB);
    p_->frame.resize(n);
    p_->group.resize(n);
    p_->coverage.resize(n);
    p_->result.resize(3 * n);
    {
        Interval timing("background");
        background<<<blocks, 256>>>(backgroundA, backgroundB, p_->frame.data, n, t);
    }

    for (size_t i = 0; i < draws.size();) {
        int lineage = draws[i].lineage;
        cv::Rect region(0, 0, a.cols, a.rows);
        if (p_->regionCompositing) {
            region = {};
            for (size_t j = i; j < draws.size() && draws[j].lineage == lineage; ++j) {
                auto bounds = drawRegion(draws[j], a.cols, a.rows);
                if (bounds.area() > 0)
                    region = region.area() ? region | bounds : bounds;
            }
        }
        if (region.empty()) {
            do {
                ++i;
            } while (i < draws.size() && draws[i].lineage == lineage);
            continue;
        }
        {
            Interval timing("clear");
            clearRegion<<<(region.area() + 255) / 256, 256>>>(p_->group.data, p_->coverage.data,
                                                              a.cols, region.x, region.y,
                                                              region.width, region.height);
        }
        do {
            const auto &d = draws[i];
            if (!d.triangles.empty()) {
                p_->drawMesh(d, a.cols, a.rows);
            } else if (!d.textureA.empty() && !d.textureB.empty()) {
                if (d.textureA.type() != CV_8UC4 || d.textureB.type() != CV_8UC4)
                    throw std::runtime_error("Sprite textures must be BGRA images");
                bool minA=d.textureA.cols>std::abs(d.extentX)*1.001f||d.textureA.rows>std::abs(d.extentY)*1.001f;
                bool minB=d.textureB.cols>std::abs(d.extentX)*1.001f||d.textureB.rows>std::abs(d.extentY)*1.001f;
                auto textureA=p_->texture(d.textureA,p_->textureA,p_->streamedMipsA,minA),
                     textureB=p_->texture(d.textureB,p_->textureB,p_->streamedMipsB,minB);
                cv::Rect r = d.bounds & cv::Rect(0, 0, a.cols, a.rows);
                Interval timing("sprite_raster");
                if (r.area() > 0)
                    sprite<<<dim3((r.width + 15) / 16, (r.height + 15) / 16), dim3(16, 16)>>>(
                        p_->group.data, p_->coverage.data, textureA, textureB, a.cols, a.rows,
                        r.x, r.y, r.width, r.height, d.centerX, d.centerY, d.extentX, d.extentY,
                        d.angle, d.t, d.color[3], d.textureOpacityA, d.textureOpacityB,
                        make_float4(d.textureClip[0], d.textureClip[1], d.textureClip[2],
                                    d.textureClip[3]));
            } else if (!d.from.empty() && d.color[3] > .00001f) {
                p_->a.upload(d.from.data(), d.from.size());
                p_->b.upload(d.to.data(), d.to.size());
                p_->points.resize(d.from.size());
                std::vector<GpuContour> edges;
                std::vector<std::vector<int>> bands((a.rows + 15) / 16);
                auto position = [&](int index) {
                    return d.from[index].y * (1 - d.t) + d.to[index].y * d.t;
                };
                for (const auto &contour : d.contours)
                    for (int k = 0; k < contour.count; ++k) {
                        int ia = contour.begin + k, ib = contour.begin + (k + 1) % contour.count;
                        float ya = position(ia), yb = position(ib);
                        if (ya == yb)
                            continue;
                        int lo = std::max(0, int(std::floor(std::min(ya, yb) / 16))),
                            hi = std::min(int(bands.size()) - 1,
                                          int(std::floor(std::max(ya, yb) / 16)));
                        int index = edges.size();
                        edges.push_back({ia, ib});
                        for (int band = lo; band <= hi; ++band)
                            bands[band].push_back(index);
                    }
                std::vector<int> starts(bands.size() + 1), refs;
                for (size_t band = 0; band < bands.size(); ++band) {
                    starts[band] = refs.size();
                    refs.insert(refs.end(), bands[band].begin(), bands[band].end());
                }
                starts.back() = refs.size();
                p_->curves.upload(edges.data(), edges.size());
                p_->offsets.upload(starts.data(), starts.size());
                p_->references.upload(refs.data(), refs.size());
                Interval timing("contour_raster");
                interpolate<<<(d.from.size() + 255) / 256, 256>>>(
                    p_->a.data, p_->b.data, p_->points.data, (int)d.from.size(), d.t);
                cv::Rect r = d.bounds & cv::Rect(0, 0, a.cols, a.rows);
                if (r.area() > 0)
                    raster<<<dim3((r.width + 15) / 16, (r.height + 15) / 16), dim3(16, 16)>>>(
                        p_->group.data, p_->coverage.data, p_->points.data, p_->curves.data,
                        p_->offsets.data, p_->references.data, a.cols, a.rows, r.x, r.y, r.width,
                        r.height, make_float4(d.color[0], d.color[1], d.color[2], d.color[3]));
            }
            ++i;
        } while (i < draws.size() && draws[i].lineage == lineage);
        {
            Interval timing("composite");
            composite<<<(region.area() + 255) / 256, 256>>>(p_->frame.data, p_->group.data,
                                                            p_->coverage.data, a.cols, region.x,
                                                            region.y, region.width, region.height);
        }
    }
    {
        Interval timing("output");
        output<<<blocks, 256>>>(p_->frame.data, p_->result.data, n);
    }

    check(cudaGetLastError(), "CUDA rasterization");
    cv::Mat result(a.size(), CV_8UC3);
    {
        Interval timing("download");
        check(cudaMemcpy(result.data, p_->result.data, n * 3, cudaMemcpyDeviceToHost),
              "download rendered frame");
    }
    if (p_->profiling) {
        recorder.resolve();
        recorder.result.downloadedBytes = n * 3;
        recorder.result.residentInputBytes = p_->residentBytes;
        recorder.result.retainedInputBytes = p_->cacheCost;
        recorder.result.inputCacheHits = p_->cacheHits;
        recorder.result.inputCacheMisses = p_->cacheMisses;
        recorder.result.wallMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began)
                .count();
        p_->lastProfile = recorder.result;
    }
    return result;
}
cv::Mat CudaBackend::warp(const GpuDraw &draw, int w, int h) {
    int n = w * h;
    p_->group.resize(n);
    p_->coverage.resize(n);
    p_->result.resize(4 * n);
    check(cudaMemset(p_->group.data, 0, n * sizeof(float4)), "clear transformation image");
    check(cudaMemset(p_->coverage.data, 0, n * sizeof(float)), "clear transformation alpha");
    bool cached = p_->cacheInputs;
    p_->cacheInputs = false;
    try {
        p_->drawMesh(draw, w, h);
    } catch (...) {
        p_->cacheInputs = cached;
        throw;
    }
    p_->cacheInputs = cached;
    transparentOutput<<<(n + 255) / 256, 256>>>(p_->group.data, p_->coverage.data, p_->result.data,
                                                n);
    check(cudaGetLastError(), "render transformed keyframe object");
    cv::Mat rgba(h, w, CV_8UC4);
    check(cudaMemcpy(rgba.data, p_->result.data, 4 * n, cudaMemcpyDeviceToHost),
          "download transformed object");
    return rgba;
}
} // namespace fm
