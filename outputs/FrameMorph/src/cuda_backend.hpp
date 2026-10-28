#pragma once
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace fm {
struct GpuProfile {
    double wallMs = 0;
    std::map<std::string, double> stages;
    size_t uploadedBytes = 0, downloadedBytes = 0;
    size_t residentInputBytes = 0, retainedInputBytes = 0, inputCacheHits = 0, inputCacheMisses = 0;
};
struct GpuPoint {
    float x, y;
};
struct GpuContour {
    int begin, count;
};
struct GpuDraw {
    std::vector<GpuPoint> from, to;
    std::vector<GpuContour> contours;
    cv::Vec4f color{1, 1, 1, 1}; // linear RGB, opacity
    cv::Rect bounds;
    float t = 0;
    int lineage = 0;
    cv::Mat textureA, textureB; // Optional BGRA sprites, preserving inserted-image detail.
    float centerX = 0, centerY = 0, extentX = 1, extentY = 1, angle = 0;
    float textureOpacityA = 1, textureOpacityB = 1;
    cv::Vec4f textureClip{0, 0, 1, 1}; // Reveal window in normalized texture coordinates.
    std::vector<GpuPoint> meshUV;
    std::vector<int> triangles;
};
class CudaBackend {
  public:
    CudaBackend();
    ~CudaBackend();
    CudaBackend(const CudaBackend &) = delete;
    CudaBackend &operator=(const CudaBackend &) = delete;
    static std::string deviceDescription();
    cv::Mat select(const cv::Mat &bgr, cv::Point seed, float tolerance);
    cv::Mat render(const cv::Mat &backgroundA, const cv::Mat &backgroundB, float t,
                   const std::vector<GpuDraw> &draws);
    cv::Mat warp(const GpuDraw &, int width, int height);
    void enableProfiling(bool);
    void enableRegionCompositing(bool);
    void enableTextureFiltering(bool);
    // Opt-in contract: cached input matrices remain immutable for their retained lifetime.
    // External/unowned matrices bypass caching. The budget bounds GPU and retained host storage.
    void enableInputCaching(bool, size_t budgetBytes = 128u * 1024u * 1024u);
    size_t inputCacheBytes() const;
    GpuProfile lastProfile() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace fm
