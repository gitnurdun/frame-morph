#include "engine.hpp"
#include <algorithm>
#include <chrono>
#include <numeric>

namespace fm {
json profileProject(const Project &project, int w, int h, int samples, int transition,
                    bool inputCaching) {
    if (samples < 8 || samples > 600)
        throw std::runtime_error("Profiling sample count must be 8 to 600");
    if (transition < 0 || transition >= int(project.transitions.size()))
        throw std::runtime_error("Choose an existing transition to profile");
    using Clock = std::chrono::steady_clock;
    auto begin = Clock::now();
    Renderer renderer(project, w, h);
    renderer.enableInputCaching(inputCaching);
    auto reference = renderer.transitionFrame(transition, .5f);
    double cold = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    std::vector<double> durations;
    for (int i = 0; i < samples; ++i) {
        auto start = Clock::now();
        renderer.transitionFrame(transition, .05f + .9f * i / samples);
        durations.push_back(
            std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    std::sort(durations.begin(), durations.end());
    double mean = std::accumulate(durations.begin(), durations.end(), 0.) / samples;
    json result = {
        {"device", CudaBackend::deviceDescription()},
        {"width", w},
        {"height", h},
        {"transition", transition + 1},
        {"samples", samples},
        {"input_cache_enabled", inputCaching},
        {"cold_first_frame_ms", cold},
        {"mean_frame_ms", mean},
        {"p95_frame_ms", durations[std::min(samples - 1, int(samples * .95))]},
        {"warm_renderer_fps", 1000 / mean},
        {"source_objects", project.slides[transition].objects.size()},
        {"notes", "Uninstrumented warm timings include CPU planning, submission, CUDA, transfers "
                  "and endpoint blending. CUDA events below are a separate instrumented run; they "
                  "are not hardware utilization or DRAM counters."}};
    renderer.enableProfiling(true);
    std::map<std::string, double> stages;
    double upload = 0, download = 0, wall = 0, hits = 0, misses = 0;
    size_t resident = 0, retained = 0;
    int instrumented = std::min(samples, 24);
    for (int i = 0; i < instrumented; ++i) {
        renderer.transitionFrame(transition, .1f + .8f * i / instrumented);
        auto p = renderer.lastGpuProfile();
        for (const auto &entry : p.stages)
            stages[entry.first] += entry.second / instrumented;
        upload += double(p.uploadedBytes) / instrumented;
        download += double(p.downloadedBytes) / instrumented;
        wall += p.wallMs / instrumented;
        hits += double(p.inputCacheHits) / instrumented;
        misses += double(p.inputCacheMisses) / instrumented;
        resident = std::max(resident, p.residentInputBytes);
        retained = std::max(retained, p.retainedInputBytes);
    }
    auto profiled = renderer.transitionFrame(transition, .5f);
    result["profiling_preserves_pixels"] = cv::norm(reference, profiled, cv::NORM_INF) == 0;
    Renderer comparison(project, w, h);
    comparison.enableInputCaching(!inputCaching);
    bool equivalent = true;
    for (float t : {0.f, .001f, .15f, .5f, .85f, .999f, 1.f})
        equivalent &= cv::norm(renderer.transitionFrame(transition, t),
                               comparison.transitionFrame(transition, t), cv::NORM_INF) == 0;
    result["input_cache_preserves_pixels"] = equivalent;
    result["input_cache_comparison_frames"] = 7;
    result["instrumented_backend_wall_ms"] = wall;
    result["cuda_event_ms"] = stages;
    result["uploaded_bytes_per_frame"] = upload;
    result["downloaded_bytes_per_frame"] = download;
    result["resident_input_cache_bytes"] = resident;
    result["retained_input_storage_cost_bytes"] = retained;
    result["input_cache_hits_per_frame"] = hits;
    result["input_cache_misses_per_frame"] = misses;
    result["explicit_upload_note"] =
        "Buffer upload bytes exclude kernel arguments and driver-internal traffic. Output is still "
        "downloaded for display/export.";
    return result;
}
} // namespace fm
