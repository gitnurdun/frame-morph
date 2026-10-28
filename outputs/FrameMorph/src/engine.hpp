#pragma once
#include "cuda_backend.hpp"
#include <array>
#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace fm {
namespace fs = std::filesystem;
using json = nlohmann::json;
struct Object {
    std::string id, name;
    cv::Mat mask;
    int layer = 0;
    json drawing;    // Editable authoring parameters, null for pixel selections.
    json generation; // Per-object operation that generated this pose from the previous keyframe.
    json updater;    // Draft, evaluated only by explicit Generate from updaters.
    std::string constructAnimation = "default", destructAnimation = "default";
    double effectStart = 0, effectEnd = 1;
    std::string effectEasing = "smooth";
    cv::Mat artwork; // Cropped BGRA patch; cached for portable text/equation rendering.
    json imageEdits; // Non-destructive settings against originalAsset.
    cv::Mat originalAsset;
    cv::Mat sourceAsset; // Original inserted image, independent of external file paths.
    cv::Point artworkOrigin{};
};
struct Slide {
    std::string id, title;
    cv::Mat image;
    cv::Mat baseImage; // Original imported/blank image before authored overlays.
    cv::Vec3b background{24, 20, 15};
    double hold = 0.5;
    double stepDuration = 1.5; // Time to the next frame; also used by explicit updater generation.
    bool blank = false;
    std::vector<Object> objects;
};
struct Link {
    std::string id, name;
    std::vector<std::string> sources, targets;
    double start = 0, end = 1;
    std::string easing = "smooth", mode = "morph";
    float bend = 0, rotation = 0;
    int layer = 0;
    // Component index pairs, followed by optional normalized boundary anchors.
    std::vector<std::pair<int, int>> componentPairs;
    std::vector<std::pair<float, float>> anchors;
    std::string offsetX, offsetY, scaleFunction, rotationFunction, opacityFunction;
    bool manual = false;
};
struct Transition {
    std::string from, to;
    double duration = 1.5;
    std::vector<Link> links;
};
struct Project {
    int width = 1920, height = 1080, fps = 60;
    std::vector<Slide> slides;
    std::vector<Transition> transitions;
    fs::path folder;
};
std::string newId();
Project deepCopy(const Project &);
Slide importSlide(const fs::path &, int width, int height);
Slide blankSlide(int width, int height, cv::Vec3b background = {42, 28, 17});
Slide duplicateSlide(const Slide &);
void renderDrawing(Slide &, Object &, const json &spec);
cv::Mat renderMedia(const json &spec, const std::function<bool()> &cancelled = {});
cv::Mat processImage(const cv::Mat &source, const json &settings,
                     const std::function<bool()> &cancelled = {});
Object combineImageObjects(const Slide &, const std::vector<std::string> &,
                           const std::string &layout, int gap = 20);
std::function<double(const std::array<double, 6> &)> compilePixelExpression(const std::string &);
void recomposeSlide(Slide &);
std::string defaultObjectName(const Slide &, const std::string &kind);
json objectFrame(const Object &);
void makeObjectEditable(Slide &, Object &);
void setObjectFrame(Slide &, Object &, const json &frame);
void setObjectColor(Slide &, Object &, const std::string &hex);
void editObjectConstruction(Slide &, Object &, const json &spec);
double evaluateExpression(const std::string &, double x);
std::function<double(double)> compileExpression(const std::string &);
std::function<double(double, double)>
compileExpression2D(const std::string &, const std::map<std::string, double> &constants = {});
std::function<double(double, double, double)>
compileExpression3D(const std::string &, const std::map<std::string, double> &constants = {});
std::map<std::string, double> sceneVariables(const Slide &, const Object * = nullptr);
json parseObjectMap(const std::string &, const Slide &, const Object &, double units = 100.);
void applyObjectMap(Slide &, Object &, const json &);
GpuDraw prepareObjectTransform(const Slide &, const Object &, const json &);
struct ObjectUpdaterPlan {
    GpuDraw source;
    bool affine = false;
    std::function<cv::Point2f(cv::Point2f, double)> evaluate;
};
json parseObjectUpdater(const std::string &, const Slide &, const Object &, double units = 100.);
ObjectUpdaterPlan prepareObjectUpdater(const Slide &, const Object &, const json &,
                                       double duration);
Slide generateUpdaterKeyframe(const Slide &, const std::map<std::string, json> &, double duration);
Slide generateKeyframe(const Slide &, const std::map<std::string, json> &operations);
std::string objectTransformSignature(const Slide &, const Object &);
void reconcileTransitions(Project &);
std::vector<Link> automaticLinks(const Slide &, const Slide &);
void saveProject(Project &, const fs::path &, bool autosave = false);
Project loadProject(const fs::path &, bool autosave = false);
json projectMetadata(const Project &);
double projectDuration(const Project &);
void validateProject(const Project &);
void createSample(const fs::path &);
void createAuthoringSample(const fs::path &);
void createTransformSample(const fs::path &);
void createStudioSample(const fs::path &);
void createMotionSample(const fs::path &);
void createUpdaterSample(const fs::path &);
void createPropertiesSample(const fs::path &);
void createToolkitSample(const fs::path &);
float ease(float t, const std::string &name);
const std::vector<std::pair<std::string, std::string>> &lifecycleEffects();
json profileProject(const Project &, int width = 960, int height = 540, int samples = 120,
                    int transition = 0, bool inputCaching = true);

class Renderer {
  public:
    explicit Renderer(Project project, int width = 960, int height = 540);
    cv::Mat frame(double seconds, const std::function<bool()> &cancelled = {});
    void warmupAt(double seconds);
    void enableProfiling(bool value) {
        gpu_.enableProfiling(value);
    }
    void enableInputCaching(bool enabled, size_t budgetBytes = 128u * 1024u * 1024u) {
        gpu_.enableInputCaching(enabled, budgetBytes);
    }
    void enableTextureFiltering(bool value){gpu_.enableTextureFiltering(value);}
    GpuProfile lastGpuProfile() const {
        return gpu_.lastProfile();
    }
    cv::Mat transitionFrame(int index, float t, const std::function<bool()> &cancelled = {});
    const std::vector<std::string> &warnings() const {
        return warnings_;
    }

  private:
    struct CurvePair {
        std::vector<cv::Point2f> a, b;
    };
    struct Part {
        std::vector<CurvePair> curves;
        cv::Vec3f colorA{}, colorB{};
        cv::Point2f centerA{}, centerB{};
        float opacityA = 1, opacityB = 1;
        bool unsafe = false;
    };
    struct Sprite {
        cv::Mat a, b;
        std::string lifecycle; // Empty for matched objects; effects apply only to absent endpoints.
        bool entering = false;
        float effectStart = 0, effectEnd = 1;
        std::string effectEasing = "smooth";
        cv::Rect boundsA, boundsB;
        float angleA = 0, angleB = 0, opacityA = 1, opacityB = 1;
    };
    struct PreparedLink {
        Link link;
        cv::Rect endpointBounds;
        std::vector<Part> parts;
        std::vector<Sprite> sprites;
        std::vector<GpuDraw> meshes;
        std::vector<ObjectUpdaterPlan> updaters;
        std::array<std::function<double(double)>, 5> functions;
    };
    struct Prepared {
        cv::Mat a, b;
        std::vector<PreparedLink> links;
    };
    Project project_;
    int width_, height_;
    float sx_, sy_, ox_, oy_;
    CudaBackend gpu_;
    std::map<int, Prepared> cache_;
    std::vector<std::string> warnings_;
    Prepared prepare(int);
};
struct ExportOptions {
    fs::path output;
    int width = 1920, height = 1080, fps = 60;
    bool pngSequence = false;
    std::string ffmpeg = "ffmpeg";
};
void exportVideo(const Project &, const ExportOptions &, std::atomic_bool *cancel = nullptr,
                 const std::function<void(int, int)> &progress = {});
int runEditor(const fs::path &initial = {});
} // namespace fm
