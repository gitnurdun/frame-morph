#include "engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fcntl.h>
#include <fstream>
#include <list>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
extern char **environ;
namespace fm {
cv::Mat renderMedia(const json &spec, const std::function<bool()> &cancelled) {
    if (cancelled && cancelled())
        throw std::runtime_error("Preview cancelled");
    static std::mutex mutex;
    static std::list<std::pair<std::string, cv::Mat>> cache;
    auto signature = spec;
    for (auto key :
         {"x", "y", "w", "h", "angle", "opacity", "stroke", "filled", "axes", "expressions"})
        signature.erase(key);
    auto key = signature.dump();
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = cache.begin(); it != cache.end(); ++it)
            if (it->first == key) {
                auto result = it->second.clone();
                cache.splice(cache.begin(), cache, it);
                return result;
            }
    }
    fs::path root = FRAMEMORPH_SOURCE_DIR;
    if (const char *env = std::getenv("FRAMEMORPH_RESOURCE_DIR"))
        root = env;
    else {
        std::error_code ec;
        auto exe = fs::read_symlink("/proc/self/exe", ec);
        if (!ec && fs::exists(exe.parent_path().parent_path() / "scripts/media_render.py"))
            root = exe.parent_path().parent_path();
    }
    auto helper = root / "scripts/media_render.py";
    if (!fs::exists(helper))
        throw std::runtime_error("Missing scripts/media_render.py");
    std::string python = "python3";
    if (const char *env = std::getenv("FRAMEMORPH_MATH_PYTHON"))
        python = env;
    else if (const char *home = std::getenv("HOME")) {
        auto conda = fs::path(home) / "anaconda3/bin/python";
        if (fs::exists(conda))
            python = conda.string();
    }
    auto dir = fs::temp_directory_path() / ("framemorph-media-" + newId());
    fs::create_directories(dir);
    struct Cleanup {
        fs::path dir;
        ~Cleanup() {
            std::error_code ec;
            fs::remove_all(dir, ec);
        }
    } cleanup{dir};
    {
        std::ofstream f(dir / "input.json");
        f << spec.dump();
    }
    std::vector<std::string> args{python, helper.string(), (dir / "input.json").string(),
                                  (dir / "output.png").string()};
    std::vector<char *> argv;
    for (auto &arg : args)
        argv.push_back(arg.data());
    argv.push_back(nullptr);
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    auto log = (dir / "helper.log").string();
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, log.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0600);
    posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
    pid_t child = -1;
    int error = posix_spawnp(&child, python.c_str(), &actions, &attr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    if (error)
        throw std::runtime_error("Cannot start artwork renderer. Check FRAMEMORPH_MATH_PYTHON.");
    int status = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(35);
    for (;;) {
        auto result = waitpid(child, &status, WNOHANG);
        if (result == child)
            break;
        if (result < 0 && errno != EINTR)
            throw std::runtime_error("Cannot wait for artwork renderer");
        if (cancelled && cancelled()) {
            kill(-child, SIGKILL);
            waitpid(child, &status, 0);
            throw std::runtime_error("Preview cancelled");
        }
        if (std::chrono::steady_clock::now() > deadline) {
            kill(-child, SIGKILL);
            waitpid(child, &status, 0);
            throw std::runtime_error("Artwork rendering exceeded 35 seconds. Simplify the input.");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::ifstream f(dir / "output.png.error");
        std::string message((std::istreambuf_iterator<char>(f)), {});
        throw std::runtime_error(message.empty()
                                     ? "Artwork renderer failed. Equation needs matplotlib; Full "
                                       "LaTeX/SVG needs the optional PyMuPDF backend."
                                     : message.substr(0, 3000));
    }
    auto image = cv::imread((dir / "output.png").string(), cv::IMREAD_UNCHANGED);
    if (image.type() != CV_8UC4 || image.total() > 16000000)
        throw std::runtime_error("Artwork renderer returned invalid or oversized RGBA data");
    {
        std::lock_guard<std::mutex> lock(mutex);
        cache.push_front({key, image.clone()});
        size_t bytes = 0;
        for (auto &entry : cache)
            bytes += entry.second.total() * 4 + entry.first.size();
        while (cache.size() > 1 && (bytes > 64 * 1024 * 1024 || cache.size() > 32)) {
            bytes -= cache.back().second.total() * 4 + cache.back().first.size();
            cache.pop_back();
        }
    }
    return image;
}
namespace {
void checkSize(int w, int h) {
    if (w < 1 || h < 1 || w > 8192 || h > 8192 || double(w) * h > 16000000)
        throw std::runtime_error(
            "Image dimensions must be 1 to 8192 pixels and at most 16 megapixels");
}
cv::Mat premultiplied(const cv::Mat &rgba, const std::function<void(cv::Mat &)> &operation) {
    cv::Mat f;
    rgba.convertTo(f, CV_32FC4, 1. / 255);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            auto &p = f.at<cv::Vec4f>(y, x);
            for (int c = 0; c < 3; ++c)
                p[c] *= p[3];
        }
    operation(f);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            auto &p = f.at<cv::Vec4f>(y, x);
            if (p[3] > 1e-6f)
                for (int c = 0; c < 3; ++c)
                    p[c] /= p[3];
        }
    cv::Mat out;
    f.convertTo(out, CV_8UC4, 255);
    return out;
}
} // namespace
cv::Mat processImage(const cv::Mat &source, const json &settings,
                     const std::function<bool()> &cancelled) {
    auto checkCancelled = [&] {
        if (cancelled && cancelled())
            throw std::runtime_error("Preview cancelled");
    };
    checkCancelled();
    if (source.type() != CV_8UC4)
        throw std::runtime_error("Image editing needs an RGBA image");
    auto bounded = [&](const char *key, double fallback, double low, double high) {
        double v = settings.value(key, fallback);
        if (!std::isfinite(v) || v < low || v > high)
            throw std::runtime_error(std::string("Invalid image setting: ") + key);
        return v;
    };
    int x = bounded("crop_x", 0, 0, source.cols - 1), y = bounded("crop_y", 0, 0, source.rows - 1),
        cw = bounded("crop_w", source.cols - x, 1, source.cols - x),
        ch = bounded("crop_h", source.rows - y, 1, source.rows - y);
    cv::Mat image = source(cv::Rect(x, y, cw, ch)).clone();
    int w = bounded("width", cw, 1, 8192), h = bounded("height", ch, 1, 8192);
    checkSize(w, h);
    if (image.cols != w || image.rows != h)
        image = premultiplied(image, [&](cv::Mat &p) {
            cv::resize(p, p, {w, h}, 0, 0, w < cw || h < ch ? cv::INTER_AREA : cv::INTER_CUBIC);
        });
    bool fx = settings.value("flip_x", false), fy = settings.value("flip_y", false);
    if (fx || fy)
        cv::flip(image, image, fx && fy ? -1 : fx ? 1 : 0);
    double brightness = bounded("brightness", 0, -1, 1), contrast = bounded("contrast", 1, 0, 4),
           saturation = bounded("saturation", 1, 0, 3), blur = bounded("blur", 0, 0, 30);
    auto tint = settings.value("tint", "#ffffff");
    if (tint.size() != 7 || tint[0] != '#' ||
        !std::all_of(tint.begin() + 1, tint.end(),
                     [](unsigned char c) { return std::isxdigit(c); }))
        throw std::runtime_error("Tint must be an RGB hex colour, such as #ffffff");
    std::array<double, 3> multiplier{std::stoi(tint.substr(5, 2), nullptr, 16) / 255.,
                                     std::stoi(tint.substr(3, 2), nullptr, 16) / 255.,
                                     std::stoi(tint.substr(1, 2), nullptr, 16) / 255.};
    auto filter = settings.value("filter", "none");
    if (filter != "none" && filter != "grayscale" && filter != "sepia" && filter != "invert" &&
        filter != "sharpen" && filter != "edges" && filter != "threshold")
        throw std::runtime_error("Unknown image filter");
    for (int row = 0; row < image.rows; ++row) {
        checkCancelled();
        for (int col = 0; col < image.cols; ++col) {
            auto &p = image.at<cv::Vec4b>(row, col);
            double b = p[0], g = p[1], r = p[2], gray = .2126 * r + .7152 * g + .0722 * b;
            if (filter == "grayscale")
                b = g = r = gray;
            else if (filter == "sepia") {
                double rr = .393 * r + .769 * g + .189 * b, gg = .349 * r + .686 * g + .168 * b;
                b = .272 * r + .534 * g + .131 * b;
                r = rr;
                g = gg;
            } else if (filter == "invert") {
                b = 255 - b;
                g = 255 - g;
                r = 255 - r;
            } else if (filter == "threshold")
                b = g = r = gray >= 127.5 ? 255 : 0;
            std::array<double, 3> colors{b, g, r};
            gray = .2126 * r + .7152 * g + .0722 * b;
            for (int c = 0; c < 3; ++c)
                p[c] = cv::saturate_cast<uchar>(
                    ((gray + (colors[c] - gray) * saturation) / 255 - .5) * contrast * 255 + 127.5 +
                    brightness * 255);
        }
    }
    checkCancelled();
    if (blur > 0)
        image = premultiplied(image, [&](cv::Mat &p) { cv::GaussianBlur(p, p, {0, 0}, blur); });
    if (filter == "sharpen") {
        cv::Mat softened;
        cv::GaussianBlur(image, softened, {0, 0}, 1.2);
        cv::Mat alpha;
        cv::extractChannel(image, alpha, 3);
        cv::addWeighted(image, 1.8, softened, -.8, 0, image);
        cv::insertChannel(alpha, image, 3);
    }
    if (filter == "edges") {
        cv::Mat gray, edges, alpha;
        cv::extractChannel(image, alpha, 3);
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        cv::Canny(gray, edges, 60, 150);
        cv::cvtColor(edges, image, cv::COLOR_GRAY2BGRA);
        cv::insertChannel(alpha, image, 3);
    }
    if (tint != "#ffffff" && tint != "#FFFFFF")
        for (int y = 0; y < image.rows; ++y) {
            checkCancelled();
            for (int x = 0; x < image.cols; ++x) {
                auto &p = image.at<cv::Vec4b>(y, x);
                for (int c = 0; c < 3; ++c)
                    p[c] = cv::saturate_cast<uchar>(p[c] * multiplier[c]);
            }
        }
    std::string map = settings.value("pixel_map", "");
    if (!map.empty()) {
        auto colon = map.find(':');
        auto prefix = map.substr(0, colon);
        prefix.erase(std::remove_if(prefix.begin(), prefix.end(),
                                    [](unsigned char c) { return std::isspace(c); }),
                     prefix.end());
        if (colon == std::string::npos || prefix != "(r,g,b,a)")
            throw std::runtime_error("Use (r,g,b,a):red,green,blue,alpha with channels in 0..1");
        std::vector<std::string> fields;
        std::string part;
        int depth = 0;
        for (char c : map.substr(colon + 1)) {
            if (c == '(')
                ++depth;
            if (c == ')')
                --depth;
            if (c == ',' && depth == 0) {
                fields.push_back(part);
                part.clear();
            } else
                part += c;
        }
        fields.push_back(part);
        if (fields.size() != 4 || depth != 0)
            throw std::runtime_error("Pixel map needs four balanced channel expressions");
        std::array<std::function<double(const std::array<double, 6> &)>, 4> functions;
        for (int c = 0; c < 4; ++c)
            functions[c] = compilePixelExpression(fields[c]);
        cv::Mat out = image.clone();
        cv::parallel_for_(cv::Range(0, image.rows), [&](const cv::Range &range) {
            for (int yy = range.start; yy < range.end; ++yy) {
                checkCancelled();
                for (int xx = 0; xx < image.cols; ++xx) {
                    auto p = image.at<cv::Vec4b>(yy, xx);
                    std::array<double, 6> values{p[2] / 255.,
                                                 p[1] / 255.,
                                                 p[0] / 255.,
                                                 p[3] / 255.,
                                                 double(xx) / std::max(1, image.cols - 1),
                                                 double(yy) / std::max(1, image.rows - 1)};
                    auto &q = out.at<cv::Vec4b>(yy, xx);
                    for (int c = 0; c < 4; ++c) {
                        double value = functions[c](values);
                        if (!std::isfinite(value))
                            throw std::runtime_error("Pixel formula returned a non-finite value");
                        q[c == 0   ? 2
                          : c == 2 ? 0
                                   : c] = cv::saturate_cast<uchar>(std::clamp(value, 0., 1.) * 255);
                    }
                }
            }
        });
        image = out;
    }
    return image;
}
Object combineImageObjects(const Slide &slide, const std::vector<std::string> &ids,
                           const std::string &layout, int gap) {
    if (ids.size() < 2)
        throw std::runtime_error("Select at least two objects to combine");
    if (layout != "overlay" && layout != "row" && layout != "column")
        throw std::runtime_error("Unknown image layout");
    if (gap < 0 || gap > 1000)
        throw std::runtime_error("Image spacing must be 0 to 1000 pixels");
    std::vector<const Object *> selected;
    cv::Rect bounds;
    std::set<std::string> seen;
    for (auto &id : ids) {
        auto it = std::find_if(slide.objects.begin(), slide.objects.end(),
                               [&](auto &o) { return o.id == id; });
        if (it == slide.objects.end() || !seen.insert(id).second)
            throw std::runtime_error("Missing or repeated object in image combination");
        selected.push_back(&*it);
        bounds |= cv::boundingRect(it->mask);
    }
    if (layout == "overlay")
        std::stable_sort(selected.begin(), selected.end(),
                         [](auto a, auto b) { return a->layer < b->layer; });
    std::vector<cv::Mat> patches;
    int w = 0, h = 0;
    for (auto o : selected) {
        cv::Mat patch;
        if (!o->artwork.empty())
            patch = o->artwork;
        else {
            auto b = cv::boundingRect(o->mask);
            cv::cvtColor(slide.image(b), patch, cv::COLOR_BGR2BGRA);
            for (int y = 0; y < patch.rows; ++y)
                for (int x = 0; x < patch.cols; ++x)
                    patch.at<cv::Vec4b>(y, x)[3] = o->mask.at<uchar>(b.y + y, b.x + x);
        }
        patches.push_back(patch);
        if (layout == "row") {
            w += patch.cols + gap;
            h = std::max(h, patch.rows);
        } else if (layout == "column") {
            h += patch.rows + gap;
            w = std::max(w, patch.cols);
        }
    }
    if (layout == "overlay") {
        w = bounds.width;
        h = bounds.height;
    } else if (layout == "row")
        w -= gap;
    else
        h -= gap;
    checkSize(w, h);
    cv::Mat merged(h, w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    int offset = 0;
    for (size_t i = 0; i < selected.size(); ++i) {
        auto &patch = patches[i];
        auto o = selected[i];
        auto b = cv::boundingRect(o->mask);
        cv::Point origin = o->artwork.empty() ? b.tl() : o->artworkOrigin;
        cv::Point at = layout == "overlay" ? origin - bounds.tl()
                       : layout == "row"   ? cv::Point(offset, (h - patch.rows) / 2)
                                           : cv::Point((w - patch.cols) / 2, offset);
        for (int y = 0; y < patch.rows; ++y)
            for (int x = 0; x < patch.cols; ++x) {
                int xx = x + at.x, yy = y + at.y;
                if (xx < 0 || yy < 0 || xx >= w || yy >= h)
                    continue;
                auto p = patch.at<cv::Vec4b>(y, x);
                auto &q = merged.at<cv::Vec4b>(yy, xx);
                float a = p[3] / 255.f, da = q[3] / 255.f, alpha = a + da * (1 - a);
                if (alpha > 0)
                    for (int c = 0; c < 3; ++c)
                        q[c] = cv::saturate_cast<uchar>((p[c] * a + q[c] * da * (1 - a)) / alpha);
                q[3] = cv::saturate_cast<uchar>(alpha * 255);
            }
        if (layout == "row")
            offset += patch.cols + gap;
        else if (layout == "column")
            offset += patch.rows + gap;
    }
    Object result;
    result.id = newId();
    result.name = defaultObjectName(slide, "combined_image");
    result.sourceAsset = merged;
    for (auto o : selected)
        result.layer = std::max(result.layer, o->layer);
    Slide temporary = slide;
    renderDrawing(temporary, result,
                  {{"kind", "image"}, {"x", bounds.x}, {"y", bounds.y}, {"w", w}, {"h", h}});
    return result;
}
} // namespace fm
