#include "engine.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <poll.h>
#include <random>
#include <set>
#include <signal.h>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace fm {
namespace {
constexpr double pi = 3.141592653589793;
bool animated(const Object &o) {
    return o.name.rfind("__", 0) == 0;
}
cv::Mat resized(const cv::Mat &m, int w, int h, int mode = cv::INTER_AREA) {
    double scale = std::min(double(w) / m.cols, double(h) / m.rows);
    int rw = std::min(w, std::max(1, int(std::lround(m.cols * scale)))),
        rh = std::min(h, std::max(1, int(std::lround(m.rows * scale))));
    cv::Mat fit;
    cv::resize(m, fit, {rw, rh}, 0, 0, mode);
    if (rw == w && rh == h)
        return fit;
    auto bg = m.at<cv::Vec3b>(0, 0);
    cv::Mat result(h, w, CV_8UC3, cv::Scalar(bg[0], bg[1], bg[2]));
    fit.copyTo(result(cv::Rect((w - rw) / 2, (h - rh) / 2, rw, rh)));
    return result;
}
float linear(float c) {
    c /= 255.f;
    return c <= .04045f ? c / 12.92f : std::pow((c + .055f) / 1.055f, 2.4f);
}
cv::Vec3f rgbLinear(cv::Scalar c) {
    return {linear(c[2]), linear(c[1]), linear(c[0])};
}
cv::Point2f center(const cv::Mat &mask) {
    auto m = cv::moments(mask, true);
    return m.m00 > 0 ? cv::Point2f(m.m10 / m.m00, m.m01 / m.m00)
                     : cv::Point2f(mask.cols * .5f, mask.rows * .5f);
}
const Object *findObject(const Slide &s, const std::string &id) {
    for (auto &o : s.objects)
        if (o.id == id)
            return &o;
    return nullptr;
}
fs::path assetPath(const fs::path &root, const std::string &name) {
    fs::path p(name);
    if (p.empty() || p.is_absolute())
        throw std::runtime_error("Project assets must have relative paths");
    for (auto &part : p)
        if (part == "..")
            throw std::runtime_error("Asset path escapes project directory");
    auto base = fs::weakly_canonical(root), full = fs::weakly_canonical(root / p);
    auto rel = full.lexically_relative(base);
    for (auto &part : rel)
        if (part == "..")
            throw std::runtime_error("Asset symlink escapes project directory");
    return full;
}
void writeImage(const fs::path &p, const cv::Mat &image) {
    if (!cv::imwrite(p.string(), image))
        throw std::runtime_error("Could not write image: " + p.string());
}
void atomicText(const fs::path &p, const std::string &data) {
    fs::path temp = p;
    temp += ".tmp";
    {
        std::ofstream f(temp, std::ios::binary);
        if (!f)
            throw std::runtime_error("Cannot write " + temp.string());
        f << data;
        f.flush();
        if (!f)
            throw std::runtime_error("Incomplete project write");
    }
    fs::rename(temp, p);
}
json linkJson(const Link &l) {
    return {{"id", l.id},
            {"name", l.name},
            {"sources", l.sources},
            {"targets", l.targets},
            {"start", l.start},
            {"end", l.end},
            {"easing", l.easing},
            {"mode", l.mode},
            {"bend", l.bend},
            {"rotation", l.rotation},
            {"layer", l.layer},
            {"component_pairs", l.componentPairs},
            {"anchors", l.anchors},
            {"offset_x", l.offsetX},
            {"offset_y", l.offsetY},
            {"scale_function", l.scaleFunction},
            {"rotation_function", l.rotationFunction},
            {"opacity_function", l.opacityFunction},
            {"manual", l.manual}};
}
Link readLink(const json &j) {
    Link l;
    l.id = j.at("id");
    l.name = j.value("name", "");
    l.sources = j.value("sources", std::vector<std::string>{});
    l.targets = j.value("targets", std::vector<std::string>{});
    l.start = j.value("start", 0.);
    l.end = j.value("end", 1.);
    l.easing = j.value("easing", "smooth");
    l.mode = j.value("mode", "morph");
    l.bend = j.value("bend", 0.f);
    l.rotation = j.value("rotation", 0.f);
    l.layer = j.value("layer", 0);
    l.componentPairs = j.value("component_pairs", std::vector<std::pair<int, int>>{});
    l.anchors = j.value("anchors", std::vector<std::pair<float, float>>{});
    l.offsetX = j.value("offset_x", "");
    l.offsetY = j.value("offset_y", "");
    l.scaleFunction = j.value("scale_function", "");
    l.rotationFunction = j.value("rotation_function", "");
    l.opacityFunction = j.value("opacity_function", "");
    l.manual = j.value("manual", false);
    return l;
}
struct Component {
    std::vector<std::vector<cv::Point2f>> loops;
    cv::Point2f c;
    cv::Vec3f color;
    double area;
};
std::vector<Component> components(const Slide &slide, const Object &obj) {
    std::vector<Component> out;
    cv::Mat appearance = slide.image;
    if (!obj.artwork.empty()) {
        appearance = slide.image.clone();
        cv::Rect dest(obj.artworkOrigin, obj.artwork.size()),
            visible = dest & cv::Rect(0, 0, appearance.cols, appearance.rows);
        for (int y = visible.y; y < visible.y + visible.height; ++y)
            for (int x = visible.x; x < visible.x + visible.width; ++x) {
                auto p = obj.artwork.at<cv::Vec4b>(y - dest.y, x - dest.x);
                appearance.at<cv::Vec3b>(y, x) = {p[0], p[1], p[2]};
            }
    }
    // Color regions stay separate even when additive selections touch.
    cv::Mat remaining;
    cv::threshold(obj.mask, remaining, 0, 255, cv::THRESH_BINARY);
    for (int colorPass = 0; colorPass < 24 && cv::countNonZero(remaining); ++colorPass) {
        std::map<int, int> histogram;
        for (int y = 0; y < remaining.rows; ++y) {
            auto m = remaining.ptr<uchar>(y);
            auto p = appearance.ptr<cv::Vec3b>(y);
            for (int x = 0; x < remaining.cols; ++x)
                if (m[x]) {
                    auto c = p[x];
                    int key = (c[0] / 24) * 121 + (c[1] / 24) * 11 + c[2] / 24;
                    ++histogram[key];
                }
        }
        if (histogram.empty())
            break;
        int best = std::max_element(histogram.begin(), histogram.end(), [](auto &a, auto &b) {
                       return a.second < b.second;
                   })->first;
        cv::Mat palette(remaining.size(), CV_8U, cv::Scalar(0));
        cv::Vec3d sum{};
        double count = 0;
        for (int y = 0; y < remaining.rows; ++y) {
            auto m = remaining.ptr<uchar>(y);
            auto p = appearance.ptr<cv::Vec3b>(y);
            for (int x = 0; x < remaining.cols; ++x)
                if (m[x]) {
                    auto c = p[x];
                    int key = (c[0] / 24) * 121 + (c[1] / 24) * 11 + c[2] / 24;
                    if (key == best) {
                        sum += cv::Vec3d(c);
                        ++count;
                    }
                }
        }
        cv::Vec3d color = sum / std::max(1., count);
        for (int y = 0; y < remaining.rows; ++y) {
            auto m = remaining.ptr<uchar>(y);
            auto dst = palette.ptr<uchar>(y);
            auto p = appearance.ptr<cv::Vec3b>(y);
            for (int x = 0; x < remaining.cols; ++x)
                if (m[x]) {
                    auto d = cv::Vec3d(p[x]) - color;
                    if (cv::norm(d) < 45 || colorPass == 23)
                        dst[x] = 255;
                }
        }
        // Include antialiasing against the known flat background, without filling holes.
        cv::Mat expanded;
        cv::dilate(palette, expanded, cv::Mat());
        cv::bitwise_and(expanded, obj.mask, expanded);
        palette = expanded;
        remaining.setTo(0, palette);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(palette, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < (int)contours.size(); ++i) {
            int depth = 0;
            for (int parent = hierarchy[i][3]; parent >= 0; parent = hierarchy[parent][3])
                ++depth;
            if (depth % 2 || std::abs(cv::contourArea(contours[i])) < .5)
                continue;
            Component component;
            component.area = std::abs(cv::contourArea(contours[i]));
            component.color = rgbLinear(cv::Scalar(color[0], color[1], color[2]));
            auto add = [&](int k) {
                std::vector<cv::Point> simplified;
                cv::approxPolyDP(contours[k], simplified, .45, true);
                std::vector<cv::Point2f> pts;
                for (auto p : simplified)
                    pts.emplace_back(p.x + .5f, p.y + .5f);
                if (pts.size() >= 3)
                    component.loops.push_back(std::move(pts));
            };
            add(i);
            for (int child = hierarchy[i][2]; child >= 0; child = hierarchy[child][0])
                add(child);
            auto mm = cv::moments(contours[i]);
            component.c = mm.m00 ? cv::Point2f(mm.m10 / mm.m00, mm.m01 / mm.m00)
                                 : cv::Point2f(contours[i][0]);
            if (!component.loops.empty())
                out.push_back(std::move(component));
        }
    }
    std::stable_sort(out.begin(), out.end(), [](auto &a, auto &b) {
        if (std::abs(a.c.y - b.c.y) > 8)
            return a.c.y < b.c.y;
        return a.c.x < b.c.x;
    });
    return out;
}
std::vector<cv::Point2f> resample(std::vector<cv::Point2f> p, int n) {
    if (p.empty())
        return std::vector<cv::Point2f>(n);
    if (cv::contourArea(p, true) < 0)
        std::reverse(p.begin(), p.end());
    std::vector<float> lengths(p.size() + 1, 0);
    for (size_t i = 0; i < p.size(); ++i)
        lengths[i + 1] = lengths[i] + cv::norm(p[(i + 1) % p.size()] - p[i]);
    std::vector<cv::Point2f> out;
    out.reserve(n);
    size_t edge = 0;
    for (int k = 0; k < n; ++k) {
        float d = lengths.back() * k / n;
        while (edge + 1 < p.size() && lengths[edge + 1] < d)
            ++edge;
        float t = (d - lengths[edge]) / std::max(1e-8f, lengths[edge + 1] - lengths[edge]);
        out.push_back(p[edge] * (1 - t) + p[(edge + 1) % p.size()] * t);
    }
    return out;
}
void align(std::vector<cv::Point2f> &a, std::vector<cv::Point2f> &b, cv::Point2f ca, cv::Point2f cb,
           const Link &link) {
    int n = (int)a.size();
    if (!n)
        return;
    if (!link.anchors.empty()) {
        auto anchors = link.anchors;
        for (auto &v : anchors) {
            v.first -= std::floor(v.first);
            v.second -= std::floor(v.second);
        }
        std::sort(anchors.begin(), anchors.end());
        if (anchors.size() == 1) {
            int shift = (int)std::lround((anchors[0].second - anchors[0].first) * n);
            shift = (shift % n + n) % n;
            std::rotate(b.begin(), b.begin() + shift, b.end());
            return;
        }
        for (size_t i = 1; i < anchors.size(); ++i)
            if (anchors[i].second <= anchors[i - 1].second)
                throw std::runtime_error("Boundary anchors must follow the same contour direction");
        auto original = b;
        for (int k = 0; k < n; ++k) {
            float u = float(k) / n;
            int seg = -1;
            for (int i = 0; i < (int)anchors.size(); ++i)
                if (anchors[i].first <= u)
                    seg = i;
            auto lo = seg < 0 ? std::make_pair(anchors.back().first - 1, anchors.back().second - 1)
                              : anchors[seg];
            auto hi = seg + 1 < (int)anchors.size()
                          ? anchors[seg + 1]
                          : std::make_pair(anchors.front().first + 1, anchors.front().second + 1);
            float v = lo.second + (hi.second - lo.second) * (u - lo.first) /
                                      std::max(1e-6f, hi.first - lo.first);
            v -= std::floor(v);
            float idx = v * n;
            int i = (int)idx;
            b[k] = original[i % n] * (1 - (idx - i)) + original[(i + 1) % n] * (idx - i);
        }
        return;
    }
    float sa = 0, sb = 0;
    for (int i = 0; i < n; ++i) {
        sa += cv::norm(a[i] - ca);
        sb += cv::norm(b[i] - cb);
    }
    sa = std::max(1.f, sa / n);
    sb = std::max(1.f, sb / n);
    double best = 1e300;
    int shift = 0;
    for (int j = 0; j < n; ++j) {
        double cost = 0;
        for (int i = 0; i < n; ++i) {
            auto d = (a[i] - ca) / sa - (b[(i + j) % n] - cb) / sb;
            cost += d.dot(d);
        }
        if (cost < best) {
            best = cost;
            shift = j;
        }
    }
    std::rotate(b.begin(), b.begin() + shift, b.end());
}
float cross(cv::Point2f a, cv::Point2f b) {
    return a.x * b.y - a.y * b.x;
}
bool selfIntersect(const std::vector<cv::Point2f> &p) {
    for (int i = 0; i < (int)p.size(); ++i)
        for (int j = i + 2; j < (int)p.size(); ++j) {
            if (i == 0 && j == (int)p.size() - 1)
                continue;
            auto a = p[i], b = p[(i + 1) % p.size()], c = p[j], d = p[(j + 1) % p.size()];
            if (cross(b - a, c - a) * cross(b - a, d - a) < -1e-5f &&
                cross(d - c, a - c) * cross(d - c, b - c) < -1e-5f)
                return true;
        }
    return false;
}
// Rectangular Hungarian assignment with deterministic tie breaking.
std::vector<std::pair<int, int>> assignment(const std::vector<std::vector<double>> &costs) {
    int n = costs.size();
    if (!n)
        return {};
    int m = costs[0].size();
    if (!m)
        return {};
    bool transpose = n > m;
    int rows = transpose ? m : n, cols = transpose ? n : m;
    std::vector<double> u(rows + 1), v(cols + 1);
    std::vector<int> p(cols + 1), way(cols + 1);
    for (int i = 1; i <= rows; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(cols + 1, 1e100);
        std::vector<bool> used(cols + 1);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = 1e100;
            for (int j = 1; j <= cols; ++j)
                if (!used[j]) {
                    double c =
                        (transpose ? costs[j - 1][i0 - 1] : costs[i0 - 1][j - 1]) - u[i0] - v[j];
                    if (c < minv[j]) {
                        minv[j] = c;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            for (int j = 0; j <= cols; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else
                    minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0]);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<std::pair<int, int>> out;
    for (int j = 1; j <= cols; ++j)
        if (p[j])
            out.push_back(transpose ? std::make_pair(j - 1, p[j] - 1)
                                    : std::make_pair(p[j] - 1, j - 1));
    std::sort(out.begin(), out.end());
    return out;
}
} // namespace
std::string newId() {
    static std::atomic<uint64_t> counter{0};
    auto t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ostringstream ss;
    ss << std::hex << t << "-" << counter++;
    return ss.str();
}
Project deepCopy(const Project &p) {
    Project q = p;
    for (auto &s : q.slides) {
        s.image = s.image.clone();
        s.baseImage = s.baseImage.clone();
        for (auto &o : s.objects) {
            o.mask = o.mask.clone();
            o.artwork = o.artwork.clone();
            o.sourceAsset = o.sourceAsset.clone();
            o.originalAsset = o.originalAsset.clone();
        }
    }
    return q;
}
Slide importSlide(const fs::path &path, int w, int h) {
    cv::Mat raw = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty())
        throw std::runtime_error("Cannot read image: " + path.string());
    if (raw.depth() != CV_8U)
        raw.convertTo(raw, CV_8U, raw.depth() == CV_16U ? 1. / 257 : 1.);
    if (raw.channels() == 1)
        cv::cvtColor(raw, raw, cv::COLOR_GRAY2BGR);
    cv::Vec3b bg{24, 20, 15};
    if (raw.channels() == 4) {
        std::vector<cv::Mat> c;
        cv::split(raw, c);
        cv::Mat bgr;
        cv::cvtColor(raw, bgr, cv::COLOR_BGRA2BGR);
        for (int y = 0; y < raw.rows; ++y)
            for (int x = 0; x < raw.cols; ++x) {
                float a = c[3].at<uchar>(y, x) / 255.f;
                bgr.at<cv::Vec3b>(y, x) =
                    cv::Vec3f(bgr.at<cv::Vec3b>(y, x)) * a + cv::Vec3f(bg) * (1 - a);
            }
        raw = bgr;
    }
    std::vector<int> channels[3];
    int step = std::max(1, std::min(raw.rows, raw.cols) / 200);
    for (int x = 0; x < raw.cols; x += step)
        for (int y : {0, raw.rows - 1})
            for (int c = 0; c < 3; ++c)
                channels[c].push_back(raw.at<cv::Vec3b>(y, x)[c]);
    for (int y = 0; y < raw.rows; y += step)
        for (int x : {0, raw.cols - 1})
            for (int c = 0; c < 3; ++c)
                channels[c].push_back(raw.at<cv::Vec3b>(y, x)[c]);
    for (int c = 0; c < 3; ++c) {
        auto &v = channels[c];
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        bg[c] = v[v.size() / 2];
    }
    Slide s;
    s.id = newId();
    s.title = path.stem().string();
    s.background = bg;
    s.image = cv::Mat(h, w, CV_8UC3, cv::Scalar(bg[0], bg[1], bg[2]));
    double scale = std::min(double(w) / raw.cols, double(h) / raw.rows);
    cv::Mat fit;
    cv::resize(raw, fit, {std::max(1, int(raw.cols * scale)), std::max(1, int(raw.rows * scale))},
               0, 0, scale < 1 ? cv::INTER_AREA : cv::INTER_CUBIC);
    fit.copyTo(s.image(cv::Rect((w - fit.cols) / 2, (h - fit.rows) / 2, fit.cols, fit.rows)));
    return s;
}
std::vector<Link> automaticLinks(const Slide &a, const Slide &b) {
    std::map<std::string, std::vector<const Object *>> aa, bb;
    for (auto &o : a.objects)
        if (cv::countNonZero(o.mask))
            aa[o.name].push_back(&o);
    for (auto &o : b.objects)
        if (cv::countNonZero(o.mask))
            bb[o.name].push_back(&o);
    std::set<std::string> names;
    for (auto &e : aa)
        names.insert(e.first);
    for (auto &e : bb)
        names.insert(e.first);
    std::vector<Link> links;
    for (auto &name : names) {
        auto &src = aa[name];
        auto &dst = bb[name];
        // Plain names stay in the residual crossfade when present on both slides.
        // Only unmatched instances need a separate construction/destruction pass.
        if (name.rfind("__", 0) != 0) {
            std::set<int> usedA, usedB;
            if (!src.empty() && !dst.empty()) {
                std::vector<std::vector<double>> costs(src.size(), std::vector<double>(dst.size()));
                for (size_t i = 0; i < src.size(); ++i)
                    for (size_t j = 0; j < dst.size(); ++j)
                        costs[i][j] = cv::norm(center(src[i]->mask) - center(dst[j]->mask));
                for (auto [i, j] : assignment(costs)) {
                    usedA.insert(i);
                    usedB.insert(j);
                }
            }
            auto absent = [&](const Object &o, bool entering) {
                Link l;
                l.id = newId();
                l.name = name;
                l.layer = o.layer;
                (entering ? l.targets : l.sources).push_back(o.id);
                links.push_back(std::move(l));
            };
            for (int i = 0; i < int(src.size()); ++i)
                if (!usedA.count(i))
                    absent(*src[i], false);
            for (int j = 0; j < int(dst.size()); ++j)
                if (!usedB.count(j))
                    absent(*dst[j], true);
            continue;
        }
        if (src.size() <= 1 || dst.size() <= 1) {
            Link l;
            l.id = newId();
            l.name = name;
            for (auto o : src) {
                l.sources.push_back(o->id);
                l.layer = std::max(l.layer, o->layer);
            }
            for (auto o : dst) {
                l.targets.push_back(o->id);
                l.layer = std::max(l.layer, o->layer);
            }
            links.push_back(l);
            continue;
        }
        std::vector<std::vector<double>> costs(src.size(), std::vector<double>(dst.size()));
        for (size_t i = 0; i < src.size(); ++i)
            for (size_t j = 0; j < dst.size(); ++j) {
                double ar = cv::countNonZero(src[i]->mask), br = cv::countNonZero(dst[j]->mask);
                auto ac = center(src[i]->mask), bc = center(dst[j]->mask);
                cv::Scalar ca = cv::mean(a.image, src[i]->mask),
                           cb = cv::mean(b.image, dst[j]->mask);
                cv::Rect ra = cv::boundingRect(src[i]->mask), rb = cv::boundingRect(dst[j]->mask);
                double aspect = std::abs(
                    std::log((double(ra.width) / ra.height) / (double(rb.width) / rb.height)));
                costs[i][j] = cv::norm(ac - bc) / a.image.cols + std::abs(std::log(ar / br)) * .2 +
                              cv::norm(ca - cb) / 255 * .2 + aspect * .15;
                if (!dst[j]->generation.is_null() &&
                    dst[j]->generation.value("source_id", "") == src[i]->id)
                    costs[i][j] -= 1000000;
            }
        std::set<int> si, ti;
        for (auto [i, j] : assignment(costs)) {
            Link l;
            l.id = newId();
            l.name = name;
            l.sources = {src[i]->id};
            l.targets = {dst[j]->id};
            l.layer = std::max(src[i]->layer, dst[j]->layer);
            links.push_back(l);
            si.insert(i);
            ti.insert(j);
        }
        for (int i = 0; i < (int)src.size(); ++i)
            if (!si.count(i)) {
                Link l;
                l.id = newId();
                l.name = name;
                l.sources = {src[i]->id};
                l.layer = src[i]->layer;
                links.push_back(l);
            }
        for (int j = 0; j < (int)dst.size(); ++j)
            if (!ti.count(j)) {
                Link l;
                l.id = newId();
                l.name = name;
                l.targets = {dst[j]->id};
                l.layer = dst[j]->layer;
                links.push_back(l);
            }
    }
    return links;
}
void reconcileTransitions(Project &p) {
    auto old = p.transitions;
    p.transitions.clear();
    for (size_t i = 1; i < p.slides.size(); ++i) {
        auto it = std::find_if(old.begin(), old.end(), [&](auto &t) {
            return t.from == p.slides[i - 1].id && t.to == p.slides[i].id;
        });
        Transition t;
        if (it != old.end())
            t = *it;
        else {
            t.from = p.slides[i - 1].id;
            t.to = p.slides[i].id;
            t.duration = p.slides[i - 1].stepDuration;
            t.links = automaticLinks(p.slides[i - 1], p.slides[i]);
        }
        p.transitions.push_back(t);
    }
}
json projectMetadata(const Project &p) {
    json j = {{"format", "framemorph"},      {"version", 3}, {"width", p.width},
              {"height", p.height},          {"fps", p.fps}, {"slides", json::array()},
              {"transitions", json::array()}};
    for (const auto &slide : p.slides)
        for (const auto &o : slide.objects) {
            auto native =
                o.drawing.contains("construction") ? o.drawing["construction"] : o.drawing;
            bool advanced = o.effectStart != 0 || o.effectEnd != 1 || o.effectEasing != "smooth" ||
                            !o.originalAsset.empty();
            if (native.is_object())
                advanced |=
                    native.value("kind", "") == "code" || native.value("kind", "") == "svg" ||
                    native.value("kind", "") == "star" || native.value("kind", "") == "diamond" ||
                    native.value("kind", "") == "hexagon" || native.contains("backend");
            const std::set<std::string> old{"default", "none",     "fade",
                                            "scale",   "slide_up", "slide_down"};
            if (advanced || !old.count(o.constructAnimation) || !old.count(o.destructAnimation))
                j["version"] = 4;
        }
    for (auto &s : p.slides) {
        json sj = {{"id", s.id},
                   {"title", s.title},
                   {"image", "assets/" + s.id + ".png"},
                   {"background", {s.background[0], s.background[1], s.background[2]}},
                   {"hold", s.hold},
                   {"step_duration", s.stepDuration},
                   {"blank", s.blank},
                   {"objects", json::array()}};
        for (auto &o : s.objects)
            sj["objects"].push_back({{"id", o.id},
                                     {"name", o.name},
                                     {"layer", o.layer},
                                     {"mask", "masks/" + o.id + ".png"},
                                     {"drawing", o.drawing},
                                     {"artwork_origin", {o.artworkOrigin.x, o.artworkOrigin.y}},
                                     {"generation", o.generation},
                                     {"updater", o.updater},
                                     {"image_edits", o.imageEdits},
                                     {"construct_animation", o.constructAnimation},
                                     {"destruct_animation", o.destructAnimation},
                                     {"effect_start", o.effectStart},
                                     {"effect_end", o.effectEnd},
                                     {"effect_easing", o.effectEasing}});
        j["slides"].push_back(sj);
    }
    for (auto &t : p.transitions) {
        json tj = {
            {"from", t.from}, {"to", t.to}, {"duration", t.duration}, {"links", json::array()}};
        for (auto &l : t.links)
            tj["links"].push_back(linkJson(l));
        j["transitions"].push_back(tj);
    }
    return j;
}
void saveProject(Project &p, const fs::path &folder, bool autosave) {
    validateProject(p);
    fs::create_directories(folder / "assets");
    fs::create_directories(folder / "masks");
    std::set<std::string> previousAssets;
    for (auto manifest : {"project.json", "autosave.json"})
        if (fs::exists(folder / manifest)) {
            try {
                std::ifstream f(folder / manifest);
                json old;
                f >> old;
                for (auto &s : old.at("slides")) {
                    previousAssets.insert(s.at("image").get<std::string>());
                    if (s.contains("base_image"))
                        previousAssets.insert(s["base_image"].get<std::string>());
                    for (auto &o : s.at("objects")) {
                        previousAssets.insert(o.at("mask").get<std::string>());
                        for (auto key : {"artwork", "source_asset", "original_asset"})
                            if (o.contains(key))
                                previousAssets.insert(o[key].get<std::string>());
                    }
                }
            } catch (...) { /* Keep unrelated or unreadable existing content. */
            }
        }
    // Assets are immutable snapshots: every save receives new paths, then the manifest commits.
    json j = projectMetadata(p);
    std::string revision = newId();
    for (size_t i = 0; i < p.slides.size(); ++i) {
        auto &s = p.slides[i];
        auto imageName = "assets/" + s.id + "-" + revision + ".png";
        j["slides"][i]["image"] = imageName;
        writeImage(folder / imageName, s.image);
        if (!s.baseImage.empty()) {
            auto name = "assets/" + s.id + "-base-" + revision + ".png";
            j["slides"][i]["base_image"] = name;
            writeImage(folder / name, s.baseImage);
        }
        for (size_t k = 0; k < s.objects.size(); ++k) {
            auto &o = s.objects[k];
            auto name = "masks/" + o.id + "-" + revision + ".png";
            auto &oj = j["slides"][i]["objects"][k];
            oj["mask"] = name;
            writeImage(folder / name, o.mask);
            if (!o.artwork.empty()) {
                auto art = "assets/" + o.id + "-art-" + revision + ".png";
                oj["artwork"] = art;
                writeImage(folder / art, o.artwork);
            }
            if (!o.originalAsset.empty()) {
                auto original = "assets/" + o.id + "-original-" + revision + ".png";
                oj["original_asset"] = original;
                writeImage(folder / original, o.originalAsset);
            }
            if (!o.sourceAsset.empty()) {
                auto src = "assets/" + o.id + "-source-" + revision + ".png";
                oj["source_asset"] = src;
                writeImage(folder / src, o.sourceAsset);
            }
        }
    }
    atomicText(folder / (autosave ? "autosave.json" : "project.json"), j.dump(2));
    if (!autosave) {
        p.folder = fs::absolute(folder);
        std::error_code ec;
        fs::remove(folder / "autosave.json", ec);
    }
    // Collect only unreferenced snapshot assets after both manifests are safely committed.
    std::set<std::string> live;
    for (auto manifest : {"project.json", "autosave.json"})
        if (fs::exists(folder / manifest)) {
            try {
                std::ifstream f(folder / manifest);
                json m;
                f >> m;
                for (auto &s : m["slides"]) {
                    live.insert(s["image"].get<std::string>());
                    if (s.contains("base_image"))
                        live.insert(s["base_image"].get<std::string>());
                    for (auto &o : s["objects"]) {
                        live.insert(o["mask"].get<std::string>());
                        for (auto key : {"artwork", "source_asset", "original_asset"})
                            if (o.contains(key))
                                live.insert(o[key].get<std::string>());
                    }
                }
            } catch (...) {
                return;
            }
        }
    for (auto &relative : previousAssets)
        if (!live.count(relative)) {
            try {
                fs::path path = assetPath(folder, relative);
                auto parent = fs::path(relative).parent_path().generic_string();
                if ((parent == "assets" || parent == "masks") && path.extension() == ".png") {
                    std::error_code ec;
                    fs::remove(path, ec);
                }
            } catch (...) { /* Never collect assets outside the managed directories. */
            }
        }
}
Project loadProject(const fs::path &folder, bool autosave) {
    fs::path root = fs::is_regular_file(folder) ? folder.parent_path() : folder;
    fs::path file =
        fs::is_regular_file(folder) ? folder : root / (autosave ? "autosave.json" : "project.json");
    std::ifstream f(file);
    if (!f)
        throw std::runtime_error("Cannot open project: " + file.string());
    json j;
    f >> j;
    if (j.value("format", "") != "framemorph" ||
        (j.value("version", 0) < 1 || j.value("version", 0) > 4))
        throw std::runtime_error("Unsupported FrameMorph project version");
    Project p;
    p.folder = fs::absolute(root);
    p.width = j.at("width");
    p.height = j.at("height");
    p.fps = j.value("fps", 60);
    if (p.width < 16 || p.height < 16 || p.width > 8192 || p.height > 8192)
        throw std::runtime_error("Invalid project canvas size");
    for (auto &sj : j.at("slides")) {
        Slide s;
        s.id = sj.at("id");
        s.title = sj.value("title", "Slide");
        s.hold = sj.value("hold", .5);
        s.stepDuration = sj.value("step_duration", 1.5);
        s.blank = sj.value("blank", false);
        auto bg = sj.at("background");
        s.background = {bg[0].get<uchar>(), bg[1].get<uchar>(), bg[2].get<uchar>()};
        s.image = cv::imread(assetPath(root, sj.at("image")).string());
        if (s.image.empty())
            throw std::runtime_error("Missing slide image: " + sj.at("image").get<std::string>());
        if (sj.contains("base_image")) {
            s.baseImage = cv::imread(assetPath(root, sj["base_image"]).string());
            if (s.baseImage.empty())
                throw std::runtime_error("Missing original slide background");
        }
        for (auto &oj : sj.at("objects")) {
            Object o;
            o.id = oj.at("id");
            o.name = oj.at("name");
            o.layer = oj.value("layer", 0);
            o.mask = cv::imread(assetPath(root, oj.at("mask")).string(), cv::IMREAD_GRAYSCALE);
            if (o.mask.empty())
                throw std::runtime_error("Missing object mask: " + o.name);
            o.drawing = oj.value("drawing", json{});
            o.generation = oj.value("generation", json{});
            o.updater = oj.value("updater", json{});
            o.imageEdits = oj.value("image_edits", json{});
            if (oj.contains("original_asset")) {
                o.originalAsset = cv::imread(assetPath(root, oj["original_asset"]).string(),
                                             cv::IMREAD_UNCHANGED);
                if (o.originalAsset.type() != CV_8UC4)
                    throw std::runtime_error("Missing original image asset");
            }
            o.constructAnimation = oj.value("construct_animation", "default");
            o.destructAnimation = oj.value("destruct_animation", "default");
            o.effectStart = oj.value("effect_start", 0.);
            o.effectEnd = oj.value("effect_end", 1.);
            o.effectEasing = oj.value("effect_easing", "smooth");
            if (oj.contains("artwork_origin"))
                o.artworkOrigin = {oj["artwork_origin"][0], oj["artwork_origin"][1]};
            if (oj.contains("artwork")) {
                o.artwork =
                    cv::imread(assetPath(root, oj["artwork"]).string(), cv::IMREAD_UNCHANGED);
                if (o.artwork.empty() || o.artwork.type() != CV_8UC4)
                    throw std::runtime_error("Missing or invalid authored artwork");
            }
            if (oj.contains("source_asset")) {
                o.sourceAsset =
                    cv::imread(assetPath(root, oj["source_asset"]).string(), cv::IMREAD_UNCHANGED);
                if (o.sourceAsset.empty())
                    throw std::runtime_error("Missing inserted image source");
            }
            s.objects.push_back(o);
        }
        p.slides.push_back(s);
    }
    for (auto &tj : j.at("transitions")) {
        Transition t;
        t.from = tj.at("from");
        t.to = tj.at("to");
        t.duration = tj.at("duration");
        for (auto &l : tj.at("links"))
            t.links.push_back(readLink(l));
        p.transitions.push_back(t);
    }
    for (size_t i = 0; i < p.transitions.size() && i < p.slides.size(); ++i)
        if (!j["slides"][i].contains("step_duration"))
            p.slides[i].stepDuration = p.transitions[i].duration;
    validateProject(p);
    return p;
}
void validateProject(const Project &p) {
    if (p.width < 16 || p.height < 16 || p.width > 8192 || p.height > 8192 || p.fps < 1 ||
        p.fps > 240)
        throw std::runtime_error("Invalid canvas size or frame rate");
    std::set<std::string> ids;
    auto validId = [](const std::string &id) {
        return !id.empty() && id.size() <= 128 &&
               std::all_of(id.begin(), id.end(),
                           [](unsigned char c) { return std::isalnum(c) || c == '-' || c == '_'; });
    };
    for (auto &s : p.slides) {
        if (!validId(s.id) || !ids.insert(s.id).second)
            throw std::runtime_error("Invalid or duplicate slide ID");
        if (s.image.type() != CV_8UC3 || s.image.size() != cv::Size(p.width, p.height))
            throw std::runtime_error("Slide dimensions differ from canvas");
        if (!std::isfinite(s.hold) || s.hold < 0 || s.hold > 3600)
            throw std::runtime_error("Invalid slide hold duration");
        if (!std::isfinite(s.stepDuration) || s.stepDuration <= 0 || s.stepDuration > 3600)
            throw std::runtime_error(
                "Frame time step must be greater than zero and at most 3600 seconds");
        for (auto &o : s.objects) {
            for (auto effect : {o.constructAnimation, o.destructAnimation})
                if (std::none_of(lifecycleEffects().begin(), lifecycleEffects().end(),
                                 [&](auto &entry) { return entry.first == effect; }))
                    throw std::runtime_error("Unknown construct/destruct animation: " + effect);
            if (!std::isfinite(o.effectStart) || !std::isfinite(o.effectEnd) || o.effectStart < 0 ||
                o.effectEnd > 1 || o.effectStart >= o.effectEnd)
                throw std::runtime_error("Effect timing must satisfy 0 <= start < end <= 1");
            if (o.effectEasing != "smooth" && o.effectEasing != "linear" &&
                o.effectEasing != "ease_in" && o.effectEasing != "ease_out")
                throw std::runtime_error("Unknown object effect easing");
            if (!validId(o.id) || !ids.insert(o.id).second)
                throw std::runtime_error("Invalid or duplicate object ID");
            if (o.mask.type() != CV_8UC1 || o.mask.size() != s.image.size())
                throw std::runtime_error("Invalid object mask: " + o.name);
        }
    }
    if (p.transitions.size() != (p.slides.empty() ? 0 : p.slides.size() - 1))
        throw std::runtime_error("Transitions must connect adjacent slides");
    for (size_t i = 0; i < p.transitions.size(); ++i) {
        auto &t = p.transitions[i];
        if (t.from != p.slides[i].id || t.to != p.slides[i + 1].id)
            throw std::runtime_error("Transition does not connect adjacent slides");
        if (!std::isfinite(t.duration) || t.duration <= 0 || t.duration > 3600)
            throw std::runtime_error("Invalid transition duration");
        std::set<std::string> usedS, usedT;
        for (auto &l : t.links) {
            if (l.id.empty() || !ids.insert(l.id).second)
                throw std::runtime_error("Duplicate or empty connection ID");
            if (!std::isfinite(l.start) || !std::isfinite(l.end) || l.start < 0 || l.end > 1 ||
                l.start >= l.end)
                throw std::runtime_error("Connection timing must satisfy 0 <= start < end <= 1");
            if (l.sources.empty() && l.targets.empty())
                throw std::runtime_error("Empty animation connection");
            for (auto &id : l.sources) {
                auto *o = findObject(p.slides[i], id);
                if (!o || (!animated(*o) && !l.targets.empty()))
                    throw std::runtime_error(
                        "Connection refers to a missing or non-participating source object");
                if (!usedS.insert(id).second)
                    throw std::runtime_error("Source object used in multiple connections; group "
                                             "splits in one connection");
            }
            for (auto &id : l.targets) {
                auto *o = findObject(p.slides[i + 1], id);
                if (!o || (!animated(*o) && !l.sources.empty()))
                    throw std::runtime_error(
                        "Connection refers to a missing or non-participating target object");
                if (!usedT.insert(id).second)
                    throw std::runtime_error("Target object used in multiple connections; group "
                                             "merges in one connection");
            }
            for (const auto &id : l.targets) {
                const auto *o = findObject(p.slides[i + 1], id);
                if (o && o->generation.is_object() &&
                    o->generation.value("kind", "") == "updater" &&
                    std::abs(t.duration * (l.end - l.start) -
                             o->generation.value("duration", -1.)) > 1e-5)
                    throw std::runtime_error("Updater timing is committed. Regenerate from the "
                                             "source slide with the desired step duration.");
            }
            if (!std::isfinite(l.bend) || !std::isfinite(l.rotation) || std::abs(l.bend) > 4 ||
                std::abs(l.rotation) > 3600)
                throw std::runtime_error("Invalid motion parameters");
            if (l.easing != "smooth" && l.easing != "linear" && l.easing != "ease_in" &&
                l.easing != "ease_out")
                throw std::runtime_error("Unknown easing");
            if (l.mode != "morph" && l.mode != "fade")
                throw std::runtime_error("Unknown animation mode");
            for (auto [a, b] : l.anchors)
                if (!std::isfinite(a) || !std::isfinite(b) || a < 0 || a >= 1 || b < 0 || b >= 1)
                    throw std::runtime_error("Anchors must lie in [0,1)");
            int functionIndex = 0;
            for (auto *expression : {&l.offsetX, &l.offsetY, &l.scaleFunction, &l.rotationFunction,
                                     &l.opacityFunction}) {
                if (!expression->empty()) {
                    auto f = compileExpression(*expression);
                    for (double t : {0., .25, .5, .75, 1.}) {
                        double v = f(t);
                        if (!std::isfinite(v) || std::abs(v) > 100000 ||
                            (functionIndex == 2 && (v <= 0 || v > 100)) ||
                            (functionIndex == 4 && (v < 0 || v > 1)))
                            throw std::runtime_error("Invalid custom motion function for " +
                                                     l.name);
                    }
                }
                ++functionIndex;
            }
        }
    }
}
double projectDuration(const Project &p) {
    double d = 0;
    for (auto &s : p.slides)
        d += s.hold;
    for (auto &t : p.transitions)
        d += t.duration;
    return d;
}
const std::vector<std::pair<std::string, std::string>> &lifecycleEffects() {
    static const std::vector<std::pair<std::string, std::string>> effects{
        {"default", "Default (fade)"},
        {"none", "None (cut)"},
        {"fade", "Fade"},
        {"scale", "Grow / shrink"},
        {"slide_up", "Slide up"},
        {"slide_down", "Slide down"},
        {"slide_left", "Slide left"},
        {"slide_right", "Slide right"},
        {"wipe_left", "Wipe from left"},
        {"wipe_right", "Wipe from right"},
        {"wipe_top", "Wipe from top"},
        {"wipe_bottom", "Wipe from bottom"},
        {"spin", "Spin & grow"},
        {"pop", "Pop"}};
    return effects;
}
float ease(float t, const std::string &name) {
    t = std::clamp(t, 0.f, 1.f);
    if (name == "linear")
        return t;
    if (name == "ease_in")
        return t * t * t;
    if (name == "ease_out")
        return 1 - std::pow(1 - t, 3);
    return t * t * t * (t * (t * 6 - 15) + 10);
}
Renderer::Renderer(Project p, int w, int h)
    : project_(std::move(p)), width_(w), height_(h),
      sx_(std::min(float(w) / project_.width, float(h) / project_.height)), sy_(sx_),
      ox_((w - project_.width * sx_) * .5f), oy_((h - project_.height * sy_) * .5f) {
    validateProject(project_);
    gpu_.enableInputCaching(true);
    if (w < 16 || h < 16 || w > 8192 || h > 8192)
        throw std::runtime_error("Invalid render dimensions");
}
Renderer::Prepared Renderer::prepare(int index) {
    Prepared out;
    auto &sa = project_.slides.at(index);
    auto &sb = project_.slides.at(index + 1);
    auto &tr = project_.transitions.at(index);
    out.a = sa.image.clone();
    out.b = sb.image.clone();
    std::set<std::string> participatingA, participatingB;
    for (auto &l : tr.links) {
        participatingA.insert(l.sources.begin(), l.sources.end());
        participatingB.insert(l.targets.begin(), l.targets.end());
    }
    for (auto *slide : {&sa, &sb}) {
        cv::Mat &img = slide == &sa ? out.a : out.b;
        auto &participating = slide == &sa ? participatingA : participatingB;
        if (!slide->baseImage.empty()) {
            Slide residual = *slide;
            residual.objects.erase(std::remove_if(residual.objects.begin(), residual.objects.end(),
                                                  [&](const Object &o) {
                                                      return participating.count(o.id) &&
                                                             !o.artwork.empty();
                                                  }),
                                   residual.objects.end());
            recomposeSlide(residual);
            img = residual.image;
        }
        for (auto &o : slide->objects)
            if (participating.count(o.id) && o.artwork.empty()) {
                cv::Mat erase;
                cv::dilate(o.mask, erase, cv::Mat());
                img.setTo(
                    cv::Scalar(slide->background[0], slide->background[1], slide->background[2]),
                    erase);
            }
    }
    out.a = resized(out.a, width_, height_);
    out.b = resized(out.b, width_, height_);
    auto makeParts = [&](const Object *oa, const Object *ob, const Link &l) {
        std::vector<Component> aa = oa ? components(sa, *oa) : std::vector<Component>{},
                               bb = ob ? components(sb, *ob) : std::vector<Component>{};
        std::vector<std::pair<int, int>> pairings = l.componentPairs;
        std::set<int> ai, bi;
        for (auto [a, b] : pairings) {
            if (a < 0 || b < 0 || a >= (int)aa.size() || b >= (int)bb.size() ||
                !ai.insert(a).second || !bi.insert(b).second)
                throw std::runtime_error("Invalid or repeated manual component pair for " + l.name);
        }
        std::vector<int> freeA, freeB;
        for (int i = 0; i < (int)aa.size(); ++i)
            if (!ai.count(i))
                freeA.push_back(i);
        for (int i = 0; i < (int)bb.size(); ++i)
            if (!bi.count(i))
                freeB.push_back(i);
        auto ca = oa ? center(oa->mask) : cv::Point2f{}, cb = ob ? center(ob->mask) : cv::Point2f{};
        std::vector<std::vector<double>> cost(freeA.size(), std::vector<double>(freeB.size()));
        for (size_t i = 0; i < freeA.size(); ++i)
            for (size_t j = 0; j < freeB.size(); ++j) {
                auto &a = aa[freeA[i]];
                auto &b = bb[freeB[j]];
                cost[i][j] = cv::norm((a.c - ca) - (b.c - cb)) / project_.width +
                             std::abs(std::log(a.area / b.area)) * .2 +
                             cv::norm(a.color - b.color) * .25 +
                             std::abs(int(a.loops.size()) - int(b.loops.size())) * .15;
            }
        for (auto [i, j] : assignment(cost)) {
            pairings.emplace_back(freeA[i], freeB[j]);
            ai.insert(freeA[i]);
            bi.insert(freeB[j]);
        }
        for (int i = 0; i < (int)aa.size(); ++i)
            if (!ai.count(i))
                pairings.emplace_back(i, -1);
        for (int i = 0; i < (int)bb.size(); ++i)
            if (!bi.count(i))
                pairings.emplace_back(-1, i);
        std::vector<Part> parts;
        for (auto [a, b] : pairings) {
            Component *ac = a >= 0 ? &aa[a] : nullptr, *bc = b >= 0 ? &bb[b] : nullptr;
            Part part;
            part.centerA = oa ? ca : (ob ? cb : cv::Point2f{});
            part.centerB = ob ? cb : part.centerA;
            part.colorA = ac ? ac->color : bc->color;
            part.colorB = bc ? bc->color : part.colorA;
            part.opacityA =
                ac ? (oa && !oa->drawing.is_null() ? oa->drawing.value("opacity", 1.f) : 1.f) : 0;
            part.opacityB =
                bc ? (ob && !ob->drawing.is_null() ? ob->drawing.value("opacity", 1.f) : 1.f) : 0;
            int loops =
                std::max(ac ? ac->loops.size() : size_t(0), bc ? bc->loops.size() : size_t(0));
            for (int k = 0; k < loops; ++k) {
                auto *al = ac && k < (int)ac->loops.size() ? &ac->loops[k] : nullptr;
                auto *bl = bc && k < (int)bc->loops.size() ? &bc->loops[k] : nullptr;
                int n = std::clamp(int(std::max(al ? cv::arcLength(*al, true) : 0.,
                                                bl ? cv::arcLength(*bl, true) : 0.) /
                                       2),
                                   64, 512);
                CurvePair curve;
                curve.a =
                    al ? resample(*al, n) : std::vector<cv::Point2f>(n, ac ? ac->c : part.centerA);
                curve.b =
                    bl ? resample(*bl, n) : std::vector<cv::Point2f>(n, bc ? bc->c : part.centerB);
                if (al && bl)
                    align(curve.a, curve.b, part.centerA, part.centerB, l);
                if (al && bl && l.mode == "morph")
                    for (float t : {.25f, .5f, .75f}) {
                        std::vector<cv::Point2f> mid(n);
                        for (int v = 0; v < n; ++v)
                            mid[v] = curve.a[v] * (1 - t) + curve.b[v] * t;
                        if (selfIntersect(mid))
                            part.unsafe = true;
                    }
                part.curves.push_back(std::move(curve));
            }
            if (part.unsafe)
                warnings_.push_back(l.name +
                                    ": self-intersecting correspondence; using fade-through "
                                    "(adjust component pairs or boundary anchors)");
            parts.push_back(std::move(part));
        }
        return parts;
    };
    for (auto &l : tr.links) {
        PreparedLink pl;
        pl.link = l;
        int fi = 0;
        for (auto *expression :
             {&l.offsetX, &l.offsetY, &l.scaleFunction, &l.rotationFunction, &l.opacityFunction}) {
            if (!expression->empty())
                pl.functions[fi] = compileExpression(*expression);
            ++fi;
        }
        std::vector<const Object *> sources, targets;
        for (auto &id : l.sources)
            sources.push_back(findObject(sa, id));
        for (auto &id : l.targets)
            targets.push_back(findObject(sb, id));
        auto append = [&](const Object *a, const Object *b) {
            for (auto *o : {a, b})
                if (o && cv::countNonZero(o->mask))
                    pl.endpointBounds |= cv::boundingRect(o->mask);
            if (a && b && b->generation.is_object() &&
                b->generation.value("kind", "") == "updater" &&
                b->generation.value("source_id", "") == a->id &&
                b->generation.value("source_signature", "") != objectTransformSignature(sa, *a))
                throw std::runtime_error(
                    "Updater source changed; regenerate its next frame from frame Properties");
            if (a && b && l.mode == "morph" && !b->generation.is_null() && a->name == b->name &&
                b->generation.value("source_signature", "") == objectTransformSignature(sa, *a)) {
                if (b->generation.value("kind", "") == "updater") {
                    if (!b->generation.contains("result_signature") ||
                        b->generation.at("result_signature") != objectTransformSignature(sb, *b))
                        throw std::runtime_error("Updater target was changed outside the editor; "
                                                 "regenerate its transition");
                    double interval = tr.duration * (l.end - l.start);
                    if (std::abs(interval - b->generation.at("duration").get<double>()) > 1e-5)
                        throw std::runtime_error("Updater duration changed; regenerate from its "
                                                 "source with the new duration");
                    pl.updaters.push_back(
                        prepareObjectUpdater(sa, *a, b->generation.at("operation"), interval));
                } else
                    pl.meshes.push_back(
                        prepareObjectTransform(sa, *a, b->generation.at("operation")));
                return;
            }
            auto textured = [](const Object *o) {
                return o && !o->sourceAsset.empty() && !o->drawing.contains("construction");
            };
            auto construction = [](const Object *o) {
                if (!o)
                    return json{};
                json spec =
                    o->drawing.contains("construction") ? o->drawing["construction"] : o->drawing;
                if (spec.is_object())
                    for (auto key :
                         {"x", "y", "w", "h", "angle", "opacity", "color", "expressions"})
                        spec.erase(key);
                return spec;
            };
            auto equivalent = [&](const Object *a, const Object *b) {
                if (!a || !b || a->artwork.empty() || b->artwork.empty())
                    return false;
                auto ca = construction(a), cb = construction(b);
                if (ca != cb || !ca.is_object())
                    return false;
                // Shapes with fixed aspect (circle, equation, text) use an identical source patch.
                // Pixel groups need equal source alpha, never just an equal category name.
                if (ca.value("kind", "") == "pixels" || ca.value("kind", "") == "group") {
                    auto &ta = a->sourceAsset.empty() ? a->artwork : a->sourceAsset;
                    auto &tb = b->sourceAsset.empty() ? b->artwork : b->sourceAsset;
                    if (ta.size() != tb.size())
                        return false;
                    cv::Mat aa, ab;
                    cv::extractChannel(ta, aa, 3);
                    cv::extractChannel(tb, ab, 3);
                    return cv::norm(aa, ab, cv::NORM_INF) == 0;
                }
                return true;
            };
            if (!a || !b || textured(a) || textured(b) || equivalent(a, b)) {
                Sprite sprite;
                auto texture = [](const Slide &s, const Object *o, cv::Rect &bounds, float &angle,
                                  float &opacity) {
                    if (!o)
                        return cv::Mat(1, 1, CV_8UC4, cv::Scalar(0, 0, 0, 0));
                    if (!o->sourceAsset.empty() && !o->drawing.is_null() &&
                        o->drawing.value("kind", "") == "image") {
                        int x = o->drawing.value("x", 0), y = o->drawing.value("y", 0),
                            w = o->drawing.value("w", 100), h = o->drawing.value("h", 100);
                        bounds = {std::min(x, x + w), std::min(y, y + h), std::max(1, std::abs(w)),
                                  std::max(1, std::abs(h))};
                        angle = o->drawing.value("angle", 0.f) * pi / 180;
                        opacity = o->drawing.value("opacity", 1.f);
                        return o->sourceAsset;
                    }
                    if (!o->artwork.empty()) {
                        bounds = cv::Rect(o->artworkOrigin, o->artwork.size());
                        return o->artwork;
                    }
                    bounds = cv::boundingRect(o->mask);
                    cv::Mat rgba;
                    cv::cvtColor(s.image(bounds), rgba, cv::COLOR_BGR2BGRA);
                    for (int y = 0; y < rgba.rows; ++y)
                        for (int x = 0; x < rgba.cols; ++x)
                            rgba.at<cv::Vec4b>(y, x)[3] =
                                o->mask.at<uchar>(y + bounds.y, x + bounds.x);
                    return rgba;
                };
                sprite.a = texture(sa, a, sprite.boundsA, sprite.angleA, sprite.opacityA);
                sprite.b = texture(sb, b, sprite.boundsB, sprite.angleB, sprite.opacityB);
                if (!a) {
                    sprite.boundsA = sprite.boundsB;
                    sprite.angleA = sprite.angleB;
                    sprite.a = sprite.b;
                    sprite.opacityA = sprite.opacityB;
                    sprite.lifecycle = b->constructAnimation;
                    sprite.entering = true;
                }
                if (!b) {
                    sprite.boundsB = sprite.boundsA;
                    sprite.angleB = sprite.angleA;
                    sprite.b = sprite.a;
                    sprite.opacityB = sprite.opacityA;
                    sprite.lifecycle = a->destructAnimation;
                }
                if (!a || !b) {
                    auto o = a ? a : b;
                    sprite.effectStart = o->effectStart;
                    sprite.effectEnd = o->effectEnd;
                    sprite.effectEasing = o->effectEasing;
                }
                pl.sprites.push_back(sprite);
            } else {
                auto parts = makeParts(a, b, l);
                pl.parts.insert(pl.parts.end(), parts.begin(), parts.end());
            }
        };
        if (sources.empty())
            for (auto b : targets)
                append(nullptr, b);
        else if (targets.empty())
            for (auto a : sources)
                append(a, nullptr);
        else if (sources.size() == 1)
            for (auto b : targets)
                append(sources[0], b);
        else if (targets.size() == 1)
            for (auto a : sources)
                append(a, targets[0]);
        else {
            std::vector<std::vector<double>> costs(sources.size(),
                                                   std::vector<double>(targets.size()));
            for (size_t i = 0; i < sources.size(); ++i)
                for (size_t j = 0; j < targets.size(); ++j)
                    costs[i][j] = cv::norm(center(sources[i]->mask) - center(targets[j]->mask));
            auto matched = assignment(costs);
            std::set<int> usedA, usedB;
            for (auto [a, b] : matched) {
                usedA.insert(a);
                usedB.insert(b);
            }
            for (int j = 0; j < (int)targets.size(); ++j)
                if (!usedB.count(j)) {
                    int i = 0;
                    for (int k = 1; k < (int)sources.size(); ++k)
                        if (costs[k][j] < costs[i][j])
                            i = k;
                    matched.emplace_back(i, j);
                }
            for (int i = 0; i < (int)sources.size(); ++i)
                if (!usedA.count(i)) {
                    int j = std::min_element(costs[i].begin(), costs[i].end()) - costs[i].begin();
                    matched.emplace_back(i, j);
                }
            for (auto [a, b] : matched)
                append(sources[a], targets[b]);
        }
        out.links.push_back(std::move(pl));
    }
    std::stable_sort(out.links.begin(), out.links.end(),
                     [](auto &a, auto &b) { return a.link.layer < b.link.layer; });
    return out;
}
void Renderer::warmupAt(double seconds) {
    for (int i = 0; i < (int)project_.transitions.size(); ++i) {
        double interval = project_.slides[i].hold + project_.transitions[i].duration;
        if (seconds <= interval) {
            if (!cache_.count(i)) {
                if (cache_.size() >= 2)
                    cache_.erase(cache_.begin());
                cache_.emplace(i, prepare(i));
            }
            return;
        }
        seconds -= interval;
    }
}
cv::Mat Renderer::transitionFrame(int index, float t, const std::function<bool()> &cancelled) {
    if (index < 0 || index >= (int)project_.transitions.size())
        throw std::runtime_error("Transition index out of range");
    t = std::clamp(t, 0.f, 1.f);
    if (t <= 0)
        return resized(project_.slides[index].image, width_, height_);
    if (t >= 1)
        return resized(project_.slides[index + 1].image, width_, height_);
    if (cancelled && cancelled())
        return {};
    if (!cache_.count(index)) {
        if (cache_.size() >= 2)
            cache_.erase(cache_.begin());
        cache_.emplace(index, prepare(index));
    }
    if (cancelled && cancelled())
        return {};
    auto &p = cache_.at(index);
    std::vector<GpuDraw> draws;
    int lineage = 0;
    std::vector<cv::Rect> exactRegions;
    bool protectEndpoints =
        (t < .06f || t > .94f) && std::any_of(p.links.begin(), p.links.end(),
                                              [](const auto &link) { return !link.parts.empty(); });
    auto protect = [&](cv::Rect region) {
        if (!protectEndpoints)
            return;
        region &= cv::Rect(0, 0, width_, height_);
        if (!region.empty())
            exactRegions.push_back(region);
    };
    auto protectMesh = [&](const GpuDraw &d) {
        if (!protectEndpoints)
            return;
        float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
        for (size_t i = 0; i < d.from.size(); ++i) {
            auto x = d.from[i].x * (1 - d.t) + d.to[i].x * d.t;
            auto y = d.from[i].y * (1 - d.t) + d.to[i].y * d.t;
            if (!std::isfinite(x + y))
                continue;
            minX = std::min(minX, x);
            minY = std::min(minY, y);
            maxX = std::max(maxX, x);
            maxY = std::max(maxY, y);
        }
        if (minX <= maxX && minY <= maxY) {
            // Clip in floating point before converting large off-canvas function results.
            int x0 = int(std::floor(std::clamp(minX - 3, 0.f, float(width_)))),
                y0 = int(std::floor(std::clamp(minY - 3, 0.f, float(height_)))),
                x1 = int(std::ceil(std::clamp(maxX + 4, 0.f, float(width_)))),
                y1 = int(std::ceil(std::clamp(maxY + 4, 0.f, float(height_))));
            protect({x0, y0, x1 - x0, y1 - y0});
        }
    };
    for (auto &pl : p.links) {
        ++lineage;
        auto &l = pl.link;
        if (!pl.updaters.empty() || !pl.meshes.empty() || !pl.sprites.empty()) {
            auto b = pl.endpointBounds;
            protect({int(b.x * sx_ + ox_) - 3, int(b.y * sy_ + oy_) - 3, int(b.width * sx_) + 7,
                     int(b.height * sy_) + 7});
        }
        float progress = std::clamp(float((t - l.start) / (l.end - l.start)), 0.f, 1.f),
              u = ease(progress, l.easing);
        float angle = l.rotation * pi / 180;
        auto function = [&](int index, float fallback) {
            if (!pl.functions[index])
                return fallback;
            double v = pl.functions[index](progress);
            if (!std::isfinite(v) || std::abs(v) > 100000 || (index == 2 && (v <= 0 || v > 100)) ||
                (index == 4 && (v < 0 || v > 1)))
                throw std::runtime_error("Motion function outside its valid range for " + l.name);
            return float(v);
        };
        cv::Point2f offset(function(0, 0), function(1, 0));
        float scale = function(2, 1), extraAngle = function(3, 0) * pi / 180,
              extraOpacity = function(4, 1);
        for (const auto &updater : pl.updaters) {
            GpuDraw d = updater.source;
            d.lineage = lineage;
            d.t = 1;
            d.color[3] = extraOpacity;
            cv::Point2f center{};
            for (size_t i = 0; i < d.from.size(); ++i) {
                auto point = updater.evaluate({d.from[i].x, d.from[i].y}, progress);
                d.to[i] = {point.x, point.y};
                center += point;
            }
            center *= 1.f / d.from.size();
            float a = extraAngle + angle * std::sin(float(pi) * progress);
            for (auto &point : d.to) {
                auto local = (cv::Point2f(point.x, point.y) - center) * scale;
                auto q = center +
                         cv::Point2f(local.x * std::cos(a) - local.y * std::sin(a),
                                     local.x * std::sin(a) + local.y * std::cos(a)) +
                         offset;
                point = {q.x * sx_ + ox_, q.y * sy_ + oy_};
            }
            // from isn't sampled at t=1, but keep it in the same output coordinate system.
            for (auto &point : d.from)
                point = {point.x * sx_ + ox_, point.y * sy_ + oy_};
            protectMesh(d);
            draws.push_back(std::move(d));
        }
        for (auto &mesh : pl.meshes) {
            GpuDraw d = mesh;
            d.lineage = lineage;
            d.t = u;
            d.color[3] = extraOpacity;
            cv::Point2f ca{}, cb{};
            int count = 0;
            for (size_t i = 0; i < mesh.from.size(); ++i)
                if (std::isfinite(mesh.to[i].x + mesh.to[i].y)) {
                    ca += cv::Point2f(mesh.from[i].x, mesh.from[i].y);
                    cb += cv::Point2f(mesh.to[i].x, mesh.to[i].y);
                    ++count;
                }
            if (!count)
                continue;
            ca *= 1.f / count;
            cb *= 1.f / count;
            auto c = ca * (1 - u) + cb * u + offset, delta = cb - ca;
            c += cv::Point2f(-delta.y, delta.x) * (2 * l.bend * u * (1 - u));
            auto rotate = [](cv::Point2f v, float a) {
                return cv::Point2f(v.x * std::cos(a) - v.y * std::sin(a),
                                   v.x * std::sin(a) + v.y * std::cos(a));
            };
            for (size_t i = 0; i < d.from.size(); ++i) {
                auto a = c + rotate(cv::Point2f(mesh.from[i].x, mesh.from[i].y) - ca,
                                    angle * u + extraAngle) *
                                 scale,
                     b = c + rotate(cv::Point2f(mesh.to[i].x, mesh.to[i].y) - cb,
                                    -angle * (1 - u) + extraAngle) *
                                 scale;
                d.from[i] = {a.x * sx_ + ox_, a.y * sy_ + oy_};
                d.to[i] = {b.x * sx_ + ox_, b.y * sy_ + oy_};
            }
            protectMesh(d);
            draws.push_back(std::move(d));
        }
        for (auto &sprite : pl.sprites) {
            GpuDraw d;
            d.lineage = lineage;
            d.t = u;
            d.textureA = sprite.a;
            d.textureB = sprite.b;
            auto a = cv::Point2f(sprite.boundsA.x + sprite.boundsA.width * .5f,
                                 sprite.boundsA.y + sprite.boundsA.height * .5f),
                 b = cv::Point2f(sprite.boundsB.x + sprite.boundsB.width * .5f,
                                 sprite.boundsB.y + sprite.boundsB.height * .5f);
            auto c = a * (1 - u) + b * u, delta = b - a;
            c += cv::Point2f(-delta.y, delta.x) * (2 * l.bend * u * (1 - u)) + offset;
            d.centerX = c.x * sx_ + ox_;
            d.centerY = c.y * sy_ + oy_;
            d.extentX = std::max(1.f, (sprite.boundsA.width * (1 - u) + sprite.boundsB.width * u) *
                                          sx_ * scale);
            d.extentY = std::max(
                1.f, (sprite.boundsA.height * (1 - u) + sprite.boundsB.height * u) * sy_ * scale);
            d.angle = sprite.angleA +
                      std::remainder(sprite.angleB - sprite.angleA, float(2 * pi)) * u +
                      angle * std::sin(float(pi) * u) + extraAngle;
            d.textureOpacityA = sprite.opacityA;
            d.textureOpacityB = sprite.opacityB;
            if (!sprite.lifecycle.empty()) {
                float phase = std::clamp((progress - sprite.effectStart) /
                                             (sprite.effectEnd - sprite.effectStart),
                                         0.f, 1.f);
                float curved = ease(phase, sprite.effectEasing),
                      amount = sprite.entering ? curved : 1 - curved;
                bool wipe = sprite.lifecycle.rfind("wipe_", 0) == 0;
                float opacity = wipe ? 1.f : amount;
                if (sprite.lifecycle == "none")
                    opacity = sprite.entering ? (phase >= 1 ? 1.f : 0.f) : (phase >= 1 ? 0.f : 1.f);
                d.textureOpacityA *= opacity;
                d.textureOpacityB *= opacity;
                if (sprite.lifecycle == "scale" || sprite.lifecycle == "spin" ||
                    sprite.lifecycle == "pop") {
                    float factor = amount;
                    if (sprite.lifecycle == "pop")
                        factor = amount == 0 ? 0
                                             : 1 + 2.70158f * std::pow(amount - 1, 3) +
                                                   1.70158f * std::pow(amount - 1, 2);
                    d.extentX *= std::max(.0001f, factor);
                    d.extentY *= std::max(.0001f, factor);
                    if (sprite.lifecycle == "spin")
                        d.angle += (sprite.entering ? 1.f : -1.f) * (1 - amount) * float(2 * pi);
                } else if (sprite.lifecycle.rfind("slide_", 0) == 0) {
                    float direction =
                        (sprite.lifecycle == "slide_left" || sprite.lifecycle == "slide_up") ? -1.f
                                                                                             : 1.f;
                    float distance = direction * (sprite.entering ? -(1 - curved) : curved);
                    if (sprite.lifecycle == "slide_left" || sprite.lifecycle == "slide_right")
                        d.centerX += distance * width_ * .2f;
                    else
                        d.centerY += distance * height_ * .2f;
                } else if (wipe) {
                    if (sprite.lifecycle == "wipe_left")
                        d.textureClip[2] = amount;
                    else if (sprite.lifecycle == "wipe_right")
                        d.textureClip[0] = 1 - amount;
                    else if (sprite.lifecycle == "wipe_top")
                        d.textureClip[3] = amount;
                    else
                        d.textureClip[1] = 1 - amount;
                    if (amount <= 0)
                        d.textureOpacityA = d.textureOpacityB = 0;
                }
            }
            float radius = std::hypot(d.extentX, d.extentY) * .5f + 2;
            d.bounds = {int(d.centerX - radius), int(d.centerY - radius), int(radius * 2 + 2),
                        int(radius * 2 + 2)};
            protect(d.bounds);
            d.color[3] = extraOpacity;
            if (l.mode == "fade" && sprite.lifecycle.empty())
                d.color[3] *= std::abs(2 * u - 1);
            draws.push_back(std::move(d));
        }
        for (auto &part : pl.parts) {
            float shapeT = u, alpha = (part.opacityA * (1 - u) + part.opacityB * u) * extraOpacity;
            if (l.mode == "fade" || part.unsafe) {
                shapeT = u < .5f ? 0 : 1;
                alpha *= std::abs(2 * u - 1);
            }
            cv::Point2f c = part.centerA * (1 - u) + part.centerB * u + offset;
            auto delta = part.centerB - part.centerA;
            c += cv::Point2f(-delta.y, delta.x) * (2 * l.bend * u * (1 - u));
            auto rotate = [](cv::Point2f v, float a) {
                float s = std::sin(a), c = std::cos(a);
                return cv::Point2f(v.x * c - v.y * s, v.x * s + v.y * c);
            };
            GpuDraw d;
            d.t = shapeT;
            d.lineage = lineage;
            auto color = part.colorA * (1 - u) + part.colorB * u;
            d.color = {color[0], color[1], color[2], alpha};
            float minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
            for (auto &curve : part.curves) {
                d.contours.push_back({(int)d.from.size(), (int)curve.a.size()});
                for (size_t i = 0; i < curve.a.size(); ++i) {
                    auto a = c + rotate(curve.a[i] - part.centerA, angle * u + extraAngle) * scale,
                         b = c + rotate(curve.b[i] - part.centerB, -angle * (1 - u) + extraAngle) *
                                     scale;
                    a.x = a.x * sx_ + ox_;
                    a.y = a.y * sy_ + oy_;
                    b.x = b.x * sx_ + ox_;
                    b.y = b.y * sy_ + oy_;
                    d.from.push_back({a.x, a.y});
                    d.to.push_back({b.x, b.y});
                    auto pt = a * (1 - shapeT) + b * shapeT;
                    minx = std::min(minx, pt.x);
                    maxx = std::max(maxx, pt.x);
                    miny = std::min(miny, pt.y);
                    maxy = std::max(maxy, pt.y);
                }
            }
            if (!d.from.empty()) {
                d.bounds = {int(std::floor(minx)) - 2, int(std::floor(miny)) - 2,
                            int(std::ceil(maxx - minx)) + 5, int(std::ceil(maxy - miny)) + 5};
                draws.push_back(std::move(d));
            }
        }
    }
    cv::Mat result = gpu_.render(p.a, p.b, t, draws);
    // Exact original patches at endpoints, smoothly handing over to reconstructed geometry.
    float endpoint = t < .06f   ? 1 - ease(t / .06f, "smooth")
                     : t > .94f ? ease((t - .94f) / .06f, "smooth")
                                : 0;
    bool approximated = std::any_of(p.links.begin(), p.links.end(),
                                    [](const auto &link) { return !link.parts.empty(); });
    if (endpoint > 0 && approximated) {
        auto original =
            resized(project_.slides[t < .5f ? index : index + 1].image, width_, height_);
        cv::Mat reconstructed = exactRegions.empty() ? cv::Mat{} : result.clone();
        cv::addWeighted(result, 1 - endpoint, original, endpoint, 0, result);
        // Contour endpoint reconstruction must never pre-empt exact textured paths or lifecycle
        // effects.
        for (const auto &region : exactRegions)
            reconstructed(region).copyTo(result(region));
    }
    return result;
}
cv::Mat Renderer::frame(double seconds, const std::function<bool()> &cancelled) {
    if (project_.slides.empty())
        throw std::runtime_error("Import at least one slide before rendering");
    seconds = std::clamp(seconds, 0., projectDuration(project_));
    for (size_t i = 0; i < project_.slides.size(); ++i) {
        auto &s = project_.slides[i];
        if (seconds <= s.hold || i == project_.slides.size() - 1)
            return resized(s.image, width_, height_);
        seconds -= s.hold;
        if (seconds < project_.transitions[i].duration)
            return transitionFrame(i, seconds / project_.transitions[i].duration, cancelled);
        seconds -= project_.transitions[i].duration;
    }
    return resized(project_.slides.back().image, width_, height_);
}
void exportVideo(const Project &p, const ExportOptions &options, std::atomic_bool *cancel,
                 const std::function<void(int, int)> &progress) {
    validateProject(p);
    if (p.slides.empty())
        throw std::runtime_error("Import slides before exporting");
    if (options.width < 16 || options.height < 16 || options.width > 8192 ||
        options.height > 8192 || options.fps < 1 || options.fps > 240)
        throw std::runtime_error("Invalid export dimensions or frame rate");
    if (!options.pngSequence && (options.width % 2 || options.height % 2))
        throw std::runtime_error("H.264 export requires even dimensions");
    if (options.output.empty())
        throw std::runtime_error("Choose an export destination");
    Renderer renderer(p, options.width, options.height);
    int count = std::max(1, int(std::ceil(projectDuration(p) * options.fps - 1e-9)));
    fs::path dest = fs::absolute(options.output),
             temp = dest.parent_path() /
                    (dest.stem().string() + ".partial-" + newId() + dest.extension().string());
    fs::create_directories(dest.parent_path());
    int fd = -1;
    pid_t child = -1;
    auto cleanup = [&] {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        if (child > 0) {
            kill(child, SIGTERM);
            waitpid(child, nullptr, 0);
            child = -1;
        }
        std::error_code ec;
        if (options.pngSequence)
            fs::remove_all(temp, ec);
        else
            fs::remove(temp, ec);
    };
    try {
        if (options.pngSequence) {
            if (fs::exists(dest))
                throw std::runtime_error("PNG sequence destination already exists");
            fs::create_directories(temp);
        } else {
            signal(SIGPIPE, SIG_IGN);
            int pipes[2];
            if (pipe2(pipes, O_CLOEXEC))
                throw std::runtime_error("Cannot create encoder pipe");
            std::vector<std::string> arguments = {options.ffmpeg,
                                                  "-hide_banner",
                                                  "-loglevel",
                                                  "error",
                                                  "-y",
                                                  "-f",
                                                  "rawvideo",
                                                  "-pixel_format",
                                                  "bgr24",
                                                  "-video_size",
                                                  std::to_string(options.width) + "x" +
                                                      std::to_string(options.height),
                                                  "-framerate",
                                                  std::to_string(options.fps),
                                                  "-i",
                                                  "pipe:0",
                                                  "-an",
                                                  "-c:v",
                                                  "libx264",
                                                  "-preset",
                                                  "medium",
                                                  "-crf",
                                                  "18",
                                                  "-pix_fmt",
                                                  "yuv420p",
                                                  "-movflags",
                                                  "+faststart",
                                                  temp.string()};
            std::vector<char *> argv;
            for (auto &arg : arguments)
                argv.push_back(arg.data());
            argv.push_back(nullptr);
            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_adddup2(&actions, pipes[0], STDIN_FILENO);
            posix_spawn_file_actions_addclose(&actions, pipes[0]);
            posix_spawn_file_actions_addclose(&actions, pipes[1]);
            int result = posix_spawnp(&child, options.ffmpeg.c_str(), &actions, nullptr,
                                      argv.data(), environ);
            posix_spawn_file_actions_destroy(&actions);
            close(pipes[0]);
            if (result) {
                close(pipes[1]);
                child = -1;
                throw std::runtime_error("Cannot launch FFmpeg: " +
                                         std::string(std::strerror(result)));
            }
            fd = pipes[1];
            fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
        }
        for (int i = 0; i < count; ++i) {
            if (cancel && cancel->load())
                throw std::runtime_error("Export cancelled");
            auto image =
                renderer.frame(double(i) / options.fps, [&] { return cancel && cancel->load(); });
            if (image.empty())
                throw std::runtime_error("Export cancelled");
            if (options.pngSequence) {
                std::ostringstream name;
                name << "frame-" << std::setw(6) << std::setfill('0') << i << ".png";
                writeImage(temp / name.str(), image);
            } else {
                size_t bytes = image.total() * image.elemSize(), offset = 0;
                while (offset < bytes) {
                    if (cancel && cancel->load())
                        throw std::runtime_error("Export cancelled");
                    ssize_t wrote = write(fd, image.data + offset, bytes - offset);
                    if (wrote < 0 && errno == EINTR)
                        continue;
                    if (wrote < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                        pollfd event{fd, POLLOUT, 0};
                        poll(&event, 1, 100);
                        continue;
                    }
                    if (wrote <= 0)
                        throw std::runtime_error("FFmpeg stopped accepting frames; check encoder "
                                                 "availability and output path");
                    offset += wrote;
                }
            }
            if (progress)
                progress(i + 1, count);
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
            int status = 0;
            waitpid(child, &status, 0);
            child = -1;
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                throw std::runtime_error("FFmpeg export failed (install FFmpeg with libx264)");
        }
        fs::rename(temp, dest);
    } catch (...) {
        cleanup();
        throw;
    }
}
} // namespace fm
