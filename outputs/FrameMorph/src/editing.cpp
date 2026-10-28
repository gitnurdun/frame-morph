#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace fm {
namespace {
void place(Slide &s, Object &o, json spec) {
    double opacity = spec.value("opacity", 1.);
    if (opacity == 0)
        spec["opacity"] = 1.;
    renderDrawing(s, o, spec);
    if (opacity == 0) {
        o.artwork = o.artwork.clone();
        for (int y = 0; y < o.artwork.rows; ++y)
            for (int x = 0; x < o.artwork.cols; ++x)
                o.artwork.at<cv::Vec4b>(y, x)[3] = 0;
        o.drawing["opacity"] = 0.;
        recomposeSlide(s);
    }
}
} // namespace
std::string defaultObjectName(const Slide &slide, const std::string &kind) {
    for (int i = 1;; ++i) {
        auto name = "__" + kind + "_" + std::to_string(i);
        if (std::none_of(slide.objects.begin(), slide.objects.end(),
                         [&](const Object &o) { return o.name == name; }))
            return name;
    }
}
json objectFrame(const Object &o) {
    if (!o.drawing.is_null() && o.drawing.value("kind", "") == "image")
        return {
            {"x", o.drawing.value("x", 0.)},         {"y", o.drawing.value("y", 0.)},
            {"w", o.drawing.value("w", 100.)},       {"h", o.drawing.value("h", 100.)},
            {"angle", o.drawing.value("angle", 0.)}, {"opacity", o.drawing.value("opacity", 1.)}};
    auto b = cv::boundingRect(o.mask);
    return {{"x", b.x},      {"y", b.y},    {"w", b.width},
            {"h", b.height}, {"angle", 0.}, {"opacity", 1.}};
}
void makeObjectEditable(Slide &s, Object &o) {
    if (!o.sourceAsset.empty() && !o.drawing.is_null() && o.drawing.value("kind", "") == "image")
        return;
    auto frame = objectFrame(o);
    auto b = cv::boundingRect(o.mask);
    if (b.empty())
        throw std::runtime_error("Cannot edit an empty object");
    json construction = o.drawing.is_null() ? json{{"kind", "pixels"}} : o.drawing;
    if (o.drawing.is_null()) {
        auto c = cv::mean(s.image, o.mask);
        char hex[8];
        std::snprintf(hex, sizeof(hex), "#%02x%02x%02x", int(c[2] + .5), int(c[1] + .5),
                      int(c[0] + .5));
        construction["color"] = hex;
    }
    if (!o.artwork.empty()) {
        o.sourceAsset = o.artwork;
        frame["x"] = o.artworkOrigin.x;
        frame["y"] = o.artworkOrigin.y;
        frame["w"] = o.artwork.cols;
        frame["h"] = o.artwork.rows;
    } else {
        cv::cvtColor(s.image(b), o.sourceAsset, cv::COLOR_BGR2BGRA);
        for (int y = 0; y < b.height; ++y)
            for (int x = 0; x < b.width; ++x)
                o.sourceAsset.at<cv::Vec4b>(y, x)[3] = o.mask.at<uchar>(y + b.y, x + b.x);
        s.baseImage = (s.baseImage.empty() ? s.image : s.baseImage).clone();
        cv::Mat erase;
        cv::dilate(o.mask, erase, cv::Mat());
        s.baseImage.setTo(cv::Scalar(s.background[0], s.background[1], s.background[2]), erase);
    }
    o.drawing = frame;
    o.drawing["kind"] = "image";
    o.drawing["construction"] = construction;
    o.drawing["color"] = construction.value("color", "#4ed5b8");
}
void setObjectFrame(Slide &s, Object &o, const json &frame) {
    for (auto key : {"x", "y", "w", "h", "angle", "opacity"}) {
        double v = frame.at(key).get<double>();
        if (!std::isfinite(v) || std::abs(v) > 100000)
            throw std::runtime_error("Invalid object geometry");
    }
    if (frame.at("w").get<double>() < 1 || frame.at("h").get<double>() < 1)
        throw std::runtime_error("Object dimensions must be positive");
    if (frame.at("opacity").get<double>() < 0 || frame.at("opacity").get<double>() > 1)
        throw std::runtime_error("Opacity must be between zero and one");
    makeObjectEditable(s, o);
    auto spec = o.drawing;
    for (auto key : {"x", "y", "w", "h", "angle", "opacity"})
        spec[key] = frame.at(key);
    o.generation = json{};
    place(s, o, spec);
}
void setObjectColor(Slide &s, Object &o, const std::string &hex) {
    unsigned r, g, b;
    char tail;
    if (hex.size() != 7 || std::sscanf(hex.c_str(), "#%2x%2x%2x%c", &r, &g, &b, &tail) != 3)
        throw std::runtime_error("Use a color such as #4ed5b8");
    makeObjectEditable(s, o);
    auto native = o.drawing.contains("construction") ? o.drawing["construction"] : o.drawing;
    auto kind = native.value("kind", "image");
    bool tint = kind == "image" || kind == "svg" || kind == "code" || kind == "group";
    if (tint) {
        if (o.originalAsset.empty())
            o.originalAsset = o.sourceAsset.clone();
        auto settings = o.imageEdits.is_object() ? o.imageEdits : json::object();
        settings["tint"] = hex;
        if (!settings.contains("width"))
            settings["width"] = int(o.drawing.value("w", 100.));
        if (!settings.contains("height"))
            settings["height"] = int(o.drawing.value("h", 100.));
        o.sourceAsset = processImage(o.originalAsset, settings);
        o.imageEdits = settings;
        auto spec = o.drawing;
        spec["color"] = hex;
        o.generation = json{};
        place(s, o, spec);
        return;
    }
    o.sourceAsset = o.sourceAsset.clone();
    for (int y = 0; y < o.sourceAsset.rows; ++y)
        for (int x = 0; x < o.sourceAsset.cols; ++x) {
            auto &p = o.sourceAsset.at<cv::Vec4b>(y, x);
            p[0] = b;
            p[1] = g;
            p[2] = r;
        }
    auto spec = o.drawing;
    spec["color"] = hex;
    if (spec.contains("construction"))
        spec["construction"]["color"] = hex;
    o.generation = json{};
    place(s, o, spec);
}
void editObjectConstruction(Slide &s, Object &o, const json &spec) {
    if (o.drawing.contains("construction")) {
        Object rebuilt;
        Slide temporary = s;
        auto kind = spec.value("kind", "");
        if (kind == "equation" || kind == "code" || kind == "svg") {
            rebuilt.artwork = renderMedia(spec);
            rebuilt.drawing = spec;
        } else
            renderDrawing(temporary, rebuilt, spec);
        o.originalAsset.release();
        o.imageEdits = json{};
        o.sourceAsset = rebuilt.artwork;
        auto placed = o.drawing;
        auto previous = o.drawing["construction"];
        if ((spec.value("kind", "") == "text" || spec.value("kind", "") == "equation") &&
            spec.value("font_size", 56.) != previous.value("font_size", 56.)) {
            double ratio = spec.value("font_size", 56.) / previous.value("font_size", 56.);
            placed["w"] = placed.value("w", 100.) * ratio;
            placed["h"] = placed.value("h", 100.) * ratio;
        }
        placed["construction"] = rebuilt.drawing;
        placed["color"] = spec.value("color", "#4ed5b8");
        o.generation = json{};
        place(s, o, placed);
    } else {
        o.generation = json{};
        renderDrawing(s, o, spec);
    }
}
} // namespace fm
