#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace fm {
namespace {
using Z = std::complex<double>;
using ComplexFunction = std::function<Z(Z, double)>;
class ComplexExpression {
    std::string text;
    size_t pos = 0;
    int depth = 0;
    std::map<std::string, double> constants;
    void space() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
            ++pos;
    }
    bool take(char c) {
        space();
        if (pos < text.size() && text[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    }
    ComplexFunction sum() {
        auto a = product();
        for (;;) {
            if (take('+')) {
                auto b = product();
                a = [a, b](Z z, double t) { return a(z, t) + b(z, t); };
            } else if (take('-')) {
                auto b = product();
                a = [a, b](Z z, double t) { return a(z, t) - b(z, t); };
            } else
                return a;
        }
    }
    ComplexFunction product() {
        auto a = unary();
        for (;;) {
            if (take('*')) {
                auto b = unary();
                a = [a, b](Z z, double t) { return a(z, t) * b(z, t); };
            } else if (take('/')) {
                auto b = unary();
                a = [a, b](Z z, double t) { return a(z, t) / b(z, t); };
            } else
                return a;
        }
    }
    ComplexFunction unary() {
        if (++depth > 64)
            throw std::runtime_error("Complex expression is nested too deeply");
        struct Guard {
            int &n;
            ~Guard() {
                --n;
            }
        } guard{depth};
        if (take('+'))
            return unary();
        if (take('-')) {
            auto a = unary();
            return [a](Z z, double t) { return -a(z, t); };
        }
        auto a = primary();
        if (take('^')) {
            auto b = unary();
            return [a, b](Z z, double t) { return std::pow(a(z, t), b(z, t)); };
        }
        return a;
    }
    ComplexFunction primary() {
        if (take('(')) {
            auto a = sum();
            if (!take(')'))
                throw std::runtime_error("Missing closing complex parenthesis");
            return a;
        }
        space();
        if (pos >= text.size())
            throw std::runtime_error("Incomplete complex expression");
        if (std::isdigit(static_cast<unsigned char>(text[pos])) || text[pos] == '.') {
            const char *begin = text.c_str() + pos;
            char *end = nullptr;
            double value = std::strtod(begin, &end);
            if (end == begin || !std::isfinite(value))
                throw std::runtime_error("Invalid complex number");
            pos += end - begin;
            return [value](Z, double) { return Z(value, 0); };
        }
        size_t begin = pos;
        while (pos < text.size() && (std::isalnum(static_cast<unsigned char>(text[pos])) ||
                                     text[pos] == '_' || text[pos] == '.'))
            ++pos;
        std::string name = text.substr(begin, pos - begin);
        if (name == "z")
            return [](Z z, double) { return z; };
        if (name == "t")
            return [](Z, double t) { return Z(t, 0); };
        if (name == "seconds") {
            double duration = constants.count("duration") ? constants.at("duration") : 1.;
            return [duration](Z, double t) { return Z(t * duration, 0); };
        }
        if (name == "i")
            return [](Z, double) { return Z(0, 1); };
        if (name == "pi")
            return [](Z, double) { return Z(3.141592653589793, 0); };
        if (name == "e")
            return [](Z, double) { return Z(2.718281828459045, 0); };
        if (name == "origin") {
            Z value(constants.at("origin_x"), constants.at("origin_y"));
            return [value](Z, double) { return value; };
        }
        if (auto found = constants.find(name); found != constants.end()) {
            double value = found->second;
            return [value](Z, double) { return Z(value, 0); };
        }
        std::function<Z(Z, double)> f;
        if (name == "sin")
            f = [](Z z, double) { return std::sin(z); };
        else if (name == "cos")
            f = [](Z z, double) { return std::cos(z); };
        else if (name == "tan")
            f = [](Z z, double) { return std::tan(z); };
        else if (name == "exp")
            f = [](Z z, double) { return std::exp(z); };
        else if (name == "log" || name == "ln")
            f = [](Z z, double) { return std::log(z); };
        else if (name == "sqrt")
            f = [](Z z, double) { return std::sqrt(z); };
        else if (name == "conj")
            f = [](Z z, double) { return std::conj(z); };
        else if (name == "abs")
            f = [](Z z, double) { return Z(std::abs(z), 0); };
        else if (name == "arg")
            f = [](Z z, double) { return Z(std::arg(z), 0); };
        else if (name == "re" || name == "real")
            f = [](Z z, double) { return Z(z.real(), 0); };
        else if (name == "im" || name == "imag")
            f = [](Z z, double) { return Z(z.imag(), 0); };
        else
            throw std::runtime_error("Unknown complex function or variable: " + name);
        if (!take('('))
            throw std::runtime_error("Complex functions need parentheses");
        auto argument = sum();
        if (!take(')'))
            throw std::runtime_error("Missing complex function closing parenthesis");
        return [f, argument](Z z, double t) { return f(argument(z, t), t); };
    }

  public:
    explicit ComplexExpression(std::string value, const std::map<std::string, double> &vars = {})
        : text(std::move(value)), constants(vars) {
        if (text.empty() || text.size() > 4096)
            throw std::runtime_error("Complex formula must contain 1 to 4096 characters");
    }
    ComplexFunction compile() {
        auto f = sum();
        space();
        if (pos != text.size())
            throw std::runtime_error(
                "Unexpected complex formula suffix; use explicit multiplication");
        return f;
    }
};

// Conservative symbolic spatial degree. Time-only expressions are constants here.
// Unknown/nonlinear spatial functions use the dense mesh, never a guessed affine fit.
class SpatialDegree {
    const std::string text;
    size_t pos = 0;
    void space() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
            ++pos;
    }
    bool take(char c) {
        space();
        if (pos < text.size() && text[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    }
    int sum() {
        int a = product();
        for (;;) {
            if (take('+') || take('-'))
                a = std::max(a, product());
            else
                return a;
        }
    }
    int product() {
        int a = power();
        for (;;) {
            if (take('*'))
                a = std::min(2, a + power());
            else if (take('/')) {
                int b = power();
                a = b ? 2 : a;
            } else
                return a;
        }
    }
    int power() {
        if (take('+') || take('-'))
            return power();
        int a = primary();
        if (take('^')) {
            int b = power();
            return a || b ? 2 : 0;
        }
        return a;
    }
    int primary() {
        if (take('(')) {
            int d = sum();
            if (!take(')'))
                throw std::runtime_error("Unbalanced updater expression");
            return d;
        }
        space();
        if (pos == text.size())
            throw std::runtime_error("Incomplete updater expression");
        if (std::isdigit(static_cast<unsigned char>(text[pos])) || text[pos] == '.') {
            char *end = nullptr;
            std::strtod(text.c_str() + pos, &end);
            pos = end - text.c_str();
            return 0;
        }
        size_t begin = pos;
        while (pos < text.size() && (std::isalnum(static_cast<unsigned char>(text[pos])) ||
                                     text[pos] == '_' || text[pos] == '.'))
            ++pos;
        if (pos == begin)
            throw std::runtime_error("Invalid updater expression");
        auto name = text.substr(begin, pos - begin);
        if (take('(')) {
            int d = sum();
            if (!take(')'))
                throw std::runtime_error("Unbalanced updater function");
            return name == "re" || name == "im" || name == "real" || name == "imag" ||
                           name == "conj"
                       ? d
                       : (d ? 2 : 0);
        }
        return name == "x" || name == "y" || name == "z" ? 1 : 0;
    }

  public:
    explicit SpatialDegree(std::string value) : text(std::move(value)) {}
    int degree() {
        int result = sum();
        space();
        return pos == text.size() ? result : 2;
    }
};
struct Mapping {
    std::function<cv::Point2f(cv::Point2f)> f;
    bool affine = false;
};
Mapping mapping(const Slide &s, const Object &o, const json &spec, int depth = 0) {
    if (depth > 2)
        throw std::runtime_error("Nested mapping sequences are not supported");
    if (spec.value("kind", "") == "sequence") {
        if (!spec.at("operations").is_array() || spec.at("operations").size() > 64)
            throw std::runtime_error("A mapping sequence supports at most 64 operations");
        std::vector<Mapping> stages;
        for (const auto &stage : spec.at("operations"))
            stages.push_back(mapping(s, o, stage, depth + 1));
        return {[stages](cv::Point2f point) {
                    for (const auto &stage : stages)
                        point = stage.f(point);
                    return point;
                },
                false};
    }
    cv::Rect bounds = cv::boundingRect(o.mask);
    cv::Point2d origin(bounds.x + bounds.width * .5, bounds.y + bounds.height * .5);
    std::string originMode = spec.value("origin_mode", "object");
    if (originMode == "canvas")
        origin = {s.image.cols * .5, s.image.rows * .5};
    else if (originMode == "custom")
        origin = {spec.value("origin_x", 0.), spec.value("origin_y", 0.)};
    double units = spec.value("units", 100.);
    if (!std::isfinite(units) || units <= .0001 || units > 1e6)
        throw std::runtime_error("Transform scale must be positive pixels per mathematical unit");
    auto constants = spec.value("constants", std::map<std::string, double>{});
    std::string kind = spec.value("kind", "matrix");
    std::function<cv::Point2d(cv::Point2d)> operation;
    bool affine = false;
    if (kind == "matrix") {
        std::string input = spec.value("matrix", "1,0,0;0,1,0;0,0,1");
        std::replace(input.begin(), input.end(), '\n', ';');
        input.erase(
            std::remove_if(input.begin(), input.end(),
                           [](unsigned char c) { return c == '[' || c == ']' || std::isspace(c); }),
            input.end());
        std::replace(input.begin(), input.end(), ';', ',');
        std::vector<double> values;
        std::istringstream stream(input);
        std::string item;
        while (std::getline(stream, item, ','))
            if (!item.empty())
                values.push_back(evaluateExpression(item, 0));
        std::array<double, 9> m{1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (values.size() == 4) {
            m = {values[0], values[1], 0, values[2], values[3], 0, 0, 0, 1};
        } else if (values.size() == 6) {
            m = {values[0], values[1], values[2], values[3], values[4], values[5], 0, 0, 1};
        } else if (values.size() == 9)
            std::copy(values.begin(), values.end(), m.begin());
        else
            throw std::runtime_error("Matrix must have 4, 6, or 9 entries (2x2, 2x3, or 3x3)");
        for (auto v : m)
            if (!std::isfinite(v))
                throw std::runtime_error("Matrix entries must be finite");
        affine = std::abs(m[6]) + std::abs(m[7]) < 1e-12 && std::abs(m[8]) > 1e-12;
        operation = [m](cv::Point2d p) {
            double w = m[6] * p.x + m[7] * p.y + m[8];
            return cv::Point2d((m[0] * p.x + m[1] * p.y + m[2]) / w,
                               (m[3] * p.x + m[4] * p.y + m[5]) / w);
        };
    } else if (kind == "map") {
        auto x = compileExpression2D(spec.value("x", "x"), constants),
             y = compileExpression2D(spec.value("y", "y"), constants);
        operation = [x, y](cv::Point2d p) { return cv::Point2d(x(p.x, p.y), y(p.x, p.y)); };
    } else if (kind == "complex") {
        auto f = ComplexExpression(spec.value("formula", "z"), constants).compile();
        operation = [f](cv::Point2d p) {
            auto z = f(Z(p.x, p.y), 1.);
            return cv::Point2d(z.real(), z.imag());
        };
    } else
        throw std::runtime_error("Unknown object transform kind");
    Mapping result;
    result.affine = affine;
    result.f = [origin, units, operation](cv::Point2f p) {
        auto q = operation({(p.x - origin.x) / units, (origin.y - p.y) / units});
        double x = origin.x + q.x * units, y = origin.y - q.y * units;
        if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) > 1e7 || std::abs(y) > 1e7)
            return cv::Point2f(std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN());
        return cv::Point2f(x, y);
    };
    return result;
}
} // namespace
std::map<std::string, double> sceneVariables(const Slide &s, const Object *o) {
    std::map<std::string, double> v{{"scene.width", s.image.cols},
                                    {"scene.height", s.image.rows},
                                    {"scene.origin_x", s.image.cols * .5},
                                    {"scene.origin_y", s.image.rows * .5},
                                    {"scene.left", 0},
                                    {"scene.top", 0},
                                    {"scene.right", s.image.cols},
                                    {"scene.bottom", s.image.rows},
                                    {"tau", 2 * CV_PI},
                                    {"duration", 1.}};
    if (o) {
        auto f = objectFrame(*o);
        double x = f["x"], y = f["y"], w = f["w"], h = f["h"];
        v.insert({{"object.x", x},
                  {"object.y", y},
                  {"object.width", w},
                  {"object.height", h},
                  {"object.center_x", x + w / 2},
                  {"object.center_y", y + h / 2}});
    }
    return v;
}
json parseObjectMap(const std::string &input, const Slide &s, const Object &o, double units) {
    if (!std::isfinite(units) || units <= .0001 || units > 1e6)
        throw std::runtime_error("Pixels per unit must be positive and at most 1000000");
    if (input.size() > 4096)
        throw std::runtime_error("Mapping is too long (maximum 4096 characters)");
    auto colon = input.find(':');
    std::string prefix = input.substr(0, colon);
    prefix.erase(std::remove_if(prefix.begin(), prefix.end(),
                                [](unsigned char c) { return std::isspace(c); }),
                 prefix.end());
    if (colon == std::string::npos || (prefix != "(x,y)" && prefix != "z"))
        throw std::runtime_error("Start with (x,y):x+1,y+2 or z:z+2. Use * for multiplication.");
    auto bounds = cv::boundingRect(o.mask);
    double cx = bounds.x + bounds.width * .5, cy = bounds.y + bounds.height * .5;
    std::map<std::string, double> vars{{"scene.width", s.image.cols / units},
                                       {"scene.height", s.image.rows / units},
                                       {"origin_x", (s.image.cols * .5 - cx) / units},
                                       {"origin_y", (cy - s.image.rows * .5) / units},
                                       {"scene.left", -cx / units},
                                       {"scene.right", (s.image.cols - cx) / units},
                                       {"scene.top", cy / units},
                                       {"scene.bottom", (cy - s.image.rows) / units},
                                       {"object.width", bounds.width / units},
                                       {"object.height", bounds.height / units},
                                       {"center_x", (cx - s.image.cols * .5) / units},
                                       {"center_y", (s.image.rows * .5 - cy) / units},
                                       {"px", 1. / units},
                                       {"pixels_per_unit", units},
                                       {"tau", 2 * CV_PI},
                                       {"duration", 1.}};
    vars["scene.origin_x"] = vars["origin_x"];
    vars["scene.origin_y"] = vars["origin_y"];
    json spec = {{"units", units}, {"origin_mode", "custom"}, {"origin_x", cx},
                 {"origin_y", cy}, {"constants", vars},       {"input", input}};
    std::string body = input.substr(colon + 1);
    if (prefix == "z") {
        spec["kind"] = "complex";
        spec["formula"] = body;
    } else {
        int depth = 0;
        size_t comma = std::string::npos;
        for (size_t i = 0; i < body.size(); ++i) {
            if (body[i] == '(')
                ++depth;
            if (body[i] == ')')
                --depth;
            if (depth < 0)
                throw std::runtime_error("Unbalanced mapping parentheses");
            if (body[i] == ',' && depth == 0) {
                if (comma != std::string::npos)
                    throw std::runtime_error("A point map needs exactly two expressions");
                comma = i;
            }
        }
        if (depth || comma == std::string::npos)
            throw std::runtime_error("Use (x,y):<new x>,<new y> with balanced parentheses");
        spec["kind"] = "map";
        spec["x"] = body.substr(0, comma);
        spec["y"] = body.substr(comma + 1);
    }
    mapping(s, o, spec); // Compile before any model mutation.
    return spec;
}
void applyObjectMap(Slide &s, Object &o, const json &operation) {
    auto draw = prepareObjectTransform(s, o, operation);
    CudaBackend gpu;
    auto rgba = gpu.warp(draw, s.image.cols, s.image.rows);
    cv::Mat alpha;
    cv::extractChannel(rgba, alpha, 3);
    auto bounds = cv::boundingRect(alpha);
    if (bounds.empty())
        throw std::runtime_error(
            "Mapping moves the entire object outside the slide or makes it empty");
    auto signature = objectTransformSignature(s, o);
    json generation = {
        {"source_id", o.generation.is_object() ? o.generation.value("parent_id", o.id) : o.id},
        {"source_signature", signature},
        {"operation", operation}};
    if (o.generation.is_object() && o.generation.value("result_signature", "") == signature) {
        generation = o.generation;
        auto prior = generation.at("operation");
        json stages =
            prior.value("kind", "") == "sequence" ? prior.at("operations") : json::array({prior});
        stages.push_back(operation);
        // Bound project complexity and prohibit recursively nested recipes.
        if (stages.size() > 64)
            throw std::runtime_error(
                "64 accumulated maps reached; duplicate or flatten the object before continuing");
        generation["operation"] = {{"kind", "sequence"}, {"operations", stages}};
    }
    if (o.artwork.empty()) {
        s.baseImage = (s.baseImage.empty() ? s.image : s.baseImage).clone();
        cv::Mat erase;
        cv::dilate(o.mask, erase, cv::Mat());
        s.baseImage.setTo(cv::Scalar(s.background[0], s.background[1], s.background[2]), erase);
    }
    o.mask = alpha;
    o.artwork = rgba(bounds).clone();
    o.artworkOrigin = bounds.tl();
    o.sourceAsset = o.artwork;
    o.drawing = {{"kind", "image"},    {"x", bounds.x}, {"y", bounds.y}, {"w", bounds.width},
                 {"h", bounds.height}, {"angle", 0},    {"opacity", 1}};
    o.generation = generation;
    o.generation["result_signature"] = objectTransformSignature(s, o);
    recomposeSlide(s);
}
GpuDraw prepareObjectTransform(const Slide &s, const Object &o, const json &spec) {
    auto transform = mapping(s, o, spec);
    GpuDraw draw;
    cv::Rect bounds;
    if (!o.artwork.empty()) {
        draw.textureA = o.artwork;
        bounds = cv::Rect(o.artworkOrigin, o.artwork.size());
    } else {
        bounds = cv::boundingRect(o.mask);
        if (bounds.empty())
            throw std::runtime_error("Cannot transform an empty object");
        cv::cvtColor(s.image(bounds), draw.textureA, cv::COLOR_BGR2BGRA);
        for (int y = 0; y < bounds.height; ++y)
            for (int x = 0; x < bounds.width; ++x)
                draw.textureA.at<cv::Vec4b>(y, x)[3] = o.mask.at<uchar>(y + bounds.y, x + bounds.x);
    }
    int columns = transform.affine ? 1 : std::clamp((bounds.width + 7) / 8, 1, 256),
        rows = transform.affine ? 1 : std::clamp((bounds.height + 7) / 8, 1, 256);
    for (int j = 0; j <= rows; ++j)
        for (int i = 0; i <= columns; ++i) {
            cv::Point2f p(bounds.x + float(bounds.width) * i / columns,
                          bounds.y + float(bounds.height) * j / rows),
                q = transform.f(p);
            draw.from.push_back({p.x, p.y});
            draw.to.push_back({q.x, q.y});
            draw.meshUV.push_back({float(i) / columns, float(j) / rows});
        }
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < columns; ++i) {
            int a = j * (columns + 1) + i, b = a + 1, c = a + columns + 1, d = c + 1;
            for (int index : {a, b, d, a, d, c})
                draw.triangles.push_back(index);
        }
    draw.t = 1;
    draw.bounds = {0, 0, s.image.cols, s.image.rows};
    return draw;
}

json parseObjectUpdater(const std::string &input, const Slide &s, const Object &o, double units) {
    std::string canonical = input;
    auto colon = canonical.find(':');
    auto prefix = canonical.substr(0, colon);
    prefix.erase(std::remove_if(prefix.begin(), prefix.end(),
                                [](unsigned char c) { return std::isspace(c); }),
                 prefix.end());
    if (colon != std::string::npos && prefix == "(x,y,t)")
        canonical = "(x,y):" + canonical.substr(colon + 1);
    auto spec = parseObjectMap(canonical, s, o, units);
    spec["input"] = input;
    spec["updater"] = true;
    return spec;
}
ObjectUpdaterPlan prepareObjectUpdater(const Slide &s, const Object &o, const json &spec,
                                       double duration) {
    if (!std::isfinite(duration) || duration <= 0 || duration > 3600)
        throw std::runtime_error(
            "Updater duration must be greater than 0 and at most 3600 seconds");
    // Obtain a dense source texture mesh from the identity map, without baking an endpoint.
    auto identity = parseObjectMap("(x,y):x,y", s, o, spec.value("units", 100.));
    ObjectUpdaterPlan result;
    result.affine =
        spec.value("kind", "") == "map"
            ? SpatialDegree(spec.at("x")).degree() <= 1 && SpatialDegree(spec.at("y")).degree() <= 1
            : SpatialDegree(spec.at("formula")).degree() <= 1;
    if (result.affine) {
        identity["kind"] = "matrix";
        identity["matrix"] = "1,0;0,1";
    }
    result.source = prepareObjectTransform(s, o, identity);
    auto vars = spec.value("constants", std::map<std::string, double>{});
    vars["duration"] = duration;
    std::function<cv::Point2d(cv::Point2d, double)> fn;
    if (spec.value("kind", "") == "map") {
        auto fx = compileExpression3D(spec.at("x"), vars),
             fy = compileExpression3D(spec.at("y"), vars);
        fn = [fx, fy](cv::Point2d p, double t) {
            return cv::Point2d(fx(p.x, p.y, t), fy(p.x, p.y, t));
        };
    } else if (spec.value("kind", "") == "complex") {
        auto f = ComplexExpression(spec.at("formula"), vars).compile();
        fn = [f](cv::Point2d p, double t) {
            auto z = f(Z(p.x, p.y), t);
            return cv::Point2d(z.real(), z.imag());
        };
    } else
        throw std::runtime_error("Updater must start with (x,y,t): or z:");
    double units = spec.at("units"), cx = spec.at("origin_x"), cy = spec.at("origin_y");
    result.evaluate = [fn, units, cx, cy](cv::Point2f p, double t) {
        auto q = fn({(p.x - cx) / units, (cy - p.y) / units}, t);
        double x = cx + units * q.x, y = cy - units * q.y;
        if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) > 1e7 || std::abs(y) > 1e7)
            throw std::runtime_error("Updater produces a non-finite or excessively distant point");
        return cv::Point2f(x, y);
    };
    // Identity at t=0 prevents a jump from the authored source frame.
    for (const auto &p : result.source.from)
        if (cv::norm(result.evaluate({p.x, p.y}, 0) - cv::Point2f(p.x, p.y)) > .05)
            throw std::runtime_error("Updater must reproduce the current pose at t=0. For a circle "
                                     "use x+r*(cos(tau*t)-1),y+r*sin(tau*t).");
    // Preflight representative times; every requested render also checks every mapped point.
    for (double t : {.25, .5, .75, 1.})
        for (const auto &p : result.source.from)
            result.evaluate({p.x, p.y}, t);
    return result;
}
Slide generateUpdaterKeyframe(const Slide &source, const std::map<std::string, json> &updaters,
                              double duration) {
    if (updaters.empty())
        throw std::runtime_error("Save an updater on at least one object first");
    Slide target = duplicateSlide(source);
    target.title = source.title + " updated";
    CudaBackend gpu;
    for (const auto &entry : updaters)
        if (std::none_of(source.objects.begin(), source.objects.end(),
                         [&](const auto &o) { return o.id == entry.first; }))
            throw std::runtime_error("Updater refers to an object missing from the source slide");
    for (size_t i = 0; i < source.objects.size(); ++i) {
        const auto &original = source.objects[i];
        auto found = updaters.find(original.id);
        if (found == updaters.end())
            continue;
        if (original.name.rfind("__", 0) != 0)
            throw std::runtime_error("Enable Animate across slides for " + original.name +
                                     " before saving an updater");
        auto spec = parseObjectUpdater(found->second.at("input"), source, original,
                                       found->second.value("units", 100.));
        auto plan = prepareObjectUpdater(source, original, spec, duration);
        auto draw = plan.source;
        for (size_t k = 0; k < draw.to.size(); ++k) {
            auto point = plan.evaluate({draw.from[k].x, draw.from[k].y}, 1.);
            draw.to[k] = {point.x, point.y};
        }
        bool identical = true;
        for (size_t k = 0; k < draw.from.size(); ++k)
            identical &= cv::norm(cv::Point2f(draw.from[k].x, draw.from[k].y) -
                                  cv::Point2f(draw.to[k].x, draw.to[k].y)) < 1e-4;
        auto &o = target.objects[i];
        if (!identical) {
            draw.t = 1;
            auto rgba = gpu.warp(draw, source.image.cols, source.image.rows);
            cv::Mat alpha;
            cv::extractChannel(rgba, alpha, 3);
            auto bounds = cv::boundingRect(alpha);
            if (bounds.empty())
                throw std::runtime_error("Updater endpoint for " + original.name +
                                         " is empty or entirely outside the slide");
            o.mask = alpha;
            o.artwork = rgba(bounds).clone();
            o.artworkOrigin = bounds.tl();
            o.sourceAsset = o.artwork;
            o.drawing = {{"kind", "image"},   {"x", bounds.x},      {"y", bounds.y},
                         {"w", bounds.width}, {"h", bounds.height}, {"angle", 0},
                         {"opacity", 1}};
        }
        o.generation = {{"kind", "updater"},
                        {"source_id", original.id},
                        {"source_signature", objectTransformSignature(source, original)},
                        {"operation", spec},
                        {"duration", duration}};
        o.generation["result_signature"] = objectTransformSignature(target, o);
        o.updater = json{};
        if (original.artwork.empty() && !identical) {
            if (target.baseImage.empty())
                target.baseImage = source.image.clone();
            cv::Mat erase;
            cv::dilate(original.mask, erase, cv::Mat());
            target.baseImage.setTo(
                cv::Scalar(source.background[0], source.background[1], source.background[2]),
                erase);
        }
    }
    recomposeSlide(target);
    return target;
}
std::string objectTransformSignature(const Slide &s, const Object &o) {
    uint64_t hash = 1469598103934665603ULL;
    auto add = [&](uchar byte) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    };
    for (int y = 0; y < o.mask.rows; ++y)
        for (int x = 0; x < o.mask.cols; ++x)
            if (uchar m = o.mask.at<uchar>(y, x)) {
                uint32_t index = y * o.mask.cols + x;
                for (int k = 0; k < 4; ++k)
                    add((index >> (8 * k)) & 255);
                add(m);
                if (o.artwork.empty()) {
                    auto p = s.image.at<cv::Vec3b>(y, x);
                    for (int c = 0; c < 3; ++c)
                        add(p[c]);
                }
            }
    if (!o.artwork.empty())
        for (size_t i = 0; i < o.artwork.total() * o.artwork.elemSize(); ++i)
            add(o.artwork.data[i]);
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}
Slide generateKeyframe(const Slide &source, const std::map<std::string, json> &operations) {
    if (operations.empty())
        throw std::runtime_error("Stage a transformation for at least one object");
    Slide target = duplicateSlide(source);
    target.title = source.title + " transformed";
    if (target.baseImage.empty())
        target.baseImage = source.image.clone();
    CudaBackend gpu;
    for (auto &entry : operations)
        if (std::none_of(source.objects.begin(), source.objects.end(),
                         [&](const Object &o) { return o.id == entry.first; }))
            throw std::runtime_error("Staged transformation refers to a missing object");
    for (size_t i = 0; i < source.objects.size(); ++i) {
        auto found = operations.find(source.objects[i].id);
        if (found == operations.end())
            continue;
        auto &original = source.objects[i];
        auto draw = prepareObjectTransform(source, original, found->second);
        auto rgba = gpu.warp(draw, source.image.cols, source.image.rows);
        cv::Mat alpha;
        cv::extractChannel(rgba, alpha, 3);
        auto bounds = cv::boundingRect(alpha);
        if (bounds.empty())
            throw std::runtime_error("Transformation makes " + original.name +
                                     " empty or entirely outside the frame");
        auto &o = target.objects[i];
        o.drawing = json{};
        o.mask = alpha;
        o.artwork = rgba(bounds).clone();
        o.artworkOrigin = bounds.tl();
        o.sourceAsset = o.artwork.clone();
        o.drawing = {{"kind", "image"},    {"x", bounds.x}, {"y", bounds.y}, {"w", bounds.width},
                     {"h", bounds.height}, {"opacity", 1},  {"angle", 0}};
        o.generation = {{"source_id", original.id}, {"operation", found->second}};
        o.generation["source_signature"] = objectTransformSignature(source, original);
        if (original.artwork.empty()) {
            cv::Mat erase;
            cv::dilate(original.mask, erase, cv::Mat());
            target.baseImage.setTo(
                cv::Scalar(source.background[0], source.background[1], source.background[2]),
                erase);
        }
    }
    recomposeSlide(target);
    return target;
}
} // namespace fm
