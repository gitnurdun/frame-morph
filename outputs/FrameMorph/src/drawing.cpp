#include "engine.hpp"
#include <ft2build.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include FT_FREETYPE_H
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
namespace fm {
namespace {
using Function = std::function<double(double, double, double)>;
thread_local std::array<double, 6> pixelValues{};
class Expression {
    std::string text_;
    std::map<std::string, double> constants_;
    bool pixels_ = false;
    size_t pos_ = 0;
    int depth_ = 0;
    void space() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])))
            ++pos_;
    }
    bool take(char c) {
        space();
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }
    Function sum() {
        auto left = product();
        for (;;) {
            if (take('+')) {
                auto right = product();
                left = [left, right](double x, double y, double t) {
                    return left(x, y, t) + right(x, y, t);
                };
            } else if (take('-')) {
                auto right = product();
                left = [left, right](double x, double y, double t) {
                    return left(x, y, t) - right(x, y, t);
                };
            } else
                return left;
        }
    }
    Function product() {
        auto left = unary();
        for (;;) {
            if (take('*')) {
                auto right = unary();
                left = [left, right](double x, double y, double t) {
                    return left(x, y, t) * right(x, y, t);
                };
            } else if (take('/')) {
                auto right = unary();
                left = [left, right](double x, double y, double t) {
                    return left(x, y, t) / right(x, y, t);
                };
            } else
                return left;
        }
    }
    Function unary() {
        if (++depth_ > 64)
            throw std::runtime_error("Expression is nested too deeply");
        struct Guard {
            int &n;
            ~Guard() {
                --n;
            }
        } guard{depth_};
        if (take('+'))
            return unary();
        if (take('-')) {
            auto f = unary();
            return [f](double x, double y, double t) { return -f(x, y, t); };
        }
        auto left = primary();
        if (take('^')) {
            auto right = unary();
            return [left, right](double x, double y, double t) {
                return std::pow(left(x, y, t), right(x, y, t));
            };
        }
        return left;
    }
    Function primary() {
        if (++depth_ > 64)
            throw std::runtime_error("Plot expression is nested too deeply");
        struct Guard {
            int &n;
            ~Guard() {
                --n;
            }
        } guard{depth_};
        if (take('(')) {
            auto f = sum();
            if (!take(')'))
                throw std::runtime_error("Missing closing parenthesis");
            return f;
        }
        space();
        if (pos_ >= text_.size())
            throw std::runtime_error("Incomplete plot expression");
        if (std::isdigit(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '.') {
            const char *start = text_.c_str() + pos_;
            char *end = nullptr;
            double v = std::strtod(start, &end);
            if (end == start || !std::isfinite(v))
                throw std::runtime_error("Invalid number in plot expression");
            pos_ += end - start;
            return [v](double, double, double) { return v; };
        }
        if (std::isalpha(static_cast<unsigned char>(text_[pos_]))) {
            size_t start = pos_;
            while (pos_ < text_.size() && (std::isalnum(static_cast<unsigned char>(text_[pos_])) ||
                                           text_[pos_] == '_' || text_[pos_] == '.'))
                ++pos_;
            auto name = text_.substr(start, pos_ - start);
            if (pixels_ && (name == "r" || name == "g" || name == "b" || name == "a")) {
                int index = name == "r" ? 0 : name == "g" ? 1 : name == "b" ? 2 : 3;
                return [index](double, double, double) { return pixelValues[index]; };
            }
            if (name == "min" || name == "max" || name == "pow" || name == "atan2" ||
                name == "clamp") {
                if (!take('('))
                    throw std::runtime_error("Function needs parentheses: " + name);
                auto a = sum();
                if (!take(','))
                    throw std::runtime_error("Missing comma in " + name);
                auto b = sum();
                Function c;
                if (name == "clamp") {
                    if (!take(','))
                        throw std::runtime_error("clamp needs value, lower, upper");
                    c = sum();
                }
                if (!take(')'))
                    throw std::runtime_error("Missing closing parenthesis in " + name);
                return [name, a, b, c](double x, double y, double t) {
                    double av = a(x, y, t), bv = b(x, y, t);
                    if (name == "min")
                        return std::min(av, bv);
                    if (name == "max")
                        return std::max(av, bv);
                    if (name == "pow")
                        return std::pow(av, bv);
                    if (name == "atan2")
                        return std::atan2(av, bv);
                    double cv = c(x, y, t);
                    if (bv > cv)
                        throw std::runtime_error("clamp lower bound exceeds upper bound");
                    return std::clamp(av, bv, cv);
                };
            }
            if (name == "x")
                return [](double x, double, double) { return x; };
            if (name == "t")
                return [](double, double, double t) { return t; };
            if (name == "seconds") {
                double duration = constants_.count("duration") ? constants_.at("duration") : 1.;
                return [duration](double, double, double t) { return t * duration; };
            }
            if (name == "y")
                return [](double, double y, double) { return y; };
            if (name == "pi")
                return [](double, double, double) { return 3.141592653589793; };
            if (name == "e")
                return [](double, double, double) { return 2.718281828459045; };
            if (auto found = constants_.find(name); found != constants_.end()) {
                double value = found->second;
                return [value](double, double, double) { return value; };
            }
            static const std::map<std::string, double (*)(double)> functions = {
                {"sin", std::sin},     {"cos", std::cos},   {"tan", std::tan},
                {"asin", std::asin},   {"acos", std::acos}, {"atan", std::atan},
                {"sqrt", std::sqrt},   {"abs", std::fabs},  {"exp", std::exp},
                {"log", std::log},     {"ln", std::log},    {"log10", std::log10},
                {"floor", std::floor}, {"ceil", std::ceil}, {"sinh", std::sinh},
                {"cosh", std::cosh},   {"tanh", std::tanh}};
            auto it = functions.find(name);
            if (it == functions.end())
                throw std::runtime_error("Unknown function or variable: " + name);
            if (!take('('))
                throw std::runtime_error("Functions need parentheses: " + name + "(x)");
            auto argument = sum();
            if (!take(')'))
                throw std::runtime_error("Missing function closing parenthesis");
            auto f = it->second;
            return [argument, f](double x, double y, double t) { return f(argument(x, y, t)); };
        }
        throw std::runtime_error("Unexpected character in plot expression");
    }

  public:
    explicit Expression(std::string text, const std::map<std::string, double> &constants = {},
                        bool pixels = false)
        : text_(std::move(text)), constants_(constants), pixels_(pixels) {
        if (text_.empty() || text_.size() > 4096)
            throw std::runtime_error("Plot expression must contain 1 to 4096 characters");
    }
    Function compile() {
        auto f = sum();
        space();
        if (pos_ != text_.size())
            throw std::runtime_error(
                "Unexpected plot expression suffix; use explicit multiplication, e.g. 2*x");
        return f;
    }
};
cv::Vec3b readColor(const std::string &hex) {
    if (hex.size() != 7 || hex[0] != '#')
        throw std::runtime_error("Use an RGB color such as #4ed5b8");
    for (size_t i = 1; i < hex.size(); ++i)
        if (!std::isxdigit(static_cast<unsigned char>(hex[i])))
            throw std::runtime_error("Invalid hex color");
    return {uchar(std::stoi(hex.substr(5, 2), nullptr, 16)),
            uchar(std::stoi(hex.substr(3, 2), nullptr, 16)),
            uchar(std::stoi(hex.substr(1, 2), nullptr, 16))};
}
void applyMask(cv::Mat &rgba, const cv::Mat &mask, cv::Vec3b color) {
    for (int y = 0; y < rgba.rows; ++y) {
        auto p = rgba.ptr<cv::Vec4b>(y);
        auto a = mask.ptr<uchar>(y);
        for (int x = 0; x < rgba.cols; ++x)
            if (a[x]) {
                float sa = a[x] / 255.f, da = p[x][3] / 255.f, alpha = sa + da * (1 - sa);
                for (int c = 0; c < 3; ++c)
                    p[x][c] =
                        cv::saturate_cast<uchar>((color[c] * sa + p[x][c] * da * (1 - sa)) / alpha);
                p[x][3] = cv::saturate_cast<uchar>(alpha * 255);
            }
    }
}
std::vector<uint32_t> codepoints(const std::string &text) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < text.size();) {
        uint32_t c = static_cast<uchar>(text[i++]);
        int n = 0;
        if (c >= 0xf0) {
            c &= 7;
            n = 3;
        } else if (c >= 0xe0) {
            c &= 15;
            n = 2;
        } else if (c >= 0xc0) {
            c &= 31;
            n = 1;
        }
        for (int k = 0; k < n; ++k) {
            if (i >= text.size() || (static_cast<uchar>(text[i]) & 0xc0) != 0x80)
                throw std::runtime_error("Invalid UTF-8 text");
            c = (c << 6) | (text[i++] & 63);
        }
        out.push_back(c);
    }
    return out;
}
void textMask(cv::Mat &mask, const std::string &value, int x, int y, int size,
              const std::string &font) {
    FT_Library lib = nullptr;
    FT_Face face = nullptr;
    if (FT_Init_FreeType(&lib))
        throw std::runtime_error("Cannot initialize FreeType");
    if (FT_New_Face(lib, font.c_str(), 0, &face)) {
        FT_Done_FreeType(lib);
        throw std::runtime_error("Cannot open font: " + font);
    }
    FT_Set_Pixel_Sizes(face, 0, size);
    int penX = x, baseline = y + size;
    FT_UInt previous = 0;
    try {
        for (auto c : codepoints(value)) {
            if (c == '\n') {
                penX = x;
                baseline += int(size * 1.3);
                previous = 0;
                continue;
            }
            FT_UInt index = FT_Get_Char_Index(face, c);
            if (previous && index && FT_HAS_KERNING(face)) {
                FT_Vector kerning;
                FT_Get_Kerning(face, previous, index, FT_KERNING_DEFAULT, &kerning);
                penX += kerning.x >> 6;
            }
            if (FT_Load_Glyph(face, index, FT_LOAD_RENDER))
                continue;
            auto glyph = face->glyph;
            auto &bitmap = glyph->bitmap;
            int left = penX + glyph->bitmap_left, top = baseline - glyph->bitmap_top;
            for (int yy = 0; yy < (int)bitmap.rows; ++yy) {
                int row = top + yy;
                if (row < 0 || row >= mask.rows)
                    continue;
                for (int xx = 0; xx < (int)bitmap.width; ++xx) {
                    int col = left + xx;
                    if (col >= 0 && col < mask.cols) {
                        uchar coverage = bitmap.buffer[yy * bitmap.pitch + xx];
                        mask.at<uchar>(row, col) = std::max(mask.at<uchar>(row, col), coverage);
                    }
                }
            }
            penX += glyph->advance.x >> 6;
            previous = index;
        }
    } catch (...) {
        FT_Done_Face(face);
        FT_Done_FreeType(lib);
        throw;
    }
    FT_Done_Face(face);
    FT_Done_FreeType(lib);
}
void implicitMask(cv::Mat &mask, const std::string &expression, cv::Rect box, double xmin,
                  double xmax, double ymin, double ymax, int stroke, bool filled) {
    std::string formula = expression;
    size_t equal = formula.find('=');
    if (equal != std::string::npos) {
        if (formula.find('=', equal + 1) != std::string::npos)
            throw std::runtime_error("Use one equality for an implicit curve");
        formula = "(" + formula.substr(0, equal) + ")-(" + formula.substr(equal + 1) + ")";
    }
    auto f = compileExpression2D(formula);
    int nx = std::clamp(box.width, 64, 1000), ny = std::clamp(box.height, 64, 1000);
    std::vector<double> grid((nx + 1) * (ny + 1));
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i)
            grid[j * (nx + 1) + i] =
                f(xmin + (xmax - xmin) * i / nx, ymax - (ymax - ymin) * j / ny);
    auto segment = [&](cv::Point2f a, cv::Point2f b) {
        cv::Point aa(cvRound(a.x * 256), cvRound(a.y * 256)),
            bb(cvRound(b.x * 256), cvRound(b.y * 256));
        cv::line(mask, aa, bb, 255, stroke, cv::LINE_AA, 8);
    };
    if (filled) {
        cv::Rect visible = box & cv::Rect(0, 0, mask.cols, mask.rows);
        for (int y = visible.y; y < visible.y + visible.height; ++y)
            for (int x = visible.x; x < visible.x + visible.width; ++x) {
                int coverage = 0;
                for (float sy : {.25f, .75f})
                    for (float sx : {.25f, .75f}) {
                        double v = f(xmin + (x + sx - box.x) / box.width * (xmax - xmin),
                                     ymax - (y + sy - box.y) / box.height * (ymax - ymin));
                        if (std::isfinite(v) && v <= 0)
                            ++coverage;
                    }
                mask.at<uchar>(y, x) = uchar(coverage * 255 / 4);
            }
    }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            double v[4] = {grid[j * (nx + 1) + i], grid[j * (nx + 1) + i + 1],
                           grid[(j + 1) * (nx + 1) + i + 1], grid[(j + 1) * (nx + 1) + i]};
            if (!std::all_of(v, v + 4, [](double x) { return std::isfinite(x); }))
                continue;
            cv::Point2f corners[4] = {{float(box.x + double(box.width) * i / nx),
                                       float(box.y + double(box.height) * j / ny)},
                                      {float(box.x + double(box.width) * (i + 1) / nx),
                                       float(box.y + double(box.height) * j / ny)},
                                      {float(box.x + double(box.width) * (i + 1) / nx),
                                       float(box.y + double(box.height) * (j + 1) / ny)},
                                      {float(box.x + double(box.width) * i / nx),
                                       float(box.y + double(box.height) * (j + 1) / ny)}};
            std::vector<cv::Point2f> intersections;
            for (int edge = 0; edge < 4; ++edge) {
                int next = (edge + 1) % 4;
                if ((v[edge] < 0) != (v[next] < 0)) {
                    double t = v[edge] / (v[edge] - v[next]);
                    intersections.push_back(corners[edge] * (1 - t) + corners[next] * t);
                }
            }
            if (intersections.size() == 2)
                segment(intersections[0], intersections[1]);
            else if (intersections.size() == 4) {
                double center =
                    f(xmin + (xmax - xmin) * (i + .5) / nx, ymax - (ymax - ymin) * (j + .5) / ny);
                if ((center < 0) == (v[0] < 0)) {
                    segment(intersections[0], intersections[1]);
                    segment(intersections[2], intersections[3]);
                } else {
                    segment(intersections[0], intersections[3]);
                    segment(intersections[1], intersections[2]);
                }
            }
        }
}
} // namespace
double evaluateExpression(const std::string &expression, double x) {
    return Expression(expression).compile()(x, 0, x);
}
std::function<double(double)> compileExpression(const std::string &expression) {
    auto f = Expression(expression).compile();
    return [f](double x) { return f(x, 0, x); };
}
std::function<double(double, double)>
compileExpression2D(const std::string &expression, const std::map<std::string, double> &constants) {
    auto f = Expression(expression, constants).compile();
    return [f](double x, double y) { return f(x, y, x); };
}
std::function<double(double, double, double)>
compileExpression3D(const std::string &expression, const std::map<std::string, double> &constants) {
    return Expression(expression, constants).compile();
}
Slide blankSlide(int width, int height, cv::Vec3b background) {
    Slide s;
    s.id = newId();
    s.title = "Blank slide";
    s.blank = true;
    s.background = background;
    s.baseImage =
        cv::Mat(height, width, CV_8UC3, cv::Scalar(background[0], background[1], background[2]));
    s.image = s.baseImage.clone();
    return s;
}
Slide duplicateSlide(const Slide &original) {
    Slide s = original;
    s.id = newId();
    s.title = original.title + " copy";
    s.image = s.image.clone();
    s.baseImage = s.baseImage.clone();
    for (auto &o : s.objects) {
        auto parent = o.id;
        o.id = newId();
        // Incoming deformation belongs to the old transition, not to this new pose.
        o.generation = {{"parent_id", parent}};
        o.updater = json{};
        o.mask = o.mask.clone();
        o.artwork = o.artwork.clone();
        o.sourceAsset = o.sourceAsset.clone();
        o.originalAsset = o.originalAsset.clone();
    }
    return s;
}
void recomposeSlide(Slide &s) {
    if (s.baseImage.empty())
        return;
    s.image = s.baseImage.clone();
    std::vector<const Object *> authored;
    for (auto &o : s.objects)
        if (!o.artwork.empty())
            authored.push_back(&o);
    std::stable_sort(authored.begin(), authored.end(),
                     [](auto *a, auto *b) { return a->layer < b->layer; });
    for (auto *o : authored) {
        cv::Rect bounds(o->artworkOrigin, o->artwork.size()),
            visible = bounds & cv::Rect(0, 0, s.image.cols, s.image.rows);
        for (int y = visible.y; y < visible.y + visible.height; ++y)
            for (int x = visible.x; x < visible.x + visible.width; ++x) {
                auto p = o->artwork.at<cv::Vec4b>(y - bounds.y, x - bounds.x);
                float a = p[3] / 255.f;
                auto &dst = s.image.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c)
                    dst[c] = cv::saturate_cast<uchar>(p[c] * a + dst[c] * (1 - a));
            }
    }
}
std::function<double(const std::array<double, 6> &)>
compilePixelExpression(const std::string &text) {
    auto f = Expression(text, {{"tau", 6.283185307179586}}, true).compile();
    return [f](const std::array<double, 6> &values) {
        pixelValues = values;
        return f(values[4], values[5], 0);
    };
}
void renderDrawing(Slide &s, Object &o, const json &spec) {
    std::string kind = spec.value("kind", "rectangle"), content = spec.value("content", ""),
                hex = spec.value("color", "#4ed5b8");
    auto color = readColor(hex);
    if (kind == "plot") {
        auto compact = content;
        compact.erase(std::remove_if(compact.begin(), compact.end(),
                                     [](unsigned char c) { return std::isspace(c); }),
                      compact.end());
        if (compact.rfind("y=", 0) == 0)
            content = compact.substr(2);
        else if (compact.find('=') != std::string::npos)
            kind = "implicit";
    }
    auto finite = [&](const char *key, double value) {
        double v = spec.value(key, value);
        if (!std::isfinite(v) || std::abs(v) > 1e6)
            throw std::runtime_error(std::string("Invalid drawing parameter: ") + key);
        return v;
    };
    int x = finite("x", 200), y = finite("y", 200), w = finite("w", 320), h = finite("h", 200),
        stroke = std::clamp(int(finite("stroke", 4)), 1, 100),
        fontSize = std::clamp(int(finite("font_size", 56)), 6, 600);
    bool filled = spec.value("filled", true);
    if (std::abs(w) > 16384 || std::abs(h) > 16384)
        throw std::runtime_error("Drawing dimensions must be at most 16384 pixels");
    if (kind != "image")
        o.sourceAsset.release();
    if (content.size() > (kind == "svg" ? 2000000 : kind == "code" ? 32000 : 16000))
        throw std::runtime_error("Drawing text/formula is too long");
    cv::Mat rgba(s.image.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0)),
        mask(s.image.size(), CV_8U, cv::Scalar(0));
    cv::Point a(x, y), b(x + w, y + h);
    cv::Rect box(std::min(x, x + w), std::min(y, y + h), std::max(1, std::abs(w)),
                 std::max(1, std::abs(h)));
    if (kind == "image") {
        cv::Mat source = o.sourceAsset;
        if (spec.contains("source")) {
            source = cv::imread(spec.at("source").get<std::string>(), cv::IMREAD_UNCHANGED);
            if (source.empty())
                throw std::runtime_error("Cannot read inserted image");
            if (source.depth() != CV_8U)
                source.convertTo(source, CV_8U, source.depth() == CV_16U ? 1. / 257 : 1.);
            if (source.channels() == 1)
                cv::cvtColor(source, source, cv::COLOR_GRAY2BGRA);
            else if (source.channels() == 3)
                cv::cvtColor(source, source, cv::COLOR_BGR2BGRA);
        }
        if (source.empty() || source.type() != CV_8UC4)
            throw std::runtime_error("Inserted image source is missing");
        o.sourceAsset = source.clone();
        cv::Mat patch;
        cv::resize(source, patch, box.size(), 0, 0,
                   box.area() < source.rows * source.cols ? cv::INTER_AREA : cv::INTER_CUBIC);
        cv::Rect visible = box & cv::Rect(0, 0, rgba.cols, rgba.rows);
        if (visible.area() > 0)
            patch(cv::Rect(visible.tl() - box.tl(), visible.size())).copyTo(rgba(visible));
    } else if (kind == "group") {
        // Child specs preserve editable construction when authored objects are grouped.
        if (!spec.contains("children"))
            throw std::runtime_error("Group has no children");
        double oldX = spec.value("original_x", double(x)),
               oldY = spec.value("original_y", double(y)),
               oldW = spec.value("original_w", double(w)),
               oldH = spec.value("original_h", double(h));
        double scaleX = w / std::max(1., oldW), scaleY = h / std::max(1., oldH);
        for (auto childSpec : spec["children"]) {
            childSpec["x"] = x + (childSpec.value("x", 0.) - oldX) * scaleX;
            childSpec["y"] = y + (childSpec.value("y", 0.) - oldY) * scaleY;
            childSpec["w"] = childSpec.value("w", 100.) * scaleX;
            childSpec["h"] = childSpec.value("h", 100.) * scaleY;
            childSpec["font_size"] =
                childSpec.value("font_size", 56.) * std::min(std::abs(scaleX), std::abs(scaleY));
            Object child;
            Slide temp = blankSlide(s.image.cols, s.image.rows);
            renderDrawing(temp, child, childSpec);
            for (int py = 0; py < child.artwork.rows; ++py)
                for (int px = 0; px < child.artwork.cols; ++px) {
                    int yy = py + child.artworkOrigin.y, xx = px + child.artworkOrigin.x;
                    if (xx < 0 || yy < 0 || xx >= rgba.cols || yy >= rgba.rows)
                        continue;
                    auto src = child.artwork.at<cv::Vec4b>(py, px);
                    auto &dst = rgba.at<cv::Vec4b>(yy, xx);
                    float a = src[3] / 255.f, da = dst[3] / 255.f, oa = a + da * (1 - a);
                    if (oa > 0)
                        for (int c = 0; c < 3; ++c)
                            dst[c] =
                                cv::saturate_cast<uchar>((src[c] * a + dst[c] * da * (1 - a)) / oa);
                    dst[3] = cv::saturate_cast<uchar>(oa * 255);
                }
        }
    } else if (kind == "rectangle")
        cv::rectangle(mask, box, 255, filled ? -1 : stroke, cv::LINE_AA);
    else if (kind == "ellipse" || kind == "circle") {
        auto center = cv::Point((a.x + b.x) / 2, (a.y + b.y) / 2);
        auto axes = cv::Size(std::max(1, box.width / 2), std::max(1, box.height / 2));
        if (kind == "circle")
            axes.height = axes.width;
        cv::ellipse(mask, center, axes, 0, 0, 360, 255, filled ? -1 : stroke, cv::LINE_AA);
    } else if (kind == "triangle") {
        std::vector<cv::Point> points{{box.x + box.width / 2, box.y},
                                      {box.x + box.width, box.y + box.height},
                                      {box.x, box.y + box.height}};
        if (filled)
            cv::fillConvexPoly(mask, points, 255, cv::LINE_AA);
        else
            cv::polylines(mask, points, true, 255, stroke, cv::LINE_AA);
    } else if (kind == "diamond" || kind == "star" || kind == "hexagon") {
        int n = kind == "diamond" ? 4 : kind == "star" ? 10 : 6;
        std::vector<cv::Point> points;
        for (int i = 0; i < n; ++i) {
            double angle = -3.141592653589793 / 2 + i * 6.283185307179586 / n;
            double radius = kind == "star" && i % 2 ? .43 : 1.;
            points.emplace_back(box.x + box.width * .5 + std::cos(angle) * box.width * .5 * radius,
                                box.y + box.height * .5 +
                                    std::sin(angle) * box.height * .5 * radius);
        }
        if (filled)
            cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{points}, 255, cv::LINE_AA);
        else
            cv::polylines(mask, points, true, 255, stroke, cv::LINE_AA);
    } else if (kind == "line")
        cv::line(mask, a, b, 255, stroke, cv::LINE_AA);
    else if (kind == "arrow")
        cv::arrowedLine(mask, a, b, 255, stroke, cv::LINE_AA, 0, .15);
    else if (kind == "text") {
        std::string font = spec.value("font", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
        textMask(mask, content, x, y, fontSize, font);
    } else if (kind == "equation" || kind == "code" || kind == "svg") {
        auto patch = renderMedia(spec);
        if (kind == "svg")
            cv::resize(patch, patch, box.size(), 0, 0, cv::INTER_AREA);
        cv::Rect dest(cv::Point(x, y), patch.size()),
            visible = dest & cv::Rect(0, 0, rgba.cols, rgba.rows);
        if (visible.area() > 0)
            patch(cv::Rect(visible.x - x, visible.y - y, visible.width, visible.height))
                .copyTo(rgba(visible));
    } else if (kind == "implicit") {
        double xmin = finite("x_min", -3), xmax = finite("x_max", 3), ymin = finite("y_min", -3),
               ymax = finite("y_max", 3);
        if (xmax <= xmin || ymax <= ymin)
            throw std::runtime_error("Implicit curve ranges must increase");
        implicitMask(mask, content, box, xmin, xmax, ymin, ymax, stroke, filled);
    } else if (kind == "plot") {
        auto f = compileExpression(content);
        double xmin = finite("x_min", -6.283185307), xmax = finite("x_max", 6.283185307),
               ymin = finite("y_min", -1.5), ymax = finite("y_max", 1.5);
        if (xmax <= xmin || ymax <= ymin || box.width < 2 || box.height < 2)
            throw std::runtime_error(
                "Plot bounds must have increasing x/y ranges and nonzero dimensions");
        if (spec.value("axes", true)) {
            cv::Mat axes(mask.size(), CV_8U, cv::Scalar(0));
            if (ymin <= 0 && ymax >= 0) {
                int py = box.y + box.height * ymax / (ymax - ymin);
                cv::line(axes, {box.x, py}, {box.x + box.width, py}, 180, 2, cv::LINE_AA);
            }
            if (xmin <= 0 && xmax >= 0) {
                int px = box.x + box.width * (-xmin) / (xmax - xmin);
                cv::line(axes, {px, box.y}, {px, box.y + box.height}, 180, 2, cv::LINE_AA);
            }
            applyMask(rgba, axes, {120, 110, 95});
        }
        bool previousValid = false;
        cv::Point previous;
        double previousY = 0;
        int samples = std::clamp(box.width * 2, 200, 12000);
        for (int i = 0; i <= samples; ++i) {
            double xv = xmin + (xmax - xmin) * i / samples, yv = f(xv),
                   py = box.y + box.height * (ymax - yv) / (ymax - ymin);
            if (!std::isfinite(py) || std::abs(py) > 1e7) {
                previousValid = false;
                continue;
            }
            cv::Point point(box.x + box.width * i / samples, int(py));
            if (previousValid && std::abs(py - previousY) < box.height * 1.5) {
                auto aa = previous, bb = point;
                if (cv::clipLine(box, aa, bb))
                    cv::line(mask, aa, bb, 255, stroke, cv::LINE_AA);
            }
            previous = point;
            previousY = py;
            previousValid = true;
        }
    } else
        throw std::runtime_error("Unknown drawing kind: " + kind);
    applyMask(rgba, mask, color);
    double angle = finite("angle", 0), opacity = finite("opacity", 1);
    if (opacity < 0 || opacity > 1)
        throw std::runtime_error("Opacity must be between 0 and 1");
    if (std::abs(angle) > .0001) {
        auto matrix = cv::getRotationMatrix2D(cv::Point2f(x + w * .5f, y + h * .5f), -angle, 1);
        cv::Mat premult, rotated;
        rgba.convertTo(premult, CV_32FC4, 1. / 255);
        for (int yy = 0; yy < premult.rows; ++yy)
            for (int xx = 0; xx < premult.cols; ++xx) {
                auto &p = premult.at<cv::Vec4f>(yy, xx);
                for (int c = 0; c < 3; ++c)
                    p[c] *= p[3];
            }
        cv::warpAffine(premult, rotated, matrix, rgba.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT);
        for (int yy = 0; yy < rotated.rows; ++yy)
            for (int xx = 0; xx < rotated.cols; ++xx) {
                auto &p = rotated.at<cv::Vec4f>(yy, xx);
                if (p[3] > 1e-6)
                    for (int c = 0; c < 3; ++c)
                        p[c] /= p[3];
            }
        rotated.convertTo(rgba, CV_8UC4, 255);
    }
    if (opacity < 1)
        for (int yy = 0; yy < rgba.rows; ++yy)
            for (int xx = 0; xx < rgba.cols; ++xx)
                rgba.at<cv::Vec4b>(yy, xx)[3] =
                    cv::saturate_cast<uchar>(rgba.at<cv::Vec4b>(yy, xx)[3] * opacity);
    cv::Mat alpha;
    cv::extractChannel(rgba, alpha, 3);
    cv::Rect bounds = cv::boundingRect(alpha);
    if (bounds.area() == 0)
        throw std::runtime_error("Drawing is empty, invisible, or entirely outside the slide");
    if (s.baseImage.empty())
        s.baseImage = s.image.clone();
    o.drawing = spec;
    o.drawing["kind"] = kind;
    o.drawing.erase("source");
    o.artworkOrigin = bounds.tl();
    o.artwork = rgba(bounds).clone();
    o.mask = alpha;
    recomposeSlide(s);
}
} // namespace fm
