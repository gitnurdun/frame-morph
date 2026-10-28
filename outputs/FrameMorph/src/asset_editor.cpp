#include "asset_editor.hpp"
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Help_View.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Multiline_Input.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_RGB_Image.H>
#include <FL/fl_ask.H>
#include <FL/fl_draw.H>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
namespace fm {
namespace {
std::chrono::steady_clock::time_point nativeDialogReady{};
void waitForNativeDialog() {
    while (std::chrono::steady_clock::now() < nativeDialogReady)
        Fl::wait(.02);
}
// Restore an initially iconified WSLg window without destroying its X11 identity.
// Recreating the native window here can repeat the stale minimized placement.
void restoreInitialPlacement(Fl_Window &window) {
    if (window.x() < -30000 || window.y() < -30000) {
        window.position(std::max(0, (Fl::w() - window.w()) / 2),
                        std::max(0, (Fl::h() - window.h()) / 2));
        window.show();
        window.redraw();
        Fl::flush();
    }
}
class ArtworkPreview : public Fl_Box {
    cv::Mat source_, display_;
    double zoom_ = 0;
    std::unique_ptr<Fl_RGB_Image> picture_;
    void rebuild() {
        if (source_.empty())
            return;
        double scale =
            zoom_ > 0 ? zoom_
                      : std::min(double(w() - 20) / source_.cols, double(h() - 20) / source_.rows);
        double left = (w() - source_.cols * scale) / 2, top = (h() - source_.rows * scale) / 2;
        picture_.reset();
        display_ = cv::Mat(h(), w(), CV_8UC3);
        // Sample only the viewport. A 16 MP image never allocates a huge zoomed preview.
        cv::parallel_for_(cv::Range(0, h()), [&](const cv::Range &rows) {
            for (int yy = rows.start; yy < rows.end; ++yy)
                for (int xx = 0; xx < w(); ++xx) {
                    double sx = (xx + .5 - left) / scale - .5, sy = (yy + .5 - top) / scale - .5;
                    int ix = int(std::floor(sx)), iy = int(std::floor(sy));
                    float fx = sx - ix, fy = sy - iy, alpha = 0;
                    cv::Vec3f colour{};
                    for (int j = 0; j < 2; ++j)
                        for (int i = 0; i < 2; ++i) {
                            int x = ix + i, y = iy + j;
                            if (x < 0 || y < 0 || x >= source_.cols || y >= source_.rows)
                                continue;
                            auto p = source_.at<cv::Vec4b>(y, x);
                            float weight = (i ? fx : 1 - fx) * (j ? fy : 1 - fy) * p[3] / 255.f;
                            alpha += weight;
                            for (int c = 0; c < 3; ++c)
                                colour[c] += p[c] * weight;
                        }
                    bool inside = xx >= left && yy >= top && xx < left + source_.cols * scale &&
                                  yy < top + source_.rows * scale;
                    int bg = inside ? ((xx / 12 + yy / 12) % 2 ? 238 : 255) : 250;
                    auto &q = display_.at<cv::Vec3b>(yy, xx);
                    for (int c = 0; c < 3; ++c)
                        q[2 - c] = cv::saturate_cast<uchar>(colour[c] + bg * (1 - alpha));
                }
        });
        picture_ = std::make_unique<Fl_RGB_Image>(display_.data, display_.cols, display_.rows, 3);
        redraw();
    }
    void draw() override {
        fl_push_clip(x(), y(), w(), h());
        fl_color(250, 251, 254);
        fl_rectf(x(), y(), w(), h());
        if (picture_)
            picture_->draw(x(), y());
        else {
            fl_color(100, 116, 139);
            fl_draw("Press Preview to check your changes", x(), y(), w(), h(), FL_ALIGN_CENTER);
        }
        fl_pop_clip();
    }

  public:
    ArtworkPreview(int x, int y, int w, int h) : Fl_Box(x, y, w, h) {}
    void set(const cv::Mat &image) {
        source_ = image;
        rebuild();
    }
    void zoom(double value) {
        zoom_ = value;
        rebuild();
    }
};
struct RenderJob {
    std::mutex mutex;
    bool done = false;
    std::atomic_bool cancelled{false};
    cv::Mat image;
    std::string error;
    json spec;
};
struct FormulaTemplate {
    const char *name, *text;
};
const std::vector<FormulaTemplate> formulas{
    {"Quadratic formula", R"(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a})"},
    {"Euler identity", R"(e^{i\pi}+1=0)"},
    {"Integral", R"(\int_{-\infty}^{\infty}e^{-x^2}\,dx=\sqrt{\pi})"},
    {"Matrix", R"(A=\begin{bmatrix}a&b\\c&d\end{bmatrix})"},
    {"Matrix transform",
     R"(\begin{pmatrix}x'\\y'\end{pmatrix}=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}\begin{pmatrix}x\\y\end{pmatrix})"},
    {"Aligned derivation",
     R"(\begin{aligned}(a+b)^2&=a^2+2ab+b^2\\(a-b)^2&=a^2-2ab+b^2\end{aligned})"},
    {"Piecewise cases", R"(f(x)=\begin{cases}x^2&x\geq0\\-x&x<0\end{cases})"},
    {"Probability", R"(P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)})"},
    {"Vector calculus", R"(\nabla\cdot\mathbf{E}=\frac{\rho}{\varepsilon_0})"},
    {"Limit and sum",
     R"(e^x=\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^n=\sum_{k=0}^{\infty}\frac{x^k}{k!})"}};
class AssetDialog {
    std::string kind_;
    json initial_;
    cv::Mat source_;
    Fl_Double_Window window_{1120, 780};
    Fl_Multiline_Input *content_ = nullptr;
    Fl_Choice *backend_ = nullptr, *templates_ = nullptr, *filter_ = nullptr;
    Fl_Input *size_ = nullptr, *color_ = nullptr, *width_ = nullptr, *height_ = nullptr,
             *brightness_ = nullptr, *contrast_ = nullptr, *saturation_ = nullptr, *blur_ = nullptr;
    Fl_Input *cropX_ = nullptr, *cropY_ = nullptr, *cropW_ = nullptr, *cropH_ = nullptr;
    Fl_Check_Button *numbers_ = nullptr, *flipX_ = nullptr, *flipY_ = nullptr, *aspect_ = nullptr;
    Fl_Input *tint_ = nullptr;
    ArtworkPreview *preview_ = nullptr;
    Fl_Box *status_ = nullptr;
    Fl_Button *apply_ = nullptr;
    std::vector<std::unique_ptr<std::function<void()>>> handlers_;
    std::shared_ptr<RenderJob> job_;
    std::thread worker_;
    json desired_, rendered_;
    cv::Mat pixels_;
    bool busy_ = false, accepted_ = false;
    template <class W> W *watch(W *w) {
        w->when(FL_WHEN_CHANGED);
        bind(w, [this] {
            apply_->deactivate();
            status_->copy_label("Changed. Preview before applying.");
        });
        return w;
    }
    void bind(Fl_Widget *w, std::function<void()> fn) {
        auto h = std::make_unique<std::function<void()>>(std::move(fn));
        w->callback(
            [](Fl_Widget *, void *p) {
                try {
                    (*static_cast<std::function<void()> *>(p))();
                } catch (const std::exception &e) {
                    fl_alert("%s", e.what());
                }
            },
            h.get());
        handlers_.push_back(std::move(h));
    }
    Fl_Input *field(int x, int y, int w, const char *name, const std::string &value) {
        auto f = new Fl_Input(x, y, w, 28, name);
        f->align(FL_ALIGN_TOP_LEFT);
        f->textsize(13);
        f->value(value.c_str());
        return watch(f);
    }
    double value(Fl_Input *f) {
        return evaluateExpression(f->value(), 0);
    }
    json settings() {
        json s = initial_;
        s["kind"] = kind_;
        s["content"] = content_->value();
        if (kind_ == "image") {
            s["pixel_map"] = content_->value();
            s["width"] = value(width_);
            s["height"] = value(height_);
            s["tint"] = tint_->value();
            s["lock_aspect"] = bool(aspect_->value());
            if (aspect_->value()) {
                s["height"] =
                    std::round(value(width_) * value(cropH_) / std::max(1., value(cropW_)));
                height_->value(std::to_string(s["height"].get<int>()).c_str());
            }
            s["crop_x"] = value(cropX_);
            s["crop_y"] = value(cropY_);
            s["crop_w"] = value(cropW_);
            s["crop_h"] = value(cropH_);
            s["brightness"] = value(brightness_);
            s["contrast"] = value(contrast_);
            s["saturation"] = value(saturation_);
            s["blur"] = value(blur_);
            static const char *filters[]{"none",    "grayscale", "sepia",    "invert",
                                         "sharpen", "edges",     "threshold"};
            s["filter"] = filters[filter_->value()];
            s["flip_x"] = bool(flipX_->value());
            s["flip_y"] = bool(flipY_->value());
        } else if (kind_ == "svg")
            s["svg_size"] = value(size_);
        else {
            s["font_size"] = value(size_);
            s["color"] = color_->value();
            if (kind_ == "equation") {
                static const char *backends[]{"auto", "mathtext", "latex"};
                s["backend"] = backends[backend_->value()];
            } else {
                static const char *languages[]{"python", "cpp", "javascript", "plain"};
                s["language"] = languages[backend_->value()];
                s["theme"] = filter_->value() ? "light" : "dark";
                s["line_numbers"] = bool(numbers_->value());
            }
        }
        return s;
    }
    void start() {
        if (busy_)
            return;
        busy_ = true;
        job_ = std::make_shared<RenderJob>();
        job_->spec = desired_;
        auto job = job_;
        auto source = source_;
        apply_->deactivate();
        status_->copy_label("Rendering preview... You can keep typing or close this window.");
        if (worker_.joinable())
            worker_.join();
        worker_ = std::thread([job, source] {
            cv::Mat image;
            std::string error;
            try {
                auto cancelled = [job] { return job->cancelled.load(); };
                image = job->spec.value("kind", "") == "image"
                            ? processImage(source, job->spec, cancelled)
                            : renderMedia(job->spec, cancelled);
            } catch (const std::exception &e) {
                error = e.what();
            }
            std::lock_guard<std::mutex> lock(job->mutex);
            job->image = std::move(image);
            job->error = std::move(error);
            job->done = true;
        });
    }
    void request() {
        desired_ = settings();
        if (!busy_)
            start();
        else {
            job_->cancelled = true;
            status_->copy_label("Latest preview queued; earlier results will be discarded.");
        }
    }
    void poll() {
        if (!busy_)
            return;
        bool done;
        cv::Mat image;
        std::string error;
        json spec;
        {
            std::lock_guard<std::mutex> lock(job_->mutex);
            done = job_->done;
            if (done) {
                image = job_->image;
                error = job_->error;
                spec = job_->spec;
            }
        }
        if (!done)
            return;
        busy_ = false;
        if (spec != desired_) {
            start();
            return;
        }
        if (!error.empty()) {
            status_->copy_label(error.c_str());
            return;
        }
        rendered_ = spec;
        pixels_ = image;
        preview_->set(image);
        if (settings() == rendered_) {
            apply_->activate();
            status_->copy_label((std::to_string(image.cols) + " x " + std::to_string(image.rows) +
                                 " pixels. Ready to apply.")
                                    .c_str());
        } else
            status_->copy_label(
                "Input changed during rendering. Press Preview for the latest version.");
    }

  public:
    AssetEditResult exercise(const fs::path &folder, std::ostream &log) {
        auto require = [&](bool condition, const std::string &message) {
            if (!condition)
                throw std::runtime_error(message);
            log << "PASS " << message << '\n';
        };
        auto wait = [&](bool success) {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(40);
            while (busy_ && std::chrono::steady_clock::now() < deadline) {
                Fl::wait(.01);
                restoreInitialPlacement(window_);
                poll();
            }
            require(!busy_, kind_ + " preview completes without blocking the event loop");
            if (success)
                require(apply_->active(),
                        kind_ + " preview enables Apply only after matching artwork is ready");
        };
        waitForNativeDialog();
        window_.position(std::max(0, (Fl::w() - window_.w()) / 2),
                         std::max(0, (Fl::h() - window_.h()) / 2));
        window_.show();
        request();
        // Queue a second request while the first is in flight, then require the latest result.
        if (kind_ == "equation") {
            content_->value(R"(A=\begin{bmatrix}1&2\\3&4\end{bmatrix})");
            backend_->value(2);
            request();
        } else if (kind_ == "image") {
            content_->value("(r,g,b,a):g,b,r,a");
            width_->value("80");
            height_->value("60");
            request();
        } else if (kind_ == "code") {
            content_->value("int square(int x) {\n    return x * x;\n}");
            backend_->value(1);
            request();
        } else {
            content_->value(
                R"(<svg xmlns="http://www.w3.org/2000/svg" width="240" height="160"><rect width="240" height="160" rx="20" fill="#386fe8"/><circle cx="120" cy="80" r="40" fill="#22aa91"/></svg>)");
            request();
        }
        wait(true);
        require(rendered_ == settings(),
                kind_ + " discards stale previews and commits the latest input");
        if (kind_ == "equation") {
            auto valid = std::string(content_->value());
            content_->value(R"(\frac{)");
            backend_->value(1);
            content_->do_callback();
            request();
            wait(false);
            require(!apply_->active() && std::string(content_->value()) == R"(\frac{)",
                    "Invalid formula preview keeps input and prevents a partial canvas edit");
            content_->value(valid.c_str());
            backend_->value(2);
            request();
            wait(true);
        }
        if (kind_ == "image")
            require(pixels_.size() == cv::Size(80, 60),
                    "Image studio previews the requested output size");
        Fl::wait(.15); // Let the newly mapped WSLg window finish presenting before capture.
        restoreInitialPlacement(window_);
        unsigned char *data = nullptr;
        for (int attempt = 0; attempt < 20 && !data; ++attempt) {
            restoreInitialPlacement(window_);
            window_.redraw();
            Fl::flush();
            window_.make_current();
            data = fl_read_image(nullptr, 0, 0, window_.w(), window_.h());
            if (!data)
                Fl::wait(.1);
        }
        log << "WINDOW " << kind_ << " shown=" << window_.shown()
            << " visible=" << window_.visible() << " at " << window_.x() << "," << window_.y()
            << " size=" << window_.w() << "," << window_.h() << " captured=" << (data != nullptr)
            << '\n';
        require(window_.shown() && data != nullptr,
                kind_ + " studio presents its native preview window");
        if (data) {
            cv::Mat rgb(window_.h(), window_.w(), CV_8UC3, data), bgr;
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            cv::imwrite((folder / ("studio-" + kind_ + ".png")).string(), bgr);
            delete[] data;
        }
        apply_->do_callback();
        require(accepted_ && !window_.shown(),
                kind_ + " Apply returns portable artwork and closes the preview editor");
        return {rendered_, pixels_.clone()};
    }
    ~AssetDialog() {
        if (job_)
            job_->cancelled = true;
        if (worker_.joinable())
            worker_.join();
        // WSLg can recycle a just-destroyed native popup while its close animation is pending.
        nativeDialogReady = std::chrono::steady_clock::now() + std::chrono::milliseconds(450);
    }
    AssetDialog(std::string kind, json initial, cv::Mat source)
        : kind_(std::move(kind)), initial_(std::move(initial)), source_(std::move(source)) {
        window_.copy_label(kind_ == "image"  ? "Image studio"
                           : kind_ == "code" ? "Source code block"
                           : kind_ == "svg"  ? "SVG artwork"
                                             : "LaTeX formula studio");
        window_.color(fl_rgb_color(238, 242, 247));
        window_.begin();
        auto title = new Fl_Box(25, 16, 1050, 30, window_.label());
        title->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        title->labelsize(21);
        content_ = new Fl_Multiline_Input(25, 105, 500, 310,
                                          kind_ == "image" ? "Pixel formula (optional)"
                                          : kind_ == "svg" ? "SVG source"
                                                           : "Content");
        content_->align(FL_ALIGN_TOP_LEFT);
        content_->textfont(FL_COURIER);
        content_->textsize(14);
        content_->wrap(kind_ != "code");
        if (kind_ == "equation")
            content_->resize(25, 120, 500, 295);
        content_->value(initial_.value(kind_ == "image" ? "pixel_map" : "content", "").c_str());
        watch(content_);
        preview_ = new ArtworkPreview(550, 105, 545, 510);
        auto *zoom = new Fl_Choice(920, 65, 175, 28);
        zoom->add("Fit preview|Actual pixels|200% zoom");
        zoom->value(0);
        bind(zoom, [this, zoom] {
            preview_->zoom(zoom->value() == 0 ? 0 : zoom->value() == 1 ? 1 : 2);
        });
        if (kind_ != "image") {
            auto *load = new Fl_Button(550, 65, 165, 28, "Load source...");
            bind(load, [this] {
                Fl_Native_File_Chooser chooser;
                chooser.title("Load source text");
                chooser.type(Fl_Native_File_Chooser::BROWSE_FILE);
                chooser.filter("Text sources\t*.{tex,txt,py,cpp,h,hpp,c,js,ts,svg}");
                if (chooser.show() != 0)
                    return;
                auto path = fs::path(chooser.filename());
                size_t limit = kind_ == "svg" ? 2000000 : kind_ == "code" ? 32000 : 16000;
                if (fs::file_size(path) > limit)
                    throw std::runtime_error("Source file is too large for this editor");
                std::ifstream f(path);
                std::string text((std::istreambuf_iterator<char>(f)), {});
                content_->value(text.c_str());
                apply_->deactivate();
                status_->copy_label("Source loaded. Preview before applying.");
            });
            auto *save = new Fl_Button(735, 65, 165, 28, "Save source...");
            bind(save, [this] {
                Fl_Native_File_Chooser chooser;
                chooser.title("Save source text");
                chooser.type(Fl_Native_File_Chooser::BROWSE_SAVE_FILE);
                chooser.options(Fl_Native_File_Chooser::SAVEAS_CONFIRM);
                chooser.preset_file(kind_ == "svg"        ? "artwork.svg"
                                    : kind_ == "equation" ? "formula.tex"
                                                          : "snippet.txt");
                if (chooser.show() != 0)
                    return;
                std::ofstream f(chooser.filename());
                if (!f)
                    throw std::runtime_error("Cannot save source file");
                f << content_->value();
                if (!f)
                    throw std::runtime_error("Could not finish writing source file");
                status_->copy_label("Source file saved.");
            });
        }
        if (kind_ == "image") {
            content_->resize(25, 105, 500, 76);
            content_->tooltip("(r,g,b,a):1-r,1-g,1-b,a; channels and x/y coordinates are 0..1. "
                              "Expressions read the pre-map image.");
            filter_ = new Fl_Choice(25, 224, 240, 28, "Filter");
            filter_->align(FL_ALIGN_TOP_LEFT);
            filter_->add("Original|Grayscale|Sepia|Invert|Sharpen|Edges|Threshold");
            std::vector<std::string> filters{"none",    "grayscale", "sepia",    "invert",
                                             "sharpen", "edges",     "threshold"};
            auto found =
                std::find(filters.begin(), filters.end(), initial_.value("filter", "none"));
            filter_->value(found == filters.end() ? 0 : found - filters.begin());
            watch(filter_);
            width_ = field(285, 224, 110, "Width",
                           std::to_string(initial_.value("width", source_.cols)));
            height_ = field(415, 224, 110, "Height",
                            std::to_string(initial_.value("height", source_.rows)));
            cropX_ = field(25, 292, 110, "Crop X", std::to_string(initial_.value("crop_x", 0)));
            cropY_ = field(155, 292, 110, "Crop Y", std::to_string(initial_.value("crop_y", 0)));
            cropW_ = field(285, 292, 110, "Crop width",
                           std::to_string(initial_.value("crop_w", source_.cols)));
            cropH_ = field(415, 292, 110, "Crop height",
                           std::to_string(initial_.value("crop_h", source_.rows)));
            brightness_ = field(25, 360, 240, "Brightness (-1..1)",
                                std::to_string(initial_.value("brightness", 0.)));
            contrast_ = field(285, 360, 240, "Contrast (0..4)",
                              std::to_string(initial_.value("contrast", 1.)));
            saturation_ = field(25, 428, 240, "Saturation (0..3)",
                                std::to_string(initial_.value("saturation", 1.)));
            blur_ = field(285, 428, 240, "Blur radius (0..30)",
                          std::to_string(initial_.value("blur", 0.)));
            aspect_ = new Fl_Check_Button(25, 184, 500, 22,
                                          "Lock output aspect to crop (width drives height)");
            aspect_->labelsize(12);
            aspect_->value(initial_.value("lock_aspect", false));
            if (aspect_->value())
                height_->deactivate();
            bind(aspect_, [this] {
                if (aspect_->value())
                    height_->deactivate();
                else
                    height_->activate();
                apply_->deactivate();
            });
            tint_ = field(25, 535, 240, "Tint (white preserves colours)",
                          initial_.value("tint", "#ffffff"));
            flipX_ = new Fl_Check_Button(25, 480, 220, 28, "Flip horizontally");
            flipY_ = new Fl_Check_Button(285, 480, 220, 28, "Flip vertically");
            flipX_->value(initial_.value("flip_x", false));
            flipY_->value(initial_.value("flip_y", false));
            watch(flipX_);
            watch(flipY_);
            auto reset = new Fl_Button(285, 535, 240, 30, "Reset to original");
            bind(reset, [this] {
                filter_->value(0);
                tint_->value("#ffffff");
                aspect_->value(0);
                height_->activate();
                content_->value("");
                brightness_->value("0");
                contrast_->value("1");
                saturation_->value("1");
                blur_->value("0");
                flipX_->value(0);
                flipY_->value(0);
                cropX_->value("0");
                cropY_->value("0");
                cropW_->value(std::to_string(source_.cols).c_str());
                cropH_->value(std::to_string(source_.rows).c_str());
                width_->value(std::to_string(source_.cols).c_str());
                height_->value(std::to_string(source_.rows).c_str());
                request();
            });
            auto invert = new Fl_Button(25, 65, 500, 28, "Insert an example pixel formula");
            bind(invert, [this] {
                content_->value("(r,g,b,a):1-r,1-g,1-b,a");
                apply_->deactivate();
            });
        } else if (kind_ != "svg") {
            size_ = field(25, 470, 125, "Font size",
                          std::to_string(initial_.value("font_size", kind_ == "code" ? 28 : 56)));
            color_ = field(170, 470, 150, "Color", initial_.value("color", "#1d293d"));
            if (kind_ == "code")
                color_->hide();
            backend_ = new Fl_Choice(340, 470, 185, 28, kind_ == "code" ? "Language" : "Renderer");
            backend_->align(FL_ALIGN_TOP_LEFT);
            if (kind_ == "equation") {
                backend_->add("Auto|Fast Mathtext|Full LaTeX");
                auto b = initial_.value("backend", "auto");
                backend_->value(b == "latex" ? 2 : b == "mathtext" ? 1 : 0);
                templates_ = new Fl_Choice(25, 65, 500, 28);
                templates_->add("Insert a formula template...");
                for (auto &item : formulas)
                    templates_->add(item.name);
                templates_->value(0);
                bind(templates_, [this] {
                    if (templates_->value() > 0) {
                        content_->value(formulas.at(templates_->value() - 1).text);
                        apply_->deactivate();
                    }
                });
            } else {
                backend_->add("Python|C++|JavaScript|Plain text");
                auto lang = initial_.value("language", "python");
                backend_->value(lang == "cpp"          ? 1
                                : lang == "javascript" ? 2
                                : lang == "plain"      ? 3
                                                       : 0);
                filter_ = new Fl_Choice(25, 535, 220, 28, "Theme");
                filter_->align(FL_ALIGN_TOP_LEFT);
                filter_->add("Dark|Light");
                filter_->value(initial_.value("theme", "dark") == "light");
                watch(filter_);
                numbers_ = new Fl_Check_Button(285, 535, 240, 28, "Show line numbers");
                numbers_->value(initial_.value("line_numbers", true));
                watch(numbers_);
            }
            watch(backend_);
        }
        if (kind_ == "svg")
            size_ = field(25, 470, 300, "Render longest edge (64..4000 px)",
                          std::to_string(initial_.value("svg_size", 1200)));
        auto notes = new Fl_Box(
            25, 590, 500, 55,
            kind_ == "image" ? "Crop uses original pixels. Resize sets canvas size.\nFilters are "
                               "saved against the original image; Undo restores edits."
            : kind_ == "equation" ? "Auto tries fast math, then Full LaTeX. Matrices, cases "
                                    "and\naligned equations use installed TinyTeX + PyMuPDF."
            : kind_ == "code" ? "Source code is displayed as artwork and is never executed.\nEdit "
                                "content here later; geometry stays editable on the canvas."
                              : "Embedded SVG paths and text become portable artwork.\nExternal "
                                "files, scripts and foreignObject are unsupported.");
        notes->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        notes->labelsize(12);
        status_ = new Fl_Box(25, 650, 1070, 60);
        status_->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP);
        status_->labelsize(12);
        auto preview = new Fl_Button(550, 725, 165, 32, "Preview");
        preview->shortcut(FL_CTRL | FL_Enter);
        preview->tooltip("Ctrl+Enter: render the latest input");
        bind(preview, [this] { request(); });
        apply_ = new Fl_Button(735, 725, 165, 32, "Apply to canvas");
        apply_->shortcut(FL_ALT | FL_Enter);
        apply_->tooltip("Alt+Enter: apply the verified preview");
        bind(apply_, [this] {
            if (settings() != rendered_) {
                request();
                return;
            }
            accepted_ = true;
            window_.hide();
        });
        apply_->deactivate();
        auto cancel = new Fl_Button(920, 725, 175, 32, "Cancel");
        bind(cancel, [this] { window_.hide(); });
        window_.end();
        window_.set_modal();
    }
    std::optional<AssetEditResult> run() {
        waitForNativeDialog();
        window_.position(std::max(0, (Fl::w() - window_.w()) / 2),
                         std::max(0, (Fl::h() - window_.h()) / 2));
        window_.show();
        request();
        int mappingPasses = 0;
        while (window_.shown()) {
            Fl::wait(.025);
            if (mappingPasses++ < 80)
                restoreInitialPlacement(window_);
            try {
                poll();
            } catch (const std::exception &e) {
                apply_->deactivate();
                status_->copy_label(e.what());
            }
        }
        if (accepted_)
            return AssetEditResult{rendered_, pixels_.clone()};
        return std::nullopt;
    }
};
} // namespace
std::optional<AssetEditResult> editAsset(const std::string &kind, const json &initial,
                                         const cv::Mat &source) {
    return AssetDialog(kind, initial, source).run();
}
void verifyAssetEditors(const fs::path &folder, std::ostream &log) {
    fs::create_directories(folder);
    AssetDialog("equation", {{"content", R"(e^{i\pi}+1=0)"}, {"font_size", 48}}, {})
        .exercise(folder, log);
    AssetDialog("code", {{"content", "def square(x):\n    return x*x"}, {"font_size", 26}}, {})
        .exercise(folder, log);
    cv::Mat image(120, 160, CV_8UC4);
    for (int y = 0; y < 120; ++y)
        for (int x = 0; x < 160; ++x)
            image.at<cv::Vec4b>(y, x) = {uchar(x), uchar(y * 2), uchar(255 - x), 255};
    AssetDialog("image", json::object(), image).exercise(folder, log);
    AssetDialog(
        "svg",
        {{"content",
          R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="80"><rect width="100" height="80" fill="red"/></svg>)"}},
        {})
        .exercise(folder, log);
    showAppHelp();
    if (!fs::exists(folder / "help-guide.png"))
        throw std::runtime_error("Help guide did not capture its contents");
    log << "PASS Built-in help opens the F1 guide with authoring, image, LaTeX and animation "
           "notes\n";
}
void showAppHelp() {
    Fl_Double_Window dialog(1020, 760, "FrameMorph guide");
    dialog.begin();
    Fl_Help_View view(15, 55, 990, 650);
    view.textsize(15);
    fs::path root = FRAMEMORPH_SOURCE_DIR;
    std::error_code ec;
    auto exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && fs::exists(exe.parent_path().parent_path() / "resources/help.html"))
        root = exe.parent_path().parent_path();
    std::ifstream f(root / "resources/help.html");
    std::string html((std::istreambuf_iterator<char>(f)), {});
    view.value(html.empty() ? "<h1>FrameMorph</h1><p>Select objects to edit Properties. Select a "
                              "frame for timing. Read README.md for more help.</p>"
                            : html.c_str());
    Fl_Choice topics(15, 15, 690, 28);
    topics.add("Getting started|Objects and frames|Animation effects|Updaters and maps|LaTeX "
               "formulas|Images and pixel formulas|Code and SVG|Shortcuts and troubleshooting");
    topics.value(0);
    topics.callback(
        [](Fl_Widget *w, void *p) {
            static const char *anchors[]{"start", "objects", "effects", "updaters",
                                         "latex", "images",  "code",    "keys"};
            static_cast<Fl_Help_View *>(p)->topline(anchors[static_cast<Fl_Choice *>(w)->value()]);
        },
        &view);
    Fl_Button close(835, 720, 170, 28, "Close");
    close.callback([](Fl_Widget *, void *p) { static_cast<Fl_Window *>(p)->hide(); }, &dialog);
    dialog.end();
    dialog.set_modal();
    dialog.show();
    struct HelpCapture {
        Fl_Window *window;
        fs::path file;
    };
    HelpCapture capture{&dialog, {}};
    auto captureHelp = [](void *data) {
        auto &c = *static_cast<HelpCapture *>(data);
        c.window->make_current();
        Fl::flush();
        unsigned char *pixels = fl_read_image(nullptr, 0, 0, c.window->w(), c.window->h());
        if (pixels) {
            cv::Mat rgb(c.window->h(), c.window->w(), CV_8UC3, pixels), bgr;
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            cv::imwrite(c.file.string(), bgr);
            delete[] pixels;
        }
        c.window->hide();
    };
    if (const char *folder = std::getenv("FRAMEMORPH_UI_TEST")) {
        capture.file = fs::path(folder) / "help-guide.png";
        Fl::add_timeout(.3, captureHelp, &capture);
    }
    while (dialog.shown())
        Fl::wait(.05);
    Fl::remove_timeout(captureHelp, &capture);
}
} // namespace fm
