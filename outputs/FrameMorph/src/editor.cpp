#include "asset_editor.hpp"
#include "engine.hpp"
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Float_Input.H>
#include <FL/Fl_Hold_Browser.H>
#include <FL/Fl_Image.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Menu_Button.H>
#include <FL/Fl_Multi_Browser.H>
#include <FL/Fl_Multiline_Input.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/fl_ask.H>
#include <FL/Fl_Color_Chooser.H>
#include <FL/fl_draw.H>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>

namespace fm {
namespace {
const Fl_Color bg = fl_rgb_color(238, 242, 247), panel = fl_rgb_color(255, 255, 255),
               surface = fl_rgb_color(245, 247, 251), line = fl_rgb_color(215, 222, 233),
               fg = fl_rgb_color(29, 41, 61), dim = fl_rgb_color(101, 116, 139),
               accent = fl_rgb_color(45, 103, 230);
std::string number(double v) {
    std::ostringstream s;
    s.precision(4);
    s << v;
    return s.str();
}
bool participating(const Object &o) {
    return o.name.rfind("__", 0) == 0;
}

class PreviewWorker {
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    bool stop_ = false, pending_ = false;
    std::shared_ptr<Project> project_;
    double time_ = 0;
    uint64_t serial_ = 0;
    std::atomic<uint64_t> latest_{0};
    cv::Mat result_;
    std::string error_, notice_;
    uint64_t delivered_ = 0;

  public:
    PreviewWorker() {
        thread_ = std::thread([this] { run(); });
    }
    ~PreviewWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++latest_;
        }
        cv_.notify_one();
        thread_.join();
    }
    void request(std::shared_ptr<Project> p, double time) {
        std::lock_guard<std::mutex> lock(mutex_);
        project_ = std::move(p);
        time_ = time;
        serial_ = ++latest_;
        pending_ = true;
        cv_.notify_one();
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++latest_;
        pending_ = false;
        delivered_ = 0;
        result_.release();
        error_.clear();
        notice_.clear();
    }
    bool poll(cv::Mat &image, std::string &error, std::string &notice) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (delivered_ && delivered_ != latest_.load()) {
            delivered_ = 0;
            result_.release();
        }
        if (!delivered_)
            return false;
        image = std::move(result_);
        error = std::move(error_);
        notice = std::move(notice_);
        delivered_ = 0;
        return true;
    }

  private:
    void run() {
        std::shared_ptr<Project> active;
        std::unique_ptr<Renderer> renderer;
        size_t warned = 0;
        for (;;) {
            std::shared_ptr<Project> p;
            double time;
            uint64_t serial;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || pending_; });
                if (stop_)
                    return;
                p = project_;
                time = time_;
                serial = serial_;
                pending_ = false;
            }
            try {
                if (p != active) {
                    renderer = std::make_unique<Renderer>(*p, 960, 540);
                    active = p;
                    warned = 0;
                }
                renderer->warmupAt(time);
                auto image = renderer->frame(time, [&] { return latest_.load() != serial; });
                if (!image.empty() && latest_.load() == serial) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    result_ = image;
                    error_.clear();
                    notice_.clear();
                    if (renderer->warnings().size() > warned) {
                        notice_ = renderer->warnings().back();
                        warned = renderer->warnings().size();
                    }
                    delivered_ = serial;
                    Fl::awake();
                }
            } catch (const std::exception &e) {
                if (latest_.load() == serial) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    error_ = e.what();
                    delivered_ = serial;
                    Fl::awake();
                }
            }
        }
    }
};

class Editor;
class Canvas : public Fl_Widget {
    Editor &e_;
    float zoom_ = 1, panX_ = 0, panY_ = 0;
    int dragX_ = 0, dragY_ = 0;
    bool panning_ = false, painting_ = false, drawingDrag_ = false;
    cv::Point previous_{-1, -1}, drawBegin_{}, drawEnd_{};

  public:
    cv::Mat preview;
    bool showingPreview = false;
    bool eyedropper = false;
    Canvas(int x, int y, int w, int h, Editor &e) : Fl_Widget(x, y, w, h), e_(e) {}
    cv::Mat screenRgb, cachedRgb_, gestureBackground_, gestureAsset_;
    std::unique_ptr<Fl_RGB_Image> cachedImage_;
    const uchar *cachedData_ = nullptr;
    uint64_t cachedRevision_ = ~uint64_t(0);
    cv::Rect cachedView_;
    cv::Size cachedSource_;
    bool objectDrag_ = false, gestureMoved_ = false, marquee_ = false;
    int gestureHandle_ = -1;
    json gestureFrame_, currentFrame_;
    std::vector<std::pair<int, json>> gestureFrames_;
    std::vector<cv::Mat> gestureAssets_;
    cv::Point gestureStart_;
    double gestureStartAngle_ = 0;
    void draw() override;
    int handle(int event) override;
    int hitObject(cv::Point) const;
    int handleAt(int, int) const;
    std::vector<cv::Point2f> framePoints(const json &) const;
    cv::Point screenPoint(cv::Point2f) const;
    void pointerDown(int, int, bool shift = false);
    void pointerMove(int, int, bool shift = false);
    void pointerUp(int, int, bool shift = false);
    void cancelGesture();
    void invalidate() {
        cachedData_ = nullptr;
    }
    cv::Point imagePoint(int x, int y) const;
    float imageScale() const;
    void reset() {
        zoom_ = 1;
        panX_ = panY_ = 0;
        redraw();
    }
};
class SlideStrip : public Fl_Widget {
    Editor &e_;
    int scroll_ = 0;
    std::map<std::string, std::pair<uint64_t, cv::Mat>> thumbnails_;
    std::map<std::string, std::unique_ptr<Fl_RGB_Image>> thumbnailImages_;

  public:
    SlideStrip(int x, int y, int w, int h, Editor &e) : Fl_Widget(x, y, w, h), e_(e) {}
    void draw() override;
    int handle(int event) override;
};
class AppWindow : public Fl_Double_Window {
  public:
    std::function<bool(int)> key;
    AppWindow(int w, int h) : Fl_Double_Window(w, h, "FrameMorph 0.8 - Slide studio") {}
    int handle(int event) override {
        if (event == FL_SHORTCUT && key && key(Fl::event_key()))
            return 1;
        return Fl_Double_Window::handle(event);
    }
};
class RibbonButton : public Fl_Button {
    int kind_;

  public:
    RibbonButton(int x, int y, int w, int h, const char *text, int kind)
        : Fl_Button(x, y, w, h, text), kind_(kind) {}
    void draw() override {
        fl_color(value() ? fl_rgb_color(214, 228, 255) : color());
        fl_rectf(x(), y(), w(), h());
        if (Fl::belowmouse() == this) {
            fl_color(line);
            fl_rect(x(), y(), w(), h());
        }
        int cx = x() + w() / 2, cy = y() + 14;
        fl_color(accent);
        fl_line_style(FL_SOLID, 2);
        switch (kind_) {
        case -1:
            fl_polygon(cx - 7, cy - 10, cx - 7, cy + 9, cx + 10, cy + 2);
            fl_line(cx - 2, cy + 1, cx + 5, cy + 10);
            break;
        case 0:
            fl_rect(cx - 11, cy - 7, 22, 15);
            break;
        case 1:
            fl_arc(cx - 9, cy - 9, 18, 18, 0, 360);
            break;
        case 4:
            fl_line(cx - 11, cy + 7, cx + 11, cy - 7);
            break;
        case 5:
            fl_line(cx - 11, cy + 7, cx + 10, cy - 7);
            fl_line(cx + 10, cy - 7, cx + 2, cy - 7);
            fl_line(cx + 10, cy - 7, cx + 10, cy + 1);
            break;
        case 6:
            fl_font(FL_HELVETICA_BOLD, 23);
            fl_draw("T", cx - 7, cy + 8);
            break;
        case 7:
            fl_font(FL_TIMES_ITALIC, 21);
            fl_draw("fx", cx - 9, cy + 7);
            break;
        case 8:
            fl_line(cx - 13, cy + 9, cx - 13, cy - 9);
            fl_line(cx - 13, cy + 9, cx + 14, cy + 9);
            fl_line(cx - 11, cy + 4, cx - 5, cy - 3, cx, cy + 2);
            fl_line(cx, cy + 2, cx + 6, cy + 5, cx + 13, cy - 7);
            break;
        case 11:
            fl_begin_loop();
            fl_vertex(cx - 10, cy + 5);
            fl_vertex(cx - 8, cy - 5);
            fl_vertex(cx, cy - 9);
            fl_vertex(cx + 10, cy - 3);
            fl_vertex(cx + 7, cy + 7);
            fl_vertex(cx - 3, cy + 8);
            fl_end_loop();
            break;
        case 9:
            fl_rect(cx - 12, cy - 9, 24, 18);
            fl_line(cx - 10, cy + 7, cx - 2, cy - 1, cx + 4, cy + 4);
            fl_line(cx + 4, cy + 4, cx + 9, cy - 1);
            fl_pie(cx + 4, cy - 6, 4, 4, 0, 360);
            break;
        case 12:
            fl_line_style(FL_DASH, 1);
            fl_rect(cx - 10, cy - 8, 20, 16);
            fl_line_style(FL_SOLID, 2);
            fl_line(cx - 6, cy + 3, cx, cy - 3, cx + 6, cy + 3);
            break;
        case 13:
        case 14:
            fl_line(cx - 8, cy + 8, cx + 8, cy - 8);
            fl_line(cx - 10, cy + 6, cx - 6, cy + 10);
            fl_line(cx + 7, cy - 5, cx + 12, cy - 5);
            if (kind_ == 13)
                fl_line(cx + 10, cy - 8, cx + 10, cy - 2);
            break;
        case 15:
            fl_rect(cx - 11, cy - 8, 22, 17);
            fl_line(cx - 5, cy, cx + 5, cy);
            fl_line(cx, cy - 5, cx, cy + 5);
            break;
        case 16:
            fl_rect(cx - 8, cy - 9, 19, 15);
            fl_rect(cx - 12, cy - 5, 19, 15);
            break;
        }
        fl_line_style(0);
        fl_color(fg);
        fl_font(FL_HELVETICA, 11);
        fl_draw(label(), x(), y() + h() - 17, w(), 15, FL_ALIGN_CENTER);
    }
};
class Editor {
  public:
    Project project;
    int slide = 0, object = -1, transition = 0, connection = -1;
    cv::Mat selection;
    CudaBackend selectionGpu;
    Canvas *canvas = nullptr;
    SlideStrip *strip = nullptr;
    Fl_Choice *tool = nullptr;
    Fl_Value_Slider *tolerance = nullptr, *brush = nullptr;
    Fl_Input *objectName = nullptr;
    Fl_Float_Input *objectLayer = nullptr, *hold = nullptr;
    Fl_Multi_Browser *objects = nullptr, *sources = nullptr, *targets = nullptr;
    Fl_Hold_Browser *connections = nullptr;
    Fl_Choice *transitionChoice = nullptr;
    Fl_Input *pairs = nullptr, *anchors = nullptr;
    Fl_Float_Input *duration = nullptr, *start = nullptr, *end = nullptr, *bend = nullptr,
                   *rotation = nullptr, *layer = nullptr;
    Fl_Float_Input *exportWidth = nullptr, *exportHeight = nullptr, *exportFps = nullptr;
    Fl_Choice *easing = nullptr, *mode = nullptr;
    Fl_Value_Slider *scrub = nullptr;
    Fl_Button *play = nullptr, *exportButton = nullptr, *cancelButton = nullptr;
    Fl_Box *status = nullptr, *clock = nullptr;
    Fl_Choice *drawType = nullptr;
    Fl_Input *drawName = nullptr, *drawColor = nullptr, *plotRange = nullptr;
    Fl_Multiline_Input *drawContent = nullptr;
    Fl_Input *drawX = nullptr, *drawY = nullptr, *drawW = nullptr, *drawH = nullptr,
             *drawStroke = nullptr, *drawFont = nullptr, *drawAngle = nullptr,
             *drawOpacity = nullptr;
    Fl_Input *functionX = nullptr, *functionY = nullptr, *functionScale = nullptr,
             *functionRotation = nullptr, *functionOpacity = nullptr;
    Fl_Check_Button *drawFilled = nullptr;
    Fl_Group *inspector = nullptr, *drawTab = nullptr, *animationTab = nullptr;
    Fl_Scroll *propertiesScroll = nullptr;
    Fl_Group *objectProperties = nullptr, *slideProperties = nullptr;
    Fl_Box *propertyTitle = nullptr, *propertyKind = nullptr, *hint = nullptr;
    Fl_Input *propX = nullptr, *propY = nullptr, *propW = nullptr, *propH = nullptr,
             *propAngle = nullptr, *propOpacity = nullptr;
    Fl_Input *slideTitle = nullptr;
    Fl_Check_Button *animateObject = nullptr;
    Fl_Button *selectToolButton = nullptr;
    uint64_t revision = 0;
    cv::Rect selectedBounds;
    bool constructionEditing = false;
    json drawDraft;
    Fl_Input *objectMap = nullptr, *mapUnits = nullptr;
    Fl_Multiline_Input *updaterInput = nullptr;
    Fl_Input *updaterUnits = nullptr, *stepDuration = nullptr;
    Fl_Hold_Browser *updaterList = nullptr;
    Fl_Box *updaterObject = nullptr;
    Fl_Choice *constructEffect = nullptr, *destructEffect = nullptr, *effectEasing = nullptr;
    Fl_Input *effectStart = nullptr, *effectEnd = nullptr;
    Fl_Button *generateUpdatersButton = nullptr;
    bool slideFocus = true;
    int placement = 0;
    std::optional<Object> clipboard;
    std::unique_ptr<AppWindow> window;
    std::vector<std::unique_ptr<std::function<void()>>> callbacks;
    struct HistoryState {
        Project project;
        cv::Mat selection;
        int slide, object;
        bool slideFocus;
        std::vector<int> selectedObjects;
    };
    std::deque<HistoryState> undoStack, redoStack;
    PreviewWorker preview;
    std::shared_ptr<Project> snapshot;
    bool dirty = false, updating = false, playing = false, exporting = false, playStarting = false;
    int previewPresented = 0;
    std::thread exportThread;
    std::atomic_bool exportCancel{false}, exportDone{false};
    std::atomic_int exportFrame{0}, exportTotal{1};
    std::mutex exportMutex;
    std::string exportError;
    fs::path exportPath;
    std::chrono::steady_clock::time_point lastTick;
    double playTime = 0;
    bool snapshotDirty = true, previewBusy = false;
    std::chrono::steady_clock::time_point lastAutosave = std::chrono::steady_clock::now();
    Editor(const fs::path &path) {
        if (!path.empty()) {
            try {
                bool recover = fs::exists(path / "autosave.json");
                project = loadProject(path, recover);
                dirty = recover;
            } catch (const std::exception &e) {
                fl_alert("%s", e.what());
            }
        }
        if (path.empty()) {
            auto first = blankSlide(project.width, project.height, {255, 255, 255});
            first.title = "Slide 1";
            project.slides.push_back(std::move(first));
        }
        cv::setNumThreads(2);
        build();
        if (dirty)
            message("Recovered autosaved changes. Save to keep them.");
        else
            message("Select an object to edit it. Insert shapes, text, pictures, or mathematics "
                    "from the ribbon.");
        refresh();
        lastTick = std::chrono::steady_clock::now();
        Fl::add_timeout(1. / 60, tickCallback, this);
    }
    ~Editor() {
        Fl::remove_timeout(tickCallback, this);
        exportCancel = true;
        if (exportThread.joinable())
            exportThread.join();
    }
    void bind(Fl_Widget *w, std::function<void()> fn) {
        auto callback = std::make_unique<std::function<void()>>([this, fn] {
            if (updating)
                return;
            try {
                fn();
            } catch (const std::exception &e) {
                message(e.what());
                fl_alert("%s", e.what());
            }
        });
        w->callback([](Fl_Widget *, void *arg) { (*static_cast<std::function<void()> *>(arg))(); },
                    callback.get());
        callbacks.push_back(std::move(callback));
    }
    Fl_Button *button(int x, int y, int w, int h, const char *label, std::function<void()> fn) {
        auto *b = new Fl_Button(x, y, w, h, label);
        b->box(FL_FLAT_BOX);
        b->color(surface);
        b->labelcolor(fg);
        b->selection_color(accent);
        b->labelsize(12);
        bind(b, std::move(fn));
        return b;
    }
    Fl_Button *ribbonButton(int x, int y, int w, int h, const char *title, int kind,
                            std::function<void()> fn) {
        auto *b = new RibbonButton(x, y, w, h, title, kind);
        b->color(surface);
        b->labelcolor(fg);
        b->tooltip(title);
        bind(b, std::move(fn));
        return b;
    }
    Fl_Box *label(int x, int y, int w, const char *text, int size = 11, Fl_Color color = dim) {
        auto *b = new Fl_Box(x, y, w, 20, text);
        b->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        b->labelsize(size);
        b->labelcolor(color);
        return b;
    }
    Fl_Float_Input *field(int x, int y, int w, const char *title, double initial) {
        label(x, y - 20, w, title);
        auto *f = new Fl_Float_Input(x, y, w, 28);
        f->value(number(initial).c_str());
        f->when(FL_WHEN_ENTER_KEY | FL_WHEN_RELEASE);
        return f;
    }
    Fl_Input *mathField(int x, int y, int w, const char *title, const char *initial) {
        label(x, y - 20, w, title);
        auto *f = new Fl_Input(x, y, w, 28);
        f->value(initial);
        f->textsize(12);
        return f;
    }
    void build() {
        Fl::scheme("gtk+");
        Fl::background(245, 247, 251);
        Fl::background2(255, 255, 255);
        Fl::foreground(29, 41, 61);
        Fl::set_color(FL_SELECTION_COLOR, 45, 103, 230);
        Fl::visible_focus(0);
        window = std::make_unique<AppWindow>(1600, 1000);
        window->color(bg);
        window->size_range(1280, 820);
        window->begin();
        label(20, 13, 180, "FrameMorph", 23, accent);
        button(170, 12, 40, 30, "New", [this] { newProject(); });
        button(215, 12, 68, 30, "Open", [this] { open(); });
        button(290, 12, 68, 30, "Save", [this] { save(false); });
        button(365, 12, 78, 30, "Save as", [this] { save(true); });
        button(458, 12, 68, 30, "Undo", [this] { undo(false); });
        button(533, 12, 68, 30, "Redo", [this] { undo(true); });
        button(618, 12, 110, 30, "Import slides", [this] { importImages(); });
        label(770, 14, 320, "Slide studio  /  2D animation", 13, dim);
        button(1150, 12, 115, 30, "Help / F1", [] { showAppHelp(); });
        exportButton = button(1290, 12, 130, 30, "Export video", [this] { beginExport(false); });
        button(1430, 12, 145, 30, "Export PNG frames", [this] { beginExport(true); });
        auto *ribbon = new Fl_Box(10, 56, 1580, 91);
        ribbon->box(FL_FLAT_BOX);
        ribbon->color(panel);
        label(20, 59, 170, "SLIDES", 10, dim);
        label(215, 59, 1120, "INSERT & EDIT", 10, dim);
        ribbonButton(20, 86, 80, 44, "New slide", 15, [this] { addBlank(); });
        ribbonButton(108, 86, 82, 44, "Duplicate", 16, [this] { duplicateCurrent(); });
        selectToolButton =
            ribbonButton(215, 86, 86, 44, "Select", -1, [this] { activateSelect(); });
        auto insert = [&](int x, int kind, const char *title) {
            ribbonButton(x, 86, 86, 44, title, kind, [this, kind] { armInsert(kind); });
        };
        insert(309, 0, "Rectangle");
        insert(403, 1, "Circle");
        insert(497, 4, "Line");
        insert(591, 5, "Arrow");
        insert(685, 6, "Text");
        ribbonButton(779, 86, 86, 44, "LaTeX", 7, [this] { openMediaEditor("equation", false); });
        insert(873, 8, "y = f(x)");
        insert(967, 11, "f(x,y) = 0");
        ribbonButton(1061, 86, 86, 44, "Picture", 9, [this] {
            insertImage();
            activateSelect();
            syncProperties();
        });
        ribbonButton(1155, 86, 106, 44, "Extract region", 12, [this] { activateRegion(0); });
        ribbonButton(1269, 86, 86, 44, "Add brush", 13, [this] { activateRegion(1); });
        ribbonButton(1363, 86, 86, 44, "Erase brush", 14, [this] { activateRegion(2); });
        auto *more = new Fl_Menu_Button(1457, 86, 118, 44, "More...");
        more->add("Ellipse");
        more->add("Triangle");
        more->add("Diamond");
        more->add("Star");
        more->add("Hexagon");
        more->add("Source code block...");
        more->add("SVG artwork...");
        more->add("Align selected left|Align selected right|Align selected top|Align selected "
                  "bottom|Distribute horizontally|Distribute vertically");
        more->tooltip("More shapes and media. Select several objects with Shift-click or a marquee "
                      "to arrange them.");
        bind(more, [this, more] {
            int v = more->value();
            if (v >= 7) {
                arrangeSelection(v - 7);
                return;
            }
            if (v >= 5)
                openMediaEditor(v == 5 ? "code" : "svg", false);
            else
                armInsert(v == 0 ? 2 : v == 1 ? 3 : v + 12);
        });
        label(20, 158, 170, "SLIDES", 11, dim);
        strip = new SlideStrip(10, 183, 185, 630, *this);
        button(20, 824, 78, 28, "Move up", [this] { moveSlide(-1); });
        button(107, 824, 78, 28, "Move down", [this] { moveSlide(1); });
        button(20, 861, 165, 28, "Delete slide", [this] { removeSlide(); });
        canvas = new Canvas(210, 158, 990, 655, *this);
        hint = new Fl_Box(215, 820, 850, 25,
                          "Click to select. Drag to move. Use the handles to resize.");
        hint->labelsize(12);
        hint->labelcolor(dim);
        hint->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        button(1080, 820, 115, 27, "Fit slide", [this] { canvas->reset(); });
        inspector = new Fl_Group(1210, 158, 375, 687, "Properties");
        inspector->align(FL_ALIGN_TOP_LEFT);
        inspector->labelsize(15);
        inspector->begin();
        drawTab = new Fl_Group(1210, 168, 375, 677);
        drawTab->begin();
        propertiesScroll = new Fl_Scroll(1215, 173, 365, 667);
        propertiesScroll->type(Fl_Scroll::VERTICAL_ALWAYS);
        propertiesScroll->begin();
        propertyTitle = new Fl_Box(1230, 186, 180, 27, "Frame properties");
        propertyTitle->labelsize(17);
        propertyTitle->labelcolor(fg);
        propertyTitle->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        button(1415, 186, 135, 27, "Frame properties", [this] {
            activateSelect();
            selectObject(-1);
            slideFocus = true;
        });
        propertyKind = new Fl_Box(1230, 222, 325, 22, "");
        propertyKind->labelsize(11);
        propertyKind->labelcolor(dim);
        propertyKind->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        objectProperties = new Fl_Group(1225, 250, 330, 1780);
        objectProperties->begin();
        objectName = mathField(1230, 279, 320, "Name", "");
        objectName->when(FL_WHEN_ENTER_KEY | FL_WHEN_RELEASE);
        bind(objectName, [this] {
            renameObject();
            syncProperties();
        });
        animateObject =
            new Fl_Check_Button(1230, 318, 320, 25, "Animate across frames (match __name)");
        animateObject->labelsize(12);
        bind(animateObject, [this] {
            if (auto *o = currentObject()) {
                auto name = o->name;
                if (animateObject->value() && name.rfind("__", 0) != 0)
                    name = "__" + name;
                else if (!animateObject->value() && name.rfind("__", 0) == 0)
                    name = name.substr(2);
                objectName->value(name.c_str());
                renameObject();
                syncProperties();
            }
        });
        label(1230, 361, 320, "POSITION & SIZE", 11, accent);
        propX = mathField(1230, 405, 150, "X", "0");
        propY = mathField(1400, 405, 150, "Y", "0");
        propW = mathField(1230, 463, 150, "Width", "100");
        propH = mathField(1400, 463, 150, "Height", "100");
        propAngle = mathField(1230, 521, 150, "Rotation (degrees)", "0");
        propOpacity = mathField(1400, 521, 150, "Opacity (0-1)", "1");
        for (auto *f : {propX, propY, propW, propH, propAngle, propOpacity}) {
            f->when(FL_WHEN_ENTER_KEY | FL_WHEN_RELEASE);
            bind(f, [this] { applyObjectGeometry(); });
        }
        button(1230, 562, 150, 28, "Center horizontally", [this] { alignObject(true); });
        button(1400, 562, 150, 28, "Center vertically", [this] { alignObject(false); });
        label(1230, 609, 320, "APPEARANCE", 11, accent);
        drawColor = mathField(1230, 653, 150, "Color / tint", "#4ed5b8");
        drawColor->when(FL_WHEN_ENTER_KEY | FL_WHEN_RELEASE);
        bind(drawColor, [this] { applyObjectColor(); });
        button(1400, 653, 150, 28, "Choose color...", [this] {
            double r = .18, g = .4, b = .9;
            if (fl_color_chooser("Object color", r, g, b)) {
                char value[8];
                std::snprintf(value, sizeof(value), "#%02x%02x%02x", int(r * 255 + .5),
                              int(g * 255 + .5), int(b * 255 + .5));
                drawColor->value(value);
                applyObjectColor();
            }
        });
        drawFont = mathField(1230, 711, 150, "Font size (px)", "56");
        drawStroke = mathField(1400, 711, 150, "Stroke width", "4");
        for (auto *f : {drawFont, drawStroke}) {
            f->when(FL_WHEN_ENTER_KEY | FL_WHEN_RELEASE);
            bind(f, [this] { applyConstruction(); });
        }
        drawFilled = new Fl_Check_Button(1230, 751, 320, 25, "Fill shape / show graph axes");
        drawFilled->resize(drawFilled->x(), drawFilled->y(), 160, drawFilled->h());
        button(1400, 751, 150, 28, "Edit content...", [this] { editSelectedContent(); });
        drawFilled->labelsize(12);
        drawFilled->value(1);
        bind(drawFilled, [this] { applyConstruction(); });
        label(1230, 801, 320, "CONSTRUCT / DESTRUCT", 11, accent);
        label(1230, 827, 320, "Only when this object appears or disappears", 11, dim);
        constructEffect = new Fl_Choice(1230, 878, 150, 28, "Construct");
        constructEffect->align(FL_ALIGN_TOP_LEFT);
        destructEffect = new Fl_Choice(1400, 878, 150, 28, "Destruct");
        destructEffect->align(FL_ALIGN_TOP_LEFT);
        for (auto &entry : lifecycleEffects()) {
            constructEffect->add(entry.second.c_str());
            destructEffect->add(entry.second.c_str());
        }
        constructEffect->value(0);
        destructEffect->value(0);
        bind(constructEffect, [this] { applyLifecycle(); });
        bind(destructEffect, [this] { applyLifecycle(); });
        label(1230, 937, 320, "UPDATER FUNCTION", 11, accent);
        updaterObject = label(1230, 965, 320, "", 11, dim);
        updaterInput = new Fl_Multiline_Input(1230, 995, 320, 82);
        updaterInput->textsize(13);
        updaterInput->tooltip("(x,y,t):new_x,new_y or z:new_z. Original local coordinates; t=0..1; "
                              "seconds=t*duration. Save here; generate in Frame properties.");
        updaterUnits = mathField(1230, 1117, 150, "Pixels per unit", "100");
        button(1400, 1117, 150, 28, "Save updater", [this] { saveUpdater(); });
        auto *presets = new Fl_Choice(1230, 1160, 150, 28);
        presets->add("Motion presets...|Circle orbit|Full rotation|Move right|Bounce "
                     "up|Pulse|Oscillate|Shear matrix|Wave deformation|Spiral turn");
        presets->value(0);
        bind(presets, [this, presets] {
            static const char *expressions[]{"",
                                             "z:z+2*(exp(i*tau*t)-1)",
                                             "z:z*exp(i*tau*t)",
                                             "z:z+2*t",
                                             "z:z+2*i*sin(pi*t)",
                                             "z:z*(1+0.2*sin(pi*t))",
                                             "z:z+sin(tau*t)",
                                             "(x,y,t):x+t*y,y",
                                             "(x,y,t):x+0.3*sin(2*y)*sin(pi*t),y",
                                             "z:z*exp(i*tau*t)*(1+0.25*t)"};
            if (presets->value() > 0)
                updaterInput->value(expressions[presets->value()]);
            presets->value(0);
        });
        button(1400, 1160, 150, 28, "Remove updater", [this] { clearUpdater(); });
        label(1230, 1201, 320, "Create the next slide in Frame properties.", 11, dim);
        label(1230, 1250, 320, "MAP THIS OBJECT NOW", 11, accent);
        objectMap = new Fl_Input(1230, 1285, 320, 36);
        objectMap->textsize(14);
        objectMap->tooltip("(x,y):x+1,y+2 or z:z+2; relative to object center. Apply updates this "
                           "object immediately.");
        mapUnits = mathField(1230, 1363, 150, "Pixels per unit", "100");
        button(1400, 1363, 150, 28, "Apply update", [this] { applyMap(); });
        button(1230, 1405, 320, 28, "Scene variables...", [this] { showVariables(); });
        label(1230, 1456, 320, "TEXT / MATHEMATICS", 11, accent);
        drawContent = new Fl_Multiline_Input(1230, 1490, 320, 83);
        drawContent->textsize(14);
        button(1230, 1584, 320, 29, "Update text or formula", [this] { applyConstruction(); });
        plotRange = mathField(1230, 1658, 320, "Graph range: xmin, xmax, ymin, ymax",
                              "-2*pi,2*pi,-1.5,1.5");
        plotRange->when(FL_WHEN_ENTER_KEY | FL_WHEN_RELEASE);
        bind(plotRange, [this] { applyConstruction(); });
        objectLayer = field(1230, 1723, 150, "Layer", 0);
        bind(objectLayer, [this] {
            if (auto *o = currentObject()) {
                int v = std::stoi(objectLayer->value());
                checkpoint();
                o->layer = v;
                recomposeSlide(*currentSlide());
                changed();
                syncProperties();
            }
        });
        button(1400, 1723, 150, 28, "Group selection", [this] { groupObjects(); });
        button(1230, 1768, 150, 28, "Bring to front", [this] { arrangeLayer(true); });
        button(1400, 1768, 150, 28, "Send to back", [this] { arrangeLayer(false); });
        button(1230, 1812, 95, 28, "Copy", [this] { copyDrawing(); });
        button(1333, 1812, 100, 28, "Paste", [this] { pasteDrawing(); });
        button(1440, 1812, 110, 28, "Delete", [this] { deleteObjects(); });
        // Make room for timing beside the lifecycle choices, in the same inspector.
        for (int i = 0; i < objectProperties->children(); ++i) {
            auto *w = objectProperties->child(i);
            if (w->y() >= 937)
                w->position(w->x(), w->y() + 160);
        }
        effectStart = mathField(1230, 950, 150, "Effect starts (%)", "0");
        effectEnd = mathField(1400, 950, 150, "Effect ends (%)", "100");
        effectEasing = new Fl_Choice(1230, 1012, 320, 28, "Effect easing");
        effectEasing->align(FL_ALIGN_TOP_LEFT);
        effectEasing->add("Smooth|Linear|Ease in|Ease out");
        effectEasing->value(0);
        bind(effectStart, [this] { applyLifecycle(); });
        bind(effectEnd, [this] { applyLifecycle(); });
        bind(effectEasing, [this] { applyLifecycle(); });
        label(1230, 1055, 320, "Timing is relative to this frame's time step.", 11, dim);
        button(1230, 2030, 320, 28, "Image filters / pixel formulas...",
               [this] { openImageEditor(); });
        auto *combine =
            new Fl_Menu_Button(1230, 2075, 320, 28, "Combine selected objects as image...");
        combine->add("Overlay at current positions|Side by side|Vertical stack");
        bind(combine, [this, combine] {
            combineSelectedImages(combine->value() == 0   ? "overlay"
                                  : combine->value() == 1 ? "row"
                                                          : "column");
        });
        objectProperties->end();
        slideProperties = new Fl_Group(1225, 250, 330, 1260);
        slideProperties->begin();
        slideTitle = mathField(1230, 279, 320, "Frame title", "");
        slideTitle->when(FL_WHEN_ENTER_KEY | FL_WHEN_RELEASE);
        bind(slideTitle, [this] {
            if (auto *s = currentSlide()) {
                checkpoint();
                s->title = slideTitle->value();
                changed();
            }
        });
        hold = field(1230, 341, 150, "Hold (seconds)", .5);
        bind(hold, [this] {
            if (auto *s = currentSlide()) {
                double v = valueOf(hold);
                if (v < 0 || v > 3600)
                    throw std::runtime_error("Frame hold must be 0 to 3600 seconds");
                checkpoint();
                s->hold = v;
                changed();
            }
        });
        button(1400, 341, 150, 28, "Background color", [this] { chooseBackground(); });
        stepDuration = mathField(1230, 407, 320, "Time step to next frame (seconds)", "1.5");
        stepDuration->when(FL_WHEN_ENTER_KEY | FL_WHEN_RELEASE);
        bind(stepDuration, [this] { applyFrameStep(); });
        generateUpdatersButton = button(1230, 456, 320, 34, "Create next slide from updaters",
                                        [this] { generateFromUpdaters(); });
        label(1230, 502, 320, "New / Duplicate uses ordinary name matching.", 11, dim);
        label(1230, 549, 320, "OBJECTS WITH SAVED UPDATERS", 11, accent);
        updaterList = new Fl_Hold_Browser(1230, 579, 320, 96);
        updaterList->textsize(12);
        bind(updaterList, [this] {
            int wanted = updaterList->value() - 1, current = 0;
            if (currentSlide())
                for (int i = 0; i < int(currentSlide()->objects.size()); ++i)
                    if (currentSlide()->objects[i].updater.is_object() && current++ == wanted) {
                        selectObject(i);
                        break;
                    }
        });
        label(1230, 708, 320, "OBJECTS ON THIS FRAME", 11, accent);
        objects = new Fl_Multi_Browser(1230, 740, 320, 135);
        objects->textsize(13);
        bind(objects, [this] { chooseObject(); });
        button(1230, 889, 150, 28, "Group selection", [this] { groupObjects(); });
        button(1400, 889, 150, 28, "Delete selection", [this] { deleteObjects(); });
        button(1400, 933, 150, 28, "Paste object", [this] { pasteDrawing(); });
        auto *combineFrame = new Fl_Menu_Button(1230, 933, 150, 28, "Combine images...");
        combineFrame->add("Overlay|Side by side|Vertical stack");
        bind(combineFrame, [this, combineFrame] {
            combineSelectedImages(combineFrame->value() == 0   ? "overlay"
                                  : combineFrame->value() == 1 ? "row"
                                                               : "column");
        });
        button(1230, 974, 320, 28, "Edit slide image...", [this] { editSlideImage(); });
        label(1230, 1004, 320, "EXTRACT FROM AN IMAGE", 12, accent);
        tool = new Fl_Choice(1230, 1046, 320, 28);
        tool->add("Color region|Add brush|Erase brush|Subtract region|Select objects");
        tool->value(4);
        bind(tool, [this] {
            if (tool->value() == 4)
                activateSelect();
            else
                activateRegion(tool->value());
        });
        tolerance = new Fl_Value_Slider(1230, 1110, 320, 26, "Color tolerance");
        tolerance->align(FL_ALIGN_TOP_LEFT);
        tolerance->type(FL_HOR_NICE_SLIDER);
        tolerance->bounds(1, 100);
        tolerance->value(18);
        tolerance->step(1);
        brush = new Fl_Value_Slider(1230, 1172, 320, 26, "Brush radius (pixels)");
        brush->align(FL_ALIGN_TOP_LEFT);
        brush->type(FL_HOR_NICE_SLIDER);
        brush->bounds(1, 120);
        brush->value(16);
        brush->step(1);
        button(1230, 1225, 150, 28, "Create object", [this] {
            createObject();
            activateSelect();
            syncProperties();
        });
        button(1400, 1225, 150, 28, "Update mask", [this] { updateMask(); });
        button(1230, 1265, 150, 28, "Clear selection", [this] {
            selection = cv::Mat{};
            canvas->redraw();
        });
        button(1400, 1265, 150, 28, "Pick background", [this] { canvas->eyedropper = true; });
        for (int i = 0; i < slideProperties->children(); ++i) {
            auto *w = slideProperties->child(i);
            if (w->y() >= 1004)
                w->position(w->x(), w->y() + 44);
        }
        slideProperties->end();
        for (auto *group : {objectProperties, slideProperties}) {
            int bottom = group->y();
            for (int i = 0; i < group->children(); ++i)
                bottom = std::max(bottom, group->child(i)->y() + group->child(i)->h());
            group->size(group->w(), bottom - group->y() + 24);
        }
        objectName->tooltip("Names beginning __ match exactly across adjacent frames. Repeated "
                            "names split or merge.");
        stepDuration->tooltip(
            "Duration to the next frame. Also controls seconds=t*duration in generated updaters.");
        drawContent->tooltip(
            "Edit native content here, or use Edit content for the preview studio.");
        propertiesScroll->end();
        drawTab->end();
        inspector->end();
        // Legacy project correspondence controls are kept off-screen for old project actions.
        // The editor exposes one context-sensitive Properties inspector.
        animationTab = new Fl_Group(1210, 190, 375, 655, "Animation");
        animationTab->begin();
        auto *motionScroll = new Fl_Scroll(1215, 195, 365, 645);
        motionScroll->type(Fl_Scroll::VERTICAL_ALWAYS);
        motionScroll->begin();
        label(1230, 207, 320, "ANIMATE BETWEEN SLIDES", 13, accent);
        transitionChoice = new Fl_Choice(1230, 250, 320, 28);
        bind(transitionChoice, [this] {
            transition = std::max(0, transitionChoice->value());
            connection = -1;
            refreshConnections();
        });
        duration = field(1230, 311, 150, "Duration (seconds)", 1.5);
        bind(duration, [this] { applyMotion(); });
        button(1400, 311, 150, 28, "Preview", [this] { previewTransition(); });
        connections = new Fl_Hold_Browser(1230, 360, 320, 100);
        connections->textsize(12);
        bind(connections, [this] {
            connection = connections->value() - 1;
            refreshMotion();
        });
        start = field(1230, 510, 150, "Start (%)", 0);
        end = field(1400, 510, 150, "End (%)", 100);
        bend = field(1230, 570, 150, "Path curvature", 0);
        rotation = field(1400, 570, 150, "Extra rotation", 0);
        layer = field(1230, 630, 150, "Animation layer", 0);
        mode = new Fl_Choice(1400, 630, 150, 28);
        mode->add("Shape morph|Fade through");
        mode->value(0);
        easing = new Fl_Choice(1230, 680, 320, 28);
        easing->add("Smooth|Linear|Ease in|Ease out");
        easing->value(0);
        for (auto *f : {start, end, bend, rotation, layer})
            bind(f, [this] { applyMotion(); });
        bind(mode, [this] { applyMotion(); });
        bind(easing, [this] { applyMotion(); });
        label(1230, 735, 320, "FUNCTIONS OF t  (0 to 1)", 12, accent);
        functionX = mathField(1230, 787, 320, "Extra X offset (pixels)", "");
        functionY = mathField(1230, 850, 320, "Extra Y offset (pixels)", "");
        functionScale = mathField(1230, 913, 320, "Scale multiplier", "");
        functionRotation = mathField(1230, 976, 320, "Extra rotation (degrees)", "");
        functionOpacity = mathField(1230, 1039, 320, "Opacity multiplier", "");
        button(1230, 1083, 320, 30, "Apply motion functions", [this] { applyFunctions(); });
        label(1230, 1140, 320, "OBJECT CONNECTIONS", 12, accent);
        sources = new Fl_Multi_Browser(1230, 1180, 150, 130);
        targets = new Fl_Multi_Browser(1400, 1180, 150, 130);
        sources->textsize(11);
        targets->textsize(11);
        button(1230, 1320, 320, 30, "Connect selected / split / merge",
               [this] { connectSelected(); });
        button(1230, 1365, 320, 28, "Rebuild automatic matches", [this] {
            checkpoint();
            rematch(true);
            changed();
            refreshConnections();
        });
        pairs = mathField(1230, 1430, 320, "Component pairs: 0:1, 1:0", "");
        anchors = mathField(1230, 1495, 320, "Boundary anchors: 0.1:0.3", "");
        button(1230, 1540, 320, 28, "Apply correspondence", [this] { applyCorrespondence(); });
        motionScroll->end();
        animationTab->end();
        animationTab->hide();
        // Construction coordinates are separate from the common object frame.
        auto *constructionFields = new Fl_Group(0, 0, 1, 1);
        constructionFields->begin();
        drawType = new Fl_Choice(0, 0, 1, 1);
        for (auto &k : drawingKinds())
            drawType->add(k.c_str());
        drawType->value(0);
        drawName = new Fl_Input(0, 0, 1, 1);
        drawX = mathField(0, 0, 1, "", "300");
        drawY = mathField(0, 0, 1, "", "320");
        drawW = mathField(0, 0, 1, "", "360");
        drawH = mathField(0, 0, 1, "", "240");
        drawAngle = mathField(0, 0, 1, "", "0");
        drawOpacity = mathField(0, 0, 1, "", "1");
        constructionFields->end();
        constructionFields->hide();
        label(215, 864, 900, "TIMELINE", 10, dim);
        play = button(215, 899, 85, 32, "Play", [this] { togglePlay(); });
        scrub = new Fl_Value_Slider(318, 900, 760, 29);
        scrub->type(FL_HOR_NICE_SLIDER);
        scrub->bounds(0, 1);
        scrub->step(.001);
        bind(scrub, [this] {
            playing = false;
            play->label("Play");
            requestPreview(scrub->value());
        });
        clock = new Fl_Box(1090, 899, 105, 30, "0.00 sec");
        clock->labelsize(12);
        clock->labelcolor(fg);
        exportWidth = field(1230, 900, 95, "Export width", 1920);
        exportHeight = field(1340, 900, 95, "Height", 1080);
        exportFps = field(1450, 900, 95, "FPS", 60);
        status = new Fl_Box(20, 958, 1300, 26, "");
        status->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        status->labelsize(12);
        status->labelcolor(dim);
        cancelButton = button(1385, 954, 190, 32, "Cancel export", [this] { exportCancel = true; });
        cancelButton->hide();
        window->end();
        window->resizable(canvas);
        window->key = [this](int key) {
            bool editing = dynamic_cast<Fl_Input_ *>(Fl::focus()) != nullptr;
            if (key == FL_Escape && !editing) {
                canvas->cancelGesture();
                placement = 0;
                activateSelect();
                return true;
            }
            if (Fl::event_state(FL_CTRL)) {
                if (key == 'n') {
                    newProject();
                    return true;
                }
                if (key == 's') {
                    save(false);
                    return true;
                }
                if (key == 'o') {
                    open();
                    return true;
                }
                if (!editing) {
                    if (key == 'z') {
                        undo(false);
                        return true;
                    }
                    if (key == 'y') {
                        undo(true);
                        return true;
                    }
                    if (key == 'd') {
                        if (slideFocus)
                            duplicateCurrent();
                        else
                            duplicateObject();
                        return true;
                    }
                    if (key == 'c') {
                        copyDrawing();
                        return true;
                    }
                    if (key == 'v') {
                        pasteDrawing();
                        return true;
                    }
                    if (key == 'a') {
                        for (int i = 1; i <= objects->size(); ++i)
                            objects->select(i);
                        return true;
                    }
                }
            }
            if (key == FL_F + 1) {
                showAppHelp();
                return true;
            }
            if (!editing) {
                if (key == FL_Delete || key == FL_BackSpace) {
                    deleteSelection();
                    return true;
                }
                if (key == FL_Left || key == FL_Right || key == FL_Up || key == FL_Down) {
                    nudgeObject(key, Fl::event_state(FL_SHIFT) ? 10 : 1);
                    return true;
                }
                if (key == ' ') {
                    togglePlay();
                    return true;
                }
            }
            return false;
        };
        window->callback(
            [](Fl_Widget *, void *data) {
                auto &e = *static_cast<Editor *>(data);
                try {
                    if (e.exporting) {
                        fl_alert("Cancel the export before closing.");
                        return;
                    }
                    if (e.dirty) {
                        int a = fl_choice("Save project changes before closing?", "Cancel",
                                          "Discard", "Save");
                        if (a == 0)
                            return;
                        if (a == 2 && !e.save(false))
                            return;
                    }
                    e.window->hide();
                } catch (const std::exception &ex) {
                    fl_alert("%s", ex.what());
                }
            },
            this);
        activateSelect();
    }
    void newProject() {
        if (dirty) {
            int answer = fl_choice("Save this project before creating a new one?", "Cancel",
                                   "Discard", "Save");
            if (answer == 0)
                return;
            if (answer == 2 && !save(false))
                return;
        }
        stopPreview();
        project = Project{};
        auto first = blankSlide(project.width, project.height, {255, 255, 255});
        first.title = "Slide 1";
        project.slides.push_back(std::move(first));
        slide = 0;
        object = -1;
        transition = 0;
        connection = -1;
        undoStack.clear();
        redoStack.clear();
        objectMap->value("");
        selection = cv::Mat{};
        snapshotDirty = true;
        dirty = false;
        ++revision;
        refresh();
        activateSelect();
        canvas->reset();
        window->label("FrameMorph 0.8 - Slide studio");
    }
    void activateSelect() {
        tool->value(4);
        placement = 0;
        selection = cv::Mat{};
        stopPreview();
        if (selectToolButton) {
            selectToolButton->color(fl_rgb_color(220, 232, 255));
            selectToolButton->redraw();
        }
        if (hint)
            hint->copy_label(
                "Click to select. Drag to move. Drag a handle to resize; Shift keeps proportions.");
        if (canvas)
            canvas->redraw();
    }
    void activateRegion(int mode) {
        stopPreview();
        tool->value(mode);
        placement = 0;
        object = -1;
        objectName->value("");
        drawTab->show();
        selectToolButton->color(surface);
        syncProperties();
        propertiesScroll->scroll_to(0, 790);
        hint->copy_label(
            "Click a color region. Shift adds pieces; Alt subtracts. Create object when ready.");
    }
    void armInsert(int kind) {
        if (!currentSlide())
            addBlank();
        stopPreview();
        selection = cv::Mat{};
        object = -1;
        drawDraft = json::object();
        drawType->value(kind);
        drawName->value("");
        drawX->value("300");
        drawY->value("300");
        drawW->value("320");
        drawH->value("220");
        drawAngle->value("0");
        drawOpacity->value("1");
        drawFont->value("56");
        drawStroke->value("4");
        drawColor->value("#386fe8");
        drawFilled->value(kind != 4 && kind != 5);
        drawContent->value(kind == 6    ? "Text"
                           : kind == 7  ? "x^2+y^2=r^2"
                           : kind == 11 ? "x^2+y^2=1"
                                        : "y=sin(x)");
        plotRange->value(kind == 11 ? "-1.5,1.5,-1.5,1.5" : "-2*pi,2*pi,-1.5,1.5");
        tool->value(4);
        placement = 1;
        selectToolButton->color(surface);
        selectToolButton->redraw();
        drawTab->show();
        syncProperties();
        hint->copy_label(
            "Drag on the slide to draw. Click once for the default size. Escape cancels.");
        canvas->window()->cursor(FL_CURSOR_CROSS);
        canvas->redraw();
    }
    void selectObject(int index, bool reveal = true, bool additive = false) {
        stopPreview();
        if (reveal)
            propertiesScroll->scroll_to(0, 0);
        object = index;
        slideFocus = false;
        objectMap->value("");
        if (!additive)
            for (int i = 1; i <= objects->size(); ++i)
                objects->select(i, 0);
        if (currentObject()) {
            objects->select(index + 1);
            objects->value(index + 1);
        }
        selection = cv::Mat{};
        syncProperties();
        selectObjectConnection();
        if (reveal)
            drawTab->show();
        canvas->redraw();
    }
    void focusObjectContent() {
        drawTab->show();
        propertiesScroll->scroll_to(0, drawContent->y() + propertiesScroll->yposition() -
                                           propertiesScroll->y() - 60);
        Fl::focus(drawContent);
        drawContent->position(std::strlen(drawContent->value()), 0);
    }
    void selectObjectConnection() {
        if (!currentObject() || project.transitions.empty())
            return;
        auto &tr = project.transitions[transition];
        for (int i = 0; i < (int)tr.links.size(); ++i) {
            auto &link = tr.links[i];
            auto &ids = tr.from == currentSlide()->id ? link.sources : link.targets;
            if (std::find(ids.begin(), ids.end(), currentObject()->id) != ids.end()) {
                connection = i;
                connections->value(i + 1);
                refreshMotion();
                break;
            }
        }
    }
    void syncProperties() {
        if (!propertyTitle || !currentSlide())
            return;
        bool hadUpdating = updating;
        updating = true;
        auto *o = currentObject();
        if (objectProperties->visible() != bool(o))
            propertiesScroll->scroll_to(0, 0);
        if (o) {
            objectProperties->show();
            propertyTitle->copy_label("Object properties");
            auto native =
                o->drawing.contains("construction") ? o->drawing["construction"] : o->drawing;
            std::string kind = native.is_null() ? "pixels" : native.value("kind", "image");
            auto count = selectedIndices().size();
            propertyKind->copy_label(
                (count > 1 ? std::to_string(count) + " selected / drag moves selection"
                           : kind + " / drag handles to resize")
                    .c_str());
            objectName->value(o->name.c_str());
            animateObject->value(participating(*o));
            auto f = objectFrame(*o);
            propX->value(number(f["x"]).c_str());
            propY->value(number(f["y"]).c_str());
            propW->value(number(f["w"]).c_str());
            propH->value(number(f["h"]).c_str());
            propAngle->value(number(f["angle"]).c_str());
            propOpacity->value(number(f["opacity"]).c_str());
            selectedBounds = cv::boundingRect(o->mask);
            objectLayer->value(number(o->layer).c_str());
            if (!native.is_null())
                loadDrawing(false);
            else {
                drawDraft = json::object();
                drawContent->value("");
            }
            bool tint = kind == "image" || kind == "svg" || kind == "code" || kind == "group";
            drawColor->copy_label(tint ? "Tint (white = original)" : "Color");
            if (tint)
                drawColor->value(o->imageEdits.is_object()
                                     ? o->imageEdits.value("tint", "#ffffff").c_str()
                                     : "#ffffff");
            bool text = kind == "text" || kind == "equation" || kind == "code" || kind == "svg",
                 math = kind == "plot" || kind == "implicit";
            if (text || math)
                drawContent->activate();
            else
                drawContent->deactivate();
            if (text)
                drawFont->activate();
            else
                drawFont->deactivate();
            if (math)
                plotRange->activate();
            else
                plotRange->deactivate();
            if (!text && kind != "image" && kind != "pixels")
                drawStroke->activate();
            else
                drawStroke->deactivate();
            if (kind == "pixels" && o->drawing.is_null()) {
                auto c = cv::mean(currentSlide()->image, o->mask);
                char hex[8];
                std::snprintf(hex, sizeof(hex), "#%02x%02x%02x", int(c[2] + .5), int(c[1] + .5),
                              int(c[0] + .5));
                drawColor->value(hex);
            }
            slideProperties->hide();
            int effect = 0;
            for (auto &entry : lifecycleEffects()) {
                if (entry.first == o->constructAnimation)
                    constructEffect->value(effect);
                if (entry.first == o->destructAnimation)
                    destructEffect->value(effect);
                ++effect;
            }
            effectStart->value(number(o->effectStart * 100).c_str());
            effectEnd->value(number(o->effectEnd * 100).c_str());
            effectEasing->value(o->effectEasing == "linear"     ? 1
                                : o->effectEasing == "ease_in"  ? 2
                                : o->effectEasing == "ease_out" ? 3
                                                                : 0);
        } else {
            objectProperties->hide();
            selectedBounds = {};
            propertyTitle->copy_label("Frame properties");
            propertyKind->copy_label("Select an object, or insert one from the ribbon.");
            slideProperties->show();
        }
        slideTitle->value(currentSlide()->title.c_str());
        hold->value(number(currentSlide()->hold).c_str());
        stepDuration->value(number(currentSlide()->stepDuration).c_str());
        updating = hadUpdating;
        propertiesScroll->redraw();
        syncUpdaters();
    }
    void commitCandidate(Object candidate, Slide &temporary) {
        checkpoint();
        currentSlide()->baseImage = temporary.baseImage;
        currentSlide()->objects[object] = std::move(candidate);
        recomposeSlide(*currentSlide());
        selection = cv::Mat{};
        changed();
        refreshObjects();
        syncProperties();
    }
    void commitFrame(const json &frame) {
        if (!currentObject())
            return;
        if (frame == objectFrame(*currentObject()))
            return;
        Object candidate = *currentObject();
        Slide temporary = *currentSlide();
        setObjectFrame(temporary, candidate, frame);
        commitCandidate(std::move(candidate), temporary);
    }
    void applyObjectGeometry() {
        if (updating || !currentObject())
            return;
        json f = {{"x", valueOf(propX)},         {"y", valueOf(propY)},
                  {"w", valueOf(propW)},         {"h", valueOf(propH)},
                  {"angle", valueOf(propAngle)}, {"opacity", valueOf(propOpacity)}};
        auto old = objectFrame(*currentObject());
        double delta = 0;
        for (auto key : {"x", "y", "w", "h", "angle", "opacity"})
            delta += std::abs(f[key].get<double>() - old[key].get<double>());
        if (delta < .001)
            return;
        commitFrame(f);
    }
    void applyObjectColor() {
        if (updating || !currentObject())
            return;
        if (currentObject()->drawing.is_object() &&
            currentObject()->drawing.value("color", "") == drawColor->value())
            return;
        Object candidate = *currentObject();
        Slide temporary = *currentSlide();
        setObjectColor(temporary, candidate, drawColor->value());
        commitCandidate(std::move(candidate), temporary);
    }
    void applyConstruction() {
        if (updating || !currentObject() || currentObject()->drawing.is_null())
            return;
        auto kind = drawDraft.value("kind", "");
        if (kind == "pixels" || kind == "image")
            return;
        Object candidate = *currentObject();
        Slide temporary = *currentSlide();
        auto spec = drawingSpec();
        window->cursor(FL_CURSOR_WAIT);
        Fl::flush();
        try {
            editObjectConstruction(temporary, candidate, spec);
            window->cursor(FL_CURSOR_DEFAULT);
        } catch (...) {
            window->cursor(FL_CURSOR_DEFAULT);
            throw;
        }
        commitCandidate(std::move(candidate), temporary);
    }
    std::vector<int> selectedIndices() const {
        std::vector<int> ids;
        for (int i = 1; i <= objects->size(); ++i)
            if (objects->selected(i))
                ids.push_back(i - 1);
        if (ids.empty() && object >= 0)
            ids.push_back(object);
        return ids;
    }
    void commitFrames(const std::vector<std::pair<int, json>> &frames) {
        if (!currentSlide() || frames.empty())
            return;
        bool modified = false;
        Slide temporary = *currentSlide();
        // Extract every selected object before changing any raster content.
        for (auto &[i, f] : frames) {
            modified |= objectFrame(temporary.objects.at(i)) != f;
            makeObjectEditable(temporary, temporary.objects.at(i));
        }
        if (!modified)
            return;
        for (auto &[i, f] : frames)
            setObjectFrame(temporary, temporary.objects.at(i), f);
        recomposeSlide(temporary);
        auto ids = selectedIndices();
        checkpoint();
        *currentSlide() = std::move(temporary);
        selection = cv::Mat{};
        changed();
        refreshObjects();
        for (int i : ids)
            objects->select(i + 1);
        syncProperties();
        canvas->redraw();
    }
    void translateSelection(double dx, double dy) {
        std::vector<std::pair<int, json>> frames;
        for (int i : selectedIndices()) {
            auto f = objectFrame(currentSlide()->objects[i]);
            f["x"] = f["x"].get<double>() + dx;
            f["y"] = f["y"].get<double>() + dy;
            frames.emplace_back(i, std::move(f));
        }
        commitFrames(frames);
    }
    cv::Rect2d frameBounds(const json &f) const {
        double w = f["w"], h = f["h"], a = f["angle"].get<double>() * CV_PI / 180;
        double bw = std::abs(w * std::cos(a)) + std::abs(h * std::sin(a)),
               bh = std::abs(w * std::sin(a)) + std::abs(h * std::cos(a));
        return {f["x"].get<double>() + (w - bw) / 2, f["y"].get<double>() + (h - bh) / 2, bw, bh};
    }
    void alignObject(bool horizontal) {
        if (!currentObject())
            return;
        cv::Rect2d all;
        bool first = true;
        for (int i : selectedIndices()) {
            auto b = frameBounds(objectFrame(currentSlide()->objects[i]));
            all = first ? b : all | b;
            first = false;
        }
        translateSelection(horizontal ? (project.width - all.width) / 2 - all.x : 0,
                           horizontal ? 0 : (project.height - all.height) / 2 - all.y);
    }
    void arrangeSelection(int mode) {
        auto ids = selectedIndices();
        if (ids.size() < 2) {
            message("Select two or more objects with Shift-click or a marquee.");
            return;
        }
        std::vector<std::pair<int, json>> frames;
        cv::Rect2d all;
        bool first = true;
        for (int i : ids) {
            auto f = objectFrame(currentSlide()->objects[i]);
            auto b = frameBounds(f);
            all = first ? b : all | b;
            first = false;
            frames.emplace_back(i, std::move(f));
        }
        if (mode >= 4) {
            if (ids.size() < 3) {
                message("Select at least three objects to distribute their spacing.");
                return;
            }
            bool horizontal = mode == 4;
            std::stable_sort(frames.begin(), frames.end(), [&](auto &a, auto &b) {
                auto x = frameBounds(a.second), y = frameBounds(b.second);
                return horizontal ? x.x + x.width / 2 < y.x + y.width / 2
                                  : x.y + x.height / 2 < y.y + y.height / 2;
            });
            auto first = frameBounds(frames.front().second),
                 last = frameBounds(frames.back().second);
            double start = horizontal ? first.x : first.y,
                   end = horizontal ? last.x + last.width : last.y + last.height, sum = 0;
            for (auto &[i, f] : frames) {
                auto b = frameBounds(f);
                sum += horizontal ? b.width : b.height;
            }
            double gap = (end - start - sum) / (frames.size() - 1), cursor = start;
            for (auto &[i, f] : frames) {
                auto b = frameBounds(f);
                auto axis = horizontal ? "x" : "y";
                f[axis] = f[axis].get<double>() + cursor - (horizontal ? b.x : b.y);
                cursor += (horizontal ? b.width : b.height) + gap;
            }
        } else
            for (auto &[i, f] : frames) {
                auto b = frameBounds(f);
                auto axis = mode < 2 ? "x" : "y";
                double delta = mode == 0   ? all.x - b.x
                               : mode == 1 ? all.x + all.width - b.x - b.width
                               : mode == 2 ? all.y - b.y
                                           : all.y + all.height - b.y - b.height;
                f[axis] = f[axis].get<double>() + delta;
            }
        commitFrames(frames);
    }
    void nudgeObject(int key, int step) {
        if (!currentObject())
            return;
        double delta = (key == FL_Left || key == FL_Up) ? -step : step;
        translateSelection(key == FL_Left || key == FL_Right ? delta : 0,
                           key == FL_Up || key == FL_Down ? delta : 0);
    }
    void duplicateObject() {
        if (currentObject()) {
            copyDrawing();
            pasteDrawing();
        } else
            duplicateCurrent();
    }
    void chooseBackground() {
        auto *s = currentSlide();
        if (!s)
            return;
        double r = s->background[2] / 255., g = s->background[1] / 255.,
               b = s->background[0] / 255.;
        if (fl_color_chooser("Slide background", r, g, b)) {
            checkpoint();
            s->background = {uchar(b * 255), uchar(g * 255), uchar(r * 255)};
            if (s->blank) {
                s->baseImage =
                    cv::Mat(s->image.size(), CV_8UC3,
                            cv::Scalar(s->background[0], s->background[1], s->background[2]));
                recomposeSlide(*s);
            }
            changed();
        }
    }
    Slide *currentSlide() {
        return slide >= 0 && slide < (int)project.slides.size() ? &project.slides[slide] : nullptr;
    }
    Object *currentObject() {
        auto *s = currentSlide();
        return s && object >= 0 && object < (int)s->objects.size() ? &s->objects[object] : nullptr;
    }
    Link *currentLink() {
        if (transition < 0 || transition >= (int)project.transitions.size())
            return nullptr;
        auto &v = project.transitions[transition].links;
        return connection >= 0 && connection < (int)v.size() ? &v[connection] : nullptr;
    }
    void message(const std::string &text) {
        status->copy_label(text.c_str());
        status->redraw();
    }
    void checkpoint() {
        undoStack.push_back(
            {project, selection.clone(), slide, object, slideFocus, selectedIndices()});
        redoStack.clear();
        while (undoStack.size() > 16)
            undoStack.pop_front();
        auto memoryUsed = [&] {
            std::set<const uchar *> seen;
            size_t bytes = 0;
            auto add = [&](const cv::Mat &m) {
                if (!m.empty() && seen.insert(m.datastart).second)
                    bytes += m.datalimit - m.datastart;
            };
            for (auto &state : undoStack) {
                add(state.selection);
                for (auto &s : state.project.slides) {
                    add(s.image);
                    add(s.baseImage);
                    for (auto &o : s.objects) {
                        add(o.mask);
                        add(o.artwork);
                        add(o.sourceAsset);
                        add(o.originalAsset);
                    }
                }
            }
            return bytes;
        };
        while (undoStack.size() > 1 && memoryUsed() > 256 * 1024 * 1024)
            undoStack.pop_front();
    }
    void changed() {
        ++revision;
        dirty = true;
        snapshotDirty = true;
        window->label("FrameMorph 0.8  /  unsaved changes");
        scrub->maximum(std::max(.01, projectDuration(project)));
        strip->redraw();
        canvas->redraw();
        if (canvas->showingPreview)
            requestPreview(scrub->value());
    }
    void undo(bool redo) {
        auto &from = redo ? redoStack : undoStack;
        auto &to = redo ? undoStack : redoStack;
        if (from.empty())
            return;
        to.push_back({project, selection.clone(), slide, object, slideFocus, selectedIndices()});
        auto state = std::move(from.back());
        from.pop_back();
        project = std::move(state.project);
        selection = state.selection;
        slide = state.slide;
        object = state.object;
        slideFocus = state.slideFocus;
        objectMap->value("");
        changed();
        refresh();
        for (int i : state.selectedObjects)
            if (i >= 0 && i < objects->size())
                objects->select(i + 1);
        syncProperties();
        canvas->redraw();
    }
    void rematch(bool force = false) {
        reconcileTransitions(project);
        for (size_t i = 0; i < project.transitions.size(); ++i) {
            auto &tr = project.transitions[i];
            auto automatic = automaticLinks(project.slides[i], project.slides[i + 1]);
            std::vector<Link> retained;
            std::set<std::string> usedA, usedB;
            if (!force)
                for (auto old : tr.links)
                    if (old.manual) {
                        auto valid = [&](const std::string &id, const Slide &s) {
                            return std::any_of(
                                s.objects.begin(), s.objects.end(),
                                [&](const Object &o) { return o.id == id && participating(o); });
                        };
                        old.sources.erase(
                            std::remove_if(old.sources.begin(), old.sources.end(),
                                           [&](auto &id) { return !valid(id, project.slides[i]); }),
                            old.sources.end());
                        old.targets.erase(std::remove_if(old.targets.begin(), old.targets.end(),
                                                         [&](auto &id) {
                                                             return !valid(id,
                                                                           project.slides[i + 1]);
                                                         }),
                                          old.targets.end());
                        if (!old.sources.empty() || !old.targets.empty()) {
                            usedA.insert(old.sources.begin(), old.sources.end());
                            usedB.insert(old.targets.begin(), old.targets.end());
                            retained.push_back(old);
                        }
                    }
            for (auto &l : automatic) {
                l.sources.erase(std::remove_if(l.sources.begin(), l.sources.end(),
                                               [&](auto &id) { return usedA.count(id); }),
                                l.sources.end());
                l.targets.erase(std::remove_if(l.targets.begin(), l.targets.end(),
                                               [&](auto &id) { return usedB.count(id); }),
                                l.targets.end());
                if (l.sources.empty() && l.targets.empty())
                    continue;
                for (auto &old : tr.links)
                    if (!old.manual && l.sources == old.sources && l.targets == old.targets &&
                        l.name == old.name) {
                        l = old;
                        break;
                    }
                l.manual = false;
                retained.push_back(l);
            }
            tr.links = std::move(retained);
        }
    }
    void refresh() {
        updating = true;
        transition = std::clamp(transition, 0, std::max(0, (int)project.transitions.size() - 1));
        transitionChoice->clear();
        for (size_t i = 0; i < project.transitions.size(); ++i) {
            std::string title = "Slide " + std::to_string(i + 1) + " -> " + std::to_string(i + 2);
            transitionChoice->add(title.c_str());
        }
        transitionChoice->value(transition);
        scrub->maximum(std::max(.01, projectDuration(project)));
        updating = false;
        refreshObjects();
        refreshConnections();
        strip->redraw();
        canvas->redraw();
    }
    void refreshObjects() {
        updating = true;
        objects->clear();
        auto *s = currentSlide();
        if (s) {
            for (auto &o : s->objects) {
                std::string title = (participating(o) ? "* " : "  ") + o.name;
                objects->add(title.c_str());
            }
            hold->value(number(s->hold).c_str());
        }
        if (currentObject()) {
            objects->select(object + 1);
            objectName->value(currentObject()->name.c_str());
            objectLayer->value(number(currentObject()->layer).c_str());
        }
        updating = false;
        syncProperties();
    }
    void refreshConnections() {
        updating = true;
        connections->clear();
        sources->clear();
        targets->clear();
        if (!project.transitions.empty()) {
            auto &t = project.transitions[transition];
            duration->value(number(t.duration).c_str());
            for (auto &l : t.links) {
                std::string title = l.name + "  " + std::to_string(l.sources.size()) + " -> " +
                                    std::to_string(l.targets.size());
                connections->add(title.c_str());
            }
            if (connection < 0 && !t.links.empty())
                connection = 0;
            if (connection >= (int)t.links.size())
                connection = t.links.size() - 1;
            if (connection >= 0)
                connections->value(connection + 1);
            for (auto &o : project.slides[transition].objects)
                if (participating(o))
                    sources->add(
                        (o.name + " [" + o.id.substr(o.id.size() > 5 ? o.id.size() - 5 : 0) + "]")
                            .c_str());
            for (auto &o : project.slides[transition + 1].objects)
                if (participating(o))
                    targets->add(
                        (o.name + " [" + o.id.substr(o.id.size() > 5 ? o.id.size() - 5 : 0) + "]")
                            .c_str());
        }
        updating = false;
        refreshMotion();
    }
    void refreshMotion() {
        bool updaterLink = false, updaterTransition = false;
        if (transition >= 0 && transition < (int)project.transitions.size()) {
            const auto &tr = project.transitions[transition];
            for (const auto &link : tr.links)
                for (const auto &id : link.targets)
                    for (const auto &o : project.slides[transition + 1].objects)
                        if (o.id == id && o.generation.is_object() &&
                            o.generation.value("kind", "") == "updater") {
                            updaterTransition = true;
                            if (currentLink() && currentLink()->id == link.id)
                                updaterLink = true;
                        }
        }
        for (auto *widget :
             std::vector<Fl_Widget *>{easing, bend, mode, pairs, anchors, start, end}) {
            if (updaterLink) {
                widget->deactivate();
                widget->tooltip("This connection runs a time function. Edit/regenerate it from the "
                                "source object's Properties.");
            } else {
                widget->activate();
                widget->tooltip(nullptr);
            }
        }
        if (updaterTransition) {
            duration->deactivate();
            duration->tooltip("Regenerate from the source slide with the desired step duration.");
        } else {
            duration->activate();
            duration->tooltip(nullptr);
        }
        updating = true;
        if (auto *l = currentLink()) {
            functionX->value(l->offsetX.c_str());
            functionY->value(l->offsetY.c_str());
            functionScale->value(l->scaleFunction.c_str());
            functionRotation->value(l->rotationFunction.c_str());
            functionOpacity->value(l->opacityFunction.c_str());
            start->value(number(l->start * 100).c_str());
            end->value(number(l->end * 100).c_str());
            bend->value(number(l->bend).c_str());
            rotation->value(number(l->rotation).c_str());
            layer->value(number(l->layer).c_str());
            const char *names[] = {"smooth", "linear", "ease_in", "ease_out"};
            for (int i = 0; i < 4; ++i)
                if (l->easing == names[i])
                    easing->value(i);
            mode->value(l->mode == "fade" ? 1 : 0);
            std::ostringstream pp, aa;
            for (auto [a, b] : l->componentPairs) {
                if (pp.tellp() > 0)
                    pp << ", ";
                pp << a << ":" << b;
            }
            for (auto [a, b] : l->anchors) {
                if (aa.tellp() > 0)
                    aa << ", ";
                aa << a << ":" << b;
            }
            pairs->value(pp.str().c_str());
            anchors->value(aa.str().c_str());
        }
        updating = false;
    }
    void selectSlide(int i) {
        if (i < 0 || i >= (int)project.slides.size())
            return;
        propertiesScroll->scroll_to(0, 0);
        slide = i;
        slideFocus = true;
        objectMap->value("");
        object = -1;
        selection = cv::Mat();
        transition = std::clamp(i - 1, 0, std::max(0, (int)project.transitions.size() - 1));
        connection = -1;
        stopPreview();
        refresh();
    }
    void stopPreview() {
        preview.cancel();
        previewBusy = false;
        playing = false;
        playStarting = false;
        play->label("Play");
        canvas->showingPreview = false;
        canvas->redraw();
    }
    void chooseObject() {
        int v = objects->value() - 1;
        if (v < 0 || !currentSlide() || v >= (int)currentSlide()->objects.size())
            return;
        object = v;
        slideFocus = false;
        objectMap->value("");
        selection = tool->value() == 4 ? cv::Mat{} : currentObject()->mask.clone();
        objectName->value(currentObject()->name.c_str());
        objectLayer->value(number(currentObject()->layer).c_str());
        stopPreview();
        syncProperties();
        drawTab->show();
    }
    void renameObject() {
        if (auto *o = currentObject()) {
            if (o->name == objectName->value())
                return;
            checkpoint();
            o->name = std::string(objectName->value()).empty()
                          ? defaultObjectName(*currentSlide(), "object")
                          : objectName->value();
            drawName->value(o->name.c_str());
            rematch();
            changed();
            refreshObjects();
            refreshConnections();
        }
    }
    void createObject() {
        if (!currentSlide() || selection.empty() || cv::countNonZero(selection) == 0)
            throw std::runtime_error("Select image pixels before creating an object");
        checkpoint();
        Object o;
        o.id = newId();
        o.name = objectName->value();
        if (o.name.empty())
            o.name = defaultObjectName(*currentSlide(), "region");
        o.layer = std::stoi(objectLayer->value());
        o.mask = selection.clone();
        currentSlide()->objects.push_back(o);
        object = currentSlide()->objects.size() - 1;
        slideFocus = false;
        rematch();
        changed();
        refresh();
        message("Object created. Shift-click adds pieces; Update mask saves corrections.");
    }
    void updateMask() {
        if (!currentObject())
            throw std::runtime_error("Choose an existing object first");
        bool movedPixels = currentObject()->drawing.contains("construction") &&
                           currentObject()->drawing["construction"].value("kind", "") == "pixels";
        if (!currentObject()->drawing.is_null() && !movedPixels)
            throw std::runtime_error("Use Properties to edit authored objects. Mask corrections "
                                     "apply to pixel selections.");
        if (selection.empty() || !cv::countNonZero(selection))
            throw std::runtime_error("Object masks cannot be empty");
        checkpoint();
        if (movedPixels) {
            auto candidate = *currentObject();
            candidate.mask = selection.clone();
            candidate.artwork.release();
            candidate.sourceAsset.release();
            candidate.drawing = json{};
            candidate.generation = json{};
            Slide temporary = *currentSlide();
            makeObjectEditable(temporary, candidate);
            setObjectFrame(temporary, candidate, objectFrame(candidate));
            currentSlide()->baseImage = temporary.baseImage;
            *currentObject() = std::move(candidate);
            recomposeSlide(*currentSlide());
        } else
            currentObject()->mask = selection.clone();
        rematch();
        changed();
        refresh();
    }
    void deleteSelection() {
        if (slideFocus)
            removeSlide();
        else
            deleteObjects();
    }
    double valueOf(Fl_Input *input) {
        auto *s = currentSlide();
        auto vars = s ? sceneVariables(*s, currentObject()) : std::map<std::string, double>{};
        double value = compileExpression2D(input->value(), vars)(0, 0);
        if (!std::isfinite(value))
            throw std::runtime_error("Property expression must be finite");
        return value;
    }
    void deleteObjects() {
        if (!currentSlide())
            return;
        std::vector<int> selected;
        for (int i = 1; i <= objects->size(); ++i)
            if (objects->selected(i))
                selected.push_back(i - 1);
        if (selected.empty())
            return;
        checkpoint();
        stopPreview();
        for (auto i = selected.rbegin(); i != selected.rend(); ++i) {
            auto &s = *currentSlide();
            auto &o = s.objects[*i];
            if (o.artwork.empty()) {
                s.baseImage = (s.baseImage.empty() ? s.image : s.baseImage).clone();
                cv::Mat erase;
                cv::dilate(o.mask, erase, cv::Mat());
                s.baseImage.setTo(cv::Scalar(s.background[0], s.background[1], s.background[2]),
                                  erase);
            }
            s.objects.erase(s.objects.begin() + *i);
        }
        recomposeSlide(*currentSlide());
        object = -1;
        selection = cv::Mat();
        rematch();
        changed();
        refresh();
    }
    void groupObjects() {
        if (!currentSlide())
            return;
        std::vector<int> selected;
        for (int i = 1; i <= objects->size(); ++i)
            if (objects->selected(i))
                selected.push_back(i - 1);
        if (selected.size() < 2)
            throw std::runtime_error("Select at least two objects in the list");
        int authored = 0, images = 0;
        for (int i : selected) {
            if (!currentSlide()->objects[i].drawing.is_null())
                ++authored;
            if (!currentSlide()->objects[i].sourceAsset.empty())
                ++images;
        }
        if (authored && authored != (int)selected.size())
            throw std::runtime_error(
                "Group drawn objects together, or group pixel selections together");
        Object merged;
        merged.id = newId();
        merged.name = objectName->value();
        merged.layer = std::stoi(objectLayer->value());
        merged.mask = cv::Mat(project.height, project.width, CV_8UC1, cv::Scalar(0));
        for (int i : selected)
            cv::bitwise_or(merged.mask, currentSlide()->objects[i].mask, merged.mask);
        if (authored) {
            auto bounds = cv::boundingRect(merged.mask);
            json spec = {{"kind", "group"},
                         {"x", bounds.x},
                         {"y", bounds.y},
                         {"w", bounds.width},
                         {"h", bounds.height},
                         {"original_x", bounds.x},
                         {"original_y", bounds.y},
                         {"original_w", bounds.width},
                         {"original_h", bounds.height},
                         {"children", json::array()}};
            if (images) {
                merged.sourceAsset = cv::Mat(bounds.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
                auto ordered = selected;
                std::stable_sort(ordered.begin(), ordered.end(), [&](int a, int b) {
                    return currentSlide()->objects[a].layer < currentSlide()->objects[b].layer;
                });
                for (int i : ordered) {
                    auto &o = currentSlide()->objects[i];
                    for (int y = 0; y < o.artwork.rows; ++y)
                        for (int x = 0; x < o.artwork.cols; ++x) {
                            int dx = x + o.artworkOrigin.x - bounds.x,
                                dy = y + o.artworkOrigin.y - bounds.y;
                            if (dx < 0 || dy < 0 || dx >= bounds.width || dy >= bounds.height)
                                continue;
                            auto src = o.artwork.at<cv::Vec4b>(y, x);
                            auto &dst = merged.sourceAsset.at<cv::Vec4b>(dy, dx);
                            float a = src[3] / 255.f, da = dst[3] / 255.f, oa = a + da * (1 - a);
                            if (oa > 0)
                                for (int c = 0; c < 3; ++c)
                                    dst[c] = cv::saturate_cast<uchar>(
                                        (src[c] * a + dst[c] * da * (1 - a)) / oa);
                            dst[3] = cv::saturate_cast<uchar>(oa * 255);
                        }
                }
                spec["kind"] = "image";
                spec.erase("children");
            } else
                for (int i : selected)
                    spec["children"].push_back(currentSlide()->objects[i].drawing);
            Slide temporary = *currentSlide();
            renderDrawing(temporary, merged, spec);
        }
        checkpoint();
        stopPreview();
        for (auto i = selected.rbegin(); i != selected.rend(); ++i) {
            auto &s = *currentSlide();
            auto &o = s.objects[*i];
            if (o.artwork.empty()) {
                s.baseImage = (s.baseImage.empty() ? s.image : s.baseImage).clone();
                cv::Mat erase;
                cv::dilate(o.mask, erase, cv::Mat());
                s.baseImage.setTo(cv::Scalar(s.background[0], s.background[1], s.background[2]),
                                  erase);
            }
            s.objects.erase(s.objects.begin() + *i);
        }
        currentSlide()->objects.push_back(merged);
        object = currentSlide()->objects.size() - 1;
        slideFocus = false;
        recomposeSlide(*currentSlide());
        selection = merged.mask.clone();
        rematch();
        changed();
        refresh();
    }
    static const std::vector<std::string> &drawingKinds() {
        static const std::vector<std::string> kinds = {
            "rectangle", "circle",   "ellipse", "triangle", "line",   "arrow",
            "text",      "equation", "plot",    "image",    "group",  "implicit",
            "code",      "svg",      "diamond", "star",     "hexagon"};
        return kinds;
    }
    json drawingSpec() {
        json spec = drawDraft.is_object() ? drawDraft : json::object();
        spec["kind"] = drawingKinds().at(drawType->value());
        spec["content"] = drawContent->value();
        spec["color"] = drawColor->value();
        std::vector<std::pair<const char *, Fl_Input *>> fields = {
            {"x", drawX},           {"y", drawY},
            {"w", drawW},           {"h", drawH},
            {"stroke", drawStroke}, {"font_size", drawFont},
            {"angle", drawAngle},   {"opacity", drawOpacity}};
        for (auto [key, input] : fields) {
            double value = valueOf(input);
            if (!std::isfinite(value))
                throw std::runtime_error(std::string("Drawing expression is not finite: ") + key);
            spec[key] = value;
            spec["expressions"][key] = input->value();
        }
        spec["filled"] = bool(drawFilled->value());
        spec["axes"] = bool(drawFilled->value());
        if (spec["kind"] == "plot" || spec["kind"] == "implicit") {
            std::istringstream in(plotRange->value());
            std::string part;
            std::vector<double> values;
            while (std::getline(in, part, ','))
                values.push_back(evaluateExpression(part, 0));
            if (values.size() != 4)
                throw std::runtime_error("Plot range needs xmin,xmax,ymin,ymax");
            spec["x_min"] = values[0];
            spec["x_max"] = values[1];
            spec["y_min"] = values[2];
            spec["y_max"] = values[3];
        }
        return spec;
    }
    void loadDrawing(bool showTab = true) {
        auto *o = currentObject();
        if (!o || o->drawing.is_null())
            throw std::runtime_error("Select an object on the canvas first");
        drawDraft = o->drawing.contains("construction") ? o->drawing["construction"] : o->drawing;
        auto kind = drawDraft.value("kind", "rectangle");
        auto &kinds = drawingKinds();
        auto found = std::find(kinds.begin(), kinds.end(), kind);
        drawType->value(found == kinds.end() ? 0 : found - kinds.begin());
        drawName->value(o->name.c_str());
        drawContent->value(drawDraft.value("content", "").c_str());
        drawColor->value(drawDraft.value("color", "#4ed5b8").c_str());
        std::vector<std::tuple<const char *, Fl_Input *, double>> fields = {
            {"x", drawX, 200},       {"y", drawY, 200},          {"w", drawW, 320},
            {"h", drawH, 200},       {"stroke", drawStroke, 4},  {"font_size", drawFont, 56},
            {"angle", drawAngle, 0}, {"opacity", drawOpacity, 1}};
        for (auto [key, input, fallback] : fields) {
            std::string value =
                drawDraft.contains("expressions") && drawDraft["expressions"].contains(key)
                    ? drawDraft["expressions"][key].get<std::string>()
                    : number(drawDraft.value(key, fallback));
            input->value(value.c_str());
        }
        drawFilled->value(kind == "plot" ? drawDraft.value("axes", true)
                                         : drawDraft.value("filled", true));
        std::string range = number(drawDraft.value("x_min", -6.283)) + "," +
                            number(drawDraft.value("x_max", 6.283)) + "," +
                            number(drawDraft.value("y_min", -1.5)) + "," +
                            number(drawDraft.value("y_max", 1.5));
        plotRange->value(range.c_str());
        if (showTab)
            drawTab->show();
    }
    void addBlank() {
        checkpoint();
        auto s = blankSlide(project.width, project.height, {255, 255, 255});
        s.title = "Blank " + std::to_string(project.slides.size() + 1);
        int index = currentSlide() ? slide + 1 : project.slides.size();
        project.slides.insert(project.slides.begin() + index, std::move(s));
        reconcileTransitions(project);
        changed();
        selectSlide(index);
        drawTab->show();
        message("New slide. Choose a tool in the ribbon, then draw on the canvas.");
    }
    void duplicateCurrent() {
        if (!currentSlide())
            throw std::runtime_error("Choose a slide to duplicate");
        checkpoint();
        auto duplicate = duplicateSlide(*currentSlide());
        project.slides.insert(project.slides.begin() + slide + 1, std::move(duplicate));
        reconcileTransitions(project);
        changed();
        selectSlide(slide + 1);
        message("Slide duplicated with animation names preserved. Change this slide to define the "
                "next pose.");
    }
    void addDrawing() {
        if (!currentSlide())
            addBlank();
        Object candidate;
        candidate.id = newId();
        candidate.name = drawName->value();
        if (candidate.name.empty())
            candidate.name =
                defaultObjectName(*currentSlide(), drawingKinds().at(drawType->value()));
        candidate.layer = currentSlide()->objects.empty()
                              ? 0
                              : std::max_element(
                                    currentSlide()->objects.begin(), currentSlide()->objects.end(),
                                    [](auto &a, auto &b) {
                                        return a.layer < b.layer;
                                    })->layer +
                                    1;
        if (drawType->value() == 9 && currentObject())
            candidate.sourceAsset = currentObject()->sourceAsset;
        Slide temporary = *currentSlide();
        renderDrawing(temporary, candidate, drawingSpec());
        checkpoint();
        if (currentSlide()->baseImage.empty())
            currentSlide()->baseImage = temporary.baseImage;
        currentSlide()->objects.push_back(candidate);
        object = currentSlide()->objects.size() - 1;
        slideFocus = false;
        recomposeSlide(*currentSlide());
        selection = candidate.mask.clone();
        rematch();
        changed();
        refresh();
        loadDrawing(false);
        message("Drawing created. Duplicate the slide, then edit its next pose.");
    }
    void updateDrawing() {
        auto *o = currentObject();
        if (!o || o->drawing.is_null())
            throw std::runtime_error(
                "Select a drawing to edit, or click Add drawing to create one");
        Object candidate = *o;
        candidate.generation = json{};
        candidate.name = std::string(drawName->value()).empty()
                             ? defaultObjectName(*currentSlide(), "object")
                             : drawName->value();
        Slide temporary = *currentSlide();
        editObjectConstruction(temporary, candidate, drawingSpec());
        checkpoint();
        currentSlide()->objects[object] = candidate;
        recomposeSlide(*currentSlide());
        selection = candidate.mask.clone();
        rematch();
        changed();
        refresh();
        loadDrawing(false);
    }
    void commitMediaAsset(const AssetEditResult &result, bool editing = false) {
        if (!currentSlide())
            addBlank();
        Slide temporary = *currentSlide();
        Object candidate;
        json spec = result.spec;
        auto kind = spec.value("kind", "equation");
        if (editing && currentObject())
            candidate = *currentObject();
        else {
            candidate.id = newId();
            candidate.name = defaultObjectName(*currentSlide(), kind);
            for (auto &o : currentSlide()->objects)
                candidate.layer = std::max(candidate.layer, o.layer + 1);
        }
        auto frame = editing ? objectFrame(candidate)
                             : json{{"x", (project.width - result.pixels.cols * .7) / 2},
                                    {"y", (project.height - result.pixels.rows * .7) / 2},
                                    {"angle", 0},
                                    {"opacity", 1}};
        double scale = std::min({1., project.width * .8 / result.pixels.cols,
                                 project.height * .7 / result.pixels.rows});
        if (editing && candidate.drawing.contains("construction")) {
            const auto &natural =
                candidate.originalAsset.empty() ? candidate.sourceAsset : candidate.originalAsset;
            frame["w"] = objectFrame(candidate)["w"].get<double>() * result.pixels.cols /
                         std::max(1, natural.cols);
            frame["h"] = objectFrame(candidate)["h"].get<double>() * result.pixels.rows /
                         std::max(1, natural.rows);
        } else {
            frame["w"] = result.pixels.cols * scale;
            frame["h"] = result.pixels.rows * scale;
        }
        if (!editing) {
            frame["x"] = (project.width - frame["w"].get<double>()) / 2;
            frame["y"] = (project.height - frame["h"].get<double>()) / 2;
        }
        frame["kind"] = "image";
        frame["construction"] = spec;
        frame["color"] = spec.value("color", "#1d293d");
        candidate.sourceAsset = result.pixels;
        candidate.originalAsset.release();
        candidate.imageEdits = json{};
        candidate.generation = json{};
        renderDrawing(temporary, candidate, frame);
        if (editing && currentObject())
            commitCandidate(std::move(candidate), temporary);
        else {
            checkpoint();
            currentSlide()->baseImage = temporary.baseImage;
            currentSlide()->objects.push_back(std::move(candidate));
            recomposeSlide(*currentSlide());
            object = currentSlide()->objects.size() - 1;
            slideFocus = false;
            rematch();
            changed();
            refresh();
        }
        selectObject(object);
    }
    void openMediaEditor(const std::string &kind, bool editing) {
        stopPreview();
        json initial = {
            {"kind", kind}, {"font_size", kind == "code" ? 28 : 56}, {"color", "#1d293d"}};
        if (editing && currentObject())
            initial = currentObject()->drawing.contains("construction")
                          ? currentObject()->drawing["construction"]
                          : currentObject()->drawing;
        if (!initial.contains("content"))
            initial["content"] = kind == "equation" ? R"(e^{i\pi}+1=0)"
                                 : kind == "code"
                                     ? "def square(x):\n    return x ** 2"
                                     : "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"300\" "
                                       "height=\"180\"><rect x=\"5\" y=\"5\" width=\"290\" "
                                       "height=\"170\" rx=\"24\" fill=\"#386fe8\"/></svg>";
        auto result = editAsset(kind, initial);
        if (result)
            commitMediaAsset(*result, editing);
    }
    void editSelectedContent() {
        if (!currentObject())
            throw std::runtime_error("Select an object first");
        auto native = currentObject()->drawing.contains("construction")
                          ? currentObject()->drawing["construction"]
                          : currentObject()->drawing;
        auto kind = native.is_object() ? native.value("kind", "") : "pixels";
        if (kind == "equation" || kind == "code" || kind == "svg")
            openMediaEditor(kind, true);
        else if (kind == "text" || kind == "plot" || kind == "implicit")
            focusObjectContent();
        else
            openImageEditor();
    }
    void commitImageEdit(const AssetEditResult &result) {
        if (!currentObject())
            throw std::runtime_error("Select an image or object first");
        Object candidate = *currentObject();
        Slide temporary = *currentSlide();
        makeObjectEditable(temporary, candidate);
        if (candidate.originalAsset.empty())
            candidate.originalAsset = candidate.sourceAsset.clone();
        candidate.sourceAsset = result.pixels;
        candidate.imageEdits = result.spec;
        candidate.generation = json{};
        auto spec = candidate.drawing;
        spec["w"] = result.pixels.cols;
        spec["h"] = result.pixels.rows;
        renderDrawing(temporary, candidate, spec);
        commitCandidate(std::move(candidate), temporary);
    }
    void openImageEditor() {
        if (!currentObject())
            throw std::runtime_error("Select an image or object first");
        stopPreview();
        Object candidate = *currentObject();
        Slide temporary = *currentSlide();
        makeObjectEditable(temporary, candidate);
        auto source =
            candidate.originalAsset.empty() ? candidate.sourceAsset : candidate.originalAsset;
        auto initial = candidate.imageEdits.is_object()
                           ? candidate.imageEdits
                           : json{{"width", int(objectFrame(candidate)["w"])},
                                  {"height", int(objectFrame(candidate)["h"])}};
        initial["width"] = int(objectFrame(candidate)["w"]);
        initial["height"] = int(objectFrame(candidate)["h"]);
        auto result = editAsset("image", initial, source);
        if (result)
            commitImageEdit(*result);
    }
    void combineSelectedImages(const std::string &layout) {
        if (!currentSlide())
            return;
        std::vector<std::string> ids;
        for (int i = 1; i <= objects->size(); ++i)
            if (objects->selected(i))
                ids.push_back(currentSlide()->objects[i - 1].id);
        auto combined = combineImageObjects(*currentSlide(), ids, layout);
        checkpoint();
        auto &s = *currentSlide();
        for (auto &o : s.objects)
            if (std::find(ids.begin(), ids.end(), o.id) != ids.end() && o.artwork.empty()) {
                s.baseImage = (s.baseImage.empty() ? s.image : s.baseImage).clone();
                cv::Mat erase;
                cv::dilate(o.mask, erase, cv::Mat());
                s.baseImage.setTo(cv::Scalar(s.background[0], s.background[1], s.background[2]),
                                  erase);
            }
        s.objects.erase(std::remove_if(s.objects.begin(), s.objects.end(),
                                       [&](auto &o) {
                                           return std::find(ids.begin(), ids.end(), o.id) !=
                                                  ids.end();
                                       }),
                        s.objects.end());
        s.objects.push_back(std::move(combined));
        object = s.objects.size() - 1;
        slideFocus = false;
        recomposeSlide(s);
        rematch();
        changed();
        refresh();
        selectObject(object);
    }
    void editSlideImage() {
        if (!currentSlide())
            return;
        stopPreview();
        auto &s = *currentSlide();
        if (std::any_of(s.objects.begin(), s.objects.end(),
                        [](auto &o) { return o.artwork.empty(); }))
            throw std::runtime_error(
                "This slide has extracted pixel objects. Edit those objects individually, or undo "
                "their extraction before editing the complete background.");
        cv::Mat rgba;
        cv::cvtColor(s.baseImage.empty() ? s.image : s.baseImage, rgba, cv::COLOR_BGR2BGRA);
        auto result =
            editAsset("image", json{{"width", project.width}, {"height", project.height}}, rgba);
        if (!result)
            return;
        Slide temporary = s;
        temporary.baseImage = cv::Mat(
            s.image.size(), CV_8UC3, cv::Scalar(s.background[0], s.background[1], s.background[2]));
        Object o;
        o.id = newId();
        o.name = defaultObjectName(s, "slide_image");
        o.layer = -10000;
        o.sourceAsset = result->pixels;
        o.originalAsset = rgba;
        o.imageEdits = result->spec;
        double scale = std::min(double(project.width) / result->pixels.cols,
                                double(project.height) / result->pixels.rows);
        renderDrawing(temporary, o,
                      {{"kind", "image"},
                       {"x", (project.width - result->pixels.cols * scale) / 2},
                       {"y", (project.height - result->pixels.rows * scale) / 2},
                       {"w", result->pixels.cols * scale},
                       {"h", result->pixels.rows * scale}});
        checkpoint();
        s.baseImage = temporary.baseImage;
        s.objects.push_back(std::move(o));
        object = s.objects.size() - 1;
        slideFocus = false;
        recomposeSlide(s);
        rematch();
        changed();
        refresh();
        selectObject(object);
    }
    void insertImage() {
        if (!currentSlide())
            addBlank();
        Fl_Native_File_Chooser chooser;
        chooser.title("Insert an image layer");
        chooser.type(Fl_Native_File_Chooser::BROWSE_FILE);
        chooser.filter("Images\t*.{png,jpg,jpeg,svg,PNG,JPG,JPEG,SVG}");
        if (chooser.show() != 0)
            return;
        auto path = fs::path(chooser.filename());
        auto extension = path.extension().string();
        if (extension == ".svg" || extension == ".SVG") {
            if (fs::file_size(path) > 2000000)
                throw std::runtime_error("SVG is larger than 2 MB");
            std::ifstream f(path);
            std::string text((std::istreambuf_iterator<char>(f)), {});
            json spec = {{"kind", "svg"}, {"content", text}, {"svg_size", 1200}};
            auto pixels = renderMedia(spec);
            commitMediaAsset({spec, pixels}, false);
            return;
        }
        auto image = cv::imread(chooser.filename(), cv::IMREAD_UNCHANGED);
        if (image.empty())
            throw std::runtime_error("Cannot read image");
        double scale = std::min({1., 600. / image.cols, 420. / image.rows});
        drawDraft = json::object();
        drawDraft["source"] = chooser.filename();
        drawType->value(9);
        drawName->value("");
        drawX->value(number((project.width - image.cols * scale) / 2).c_str());
        drawY->value(number((project.height - image.rows * scale) / 2).c_str());
        drawW->value(number(image.cols * scale).c_str());
        drawH->value(number(image.rows * scale).c_str());
        drawAngle->value("0");
        drawOpacity->value("1");
        drawTab->show();
        addDrawing();
    }
    void alignDrawing(bool horizontal) {
        loadDrawing(false);
        double value = horizontal ? (project.width - valueOf(drawW)) / 2
                                  : (project.height - valueOf(drawH)) / 2;
        (horizontal ? drawX : drawY)->value(number(value).c_str());
        updateDrawing();
    }
    void arrangeLayer(bool front) {
        auto *o = currentObject();
        if (!o)
            return;
        int value = o->layer;
        for (auto &other : currentSlide()->objects)
            value = front ? std::max(value, other.layer) : std::min(value, other.layer);
        checkpoint();
        currentObject()->layer = value + (front ? 1 : -1);
        recomposeSlide(*currentSlide());
        changed();
        refresh();
    }
    void copyDrawing() {
        auto *o = currentObject();
        if (!o)
            throw std::runtime_error("Select an object first");
        clipboard = *o;
        clipboard->mask = o->mask.clone();
        clipboard->artwork = o->artwork.clone();
        clipboard->sourceAsset = o->sourceAsset.clone();
        clipboard->originalAsset = o->originalAsset.clone();
        if (o->drawing.is_null()) {
            auto r = cv::boundingRect(o->mask);
            cv::cvtColor(currentSlide()->image(r), clipboard->sourceAsset, cv::COLOR_BGR2BGRA);
            for (int y = 0; y < r.height; ++y)
                for (int x = 0; x < r.width; ++x)
                    clipboard->sourceAsset.at<cv::Vec4b>(y, x)[3] =
                        o->mask.at<uchar>(r.y + y, r.x + x);
            clipboard->drawing = {{"kind", "image"}, {"x", r.x},      {"y", r.y},   {"w", r.width},
                                  {"h", r.height},   {"opacity", 1.}, {"angle", 0.}};
        }
        message("Copied object. Paste it on this or another keyframe.");
    }
    void pasteDrawing() {
        if (!clipboard)
            throw std::runtime_error("Copy an object first");
        if (!currentSlide())
            addBlank();
        Object candidate = *clipboard;
        candidate.id = newId();
        bool sameName = std::any_of(currentSlide()->objects.begin(), currentSlide()->objects.end(),
                                    [&](auto &o) { return o.name == candidate.name; });
        auto frame = objectFrame(candidate);
        if (sameName) {
            auto native = candidate.drawing.contains("construction")
                              ? candidate.drawing["construction"]
                              : candidate.drawing;
            candidate.name = defaultObjectName(*currentSlide(), native.value("kind", "object"));
            frame["x"] = frame["x"].get<double>() + 24;
            frame["y"] = frame["y"].get<double>() + 24;
        }
        Slide temporary = *currentSlide();
        setObjectFrame(temporary, candidate, frame);
        checkpoint();
        if (currentSlide()->baseImage.empty())
            currentSlide()->baseImage = temporary.baseImage;
        currentSlide()->objects.push_back(candidate);
        object = currentSlide()->objects.size() - 1;
        slideFocus = false;
        recomposeSlide(*currentSlide());
        selection = candidate.mask.clone();
        rematch();
        changed();
        refresh();
        loadDrawing();
    }
    void applyLifecycle() {
        if (updating || !currentObject())
            return;
        auto construct = lifecycleEffects().at(constructEffect->value()).first,
             destruct = lifecycleEffects().at(destructEffect->value()).first;
        double begin = valueOf(effectStart) / 100., finish = valueOf(effectEnd) / 100.;
        if (!std::isfinite(begin) || !std::isfinite(finish) || begin < 0 || finish > 1 ||
            begin >= finish)
            throw std::runtime_error("Effect timing must satisfy 0 <= start < end <= 100%");
        static const char *curves[]{"smooth", "linear", "ease_in", "ease_out"};
        std::string curve = curves[effectEasing->value()];
        if (construct == currentObject()->constructAnimation &&
            destruct == currentObject()->destructAnimation &&
            begin == currentObject()->effectStart && finish == currentObject()->effectEnd &&
            curve == currentObject()->effectEasing)
            return;
        checkpoint();
        currentObject()->constructAnimation = construct;
        currentObject()->destructAnimation = destruct;
        currentObject()->effectStart = begin;
        currentObject()->effectEnd = finish;
        currentObject()->effectEasing = curve;
        rematch();
        changed();
    }
    void applyFrameStep() {
        if (updating || !currentSlide())
            return;
        double step = valueOf(stepDuration);
        if (!std::isfinite(step) || step <= 0 || step > 3600)
            throw std::runtime_error(
                "Time step must be greater than zero and at most 3600 seconds");
        if (step == currentSlide()->stepDuration)
            return;
        bool committed = false;
        if (slide < int(project.transitions.size())) {
            const auto &destination = project.slides[slide + 1];
            for (const auto &o : destination.objects)
                if (o.generation.is_object() && o.generation.value("kind", "") == "updater")
                    committed = true;
        }
        checkpoint();
        currentSlide()->stepDuration = step;
        if (!committed && slide < int(project.transitions.size()))
            project.transitions[slide].duration = step;
        changed();
        if (committed)
            message("Time step saved for the next generation. Create a new slide to regenerate "
                    "updater timing.");
        refreshConnections();
    }
    void syncUpdaters() {
        if (!updaterInput)
            return;
        updaterList->clear();
        if (auto *s = currentSlide())
            for (const auto &o : s->objects)
                if (o.updater.is_object())
                    updaterList->add(o.name.c_str());
        if (updaterList->size())
            generateUpdatersButton->activate();
        else
            generateUpdatersButton->deactivate();
        if (auto *o = currentObject()) {
            updaterObject->copy_label(("Object: " + o->name).c_str());
            updaterInput->activate();
            updaterInput->value(o->updater.is_object() ? o->updater.value("input", "").c_str()
                                                       : "");
            updaterUnits->value(
                number(o->updater.is_object() ? o->updater.value("units", 100.) : 100.).c_str());
        } else {
            updaterObject->copy_label("Select an object on the slide");
            updaterInput->value("");
            updaterInput->deactivate();
        }
    }
    void saveUpdater() {
        if (!currentObject())
            throw std::runtime_error("Select an object first");
        if (!participating(*currentObject()))
            throw std::runtime_error("Enable Animate across slides in Properties first");
        auto spec = parseObjectUpdater(updaterInput->value(), *currentSlide(), *currentObject(),
                                       valueOf(updaterUnits));
        prepareObjectUpdater(*currentSlide(), *currentObject(), spec, currentSlide()->stepDuration);
        checkpoint();
        currentObject()->updater = spec;
        changed();
        syncUpdaters();
        message("Updater saved. Add functions to other objects, then create the next slide in "
                "Frame properties.");
    }
    void clearUpdater() {
        if (!currentObject() || currentObject()->updater.is_null())
            return;
        checkpoint();
        currentObject()->updater = json{};
        changed();
        syncUpdaters();
        message("Draft updater removed. Existing generated transitions remain unchanged.");
    }
    void generateFromUpdaters() {
        if (!currentSlide())
            throw std::runtime_error("Choose a source slide first");
        std::map<std::string, json> staged;
        for (const auto &o : currentSlide()->objects)
            if (o.updater.is_object())
                staged[o.id] = o.updater;
        double step = currentSlide()->stepDuration;
        window->cursor(FL_CURSOR_WAIT);
        Fl::flush();
        Slide next;
        try {
            next = generateUpdaterKeyframe(*currentSlide(), staged, step);
        } catch (...) {
            window->cursor(FL_CURSOR_DEFAULT);
            throw;
        }
        window->cursor(FL_CURSOR_DEFAULT);
        stopPreview();
        checkpoint();
        int sourceIndex = slide;
        project.slides.insert(project.slides.begin() + slide + 1, std::move(next));
        reconcileTransitions(project);
        auto &incoming = project.transitions[sourceIndex];
        incoming.duration = step;
        for (auto &link : incoming.links)
            link.easing = "linear";
        changed();
        selectSlide(sourceIndex + 1);
        drawTab->show();
        message("Generated the endpoint and continuous updater motion for " +
                std::to_string(staged.size()) + " object(s). Press Play.");
    }
    void applyMap() {
        if (!currentObject())
            throw std::runtime_error("Select an object to map");
        auto operation = parseObjectMap(objectMap->value(), *currentSlide(), *currentObject(),
                                        valueOf(mapUnits));
        Object candidate = *currentObject();
        Slide temporary = *currentSlide();
        applyObjectMap(temporary, candidate, operation);
        stopPreview();
        commitCandidate(std::move(candidate), temporary);
        objectMap->value("");
        slideFocus = false;
        message(
            "Object updated. Enter another map, or duplicate the slide to create the next pose.");
    }
    void showVariables() {
        fl_message("PROPERTY FIELDS (pixels, X right / Y down)\n"
                   "scene.width, scene.height, scene.origin_x, scene.origin_y\n"
                   "scene.left / right / top / bottom, tau, pi, e\n"
                   "object.x / y / width / height / center_x / center_y\n\n"
                   "OBJECT MAPS (object center = 0, X right / Y up)\n"
                   "1 unit = Pixels per unit (default 100 px); px = 1 pixel\n"
                   "origin_x, origin_y = scene center in these local coordinates\n"
                   "origin = origin_x + i*origin_y (complex maps only)\n"
                   "scene.width / height, object.width / height are in map units\n"
                   "scene.left / right / top / bottom = local scene edges\n"
                   "center_x / center_y = object center measured from scene center\n\n"
                   "(x,y):x+1,y+2    move right 1 unit and up 2\n"
                   "(x,y):x+origin_x,y+origin_y    move to scene center\n"
                   "z:z*exp(i*pi/4)    rotate 45 degrees counterclockwise\n"
                   "z:z+origin    move to scene center\n"
                   "(x,y):2*x,0.5*y    matrix diag(2, 0.5)\n"
                   "(x,y):x+20*px,y    move right 20 pixels\n"
                   "Use explicit multiplication. Changes are undoable.");
    }
    void importImages() {
        Fl_Native_File_Chooser chooser;
        chooser.title("Import PNG or JPG keyframes");
        chooser.type(Fl_Native_File_Chooser::BROWSE_MULTI_FILE);
        chooser.filter("Images\t*.{png,jpg,jpeg,PNG,JPG,JPEG}");
        if (chooser.show() != 0)
            return;
        std::vector<Slide> slides;
        for (int i = 0; i < chooser.count(); ++i)
            slides.push_back(importSlide(chooser.filename(i), project.width, project.height));
        if (slides.empty())
            return;
        checkpoint();
        project.slides.insert(project.slides.end(), slides.begin(), slides.end());
        reconcileTransitions(project);
        changed();
        selectSlide(project.slides.size() - slides.size());
    }
    bool save(bool saveAs) {
        fs::path folder = project.folder;
        if (saveAs || folder.empty()) {
            Fl_Native_File_Chooser chooser;
            chooser.title("Choose project folder");
            chooser.type(Fl_Native_File_Chooser::BROWSE_SAVE_DIRECTORY);
            chooser.options(Fl_Native_File_Chooser::NEW_FOLDER);
            if (chooser.show() != 0)
                return false;
            folder = chooser.filename();
            if (fs::exists(folder / "project.json") && fs::absolute(folder) != project.folder) {
                if (fl_choice("Replace the FrameMorph project in this folder?", "Cancel", "Replace",
                              nullptr) != 1)
                    return false;
            }
        }
        saveProject(project, folder);
        dirty = false;
        window->label("FrameMorph 0.8 - Slide studio");
        message("Saved project: " + folder.string());
        return true;
    }
    void open() {
        if (dirty && fl_choice("Open another project and discard unsaved changes?", "Cancel",
                               "Open", nullptr) != 1)
            return;
        Fl_Native_File_Chooser chooser;
        chooser.title("Open FrameMorph project folder");
        chooser.type(Fl_Native_File_Chooser::BROWSE_DIRECTORY);
        if (chooser.show() != 0)
            return;
        fs::path folder = chooser.filename();
        bool recover = fs::exists(folder / "autosave.json");
        auto p = loadProject(folder, recover);
        project = std::move(p);
        dirty = recover;
        snapshotDirty = true;
        undoStack.clear();
        redoStack.clear();
        slide = 0;
        object = -1;
        transition = 0;
        connection = -1;
        selection = cv::Mat();
        stopPreview();
        refresh();
        message(recover ? "Recovered autosaved project. Save to keep changes." : "Project opened.");
    }
    void moveSlide(int delta) {
        int target = slide + delta;
        if (!currentSlide() || target < 0 || target >= (int)project.slides.size())
            return;
        checkpoint();
        std::swap(project.slides[slide], project.slides[target]);
        reconcileTransitions(project);
        changed();
        selectSlide(target);
    }
    void removeSlide() {
        if (!currentSlide())
            return;
        checkpoint();
        stopPreview();
        project.slides.erase(project.slides.begin() + slide);
        if (project.slides.empty()) {
            project.slides.push_back(blankSlide(project.width, project.height, {255, 255, 255}));
            project.slides.back().title = "Slide 1";
        }
        slideFocus = true;
        reconcileTransitions(project);
        slide = std::min(slide, std::max(0, (int)project.slides.size() - 1));
        object = -1;
        selection = cv::Mat();
        changed();
        refresh();
    }
    void applyMotion() {
        if (project.transitions.empty())
            return;
        auto candidate = project;
        auto &t = candidate.transitions[transition];
        t.duration = std::stod(duration->value());
        if (connection >= 0 && connection < (int)t.links.size()) {
            auto &l = t.links[connection];
            l.start = std::stod(start->value()) / 100;
            l.end = std::stod(end->value()) / 100;
            l.bend = std::stof(bend->value());
            l.rotation = std::stof(rotation->value());
            l.layer = std::stoi(layer->value());
            const char *names[] = {"smooth", "linear", "ease_in", "ease_out"};
            l.easing = names[easing->value()];
            l.mode = mode->value() ? "fade" : "morph";
            l.manual = true;
        }
        validateProject(candidate);
        checkpoint();
        project = std::move(candidate);
        changed();
    }
    void applyFunctions() {
        if (!currentLink())
            throw std::runtime_error("Choose an animation connection in Motion first");
        auto candidate = project;
        auto &l = candidate.transitions[transition].links[connection];
        l.offsetX = functionX->value();
        l.offsetY = functionY->value();
        l.scaleFunction = functionScale->value();
        l.rotationFunction = functionRotation->value();
        l.opacityFunction = functionOpacity->value();
        l.manual = true;
        validateProject(candidate);
        checkpoint();
        project = std::move(candidate);
        changed();
        previewTransition();
        message("Applied mathematical motion. t is normalized connection time.");
    }
    void connectSelected() {
        if (project.transitions.empty())
            return;
        Link l;
        l.id = newId();
        l.manual = true;
        int i = 1;
        for (auto &o : project.slides[transition].objects)
            if (participating(o)) {
                if (sources->selected(i))
                    l.sources.push_back(o.id);
                ++i;
            }
        i = 1;
        for (auto &o : project.slides[transition + 1].objects)
            if (participating(o)) {
                if (targets->selected(i))
                    l.targets.push_back(o.id);
                ++i;
            }
        if (l.sources.empty() && l.targets.empty())
            throw std::runtime_error("Select source and/or target objects");
        l.name = objectName->value();
        if (l.name.empty())
            l.name = "__connection";
        checkpoint();
        auto &links = project.transitions[transition].links;
        // Preserve unselected objects from connections affected by the edit.
        for (auto &old : links) {
            for (auto &id : l.sources)
                old.sources.erase(std::remove(old.sources.begin(), old.sources.end(), id),
                                  old.sources.end());
            for (auto &id : l.targets)
                old.targets.erase(std::remove(old.targets.begin(), old.targets.end(), id),
                                  old.targets.end());
        }
        links.erase(
            std::remove_if(links.begin(), links.end(),
                           [](auto &old) { return old.sources.empty() && old.targets.empty(); }),
            links.end());
        links.push_back(l);
        connection = links.size() - 1;
        changed();
        refreshConnections();
        message("Connection updated. Multiple sources converge; multiple targets copy.");
    }
    void applyCorrespondence() {
        if (!currentLink())
            throw std::runtime_error("Choose a connection in the Motion tab");
        Link copy = *currentLink();
        copy.manual = true;
        copy.componentPairs.clear();
        copy.anchors.clear();
        auto parse = [](std::string value, const std::function<void(double, double)> &add) {
            std::replace(value.begin(), value.end(), ',', ' ');
            std::istringstream in(value);
            std::string item;
            while (in >> item) {
                size_t colon = item.find(':');
                if (colon == std::string::npos)
                    throw std::runtime_error("Use pairs such as 0:1, 1:0");
                size_t aUsed = 0, bUsed = 0;
                double a = std::stod(item.substr(0, colon), &aUsed),
                       b = std::stod(item.substr(colon + 1), &bUsed);
                if (aUsed != colon || bUsed != item.size() - colon - 1)
                    throw std::runtime_error("Invalid correspondence pair");
                add(a, b);
            }
        };
        parse(pairs->value(), [&](double a, double b) {
            if (a < 0 || b < 0 || a != std::floor(a) || b != std::floor(b))
                throw std::runtime_error("Component indices must be nonnegative integers");
            copy.componentPairs.emplace_back(a, b);
        });
        parse(anchors->value(), [&](double a, double b) {
            if (a < 0 || a >= 1 || b < 0 || b >= 1)
                throw std::runtime_error("Boundary anchors must be in [0,1)");
            copy.anchors.emplace_back(a, b);
        });
        checkpoint();
        *currentLink() = copy;
        changed();
        previewTransition();
    }
    void requestPreview(double time) {
        if (project.slides.empty())
            return;
        if (snapshotDirty || !snapshot) {
            snapshot = std::make_shared<Project>(project);
            snapshotDirty = false;
        }
        canvas->showingPreview = true;
        previewBusy = true;
        preview.request(snapshot, time);
        clock->copy_label((number(time) + " sec").c_str());
    }
    void previewTransition() {
        if (project.transitions.empty())
            return;
        double time = 0;
        for (int i = 0; i < transition; ++i)
            time += project.slides[i].hold + project.transitions[i].duration;
        time += project.slides[transition].hold + project.transitions[transition].duration * .5;
        scrub->value(time);
        requestPreview(time);
    }
    void togglePlay() {
        if (project.slides.empty())
            return;
        playing = !playing;
        play->label(playing ? "Pause" : "Play");
        playTime = scrub->value();
        if (playTime >= projectDuration(project) - .02)
            playTime = 0;
        lastTick = std::chrono::steady_clock::now();
        if (playing) {
            playStarting = true;
            requestPreview(playTime);
            message("Preparing preview...");
        } else
            playStarting = false;
    }
    void beginExport(bool png) {
        if (exporting)
            throw std::runtime_error("An export is already running");
        validateProject(project);
        if (project.slides.empty())
            throw std::runtime_error("Import images before exporting");
        Fl_Native_File_Chooser chooser;
        chooser.title(png ? "Choose a new PNG sequence folder" : "Export animation video");
        chooser.type(png ? Fl_Native_File_Chooser::BROWSE_SAVE_DIRECTORY
                         : Fl_Native_File_Chooser::BROWSE_SAVE_FILE);
        if (!png) {
            chooser.filter("MP4 video\t*.mp4");
            chooser.preset_file("animation.mp4");
            chooser.options(Fl_Native_File_Chooser::SAVEAS_CONFIRM);
        }
        if (chooser.show() != 0)
            return;
        ExportOptions options;
        options.output = chooser.filename();
        options.pngSequence = png;
        options.width = std::stoi(exportWidth->value());
        options.height = std::stoi(exportHeight->value());
        options.fps = std::stoi(exportFps->value());
        if (options.width < 16 || options.height < 16 || options.width > 8192 ||
            options.height > 8192 || options.fps < 1 || options.fps > 240)
            throw std::runtime_error("Invalid export dimensions or frame rate");
        if (!png && options.output.extension() != ".mp4")
            options.output += ".mp4";
        if (exportThread.joinable())
            exportThread.join();
        exportCancel = false;
        exportDone = false;
        exportFrame = 0;
        exportTotal = 1;
        exportError.clear();
        exportPath = options.output;
        exporting = true;
        cancelButton->show();
        exportButton->deactivate();
        auto copy = deepCopy(project);
        exportThread = std::thread([this, copy = std::move(copy), options] {
            try {
                exportVideo(copy, options, &exportCancel, [this](int i, int total) {
                    exportFrame = i;
                    exportTotal = total;
                });
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(exportMutex);
                exportError = e.what();
            }
            exportDone = true;
            Fl::awake();
        });
        message("Rendering video with CUDA...");
    }
    static void tickCallback(void *data) {
        auto &e = *static_cast<Editor *>(data);
        try {
            e.tick();
        } catch (const std::exception &ex) {
            e.message(ex.what());
        }
        Fl::repeat_timeout(1. / 60, tickCallback, data);
    }
    void tick() {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastTick).count();
        lastTick = now;
        cv::Mat image;
        std::string error, notice;
        if (preview.poll(image, error, notice)) {
            previewBusy = false;
            if (!error.empty()) {
                playing = false;
                play->label("Play");
                message(error);
            } else {
                canvas->preview = image;
                ++previewPresented;
                if (playStarting) {
                    playStarting = false;
                    elapsed = 0;
                    message("Playing");
                }
                canvas->redraw();
                if (!notice.empty() && !exporting)
                    message(notice);
            }
        }
        if (playing && !playStarting) {
            playTime += elapsed;
            if (playTime >= projectDuration(project)) {
                playTime = projectDuration(project);
                playing = false;
                play->label("Play");
                requestPreview(playTime);
            }
            scrub->value(playTime);
            if (!previewBusy)
                requestPreview(playTime);
        }
        if (exporting) {
            if (exportDone) {
                if (exportThread.joinable())
                    exportThread.join();
                exporting = false;
                cancelButton->hide();
                exportButton->activate();
                std::lock_guard<std::mutex> lock(exportMutex);
                message(exportError.empty() ? "Exported: " + exportPath.string() : exportError);
            } else
                message("Rendering " + std::to_string(exportFrame.load()) + " / " +
                        std::to_string(exportTotal.load()) + " frames. Editing remains available.");
        }
        if (dirty && !exporting && std::chrono::duration<double>(now - lastAutosave).count() > 60) {
            fs::path folder = project.folder;
            if (folder.empty())
                folder = fs::current_path() / "recovery" / project.slides.front().id;
            saveProject(project, folder, true);
            lastAutosave = now;
            message("Autosaved recovery snapshot.");
        }
    }
};
float Canvas::imageScale() const {
    return std::min(float(w() - 64) / e_.project.width, float(h() - 64) / e_.project.height) *
           zoom_;
}
cv::Point Canvas::imagePoint(int px, int py) const {
    float s = imageScale();
    return {int(std::floor((px - x() - w() * .5f + e_.project.width * s * .5f - panX_) / s)),
            int(std::floor((py - y() - h() * .5f + e_.project.height * s * .5f - panY_) / s))};
}
cv::Point Canvas::screenPoint(cv::Point2f p) const {
    float s = imageScale();
    return {int(std::round(x() + w() * .5f - e_.project.width * s * .5f + panX_ + p.x * s)),
            int(std::round(y() + h() * .5f - e_.project.height * s * .5f + panY_ + p.y * s))};
}
std::vector<cv::Point2f> Canvas::framePoints(const json &f) const {
    float x = f["x"], y = f["y"], w = f["w"], h = f["h"],
          a = f["angle"].get<float>() * float(CV_PI) / 180;
    cv::Point2f center(x + w / 2, y + h / 2);
    std::vector<cv::Point2f> result;
    for (auto p : std::vector<cv::Point2f>{{-w / 2, -h / 2},
                                           {0, -h / 2},
                                           {w / 2, -h / 2},
                                           {w / 2, 0},
                                           {w / 2, h / 2},
                                           {0, h / 2},
                                           {-w / 2, h / 2},
                                           {-w / 2, 0},
                                           {0, -h / 2 - 28 / imageScale()}})
        result.push_back(center + cv::Point2f(p.x * std::cos(a) - p.y * std::sin(a),
                                              p.x * std::sin(a) + p.y * std::cos(a)));
    return result;
}
int Canvas::handleAt(int px, int py) const {
    if (!e_.currentObject() || showingPreview)
        return -1;
    auto points = framePoints(objectFrame(*e_.currentObject()));
    for (int i = 0; i < 9; ++i) {
        auto p = screenPoint(points[i]);
        if (std::abs(px - p.x) <= 7 && std::abs(py - p.y) <= 7)
            return i;
    }
    return -1;
}
int Canvas::hitObject(cv::Point p) const {
    auto *slide = e_.currentSlide();
    if (!slide || p.x < 0 || p.y < 0 || p.x >= slide->image.cols || p.y >= slide->image.rows)
        return -1;
    int hit = -1;
    for (int i = 0; i < (int)slide->objects.size(); ++i) {
        auto &o = slide->objects[i];
        if (hit >= 0 && o.layer < slide->objects[hit].layer)
            continue;
        bool inside = o.mask.at<uchar>(p) > 8;
        auto kind = o.drawing.is_null() ? std::string("pixels")
                    : o.drawing.contains("construction")
                        ? o.drawing["construction"].value("kind", "")
                        : o.drawing.value("kind", "");
        if (!inside && (kind == "text" || kind == "equation" || kind == "image")) {
            auto points = framePoints(objectFrame(o));
            std::vector<cv::Point2f> corners{points[0], points[2], points[4], points[6]};
            inside = cv::pointPolygonTest(corners, p, false) >= 0;
        }
        if (inside)
            hit = i;
    }
    return hit;
}
void Canvas::pointerDown(int px, int py, bool shift) {
    Fl::focus(this);
    e_.slideFocus = false;
    e_.stopPreview();
    auto p = imagePoint(px, py);
    if (p.x < 0 || p.y < 0 || p.x >= e_.project.width || p.y >= e_.project.height)
        return;
    if (e_.placement == 1) {
        drawingDrag_ = true;
        drawBegin_ = drawEnd_ = p;
        return;
    }
    int handle = handleAt(px, py), hit = handle >= 0 ? e_.object : hitObject(p);
    if (hit < 0) {
        e_.selectObject(-1);
        marquee_ = true;
        drawBegin_ = drawEnd_ = p;
        redraw();
        return;
    }
    if (handle < 0)
        e_.selectObject(hit, true, shift || e_.objects->selected(hit + 1));
    gestureFrames_.clear();
    gestureAssets_.clear();
    for (int i : handle < 0 ? e_.selectedIndices() : std::vector<int>{hit})
        gestureFrames_.emplace_back(i, objectFrame(e_.currentSlide()->objects[i]));
    std::stable_sort(gestureFrames_.begin(), gestureFrames_.end(), [&](auto &a, auto &b) {
        return e_.currentSlide()->objects[a.first].layer <
               e_.currentSlide()->objects[b.first].layer;
    });
    objectDrag_ = true;
    gestureMoved_ = false;
    gestureHandle_ = handle;
    gestureStart_ = p;
    gestureFrame_ = currentFrame_ = objectFrame(*e_.currentObject());
    double cx = gestureFrame_["x"].get<double>() + gestureFrame_["w"].get<double>() / 2,
           cy = gestureFrame_["y"].get<double>() + gestureFrame_["h"].get<double>() / 2;
    gestureStartAngle_ = std::atan2(p.y - cy, p.x - cx);
    gestureBackground_.release();
    gestureAsset_.release();
    redraw();
}
void Canvas::pointerMove(int px, int py, bool shift) {
    auto p = imagePoint(px, py);
    if (drawingDrag_ || marquee_) {
        drawEnd_ = p;
        redraw();
        return;
    }
    if (!objectDrag_)
        return;
    double dx = p.x - gestureStart_.x, dy = p.y - gestureStart_.y;
    if (!gestureMoved_ && std::hypot(dx, dy) * imageScale() < 2)
        return;
    if (!gestureMoved_) {
        Slide residual = *e_.currentSlide();
        std::set<int> ids;
        gestureAssets_.clear();
        for (auto &[i, f] : gestureFrames_) {
            auto &o = e_.currentSlide()->objects[i];
            ids.insert(i);
            cv::Mat asset;
            if (!o.artwork.empty())
                asset = o.drawing.value("kind", "") == "image" && !o.sourceAsset.empty()
                            ? o.sourceAsset
                            : o.artwork;
            else {
                auto b = cv::boundingRect(o.mask);
                cv::cvtColor(e_.currentSlide()->image(b), asset, cv::COLOR_BGR2BGRA);
                for (int y = 0; y < b.height; ++y)
                    for (int x = 0; x < b.width; ++x)
                        asset.at<cv::Vec4b>(y, x)[3] = o.mask.at<uchar>(b.y + y, b.x + x);
                residual.baseImage =
                    (residual.baseImage.empty() ? residual.image : residual.baseImage).clone();
                cv::Mat erase;
                cv::dilate(o.mask, erase, cv::Mat());
                residual.baseImage.setTo(cv::Scalar(residual.background[0], residual.background[1],
                                                    residual.background[2]),
                                         erase);
            }
            gestureAssets_.push_back(asset);
        }
        std::vector<Object> remaining;
        for (int i = 0; i < int(residual.objects.size()); ++i)
            if (!ids.count(i))
                remaining.push_back(residual.objects[i]);
        residual.objects = std::move(remaining);
        recomposeSlide(residual);
        gestureBackground_ = residual.image;
        gestureMoved_ = true;
    }
    currentFrame_ = gestureFrame_;
    double x = gestureFrame_["x"], y = gestureFrame_["y"], w = gestureFrame_["w"],
           h = gestureFrame_["h"], angle = gestureFrame_["angle"];
    if (gestureHandle_ == 8) {
        double a =
            (std::atan2(p.y - y - h / 2, p.x - x - w / 2) - gestureStartAngle_) * 180 / CV_PI +
            angle;
        if (shift)
            a = std::round(a / 15) * 15;
        currentFrame_["angle"] = a;
    } else if (gestureHandle_ < 0) {
        if (shift) {
            if (std::abs(dx) > std::abs(dy))
                dy = 0;
            else
                dx = 0;
        }
        currentFrame_["x"] = x + dx;
        currentFrame_["y"] = y + dy;
    } else {
        double radians = angle * CV_PI / 180, u = dx * std::cos(radians) + dy * std::sin(radians),
               v = -dx * std::sin(radians) + dy * std::cos(radians);
        bool left = gestureHandle_ == 0 || gestureHandle_ == 6 || gestureHandle_ == 7,
             right = gestureHandle_ == 2 || gestureHandle_ == 3 || gestureHandle_ == 4,
             top = gestureHandle_ <= 2, bottom = gestureHandle_ >= 4 && gestureHandle_ <= 6;
        double l = -w / 2, r = w / 2, t = -h / 2, b = h / 2;
        if (left)
            l = std::min(r - 8, l + u);
        if (right)
            r = std::max(l + 8, r + u);
        if (top)
            t = std::min(b - 8, t + v);
        if (bottom)
            b = std::max(t + 8, b + v);
        double nw = r - l, nh = b - t;
        if (shift) {
            if (std::abs(nw / w - 1) > std::abs(nh / h - 1)) {
                nh = nw * h / w;
                if (top)
                    t = b - nh;
                else
                    b = t + nh;
            } else {
                nw = nh * w / h;
                if (left)
                    l = r - nw;
                else
                    r = l + nw;
            }
        }
        double cu = (l + r) / 2, cv = (t + b) / 2;
        currentFrame_["x"] = x + w / 2 + cu * std::cos(radians) - cv * std::sin(radians) - nw / 2;
        currentFrame_["y"] = y + h / 2 + cu * std::sin(radians) + cv * std::cos(radians) - nh / 2;
        currentFrame_["w"] = nw;
        currentFrame_["h"] = nh;
    }
    e_.propX->value(number(currentFrame_["x"]).c_str());
    e_.propY->value(number(currentFrame_["y"]).c_str());
    e_.propW->value(number(currentFrame_["w"]).c_str());
    e_.propH->value(number(currentFrame_["h"]).c_str());
    e_.propAngle->value(number(currentFrame_["angle"]).c_str());
    redraw();
}
void Canvas::pointerUp(int px, int py, bool shift) {
    if (drawingDrag_) {
        drawingDrag_ = false;
        drawEnd_ = imagePoint(px, py);
        int kind = e_.drawType->value();
        auto start = drawBegin_, end = drawEnd_;
        bool click = cv::norm(start - end) * imageScale() < 3;
        e_.drawX->value(
            number(kind == 4 || kind == 5 ? start.x : std::min(start.x, end.x)).c_str());
        e_.drawY->value(
            number(kind == 4 || kind == 5 ? start.y : std::min(start.y, end.y)).c_str());
        double w = click ? (kind == 6   ? 300
                            : kind == 7 ? 360
                                        : 320)
                         : end.x - start.x,
               h = click ? (kind == 6 || kind == 7 ? 80 : 220) : end.y - start.y;
        if (kind != 4 && kind != 5) {
            w = std::max(8., std::abs(w));
            h = std::max(8., std::abs(h));
        }
        if (kind == 1 || shift)
            h = (h < 0 ? -1 : 1) * std::abs(w);
        e_.drawW->value(number(w).c_str());
        e_.drawH->value(number(h).c_str());
        e_.placement = 0;
        e_.addDrawing();
        e_.activateSelect();
        e_.syncProperties();
        window()->cursor(FL_CURSOR_DEFAULT);
        return;
    }
    if (marquee_) {
        marquee_ = false;
        drawEnd_ = imagePoint(px, py);
        cv::Rect r(std::min(drawBegin_.x, drawEnd_.x), std::min(drawBegin_.y, drawEnd_.y),
                   std::abs(drawBegin_.x - drawEnd_.x), std::abs(drawBegin_.y - drawEnd_.y));
        int active = -1;
        for (int i = 0; i < (int)e_.currentSlide()->objects.size(); ++i) {
            auto b = cv::boundingRect(e_.currentSlide()->objects[i].mask);
            if ((r & b).area() > 0) {
                e_.objects->select(i + 1);
                active = i;
            }
        }
        e_.object = active;
        e_.syncProperties();
        redraw();
        return;
    }
    if (objectDrag_) {
        pointerMove(px, py, shift);
        bool moved = gestureMoved_;
        auto frame = currentFrame_;
        objectDrag_ = false;
        gestureMoved_ = false;
        gestureBackground_.release();
        gestureAsset_.release();
        if (moved) {
            if (gestureHandle_ < 0) {
                double dx = frame["x"].get<double>() - gestureFrame_["x"].get<double>(),
                       dy = frame["y"].get<double>() - gestureFrame_["y"].get<double>();
                auto frames = gestureFrames_;
                for (auto &[i, f] : frames) {
                    f["x"] = f["x"].get<double>() + dx;
                    f["y"] = f["y"].get<double>() + dy;
                }
                e_.commitFrames(frames);
            } else
                e_.commitFrame(frame);
        }
        gestureFrames_.clear();
        gestureAssets_.clear();
        redraw();
    }
}
void Canvas::cancelGesture() {
    objectDrag_ = drawingDrag_ = marquee_ = false;
    gestureMoved_ = false;
    gestureFrames_.clear();
    gestureAssets_.clear();
    gestureBackground_.release();
    gestureAsset_.release();
    e_.syncProperties();
    window()->cursor(FL_CURSOR_DEFAULT);
    redraw();
}
void Canvas::draw() {
    fl_push_clip(x(), y(), w(), h());
    fl_color(bg);
    fl_rectf(x(), y(), w(), h());
    if (!e_.currentSlide()) {
        fl_color(dim);
        fl_font(FL_HELVETICA, 17);
        fl_draw("Add a slide to begin.", x(), y(), w(), h(), FL_ALIGN_CENTER);
        fl_pop_clip();
        return;
    }
    cv::Mat source = objectDrag_ && gestureMoved_         ? gestureBackground_
                     : showingPreview && !preview.empty() ? preview
                                                          : e_.currentSlide()->image;
    auto topLeft = screenPoint({0, 0});
    int iw = std::lround(e_.project.width * imageScale()),
        ih = std::lround(e_.project.height * imageScale());
    cv::Rect screen(topLeft.x, topLeft.y, iw, ih), view = screen & cv::Rect(x(), y(), w(), h());
    if (view.area() > 0) {
        if (cachedData_ != source.data || cachedRevision_ != e_.revision || cachedView_ != screen ||
            cachedSource_ != source.size() || cachedRgb_.size() != view.size()) {
            double sx = double(iw) / source.cols, sy = double(ih) / source.rows;
            cv::Mat resized;
            cv::Matx23d m(sx, 0, screen.x - view.x, 0, sy, screen.y - view.y);
            cv::warpAffine(source, resized, m, view.size(),
                           sx < 1 ? cv::INTER_AREA : cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            cv::cvtColor(resized, cachedRgb_, cv::COLOR_BGR2RGB);
            cachedImage_ = std::make_unique<Fl_RGB_Image>(cachedRgb_.data, cachedRgb_.cols,
                                                          cachedRgb_.rows, 3, cachedRgb_.step);
            cachedData_ = source.data;
            cachedRevision_ = e_.revision;
            cachedView_ = screen;
            cachedSource_ = source.size();
        }
        screenRgb = cachedRgb_;
        if (objectDrag_ && gestureMoved_) {
            screenRgb = cachedRgb_.clone();
            for (size_t layerIndex = 0; layerIndex < gestureFrames_.size(); ++layerIndex) {
                auto frame = currentFrame_;
                if (gestureHandle_ < 0) {
                    frame = gestureFrames_[layerIndex].second;
                    frame["x"] = frame["x"].get<double>() + currentFrame_["x"].get<double>() -
                                 gestureFrame_["x"].get<double>();
                    frame["y"] = frame["y"].get<double>() + currentFrame_["y"].get<double>() -
                                 gestureFrame_["y"].get<double>();
                }
                auto &asset = gestureAssets_[layerIndex];
                auto points = framePoints(frame);
                std::vector<cv::Point2f> dst;
                for (int i : {0, 2, 6}) {
                    auto p = screenPoint(points[i]);
                    dst.push_back(cv::Point2f(p.x - view.x, p.y - view.y));
                }
                std::vector<cv::Point2f> src{
                    {0, 0}, {float(asset.cols), 0}, {0, float(asset.rows)}};
                cv::Mat layer;
                auto m = cv::getAffineTransform(src, dst);
                cv::warpAffine(asset, layer, m, view.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
                float opacity = frame["opacity"];
                for (int y = 0; y < layer.rows; ++y) {
                    auto *p = layer.ptr<cv::Vec4b>(y);
                    auto *d = screenRgb.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < layer.cols; ++x)
                        if (p[x][3]) {
                            float a = p[x][3] / 255.f * opacity;
                            for (int c = 0; c < 3; ++c)
                                d[x][c] =
                                    cv::saturate_cast<uchar>(p[x][2 - c] * a + d[x][c] * (1 - a));
                        }
                }
            }
        } else if (!showingPreview && e_.tool->value() != 4 && !e_.selection.empty()) {
            screenRgb = cachedRgb_.clone();
            cv::Mat selection;
            double sx = double(iw) / e_.selection.cols, sy = double(ih) / e_.selection.rows;
            cv::warpAffine(e_.selection, selection,
                           cv::Matx23d(sx, 0, screen.x - view.x, 0, sy, screen.y - view.y),
                           view.size(), cv::INTER_NEAREST);
            cv::Mat tint(view.size(), CV_8UC3, cv::Scalar(45, 103, 230)), blend;
            cv::addWeighted(screenRgb, .65, tint, .35, 0, blend);
            blend.copyTo(screenRgb, selection);
        }
        fl_color(fl_rgb_color(195, 203, 215));
        fl_rectf(screen.x + 3, screen.y + 4, screen.width, screen.height);
        if (screenRgb.data == cachedRgb_.data)
            cachedImage_->draw(view.x, view.y);
        else
            fl_draw_image(screenRgb.data, view.x, view.y, view.width, view.height, 3,
                          screenRgb.step);
    }
    fl_color(line);
    fl_rect(screen.x - 1, screen.y - 1, screen.width + 2, screen.height + 2);
    if (!showingPreview && e_.tool->value() == 4) {
        for (int i = 1; i <= e_.objects->size(); ++i)
            if (e_.objects->selected(i) && i - 1 != e_.object) {
                auto f = objectFrame(e_.currentSlide()->objects[i - 1]);
                if (objectDrag_ && gestureMoved_ && gestureHandle_ < 0) {
                    f["x"] = f["x"].get<double>() + currentFrame_["x"].get<double>() -
                             gestureFrame_["x"].get<double>();
                    f["y"] = f["y"].get<double>() + currentFrame_["y"].get<double>() -
                             gestureFrame_["y"].get<double>();
                }
                double padding = 6 / imageScale();
                f["x"] = f["x"].get<double>() - padding / 2;
                f["y"] = f["y"].get<double>() - padding / 2;
                f["w"] = f["w"].get<double>() + padding;
                f["h"] = f["h"].get<double>() + padding;
                auto ps = framePoints(f);
                fl_color(accent);
                fl_line_style(FL_DASH, 1);
                for (int k = 0; k < 4; ++k) {
                    auto a = screenPoint(ps[k * 2]), b = screenPoint(ps[((k + 1) % 4) * 2]);
                    fl_line(a.x, a.y, b.x, b.y);
                }
            }
        fl_line_style(0);
        if (e_.currentObject()) {
            auto ps = framePoints(objectDrag_ ? currentFrame_ : objectFrame(*e_.currentObject()));
            fl_color(accent);
            fl_line_style(FL_SOLID, 2);
            for (int k = 0; k < 4; ++k) {
                auto a = screenPoint(ps[k * 2]), b = screenPoint(ps[((k + 1) % 4) * 2]);
                fl_line(a.x, a.y, b.x, b.y);
            }
            fl_line_style(0);
            auto n = screenPoint(ps[1]), r = screenPoint(ps[8]);
            fl_line(n.x, n.y, r.x, r.y);
            for (int i = 0; i < 9; ++i) {
                auto p = screenPoint(ps[i]);
                fl_color(panel);
                if (i == 8)
                    fl_pie(p.x - 5, p.y - 5, 10, 10, 0, 360);
                else
                    fl_rectf(p.x - 4, p.y - 4, 8, 8);
                fl_color(accent);
                if (i == 8)
                    fl_arc(p.x - 5, p.y - 5, 10, 10, 0, 360);
                else
                    fl_rect(p.x - 4, p.y - 4, 8, 8);
            }
        }
    }
    if (drawingDrag_ || marquee_) {
        auto a = screenPoint(drawBegin_), b = screenPoint(drawEnd_);
        fl_color(accent);
        fl_line_style(FL_DASH, 1);
        fl_rect(std::min(a.x, b.x), std::min(a.y, b.y), std::max(1, std::abs(b.x - a.x)),
                std::max(1, std::abs(b.y - a.y)));
        fl_line_style(0);
    }
    fl_pop_clip();
}
int Canvas::handle(int event) {
    try {
        if (event == FL_MOUSEWHEEL) {
            zoom_ = std::clamp(zoom_ * float(std::pow(1.10f, -Fl::event_dy())), .25f, 12.f);
            redraw();
            return 1;
        }
        if (event == FL_ENTER || event == FL_MOVE) {
            if (e_.tool->value() == 4 && !e_.placement) {
                int handle = handleAt(Fl::event_x(), Fl::event_y());
                window()->cursor(
                    handle >= 0 ? (handle == 8 ? FL_CURSOR_CROSS
                                               : (handle == 1 || handle == 5   ? FL_CURSOR_NS
                                                  : handle == 3 || handle == 7 ? FL_CURSOR_WE
                                                  : handle == 0 || handle == 4 ? FL_CURSOR_NWSE
                                                                               : FL_CURSOR_NESW))
                    : hitObject(imagePoint(Fl::event_x(), Fl::event_y())) >= 0 ? FL_CURSOR_HAND
                                                                               : FL_CURSOR_DEFAULT);
            }
            return 1;
        }
        if (event == FL_LEAVE) {
            window()->cursor(FL_CURSOR_DEFAULT);
            return 1;
        }
        if (event == FL_PUSH) {
            if (!e_.currentSlide())
                return 1;
            Fl::focus(this);
            dragX_ = Fl::event_x();
            dragY_ = Fl::event_y();
            if (Fl::event_button() == FL_RIGHT_MOUSE || Fl::event_button() == FL_MIDDLE_MOUSE) {
                panning_ = true;
                return 1;
            }
            if (Fl::event_button() != FL_LEFT_MOUSE)
                return 0;
            auto p = imagePoint(dragX_, dragY_);
            if (p.x < 0 || p.y < 0 || p.x >= e_.project.width || p.y >= e_.project.height)
                return 1;
            if (eyedropper) {
                e_.checkpoint();
                e_.currentSlide()->background = e_.currentSlide()->image.at<cv::Vec3b>(p);
                eyedropper = false;
                e_.changed();
                return 1;
            }
            if (e_.tool->value() == 4 || e_.placement == 1) {
                pointerDown(dragX_, dragY_, Fl::event_state(FL_SHIFT));
                if (Fl::event_clicks() && e_.currentObject() && e_.drawContent->active()) {
                    cancelGesture();
                    e_.focusObjectContent();
                }
                return 1;
            }
            e_.stopPreview();
            e_.checkpoint();
            if (e_.selection.empty())
                e_.selection = cv::Mat(e_.project.height, e_.project.width, CV_8UC1, cv::Scalar(0));
            int tool = e_.tool->value();
            if (tool == 1 || tool == 2) {
                painting_ = true;
                previous_ = p;
                cv::circle(e_.selection, p, int(e_.brush->value()), tool == 2 ? 0 : 255, -1);
            } else {
                auto region =
                    e_.selectionGpu.select(e_.currentSlide()->image, p, e_.tolerance->value());
                if (tool == 3 || Fl::event_state(FL_ALT))
                    e_.selection.setTo(0, region);
                else if (Fl::event_state(FL_SHIFT))
                    cv::bitwise_or(e_.selection, region, e_.selection);
                else
                    e_.selection = region;
            }
            redraw();
            return 1;
        }
        if (event == FL_DRAG) {
            if (panning_) {
                panX_ += Fl::event_x() - dragX_;
                panY_ += Fl::event_y() - dragY_;
                dragX_ = Fl::event_x();
                dragY_ = Fl::event_y();
                redraw();
                return 1;
            }
            if (painting_) {
                auto p = imagePoint(Fl::event_x(), Fl::event_y());
                cv::line(e_.selection, previous_, p, e_.tool->value() == 2 ? 0 : 255,
                         int(e_.brush->value() * 2), cv::LINE_8);
                previous_ = p;
                redraw();
                return 1;
            }
            pointerMove(Fl::event_x(), Fl::event_y(), Fl::event_state(FL_SHIFT));
            return 1;
        }
        if (event == FL_RELEASE) {
            panning_ = painting_ = false;
            pointerUp(Fl::event_x(), Fl::event_y(), Fl::event_state(FL_SHIFT));
            return 1;
        }
    } catch (const std::exception &ex) {
        cancelGesture();
        e_.message(ex.what());
        fl_alert("%s", ex.what());
    }
    return Fl_Widget::handle(event);
}
void SlideStrip::draw() {
    fl_push_clip(x(), y(), w(), h());
    fl_color(panel);
    fl_rectf(x(), y(), w(), h());
    for (int i = 0; i < (int)e_.project.slides.size(); ++i) {
        int top = y() + i * 130 - scroll_;
        if (top + 125 < y() || top > y() + h())
            continue;
        fl_color(i == e_.slide ? accent : line);
        fl_rectf(x() + 4, top + 3, w() - 8, 119);
        auto &slide = e_.project.slides[i];
        auto &cached = thumbnails_[slide.id];
        if (cached.second.empty() || cached.first != e_.revision ||
            cached.second.cols != w() - 16) {
            cv::Mat bgr;
            cv::resize(slide.image, bgr, {w() - 16, 87}, 0, 0, cv::INTER_AREA);
            cv::cvtColor(bgr, cached.second, cv::COLOR_BGR2RGB);
            cached.first = e_.revision;
            thumbnailImages_[slide.id] = std::make_unique<Fl_RGB_Image>(
                cached.second.data, cached.second.cols, cached.second.rows, 3, cached.second.step);
        }
        thumbnailImages_[slide.id]->draw(x() + 8, top + 7);
        fl_color(i == e_.slide ? bg : fg);
        fl_font(FL_HELVETICA, 11);
        std::string caption = std::to_string(i + 1) + "  " + e_.project.slides[i].title;
        if (caption.size() > 24)
            caption = caption.substr(0, 22) + "...";
        fl_draw(caption.c_str(), x() + 12, top + 111);
    }
    fl_pop_clip();
}
int SlideStrip::handle(int event) {
    if (event == FL_FOCUS || event == FL_UNFOCUS)
        return 1;
    if (event == FL_PUSH) {
        Fl::focus(this);
        int index = (Fl::event_y() - y() + scroll_) / 130;
        e_.selectSlide(index);
        return 1;
    }
    if (event == FL_MOUSEWHEEL) {
        scroll_ = std::clamp(scroll_ + Fl::event_dy() * 40, 0,
                             std::max(0, (int)e_.project.slides.size() * 130 - h()));
        redraw();
        return 1;
    }
    return Fl_Widget::handle(event);
}
} // namespace
int runEditor(const fs::path &path) {
    Fl::lock();
    Editor editor(path);
    editor.window->show();
    struct Smoke {
        Editor *editor;
        fs::path output;
        int status = 0;
    };
    Smoke smoke{&editor, {}};
    if (const char *output = std::getenv("FRAMEMORPH_UI_TEST")) {
        smoke.output = output;
        fs::create_directories(smoke.output);
        Fl::add_timeout(
            .8,
            [](void *data) {
                auto &test = *static_cast<Smoke *>(data);
                auto &e = *test.editor;
                auto playbackProject = e.project;
                std::ofstream log(test.output / "ui-test.log");
                auto require = [&](bool ok, const char *label) {
                    if (!ok)
                        throw std::runtime_error(label);
                    log << "PASS " << label << '\n';
                };
                auto capture = [&](const fs::path &file) {
                    e.window->make_current();
                    e.window->redraw();
                    Fl::flush();
                    unsigned char *pixels =
                        fl_read_image(nullptr, 0, 0, e.window->w(), e.window->h());
                    if (!pixels)
                        throw std::runtime_error("Editor screenshot capture failed");
                    cv::Mat rgb(e.window->h(), e.window->w(), CV_8UC3, pixels), bgr;
                    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                    cv::imwrite(file.string(), bgr);
                    delete[] pixels;
                };
                try {
                    require(e.project.slides.size() >= 2, "Editor opens a multi-slide project");
                    e.selectSlide(0);
                    capture(test.output / "editor.png");
                    auto count = e.project.slides[0].objects.size();
                    e.selection = e.selectionGpu.select(e.project.slides[0].image, {365, 570}, 18);
                    require(cv::countNonZero(e.selection) > 1000,
                            "Editor color selection produces a mask");
                    e.objectName->value("__ui_test");
                    e.createObject();
                    require(e.currentSlide()->objects.size() == count + 1,
                            "Create object stores the selection");
                    e.objectName->value("__ui_renamed");
                    e.objectName->do_callback();
                    require(e.currentObject()->name == "__ui_renamed",
                            "Object name input commits through its callback");
                    e.undo(false);
                    require(e.currentObject()->name == "__ui_test",
                            "Undo restores the object name");
                    e.undo(true);
                    require(e.currentObject()->name == "__ui_renamed",
                            "Redo reapplies the object name");
                    auto oldPixels = cv::countNonZero(e.currentObject()->mask);
                    cv::circle(e.selection, {365, 570}, 20, 0, -1);
                    e.updateMask();
                    require(cv::countNonZero(e.currentObject()->mask) < oldPixels,
                            "Mask corrections update the object");
                    e.undo(false);
                    require(cv::countNonZero(e.currentObject()->mask) == oldPixels,
                            "Undo restores the previous mask");
                    std::string first = e.project.slides[0].id;
                    e.moveSlide(1);
                    require(e.project.slides[1].id == first && e.slide == 1,
                            "Slide reordering updates the active slide");
                    fm::validateProject(e.project);
                    e.undo(false);
                    require(e.project.slides[0].id == first,
                            "Undo restores slide order and adjacency");
                    e.transition = 0;
                    e.connection = 0;
                    e.refreshConnections();
                    e.duration->value("2.1");
                    e.bend->value("0.3");
                    e.applyMotion();
                    require(e.project.transitions[0].duration == 2.1 &&
                                std::abs(e.project.transitions[0].links[0].bend - .3f) < .0001f,
                            "Motion controls update the transition");
                    capture(test.output / "editor-selection.png");
                    e.addBlank();
                    require(e.currentSlide()->blank,
                            "Blank keyframe action creates an authoring canvas");
                    e.drawDraft = json::object();
                    e.drawType->value(1);
                    e.drawName->value("__authored_qa");
                    e.drawX->value("100+20*cos(0)");
                    e.drawY->value("100");
                    e.drawW->value("180");
                    e.drawH->value("180");
                    e.addDrawing();
                    require(e.currentObject()->drawing["x"].get<double>() == 120,
                            "Drawing coordinates accept mathematical expressions");
                    std::string authoredId = e.currentSlide()->id;
                    e.duplicateCurrent();
                    require(e.currentSlide()->id != authoredId && e.currentObject() == nullptr,
                            "Duplicate action selects a new editable keyframe");
                    e.object = 0;
                    e.loadDrawing();
                    e.drawX->value("800");
                    e.updateDrawing();
                    require(e.currentObject()->drawing["x"].get<double>() == 800,
                            "Draw inspector edits the duplicated pose");
                    e.copyDrawing();
                    e.pasteDrawing();
                    require(e.currentSlide()->objects.size() == 2,
                            "Object copy/paste creates an editable layer");
                    e.alignObject(true);
                    require(std::abs(objectFrame(*e.currentObject())["x"].get<double>() +
                                     objectFrame(*e.currentObject())["w"].get<double>() * .5 -
                                     e.project.width * .5) < .01,
                            "Horizontal alignment centers authored geometry");
                    e.drawDraft = json::object();
                    e.drawType->value(11);
                    e.drawName->value("__implicit_qa");
                    e.drawContent->value("(x^2+y^2-1)^3-x^2*y^3=0");
                    e.drawX->value("380");
                    e.drawY->value("260");
                    e.drawW->value("360");
                    e.drawH->value("360");
                    e.plotRange->value("-1.6,1.6,-1.6,1.6");
                    e.addDrawing();
                    require(e.currentObject()->drawing["kind"] == "implicit",
                            "Implicit equation toolbox action creates a named object");
                    e.refreshConnections();
                    e.functionY->value("-80*sin(pi*t)");
                    e.applyFunctions();
                    require(e.currentLink()->offsetY == "-80*sin(pi*t)",
                            "Function inspector commits normalized-time motion");
                    e.stopPreview();
                    e.drawTab->show();
                    capture(test.output / "editor-drawing.png");
                    auto beforeCount = e.project.slides.size();
                    auto beforeMap = e.currentSlide()->image.clone();
                    e.selectObject(0);
                    auto mappedId = e.currentObject()->id, mappedName = e.currentObject()->name;
                    auto beforeBounds = cv::boundingRect(e.currentObject()->mask);
                    e.objectMap->value("(x,y):x+0.5,y");
                    e.mapUnits->value("100");
                    e.applyMap();
                    require(e.project.slides.size() == beforeCount &&
                                e.currentObject()->id == mappedId &&
                                e.currentObject()->name == mappedName,
                            "Apply update edits the current object without creating slides or "
                            "changing identity");
                    require(!*e.objectMap->value() &&
                                cv::boundingRect(e.currentObject()->mask).x >= beforeBounds.x + 49,
                            "Applying a point map moves artwork and clears the function input");
                    e.objectMap->value("z:z+0.25*i");
                    e.applyMap();
                    require(e.currentObject()->generation["operation"]["kind"] == "sequence",
                            "Successive maps compose and preserve a smooth animation recipe");
                    e.undo(false);
                    e.undo(false);
                    require(cv::norm(e.currentSlide()->image, beforeMap, cv::NORM_INF) == 0,
                            "Each in-place map is one complete undo action");
                    e.selectObject(0);
                    auto beforeInvalid = e.currentSlide()->image.clone();
                    e.objectMap->value("z:unknown(z)");
                    bool rejected = false;
                    try {
                        e.applyMap();
                    } catch (...) {
                        rejected = true;
                    }
                    require(rejected && *e.objectMap->value() &&
                                cv::norm(e.currentSlide()->image, beforeInvalid, cv::NORM_INF) == 0,
                            "Invalid maps retain the expression and leave artwork unchanged");
                    e.objectMap->value("");
                    e.propertiesScroll->scroll_to(0, 0);
                    capture(test.output / "editor-transforms.png");
                    e.requestPreview(0);
                    e.stopPreview();
                    cv::Mat cancelledPreview;
                    std::string previewError, previewNotice;
                    require(!e.previewBusy && !e.canvas->showingPreview &&
                                !e.preview.poll(cancelledPreview, previewError, previewNotice),
                            "Returning to edit mode cancels pending preview work");
                    // Exercise the exact pointer methods used by the native event handler.
                    e.stopPreview();
                    e.project = Project{};
                    e.project.slides.push_back(blankSlide(1920, 1080, {255, 255, 255}));
                    e.project.slides[0].title = "Interaction checks";
                    e.slide = 0;
                    e.object = -1;
                    e.transition = 0;
                    e.connection = -1;
                    e.undoStack.clear();
                    e.redoStack.clear();
                    ++e.revision;
                    e.refresh();
                    e.activateSelect();
                    e.canvas->reset();
                    require(e.tool->value() == 4, "Select is the default authoring tool");
                    e.armInsert(0);
                    auto start = e.canvas->screenPoint({200, 200}),
                         end = e.canvas->screenPoint({500, 400});
                    e.canvas->pointerDown(start.x, start.y);
                    e.canvas->pointerMove(end.x, end.y);
                    e.canvas->pointerUp(end.x, end.y);
                    Fl::flush();
                    require(e.currentSlide()->objects.size() == 1 &&
                                e.currentObject()->name == "__rectangle_1",
                            "Ribbon drag creates a selected object with a default name");
                    require(e.drawTab->visible() && e.objectProperties->visible(),
                            "Selecting an object reveals its unified properties automatically");
                    auto firstId = e.currentObject()->id;
                    auto f = objectFrame(*e.currentObject());
                    auto initialMask = e.currentObject()->mask.clone();
                    auto click = e.canvas->screenPoint({350, 300});
                    size_t history = e.undoStack.size();
                    e.canvas->pointerDown(click.x, click.y);
                    e.canvas->pointerUp(click.x, click.y);
                    require(e.undoStack.size() == history,
                            "Clicking an object does not create an undo entry");
                    auto oldImage = e.currentSlide()->image.clone();
                    e.canvas->pointerDown(click.x, click.y);
                    auto move = e.canvas->screenPoint({450, 350});
                    e.canvas->pointerMove(move.x, move.y);
                    Fl::flush();
                    require(objectFrame(*e.currentObject()) == f && e.undoStack.size() == history,
                            "Live dragging previews artwork without mutating the project");
                    capture(test.output / "live-drag.png");
                    e.canvas->pointerUp(move.x, move.y);
                    auto moved = objectFrame(*e.currentObject());
                    require(std::abs(moved["x"].get<double>() - f["x"].get<double>() - 100) < 4 &&
                                std::abs(moved["y"].get<double>() - f["y"].get<double>() - 50) < 4,
                            "Pointer drag commits the requested position");
                    require(e.undoStack.size() == history + 1,
                            "A completed drag creates exactly one undo entry");
                    e.undo(false);
                    require(cv::norm(e.currentSlide()->image, oldImage, cv::NORM_INF) == 0,
                            "Undo restores the exact image before a drag");
                    e.undo(true);
                    require(e.currentObject()->id == firstId,
                            "Redo preserves permanent object identity");
                    auto ps = e.canvas->framePoints(objectFrame(*e.currentObject()));
                    auto corner = e.canvas->screenPoint(ps[4]);
                    auto beforeResize = objectFrame(*e.currentObject());
                    require(e.canvas->handleAt(corner.x, corner.y) == 4,
                            "Bounding-box corner is hit-testable in screen coordinates");
                    auto resize =
                        e.canvas->screenPoint({float(ps[4].x + 150), float(ps[4].y + 100)});
                    e.canvas->pointerDown(corner.x, corner.y);
                    e.canvas->pointerMove(resize.x, resize.y, true);
                    e.canvas->pointerUp(resize.x, resize.y, true);
                    auto resized = objectFrame(*e.currentObject());
                    require(resized["w"].get<double>() > beforeResize["w"].get<double>() * 1.3,
                            "Corner handle resizes the selected object");
                    require(std::abs(resized["w"].get<double>() / resized["h"].get<double>() -
                                     beforeResize["w"].get<double>() /
                                         beforeResize["h"].get<double>()) < .02,
                            "Shift-resize preserves aspect ratio");
                    ps = e.canvas->framePoints(resized);
                    auto rotationHandle = e.canvas->screenPoint(ps[8]);
                    cv::Point2f center(resized["x"].get<float>() + resized["w"].get<float>() / 2,
                                       resized["y"].get<float>() + resized["h"].get<float>() / 2);
                    auto offset = ps[8] - center;
                    auto rotated = e.canvas->screenPoint(
                        center + cv::Point2f((offset.x - offset.y) * .7071068f,
                                             (offset.x + offset.y) * .7071068f));
                    e.canvas->pointerDown(rotationHandle.x, rotationHandle.y);
                    e.canvas->pointerMove(rotated.x, rotated.y, true);
                    e.canvas->pointerUp(rotated.x, rotated.y, true);
                    require(std::abs(objectFrame(*e.currentObject())["angle"].get<double>() - 45) <
                                1,
                            "Rotation handle supports snapped 15-degree increments");
                    e.propX->value("700+20");
                    e.propX->do_callback();
                    require(std::abs(objectFrame(*e.currentObject())["x"].get<double>() - 720) <
                                .01,
                            "Properties apply a coordinate expression without an Apply button");
                    e.drawColor->value("#d9485f");
                    e.drawColor->do_callback();
                    require(e.currentObject()->drawing["construction"]["color"] == "#d9485f",
                            "Color control updates the selected object and preserves native "
                            "construction");
                    e.propOpacity->value("0");
                    e.propOpacity->do_callback();
                    require(objectFrame(*e.currentObject())["opacity"] == 0 &&
                                cv::countNonZero(e.currentObject()->mask) > 0,
                            "Fully transparent objects remain selectable through their geometry");
                    e.propOpacity->value("1");
                    e.propOpacity->do_callback();
                    auto beforeNudge = objectFrame(*e.currentObject());
                    e.nudgeObject(FL_Right, 10);
                    require(objectFrame(*e.currentObject())["x"].get<double>() ==
                                beforeNudge["x"].get<double>() + 10,
                            "Arrow-key nudging updates object geometry");
                    e.armInsert(1);
                    start = e.canvas->screenPoint({1250, 300});
                    end = e.canvas->screenPoint({1450, 500});
                    e.canvas->pointerDown(start.x, start.y);
                    e.canvas->pointerUp(end.x, end.y);
                    auto circleName = e.currentObject()->name;
                    e.armInsert(1);
                    start = e.canvas->screenPoint({1400, 650});
                    end = e.canvas->screenPoint({1560, 810});
                    e.canvas->pointerDown(start.x, start.y);
                    e.canvas->pointerUp(end.x, end.y);
                    require(circleName == "__circle_1" && e.currentObject()->name == "__circle_2",
                            "Repeated insertions receive distinct automatic names");
                    e.objectName->value("__custom_name");
                    e.objectName->do_callback();
                    require(e.currentObject()->name == "__custom_name",
                            "A provided name overrides the automatic name");
                    e.animateObject->value(0);
                    e.animateObject->do_callback();
                    require(e.currentObject()->name == "custom_name",
                            "Animation participation can be changed with a checkbox");
                    e.animateObject->value(1);
                    e.animateObject->do_callback();
                    e.armInsert(6);
                    start = e.canvas->screenPoint({180, 700});
                    e.canvas->pointerDown(start.x, start.y);
                    e.canvas->pointerUp(start.x, start.y);
                    e.drawContent->value("Click, edit, animate.");
                    e.applyConstruction();
                    require(e.currentObject()->drawing["content"] == "Click, edit, animate.",
                            "The same properties panel edits text content");
                    e.focusObjectContent();
                    require(e.drawContent->y() >= e.propertiesScroll->y() &&
                                e.drawContent->y() + e.drawContent->h() <
                                    e.propertiesScroll->y() + e.propertiesScroll->h(),
                            "The double-click content action scrolls the relocated text editor "
                            "into view");
                    e.propX->value("250");
                    e.propX->do_callback();
                    e.drawFont->value("72");
                    e.drawFont->do_callback();
                    require(e.currentObject()->drawing["construction"]["font_size"] == 72,
                            "Font editing still works after moving an object");
                    e.copyDrawing();
                    auto copiedName = e.currentObject()->name;
                    e.pasteDrawing();
                    require(e.currentObject()->name != copiedName,
                            "Pasted objects get a distinct default name on the same slide");
                    auto checkFolder = test.output / "edited-project";
                    auto exact = e.currentSlide()->image.clone();
                    saveProject(e.project, checkFolder);
                    auto reopened = loadProject(checkFolder);
                    require(cv::norm(reopened.slides[0].image, exact, cv::NORM_INF) == 0 &&
                                reopened.slides[0].objects[0].drawing.contains("construction"),
                            "Direct manipulation and construction metadata survive save/reopen");
                    // Stress selection and live repaint with many full-resolution object masks.
                    for (int i = 0; i < 36; ++i) {
                        Object o;
                        o.id = newId();
                        o.name = "__stress_" + std::to_string(i);
                        o.layer = -1;
                        Slide temp = *e.currentSlide();
                        renderDrawing(temp, o,
                                      {{"kind", "rectangle"},
                                       {"x", 70 + (i % 9) * 190},
                                       {"y", 70 + (i / 9) * 230},
                                       {"w", 65},
                                       {"h", 40},
                                       {"color", "#a5b8d5"}});
                        e.currentSlide()->objects.push_back(std::move(o));
                    }
                    recomposeSlide(*e.currentSlide());
                    ++e.revision;
                    e.refresh();
                    e.activateSelect();
                    e.selectObject(1);
                    Fl::flush();
                    auto target = e.canvas->screenPoint({1350, 400});
                    std::vector<double> selectionTimes, dragTimes;
                    for (int i = 0; i < 30; ++i) {
                        e.selectObject(-1);
                        auto begin = std::chrono::steady_clock::now();
                        e.canvas->pointerDown(target.x, target.y);
                        e.canvas->pointerUp(target.x, target.y);
                        Fl::flush();
                        selectionTimes.push_back(std::chrono::duration<double, std::milli>(
                                                     std::chrono::steady_clock::now() - begin)
                                                     .count());
                    }
                    require(e.object == 1,
                            "Hit testing selects the front object in a crowded slide");
                    auto beforeGesture = objectFrame(*e.currentObject());
                    auto undoBefore = e.undoStack.size();
                    e.canvas->pointerDown(target.x, target.y);
                    for (int i = 0; i < 60; ++i) {
                        auto begin = std::chrono::steady_clock::now();
                        e.canvas->pointerMove(target.x + i % 20, target.y + i % 15);
                        Fl::flush();
                        dragTimes.push_back(std::chrono::duration<double, std::milli>(
                                                std::chrono::steady_clock::now() - begin)
                                                .count());
                    }
                    e.canvas->cancelGesture();
                    require(objectFrame(*e.currentObject()) == beforeGesture &&
                                e.undoStack.size() == undoBefore,
                            "Escape cancels a drag without changing the project or undo history");
                    auto report = [&](std::vector<double> times, const char *label) {
                        std::sort(times.begin(), times.end());
                        double sum = std::accumulate(times.begin(), times.end(), 0.);
                        log << "PERFORMANCE " << label << ": mean " << sum / times.size()
                            << " ms; p95 " << times[int(times.size() * .95)] << " ms; objects "
                            << e.currentSlide()->objects.size() << '\n';
                    };
                    report(selectionTimes, "click selection plus repaint");
                    report(dragTimes, "live drag plus repaint");
                    e.selectObject(0);
                    e.drawTab->show();
                    e.propertiesScroll->scroll_to(0, 0);
                    capture(test.output / "editor-v02.png");
                    auto box = objectFrame(*e.currentObject());
                    e.window->resize(0, 0, 1360, 860);
                    Fl::flush();
                    e.selectObject(0);
                    require(e.canvas->handleAt(
                                e.canvas->screenPoint(e.canvas->framePoints(box)[4]).x,
                                e.canvas->screenPoint(e.canvas->framePoints(box)[4]).y) == 4,
                            "Selection handles track a resized editor window");
                    capture(test.output / "editor-compact.png");

                    require(e.inspector->children() == 1 && !e.animationTab->visible(),
                            "The inspector exposes Properties only, with no Animation, Layers or "
                            "Updaters tabs");
                    e.selectObject(-1);
                    e.propertiesScroll->scroll_to(0, 0);
                    require(e.slideProperties->visible() && !e.objectProperties->visible() &&
                                e.stepDuration->parent() == e.slideProperties &&
                                e.generateUpdatersButton->parent() == e.slideProperties,
                            "Frame properties owns both the time step and Create next slide from "
                            "updaters");
                    capture(test.output / "editor-frame-properties.png");
                    double previousStep = e.currentSlide()->stepDuration;
                    e.stepDuration->value("2.5");
                    e.applyFrameStep();
                    require(e.currentSlide()->stepDuration == 2.5,
                            "Editing frame time step updates the model immediately");
                    e.undo(false);
                    require(e.currentSlide()->stepDuration == previousStep,
                            "Undo restores frame time step");
                    e.selectObject(0);
                    require(e.updaterInput->parent() == e.objectProperties &&
                                e.constructEffect->parent() == e.objectProperties &&
                                e.destructEffect->parent() == e.objectProperties &&
                                !e.slideProperties->visible(),
                            "Object properties owns updater and construct/destruct choices without "
                            "frame controls");
                    e.constructEffect->value(3);
                    e.destructEffect->value(1);
                    e.applyLifecycle();
                    require(e.currentObject()->constructAnimation == "scale" &&
                                e.currentObject()->destructAnimation == "none",
                            "Object lifecycle choices save to the selected object");
                    e.undo(false);
                    require(e.currentObject()->constructAnimation == "default" &&
                                e.currentObject()->destructAnimation == "default",
                            "Object lifecycle choices are one undo action");
                    e.selectObject(0);
                    e.propertiesScroll->scroll_to(0, 660);
                    e.selectSlide(e.slide);
                    require(e.slideProperties->visible() && e.propertiesScroll->yposition() == 0 &&
                                e.generateUpdatersButton->y() >= e.propertiesScroll->y() &&
                                e.generateUpdatersButton->y() + e.generateUpdatersButton->h() <
                                    e.propertiesScroll->y() + e.propertiesScroll->h(),
                            "Selecting a frame thumbnail returns its timing and generation button "
                            "to view");
                    e.selectObject(0);
                    e.drawTab->show();
                    auto originalUpdaterCount = e.project.slides.size();
                    int updaterSourceIndex = e.slide;
                    e.updaterInput->value("z:z+0.4*(exp(i*tau*t)-1)");
                    e.stepDuration->value("2");
                    e.applyFrameStep();
                    e.saveUpdater();
                    auto updaterSourceImage = e.currentSlide()->image.clone();
                    e.selectObject(1);
                    require(
                        e.objectProperties->visible() && !e.slideProperties->visible(),
                        "Selecting another object exposes its updater within Object properties");
                    e.updaterInput->value("(x,y,t):x+0.2*t,y");
                    e.saveUpdater();
                    require(e.updaterList->size() == 2, "Per-object updater actions save functions "
                                                        "for several objects on one slide");
                    e.propertiesScroll->scroll_to(0, 660);
                    capture(test.output / "editor-object-updater.png");
                    e.selectObject(-1);
                    e.slideFocus = true;
                    e.propertiesScroll->scroll_to(0, 0);
                    require(e.generateUpdatersButton->visible_r() &&
                                e.generateUpdatersButton->active() &&
                                e.currentSlide()->stepDuration == 2.,
                            "The frame generation button becomes available after saving object "
                            "updaters");
                    capture(test.output / "editor-frame-properties.png");
                    e.generateUpdatersButton->do_callback();
                    require(e.project.slides.size() == originalUpdaterCount + 1 &&
                                e.slide == updaterSourceIndex + 1 &&
                                e.project.transitions[updaterSourceIndex].duration == 2.,
                            "Generate from updaters commits one endpoint and its chosen transition "
                            "duration");
                    require(e.project.slides[updaterSourceIndex].objects[0].updater.is_object() &&
                                e.currentSlide()->objects[0].updater.is_null(),
                            "Saved source functions remain editable and do not become active on "
                            "the generated slide");
                    fm::validateProject(e.project);
                    e.selectObject(0);
                    require(!e.easing->active() && !e.duration->active(),
                            "Updater connections identify function-owned timing instead of "
                            "offering inactive morph controls");
                    e.propertiesScroll->scroll_to(0, 660);
                    capture(test.output / "editor-updaters.png");
                    e.propertiesScroll->scroll_to(0, 0);
                    e.undo(false);
                    require(e.project.slides.size() == originalUpdaterCount &&
                                cv::norm(e.currentSlide()->image, updaterSourceImage,
                                         cv::NORM_INF) == 0,
                            "Updater generation is one undo action and preserves the source frame");
                    e.duplicateCurrent();
                    require(e.currentSlide()->objects[0].updater.is_null() &&
                                cv::norm(e.currentSlide()->image, updaterSourceImage,
                                         cv::NORM_INF) == 0,
                            "The ordinary Duplicate action does not run saved updaters");
                    e.undo(false);
                    e.clearUpdater();
                    e.drawTab->show();
                    e.selectObject(0);
                    // Exercise the same keyboard dispatch as native FLTK shortcut events.
                    e.window->resize(0, 0, 1600, 1000);
                    e.selectObject(0);
                    Fl::focus(e.canvas);
                    auto deletedObjectId = e.currentObject()->id;
                    auto objectCount = e.currentSlide()->objects.size();
                    require(e.window->key(FL_Delete) &&
                                e.currentSlide()->objects.size() == objectCount - 1,
                            "Delete key removes the canvas-selected object");
                    e.undo(false);
                    require(e.currentObject()->id == deletedObjectId && !e.slideFocus,
                            "Undo restores a deleted object and its keyboard selection context");
                    Fl::focus(e.objectName);
                    auto textObjectCount = e.currentSlide()->objects.size();
                    require(!e.window->key(FL_Delete) &&
                                e.currentSlide()->objects.size() == textObjectCount,
                            "Delete in a text input does not delete the object");
                    e.duplicateCurrent();
                    e.selectSlide(0);
                    Fl::focus(e.strip);
                    auto slideCount = e.project.slides.size();
                    auto deletedSlideId = e.currentSlide()->id;
                    require(e.window->key(FL_Delete) && e.project.slides.size() == slideCount - 1,
                            "Delete key on the thumbnail strip removes the selected slide");
                    e.undo(false);
                    require(e.currentSlide()->id == deletedSlideId && e.slideFocus,
                            "Undo restores a deleted slide and its adjacency");
                    e.project = Project{};
                    e.project.width = 320;
                    e.project.height = 180;
                    auto raw = blankSlide(320, 180, {255, 255, 255});
                    Object rawObject;
                    rawObject.id = newId();
                    rawObject.name = "__raw";
                    rawObject.mask = cv::Mat(180, 320, CV_8U, cv::Scalar(0));
                    cv::circle(rawObject.mask, {80, 90}, 25, 255, -1);
                    raw.baseImage.setTo(cv::Scalar(20, 50, 120), rawObject.mask);
                    raw.image = raw.baseImage.clone();
                    raw.objects.push_back(rawObject);
                    e.project.slides.push_back(raw);
                    e.slide = 0;
                    e.object = -1;
                    e.refresh();
                    e.selectObject(0);
                    Fl::focus(e.canvas);
                    e.window->key(FL_Delete);
                    require(e.currentSlide()->image.at<cv::Vec3b>(90, 80) ==
                                cv::Vec3b(255, 255, 255),
                            "Deleting an extracted pixel object erases its visible footprint");
                    e.undo(false);
                    require(e.currentSlide()->image.at<cv::Vec3b>(90, 80) == cv::Vec3b(20, 50, 120),
                            "Undo restores deleted pixel artwork exactly");
                    e.selectSlide(0);
                    Fl::focus(e.strip);
                    e.window->key(FL_Delete);
                    require(e.project.slides.size() == 1 && e.currentSlide()->objects.empty(),
                            "Deleting the final slide leaves a usable blank canvas");
                    e.undo(false);
                    require(e.currentSlide()->objects.size() == 1,
                            "Deleting the final slide can be undone");
                    e.selectObject(0);
                    e.syncProperties();
                    e.propX->value("scene.origin_x-object.width/2");
                    e.applyObjectGeometry();
                    auto centeredFrame = objectFrame(*e.currentObject());
                    require(std::abs(centeredFrame["x"].get<double>() +
                                     centeredFrame["w"].get<double>() / 2 - 160) < .01,
                            "Property fields accept built-in scene and object variables");
                    // Multi-object gestures remain a single reversible edit.
                    e.project = Project{};
                    e.project.width = 1920;
                    e.project.height = 1080;
                    e.project.slides.push_back(blankSlide(1920, 1080, {255, 255, 255}));
                    for (int i = 0; i < 3; ++i) {
                        Object o;
                        o.id = newId();
                        o.name = "__arrange_" + std::to_string(i);
                        renderDrawing(e.project.slides[0], o,
                                      {{"kind", "rectangle"},
                                       {"x", 200. + 400 * i},
                                       {"y", 200. + 100 * i},
                                       {"w", 150.},
                                       {"h", 100.},
                                       {"color", "#368ce7"}});
                        e.project.slides[0].objects.push_back(o);
                    }
                    recomposeSlide(e.project.slides[0]);
                    e.slide = 0;
                    e.object = 0;
                    e.refresh();
                    e.selectObject(0);
                    e.selectObject(1, true, true);
                    e.selectObject(2, true, true);
                    require(e.selectedIndices().size() == 3,
                            "Shift selection retains several independently editable objects");
                    auto groupBefore = e.currentSlide()->image.clone();
                    auto firstFrame = objectFrame(e.currentSlide()->objects[0]);
                    auto lastFrame = objectFrame(e.currentSlide()->objects[2]);
                    e.undoStack.clear();
                    e.redoStack.clear();
                    auto groupHistory = e.undoStack.size();
                    auto groupStart = e.canvas->screenPoint({1075, 450}),
                         groupEnd = e.canvas->screenPoint({1175, 500});
                    e.canvas->pointerDown(groupStart.x, groupStart.y);
                    e.canvas->pointerMove(groupEnd.x, groupEnd.y);
                    Fl::flush();
                    require(e.selectedIndices().size() == 3 &&
                                objectFrame(e.currentSlide()->objects[0]) == firstFrame,
                            "Dragging a selected member preserves the group and previews without "
                            "changing geometry");
                    capture(test.output / "multi-object-drag.png");
                    e.canvas->pointerUp(groupEnd.x, groupEnd.y);
                    double groupDx = objectFrame(e.currentSlide()->objects[0])["x"].get<double>() -
                                     firstFrame["x"].get<double>();
                    require(
                        std::abs(groupDx - 100) < 4 &&
                            std::abs(objectFrame(e.currentSlide()->objects[2])["x"].get<double>() -
                                     lastFrame["x"].get<double>() - groupDx) < .01 &&
                            e.undoStack.size() == groupHistory + 1,
                        "Dragging translates every selected object with one undo entry");
                    e.undo(false);
                    require(cv::norm(groupBefore, e.currentSlide()->image, cv::NORM_INF) == 0 &&
                                e.selectedIndices().size() == 3,
                            "Undo restores all moved objects and their exact artwork");
                    e.selectObject(0);
                    e.selectObject(1, true, true);
                    e.selectObject(2, true, true);
                    e.nudgeObject(FL_Down, 10);
                    require(objectFrame(e.currentSlide()->objects[0])["y"].get<double>() ==
                                    firstFrame["y"].get<double>() + 10 &&
                                e.selectedIndices().size() == 3,
                            "Arrow keys nudge the entire selection and retain its members");
                    e.arrangeSelection(2);
                    double alignedY = objectFrame(e.currentSlide()->objects[0])["y"];
                    require(objectFrame(e.currentSlide()->objects[1])["y"] == alignedY &&
                                objectFrame(e.currentSlide()->objects[2])["y"] == alignedY,
                            "Arrange aligns multiple object edges");
                    auto second = objectFrame(e.currentSlide()->objects[1]);
                    second["x"] = 720.;
                    e.commitFrames({{1, second}});
                    e.arrangeSelection(4);
                    double xa = objectFrame(e.currentSlide()->objects[0])["x"],
                           xb = objectFrame(e.currentSlide()->objects[1])["x"],
                           xc = objectFrame(e.currentSlide()->objects[2])["x"];
                    require(std::abs(xb - xa - (xc - xb)) < .01,
                            "Horizontal distribution equalizes gaps while preserving the outside "
                            "objects");
                    e.alignObject(true);
                    auto left = e.frameBounds(objectFrame(e.currentSlide()->objects[0])),
                         right = e.frameBounds(objectFrame(e.currentSlide()->objects[2]));
                    require(std::abs((left.x + right.x + right.width) / 2 - 960) < .01,
                            "Center horizontally centers the whole selection without collapsing "
                            "its spacing");
                    e.stopPreview();
                    e.project = playbackProject;
                    e.slide = 0;
                    e.object = -1;
                    e.snapshotDirty = true;
                    e.refresh();
                    e.dirty = false;
                    e.scrub->value(0);
                    e.togglePlay();
                    auto waitStart = std::chrono::steady_clock::now();
                    while (e.playStarting && std::chrono::duration<double>(
                                                 std::chrono::steady_clock::now() - waitStart)
                                                     .count() < 15)
                        Fl::wait(.002);
                    require(!e.playStarting && e.playing,
                            "Playback waits for prepared geometry before starting its clock");
                    int presented = e.previewPresented;
                    auto cadenceStart = std::chrono::steady_clock::now();
                    while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                         cadenceStart)
                               .count() < 2.5)
                        Fl::wait(.002);
                    double cadenceSeconds = std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() - cadenceStart)
                                                .count();
                    log << "PERFORMANCE live FLTK preview presentations: "
                        << (e.previewPresented - presented) / cadenceSeconds << " fps over "
                        << cadenceSeconds << " seconds\n";
                    require(
                        e.previewPresented - presented > 35,
                        "Live playback presents frames continuously across the first transition");
                    e.stopPreview();
                    verifyAssetEditors(test.output, log);
                    // The same commit functions are used by studio Apply actions.
                    e.selectObject(0);
                    auto beforeImage = e.currentSlide()->image.clone();
                    auto imageId = e.currentObject()->id;
                    auto media = renderMedia({{"kind", "code"},
                                              {"content", "def square(x):\n    return x*x"},
                                              {"font_size", 28}});
                    e.commitMediaAsset({{{"kind", "code"},
                                         {"content", "def square(x):\n    return x*x"},
                                         {"font_size", 28}},
                                        media},
                                       false);
                    require(e.currentObject()->drawing["construction"]["kind"] == "code",
                            "Source code studio commits an editable construction object");
                    auto assetId = e.currentObject()->id;
                    e.commitImageEdit(
                        {{{"filter", "invert"}, {"width", media.cols}, {"height", media.rows}},
                         processImage(media, {{"filter", "invert"}})});
                    require(e.currentObject()->id == assetId &&
                                !e.currentObject()->originalAsset.empty() &&
                                e.currentObject()->imageEdits["filter"] == "invert",
                            "Image studio applies non-destructive settings without changing object "
                            "identity");
                    e.undo(false);
                    require(e.currentObject()->originalAsset.empty(),
                            "Undo restores an object before its first image-filter edit");
                    e.undo(false);
                    require(e.currentObject()->id == imageId &&
                                cv::norm(e.currentSlide()->image, beforeImage, cv::NORM_INF) == 0,
                            "Adding studio artwork is one complete undo action");
                    log << "UI smoke test completed\n";
                } catch (const std::exception &ex) {
                    log << "FAIL " << ex.what() << '\n';
                    test.status = 1;
                }
                e.dirty = false;
                e.window->hide();
            },
            &smoke);
    }
    struct Capture {
        Editor *editor;
        fs::path file;
    };
    Capture capture{&editor, {}};
    if (const char *file = std::getenv("FRAMEMORPH_CAPTURE")) {
        capture.file = file;
        Fl::add_timeout(
            1.,
            [](void *data) {
                auto &c = *static_cast<Capture *>(data);
                auto &e = *c.editor;
                e.selectObject(-1);
                e.slideFocus = true;
                e.propertiesScroll->scroll_to(0, 0);
                e.selection = cv::Mat();
                e.canvas->redraw();
                e.window->make_current();
                Fl::flush();
                unsigned char *pixels = fl_read_image(nullptr, 0, 0, e.window->w(), e.window->h());
                if (pixels) {
                    cv::Mat rgb(e.window->h(), e.window->w(), CV_8UC3, pixels), bgr;
                    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                    cv::imwrite(c.file.string(), bgr);
                    delete[] pixels;
                }
            },
            &capture);
    }
    int result = Fl::run();
    return smoke.status ? smoke.status : result;
}
} // namespace fm
