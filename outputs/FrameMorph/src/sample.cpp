#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fm {
namespace {
const cv::Scalar ink(42, 28, 17), muted(173, 161, 142), white(244, 239, 228), teal(206, 213, 64),
    gold(85, 205, 255), coral(112, 130, 250), violet(247, 151, 186);
void text(cv::Mat &image, const std::string &value, cv::Point pos, double scale, cv::Scalar color,
          int weight = 2) {
    cv::putText(image, value, pos, cv::FONT_HERSHEY_SIMPLEX, scale, color, weight, cv::LINE_AA);
}
void paint(Slide &s, const cv::Mat &mask, cv::Scalar color) {
    for (int y = 0; y < mask.rows; ++y) {
        auto m = mask.ptr<uchar>(y);
        auto p = s.image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x)
            if (m[x]) {
                float a = m[x] / 255.f;
                for (int c = 0; c < 3; ++c)
                    p[x][c] = cv::saturate_cast<uchar>(p[x][c] * (1 - a) + color[c] * a);
            }
    }
}
Object object(Slide &s, const std::string &name, cv::Scalar color,
              const std::function<void(cv::Mat &)> &draw, int layer = 0) {
    Object o;
    o.id = newId();
    o.name = name;
    o.layer = layer;
    o.mask = cv::Mat(s.image.size(), CV_8UC1, cv::Scalar(0));
    draw(o.mask);
    paint(s, o.mask, color);
    s.objects.push_back(o);
    return o;
}
} // namespace
void createSample(const fs::path &folder) {
    Project p;
    const char *titles[] = {"ONE SHAPE. MANY POSSIBILITIES.", "CORRESPONDENCE BECOMES MOTION.",
                            "ONE IDEA CAN BECOME MANY.", "BRING THE PIECES TOGETHER."};
    const char *subtitles[] = {"Select pixels. Give them a name. Tell a visual story.",
                               "Outlines move. Holes remain. Color evolves.",
                               "Repeated names create copies with independent destinations.",
                               "The same engine reverses the journey: many to one."};
    for (int k = 0; k < 4; ++k) {
        Slide s;
        s.id = newId();
        s.title = "0" + std::to_string(k + 1) + " - " + titles[k];
        s.hold = 1.1;
        s.background = {42, 28, 17};
        s.image = cv::Mat(p.height, p.width, CV_8UC3, ink);
        text(s.image, "F R A M E M O R P H", {110, 85}, .65, teal, 1);
        text(s.image, "CUDA / VISUAL STORIES", {1440, 85}, .5, muted, 1);
        cv::line(s.image, {110, 115}, {1810, 115}, cv::Scalar(82, 65, 46), 1, cv::LINE_AA);
        text(s.image, "FROM KEYFRAMES TO CONTINUOUS MOTION.", {110, 195}, 1.25, white, 2);
        text(s.image, "Select pixels. Give them a name. Shape the movement.", {110, 245}, .66,
             muted, 1);
        for (int x = 150; x < 1800; x += 60)
            for (int y = 330; y < 830; y += 60)
                cv::circle(s.image, {x, y}, 1, cv::Scalar(66, 48, 31), -1);
        text(s.image, "SHAPE", {180, 885}, .55, muted, 1);
        text(s.image, "TOPOLOGY", {790, 885}, .55, muted, 1);
        text(s.image, "COPY / CONVERGE", {1370, 885}, .55, muted, 1);
        cv::line(s.image, {110, 940}, {1810, 940}, cv::Scalar(82, 65, 46), 1, cv::LINE_AA);
        text(s.image, "IMAGE KEYFRAMES  +  NAMED OBJECTS  =  CONTINUOUS MOTION", {110, 1005}, .55,
             muted, 1);
        text(s.image, "C++ / CUDA", {1590, 1005}, .6, teal, 1);
        cv::Scalar shapeColor = k % 2 ? cv::Scalar(245, 170, 100) : teal;
        object(s, "__shape", shapeColor, [&](cv::Mat &m) {
            if (k == 0)
                cv::circle(m, {365, 570}, 138, 255, -1, cv::LINE_AA);
            else if (k == 1)
                cv::rectangle(m, {240, 430}, {515, 705}, 255, -1, cv::LINE_AA);
            else if (k == 2) {
                std::vector<cv::Point> v{{390, 415}, {555, 710}, {225, 710}};
                cv::fillConvexPoly(m, v, 255, cv::LINE_AA);
            } else
                cv::ellipse(m, {365, 570}, {160, 105}, -25, 0, 360, 255, -1, cv::LINE_AA);
        });
        object(
            s, "__ring", gold,
            [&](cv::Mat &m) {
                if (k == 0 || k == 3) {
                    cv::circle(m, {970, 570}, 145, 255, -1, cv::LINE_AA);
                    cv::circle(m, {970, 570}, 86, 0, -1, cv::LINE_AA);
                } else {
                    cv::rectangle(m, {825, 425}, {1115, 715}, 255, -1, cv::LINE_AA);
                    cv::rectangle(m, {880, 480}, {1060, 660}, 0, -1, cv::LINE_AA);
                }
            },
            1);
        if (k == 0 || k == 3)
            object(
                s, "__particles", violet,
                [&](cv::Mat &m) { cv::circle(m, {1530, 570}, 95, 255, -1, cv::LINE_AA); }, 2);
        else {
            int n = k == 1 ? 2 : 3;
            for (int i = 0; i < n; ++i)
                object(
                    s, "__particles", violet,
                    [&, i, n](cv::Mat &m) {
                        float a = 2 * 3.14159265f * i / n - 1.5707963f;
                        cv::circle(m, {int(1530 + 115 * std::cos(a)), int(570 + 115 * std::sin(a))},
                                   55, 255, -1, cv::LINE_AA);
                    },
                    2);
        }
        text(s.image, "__shape", {285, 790}, .65, shapeColor, 1);
        text(s.image, "__ring", {900, 790}, .65, gold, 1);
        text(s.image, "__particles", {1430, 790}, .65, violet, 1);
        p.slides.push_back(std::move(s));
    }
    reconcileTransitions(p);
    // Keep the particle family together for the two-to-three transition.
    for (size_t i = 0; i < p.transitions.size(); ++i) {
        auto &t = p.transitions[i];
        t.duration = 2.4;
        std::vector<std::string> a, b;
        for (auto &o : p.slides[i].objects)
            if (o.name == "__particles")
                a.push_back(o.id);
        for (auto &o : p.slides[i + 1].objects)
            if (o.name == "__particles")
                b.push_back(o.id);
        t.links.erase(std::remove_if(t.links.begin(), t.links.end(),
                                     [](auto &l) { return l.name == "__particles"; }),
                      t.links.end());
        Link l;
        l.id = newId();
        l.name = "__particles";
        l.sources = a;
        l.targets = b;
        l.bend = i == 2 ? -.4f : .3f;
        l.layer = 2;
        t.links.push_back(l);
        for (auto &link : t.links)
            if (link.name == "__shape") {
                link.rotation = i == 0 ? 60 : -40;
                link.bend = .12f;
            }
    }
    saveProject(p, folder);
}
void createAuthoringSample(const fs::path &folder) {
    Project p;
    auto add = [](Slide &s, std::string name, json spec, int layer = 0, cv::Mat source = {}) {
        Object o;
        o.id = newId();
        o.name = std::move(name);
        o.layer = layer;
        o.sourceAsset = std::move(source);
        renderDrawing(s, o, spec);
        s.objects.push_back(std::move(o));
        recomposeSlide(s);
    };
    cv::Mat image(180, 260, CV_8UC4);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x) {
            float u = float(x) / image.cols, v = float(y) / image.rows;
            float wave = .5f + .5f * std::sin(18 * u + 7 * v);
            image.at<cv::Vec4b>(y, x) = {uchar(100 + 145 * u), uchar(65 + 165 * wave),
                                         uchar(55 + 170 * v), 255};
        }
    for (int k = 0; k < 3; ++k) {
        auto s = blankSlide(p.width, p.height);
        s.title = "Authoring " + std::to_string(k + 1);
        s.hold = 1.2;
        add(s, "heading",
            {{"kind", "text"},
             {"content", "A visual story, built from mathematics."},
             {"font_size", 62},
             {"x", 110},
             {"y", 90},
             {"color", "#e9eff6"}});
        add(s, "subheading",
            {{"kind", "text"},
             {"content", "Draw. Duplicate. Change the next pose. Animate."},
             {"font_size", 30},
             {"x", 114},
             {"y", 185},
             {"color", "#8b9db1"}});
        add(s, "plot-label",
            {{"kind", "text"},
             {"content", "FUNCTION PLOT"},
             {"font_size", 21},
             {"x", 118},
             {"y", 320},
             {"color", "#8b9db1"}});
        add(s, "image-label",
            {{"kind", "text"},
             {"content", "EDITABLE IMAGE LAYER"},
             {"font_size", 21},
             {"x", 1240},
             {"y", 650},
             {"color", "#8b9db1"}});
        add(s, "implicit-label",
            {{"kind", "text"},
             {"content", "IMPLICIT EQUATION"},
             {"font_size", 21},
             {"x", 1240},
             {"y", 280},
             {"color", "#8b9db1"}});
        add(s, "axes",
            {{"kind", "line"},
             {"x", 120},
             {"y", 600},
             {"w", 940},
             {"h", 0},
             {"stroke", 2},
             {"color", "#3b4a5e"}});
        add(s, "axes",
            {{"kind", "line"},
             {"x", 590},
             {"y", 380},
             {"w", 0},
             {"h", 440},
             {"stroke", 2},
             {"color", "#3b4a5e"}});
        std::string functions[] = {"sin(x)", "cos(x)", "exp(-x^2/8)*sin(2*x)"};
        add(s, "__wave",
            {{"kind", "plot"},
             {"content", functions[k]},
             {"x", 120},
             {"y", 400},
             {"w", 940},
             {"h", 400},
             {"x_min", -6.283185307},
             {"x_max", 6.283185307},
             {"y_min", -1.5},
             {"y_max", 1.5},
             {"axes", false},
             {"stroke", 5},
             {"color", "#4ed5b8"}},
            1);
        std::string equations[] = {"y=\\sin(x)", "y=\\cos(x)", "y=e^{-x^2/8}\\sin(2x)"};
        add(s, "__equation",
            {{"kind", "equation"},
             {"content", equations[k]},
             {"font_size", 48},
             {"x", 300},
             {"y", 870},
             {"color", "#ffd05d"}},
            2);
        std::string implicit[] = {"x^2+y^2=1", "(x^2+y^2-1)^3-x^2*y^3=0", "x^4+y^4=1"};
        add(s, "__shape",
            {{"kind", "implicit"},
             {"content", implicit[k]},
             {"x", 1280 + k * 15},
             {"y", 320},
             {"w", 350},
             {"h", 300},
             {"x_min", -1.6},
             {"x_max", 1.6},
             {"y_min", -1.4},
             {"y_max", 1.4},
             {"color", k == 1 ? "#69aaf5" : "#4ed5b8"},
             {"filled", true},
             {"stroke", 2}},
            3);
        add(s, "__image",
            {{"kind", "image"},
             {"x", 1240 + k * 30},
             {"y", 720},
             {"w", 400 - k * 20},
             {"h", 220},
             {"angle", k == 1 ? -9 : 0},
             {"opacity", .95}},
            4, image.clone());
        p.slides.push_back(std::move(s));
    }
    reconcileTransitions(p);
    for (auto &t : p.transitions) {
        t.duration = 2.4;
        for (auto &l : t.links) {
            if (l.name == "__shape") {
                l.offsetY = "-25*sin(pi*t)";
                l.scaleFunction = "1+0.08*sin(pi*t)";
            }
            if (l.name == "__equation")
                l.mode = "fade";
        }
    }
    saveProject(p, folder);
}
void createTransformSample(const fs::path &folder) {
    Project p;
    auto s = blankSlide(p.width, p.height);
    s.title = "Before transformation";
    s.hold = 1.4;
    auto add = [&](std::string name, json spec, cv::Mat source = {}) {
        Object o;
        o.id = newId();
        o.name = std::move(name);
        o.sourceAsset = std::move(source);
        renderDrawing(s, o, spec);
        s.objects.push_back(std::move(o));
        recomposeSlide(s);
        return s.objects.back().id;
    };
    auto text = [&](std::string content, int x, int y, int size, std::string color) {
        add("caption", {{"kind", "text"},
                        {"content", content},
                        {"x", x},
                        {"y", y},
                        {"font_size", size},
                        {"color", color}});
    };
    text("Transform objects. Generate the next frame.", 110, 85, 62, "#e9eff6");
    text("One slide, three independent operations. The artwork follows the mathematics.", 114, 182,
         29, "#8b9db1");
    std::string labels[] = {"MATRIX", "COMPLEX FUNCTION", "COORDINATE MAP"};
    std::string formulas[] = {"\\begin{matrix}0.95&0.35\\\\-0.15&0.95\\end{matrix}",
                              "z\\mapsto z+0.22z^2", "(x,y)\\mapsto (x+0.35\\sin(1.8y),y)"};
    std::vector<std::string> ids;
    for (int panel = 0; panel < 3; ++panel) {
        int left = 110 + panel * 620;
        text(labels[panel], left + 15, 315, 22, "#8b9db1");
        cv::Mat artwork(360, 360, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        for (int k = 0; k < 9; ++k) {
            int v = 20 + k * 40;
            cv::line(artwork, {v, 20}, {v, 340}, {119, 89, 60, 255}, 1, cv::LINE_AA);
            cv::line(artwork, {20, v}, {340, v}, {119, 89, 60, 255}, 1, cv::LINE_AA);
        }
        cv::line(artwork, {180, 16}, {180, 344}, {180, 161, 140, 255}, 2, cv::LINE_AA);
        cv::line(artwork, {16, 180}, {344, 180}, {180, 161, 140, 255}, 2, cv::LINE_AA);
        cv::Scalar color = panel == 0   ? cv::Scalar(184, 213, 78, 255)
                           : panel == 1 ? cv::Scalar(245, 170, 105, 255)
                                        : cv::Scalar(93, 208, 255, 255);
        cv::circle(artwork, {180, 180}, 100, color, 4, cv::LINE_AA);
        cv::arrowedLine(artwork, {180, 180}, {280, 100}, color, 5, cv::LINE_AA, 0, .14);
        cv::circle(artwork, {240, 230}, 16, color, -1, cv::LINE_AA);
        ids.push_back(add("__panel_" + std::to_string(panel + 1),
                          {{"kind", "image"}, {"x", left + 55}, {"y", 425}, {"w", 360}, {"h", 360}},
                          artwork));
        if (panel == 0) {
            // Mathtext has no matrix environment; lay out the entries as plain text.
            text("[ 0.95   0.35 ]", left + 100, 850, 28, "#4ed5b8");
            text("[-0.15   0.95 ]", left + 100, 893, 28, "#4ed5b8");
        } else {
            add("formula", {{"kind", "equation"},
                            {"content", formulas[panel]},
                            {"x", left + (panel == 1 ? 65 : 0)},
                            {"y", 855},
                            {"font_size", panel == 1 ? 34 : 24},
                            {"color", panel == 1 ? "#69aaf5" : "#ffd05d"}});
        }
    }
    std::map<std::string, json> operations{
        {ids[0], {{"kind", "matrix"}, {"matrix", "0.95,0.35;-0.15,0.95"}, {"units", 150.}}},
        {ids[1], {{"kind", "complex"}, {"formula", "z+0.22*z^2"}, {"units", 150.}}},
        {ids[2], {{"kind", "map"}, {"x", "x+0.35*sin(1.8*y)"}, {"y", "y"}, {"units", 150.}}}};
    auto next = generateKeyframe(s, operations);
    next.title = "Generated result";
    p.slides = {std::move(s), std::move(next)};
    reconcileTransitions(p);
    p.transitions[0].duration = 3.6;
    saveProject(p, folder);
}
void createStudioSample(const fs::path &folder) {
    Project p;
    auto s = blankSlide(p.width, p.height, {255, 255, 255});
    s.title = "Select. Edit. Animate.";
    s.hold = 1.2;
    auto add = [&](std::string name, json spec) {
        Object o;
        o.id = newId();
        o.name = std::move(name);
        renderDrawing(s, o, spec);
        s.objects.push_back(std::move(o));
        recomposeSlide(s);
    };
    auto text = [&](const std::string &name, const std::string &content, int x, int y, int size,
                    const std::string &color) {
        add(name, {{"kind", "text"},
                   {"content", content},
                   {"x", x},
                   {"y", y},
                   {"font_size", size},
                   {"color", color}});
    };
    text("heading", "A slide, made to move.", 130, 90, 76, "#1d293d");
    text("subtitle", "Select an object. Edit its properties. Shape the next moment.", 135, 205, 32,
         "#66758c");
    add("left-panel", {{"kind", "rectangle"},
                       {"x", 110},
                       {"y", 325},
                       {"w", 650},
                       {"h", 555},
                       {"color", "#f1f5fb"},
                       {"filled", true}});
    add("right-panel", {{"kind", "rectangle"},
                        {"x", 805},
                        {"y", 325},
                        {"w", 1000},
                        {"h", 555},
                        {"color", "#f1f5fb"},
                        {"filled", true}});
    text("left-label", "01 / SELECT & EDIT", 165, 365, 21, "#66758c");
    text("right-label", "02 / EXPRESS THE IDEA", 865, 365, 21, "#66758c");
    add("__shape", {{"kind", "circle"},
                    {"x", 300},
                    {"y", 460},
                    {"w", 250},
                    {"h", 250},
                    {"color", "#386fe8"},
                    {"filled", true}});
    text("left-caption", "Drag. Resize. Rotate.", 240, 770, 29, "#1d293d");
    add("graph-axis", {{"kind", "line"},
                       {"x", 900},
                       {"y", 577},
                       {"w", 800},
                       {"h", 0},
                       {"stroke", 2},
                       {"color", "#9aa8ba"}});
    add("graph-axis", {{"kind", "line"},
                       {"x", 1300},
                       {"y", 455},
                       {"w", 0},
                       {"h", 245},
                       {"stroke", 2},
                       {"color", "#9aa8ba"}});
    add("__curve", {{"kind", "plot"},
                    {"content", "sin(x)"},
                    {"x", 900},
                    {"y", 455},
                    {"w", 800},
                    {"h", 245},
                    {"x_min", -6.283185307},
                    {"x_max", 6.283185307},
                    {"y_min", -1.5},
                    {"y_max", 1.5},
                    {"axes", false},
                    {"stroke", 5},
                    {"color", "#386fe8"}});
    add("__formula", {{"kind", "equation"},
                      {"content", "y=\\sin(x)"},
                      {"x", 1140},
                      {"y", 760},
                      {"font_size", 43},
                      {"color", "#1d293d"}});
    text("footer", "Named objects connect the keyframes. Mathematical functions add precision.",
         135, 955, 25, "#66758c");
    auto next = duplicateSlide(s);
    next.title = "The next pose";
    for (auto &o : next.objects) {
        auto spec = o.drawing;
        if (o.name == "__shape") {
            spec["kind"] = "rectangle";
            spec["x"] = 265;
            spec["y"] = 495;
            spec["w"] = 330;
            spec["h"] = 180;
            spec["color"] = "#8262df";
            renderDrawing(next, o, spec);
        } else if (o.name == "__curve") {
            spec["content"] = "cos(x)";
            renderDrawing(next, o, spec);
        } else if (o.name == "__formula") {
            spec["content"] = "y=\\cos(x)";
            renderDrawing(next, o, spec);
        }
    }
    recomposeSlide(next);
    p.slides = {s, next};
    reconcileTransitions(p);
    p.transitions[0].duration = 2.8;
    for (auto &l : p.transitions[0].links)
        if (l.name == "__formula")
            l.mode = "fade";
    saveProject(p, folder);
}
void createMotionSample(const fs::path &folder) {
    Project p;
    auto s = blankSlide(1920, 1080, {255, 255, 255});
    s.title = "Compose the first pose";
    s.hold = .8;
    auto add = [&](const std::string &name, const json &spec, cv::Mat image = cv::Mat{}) {
        Object o;
        o.id = newId();
        o.name = name;
        o.sourceAsset = image;
        renderDrawing(s, o, spec);
        s.objects.push_back(o);
        recomposeSlide(s);
    };
    auto text = [&](std::string content, int x, int y, int size, std::string color = "#66758c") {
        add("caption", {{"kind", "text"},
                        {"content", content},
                        {"x", x},
                        {"y", y},
                        {"font_size", size},
                        {"color", color}});
    };
    text("The next pose is yours.", 120, 95, 76, "#1d293d");
    text("Duplicate a slide. Edit each object. Let the names connect the motion.", 125, 210, 30);
    for (int i = 0; i < 3; ++i)
        add("panel", {{"kind", "rectangle"},
                      {"x", 95 + i * 590},
                      {"y", 340},
                      {"w", 550},
                      {"h", 520},
                      {"color", "#f1f5fb"}});
    text("01 / SHAPE", 145, 375, 24);
    text("02 / ROTATION", 735, 375, 24);
    text("03 / FUNCTION", 1325, 375, 24);
    add("__shape",
        {{"kind", "circle"}, {"x", 235}, {"y", 510}, {"w", 270}, {"h", 270}, {"color", "#386fe8"}});
    add("__bar", {{"kind", "rectangle"},
                  {"x", 800},
                  {"y", 625},
                  {"w", 320},
                  {"h", 18},
                  {"color", "#22aa91"}});
    cv::Mat grid(340, 340, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    for (int i = 20; i <= 320; i += 30) {
        cv::line(grid, {i, 20}, {i, 320}, {230, 170, 90, 230}, 2, cv::LINE_AA);
        cv::line(grid, {20, i}, {320, i}, {230, 170, 90, 230}, 2, cv::LINE_AA);
    }
    cv::circle(grid, {170, 170}, 100, {226, 101, 133, 255}, 4, cv::LINE_AA);
    add("__grid", {{"kind", "image"}, {"x", 1380}, {"y", 470}, {"w", 340}, {"h", 340}}, grid);
    text("Properties > Apply update", 1275, 895, 26, "#1d293d");
    text("z:z+0.2*z^2", 1370, 945, 28, "#8262df");
    text("Exact keyframes. Independent object maps. Smooth timing.", 125, 965, 26);
    auto next = duplicateSlide(s);
    next.title = "Edit objects in place";
    for (auto &o : next.objects) {
        if (o.name == "__shape") {
            auto spec = o.drawing;
            spec["kind"] = "rectangle";
            spec["color"] = "#8262df";
            renderDrawing(next, o, spec);
        } else if (o.name == "__bar") {
            auto frame = objectFrame(o);
            frame["angle"] = 90.;
            setObjectFrame(next, o, frame);
        } else if (o.name == "__grid")
            applyObjectMap(next, o, parseObjectMap("z:z+0.2*z^2", next, o, 150.));
    }
    recomposeSlide(next);
    p.slides = {s, next};
    reconcileTransitions(p);
    p.transitions[0].duration = 3.2;
    saveProject(p, folder);
}

void createUpdaterSample(const fs::path &folder) {
    Project project;
    auto slide = blankSlide(1920, 1080, {255, 255, 255});
    slide.title = "Functions describe the whole motion";
    slide.stepDuration = 2.;
    slide.hold = .6;
    auto add = [&](const std::string &name, const json &spec, cv::Mat asset = cv::Mat{}) {
        Object o;
        o.id = newId();
        o.name = name;
        o.sourceAsset = asset;
        renderDrawing(slide, o, spec);
        slide.objects.push_back(o);
        recomposeSlide(slide);
        return int(slide.objects.size() - 1);
    };
    auto text = [&](const std::string &content, int x, int y, int size,
                    std::string color = "#66758c") {
        add("caption", {{"kind", "text"},
                        {"content", content},
                        {"x", x},
                        {"y", y},
                        {"font_size", size},
                        {"color", color}});
    };
    text("The path lives in the function.", 105, 85, 72, "#1d293d");
    text("Equal endpoints can still contain a whole circle, a full turn, or a continuous "
         "deformation.",
         110, 195, 29);
    for (int i = 0; i < 3; ++i)
        add("panel", {{"kind", "rectangle"},
                      {"x", 85 + 590 * i},
                      {"y", 315},
                      {"w", 550},
                      {"h", 530},
                      {"color", "#f1f5fb"}});
    text("01 / ORBIT", 130, 355, 23);
    text("02 / ROTATE", 720, 355, 23);
    text("03 / DEFORM", 1310, 355, 23);
    add("orbit-path", {{"kind", "circle"},
                       {"x", 255},
                       {"y", 475},
                       {"w", 220},
                       {"h", 220},
                       {"filled", false},
                       {"stroke", 2},
                       {"color", "#cad5e6"}});
    int orb = add(
        "__orbit",
        {{"kind", "circle"}, {"x", 449}, {"y", 559}, {"w", 52}, {"h", 52}, {"color", "#386fe8"}});
    int bar = add("__spin", {{"kind", "rectangle"},
                             {"x", 790},
                             {"y", 575},
                             {"w", 320},
                             {"h", 20},
                             {"color", "#22aa91"}});
    cv::Mat grid(300, 300, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    for (int i = 15; i <= 285; i += 30) {
        cv::line(grid, {i, 15}, {i, 285}, {230, 170, 90, 240}, 2, cv::LINE_AA);
        cv::line(grid, {15, i}, {285, i}, {230, 170, 90, 240}, 2, cv::LINE_AA);
    }
    cv::circle(grid, {150, 150}, 95, {226, 101, 133, 255}, 4, cv::LINE_AA);
    int deform =
        add("__warp", {{"kind", "image"}, {"x", 1400}, {"y", 440}, {"w", 300}, {"h", 300}}, grid);
    text("z:z+1.1*(exp(i*tau*t)-1)", 115, 890, 24, "#386fe8");
    text("z:z*exp(i*tau*t)", 775, 890, 24, "#22aa91");
    text("z:z+0.2*t*z^2", 1390, 890, 24, "#8262df");
    text("Step: 2 seconds    |    t: 0 to 1    |    seconds = 2*t", 110, 980, 28, "#1d293d");
    std::map<std::string, json> operations;
    for (auto entry : std::vector<std::pair<int, std::string>>{{orb, "z:z+1.1*(exp(i*tau*t)-1)"},
                                                               {bar, "z:z*exp(i*tau*t)"},
                                                               {deform, "z:z+0.2*t*z^2"}}) {
        auto &o = slide.objects[entry.first];
        o.updater = parseObjectUpdater(entry.second, slide, o, 100.);
        operations[o.id] = o.updater;
    }
    auto next = generateUpdaterKeyframe(slide, operations, 2.);
    next.title = "Endpoint of the updater step";
    project.slides = {slide, next};
    reconcileTransitions(project);
    project.transitions[0].duration = 2.;
    for (auto &l : project.transitions[0].links)
        l.easing = "linear";
    saveProject(project, folder);
}

void createPropertiesSample(const fs::path &folder) {
    Project p;
    auto source = blankSlide(1920, 1080, {255, 255, 255});
    source.title = "Objects own their motion";
    source.hold = .8;
    source.stepDuration = 2.4;
    auto add = [](Slide &s, const std::string &name, const json &spec) -> int {
        Object o;
        o.id = newId();
        o.name = name;
        renderDrawing(s, o, spec);
        s.objects.push_back(std::move(o));
        recomposeSlide(s);
        return int(s.objects.size() - 1);
    };
    auto text = [&](const std::string &content, int x, int y, int size,
                    std::string color = "#64748b") {
        add(source, "caption",
            {{"kind", "text"},
             {"content", content},
             {"x", x},
             {"y", y},
             {"font_size", size},
             {"color", color}});
    };
    text("One canvas. Every property.", 110, 80, 76, "#1d293d");
    text("Objects own their motion. Frames own their timing.", 115, 198, 32);
    for (int i = 0; i < 3; ++i)
        add(source, "panel",
            {{"kind", "rectangle"},
             {"x", 85 + i * 590},
             {"y", 315},
             {"w", 550},
             {"h", 530},
             {"color", "#f1f5fb"}});
    text("01 / MATCH BY NAME", 130, 355, 23);
    text("02 / APPEAR & DISAPPEAR", 720, 355, 23);
    text("03 / UPDATER FUNCTION", 1310, 355, 23);
    int shape = add(
        source, "__shape",
        {{"kind", "circle"}, {"x", 245}, {"y", 495}, {"w", 220}, {"h", 220}, {"color", "#386fe8"}});
    int exit = add(source, "old-note",
                   {{"kind", "rectangle"},
                    {"x", 850},
                    {"y", 555},
                    {"w", 200},
                    {"h", 100},
                    {"color", "#ea7669"}});
    source.objects[exit].destructAnimation = "slide_up";
    add(source, "orbit-path",
        {{"kind", "circle"},
         {"x", 1430},
         {"y", 485},
         {"w", 240},
         {"h", 240},
         {"filled", false},
         {"stroke", 2},
         {"color", "#c8d4e6"}});
    int orbit = add(
        source, "__orbit",
        {{"kind", "circle"}, {"x", 1644}, {"y", 579}, {"w", 52}, {"h", 52}, {"color", "#22aa91"}});
    text("__shape : circle to square", 125, 890, 25, "#386fe8");
    text("Destruct: slide up / Construct: grow", 700, 890, 24, "#8262df");
    text("z:z+1.2*(exp(i*tau*t)-1)", 1300, 890, 24, "#22aa91");
    text("Select an object for its properties. Select a frame to set the time step and generate.",
         115, 985, 27, "#1d293d");
    auto &o = source.objects[orbit];
    o.updater = parseObjectUpdater("z:z+1.2*(exp(i*tau*t)-1)", source, o, 100.);
    auto target = generateUpdaterKeyframe(source, {{o.id, o.updater}}, source.stepDuration);
    target.title = "Matched shape, lifecycle effects, continuous orbit";
    auto spec = target.objects[shape].drawing;
    spec["kind"] = "rectangle";
    spec["color"] = "#8262df";
    renderDrawing(target, target.objects[shape], spec);
    target.objects.erase(target.objects.begin() + exit);
    int enter = add(target, "new-note",
                    {{"kind", "triangle"},
                     {"x", 850},
                     {"y", 520},
                     {"w", 200},
                     {"h", 170},
                     {"color", "#8262df"}});
    target.objects[enter].constructAnimation = "scale";
    recomposeSlide(target);
    p.slides = {source, target};
    reconcileTransitions(p);
    saveProject(p, folder);
}

void createToolkitSample(const fs::path &folder) {
    Project project;
    auto source = blankSlide(1920, 1080, {255, 255, 255});
    source.title = "An idea, in any language";
    source.hold = 1.;
    source.stepDuration = 3.2;
    auto add = [&](const std::string &name, const json &spec, cv::Mat asset = cv::Mat{}) {
        Object o;
        o.id = newId();
        o.name = name;
        o.sourceAsset = asset;
        renderDrawing(source, o, spec);
        source.objects.push_back(std::move(o));
        recomposeSlide(source);
        return int(source.objects.size() - 1);
    };
    auto text = [&](const std::string &value, int x, int y, int size,
                    std::string color = "#64748b") {
        add("caption", {{"kind", "text"},
                        {"content", value},
                        {"x", x},
                        {"y", y},
                        {"font_size", size},
                        {"color", color}});
    };
    text("An idea, in any language.", 105, 55, 76, "#1d293d");
    text("LaTeX, code, vectors, images and functions. One canvas. One animation engine.", 110, 160,
         30);
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 3; ++col)
            add("panel", {{"kind", "rectangle"},
                          {"x", 85 + 590 * col},
                          {"y", 255 + 360 * row},
                          {"w", 550},
                          {"h", 320},
                          {"color", "#f1f5fb"}});
    text("01 / FULL LATEX", 125, 285, 23);
    text("02 / SOURCE CODE", 715, 285, 23);
    text("03 / SVG ARTWORK", 1305, 285, 23);
    text("04 / IMAGE FORMULAS", 125, 645, 23);
    text("05 / APPEAR & DISAPPEAR", 715, 645, 23);
    text("06 / CONTINUOUS UPDATERS", 1305, 645, 23);
    auto media = [&](const std::string &name, const json &spec, int cx, int cy, int maxw,
                     int maxh) {
        auto pixels = renderMedia(spec);
        double scale = std::min({1., double(maxw) / pixels.cols, double(maxh) / pixels.rows});
        int i = add(name,
                    {{"kind", "image"},
                     {"x", cx - pixels.cols * scale / 2},
                     {"y", cy - pixels.rows * scale / 2},
                     {"w", pixels.cols * scale},
                     {"h", pixels.rows * scale},
                     {"construction", spec}},
                    pixels);
        return i;
    };
    int formula = media(
        "__matrix",
        {{"kind", "equation"},
         {"backend", "latex"},
         {"font_size", 46},
         {"color", "#386fe8"},
         {"content",
          R"(R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix})"}},
        360, 440, 450, 160);
    media("code-example",
          {{"kind", "code"},
           {"language", "python"},
           {"font_size", 25},
           {"theme", "dark"},
           {"line_numbers", true},
           {"content",
            "def transform(z, t):\n    angle = 2 * pi * t\n    return z * exp(1j * angle)"}},
          950, 440, 470, 170);
    int svg = media(
        "__vector",
        {{"kind", "svg"},
         {"content",
          R"(<svg xmlns="http://www.w3.org/2000/svg" width="260" height="180"><path d="M30 130 Q130 -20 230 130" fill="none" stroke="#386fe8" stroke-width="12"/><path d="M30 130 L230 130" stroke="#cbd5e1" stroke-width="3"/><circle cx="130" cy="55" r="20" fill="#22aa91"/><circle cx="30" cy="130" r="12" fill="#8262df"/><circle cx="230" cy="130" r="12" fill="#ea7669"/></svg>)"}},
        1540, 440, 300, 180);
    cv::Mat picture(180, 420, CV_8UC4);
    for (int y = 0; y < picture.rows; ++y)
        for (int x = 0; x < picture.cols; ++x)
            picture.at<cv::Vec4b>(y, x) = {uchar(90 + 120. * x / picture.cols),
                                           uchar(80 + 130. * y / picture.rows),
                                           uchar(220 - 120. * x / picture.cols), 255};
    for (int x = 20; x < 420; x += 40)
        cv::line(picture, {x, 0}, {x, 180}, cv::Scalar(255, 255, 255, 255), 1, cv::LINE_AA);
    cv::circle(picture, {210, 90}, 55, cv::Scalar(255, 255, 255, 255), 5, cv::LINE_AA);
    int image = add("__image", {{"kind", "image"}, {"x", 150}, {"y", 710}, {"w", 420}, {"h", 180}},
                    picture);
    int outgoing = add("old-marker", {{"kind", "rectangle"},
                                      {"x", 845},
                                      {"y", 760},
                                      {"w", 210},
                                      {"h", 70},
                                      {"color", "#ea7669"}});
    source.objects[outgoing].destructAnimation = "slide_left";
    add("orbit-path", {{"kind", "circle"},
                       {"x", 1480},
                       {"y", 720},
                       {"w", 140},
                       {"h", 140},
                       {"filled", false},
                       {"stroke", 2},
                       {"color", "#cbd5e1"}});
    int orbit = add(
        "__orbit",
        {{"kind", "circle"}, {"x", 1603}, {"y", 773}, {"w", 34}, {"h", 34}, {"color", "#22aa91"}});
    text("Select. Preview. Apply. Every edit stays in the project.", 110, 985, 29, "#1d293d");
    text("F1 opens the built-in guide.", 1400, 990, 26, "#386fe8");
    std::map<std::string, json> updates;
    for (auto entry : std::vector<std::pair<int, std::string>>{{orbit, "z:z+0.7*(exp(i*tau*t)-1)"},
                                                               {svg, "z:z*exp(i*tau*t)"}}) {
        auto &o = source.objects[entry.first];
        o.updater = parseObjectUpdater(entry.second, source, o);
        updates[o.id] = o.updater;
    }
    auto next = generateUpdaterKeyframe(source, updates, source.stepDuration);
    next.title = "Functions transform every kind of artwork";
    applyObjectMap(next, next.objects[formula],
                   parseObjectMap("(x,y):1.08*x,1.08*y", next, next.objects[formula]));
    next.objects[image].originalAsset = picture;
    next.objects[image].imageEdits = {{"pixel_map", "(r,g,b,a):g,b,r,a"}, {"saturation", 1.2}};
    next.objects[image].sourceAsset = processImage(picture, next.objects[image].imageEdits);
    renderDrawing(next, next.objects[image], next.objects[image].drawing);
    next.objects.erase(next.objects.begin() + outgoing);
    Object star;
    star.id = newId();
    star.name = "__new_star";
    star.constructAnimation = "pop";
    star.effectStart = .1;
    star.effectEnd = .85;
    renderDrawing(
        next, star,
        {{"kind", "star"}, {"x", 860}, {"y", 705}, {"w", 180}, {"h", 180}, {"color", "#8262df"}});
    next.objects.push_back(star);
    recomposeSlide(next);
    project.slides = {source, next};
    reconcileTransitions(project);
    saveProject(project, folder);
}
} // namespace fm
