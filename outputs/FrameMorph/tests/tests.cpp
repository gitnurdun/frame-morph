#include "engine.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <stdexcept>

namespace {
int passed = 0;
void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
    ++passed;
    std::cout << "PASS " << message << '\n';
}
template <class Fn> void rejects(Fn fn, const std::string &message) {
    bool rejected = false;
    try {
        fn();
    } catch (...) {
        rejected = true;
    }
    require(rejected, message);
}
fm::Project fixture() {
    fm::Project p;
    p.width = 320;
    p.height = 180;
    p.fps = 12;
    for (int k = 0; k < 2; ++k) {
        fm::Slide s;
        s.id = fm::newId();
        s.title = "Test " + std::to_string(k);
        s.background = {25, 20, 15};
        s.hold = .2;
        s.image = cv::Mat(p.height, p.width, CV_8UC3, cv::Scalar(25, 20, 15));
        fm::Object o;
        o.id = fm::newId();
        o.name = "__shape";
        o.mask = cv::Mat(p.height, p.width, CV_8U, cv::Scalar(0));
        if (!k)
            cv::circle(o.mask, {80, 90}, 30, 255, -1);
        else
            cv::rectangle(o.mask, {210, 60}, {270, 120}, 255, -1);
        s.image.setTo(k ? cv::Scalar(80, 130, 240) : cv::Scalar(200, 210, 70), o.mask);
        s.objects.push_back(o);
        p.slides.push_back(s);
    }
    fm::reconcileTransitions(p);
    p.transitions[0].duration = 1.;
    return p;
}
} // namespace
int main() {
    try {
        std::cout << fm::CudaBackend::deviceDescription() << '\n';
        fm::CudaBackend gpu;
        cv::Mat image(80, 120, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::rectangle(image, {10, 10}, {40, 40}, cv::Scalar(50, 200, 70), -1);
        cv::rectangle(image, {65, 10}, {95, 40}, cv::Scalar(50, 200, 70), -1);
        auto mask = gpu.select(image, {20, 20}, 5);
        require(cv::countNonZero(mask) == 31 * 31,
                "GPU selection excludes disconnected same-color regions");
        require(mask.at<uchar>(20, 75) == 0, "Disconnected object remains unselected");
        cv::line(image, {40, 40}, {65, 65}, cv::Scalar(50, 200, 70), 1);
        mask = gpu.select(image, {20, 20}, 5);
        require(mask.at<uchar>(65, 65) == 255, "Selection uses eight-connectivity");
        cv::Mat gradient(5, 100, CV_8UC3);
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 100; ++x)
                gradient.at<cv::Vec3b>(y, x) = {uchar(x * 2), uchar(x * 2), uchar(x * 2)};
        auto g = gpu.select(gradient, {5, 2}, 8);
        require(g.at<uchar>(2, 90) == 0, "Seed-relative tolerance does not drift across gradients");
        cv::Mat jpeg;
        std::vector<uchar> encoded;
        cv::imencode(".jpg", image, encoded, {cv::IMWRITE_JPEG_QUALITY, 85});
        jpeg = cv::imdecode(encoded, cv::IMREAD_COLOR);
        auto j = gpu.select(jpeg, {20, 20}, 12);
        require(cv::countNonZero(j) > 800 && cv::countNonZero(j) < 1400,
                "JPG compression tolerated without background leakage");
        // Compare GPU selection against an independent CPU Lab threshold + flood fill.
        cv::Mat floatImage, lab;
        image.convertTo(floatImage, CV_32F, 1. / 255);
        cv::cvtColor(floatImage, lab, cv::COLOR_BGR2Lab);
        cv::Mat eligible(image.size(), CV_8U, cv::Scalar(0));
        auto seed = lab.at<cv::Vec3f>(20, 20);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                if (cv::norm(lab.at<cv::Vec3f>(y, x) - seed) <= 8)
                    eligible.at<uchar>(y, x) = 255;
        cv::floodFill(eligible, {20, 20}, 128, nullptr, 0, 0, 8 | cv::FLOODFILL_FIXED_RANGE);
        cv::Mat reference = eligible == 128;
        mask = gpu.select(image, {20, 20}, 8);
        require(cv::norm(reference, mask, cv::NORM_INF) == 0,
                "CUDA labeling agrees with CPU reference");
        auto p = fixture();
        fm::validateProject(p);
        fm::Renderer renderer(p, 320, 180);
        require(cv::norm(renderer.transitionFrame(0, 0), p.slides[0].image, cv::NORM_INF) == 0,
                "Source keyframe reproduced exactly");
        require(cv::norm(renderer.transitionFrame(0, 1), p.slides[1].image, cv::NORM_INF) == 0,
                "Target keyframe reproduced exactly");
        auto mid = renderer.transitionFrame(0, .5);
        require(cv::norm(mid, renderer.transitionFrame(0, .5), cv::NORM_INF) == 0,
                "Repeated rendering is deterministic");
        require(cv::norm(cv::Vec3d(mid.at<cv::Vec3b>(90, 80)) - cv::Vec3d(p.slides[0].background)) <
                    3,
                "Moving object leaves no source ghost");
        require(cv::norm(cv::Vec3d(mid.at<cv::Vec3b>(90, 160)) -
                         cv::Vec3d(p.slides[0].background)) > 50,
                "Shape passes through the intermediate position");
        auto ring = fixture();
        for (auto &s : ring.slides) {
            s.image.setTo(cv::Scalar(25, 20, 15));
            auto &o = s.objects[0];
            o.mask.setTo(0);
            cv::circle(o.mask, {160, 90}, 45, 255, -1);
            cv::circle(o.mask, {160, 90}, 23, 0, -1);
            s.image.setTo(cv::Scalar(60, 210, 240), o.mask);
        }
        fm::Renderer ringRenderer(ring, 320, 180);
        auto ringMid = ringRenderer.transitionFrame(0, .5);
        require(cv::norm(cv::Vec3d(ringMid.at<cv::Vec3b>(90, 160)) -
                         cv::Vec3d(ring.slides[0].background)) < 3,
                "Hole remains open throughout ring morph");
        require(ringMid.at<cv::Vec3b>(90, 195)[1] > 150, "Ring body remains visible");
        auto copies = fixture();
        auto original = copies.slides[1].objects[0];
        auto other = original;
        other.id = fm::newId();
        other.mask = original.mask.clone();
        copies.slides[1].objects.push_back(other);
        copies.transitions[0].links = fm::automaticLinks(copies.slides[0], copies.slides[1]);
        require(copies.transitions[0].links.size() == 1 &&
                    copies.transitions[0].links[0].targets.size() == 2,
                "Repeated target names create one-to-many connection");
        fm::Renderer copiesRenderer(copies, 320, 180);
        auto cframe = copiesRenderer.transitionFrame(0, .5);
        require(cv::norm(cframe, mid, cv::NORM_INF) == 0,
                "Overlapping copies do not brighten or thicken edges");
        auto merge = copies;
        std::swap(merge.slides[0], merge.slides[1]);
        merge.transitions.clear();
        fm::reconcileTransitions(merge);
        require(merge.transitions[0].links[0].sources.size() == 2 &&
                    merge.transitions[0].links[0].targets.size() == 1,
                "Repeated source names create many-to-one connection");
        fm::Renderer mergeRenderer(merge, 320, 180);
        require(!mergeRenderer.transitionFrame(0, .5).empty(), "Many-to-one renders successfully");
        auto grouped = fixture();
        auto &o = grouped.slides[0].objects[0];
        cv::rectangle(o.mask, {100, 130}, {115, 145}, 255, -1);
        grouped.slides[0].image.setTo(cv::Scalar(230, 60, 100),
                                      o.mask & (cv::Mat(o.mask.size(), CV_8U, cv::Scalar(255))));
        fm::Renderer groupedRenderer(grouped, 320, 180);
        require(!groupedRenderer.transitionFrame(0, .5).empty(),
                "Disconnected components inside an object render together");
        auto unmatched = fixture();
        unmatched.slides[1].objects[0].name = "__different";
        unmatched.transitions[0].links =
            fm::automaticLinks(unmatched.slides[0], unmatched.slides[1]);
        require(unmatched.transitions[0].links.size() == 2,
                "Unmatched animated objects receive entrance/exit connections");
        fm::Renderer unmatchedRenderer(unmatched, 320, 180);
        require(!unmatchedRenderer.transitionFrame(0, .5).empty(),
                "Unmatched objects use the default fade");
        auto ordinary = fixture();
        ordinary.slides[0].objects[0].name = "ordinary";
        ordinary.slides[1].objects[0].name = "ordinary";
        ordinary.transitions[0].links = fm::automaticLinks(ordinary.slides[0], ordinary.slides[1]);
        require(ordinary.transitions[0].links.empty(),
                "Names without __ remain part of residual slides");
        auto caseSensitive = fixture();
        caseSensitive.slides[1].objects[0].name = "__Shape";
        caseSensitive.transitions[0].links =
            fm::automaticLinks(caseSensitive.slides[0], caseSensitive.slides[1]);
        require(caseSensitive.transitions[0].links.size() == 2,
                "Object matching is case sensitive");
        auto root = fm::fs::temp_directory_path() / ("framemorph-tests-" + fm::newId());
        fm::fs::create_directories(root);
        fm::saveProject(p, root / "project");
        auto reopened = fm::loadProject(root / "project");
        require(cv::norm(reopened.slides[0].image, p.slides[0].image, cv::NORM_INF) == 0 &&
                    cv::norm(reopened.slides[0].objects[0].mask, p.slides[0].objects[0].mask,
                             cv::NORM_INF) == 0,
                "Save/reopen preserves image and mask bytes");
        auto copy = fm::deepCopy(p);
        copy.slides[0].objects[0].mask.setTo(0);
        require(cv::countNonZero(p.slides[0].objects[0].mask) > 0,
                "Undo snapshots do not share mutable masks");
        p.slides[0].title = "Recovered edit";
        fm::saveProject(p, root / "project", true);
        require(fm::loadProject(root / "project", true).slides[0].title == "Recovered edit",
                "Autosave recovery loads uncommitted changes");
        require(fm::loadProject(root / "project").slides[0].title != "Recovered edit",
                "Autosave preserves last explicit save");
        fm::saveProject(p, root / "project");
        require(!fm::fs::exists(root / "project" / "autosave.json"),
                "Explicit save clears recovery manifest");
        fm::fs::copy(root / "project", root / "relocated", fm::fs::copy_options::recursive);
        require(fm::loadProject(root / "relocated").slides.size() == 2,
                "Project assets remain portable after relocation");
        auto bad = fixture();
        bad.transitions[0].links[0].end = 0;
        rejects([&] { fm::validateProject(bad); }, "Invalid connection timing rejected");
        bad = fixture();
        bad.transitions[0].links[0].sources = {"missing"};
        rejects([&] { fm::validateProject(bad); }, "Missing object references rejected");
        bad = fixture();
        bad.transitions[0].links.push_back(bad.transitions[0].links[0]);
        bad.transitions[0].links.back().id = fm::newId();
        rejects([&] { fm::validateProject(bad); }, "Conflicting duplicate connections rejected");
        auto manual = fixture();
        manual.transitions[0].links[0].anchors = {{.2f, .35f}};
        manual.transitions[0].links[0].componentPairs = {{0, 0}};
        manual.transitions[0].links[0].bend = .4;
        manual.transitions[0].links[0].rotation = 70;
        fm::Renderer manualRenderer(manual, 320, 180);
        require(!manualRenderer.transitionFrame(0, .5).empty(),
                "Manual contour pairing, boundary anchors, curved paths, and rotation render");
        auto timings = fixture();
        timings.transitions[0].links[0].start = .3;
        timings.transitions[0].links[0].end = .7;
        fm::Renderer timed(timings, 320, 180);
        auto before = timed.transitionFrame(0, .15);
        require(before.at<cv::Vec3b>(90, 80)[1] > 150, "Object waits until its scheduled start");
        auto after = timed.transitionFrame(0, .85);
        require(after.at<cv::Vec3b>(90, 240)[1] > 100, "Object holds target after scheduled end");
        fm::ExportOptions options;
        options.output = root / "test.mp4";
        options.width = 320;
        options.height = 180;
        options.fps = 12;
        fm::exportVideo(p, options);
        require(fm::fs::file_size(options.output) > 1000, "H.264 export creates a nonempty video");
        options.pngSequence = true;
        options.output = root / "sequence";
        fm::exportVideo(p, options);
        int frames = 0;
        for (auto &e : fm::fs::directory_iterator(options.output))
            ++frames;
        require(frames == int(std::ceil(fm::projectDuration(p) * options.fps - 1e-9)),
                "PNG sequence frame count matches timeline duration");
        require(cv::norm(cv::imread((options.output / "frame-000000.png").string()),
                         p.slides[0].image, cv::NORM_INF) == 0,
                "Export and preview share exact keyframe output");
        std::atomic_bool cancel{true};
        options.pngSequence = false;
        options.output = root / "cancel.mp4";
        rejects([&] { fm::exportVideo(p, options, &cancel); }, "Export cancellation is reported");
        require(!fm::fs::exists(options.output), "Cancelled export leaves no final output");
        options.ffmpeg = "/nonexistent/framemorph-encoder";
        options.output = root / "missing-encoder.mp4";
        rejects([&] { fm::exportVideo(p, options); }, "Missing FFmpeg produces a clear failure");
        require(!fm::fs::exists(options.output), "Encoder failure leaves no final output");
        require(std::abs(fm::evaluateExpression("sin(pi/2)+2^3^2", 0) - 513) < 1e-6,
                "Math expressions support functions, constants, and right-associative powers");
        require(fm::evaluateExpression("-x^2", 3) == -9,
                "Unary minus follows mathematical power precedence");
        require(std::abs(fm::evaluateExpression("80*sin(pi*t)", .5) - 80) < 1e-6,
                "Normalized animation time is available as t");
        rejects([] { fm::evaluateExpression("2x", 0); },
                "Invalid implicit multiplication is rejected");
        rejects([] { fm::evaluateExpression("unknown(x)", 0); },
                "Unknown expression functions are rejected");
        fm::Project native;
        native.width = 320;
        native.height = 180;
        native.fps = 12;
        auto ns = fm::blankSlide(320, 180);
        fm::Object shape;
        shape.id = fm::newId();
        shape.name = "__native";
        fm::json shapeSpec = {{"kind", "circle"}, {"x", 30}, {"y", 45},
                              {"w", 70},          {"h", 70}, {"color", "#4ed5b8"}};
        fm::renderDrawing(ns, shape, shapeSpec);
        ns.objects.push_back(shape);
        fm::recomposeSlide(ns);
        require(ns.image.at<cv::Vec3b>(80, 65)[1] > 180, "Native circle draws on a blank slide");
        auto base = ns.baseImage.clone();
        auto second = fm::duplicateSlide(ns);
        require(second.id != ns.id && second.objects[0].id != shape.id &&
                    second.objects[0].name == shape.name,
                "Duplicate slide creates new IDs and preserves animation names");
        auto moved = shapeSpec;
        moved["kind"] = "rectangle";
        moved["x"] = 220;
        fm::renderDrawing(second, second.objects[0], moved);
        fm::recomposeSlide(second);
        require(cv::norm(cv::Vec3d(second.image.at<cv::Vec3b>(80, 65)) -
                         cv::Vec3d(second.background)) < 2,
                "Moving authored shape restores the clean background");
        native.slides = {ns, second};
        fm::reconcileTransitions(native);
        fm::Renderer nativeRenderer(native, 320, 180);
        require(!nativeRenderer.transitionFrame(0, .5).empty(),
                "Authored objects animate through the CUDA contour engine");
        fm::Object text;
        text.id = fm::newId();
        text.name = "text";
        fm::renderDrawing(ns, text,
                          {{"kind", "text"},
                           {"content", "Hello, \xCF\x80!"},
                           {"x", 5},
                           {"y", 2},
                           {"font_size", 20},
                           {"color", "#e9eff6"}});
        require(cv::countNonZero(text.mask) > 150, "FreeType draws UTF-8 text with Greek glyphs");
        fm::Object plot;
        plot.id = fm::newId();
        plot.name = "__plot";
        fm::renderDrawing(ns, plot,
                          {{"kind", "plot"},
                           {"content", "sin(x)"},
                           {"x", 5},
                           {"y", 20},
                           {"w", 300},
                           {"h", 130},
                           {"axes", true},
                           {"stroke", 2}});
        require(cv::countNonZero(plot.mask) > 500, "Function plot renders its curve and axes");
        fm::Object explicitPlot;
        explicitPlot.id = fm::newId();
        explicitPlot.name = "__explicit";
        fm::renderDrawing(ns, explicitPlot,
                          {{"kind", "plot"},
                           {"content", "y=sin(x)"},
                           {"x", 5},
                           {"y", 20},
                           {"w", 300},
                           {"h", 130},
                           {"axes", true},
                           {"stroke", 2}});
        require(cv::norm(explicitPlot.mask, plot.mask, cv::NORM_INF) == 0,
                "y=f(x) syntax produces the same explicit curve");
        fm::Object implicit;
        implicit.id = fm::newId();
        implicit.name = "__implicit";
        fm::json implicitSpec = {{"kind", "implicit"},
                                 {"content", "x^2+y^2=1"},
                                 {"x", 60},
                                 {"y", 10},
                                 {"w", 160},
                                 {"h", 160},
                                 {"x_min", -2},
                                 {"x_max", 2},
                                 {"y_min", -2},
                                 {"y_max", 2},
                                 {"filled", true},
                                 {"stroke", 1}};
        fm::renderDrawing(ns, implicit, implicitSpec);
        require(implicit.mask.at<uchar>(90, 140) == 255 && implicit.mask.at<uchar>(15, 65) == 0,
                "Implicit equation creates a filled circle with correct interior/exterior");
        implicitSpec["content"] = "(x^2+y^2-1)*(x^2+y^2-0.25)=0";
        fm::renderDrawing(ns, implicit, implicitSpec);
        require(implicit.mask.at<uchar>(90, 140) == 0 && implicit.mask.at<uchar>(90, 170) > 0,
                "Implicit filled regions preserve holes between level sets");
        implicitSpec["content"] = "(x^2+y^2-1)^3-x^2*y^3=0";
        implicitSpec["filled"] = false;
        fm::renderDrawing(ns, implicit, implicitSpec);
        require(cv::countNonZero(implicit.mask) > 100 && implicit.mask.at<uchar>(90, 140) == 0,
                "Marching squares creates an outline for a nonlinear implicit heart");
        fm::Object eq;
        eq.id = fm::newId();
        eq.name = "__equation";
        fm::renderDrawing(ns, eq,
                          {{"kind", "equation"},
                           {"content", "\\frac{x^2}{2}+\\alpha"},
                           {"x", 10},
                           {"y", 15},
                           {"font_size", 24}});
        require(cv::countNonZero(eq.mask) > 100,
                "LaTeX-style math notation renders as a selectable object");
        require(cv::countNonZero(eq.mask) < eq.artwork.total() * .85,
                "Equation raster keeps a transparent background");
        cv::Mat pattern(40, 60, CV_8UC4);
        for (int y = 0; y < 40; ++y)
            for (int x = 0; x < 60; ++x)
                pattern.at<cv::Vec4b>(y, x) = {uchar(x * 4), uchar(y * 6), uchar((x + y) * 2), 255};
        auto imageSlide = fm::blankSlide(320, 180);
        fm::Object imageObject;
        imageObject.id = fm::newId();
        imageObject.name = "__image";
        imageObject.sourceAsset = pattern;
        fm::json imageSpec = {{"kind", "image"}, {"x", 20}, {"y", 20}, {"w", 100}, {"h", 65}};
        fm::renderDrawing(imageSlide, imageObject, imageSpec);
        imageSlide.objects.push_back(imageObject);
        fm::recomposeSlide(imageSlide);
        auto imageTarget = fm::duplicateSlide(imageSlide);
        auto destination = imageSpec;
        destination["x"] = 190;
        destination["y"] = 90;
        fm::renderDrawing(imageTarget, imageTarget.objects[0], destination);
        fm::Project images;
        images.width = 320;
        images.height = 180;
        images.fps = 12;
        images.slides = {imageSlide, imageTarget};
        fm::reconcileTransitions(images);
        fm::Renderer imageRenderer(images, 320, 180);
        auto imageMid = imageRenderer.transitionFrame(0, .5);
        std::set<int> unique;
        for (int y = 55; y < 110; ++y)
            for (int x = 110; x < 190; ++x) {
                auto c = imageMid.at<cv::Vec3b>(y, x);
                unique.insert(c[0] * 65536 + c[1] * 256 + c[2]);
            }
        require(unique.size() > 100,
                "Inserted images retain texture and color detail during GPU motion");
        require(cv::norm(cv::Vec3d(imageMid.at<cv::Vec3b>(30, 30)) -
                         cv::Vec3d(imageSlide.background)) < 2,
                "Moving image layers reveal the underlying slide");
        fm::saveProject(images, root / "image-project");
        auto imageReopened = fm::loadProject(root / "image-project");
        require(cv::norm(imageReopened.slides[0].objects[0].sourceAsset, pattern, cv::NORM_INF) ==
                    0,
                "Inserted image originals are embedded for portable editing");
        auto edited = imageReopened.slides[0].objects[0].drawing;
        edited["w"] = 120;
        fm::renderDrawing(imageReopened.slides[0], imageReopened.slides[0].objects[0], edited);
        require(imageReopened.slides[0].objects[0].artwork.cols == 120,
                "Reopened image layers remain editable without their original file");
        auto rotationProject = images;
        rotationProject.slides[0] = fm::duplicateSlide(imageSlide);
        auto pose = imageSpec;
        pose["x"] = 60;
        pose["y"] = 50;
        pose["w"] = 100;
        pose["h"] = 40;
        fm::renderDrawing(rotationProject.slides[0], rotationProject.slides[0].objects[0], pose);
        rotationProject.slides[1] = fm::duplicateSlide(rotationProject.slides[0]);
        pose["angle"] = 90;
        fm::renderDrawing(rotationProject.slides[1], rotationProject.slides[1].objects[0], pose);
        rotationProject.transitions.clear();
        fm::reconcileTransitions(rotationProject);
        fm::Renderer rotationRenderer(rotationProject, 320, 180);
        auto rotatedMid = rotationRenderer.transitionFrame(0, .5);
        require(cv::norm(cv::Vec3d(rotatedMid.at<cv::Vec3b>(70, 150)) -
                         cv::Vec3d(imageSlide.background)) < 3,
                "Image rotation has one interpolated footprint without crossfade ghosts");
        require(cv::norm(cv::Vec3d(rotatedMid.at<cv::Vec3b>(95, 135)) -
                         cv::Vec3d(imageSlide.background)) > 30,
                "Image texture follows the intermediate rotation");
        fm::Renderer squareExport(p, 180, 180);
        auto squareFrame = squareExport.transitionFrame(0, .5);
        require(cv::norm(cv::Vec3d(squareFrame.at<cv::Vec3b>(10, 90)) -
                         cv::Vec3d(p.slides[0].background)) < 3 &&
                    cv::norm(cv::Vec3d(squareFrame.at<cv::Vec3b>(90, 90)) -
                             cv::Vec3d(p.slides[0].background)) > 50,
                "Different output aspect ratios add padding and preserve object geometry");
        fm::saveProject(native, root / "native-project");
        auto nativeLoaded = fm::loadProject(root / "native-project");
        require(nativeLoaded.slides[0].objects[0].drawing == shapeSpec,
                "Authored drawing parameters survive save/reopen");
        nativeLoaded.slides[0].objects.clear();
        fm::recomposeSlide(nativeLoaded.slides[0]);
        require(cv::norm(nativeLoaded.slides[0].image, base, cv::NORM_INF) == 0,
                "Deleting authored objects restores the original base image");
        cv::imwrite((root / "native-project/assets/user-kept.png").string(), base);
        fm::saveProject(native, root / "native-project");
        require(fm::fs::exists(root / "native-project/assets/user-kept.png"),
                "Snapshot collection preserves unrelated user assets");
        auto custom = fixture();
        custom.transitions[0].links[0].offsetY = "-30*sin(pi*t)";
        custom.transitions[0].links[0].scaleFunction = "1+0.1*sin(pi*t)";
        fm::Renderer customRenderer(custom, 320, 180);
        auto customMid = customRenderer.transitionFrame(0, .5);
        require(cv::norm(cv::Vec3d(customMid.at<cv::Vec3b>(60, 160)) -
                         cv::Vec3d(custom.slides[0].background)) > 50,
                "Motion functions control offsets and scale in rendered frames");
        custom.transitions[0].links[0].scaleFunction = "-1";
        rejects([&] { fm::validateProject(custom); },
                "Invalid negative function scale is rejected");
        auto transformSource = fixture().slides[0];
        auto sourceId = transformSource.objects[0].id;
        auto originalSource = transformSource.image.clone();
        fm::json translation = {
            {"kind", "matrix"}, {"matrix", "1,0,1;0,1,0;0,0,1"}, {"units", 100.}};
        auto transformed = fm::generateKeyframe(transformSource, {{sourceId, translation}});
        require(transformed.objects[0].name == transformSource.objects[0].name &&
                    transformed.objects[0].id != sourceId,
                "Generated keyframe preserves names and creates independent object IDs");
        require(cv::norm(transformSource.image, originalSource, cv::NORM_INF) == 0,
                "Keyframe generation leaves the source slide unchanged");
        require(transformed.objects[0].mask.at<uchar>(90, 180) > 200 &&
                    transformed.objects[0].mask.at<uchar>(90, 80) == 0,
                "Homogeneous matrix transforms only the selected object's position");
        fm::Project generated;
        generated.width = 320;
        generated.height = 180;
        generated.fps = 12;
        generated.slides = {transformSource, transformed};
        fm::reconcileTransitions(generated);
        fm::Renderer generatedRenderer(generated, 320, 180);
        auto generatedMid = generatedRenderer.transitionFrame(0, .5);
        require(cv::norm(cv::Vec3d(generatedMid.at<cv::Vec3b>(90, 130)) -
                         cv::Vec3d(transformSource.background)) > 50,
                "Generated matrix keyframe animates along the actual transformation mesh");
        require(cv::norm(generatedMid, generatedRenderer.transitionFrame(0, .5), cv::NORM_INF) == 0,
                "Transformation mesh rendering is deterministic");
        auto complexTranslation = fm::generateKeyframe(
            transformSource,
            {{sourceId, {{"kind", "complex"}, {"formula", "z+1"}, {"units", 100.}}}});
        require(cv::norm(complexTranslation.objects[0].mask, transformed.objects[0].mask,
                         cv::NORM_INF) < 3,
                "Complex translation agrees with the equivalent matrix");
        auto squared = fm::generateKeyframe(
            transformSource,
            {{sourceId, {{"kind", "complex"}, {"formula", "z^2"}, {"units", 100.}}}});
        int squaredArea = cv::countNonZero(squared.objects[0].mask);
        require(squaredArea > 100 && squaredArea < 700,
                "Nonlinear complex z^2 maps a filled object without parity cancellation");
        auto coordinateMap = fm::generateKeyframe(
            transformSource,
            {{sourceId, {{"kind", "map"}, {"x", "x+0.3*y^2"}, {"y", "y"}, {"units", 100.}}}});
        require(cv::countNonZero(coordinateMap.objects[0].mask) > 1000,
                "Coordinate functions generate a nonlinear warped keyframe");
        auto rotated = fm::generateKeyframe(
            transformSource, {{sourceId,
                               {{"kind", "matrix"},
                                {"matrix", "cos(pi/4),-sin(pi/4);sin(pi/4),cos(pi/4)"},
                                {"units", 100.}}}});
        require(cv::countNonZero(rotated.objects[0].mask) > 2000,
                "Matrix entries accept mathematical expressions");
        auto unchanged = transformSource;
        fm::Object marker;
        marker.id = fm::newId();
        marker.name = "__marker";
        marker.mask = cv::Mat(180, 320, CV_8U, cv::Scalar(0));
        cv::circle(marker.mask, {240, 140}, 10, 255, -1);
        unchanged.image = unchanged.image.clone();
        unchanged.image.setTo(cv::Scalar(230, 60, 180), marker.mask);
        unchanged.objects.push_back(marker);
        auto selectedOnly = fm::generateKeyframe(unchanged, {{sourceId, translation}});
        require(selectedOnly.image.at<cv::Vec3b>(140, 240) ==
                        unchanged.image.at<cv::Vec3b>(140, 240) &&
                    cv::norm(selectedOnly.objects[1].mask, marker.mask, cv::NORM_INF) == 0,
                "Objects without staged transformations keep their original pose");
        fm::json markerTransform = {{"kind", "complex"}, {"formula", "z-0.2"}, {"units", 100.}};
        std::map<std::string, fm::json> batchOperations{{sourceId, translation},
                                                        {marker.id, markerTransform}};
        auto batch = fm::generateKeyframe(unchanged, batchOperations);
        require(batch.objects[1].mask.at<uchar>(140, 220) > 200 &&
                    batch.objects[0].mask.at<uchar>(90, 180) > 200,
                "Several per-object transformations generate one next keyframe together");
        auto repeatedSource = transformSource;
        repeatedSource.image = transformSource.image.clone();
        auto twin = transformSource.objects[0];
        twin.id = fm::newId();
        twin.mask = cv::Mat(180, 320, CV_8U, cv::Scalar(0));
        cv::circle(twin.mask, {240, 90}, 30, 255, -1);
        repeatedSource.image.setTo(cv::Scalar(200, 210, 70), twin.mask);
        repeatedSource.objects.push_back(twin);
        fm::json right = {{"kind", "complex"}, {"formula", "z+1.6"}, {"units", 100.}},
                 left = {{"kind", "complex"}, {"formula", "z-1.6"}, {"units", 100.}};
        auto swapped = fm::generateKeyframe(repeatedSource, {{sourceId, right}, {twin.id, left}});
        auto swappedLinks = fm::automaticLinks(repeatedSource, swapped);
        bool lineagePreserved = swappedLinks.size() == 2;
        for (auto &link : swappedLinks) {
            auto it = std::find_if(swapped.objects.begin(), swapped.objects.end(),
                                   [&](const auto &o) { return o.id == link.targets[0]; });
            lineagePreserved &=
                it != swapped.objects.end() && it->generation["source_id"] == link.sources[0];
        }
        require(lineagePreserved,
                "Generated repeated-name objects retain identity even when they exchange places");
        auto compactMatrix = fm::generateKeyframe(
            transformSource,
            {{sourceId, {{"kind", "matrix"}, {"matrix", "1,0,1\n0,1,0"}, {"units", 100.}}}});
        require(
            cv::norm(compactMatrix.objects[0].mask, transformed.objects[0].mask, cv::NORM_INF) == 0,
            "Multiline 2x3 matrices agree with homogeneous 3x3 matrices");
        fm::saveProject(generated, root / "generated-project");
        auto regenerated = fm::loadProject(root / "generated-project");
        require(regenerated.slides[1].objects[0].generation["operation"]["matrix"] ==
                    translation["matrix"],
                "Transformation recipes survive save and reopen");
        fm::Renderer regeneratedRenderer(regenerated, 320, 180);
        require(cv::norm(regeneratedRenderer.transitionFrame(0, .5), generatedMid, cv::NORM_INF) ==
                    0,
                "Reopened generated keyframe preserves its deformation animation");
        rejects(
            [&] {
                fm::generateKeyframe(transformSource,
                                     {{sourceId, {{"kind", "matrix"}, {"matrix", "1,2,3"}}}});
            },
            "Malformed transformation matrices are rejected");
        rejects(
            [&] {
                fm::generateKeyframe(
                    transformSource,
                    {{sourceId, {{"kind", "complex"}, {"formula", "z"}, {"units", 0}}}});
            },
            "Zero transformation scale is rejected");
        rejects([&] { fm::generateKeyframe(transformSource, {{"missing-object", translation}}); },
                "Staged transformations cannot refer to missing objects");
        rejects(
            [&] {
                fm::generateKeyframe(
                    transformSource,
                    {{sourceId, {{"kind", "complex"}, {"formula", "unknown(z)"}}}});
            },
            "Unknown complex functions produce a clear rejection");
        auto direct = fixture();
        auto directBefore = direct.slides[0].image.clone();
        auto sharedBefore = direct;
        auto movedFrame = fm::objectFrame(direct.slides[0].objects[0]);
        movedFrame["x"] = 150.;
        movedFrame["y"] = 50.;
        fm::setObjectFrame(direct.slides[0], direct.slides[0].objects[0], movedFrame);
        require(direct.slides[0].image.at<cv::Vec3b>(90, 80) == direct.slides[0].background,
                "Moving a pixel object removes its original footprint from the slide");
        require(
            cv::norm(sharedBefore.slides[0].image, directBefore, cv::NORM_INF) == 0 &&
                sharedBefore.slides[0].objects[0].drawing.is_null(),
            "Direct manipulation preserves immutable snapshots without copying unrelated assets");
        require(direct.slides[0].objects[0].drawing["construction"]["kind"] == "pixels",
                "Moved pixel objects retain their construction category");
        auto beforeColor = direct;
        fm::setObjectColor(direct.slides[0], direct.slides[0].objects[0], "#f04060");
        require(beforeColor.slides[0].objects[0].sourceAsset.at<cv::Vec4b>(30, 30) !=
                    direct.slides[0].objects[0].sourceAsset.at<cv::Vec4b>(30, 30),
                "Recoloring uses copy-on-write and preserves prior image assets");
        auto invisible = fm::objectFrame(direct.slides[0].objects[0]);
        invisible["opacity"] = 0.;
        fm::setObjectFrame(direct.slides[0], direct.slides[0].objects[0], invisible);
        require(cv::countNonZero(direct.slides[0].objects[0].mask) > 0 &&
                    direct.slides[0].image.at<cv::Vec3b>(80, 180) == direct.slides[0].background,
                "Zero opacity hides artwork while preserving selectable geometry");
        invisible["opacity"] = 1.;
        fm::setObjectFrame(direct.slides[0], direct.slides[0].objects[0], invisible);
        require(direct.slides[0].image.at<cv::Vec3b>(80, 180) != direct.slides[0].background,
                "Increasing opacity restores cached artwork without reconstructing it");
        fm::saveProject(direct, root / "direct-project");
        auto directReload = fm::loadProject(root / "direct-project");
        require(
            cv::norm(directReload.slides[0].image, direct.slides[0].image, cv::NORM_INF) == 0 &&
                directReload.slides[0].objects[0].drawing.contains("construction"),
            "Directly manipulated object geometry, construction, and artwork survive reopening");
        auto naming = fm::blankSlide(320, 180);
        fm::Object named;
        named.name = fm::defaultObjectName(naming, "circle");
        naming.objects.push_back(named);
        require(named.name == "__circle_1" &&
                    fm::defaultObjectName(naming, "circle") == "__circle_2",
                "Default object names are unique within a slide");
        auto invalidFrame = invisible;
        invalidFrame["w"] = -1;
        rejects(
            [&] {
                fm::setObjectFrame(direct.slides[0], direct.slides[0].objects[0], invalidFrame);
            },
            "Invalid direct-manipulation dimensions are rejected");
        rejects(
            [&] { fm::setObjectColor(direct.slides[0], direct.slides[0].objects[0], "#gg0055"); },
            "Invalid property colors are rejected before editing assets");

        // Prefix inference, local coordinates, scene symbols, transactionality, and saved recipes.
        auto mapSource = fm::blankSlide(640, 360, {255, 255, 255});
        fm::Object mapObject;
        mapObject.id = fm::newId();
        mapObject.name = "__map";
        fm::renderDrawing(mapSource, mapObject,
                          {{"kind", "rectangle"},
                           {"x", 100},
                           {"y", 100},
                           {"w", 80},
                           {"h", 40},
                           {"color", "#2050b0"}});
        mapSource.objects.push_back(mapObject);
        fm::recomposeSlide(mapSource);
        auto mapSlide = fm::duplicateSlide(mapSource);
        auto &mapped = mapSlide.objects[0];
        auto stableId = mapped.id;
        auto originalBounds = cv::boundingRect(mapped.mask);
        auto pointSpec = fm::parseObjectMap(" ( x , y ) : x+0.5,y+0.25", mapSlide, mapped);
        fm::applyObjectMap(mapSlide, mapped, pointSpec);
        auto translatedBounds = cv::boundingRect(mapped.mask);
        require(
            std::abs(translatedBounds.x - originalBounds.x - 50) <= 1 &&
                std::abs(translatedBounds.y - originalBounds.y + 25) <= 1,
            "Point-map prefix is inferred and translations use center-relative X-right Y-up units");
        require(mapped.id == stableId && mapped.name == "__map" &&
                    mapped.generation["source_id"] == mapObject.id,
                "In-place mapping preserves identity and links a duplicated pose to its source");
        fm::applyObjectMap(mapSlide, mapped, fm::parseObjectMap("z:z+0.2*i", mapSlide, mapped));
        require(mapped.generation["operation"]["operations"].size() == 2,
                "Repeated in-place maps compose into one non-nested recipe");
        fm::Project mappedProject;
        mappedProject.width = 640;
        mappedProject.height = 360;
        mappedProject.slides = {mapSource, mapSlide};
        fm::reconcileTransitions(mappedProject);
        fm::Renderer mapRenderer(mappedProject, 640, 360);
        auto mapMid = mapRenderer.transitionFrame(0, .5);
        require(
            cv::norm(cv::Vec3d(mapMid.at<cv::Vec3b>(97, 165)) - cv::Vec3d(255, 255, 255)) > 50,
            "Duplicated keyframe with sequential maps renders the actual deformation at midpoint");
        auto third = fm::duplicateSlide(mapSlide);
        fm::applyObjectMap(third, third.objects[0],
                           fm::parseObjectMap("z:z-0.25", third, third.objects[0]));
        require(third.objects[0].generation["operation"]["kind"] == "complex" &&
                    third.objects[0].generation["source_signature"] ==
                        fm::objectTransformSignature(mapSlide, mapped),
                "Duplicating a mapped slide starts a fresh incoming map recipe");
        fm::saveProject(mappedProject, root / "inplace-project");
        auto loadedMapped = fm::loadProject(root / "inplace-project");
        fm::Renderer loadedMapRenderer(loadedMapped, 640, 360);
        require(cv::norm(mapMid, loadedMapRenderer.transitionFrame(0, .5), cv::NORM_INF) == 0,
                "Sequential in-place deformation survives a portable save and reopen");
        auto centerTest = mapSource;
        fm::applyObjectMap(centerTest, centerTest.objects[0],
                           fm::parseObjectMap("z:z+origin", centerTest, centerTest.objects[0]));
        auto centered = cv::boundingRect(centerTest.objects[0].mask);
        require(std::abs(centered.x + centered.width * .5 - 320) < 1 &&
                    std::abs(centered.y + centered.height * .5 - 180) < 1,
                "Built-in complex origin moves an object exactly to the scene center");
        auto pointCenter = mapSource;
        fm::applyObjectMap(pointCenter, pointCenter.objects[0],
                           fm::parseObjectMap("(x,y):x+scene.origin_x,y+origin_y", pointCenter,
                                              pointCenter.objects[0]));
        require(cv::norm(centerTest.objects[0].mask, pointCenter.objects[0].mask, cv::NORM_INF) ==
                    0,
                "Point and complex maps agree on scene origin variables");
        auto pixelMap = mapSource;
        fm::applyObjectMap(
            pixelMap, pixelMap.objects[0],
            fm::parseObjectMap("(x,y):x+20*px,y", pixelMap, pixelMap.objects[0], 80));
        require(std::abs(cv::boundingRect(pixelMap.objects[0].mask).x - originalBounds.x - 20) <= 1,
                "px stays exactly one pixel when mathematical unit scale changes");
        auto vars = fm::sceneVariables(mapSource, &mapObject);
        require(fm::compileExpression2D("scene.origin_x-object.width/2", vars)(0, 0) ==
                    320 - vars.at("object.width") / 2,
                "Property expressions resolve scene and object pixel variables without substring "
                "replacement");
        require(
            fm::parseObjectMap("(x,y):x+sin(y),cos(x)", mapSource, mapObject)["kind"] == "map" &&
                fm::parseObjectMap("z:z*exp(i*pi/4)", mapSource, mapObject)["kind"] == "complex",
            "Function prefixes infer point or complex parsing without a type dropdown");
        for (const auto &bad :
             {"x+1,y", "(x,y):x,y,z", "(x,y):sin(x,y", "z:does_not_exist", "(x,y):x+,y"})
            rejects([&] { fm::parseObjectMap(bad, mapSource, mapObject); },
                    std::string("Reject malformed object map: ") + bad);
        auto failureImage = mapSlide.image.clone();
        auto failureSignature = fm::objectTransformSignature(mapSlide, mapped);
        rejects(
            [&] {
                fm::applyObjectMap(mapSlide, mapped,
                                   fm::parseObjectMap("z:z+100000", mapSlide, mapped));
            },
            "Entirely off-canvas in-place maps are rejected");
        require(fm::objectTransformSignature(mapSlide, mapped) == failureSignature &&
                    cv::norm(failureImage, mapSlide.image, cv::NORM_INF) == 0,
                "Failed mapping leaves both object buffers and slide pixels unchanged");
        auto rotationSource = fm::blankSlide(640, 360, {255, 255, 255});
        fm::Object thin;
        thin.id = fm::newId();
        thin.name = "__bar";
        fm::renderDrawing(rotationSource, thin,
                          {{"kind", "rectangle"},
                           {"x", 200},
                           {"y", 140},
                           {"w", 200},
                           {"h", 16},
                           {"color", "#202020"}});
        rotationSource.objects.push_back(thin);
        fm::recomposeSlide(rotationSource);
        auto rotationTarget = fm::duplicateSlide(rotationSource);
        auto rotateFrame = fm::objectFrame(rotationTarget.objects[0]);
        rotateFrame["angle"] = 90.;
        fm::setObjectFrame(rotationTarget, rotationTarget.objects[0], rotateFrame);
        fm::Project rigidProject;
        rigidProject.width = 640;
        rigidProject.height = 360;
        rigidProject.slides = {rotationSource, rotationTarget};
        fm::reconcileTransitions(rigidProject);
        fm::Renderer rigid(rigidProject, 640, 360);
        auto diagonal = rigid.transitionFrame(0, .5);
        cv::Mat gray, bar;
        cv::cvtColor(diagonal, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, bar, 160, 255, cv::THRESH_BINARY_INV);
        std::vector<cv::Point> barPixels;
        cv::findNonZero(bar, barPixels);
        auto rect = cv::minAreaRect(barPixels);
        require(std::max(rect.size.width, rect.size.height) > 195 &&
                    std::min(rect.size.width, rect.size.height) < 20,
                "Mid-rotation preserves a thin object's length and width instead of collapsing its "
                "contours");
        require(rigid.warnings().empty(),
                "Rigid pose edits do not trigger self-intersection fade-through");
        cv::imwrite((root / "smooth-rotation-midpoint.png").string(), diagonal);

        {
            require(fm::compileExpression3D("x+2*y+3*t")(4, 5, .25) == 14.75,
                    "Updater expressions keep x, y and normalized time independent");
            auto updaterSource = fm::blankSlide(640, 360, {255, 255, 255});
            fm::Object mover;
            mover.id = fm::newId();
            mover.name = "__orbit";
            fm::renderDrawing(updaterSource, mover,
                              {{"kind", "circle"},
                               {"x", 370},
                               {"y", 150},
                               {"w", 32},
                               {"h", 32},
                               {"color", "#386fe8"}});
            updaterSource.objects.push_back(mover);
            fm::recomposeSlide(updaterSource);
            auto orbit = fm::parseObjectUpdater("(x,y,t):x+0.8*(cos(tau*t)-1),y+0.8*sin(tau*t)",
                                                updaterSource, mover);
            auto orbitPlan = fm::prepareObjectUpdater(updaterSource, mover, orbit, 2.);
            require(orbitPlan.affine && orbitPlan.source.from.size() == 4,
                    "Circular translation uses a four-corner updater instead of a dense "
                    "deformation mesh");
            auto warpSpec = fm::parseObjectUpdater("z:z+0.2*t*z^2", updaterSource, mover);
            auto warpPlan = fm::prepareObjectUpdater(updaterSource, mover, warpSpec, 2.);
            require(!warpPlan.affine && warpPlan.source.from.size() > 4,
                    "Nonlinear spatial formulas use the dense mesh rather than an affine guess");
            auto nonlinearPoint =
                fm::parseObjectUpdater("(x,y,t):x+t*sin(y),y", updaterSource, mover);
            require(!fm::prepareObjectUpdater(updaterSource, mover, nonlinearPoint, 2.).affine,
                    "A spatial sine cannot enter the affine fast path");
            auto secondsSpec = fm::parseObjectUpdater("z:z+seconds", updaterSource, mover);
            auto secondsPlan = fm::prepareObjectUpdater(updaterSource, mover, secondsSpec, 2.);
            auto point = cv::Point2f(386, 166);
            require(std::abs(secondsPlan.evaluate(point, .5).x - point.x - 100) < .01,
                    "Two-second steps expose seconds=2*t without multiplying normalized-time "
                    "coordinates");
            auto slowerPlan = fm::prepareObjectUpdater(updaterSource, mover, secondsSpec, 4.);
            require(std::abs(slowerPlan.evaluate(point, .5).x - point.x - 200) < .01,
                    "Seconds-based updater rates scale with the chosen step duration");
            auto loopTarget = fm::generateUpdaterKeyframe(updaterSource, {{mover.id, orbit}}, 2.);
            require(cv::norm(loopTarget.image, updaterSource.image, cv::NORM_INF) == 0,
                    "A closed updater loop preserves exact equal start and end keyframes");
            fm::Project loop;
            loop.width = 640;
            loop.height = 360;
            loop.slides = {updaterSource, loopTarget};
            fm::reconcileTransitions(loop);
            loop.transitions[0].duration = 2.;
            loop.transitions[0].links[0].easing = "linear";
            fm::Renderer loopRenderer(loop, 640, 360);
            auto quarter = loopRenderer.transitionFrame(0, .25),
                 half = loopRenderer.transitionFrame(0, .5);
            require(cv::norm(cv::Vec3d(quarter.at<cv::Vec3b>(86, 306)) - cv::Vec3d(255, 255, 255)) >
                            50 &&
                        cv::norm(cv::Vec3d(half.at<cv::Vec3b>(166, 226)) -
                                 cv::Vec3d(255, 255, 255)) > 50,
                    "Equal endpoint poses still animate along the circular updater path");
            require(quarter.at<cv::Vec3b>(166, 386) == cv::Vec3b(255, 255, 255),
                    "Updater animation leaves no stationary source duplicate");
            loopRenderer.transitionFrame(0, .9);
            require(cv::norm(quarter, loopRenderer.transitionFrame(0, .25), cv::NORM_INF) == 0,
                    "Updater scrubbing evaluates the original pose and is independent of playback "
                    "history");
            loop.slides[0].objects[0].updater = orbit;
            fm::saveProject(loop, root / "updater-project");
            auto reopenedLoop = fm::loadProject(root / "updater-project");
            require(
                fm::projectMetadata(reopenedLoop)["version"] == 3 &&
                    reopenedLoop.slides[0].objects[0].updater["input"] == orbit["input"],
                "Versioned projects preserve draft updaters and protect their time-based recipes");
            fm::Renderer savedLoop(reopenedLoop, 640, 360);
            require(cv::norm(quarter, savedLoop.transitionFrame(0, .25), cv::NORM_INF) == 0,
                    "Circular updater playback survives save and reopen");
            auto plainDuplicate = fm::duplicateSlide(loop.slides[0]);
            require(plainDuplicate.objects[0].updater.is_null() &&
                        cv::norm(plainDuplicate.image, loop.slides[0].image, cv::NORM_INF) == 0,
                    "Ordinary slide duplication does not execute or copy pending updater drafts");
            auto staleLoop = loop;
            staleLoop.transitions[0].duration = 3.;
            rejects([&] { fm::validateProject(staleLoop); },
                    "Changing committed updater duration requires explicit regeneration");
            rejects(
                [&] {
                    fm::prepareObjectUpdater(updaterSource, mover,
                                             fm::parseObjectUpdater("z:z+1", updaterSource, mover),
                                             2.);
                },
                "Non-identity updater starts are rejected to prevent an initial jump");
            rejects(
                [&] {
                    fm::prepareObjectUpdater(
                        updaterSource, mover,
                        fm::parseObjectUpdater("z:z/(1-2*t)", updaterSource, mover), 2.);
                },
                "A singular updater is caught by temporal preflight");
            rejects([&] { fm::generateUpdaterKeyframe(updaterSource, {{mover.id, orbit}}, 0); },
                    "Zero updater step duration is rejected");
            rejects([&] { fm::generateUpdaterKeyframe(updaterSource, {}, 2); },
                    "Generating with no saved updaters is rejected");
            auto other = updaterSource.objects[0];
            other.id = fm::newId();
            other.name = "__second";
            fm::renderDrawing(updaterSource, other,
                              {{"kind", "rectangle"},
                               {"x", 120},
                               {"y", 260},
                               {"w", 24},
                               {"h", 24},
                               {"color", "#e0a030"}});
            updaterSource.objects.push_back(other);
            fm::recomposeSlide(updaterSource);
            auto shift = fm::parseObjectUpdater("z:z+0.5*t", updaterSource, other);
            auto batchUpdaters = fm::generateUpdaterKeyframe(
                updaterSource, {{mover.id, orbit}, {other.id, shift}}, 2.);
            require(std::abs(cv::boundingRect(batchUpdaters.objects[1].mask).x -
                             cv::boundingRect(other.mask).x - 50) <= 1,
                    "Several saved object updaters generate one next slide together");
            auto onlyFirst = fm::generateUpdaterKeyframe(updaterSource, {{mover.id, orbit}}, 2.);
            require(cv::norm(onlyFirst.objects[1].mask, other.mask, cv::NORM_INF) == 0 &&
                        cv::norm(onlyFirst.image, updaterSource.image, cv::NORM_INF) == 0,
                    "Objects without updaters retain their original pose");
            auto profiledLoop = fm::profileProject(loop, 640, 360, 12);
            require(
                profiledLoop["profiling_preserves_pixels"].get<bool>() &&
                    profiledLoop["cuda_event_ms"].contains("mesh_raster") &&
                    profiledLoop["downloaded_bytes_per_frame"].get<double>() == 640. * 360 * 3,
                "Opt-in CUDA timing reports actual stages and byte counts without altering pixels");
            cv::imwrite((root / "updater-quarter.png").string(), quarter);
            // Compare like-for-like linear translations, with profiling disabled during the warm
            // timings.
            auto ordinarySource = updaterSource;
            auto ordinaryTarget = fm::duplicateSlide(ordinarySource);
            auto moved = fm::objectFrame(ordinaryTarget.objects[0]);
            moved["x"] = moved["x"].get<double>() + 100.;
            fm::setObjectFrame(ordinaryTarget, ordinaryTarget.objects[0], moved);
            fm::Project ordinary;
            ordinary.width = 640;
            ordinary.height = 360;
            ordinary.slides = {ordinarySource, ordinaryTarget};
            fm::reconcileTransitions(ordinary);
            ordinary.transitions[0].duration = 2.;
            for (auto &l : ordinary.transitions[0].links)
                l.easing = "linear";
            auto linearSpec =
                fm::parseObjectUpdater("z:z+t", ordinarySource, ordinarySource.objects[0]);
            auto updatedLinear = fm::generateUpdaterKeyframe(
                ordinarySource, {{ordinarySource.objects[0].id, linearSpec}}, 2.);
            auto withUpdater = ordinary;
            withUpdater.slides[1] = updatedLinear;
            withUpdater.transitions.clear();
            fm::reconcileTransitions(withUpdater);
            withUpdater.transitions[0].duration = 2.;
            auto plainReport = fm::profileProject(ordinary, 960, 540, 60),
                 updaterReport = fm::profileProject(withUpdater, 960, 540, 60);
            fm::json comparison = {
                {"ordinary", plainReport}, {"updater", updaterReport}, {"circle", profiledLoop}};
            std::ofstream report(root / "updater-performance.json");
            report << comparison.dump(2);
            std::cout << "UPDATER BENCHMARK 960x540 translation: ordinary "
                      << plainReport["mean_frame_ms"] << " ms, updater "
                      << updaterReport["mean_frame_ms"] << " ms\n";
        }

        {
            fm::CudaBackend regionGpu;
            cv::Mat background(180, 320, CV_8UC3, cv::Scalar(220, 230, 240));
            cv::Mat tex(48, 64, CV_8UC4, cv::Scalar(0, 0, 0, 0));
            cv::circle(tex, {32, 24}, 20, cv::Scalar(80, 160, 240, 210), -1, cv::LINE_AA);
            bool identical = true, cacheIdentical = true;
            fm::CudaBackend cachedGpu;
            cachedGpu.enableInputCaching(true);
            cachedGpu.enableProfiling(true);
            for (float t : {.03f, .25f, .5f, .76f, .98f}) {
                fm::GpuDraw shape;
                shape.lineage = 1;
                shape.t = t;
                shape.color = {.1f, .2f, .5f, .8f};
                shape.from = {{-20, 10}, {90, 10}, {90, 90}, {-20, 90},
                              {15, 30},  {45, 30}, {45, 65}, {15, 65}};
                shape.to = shape.from;
                for (auto &point : shape.to) {
                    point.x += 190;
                    point.y += 65;
                }
                shape.contours = {{0, 4}, {4, 4}};
                shape.bounds = {-25, 5, 320, 160};
                auto copy = shape;
                for (auto &point : copy.from)
                    point.x += 2;
                for (auto &point : copy.to)
                    point.x += 2;
                fm::GpuDraw sprite;
                sprite.lineage = 2;
                sprite.textureA = tex;
                sprite.textureB = tex;
                sprite.centerX = -15 + 355 * t;
                sprite.centerY = 130;
                sprite.extentX = 90;
                sprite.extentY = 60;
                sprite.angle = t * 3.14f;
                sprite.bounds = {int(sprite.centerX - 60), 70, 120, 120};
                sprite.t = t;
                fm::GpuDraw mesh;
                mesh.lineage = 3;
                mesh.textureA = tex;
                mesh.t = 1;
                mesh.from = {{0, 0}, {64, 0}, {64, 48}, {0, 48}};
                mesh.to = {{160, 20}, {240, 35}, {220, 100}, {140, 80}};
                mesh.meshUV = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
                mesh.triangles = {0, 1, 2, 0, 2, 3};
                std::vector<fm::GpuDraw> draws = {shape, copy, sprite, mesh};
                regionGpu.enableRegionCompositing(false);
                auto full = regionGpu.render(background, background, .5, draws);
                regionGpu.enableRegionCompositing(true);
                auto clipped = regionGpu.render(background, background, .5, draws);
                identical &= cv::norm(full, clipped, cv::NORM_INF) == 0;
                cacheIdentical &=
                    cv::norm(clipped, cachedGpu.render(background, background, .5, draws),
                             cv::NORM_INF) == 0;
            }
            require(cacheIdentical, "Resident GPU inputs preserve overlapping copies, holes, "
                                    "clipped sprites and warped meshes exactly");
            require(identical,
                    "Region-limited clearing/compositing exactly matches full-frame rendering for "
                    "holes, overlapping copies, clipped sprites and warped meshes");
        }
        {
            fm::CudaBackend reference, cached;
            cached.enableInputCaching(true, 1600);
            cached.enableProfiling(true);
            cv::Mat bg(16, 16, CV_8UC3, cv::Scalar(20, 40, 60));
            fm::GpuDraw d;
            d.lineage = 1;
            d.centerX = 8;
            d.centerY = 8;
            d.extentX = 14;
            d.extentY = 14;
            d.bounds = {0, 0, 16, 16};
            d.t = .35;
            d.textureA = cv::Mat(16, 16, CV_8UC4, cv::Scalar(40, 100, 200, 190));
            d.textureB = cv::Mat(16, 16, CV_8UC4, cv::Scalar(210, 50, 80, 220));
            bool identical = true, bounded = true;
            for (int i = 0; i < 24; ++i) {
                d.textureA = d.textureA.clone();
                d.textureA.setTo(cv::Scalar(i * 9, 100, 200, 190));
                auto a = reference.render(bg, bg, .2, {d}), b = cached.render(bg, bg, .2, {d});
                identical &= cv::norm(a, b, cv::NORM_INF) == 0;
                bounded &= cached.inputCacheBytes() <= 1600;
            }
            require(identical && bounded,
                    "Tiny GPU cache preserves pixels across eviction and pinned-texture fallback");
            cached.enableInputCaching(true);
            cached.render(bg, bg, .2, {d});
            cached.render(bg, bg, .2, {d});
            auto stats = cached.lastProfile();
            require(stats.uploadedBytes == 0 && stats.inputCacheHits == 4 &&
                        stats.residentInputBytes > 0,
                    "Warm immutable backgrounds and sprites avoid all explicit image uploads");
            cached.enableInputCaching(false);
            require(cached.inputCacheBytes() == 0,
                    "Disabling GPU input cache releases its resident allocations");
            auto before = reference.render(bg, bg, 0, {});
            bg.setTo(cv::Scalar(100, 120, 140));
            require(cv::norm(before, reference.render(bg, bg, 0, {}), cv::NORM_INF) > 0,
                    "Default raw CUDA backend observes mutable caller inputs");
            cv::Mat parent(128, 128, CV_8UC3, cv::Scalar(17, 37, 97));
            auto roi = parent(cv::Rect(3, 4, 16, 16));
            cached.enableInputCaching(true, 1600);
            auto roiResult = cached.render(roi, roi, .4, {});
            require(
                cv::norm(roiResult, reference.render(roi.clone(), roi.clone(), .4, {}),
                         cv::NORM_INF) == 0 &&
                    cached.inputCacheBytes() == 0,
                "Strided ROI renders correctly without retaining an oversized parent allocation");
            std::vector<uchar> external(16 * 16 * 3, 90);
            cv::Mat unowned(16, 16, CV_8UC3, external.data());
            cached.enableInputCaching(true);
            cached.render(unowned, unowned, 0, {});
            require(cached.inputCacheBytes() == 0,
                    "Externally owned image memory bypasses resident GPU caching");
            auto p = fixture();
            fm::Renderer a(p, 320, 180), b(p, 320, 180);
            a.enableInputCaching(false);
            bool exact = true;
            for (float t : {0.f, .001f, .15f, .5f, .85f, .999f, 1.f, .3f})
                exact &=
                    cv::norm(a.transitionFrame(0, t), b.transitionFrame(0, t), cv::NORM_INF) == 0;
            require(exact,
                    "Resident GPU inputs preserve contour morphs and exact keyframe endpoints");
        }
        {
            fm::Project p;
            p.width = 640;
            p.height = 360;
            auto s = fm::blankSlide(640, 360, {255, 255, 255});
            for (int i = 0; i < 4; ++i) {
                fm::Object o;
                o.id = fm::newId();
                o.name = "__steady" + std::to_string(i);
                fm::renderDrawing(s, o,
                                  {{"kind", i % 2 ? "text" : "circle"},
                                   {"x", 30. + 150 * i},
                                   {"y", 100.},
                                   {"w", 120.},
                                   {"h", 120.},
                                   {"font_size", 40},
                                   {"content", "Abc"},
                                   {"color", "#152850"},
                                   {"opacity", i < 2 ? 1. : .5}});
                s.objects.push_back(o);
            }
            fm::recomposeSlide(s);
            p.slides.push_back(s);
            auto next = s;
            next.id = fm::newId();
            for (auto &o : next.objects)
                o.id = fm::newId();
            p.slides.push_back(next);
            fm::reconcileTransitions(p);
            fm::Renderer renderer(p, 640, 360);
            bool exact = true;
            for (float t : {.00001f, .1f, .5f, .9f, .99999f})
                exact &= cv::norm(s.image, renderer.transitionFrame(0, t), cv::NORM_INF) == 0;
            require(exact, "Stationary opaque/translucent shapes and text have no edge or opacity "
                           "flash at playback start or endpoints");
            cv::imwrite((root / "stable-transparency-keyframe.png").string(), s.image);
            cv::imwrite((root / "stable-transparency-motion.png").string(),
                        renderer.transitionFrame(0, .5));
            fm::CudaBackend gpu;
            cv::Mat bg(16, 16, CV_8UC3, cv::Scalar(255, 255, 255));
            fm::GpuDraw d;
            d.lineage = 1;
            d.bounds = {0, 0, 16, 16};
            d.centerX = 8;
            d.centerY = 8;
            d.extentX = 16;
            d.extentY = 16;
            bool coverage = true;
            for (int alpha : {0, 1, 31, 64, 127, 128, 192, 254, 255}) {
                d.textureA = cv::Mat(16, 16, CV_8UC4, cv::Scalar(20, 80, 140, alpha));
                d.textureB = d.textureA;
                auto frame = gpu.render(bg, bg, 0, {d});
                auto pixel = frame.at<cv::Vec3b>(8, 8);
                for (int c = 0; c < 3; ++c) {
                    double source = c == 0 ? 20 : c == 1 ? 80 : 140;
                    double expected = std::round(source * alpha / 255. + 255 * (1 - alpha / 255.));
                    coverage &= std::abs(pixel[c] - expected) <= 1;
                }
            }
            require(coverage, "CUDA coverage matches authored-image alpha compositing from "
                              "transparent through opaque");
        }
        {
            auto lifecycle = fixture();
            // Keep a simultaneous contour morph to exercise endpoint-patch protection.
            auto disappearing = lifecycle.slides[0].objects[0];
            disappearing.id = fm::newId();
            disappearing.name = "plain exit";
            disappearing.mask = cv::Mat(180, 320, CV_8U, cv::Scalar(0));
            cv::rectangle(disappearing.mask, {10, 10}, {40, 40}, 255, -1);
            auto appearing = disappearing;
            appearing.id = fm::newId();
            appearing.name = "plain enter";
            appearing.mask = cv::Mat(180, 320, CV_8U, cv::Scalar(0));
            cv::rectangle(appearing.mask, {280, 10}, {310, 40}, 255, -1);
            disappearing.destructAnimation = "none";
            appearing.constructAnimation = "none";
            lifecycle.slides[0].objects.push_back(disappearing);
            lifecycle.slides[1].objects.push_back(appearing);
            lifecycle.slides[0].image.setTo(cv::Scalar(80, 220, 100), disappearing.mask);
            lifecycle.slides[1].image.setTo(cv::Scalar(80, 220, 100), appearing.mask);
            lifecycle.transitions.clear();
            fm::reconcileTransitions(lifecycle);
            fm::validateProject(lifecycle);
            require(lifecycle.transitions[0].links.size() == 3,
                    "Unmatched plain names receive lifecycle connections alongside automatic __ "
                    "morphs");
            fm::Renderer cut(lifecycle, 320, 180);
            auto almostEnd = cut.transitionFrame(0, .98);
            require(almostEnd.at<cv::Vec3b>(25, 25)[1] > 215 &&
                        almostEnd.at<cv::Vec3b>(25, 295)[1] < 25,
                    "None cuts at the next keyframe without endpoint-blend ghosts during another "
                    "morph");
            require(cv::norm(cut.transitionFrame(0, 1), lifecycle.slides[1].image, cv::NORM_INF) ==
                        0,
                    "Lifecycle cuts preserve the exact destination keyframe");
            lifecycle.slides[0].objects.back().destructAnimation = "fade";
            lifecycle.slides[1].objects.back().constructAnimation = "fade";
            fm::Renderer fade(lifecycle, 320, 180);
            auto fading = fade.transitionFrame(0, .5);
            // Authored-image coverage: half of source green220 over background20 is120.
            require(std::abs(int(fading.at<cv::Vec3b>(25, 25)[1]) - 120) <= 1 &&
                        std::abs(int(fading.at<cv::Vec3b>(25, 295)[1]) - 120) <= 1,
                    "Construct and destruct fade independently with authored-image coverage at "
                    "half progress");
            lifecycle.slides[0].objects.back().destructAnimation = "scale";
            lifecycle.slides[1].objects.back().constructAnimation = "scale";
            fm::Renderer scaled(lifecycle, 320, 180);
            auto growing = scaled.transitionFrame(0, .5);
            require(
                growing.at<cv::Vec3b>(25, 25)[1] > 90 && growing.at<cv::Vec3b>(12, 12)[1] < 25 &&
                    growing.at<cv::Vec3b>(25, 295)[1] > 90 &&
                    growing.at<cv::Vec3b>(12, 282)[1] < 25,
                "Grow and shrink change object size around its center without stationary copies");
            for (auto effect : {"slide_up", "slide_down"}) {
                lifecycle.slides[0].objects.back().destructAnimation = effect;
                lifecycle.slides[1].objects.back().constructAnimation = effect;
                fm::Renderer sliding(lifecycle, 320, 180);
                auto moving = sliding.transitionFrame(0, .5);
                int exitY = std::string(effect) == "slide_up" ? 7 : 43;
                int enterY = std::string(effect) == "slide_up" ? 43 : 7;
                require(moving.at<cv::Vec3b>(exitY, 25)[1] > 90 &&
                            moving.at<cv::Vec3b>(enterY, 295)[1] > 90,
                        std::string("Construct/destruct ") + effect +
                            " follows the directional path");
            }
            auto paired = fixture();
            paired.slides[0].objects[0].destructAnimation = "none";
            paired.slides[1].objects[0].constructAnimation = "scale";
            fm::Renderer pairedRenderer(paired, 320, 180);
            require(cv::norm(pairedRenderer.transitionFrame(0, .5), mid, cv::NORM_INF) == 0,
                    "Matched __ objects morph and ignore construct/destruct settings");
            auto plainPaired = paired;
            for (auto &slide : plainPaired.slides)
                slide.objects[0].name = "same plain name";
            plainPaired.transitions.clear();
            fm::reconcileTransitions(plainPaired);
            require(plainPaired.transitions[0].links.empty(),
                    "Plain names present in both frames keep the default transition despite "
                    "lifecycle choices");
            auto extra = plainPaired.slides[0].objects[0];
            extra.id = fm::newId();
            plainPaired.slides[0].objects.push_back(extra);
            auto extraLinks = fm::automaticLinks(plainPaired.slides[0], plainPaired.slides[1]);
            require(extraLinks.size() == 1 && extraLinks[0].sources.size() == 1 &&
                        extraLinks[0].targets.empty(),
                    "Repeated plain names pair existing instances and animate only the "
                    "disappearing extra");
            lifecycle.slides[0].stepDuration = 2.75;
            lifecycle.slides[1].stepDuration = 4.25;
            lifecycle.transitions.clear();
            fm::reconcileTransitions(lifecycle);
            require(lifecycle.transitions[0].duration == 2.75,
                    "New ordinary transitions inherit the source frame time step");
            fm::saveProject(lifecycle, root / "properties");
            auto props = fm::loadProject(root / "properties");
            require(props.slides[0].stepDuration == 2.75 && props.slides[1].stepDuration == 4.25 &&
                        props.slides[0].objects.back().destructAnimation == "slide_down" &&
                        props.slides[1].objects.back().constructAnimation == "slide_down",
                    "Frame time steps and both object lifecycle choices survive save and reopen");
            auto duplicate = fm::duplicateSlide(props.slides[0]);
            require(duplicate.stepDuration == 2.75 &&
                        duplicate.objects.back().destructAnimation == "slide_down",
                    "Duplicating a frame preserves its time step and object animation properties");
            auto invalid = props;
            invalid.slides[0].stepDuration = 0;
            rejects([&] { fm::validateProject(invalid); }, "Zero frame time step rejected");
            invalid = props;
            invalid.slides[0].objects[0].constructAnimation = "unknown";
            rejects([&] { fm::validateProject(invalid); },
                    "Unknown lifecycle effects rejected clearly");
            auto legacyFile = root / "properties" / "project.json";
            fm::json legacy;
            {
                std::ifstream f(legacyFile);
                f >> legacy;
            }
            legacy["version"] = 1;
            for (auto &slide : legacy["slides"]) {
                slide.erase("step_duration");
                for (auto &object : slide["objects"]) {
                    object.erase("construct_animation");
                    object.erase("destruct_animation");
                }
            }
            {
                std::ofstream f(legacyFile);
                f << legacy.dump(2);
            }
            auto migrated = fm::loadProject(root / "properties");
            require(migrated.slides[0].stepDuration == 2.75 &&
                        migrated.slides[0].objects.back().destructAnimation == "default",
                    "Legacy projects derive frame time step from outgoing duration and default "
                    "their effects");
            cv::imwrite((root / "lifecycle-midpoint.png").string(), growing);
        }
        {
            fm::Project mixed;
            mixed.width = 320;
            mixed.height = 180;
            auto s = fm::blankSlide(320, 180, {255, 255, 255});
            s.stepDuration = 2.;
            fm::Object shape;
            shape.id = fm::newId();
            shape.name = "__morph";
            fm::renderDrawing(s, shape,
                              {{"kind", "circle"},
                               {"x", 40},
                               {"y", 100},
                               {"w", 50},
                               {"h", 50},
                               {"color", "#386fe8"}});
            s.objects.push_back(shape);
            fm::Object orb;
            orb.id = fm::newId();
            orb.name = "__orbit";
            fm::renderDrawing(s, orb,
                              {{"kind", "circle"},
                               {"x", 200},
                               {"y", 20},
                               {"w", 10},
                               {"h", 10},
                               {"color", "#22aa91"}});
            s.objects.push_back(orb);
            fm::recomposeSlide(s);
            auto updater = fm::parseObjectUpdater("z:z+0.3*(exp(i*tau*t)-1)", s, s.objects[1]);
            auto next = fm::generateUpdaterKeyframe(s, {{orb.id, updater}}, 2.);
            auto spec = next.objects[0].drawing;
            spec["kind"] = "rectangle";
            spec["x"] = 210;
            fm::renderDrawing(next, next.objects[0], spec);
            fm::recomposeSlide(next);
            mixed.slides = {s, next};
            fm::reconcileTransitions(mixed);
            auto isolated = fm::deepCopy(mixed);
            for (auto &slide : isolated.slides) {
                slide.objects.erase(slide.objects.begin());
                fm::recomposeSlide(slide);
            }
            isolated.transitions.clear();
            fm::reconcileTransitions(isolated);
            fm::Renderer together(mixed, 320, 180), alone(isolated, 320, 180);
            bool equal = true;
            for (float t : {.01f, .03f, .97f, .99f}) {
                auto combined = together.transitionFrame(0, t),
                     original = alone.transitionFrame(0, t);
                equal &= cv::norm(combined(cv::Rect(0, 0, 320, 65)),
                                  original(cv::Rect(0, 0, 320, 65)), cv::NORM_INF) == 0;
            }
            require(equal, "A simultaneous contour morph never introduces endpoint ghosts into a "
                           "circular updater path");
        }
        {
            cv::Mat image(48, 64, CV_8UC4);
            for (int y = 0; y < image.rows; ++y)
                for (int x = 0; x < image.cols; ++x)
                    image.at<cv::Vec4b>(y, x) = {uchar(x * 3), uchar(y * 5), uchar(220 - x),
                                                 uchar(80 + x * 2)};
            auto untouched = image.clone();
            rejects([&] { fm::processImage(image, fm::json::object(), [] { return true; }); },
                    "Cancelled image previews stop before processing");
            rejects(
                [&] {
                    fm::renderMedia({{"kind", "code"}, {"content", "test"}}, [] { return true; });
                },
                "Cancelled artwork previews stop before launching a helper");
            auto swap = fm::processImage(image, {{"pixel_map", "(r,g,b,a):g,b,r,a"}});
            auto p = image.at<cv::Vec4b>(10, 20), q = swap.at<cv::Vec4b>(10, 20);
            require(q[2] == p[1] && q[1] == p[0] && q[0] == p[2] && q[3] == p[3],
                    "Pixel maps read immutable source channels and preserve alpha");
            require(cv::norm(image, untouched, cv::NORM_INF) == 0,
                    "Image processing never mutates the saved original");
            auto gradient =
                fm::processImage(image, {{"pixel_map", "(r,g,b,a):clamp(2*r,0,1),g,b,a*x"}});
            require(gradient.at<cv::Vec4b>(0, 0)[3] == 0 &&
                        gradient.at<cv::Vec4b>(0, 63)[3] == image.at<cv::Vec4b>(0, 63)[3],
                    "Pixel formulas support normalized coordinates and clamp");
            auto crop = fm::processImage(image, {{"crop_x", 10},
                                                 {"crop_y", 8},
                                                 {"crop_w", 20},
                                                 {"crop_h", 16},
                                                 {"width", 20},
                                                 {"height", 16},
                                                 {"flip_x", true}});
            require(crop.size() == cv::Size(20, 16) &&
                        crop.at<cv::Vec4b>(0, 0) == image.at<cv::Vec4b>(8, 29),
                    "Crop and horizontal flip use original image coordinates");
            auto tinted = fm::processImage(image, {{"tint", "#80ff80"}});
            require(tinted.at<cv::Vec4b>(10, 20)[1] == image.at<cv::Vec4b>(10, 20)[1] &&
                        tinted.at<cv::Vec4b>(10, 20)[2] < image.at<cv::Vec4b>(10, 20)[2] &&
                        tinted.at<cv::Vec4b>(10, 20)[3] == image.at<cv::Vec4b>(10, 20)[3],
                    "Image tint preserves detail and alpha instead of flattening the picture");
            auto gray = fm::processImage(image, {{"filter", "grayscale"}});
            q = gray.at<cv::Vec4b>(10, 20);
            require(q[0] == q[1] && q[1] == q[2] && q[3] == p[3],
                    "Grayscale preserves original transparency");
            for (auto filter : {"sepia", "invert", "sharpen", "edges", "threshold"}) {
                auto result = fm::processImage(image, {{"filter", filter}});
                cv::Mat a, b;
                cv::extractChannel(result, a, 3);
                cv::extractChannel(image, b, 3);
                require(result.type() == CV_8UC4 && cv::norm(a, b, cv::NORM_INF) == 0,
                        std::string("Image filter preserves alpha: ") + filter);
            }
            cv::Mat transparent(2, 2, CV_8UC4, cv::Scalar(0, 0, 0, 0));
            transparent.at<cv::Vec4b>(0, 0) = {0, 0, 255, 255};
            auto enlarged =
                fm::processImage(transparent, {{"width", 16}, {"height", 16}, {"blur", .8}});
            bool noHalo = true;
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x) {
                    auto v = enlarged.at<cv::Vec4b>(y, x);
                    if (v[3] > 5)
                        noHalo &= v[2] > 250 && v[0] < 2;
                }
            require(noHalo, "Alpha-aware resizing and blur avoid dark transparent-edge halos");
            rejects([&] { fm::processImage(image, {{"pixel_map", "(r,g,b,a):1/(r-r),g,b,a"}}); },
                    "Non-finite pixel formulas leave the original intact");
            rejects([&] { fm::processImage(image, {{"pixel_map", "(r,g,b):r,g,b"}}); },
                    "Pixel maps require four output channels");
            rejects([&] { fm::processImage(image, {{"width", 8192}, {"height", 8192}}); },
                    "Oversized image processing rejected before allocation");
            rejects([&] { fm::processImage(image, {{"crop_x", 50}, {"crop_w", 20}}); },
                    "Out-of-range image crops rejected");
            auto scene = fm::blankSlide(320, 180, {255, 255, 255});
            for (int i = 0; i < 2; ++i) {
                fm::Object object;
                object.id = fm::newId();
                object.name = "__image_" + std::to_string(i);
                object.sourceAsset =
                    cv::Mat(30, 40, CV_8UC4, cv::Scalar(i ? 200 : 10, 50, i ? 10 : 230, 180));
                fm::renderDrawing(
                    scene, object,
                    {{"kind", "image"}, {"x", 20 + i * 15}, {"y", 20}, {"w", 40}, {"h", 30}});
                scene.objects.push_back(object);
            }
            fm::recomposeSlide(scene);
            auto tintObject = scene.objects[0];
            auto tintSlide = scene;
            auto neutral = tintObject.sourceAsset.clone();
            fm::setObjectColor(tintSlide, tintObject, "#80ff80");
            fm::setObjectColor(tintSlide, tintObject, "#ffffff");
            require(cv::norm(tintObject.sourceAsset, neutral, cv::NORM_INF) == 0 &&
                        !tintObject.originalAsset.empty(),
                    "Changing an image tint back to white recovers its original pixels");
            std::vector<std::string> ids{scene.objects[0].id, scene.objects[1].id};
            auto overlay = fm::combineImageObjects(scene, ids, "overlay"),
                 row = fm::combineImageObjects(scene, ids, "row", 8),
                 column = fm::combineImageObjects(scene, ids, "column", 8);
            require(overlay.sourceAsset.cols == 55 && overlay.sourceAsset.rows == 30 &&
                        overlay.sourceAsset.at<cv::Vec4b>(10, 20)[3] > 220,
                    "Overlay combines images with alpha in layer order");
            require(row.sourceAsset.size() == cv::Size(88, 30) &&
                        column.sourceAsset.size() == cv::Size(40, 68),
                    "Image combination supports horizontal and vertical layouts with spacing");
            require(scene.objects.size() == 2,
                    "Preparing an image combination does not mutate the source slide");
            rejects([&] { fm::combineImageObjects(scene, {ids[0], ids[0]}, "row"); },
                    "Repeated objects rejected in image combinations");
            auto edited = scene.objects[0];
            edited.originalAsset = edited.sourceAsset.clone();
            edited.imageEdits = {{"filter", "invert"}};
            edited.sourceAsset = fm::processImage(edited.originalAsset, edited.imageEdits);
            fm::renderDrawing(scene, edited, edited.drawing);
            scene.objects[0] = edited;
            fm::recomposeSlide(scene);
            fm::Project portable;
            portable.width = 320;
            portable.height = 180;
            portable.slides = {scene};
            fm::saveProject(portable, root / "image-edits");
            auto loaded = fm::loadProject(root / "image-edits");
            require(fm::projectMetadata(loaded)["version"] == 4 &&
                        loaded.slides[0].objects[0].imageEdits == edited.imageEdits &&
                        cv::norm(loaded.slides[0].objects[0].originalAsset, edited.originalAsset,
                                 cv::NORM_INF) == 0,
                    "Non-destructive image originals and recipes survive portable versioned saves");
            auto matrix = fm::renderMedia({{"kind", "equation"},
                                           {"backend", "latex"},
                                           {"content", R"(A=\begin{bmatrix}1&2\\3&4\end{bmatrix})"},
                                           {"font_size", 42},
                                           {"color", "#386fe8"}});
            cv::Mat alpha;
            cv::extractChannel(matrix, alpha, 3);
            require(matrix.type() == CV_8UC4 && cv::countNonZero(alpha) > 100 &&
                        cv::countNonZero(alpha) < alpha.total(),
                    "Full LaTeX renders a transparent coloured matrix");
            auto cases =
                fm::renderMedia({{"kind", "equation"},
                                 {"backend", "auto"},
                                 {"content", R"(f(x)=\begin{cases}x^2&x\geq0\\-x&x<0\end{cases})"},
                                 {"font_size", 36}});
            require(!cases.empty(), "Auto equation rendering falls back to Full LaTeX for cases");
            rejects(
                [&] {
                    fm::renderMedia(
                        {{"kind", "equation"}, {"backend", "latex"}, {"content", R"(\frac{)"}});
                },
                "Invalid LaTeX returns its compilation error");
            auto code = fm::renderMedia(
                {{"kind", "code"},
                 {"language", "python"},
                 {"content", "raise RuntimeError('source code must never execute')"},
                 {"font_size", 20}});
            require(!code.empty() && code.channels() == 4,
                    "Source code blocks are highlighted artwork and never execute");
            std::string svg =
                R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect x="5" y="5" width="190" height="90" fill="#386fe8"/></svg>)";
            auto vector = fm::renderMedia({{"kind", "svg"}, {"content", svg}, {"svg_size", 400}});
            require(vector.cols == 400 && vector.rows == 200,
                    "SVG source renders to bounded portable artwork with its aspect ratio");
            rejects(
                [&] {
                    fm::renderMedia(
                        {{"kind", "svg"},
                         {"content",
                          R"(<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.com/a.png"/></svg>)"}});
                },
                "SVG external resources are rejected clearly");
            cv::imwrite((root / "full-latex-matrix.png").string(), matrix);
            cv::imwrite((root / "source-code.png").string(), code);
            cv::imwrite((root / "svg-artwork.png").string(), vector);
        }
        {
            auto effects = fixture();
            effects.slides[1].objects[0].name = "__other";
            effects.transitions[0].links = fm::automaticLinks(effects.slides[0], effects.slides[1]);
            bool good = true;
            for (auto &effect : fm::lifecycleEffects()) {
                effects.slides[0].objects[0].destructAnimation = effect.first;
                effects.slides[1].objects[0].constructAnimation = effect.first;
                fm::validateProject(effects);
                fm::Renderer renderer(effects, 320, 180);
                for (float t : {.01f, .25f, .5f, .75f, .99f})
                    good &= !renderer.transitionFrame(0, t).empty();
            }
            require(good,
                    "All fourteen construct/destruct effects render at five transition times");
            effects.slides[0].objects[0].destructAnimation = "wipe_left";
            effects.slides[1].objects[0].constructAnimation = "wipe_left";
            fm::Renderer wiping(effects, 320, 180);
            auto middle = wiping.transitionFrame(0, .5);
            require(middle.at<cv::Vec3b>(90, 225)[1] > 100 && middle.at<cv::Vec3b>(90, 255)[1] < 30,
                    "Wipe reveals artwork without shrinking its geometry");
            effects.slides[1].objects[0].constructAnimation = "fade";
            effects.slides[1].objects[0].effectStart = .25;
            effects.slides[1].objects[0].effectEnd = .75;
            effects.slides[1].objects[0].effectEasing = "linear";
            fm::Renderer timedEffects(effects, 320, 180);
            require(timedEffects.transitionFrame(0, .1).at<cv::Vec3b>(90, 240)[1] < 30 &&
                        timedEffects.transitionFrame(0, .9).at<cv::Vec3b>(90, 240)[1] > 100,
                    "Per-object effect timing holds before and after its scheduled interval");
            require(fm::projectMetadata(effects)["version"] == 4,
                    "Expanded effects and timing are protected by project format version 4");
            auto invalid = effects;
            invalid.slides[0].objects[0].effectStart = .9;
            invalid.slides[0].objects[0].effectEnd = .1;
            rejects([&] { fm::validateProject(invalid); }, "Reversed effect timing rejected");
        }
        auto benchmark = fixture();
        auto started = std::chrono::steady_clock::now();
        fm::Renderer speed(benchmark, 960, 540);
        speed.transitionFrame(0, .5);
        started = std::chrono::steady_clock::now();
        for (int i = 0; i < 30; ++i)
            speed.transitionFrame(0, .1f + .8f * i / 30);
        double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count() /
            30;
        std::cout << "BENCHMARK simple 960x540 morph: " << ms << " ms/frame, " << 1000 / ms
                  << " fps\n";
        std::cout << "Artifacts: " << root << "\n" << passed << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL after " << passed << " checks: " << e.what() << '\n';
        return 1;
    }
}
