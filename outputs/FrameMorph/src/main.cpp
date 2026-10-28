#include "engine.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char **argv) {
    try {
        if (argc == 1)
            return fm::runEditor();
        std::string command = argv[1];
        if (command == "--help" || command == "help") {
            std::cout
                << "FrameMorph - image keyframes, intentional motion\n\n"
                   "  framemorph [project-folder]                  Open editor\n"
                   "  framemorph edit [project-folder]             Open editor\n"
                   "  framemorph sample-toolkit <folder>           Create LaTeX, SVG, code and "
                   "image demo\n"
                   "  framemorph sample-properties <folder>        Create Properties workflow "
                   "example\n"
                   "  framemorph sample <folder>                   Create demonstration project\n"
                   "  framemorph sample-studio <folder>            Create an editable studio "
                   "example\n"
                   "  framemorph sample-motion <folder>            Create smooth motion / in-place "
                   "map example\n"
                   "  framemorph sample-authoring <folder>         Create a native drawing / math "
                   "demo\n"
                   "  framemorph sample-transforms <folder>        Create matrix / complex map "
                   "demo\n"
                   "  framemorph sample-updaters <folder>          Create orbit / spin / "
                   "deformation example\n"
                   "  framemorph profile <project> --output profile.json [--samples 120 --width "
                   "960 --height 540 --uncached]\n"
                   "  framemorph info                              Describe CUDA device\n"
                   "  framemorph validate <project-folder>         Check project and assets\n"
                   "  framemorph generate <project> --operations ops.json --output <folder> "
                   "[--slide 1]\n"
                   "  framemorph frame <project> --output f.png [--time 1.5] [--width 960 --height "
                   "540]\n"
                   "  framemorph render <project> --output f.mp4 [--width 1920 --height 1080 --fps "
                   "60]\n"
                   "                    [--png-sequence] [--ffmpeg /path/to/ffmpeg]\n";
            return 0;
        }
        if (command == "info") {
            std::cout << fm::CudaBackend::deviceDescription() << '\n';
            return 0;
        }
        if (command == "edit")
            return fm::runEditor(argc > 2 ? fm::fs::path(argv[2]) : fm::fs::path{});
        if (command == "profile") {
            if (argc < 3)
                throw std::runtime_error("Provide a project to profile");
            int width = 960, height = 540, samples = 120, transition = 1;
            bool inputCaching = true;
            fm::fs::path output;
            for (int i = 3; i < argc; ++i) {
                std::string flag = argv[i];
                if (flag == "--uncached") {
                    inputCaching = false;
                    continue;
                }
                if (i + 1 >= argc)
                    throw std::runtime_error("Missing profiling option value");
                std::string value = argv[++i];
                if (flag == "--width")
                    width = std::stoi(value);
                else if (flag == "--height")
                    height = std::stoi(value);
                else if (flag == "--samples")
                    samples = std::stoi(value);
                else if (flag == "--transition")
                    transition = std::stoi(value);
                else if (flag == "--output")
                    output = value;
                else
                    throw std::runtime_error("Unknown profiling option: " + flag);
            }
            auto report = fm::profileProject(fm::loadProject(argv[2]), width, height, samples,
                                             transition - 1, inputCaching);
            if (!output.empty()) {
                std::ofstream file(output);
                if (!file)
                    throw std::runtime_error("Cannot write profiling report");
                file << report.dump(2) << '\n';
            }
            std::cout << report.dump(2) << '\n';
            return 0;
        }
        if (command == "generate") {
            if (argc < 3)
                throw std::runtime_error("Provide a source project folder");
            fm::fs::path operationsFile, output;
            int slide = 1;
            for (int i = 3; i < argc; ++i) {
                std::string flag = argv[i];
                if (i + 1 >= argc)
                    throw std::runtime_error("Missing value for " + flag);
                std::string value = argv[++i];
                if (flag == "--operations")
                    operationsFile = value;
                else if (flag == "--output")
                    output = value;
                else if (flag == "--slide")
                    slide = std::stoi(value);
                else
                    throw std::runtime_error("Unknown option: " + flag);
            }
            if (operationsFile.empty() || output.empty())
                throw std::runtime_error(
                    "Use --operations ops.json and --output <new project folder>");
            if (fm::fs::exists(output / "project.json"))
                throw std::runtime_error("Output project already exists; choose a new folder");
            auto project = fm::loadProject(argv[2]);
            if (slide < 1 || slide > (int)project.slides.size())
                throw std::runtime_error("Source slide number is outside the project");
            std::ifstream stream(operationsFile);
            if (!stream)
                throw std::runtime_error("Cannot open operations JSON file");
            auto specs = fm::json::parse(stream);
            if (!specs.is_object())
                throw std::runtime_error(
                    "Operations JSON must map object names or IDs to transformations");
            std::map<std::string, fm::json> operations;
            auto &source = project.slides[slide - 1];
            for (auto &entry : specs.items()) {
                bool matched = false;
                for (auto &o : source.objects)
                    if (o.id == entry.key() || o.name == entry.key()) {
                        operations[o.id] = entry.value();
                        matched = true;
                    }
                if (!matched)
                    throw std::runtime_error("No source object matches " + entry.key());
            }
            auto next = fm::generateKeyframe(source, operations);
            project.slides.insert(project.slides.begin() + slide, std::move(next));
            fm::reconcileTransitions(project);
            fm::saveProject(project, output);
            std::cout << "Generated keyframe " << slide + 1 << " in " << output << '\n';
            return 0;
        }
        if (command == "sample" || command == "sample-toolkit" || command == "sample-properties" ||
            command == "sample-updaters" || command == "sample-motion" ||
            command == "sample-studio" || command == "sample-authoring" ||
            command == "sample-transforms") {
            if (argc < 3)
                throw std::runtime_error("Provide a sample project folder");
            if (fm::fs::exists(fm::fs::path(argv[2]) / "project.json"))
                throw std::runtime_error("Sample project already exists; choose a new folder");
            if (command == "sample")
                fm::createSample(argv[2]);
            else if (command == "sample-authoring")
                fm::createAuthoringSample(argv[2]);
            else if (command == "sample-studio")
                fm::createStudioSample(argv[2]);
            else if (command == "sample-motion")
                fm::createMotionSample(argv[2]);
            else if (command == "sample-toolkit")
                fm::createToolkitSample(argv[2]);
            else if (command == "sample-properties")
                fm::createPropertiesSample(argv[2]);
            else if (command == "sample-updaters")
                fm::createUpdaterSample(argv[2]);
            else
                fm::createTransformSample(argv[2]);
            std::cout << "Created " << argv[2] << '\n';
            return 0;
        }
        if (command == "render" || command == "frame" || command == "validate") {
            if (argc < 3)
                throw std::runtime_error("Provide a project folder");
            auto p = fm::loadProject(argv[2]);
            if (command == "validate") {
                std::cout << p.slides.size() << " slides, " << p.transitions.size()
                          << " transitions, " << fm::projectDuration(p) << " seconds: valid\n";
                return 0;
            }
            fm::ExportOptions options;
            double time = 0;
            for (int i = 3; i < argc; ++i) {
                std::string flag = argv[i];
                if (flag == "--png-sequence") {
                    options.pngSequence = true;
                    continue;
                }
                if (i + 1 >= argc)
                    throw std::runtime_error("Missing value for " + flag);
                std::string value = argv[++i];
                if (flag == "--output")
                    options.output = value;
                else if (flag == "--width")
                    options.width = std::stoi(value);
                else if (flag == "--height")
                    options.height = std::stoi(value);
                else if (flag == "--fps")
                    options.fps = std::stoi(value);
                else if (flag == "--ffmpeg")
                    options.ffmpeg = value;
                else if (flag == "--time")
                    time = std::stod(value);
                else
                    throw std::runtime_error("Unknown option: " + flag);
            }
            if (options.output.empty())
                throw std::runtime_error("Use --output to choose a destination");
            if (command == "frame") {
                fm::Renderer r(p, options.width, options.height);
                auto image = r.frame(time);
                if (!cv::imwrite(options.output.string(), image))
                    throw std::runtime_error("Cannot write frame");
                for (auto &w : r.warnings())
                    std::cerr << w << '\n';
            } else
                fm::exportVideo(p, options, nullptr, [](int frame, int count) {
                    if (frame % 60 == 0 || frame == count)
                        std::cerr << "\rRendering " << frame << " / " << count << std::flush;
                });
            std::cout << "\nSaved " << options.output << '\n';
            return 0;
        }
        return fm::runEditor(argv[1]);
    } catch (const std::exception &e) {
        std::cerr << "FrameMorph: " << e.what() << '\n';
        return 1;
    }
}
