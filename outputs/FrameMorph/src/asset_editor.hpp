#pragma once
#include "engine.hpp"
#include <optional>
#include <ostream>
namespace fm {
struct AssetEditResult {
    json spec;
    cv::Mat pixels;
};
std::optional<AssetEditResult> editAsset(const std::string &kind, const json &initial,
                                         const cv::Mat &source = {});
void showAppHelp();
void verifyAssetEditors(const fs::path &, std::ostream &);
} // namespace fm
