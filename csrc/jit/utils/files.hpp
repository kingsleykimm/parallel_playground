#pragma once
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <moe_cuda/error.hpp>
#include <random>
#include <runtime/format.hpp>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
// recursive iteration of files
static std::vector<fs::path> all_files_in_dir(const fs::path &directory_path) {
  // recurse
  std::vector<fs::path> files;

  std::function<void(const fs::path &)> collect;

  collect = [&](const fs::path &cur_path) {
    if (fs::is_directory(cur_path)) {
      for (auto sub_path : fs::directory_iterator(cur_path)) {
        collect(sub_path);
      }
    } else {
      files.push_back(cur_path);
    }
  };

  collect(directory_path);

  // consistent ordering, since it's mostly used for hashing
  std::sort(files.begin(), files.end());
  return files;
}

inline fs::path make_dir(const fs::path &file_path) {
  std::error_code ec;
  bool created = fs::create_directories(file_path, ec);

  if (!created && ec) {
    HOST_ASSERT(false,
                fmt::format("Failed to make directory {}, with error code {}",
                            (file_path).string(), ec.value())
                    .c_str());
  }
  return file_path;
}

static std::string get_uuid() {
  std::random_device rd;
  int pid = getpid();
  std::mt19937_64 gen(
      rd() ^ std::chrono::steady_clock::now().time_since_epoch().count());
  // to generate an uuid, use one random number, the process id and the current
  // time
  std::uniform_int_distribution<uint32_t> dist;
  std::stringstream ss;
  ss << pid << std::hex << std::setfill('0') << std::setw(8) << dist(gen);
  return ss.str();
}

// we run command and get capture output + exit codes
static std::pair<int, std::string> run_command(std::string command) {
  // combine stderr and stdout first, stderr > &stdout
  command += " 2>&1";
  // pipe cleanup function
  const auto &cleanup = [&](FILE *f) {
    if (f)
      pclose(f);
  };
  std::unique_ptr<FILE, decltype(cleanup)> command_pipe(
      popen(command.c_str(), "r"), cleanup);
  HOST_ASSERT(command_pipe != nullptr, "");

  std::array<char, 512> buffer;
  std::string output;

  while (fgets(buffer.data(), 512, command_pipe.get())) {
    output += buffer.data();
  }
  const auto exitcode = WEXITSTATUS(pclose(command_pipe.release()));
  return {exitcode, output};
}
