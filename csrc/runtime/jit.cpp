// Copyright (c) 2026, Lu Lu

#include "runtime/jit.hpp"

#include <dlfcn.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace deep_ep {
namespace {

constexpr uint32_t kDefaultArch = 3510U;

// EP_JIT_* is the deployed name; DEEPEP_JIT_* is accepted during migration so
// existing scripts keep working and are warned once.
std::string env_text(const char* name)
{
    const std::string ep_name = std::string("EP_JIT_") + name;
    const std::string legacy_name = std::string("DEEPEP_JIT_") + name;
    if (const char* value = std::getenv(ep_name.c_str()); value != nullptr) return std::string(value);
    if (const char* value = std::getenv(legacy_name.c_str()); value != nullptr) {
        fprintf(stderr, "warning: %s is deprecated, use %s\n", legacy_name.c_str(), ep_name.c_str());
        return std::string(value);
    }
    return std::string();
}

std::string required_env(const char* name)
{
    const std::string value = env_text(name);
    if (value.empty()) throw std::runtime_error(std::string("DeepJIT requires EP_JIT_") + name);
    return value;
}

std::filesystem::path module_dir()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&jit_compile), &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("DeepJIT cannot locate the loaded extension");
    return std::filesystem::absolute(std::filesystem::path(info.dli_fname)).lexically_normal().parent_path();
}

std::filesystem::path source_root()
{
    const auto packaged = module_dir() / "_kernel_src" / "csrc";
    if (std::filesystem::is_directory(packaged)) return packaged;
    const auto checkout = module_dir().parent_path() / "csrc";
    if (std::filesystem::is_directory(checkout)) return checkout;
    throw std::runtime_error("DeepJIT cannot find csrc; install the wheel or run from the checkout");
}

// Device headers live in deep_ep/include, packaged next to the extension.
std::filesystem::path include_root()
{
    const auto packaged = module_dir() / "include";
    if (std::filesystem::is_directory(packaged)) return packaged;
    const auto kernel_src = module_dir() / "_kernel_src" / "deep_ep" / "include";
    if (std::filesystem::is_directory(kernel_src)) return kernel_src;
    const auto checkout = module_dir().parent_path() / "deep_ep" / "include";
    if (std::filesystem::is_directory(checkout)) return checkout;
    throw std::runtime_error("DeepJIT cannot find deep_ep/include; install the wheel or run from the checkout");
}

std::filesystem::path shmem_root()
{
    const auto packaged = module_dir() / "_kernel_src" / "shmem";
    if (std::filesystem::is_regular_file(packaged / "include" / "shmem.h")) return packaged;
    const auto checkout = module_dir().parent_path() / "third-party/shmem/install/shmem";
    if (std::filesystem::is_regular_file(checkout / "include" / "shmem.h")) return checkout;
    throw std::runtime_error("DeepJIT cannot find the SHMEM device headers");
}

std::string host_arch()
{
    utsname value{};
    if (uname(&value) == 0 && value.machine[0] != '\0') return value.machine;
    return "aarch64";
}

uint32_t target_arch()
{
    const std::string value = env_text("TARGET_ARCH");
    if (value.empty()) return kDefaultArch;
    try {
        const auto arch = std::stoul(value);
        if (arch == 0U || arch > 65535U) throw std::out_of_range("arch");
        return static_cast<uint32_t>(arch);
    } catch (const std::exception&) {
        throw std::runtime_error("EP_JIT_TARGET_ARCH must be a positive integer");
    }
}

std::vector<std::filesystem::path> include_dirs()
{
    const auto home = std::filesystem::canonical(required_env("ASCEND_HOME_PATH"));
    const auto arch = host_arch();
    const auto shmem = shmem_root();
    const auto megamoe = module_dir() / "_kernel_src" / "megamoe";
    std::vector<std::filesystem::path> dirs = {
        source_root(),
        include_root(),
        megamoe / "csrc",
        megamoe / "catlass" / "include",
        megamoe / "catlass" / "examples" / "common",
        megamoe / "tensor_api",
        megamoe / "tensor_api" / "include",
        shmem / "include",
        shmem / "src",
        shmem / "src/device",
        shmem / "src/device_simt",
        home / "include",
        home / "include/experiment/runtime",
        home / "asc/include",
        home / (arch + "-linux") / "include",
        home / (arch + "-linux") / "asc/include",
        home / "aarch64-linux/include",
        home / "compiler/tikcpp",
        home / "compiler/tikcpp/tikcfw",
        home / "compiler/tikcpp/tikcfw/impl",
        home / "compiler/tikcpp/tikcfw/interface",
    };
    return dirs;
}

std::unique_ptr<deep_jit::Runtime<deep_jit::Ascend>>& runtime_storage()
{
    static std::unique_ptr<deep_jit::Runtime<deep_jit::Ascend>> runtime;
    return runtime;
}

}  // namespace

deep_jit::Runtime<deep_jit::Ascend>& jit_runtime()
{
    static std::once_flag once;
    std::call_once(once, [] {
        const deep_jit::Config config(source_root(), "EP", "", include_dirs(), {"deep_ep/"});
        runtime_storage() = std::make_unique<deep_jit::Runtime<deep_jit::Ascend>>(config);
        runtime_storage()->default_compiler_options.arch = static_cast<int>(target_arch());
    });
    return *runtime_storage();
}

std::shared_ptr<JitKernel> jit_compile(const std::string& name, const std::string& source,
                                       const std::vector<std::string>& extra_flags)
{
    deep_jit::ascend::CompilerOptions options;
    options.arch = static_cast<int>(target_arch());
    options.debug_info = deep_jit::get_env<bool>("EP_JIT_KERNEL_DEBUG_INFO", false);
    options.dump_asm = deep_jit::get_env<bool>("EP_JIT_DUMP_ASM", false);
    options.extra_bisheng_flags = extra_flags;
    return jit_runtime().compile(name, source, options);
}

}  // namespace deep_ep
