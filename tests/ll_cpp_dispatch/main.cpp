// MPI 协调 + ACL + SHMEM 的低时延 dispatch 直调示例（不经过 Python/torch）。
// 结构对齐 Ziha0-Zeng/DeepEP:tests/ll_cpp_bench 的 runner：MPI_Init → 拓扑校验 →
// aclInit/aclrtSetDevice/aclrtCreateStream → deepep::Session(mpi=true) → 调用算子 → 校验。
#include <mpi.h>

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/dispatch/low_latency/common/comm_args.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
#include "ll_dispatch_c_api.h"
#include "runtime/session.hpp"

namespace {

void check(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}

void mpi_check(int status, const char* operation)
{
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
}

// 每台物理主机必须承载 kLlPhysicalRanksPerHost(=16) 个连续 rank。
struct Coordinator {
    int rank = 0;
    int world = 0;
    int local_rank = 0;
    int local_size = 0;

    Coordinator()
    {
        mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI error handler");
        mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI rank");
        mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &world), "MPI size");
        if (world < 16 || world > 256 || (world & (world - 1)))
            throw std::runtime_error("use a power-of-two world size in [16, 256]");
        MPI_Comm local = MPI_COMM_NULL;
        mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local),
                  "MPI local group");
        mpi_check(MPI_Comm_rank(local, &local_rank), "MPI local rank");
        mpi_check(MPI_Comm_size(local, &local_size), "MPI local size");
        mpi_check(MPI_Comm_free(&local), "MPI free local group");
        const int host_first = rank / 16 * 16;
        const int expected = std::min(16, world - host_first);
        if (local_size != expected || rank - local_rank != host_first)
            throw std::runtime_error("each host must own 16 consecutive ranks");
    }

    void barrier() const
    {
        mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI barrier");
    }
};

struct Options {
    int tokens = 32;      // 每卡输入 token 数 b
    int hidden = 3584;    // h
    int topk = 8;         // k
    int experts = 384;    // 全局专家数 e
    int capacity = 0;     // 每卡容量上限；0 表示等于 tokens
    int warmup = 5;
    int iters = 20;
    int timeout = 300;
    int device_base = 0;
    bool fp16 = false;
    std::string backend = "rdma";
};

int positive(const char* text, const char* name)
{
    const long value = std::strtol(text, nullptr, 10);
    if (value <= 0 || value > (1 << 20)) throw std::invalid_argument(std::string("bad ") + name + ": " + text);
    return int(value);
}

Options parse(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
            return argv[++i];
        };
        if (key == "--tokens") opt.tokens = positive(next(), "tokens");
        else if (key == "--hidden") opt.hidden = positive(next(), "hidden");
        else if (key == "--topk") opt.topk = positive(next(), "topk");
        else if (key == "--experts") opt.experts = positive(next(), "experts");
        else if (key == "--capacity") opt.capacity = positive(next(), "capacity");
        else if (key == "--warmup") opt.warmup = positive(next(), "warmup");
        else if (key == "--iters") opt.iters = positive(next(), "iters");
        else if (key == "--timeout") opt.timeout = positive(next(), "timeout");
        else if (key == "--device-base") opt.device_base = int(std::strtol(next(), nullptr, 10));
        else if (key == "--dtype") {
            const std::string value = next();
            if (value == "fp16") opt.fp16 = true;
            else if (value == "bf16") opt.fp16 = false;
            else throw std::invalid_argument("--dtype must be bf16 or fp16");
        } else if (key == "--backend") {
            opt.backend = next();
            if (opt.backend != "mte" && opt.backend != "rdma")
                throw std::invalid_argument("--backend must be mte or rdma");
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (opt.capacity == 0) opt.capacity = opt.tokens;
    if (opt.capacity < opt.tokens || opt.capacity > 256)
        throw std::invalid_argument("capacity must be in [tokens, 256]");
    if (opt.topk > opt.experts) throw std::invalid_argument("topk must be <= experts");
    return opt;
}

struct DeviceBuffer {
    void* pointer = nullptr;
    size_t bytes = 0;

    DeviceBuffer(size_t size, bool zero) : bytes(size)
    {
        check(aclrtMalloc(&pointer, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        if (zero) check(aclrtMemset(pointer, size, 0, size), "aclrtMemset");
    }
    ~DeviceBuffer()
    {
        if (pointer) aclrtFree(pointer);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// 每个 rank 用同一套确定性路由，便于在 CPU 上重建参考结果。
int32_t route_expert(int source, int token, int slot, int experts)
{
    return int32_t((int64_t(source) * 7919 + int64_t(token) * 131 + int64_t(slot) * 17) % experts);
}

void copy_to_device(void* destination, const void* source, size_t bytes)
{
    check(aclrtMemcpy(destination, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)");
}

void run(const Coordinator& peers, const Options& opt)
{
    const int b = opt.tokens;
    const int h = opt.hidden;
    const int k = opt.topk;
    const int e = opt.experts;
    const int world = peers.world;
    const int local_experts = e / world;
    const size_t element = 2;  // bf16 / fp16
    const int device = opt.device_base + peers.local_rank;

    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    check(aclrtCreateStream(&stream), "aclrtCreateStream");
    setenv("DEEPEP_LL_BACKEND", opt.backend.c_str(), 1);

    {
        deepep::Session session(peers.rank, world, DispatchDedup::kEpServerDedupPeerBytes, "", unsigned(opt.timeout),
                                false, false, /*mpi=*/true);
        const uint32_t rows = ll_dispatch_capacity_rows(uint32_t(world), b, h, k, e);

        DeviceBuffer input(size_t(b) * h * element, false);
        DeviceBuffer indices(size_t(b) * k * sizeof(int32_t), false);
        DeviceBuffer weights(size_t(b) * k * sizeof(float), false);
        DeviceBuffer output(size_t(rows) * h * element, false);
        DeviceBuffer counts(size_t(local_experts) * sizeof(int64_t), true);
        DeviceBuffer send_counts(size_t(e) * sizeof(int32_t), true);
        DeviceBuffer token_type(size_t(rows) * sizeof(int32_t), true);
        DeviceBuffer destination_index(size_t(rows) * 3 * sizeof(int32_t), true);
        DeviceBuffer relay_read_index(size_t(rows) * k * 2 * sizeof(int32_t), true);
        DeviceBuffer source_mask(size_t(b) * sizeof(int32_t), true);
        DeviceBuffer source_weights(size_t(b) * k * sizeof(float), true);
        DeviceBuffer comm_args(sizeof(DispatchDedup::CommArgs), true);
        DeviceBuffer tiling(sizeof(DispatchDedup::EpServerDedupTilingData), true);

        ll_dispatch_make_comm_args(session, comm_args.pointer);
        ll_dispatch_make_tiling(uint32_t(world), b, h, k, e, opt.capacity, opt.fp16, tiling.pointer);

        std::vector<uint16_t> host_input(size_t(b) * h, 0x3F80U);  // 1.0（bf16 与 fp16 同为 0x3F80）
        std::vector<int32_t> host_indices(size_t(b) * k);
        std::vector<float> host_weights(size_t(b) * k);
        for (int token = 0; token < b; ++token) {
            for (int slot = 0; slot < k; ++slot) {
                const size_t at = size_t(token) * k + slot;
                host_indices[at] = route_expert(peers.rank, token, slot, e);
                host_weights[at] = 1.0F / float(k);
            }
        }
        copy_to_device(input.pointer, host_input.data(), host_input.size() * sizeof(uint16_t));
        copy_to_device(indices.pointer, host_indices.data(), host_indices.size() * sizeof(int32_t));
        copy_to_device(weights.pointer, host_weights.data(), host_weights.size() * sizeof(float));

        LlDispatchTensors tensors;
        tensors.input = input.pointer;
        tensors.indices = indices.pointer;
        tensors.weights = weights.pointer;
        tensors.output = output.pointer;
        tensors.expert_recv_counts = counts.pointer;
        tensors.send_counts = send_counts.pointer;
        tensors.token_type = token_type.pointer;
        tensors.destination_index = destination_index.pointer;
        tensors.relay_read_index = relay_read_index.pointer;
        tensors.source_mask = source_mask.pointer;
        tensors.source_weights = source_weights.pointer;
        tensors.comm_args = comm_args.pointer;
        tensors.tiling = tiling.pointer;

        aclrtEvent start = nullptr;
        aclrtEvent stop = nullptr;
        check(aclrtEventCreate(&start), "aclrtEventCreate");
        check(aclrtEventCreate(&stop), "aclrtEventCreate");
        for (int i = 0; i < opt.warmup + opt.iters; ++i) {
            if (i == opt.warmup) check(aclrtRecordEvent(start, stream), "aclrtRecordEvent");
            ll_dispatch_run(session, stream, tensors, b, h, k, e, int(rows));
        }
        check(aclrtRecordEvent(stop, stream), "aclrtRecordEvent");
        check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
        float elapsed = 0.0F;
        check(aclrtEventElapsedTime(&elapsed, start, stop), "aclrtEventElapsedTime");
        check(aclrtEventDestroy(start), "aclrtEventDestroy");
        check(aclrtEventDestroy(stop), "aclrtEventDestroy");

        std::vector<int64_t> local(size_t(local_experts), 0);
        check(aclrtMemcpy(local.data(), local.size() * sizeof(int64_t), counts.pointer, local.size() * sizeof(int64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy(D2H)");

        // 参考结果：每个 (token, slot) 恰好落到一个目标卡的本地专家上。
        std::vector<int64_t> expected(size_t(world) * local_experts, 0);
        for (int source = 0; source < world; ++source)
            for (int token = 0; token < b; ++token)
                for (int slot = 0; slot < k; ++slot) {
                    const int64_t expert = route_expert(source, token, slot, e);
                    ++expected[size_t(expert / local_experts) * local_experts + size_t(expert % local_experts)];
                }

        std::vector<int64_t> gathered(expected.size(), 0);
        mpi_check(MPI_Allgather(local.data(), local_experts, MPI_LONG_LONG, gathered.data(), local_experts,
                                MPI_LONG_LONG, MPI_COMM_WORLD),
                  "MPI_Allgather");

        int64_t total = 0;
        int mismatches = 0;
        std::string first_mismatch;
        for (size_t i = 0; i < gathered.size(); ++i) {
            total += gathered[i];
            if (gathered[i] == expected[i]) continue;
            if (!mismatches) {
                first_mismatch = "rank=" + std::to_string(i / local_experts) + " local_expert=" +
                                 std::to_string(i % local_experts) + " got=" + std::to_string(gathered[i]) +
                                 " want=" + std::to_string(expected[i]);
            }
            ++mismatches;
        }
        int local_ok = (mismatches == 0 && total == int64_t(b) * k) ? 1 : 0;
        int all_ok = 0;
        mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce");
        if (!peers.rank) {
            std::cout << "tokens=" << b << " hidden=" << h << " topk=" << k << " experts=" << e << " world=" << world
                      << " capacity=" << opt.capacity << " backend=" << opt.backend
                      << " rows=" << rows << std::endl;
            std::cout << "dispatch avg=" << (elapsed * 1000.0F / float(opt.iters)) << "us over " << opt.iters
                      << " iters (warmup " << opt.warmup << ")" << std::endl;
            std::cout << "expert counts total=" << total << " expect=" << (int64_t(b) * k)
                      << " mismatches=" << mismatches << std::endl;
            if (!first_mismatch.empty()) std::cout << "first mismatch: " << first_mismatch << std::endl;
        }
        if (!all_ok) throw std::runtime_error("dispatch expert count mismatch");

        peers.barrier();
        session.close(true);
        peers.barrier();
    }

    check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    check(aclrtResetDevice(device), "aclrtResetDevice");
    check(aclFinalize(), "aclFinalize");
}

void usage()
{
    std::cout << "LL C++ dispatch sample (MPI bootstrap)\n"
                 "--tokens 32 --hidden 3584 --topk 8 --experts 384 --capacity 32\n"
                 "--dtype bf16|fp16 --backend rdma|mte --warmup 5 --iters 20 --timeout 300 --device-base 0\n";
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--help") {
        usage();
        return 0;
    }
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed" << std::endl;
        return 1;
    }
    int rank = -1;
    try {
        const Coordinator peers;
        rank = peers.rank;
        run(peers, parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << " FAIL: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
