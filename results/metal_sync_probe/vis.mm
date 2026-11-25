// Does a NEW compute dispatch see a CPU write made while its command buffer is already running?
// (A running kernel does not: RESULTS 1u.) One command buffer of N short dispatches; each spins ~SPIN iterations,
// then reads a shared flag and records it. The CPU writes the flag a few ms after commit.
// Modes: 0 = one encoder, N dispatches; 1 = one encoder per dispatch; 2 = one encoder, memoryBarrier between.
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <thread>
#include <atomic>
#include <vector>

static const char* SRC = R"(
#include <metal_stdlib>
using namespace metal;
kernel void probe(device atomic_uint* flag [[buffer(0)]], device uint* out [[buffer(1)]],
                  constant uint& idx [[buffer(2)]], constant uint& spin [[buffer(3)]],
                  device float* sink [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
  float a = float(tid) + 1.0f;
  for (uint i = 0; i < spin; ++i) a = fma(a, 1.0000001f, 1e-7f);
  if (tid == 0) { out[idx] = atomic_load_explicit(flag, memory_order_relaxed); sink[0] = a; }
}
kernel void probe_plain(device uint* flag [[buffer(0)]], device uint* out [[buffer(1)]],
                  constant uint& idx [[buffer(2)]], constant uint& spin [[buffer(3)]],
                  device float* sink [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
  float a = float(tid) + 1.0f;
  for (uint i = 0; i < spin; ++i) a = fma(a, 1.0000001f, 1e-7f);
  if (tid == 0) { out[idx] = ((volatile device uint*)flag)[0]; sink[0] = a; }
}
)";

int main(int argc, char** argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  int N = argc > 2 ? atoi(argv[2]) : 2000;
  uint32_t spin = argc > 3 ? atoi(argv[3]) : 2000;
  int delay_us = argc > 4 ? atoi(argv[4]) : 5000;
  int plain = argc > 5 ? atoi(argv[5]) : 0;
  int wc = argc > 6 ? atoi(argv[6]) : 0;
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(SRC) options:nil error:&err];
    if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:plain ? @"probe_plain" : @"probe"] error:&err];
    MTLResourceOptions fo = MTLResourceStorageModeShared | (wc ? MTLResourceCPUCacheModeWriteCombined : 0);
    id<MTLBuffer> flag = [dev newBufferWithLength:16 options:fo];
    id<MTLBuffer> out = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> sink = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (int rep = 0; rep < 3; ++rep) {
      ((volatile uint32_t*)flag.contents)[0] = 0;
      memset(out.contents, 0xff, N * 4);
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = nil;
      for (uint32_t i = 0; i < (uint32_t)N; ++i) {
        if (!enc || mode == 1) {
          if (enc) [enc endEncoding];
          enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:ps];
          [enc setBuffer:flag offset:0 atIndex:0];
          [enc setBuffer:out offset:0 atIndex:1];
          [enc setBuffer:sink offset:0 atIndex:4];
          [enc setBytes:&spin length:4 atIndex:3];
        }
        [enc setBytes:&i length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        if (mode == 2) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      [enc endEncoding];
      std::vector<id<MTLCommandBuffer>> cbs;
      if (mode == 3) {  // N command buffers of one dispatch each, all committed before the write
        for (uint32_t i = 0; i < (uint32_t)N; ++i) {
          id<MTLCommandBuffer> c = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
          [e setComputePipelineState:ps];
          [e setBuffer:flag offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1]; [e setBuffer:sink offset:0 atIndex:4];
          [e setBytes:&spin length:4 atIndex:3]; [e setBytes:&i length:4 atIndex:2];
          [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
          [e endEncoding];
          cbs.push_back(c);
        }
        cb = cbs.back();
      }
      auto t0 = std::chrono::steady_clock::now();
      if (mode == 3) { for (auto& c : cbs) [c commit]; } else [cb commit];
      std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
      ((volatile uint32_t*)flag.contents)[0] = 1;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      auto tw = std::chrono::steady_clock::now();
      [cb waitUntilCompleted];
      auto t1 = std::chrono::steady_clock::now();
      double gpu_ms = ((mode == 3 ? cb.GPUEndTime : cb.GPUEndTime) - (mode == 3 ? cbs.front().GPUStartTime : cb.GPUStartTime)) * 1e3;
      double per = gpu_ms / N;
      uint32_t* o = (uint32_t*)out.contents;
      int first = -1, ones = 0;
      for (int i = 0; i < N; ++i) { if (o[i] == 1) { ones++; if (first < 0) first = i; } }
      double write_ms = std::chrono::duration<double, std::milli>(tw - t0).count();
      double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      printf("mode %d plain %d wc %d rep %d: N %d gpu %.2f ms (%.1f us/dispatch), CPU wrote at %.2f ms after commit, "
             "first dispatch seeing it %d (~%.2f ms into GPU time), dispatches seeing it %d, last=%u, wall %.2f\n",
             mode, plain, wc, rep, N, gpu_ms, per * 1e3, write_ms, first, first >= 0 ? first * per : -1.0, ones,
             o[N - 1], wall_ms);
    }
  }
  return 0;
}
