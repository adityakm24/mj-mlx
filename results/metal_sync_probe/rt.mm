// Where does a GPU->CPU->GPU MTLSharedEvent round trip's time go (M4)? One command buffer of R "layers":
//   kernel (spin ~K iters) ; [end encoder] ; signal(e, 2i+1) ; wait(e, 2i+2) ; [new encoder]
// mode 0: no events, one encoder (baseline: kernels back to back)
// mode 1: encoder break per layer only (end + new encoder, no events)
// mode 2: encoder break + signal + wait, the wait value pre-signaled by the CPU before commit (wait already satisfied)
// mode 3: encoder break + signal + wait, a CPU thread spins on signaledValue >= 2i+1 then sets 2i+2 (the real pattern)
// mode 4: as 3 but the CPU thread uses waitUntilSignaledValue (blocking) instead of spinning
// mode 5: as 3 but the GPU signal/wait go through MTLEvent?  (not possible for CPU: skipped) -> signal only, no wait
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <thread>

static const char* SRC = R"(
#include <metal_stdlib>
using namespace metal;
kernel void work(device float* sink [[buffer(0)]], constant uint& spin [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  float a = float(tid) + 1.0f;
  for (uint i = 0; i < spin; ++i) a = fma(a, 1.0000001f, 1e-7f);
  if (a == 0.0f) sink[tid] = a;
}
)";

int main(int argc, char** argv) {
  int mode = atoi(argv[1]);
  int R = argc > 2 ? atoi(argv[2]) : 40;
  uint32_t spin = argc > 3 ? atoi(argv[3]) : 2000;
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(SRC) options:nil error:&err];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"work"] error:&err];
    id<MTLBuffer> sink = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLSharedEvent> ev = [dev newSharedEvent];
    uint64_t base = 0;
    double tot = 0; int reps = 6;
    for (int rep = 0; rep < reps + 1; ++rep) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (int i = 0; i < R; ++i) {
        if (!enc) enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:sink offset:0 atIndex:0];
        [enc setBytes:&spin length:4 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (mode >= 1) { [enc endEncoding]; enc = nil; }
        if (mode >= 2) {
          [cb encodeSignalEvent:ev value:base + 2 * i + 1];
          if (mode != 5) [cb encodeWaitForEvent:ev value:base + 2 * i + 2];
        }
      }
      if (enc) [enc endEncoding];
      if (mode == 2) ev.signaledValue = base + 2 * R;  // every wait already satisfied (signals below it are no-ops)
      std::thread th;
      if (mode == 3 || mode == 4) {
        th = std::thread([&, base] {
          for (int i = 0; i < R; ++i) {
            uint64_t v = base + 2 * i + 1;
            if (mode == 3) { while (ev.signaledValue < v) {} }
            else [ev waitUntilSignaledValue:v timeoutMS:10000];
            ev.signaledValue = v + 1;
          }
        });
      }
      [cb commit];
      [cb waitUntilCompleted];
      if (th.joinable()) th.join();
      double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
      if (rep > 0) tot += ms;
      base += 2 * R + 2;
      if (mode == 2 || mode == 5) ev.signaledValue = base;
    }
    double per = tot / reps / R * 1e3;
    printf("mode %d R %d spin %u: %.1f us per layer (GPU time per command buffer / R)\n", mode, R, spin, per);
  }
  return 0;
}
