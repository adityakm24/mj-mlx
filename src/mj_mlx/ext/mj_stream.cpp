// Copyright 2026 the MilliJoule authors
// SPDX-License-Identifier: Apache-2.0
// MilliJoule: GPU-driven expert streaming for MLX on Apple silicon (prototype).
//
// One decode layer of a streamed MoE, without any host sync in the Python step:
//   GPU: ... router -> [StreamRoute: signal route_event = v, commit] -> [wait ready_event = v] -> experts ...
//   IO thread (C++): waits route_event >= v, reads this layer's routing straight from unified memory, makes the
//   routed experts resident in the layer's slab (preadv from the repacked expert file into the slot arrays, LRU),
//   writes each (token, k) pair's slot id into the primitive's output buffer, signals ready_event = v, then starts
//   reads for the predicted experts of the next layers (prefetch). The GPU's expert matmul then runs on those
//   slots: the same gather_qmm call as expert_stream's speculative fast path, so results are bit-identical.
// The Python host only builds the graph; it never waits for routing or for the SSD mid-step.

#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>

#include "mlx/allocator.h"
#include "mlx/array.h"
// Device keeps the per-command-buffer op / MB limits (MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER, read once at
// startup) private; set_buffer_limits() below changes them between steps (gated decode wants one buffer per step,
// prefill needs small ones), so the header is included with its private members visible.
#define private public
#include "mlx/backend/metal/device.h"
#undef private
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int F_NOCACHE_ = 48;
constexpr int F_RDAHEAD_ = 45;

class Pool {
 public:
  explicit Pool(int n) {
    for (int i = 0; i < n; ++i)
      threads_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> g(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  std::shared_future<ssize_t> submit(std::function<ssize_t()> f) {
    auto task = std::make_shared<std::packaged_task<ssize_t()>>(std::move(f));
    auto fut = task->get_future().share();
    {
      std::lock_guard<std::mutex> g(m_);
      q_.push_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

 private:
  void loop() {
    for (;;) {
      std::function<void()> f;
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return;
        f = std::move(q_.front());
        q_.pop_front();
      }
      f();
    }
  }
  std::vector<std::thread> threads_;
  std::deque<std::function<void()>> q_;
  std::mutex m_;
  std::condition_variable cv_;
  bool stop_ = false;
};

struct Slab {
  int S = 0, E = 0;
  int fd = -1;
  size_t stride = 0;
  std::vector<uint8_t*> part_ptr;  // per part: base of the [S, ...] slot array
  std::vector<size_t> part_bytes;  // per part: bytes per slot
  std::vector<mx::array> keep;     // keeps the slot arrays alive
  std::vector<int> slot_of, expert_of;
  std::list<int> lru;  // resident experts, least recently used first
  std::unordered_map<int, std::list<int>::iterator> lru_pos;
  std::unordered_map<int, std::shared_future<ssize_t>> pending;  // expert -> in-flight read
  std::unordered_set<int> prefetched;

  void touch(int e) {
    auto it = lru_pos.find(e);
    if (it != lru_pos.end()) lru.erase(it->second);
    lru.push_back(e);
    lru_pos[e] = std::prev(lru.end());
  }
  void drop(int e) {
    auto it = lru_pos.find(e);
    if (it != lru_pos.end()) {
      lru.erase(it->second);
      lru_pos.erase(it);
    }
  }
  int take_slot(const std::unordered_set<int>& protect) {
    for (int s = S - 1; s >= 0; --s)
      if (expert_of[s] < 0) return s;
    for (int e : lru) {
      if (protect.count(e) || pending.count(e)) continue;
      int s = slot_of[e];
      slot_of[e] = -1;
      expert_of[s] = -1;
      drop(e);
      prefetched.erase(e);
      return s;
    }
    return -1;
  }
};

struct Job {
  int layer;
  uint64_t value;
  mx::array idx;                 // routing of this layer, [T, k] (uint32 or int32)
  std::vector<mx::array> preds;  // predicted routing of the next layers, nearest first
  mx::array out;                 // slot id per (token, k), int32 [T*k]
};

class Streamer;

class StreamRoute : public mx::UnaryPrimitive {
 public:
  StreamRoute(mx::Stream s, Streamer* st, int layer) : mx::UnaryPrimitive(s), st_(st), layer_(layer) {}
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("[mj_stream] GPU only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;
  const char* name() const override {
    return "MJStreamRoute";
  }

 private:
  Streamer* st_;
  int layer_;
};

class Streamer {
 public:
  Streamer(const std::string& path, size_t stride, int threads) : path_(path), stride_(stride), pool_(threads) {
    auto* dev = mx::metal::device(mx::Device::gpu).mtl_device();
    route_ev_ = dev->newSharedEvent();
    ready_ev_ = dev->newSharedEvent();
    setup_keepalive(dev);
    io_ = std::thread([this] { io_loop(); });
  }
  ~Streamer() {
    {
      std::lock_guard<std::mutex> g(m_);
      stop_ = true;
    }
    cv_.notify_all();
    io_.join();
    for (auto& s : slabs_)
      if (s.fd >= 0) close(s.fd);
    route_ev_->release();
    ready_ev_->release();
  }

  void add_layer(int layer, int num_experts, const std::vector<mx::array>& parts, const std::vector<int>& slot_of) {
    if ((int)slabs_.size() <= layer) slabs_.resize(layer + 1);
    Slab& s = slabs_[layer];
    if (s.fd >= 0) {  // already registered: take over the Python slab's current residency (handoff after prefill)
      drain();
      import_slots(s, slot_of);
      return;
    }
    char name[64];
    snprintf(name, sizeof(name), "/layer_%02d.bin", layer);
    s.fd = open((path_ + name).c_str(), O_RDONLY);
    if (s.fd < 0) throw std::runtime_error("[mj_stream] cannot open " + path_ + name);
    fcntl(s.fd, F_RDAHEAD_, 0);
    fcntl(s.fd, F_NOCACHE_, 1);
    s.E = num_experts;
    s.S = parts[0].shape(0);
    s.stride = stride_;
    for (auto& p : parts) {
      s.part_ptr.push_back(const_cast<uint8_t*>(p.data<uint8_t>()));
      s.part_bytes.push_back(p.nbytes() / s.S);
      s.keep.push_back(p);
    }
    import_slots(s, slot_of);
  }

  static void import_slots(Slab& s, const std::vector<int>& slot_of) {
    s.slot_of.assign(s.E, -1);
    s.expert_of.assign(s.S, -1);
    s.lru.clear();
    s.lru_pos.clear();
    s.pending.clear();
    s.prefetched.clear();
    for (int e = 0; e < s.E && e < (int)slot_of.size(); ++e)
      if (slot_of[e] >= 0) {
        s.slot_of[e] = slot_of[e];
        s.expert_of[slot_of[e]] = e;
        s.touch(e);
      }
  }

  // GPU keep-alive: while this layer's expert reads are in flight the GPU would idle and drop its clocks (the next
  // kernels then run 2-4x slower, RESULTS 1u). A tiny spin kernel on a separate queue, sized to the expected read
  // time, keeps it awake.
  void setup_keepalive(MTL::Device* dev) {
    const char* src = R"(
#include <metal_stdlib>
using namespace metal;
kernel void mj_spin(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                    uint tid [[thread_position_in_grid]]) {
  float a = float(tid) * 1e-7f + 1.0f;
  for (uint i = 0; i < iters; ++i) { a = fma(a, 1.0000001f, 1e-7f); }
  out[tid] = a;
})";
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    NS::Error* err = nullptr;
    auto* lib = dev->newLibrary(NS::String::string(src, NS::UTF8StringEncoding), nullptr, &err);
    if (!lib) throw std::runtime_error("[mj_stream] keep-alive kernel compile failed");
    auto* fn = lib->newFunction(NS::String::string("mj_spin", NS::UTF8StringEncoding));
    spin_pso_ = dev->newComputePipelineState(fn, &err);
    fn->release();
    lib->release();
    ka_queue_ = dev->newCommandQueue();
    ka_out_ = dev->newBuffer(64 * sizeof(float), MTL::ResourceStorageModeShared);
    // calibrate: GPU seconds per iteration (warm, back to back)
    for (int r = 0; r < 3; ++r) {
      auto* cb = launch_spin(200000);
      cb->waitUntilCompleted();
      spin_s_per_iter_ = (cb->GPUEndTime() - cb->GPUStartTime()) / 200000.0;
    }
    pool->release();
  }

  MTL::CommandBuffer* launch_spin(uint32_t iters) {
    auto* cb = ka_queue_->commandBuffer();
    auto* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(spin_pso_);
    enc->setBuffer(ka_out_, 0, 0);
    enc->setBytes(&iters, sizeof(iters), 1);
    enc->dispatchThreads(MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
    enc->endEncoding();
    cb->commit();
    return cb;
  }

  std::vector<int> slot_of(int layer) {
    drain();
    return slabs_.at(layer).slot_of;
  }

  mx::array route(int layer, const mx::array& idx, const std::vector<mx::array>& preds) {
    // The IO thread reads these buffers as packed [T*k]: a top-k slice of an argpartition is strided for T > 1.
    std::vector<mx::array> inputs{mx::contiguous(idx)};
    for (auto& p : preds) inputs.push_back(mx::contiguous(p));
    auto s = mx::default_stream(mx::Device::gpu);
    return mx::array({(int)idx.size()}, mx::int32, std::make_shared<StreamRoute>(s, this, layer), inputs);
  }

  // Called from StreamRoute::eval_gpu while MLX encodes the step.
  void encode(int layer, const std::vector<mx::array>& inputs, mx::array& out) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    uint64_t v = ++counter_;
    std::vector<mx::array> preds(inputs.begin() + 1, inputs.end());
    {
      std::lock_guard<std::mutex> g(m_);
      jobs_.push_back(Job{layer, v, inputs[0], preds, out});
    }
    cv_.notify_one();
    auto& enc = mx::metal::get_command_encoder(out.primitive().stream());
    // Exact positions in MLX's own command buffer (MLX commits it every few dozen ops; committing it here would
    // break MLX's completion handler): signal after the router's kernels, then wait before any later kernel.
    enc.end_encoding();
    auto* cb = enc.get_command_buffer();
    cb->encodeSignalEvent(route_ev_, v);
    cb->encodeWait(ready_ev_, v);

  }

  std::unordered_map<std::string, double> stats() {
    drain();
    return {{"hits", (double)hits_}, {"misses", (double)misses_}, {"prefetch_issued", (double)pf_issued_},
            {"prefetch_used", (double)pf_used_}, {"bytes_read", (double)bytes_read_},
            {"route_wait_s", route_wait_s_}, {"read_wait_s", read_wait_s_}, {"jobs", (double)n_jobs_},
            {"keepalive_launched", (double)ka_launched_}, {"spin_us_per_kiter", spin_s_per_iter_ * 1e9}};
  }
  void reset_stats() {
    drain();
    hits_ = misses_ = pf_issued_ = pf_used_ = bytes_read_ = n_jobs_ = ka_launched_ = 0;
    route_wait_s_ = read_wait_s_ = 0;
  }

 private:
  void drain() {  // wait until the IO thread has no queued jobs and no reads in flight
    std::unique_lock<std::mutex> l(m_);
    idle_cv_.wait(l, [this] { return jobs_.empty() && !busy_; });
    l.unlock();
    for (auto& s : slabs_)
      for (auto& [e, f] : s.pending) bytes_read_ += f.get();
    for (auto& s : slabs_) s.pending.clear();
  }

  // Retire completed prefetch reads: an expert stays protected only while its read is in flight (mispredicted
  // prefetches would otherwise pin their slots forever).
  void harvest(Slab& s) {
    for (auto it = s.pending.begin(); it != s.pending.end();) {
      if (it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        bytes_read_ += it->second.get();
        it = s.pending.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::shared_future<ssize_t> read(Slab& s, int e, int slot) {
    std::vector<iovec> iov;
    for (size_t p = 0; p < s.part_ptr.size(); ++p)
      iov.push_back(iovec{s.part_ptr[p] + (size_t)slot * s.part_bytes[p], s.part_bytes[p]});
    int fd = s.fd;
    off_t off = (off_t)e * (off_t)s.stride;
    return pool_.submit([fd, off, iov]() -> ssize_t {
      ssize_t n = preadv(fd, iov.data(), (int)iov.size(), off);
      if (n < 0) throw std::runtime_error("[mj_stream] preadv failed");
      return n;
    });
  }

  void prefetch(Slab& s, const mx::array& pred) {
    const int n = (int)pred.size();
    std::vector<int> want;
    std::unordered_set<int> seen;
    if (pred.dtype() == mx::uint32) {
      auto* p = pred.data<uint32_t>();
      for (int i = 0; i < n; ++i) if (seen.insert((int)p[i]).second) want.push_back((int)p[i]);
    } else {
      auto* p = pred.data<int32_t>();
      for (int i = 0; i < n; ++i) if (seen.insert(p[i]).second) want.push_back(p[i]);
    }
    if ((int)want.size() > s.S) return;
    harvest(s);
    std::unordered_set<int> protect(want.begin(), want.end());
    for (auto& [e, f] : s.pending) protect.insert(e);
    for (int e : want) {
      if (s.slot_of[e] >= 0) continue;  // resident or in flight
      int slot = s.take_slot(protect);
      if (slot < 0) return;
      s.slot_of[e] = slot;
      s.expert_of[slot] = e;
      s.touch(e);
      s.pending[e] = read(s, e, slot);
      s.prefetched.insert(e);
      ++pf_issued_;
    }
  }

  void run(Job& j) {
    Slab& s = slabs_.at(j.layer);
    auto t0 = Clock::now();
    // Spin briefly on the event value (wake latency of a few us), then fall back to the blocking wait.
    if (route_ev_->signaledValue() < j.value) {
      auto spin_until = Clock::now() + std::chrono::microseconds(spin_us_);
      while (route_ev_->signaledValue() < j.value && Clock::now() < spin_until) {
        __builtin_arm_yield();
      }
      if (route_ev_->signaledValue() < j.value && !route_ev_->waitUntilSignaledValue(j.value, 60000))
        throw std::runtime_error("[mj_stream] route timeout");
    }
    auto t1 = Clock::now();
    route_wait_s_ += std::chrono::duration<double>(t1 - t0).count();
    const int n = (int)j.idx.size();
    std::vector<int> ids(n);
    if (j.idx.dtype() == mx::uint32) {
      auto* p = j.idx.data<uint32_t>();
      for (int i = 0; i < n; ++i) ids[i] = (int)p[i];
    } else {
      auto* p = j.idx.data<int32_t>();
      for (int i = 0; i < n; ++i) ids[i] = p[i];
    }
    std::unordered_set<int> need(ids.begin(), ids.end());
    harvest(s);
    std::vector<std::shared_future<ssize_t>> wait;
    for (int e : need) {
      auto pit = s.pending.find(e);
      if (pit != s.pending.end()) {  // its prefetch read must land first
        wait.push_back(pit->second);
        s.pending.erase(pit);
      }
      if (s.prefetched.erase(e)) ++pf_used_;
    }
    for (int e : need) {
      if (s.slot_of[e] >= 0) {
        ++hits_;
        s.touch(e);
        continue;
      }
      ++misses_;
      int slot = s.take_slot(need);
      if (slot < 0) {
        int nres = 0;
        for (int x : s.expert_of) nres += x >= 0;
        throw std::runtime_error("[mj_stream] no free slot: layer " + std::to_string(j.layer) + " S " +
                                 std::to_string(s.S) + " need " + std::to_string(need.size()) + " ids " +
                                 std::to_string(n) + " resident " + std::to_string(nres) + " lru " +
                                 std::to_string(s.lru.size()) + " pending " + std::to_string(s.pending.size()));
      }
      s.slot_of[e] = slot;
      s.expert_of[slot] = e;
      s.touch(e);
      wait.push_back(read(s, e, slot));
    }
    if (keepalive_ && !wait.empty() && spin_s_per_iter_ > 0) {
      size_t rec = 0;
      for (auto b : s.part_bytes) rec += b;
      double est = (double)rec * (double)wait.size() / (ka_gbps_ * 1e9);  // expected read time
      NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
      launch_spin((uint32_t)std::min(est / spin_s_per_iter_, 4.0e6));
      pool->release();
      ++ka_launched_;
    }
    for (auto& f : wait) bytes_read_ += f.get();
    read_wait_s_ += std::chrono::duration<double>(Clock::now() - t1).count();
    auto* o = j.out.data<int32_t>();
    for (int i = 0; i < n; ++i) o[i] = s.slot_of[ids[i]];
    ready_ev_->setSignaledValue(j.value);
    for (size_t d = 0; d < j.preds.size(); ++d)
      if (j.layer + 1 + (int)d < (int)slabs_.size()) prefetch(slabs_[j.layer + 1 + d], j.preds[d]);
    ++n_jobs_;
  }

  void io_loop() {
    for (;;) {
      Job j{0, 0, mx::array(0), {}, mx::array(0)};
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return stop_ || !jobs_.empty(); });
        if (stop_ && jobs_.empty()) return;
        j = jobs_.front();
        jobs_.pop_front();
        busy_ = true;
      }
      run(j);
      {
        std::lock_guard<std::mutex> g(m_);
        busy_ = false;
      }
      idle_cv_.notify_all();
    }
  }

  std::string path_;
  size_t stride_;
  Pool pool_;
  std::vector<Slab> slabs_;
  MTL::SharedEvent* route_ev_;
  MTL::SharedEvent* ready_ev_;
  std::atomic<uint64_t> counter_{0};
  std::deque<Job> jobs_;
  std::mutex m_;
  std::condition_variable cv_, idle_cv_;
  bool stop_ = false, busy_ = false;
  std::thread io_;
  uint64_t hits_ = 0, misses_ = 0, pf_issued_ = 0, pf_used_ = 0, bytes_read_ = 0, n_jobs_ = 0;
  double route_wait_s_ = 0, read_wait_s_ = 0;

 public:
  int spin_us_ = 1000;
  bool keepalive_ = false;
  double ka_gbps_ = 2.5;  // assumed SSD rate for sizing the keep-alive spin

 private:
  MTL::ComputePipelineState* spin_pso_ = nullptr;
  MTL::CommandQueue* ka_queue_ = nullptr;
  MTL::Buffer* ka_out_ = nullptr;
  double spin_s_per_iter_ = 0;
  uint64_t ka_launched_ = 0;
};

void StreamRoute::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
  st_->encode(layer_, inputs, out);
}


// ---- Host gate: queue a step on the GPU before its host-written inputs exist ------------------------------------
// gate(x, v) returns x unchanged, but every kernel after it in the stream waits until the host calls signal(v).
// Lean gpt-oss decode builds step N+1 on a host embedding buffer while step N runs; with the gate the whole of N+1
// is encoded and committed right away, and the host only writes the new token's embedding row and signals once N's
// token is known, so the GPU waits ~the host rescore instead of the rescore plus encoding N+1 (RESULTS 1zc).
class HostGate;

class GateWait : public mx::UnaryPrimitive {
 public:
  GateWait(mx::Stream s, HostGate* g, uint64_t v) : mx::UnaryPrimitive(s), g_(g), v_(v) {}
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("[mj_gate] GPU only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;
  const char* name() const override {
    return "MJHostGate";
  }

 private:
  HostGate* g_;
  uint64_t v_;
};

class HostGate {
 public:
  HostGate() {
    ev_ = mx::metal::device(mx::Device::gpu).mtl_device()->newSharedEvent();
  }
  ~HostGate() {
    release_all();
    ev_->release();
  }
  // A new gate value; kernels after gate(x, v) wait for signal(v) (or any later value).
  uint64_t next() {
    return ++issued_;
  }
  mx::array gate(const mx::array& x, uint64_t v) {
    auto s = mx::default_stream(mx::Device::gpu);
    return mx::array(x.shape(), x.dtype(), std::make_shared<GateWait>(s, this, v), {x});
  }
  void signal(uint64_t v) {
    if (v > ev_->signaledValue()) ev_->setSignaledValue(v);
  }
  // Opens every gate issued so far (teardown or an abandoned step: the GPU must never wait forever).
  void release_all() {
    signal(issued_);
  }
  uint64_t value() {
    return ev_->signaledValue();
  }
  void encode(const std::vector<mx::array>& inputs, mx::array& out, uint64_t v, mx::Stream s) {
    out.copy_shared_buffer(inputs[0]);
    auto& enc = mx::metal::get_command_encoder(s);
    enc.end_encoding();
    enc.get_command_buffer()->encodeWait(ev_, v);
  }

 private:
  MTL::SharedEvent* ev_;
  uint64_t issued_ = 0;
};

void GateWait::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
  g_->encode(inputs, out, v_, stream());
}

}  // namespace

NB_MODULE(_mj_stream, m) {
  m.doc() = "GPU-driven expert streaming for MLX (MilliJoule prototype)";
  nb::class_<Streamer>(m, "Streamer")
      .def(nb::init<const std::string&, size_t, int>(), nb::arg("path"), nb::arg("stride"), nb::arg("threads") = 8)
      .def("add_layer", &Streamer::add_layer, nb::arg("layer"), nb::arg("num_experts"), nb::arg("parts"),
           nb::arg("slot_of"))
      .def("route", &Streamer::route, nb::arg("layer"), nb::arg("idx"), nb::arg("preds"))
      .def("slot_of", &Streamer::slot_of)
      .def("stats", &Streamer::stats)
      .def("reset_stats", &Streamer::reset_stats)
      .def_rw("spin_us", &Streamer::spin_us_)
      .def_rw("keepalive", &Streamer::keepalive_)
      .def_rw("keepalive_gbps", &Streamer::ka_gbps_);
  m.def("buffer_limits", []() {
    auto& d = mx::metal::device(mx::Device::gpu);
    return std::vector<int>{d.max_ops_per_buffer_, d.max_mb_per_buffer_};
  });
  m.def("set_buffer_limits", [](int ops, int mb) {
    auto& d = mx::metal::device(mx::Device::gpu);
    d.max_ops_per_buffer_ = ops;
    d.max_mb_per_buffer_ = mb;
  }, nb::arg("max_ops"), nb::arg("max_mb"));
  nb::class_<HostGate>(m, "HostGate")
      .def(nb::init<>())
      .def("next", &HostGate::next)
      .def("gate", &HostGate::gate, nb::arg("x"), nb::arg("value"))
      .def("signal", &HostGate::signal, nb::arg("value"))
      .def("release_all", &HostGate::release_all)
      .def("value", &HostGate::value);
}
