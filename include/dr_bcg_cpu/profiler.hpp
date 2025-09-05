#pragma once

#include <array>
#include <chrono>

enum class Event { GetXi, GetX, GetResidual, GetWZeta, GetS, GetSigma, Count };

struct SolverTimings {
    std::array<std::chrono::nanoseconds, static_cast<size_t>(Event::Count)>
        totals;
};

#ifndef PROFILE_SOLVER
#define PROFILE_SOLVER 0
#endif

template <bool Enable> class Profiler;

template <> class Profiler<true> {
  public:
    std::array<std::chrono::nanoseconds, static_cast<size_t>(Event::Count)>
        totals{};
    void add(Event e, std::chrono::nanoseconds dt) noexcept {
        totals[static_cast<size_t>(e)] += dt;
    };
};

template <> class Profiler<false> {
  public:
    inline void add(Event, std::chrono::nanoseconds) noexcept {};
};

template <bool Enable> class ScopedEvent {
    using clock = std::chrono::steady_clock;
    Profiler<Enable> &p_;
    Event e_;
    clock::time_point t0_;

  public:
    ScopedEvent(Profiler<Enable> &p, Event e) noexcept
        : p_(p), e_(e), t0_(clock::now()) {};
    ~ScopedEvent() noexcept {
        if constexpr (Enable) {
            auto t1 = clock::now();
            p_.add(e_, std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t1 - t0_));
        }
    }
};

#define PROFILE_BLOCK(prof, ev)                                                \
    ScopedEvent<(PROFILE_SOLVER != 0)> _scoped_##__LINE__ { prof, ev }
