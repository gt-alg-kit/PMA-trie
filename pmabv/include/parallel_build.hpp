#pragma once

namespace pmabv {

struct ParallelBuildTag { explicit ParallelBuildTag() = default; };
inline constexpr ParallelBuildTag parallel_build_tag{};
// With ParallelBuildTag: "this argument is a complete packed run" (bits, the
// payload, the flags and, for the supported id keys, the fields and their
// samples) -- BitPMAImpl's PackedSeqTag constructor.
struct PackedSeqTag { explicit PackedSeqTag() = default; };

}  // namespace pmabv
