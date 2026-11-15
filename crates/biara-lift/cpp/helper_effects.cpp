// What QEMU helpers read and write through `env` (see src/helper_effects.rs),
// from their bitcode: TCG only knows whether a helper may touch globals at
// all, so lifted code would otherwise store every global to env before each
// such helper and reload every one after it.
//
// A helper's pointer parameters are followed through GEPs to the loads,
// stores and memory intrinsics that use them, and into the defined functions
// they are passed to. A variable index widens the offset to the index's known
// range (`fptags[fpstt & 7]`). Anything else (an unbounded index, the pointer
// escaping, a call to code without a body) makes the helper's effect unknown. Calls that do not return (exception raising) are not followed:
// lifted code does not continue past them.

#include "llvm-c/Types.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {
struct Range {
  int64_t Lo, Hi; // [Lo, Hi)
};

// Byte ranges read and written relative to a pointer; `Unknown`: anything.
struct Effects {
  bool Unknown = false;
  std::vector<Range> Reads, Writes;

  void merge(const Effects &O, int64_t Shift) {
    Unknown |= O.Unknown;
    for (const Range &R : O.Reads)
      Reads.push_back({R.Lo + Shift, R.Hi + Shift});
    for (const Range &R : O.Writes)
      Writes.push_back({R.Lo + Shift, R.Hi + Shift});
  }
};

struct Analysis {
  const DataLayout &DL;
  std::map<std::pair<const Function *, unsigned>, Effects> Done;
  std::set<std::pair<const Function *, unsigned>> Active;

  // Uses of `P`, which points [Lo, Hi] bytes past the parameter.
  void uses(const Value *P, int64_t Lo, int64_t Hi, Effects &E) {
    for (const Use &U : P->uses()) {
      if (E.Unknown)
        return;
      const User *Us = U.getUser();
      if (auto *G = dyn_cast<GEPOperator>(Us); G && G->getPointerOperand() == P) {
        auto Delta = offsets(*G);
        if (!Delta) {
          E.Unknown = true;
          return;
        }
        uses(G, Lo + Delta->first, Hi + Delta->second, E);
      } else if (auto *L = dyn_cast<LoadInst>(Us)) {
        E.Reads.push_back({Lo, Hi + (int64_t)DL.getTypeStoreSize(L->getType())});
      } else if (auto *S = dyn_cast<StoreInst>(Us); S && S->getPointerOperand() == P &&
                                                     S->getValueOperand() != P) {
        int64_t Size = DL.getTypeStoreSize(S->getValueOperand()->getType());
        E.Writes.push_back({Lo, Hi + Size});
      } else if (auto *CB = dyn_cast<CallBase>(Us); CB && CB->isArgOperand(&U)) {
        call(*CB, CB->getArgOperandNo(&U), Lo, Hi, E);
      } else if (isa<ICmpInst>(Us)) {
        // Comparing the pointer reads nothing.
      } else {
        E.Unknown = true;
      }
    }
  }

  // The smallest and largest byte offset `G` adds, if its indices are
  // bounded (a zero-extended 32-bit index still rules out what lies below).
  std::optional<std::pair<int64_t, int64_t>> offsets(const GEPOperator &G) {
    unsigned Bits = DL.getIndexTypeSizeInBits(G.getType());
    SmallMapVector<Value *, APInt, 4> Variable;
    APInt Constant(Bits, 0);
    if (!G.collectOffset(DL, Bits, Variable, Constant))
      return std::nullopt;
    int64_t Lo = Constant.getSExtValue(), Hi = Lo;
    const int64_t Limit = int64_t(1) << 40;
    for (auto &[V, Scale] : Variable) {
      ConstantRange R =
          ConstantRange::fromKnownBits(computeKnownBits(V, DL), /*IsSigned=*/true);
      if (R.isFullSet() || R.getSignedMin().getSignificantBits() > 40 ||
          R.getSignedMax().getSignificantBits() > 40)
        return std::nullopt;
      int64_t A = R.getSignedMin().getSExtValue() * Scale.getSExtValue();
      int64_t B = R.getSignedMax().getSExtValue() * Scale.getSExtValue();
      Lo += std::min(A, B);
      Hi += std::max(A, B);
      if (Hi - Lo > Limit)
        return std::nullopt;
    }
    return std::make_pair(Lo, Hi);
  }

  void call(const CallBase &CB, unsigned Arg, int64_t Lo, int64_t Hi, Effects &E) {
    if (auto *I = dyn_cast<IntrinsicInst>(&CB)) {
      switch (I->getIntrinsicID()) {
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::prefetch:
        return;
      default:
        break;
      }
      if (auto *M = dyn_cast<MemIntrinsic>(I)) {
        auto *Len = dyn_cast<ConstantInt>(M->getLength());
        if (!Len) {
          E.Unknown = true;
          return;
        }
        Range R{Lo, Hi + (int64_t)Len->getZExtValue()};
        (Arg == 0 ? E.Writes : E.Reads).push_back(R);
        return;
      }
      E.Unknown = true;
      return;
    }
    const Function *Callee = CB.getCalledFunction();
    if (Callee && Callee->doesNotReturn())
      return;
    if (!Callee || Callee->isDeclaration() || Arg >= Callee->arg_size()) {
      E.Unknown = true;
      return;
    }
    const Effects &Sub = of(*Callee, Arg);
    E.Unknown |= Sub.Unknown;
    for (const Range &R : Sub.Reads)
      E.Reads.push_back({R.Lo + Lo, R.Hi + Hi});
    for (const Range &R : Sub.Writes)
      E.Writes.push_back({R.Lo + Lo, R.Hi + Hi});
  }

  const Effects &of(const Function &F, unsigned Arg) {
    auto Key = std::make_pair(&F, Arg);
    if (auto It = Done.find(Key); It != Done.end())
      return It->second;
    Effects E;
    if (!Active.insert(Key).second) {
      // Recursion: give up rather than iterate to a fixpoint.
      static const Effects Anything{true, {}, {}};
      return Anything;
    }
    uses(F.getArg(Arg), 0, 0, E);
    Active.erase(Key);
    return Done[Key] = std::move(E);
  }
};

bool overlaps(const std::vector<Range> &Rs, int64_t Lo, int64_t Hi) {
  for (const Range &R : Rs)
    if (R.Lo < Hi && Lo < R.Hi)
      return true;
  return false;
}
} // namespace

// For each helper `Symbols[I]` and each global `J` (`Offsets[J]`, `Sizes[J]`
// bytes into env), whether the helper may read (`Reads[I * M + J]`) or write
// (`Writes[I * M + J]`) it through its pointer parameters. Helpers missing
// from the bitcode at `Path`, or whose effect is unknown, may do both.
// Returns 0, or 1 with `*Err` set (malloc'ed).
extern "C" int biara_helper_effects(const char *Path, const char *const *Symbols, size_t N,
                                    const int64_t *Offsets, const uint32_t *Sizes, size_t M,
                                    uint8_t *Reads, uint8_t *Writes, char **Err) {
  std::memset(Reads, 1, N * M);
  std::memset(Writes, 1, N * M);
  LLVMContext Cx;
  auto Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer) {
    *Err = strdup((std::string(Path) + ": " + Buffer.getError().message()).c_str());
    return 1;
  }
  auto Module = parseBitcodeFile(**Buffer, Cx);
  if (!Module) {
    *Err = strdup((std::string(Path) + ": " + toString(Module.takeError())).c_str());
    return 1;
  }
  Analysis A{(*Module)->getDataLayout(), {}, {}};
  for (size_t I = 0; I < N; ++I) {
    const Function *F = (*Module)->getFunction(Symbols[I]);
    if (!F || F->isDeclaration())
      continue;
    Effects E;
    for (const Argument &P : F->args())
      if (P.getType()->isPointerTy())
        E.merge(A.of(*F, P.getArgNo()), 0);
    if (E.Unknown)
      continue;
    for (size_t J = 0; J < M; ++J) {
      int64_t Lo = Offsets[J], Hi = Offsets[J] + Sizes[J];
      Reads[I * M + J] = overlaps(E.Reads, Lo, Hi);
      Writes[I * M + J] = overlaps(E.Writes, Lo, Hi);
    }
  }
  return 0;
}
