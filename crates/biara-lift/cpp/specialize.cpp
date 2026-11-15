// Helper specialization (see src/decompile.rs): QEMU's translator computes
// some guest state lazily through helpers that take the operation as an
// argument (x86 condition codes: `helper_cc_compute_all(dst, src1, src2,
// cc_op)`). Where the operation is known at translation time, TCG inlines the
// matching computation itself; where it only becomes constant after LLVM has
// carried it across blocks, the lifted code still calls the helper, and the C
// shows `helper_cc_compute_nz(x, 0xcd8, 27)` instead of `x == 0`.
//
// This pass inlines QEMU's own definition at the calls whose operation
// argument is constant, so the usual simplification folds it to the
// expression it stands for. Calls with a variable operation keep calling
// the declared helper; no helper definition is left in the module.

#include "llvm-c/Types.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

namespace {
// Calls of `F` whose argument `Selector` is a constant.
std::vector<CallBase *> constantCalls(Function &F, uint32_t Selector) {
  std::vector<CallBase *> Out;
  for (User *U : F.users())
    if (auto *CB = dyn_cast<CallBase>(U);
        CB && CB->getCalledFunction() == &F && Selector < CB->arg_size() &&
        isa<ConstantInt>(CB->getArgOperand(Selector)))
      Out.push_back(CB);
  return Out;
}

int fail(char **Err, const std::string &Message) {
  *Err = strdup(Message.c_str());
  return -1;
}
} // namespace

// Inline the helpers `Names[I]` from the bitcode at `Path` at their calls in
// `M` where argument `Selectors[I]` is constant. Returns the number of calls
// inlined, or -1 with `*Err` set (malloc'ed).
extern "C" long biara_specialize_helpers(LLVMModuleRef MRef, const char *Path,
                                         const char *const *Names, const uint32_t *Selectors,
                                         size_t N, char **Err) {
  Module &M = *unwrap(MRef);
  std::vector<std::pair<std::string, uint32_t>> Wanted;
  for (size_t I = 0; I < N; ++I)
    if (Function *F = M.getFunction(Names[I]);
        F && F->isDeclaration() && !constantCalls(*F, Selectors[I]).empty())
      Wanted.emplace_back(Names[I], Selectors[I]);
  if (Wanted.empty())
    return 0;

  auto Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return fail(Err, std::string(Path) + ": " + Buffer.getError().message());
  auto Helpers = getOwningLazyBitcodeModule(std::move(*Buffer), M.getContext());
  if (!Helpers)
    return fail(Err, std::string(Path) + ": " + toString(Helpers.takeError()));
  (*Helpers)->setTargetTriple(M.getTargetTriple());
  (*Helpers)->setDataLayout(M.getDataLayout());

  // Only the wanted definitions and what they use: link them into a module
  // that declares just those, then that module into `M`.
  auto Picked = std::make_unique<Module>("biara.helpers", M.getContext());
  Picked->setTargetTriple(M.getTargetTriple());
  Picked->setDataLayout(M.getDataLayout());
  for (auto &[Name, Selector] : Wanted)
    Picked->getOrInsertFunction(Name, M.getFunction(Name)->getFunctionType());
  if (Linker::linkModules(*Picked, std::move(*Helpers), Linker::Flags::LinkOnlyNeeded))
    return fail(Err, std::string(Path) + ": linking the helpers failed");
  std::set<Function *> Before;
  for (Function &F : M)
    Before.insert(&F);
  if (Linker::linkModules(M, std::move(Picked)))
    return fail(Err, "linking the helpers into the module failed");

  long Inlined = 0;
  for (auto &[Name, Selector] : Wanted) {
    Function *F = M.getFunction(Name);
    if (!F || F->isDeclaration())
      continue;
    for (CallBase *CB : constantCalls(*F, Selector)) {
      if (CB->getFunctionType() != F->getFunctionType())
        continue;
      InlineFunctionInfo Info;
      Inlined += InlineFunction(*CB, Info).isSuccess();
    }
  }
  // Back to declarations: the rest of the calls keep naming the helper, and
  // what only the definitions used goes.
  std::vector<Function *> Linked;
  for (Function &F : M)
    if (!Before.count(&F))
      Linked.push_back(&F);
  for (auto &[Name, Selector] : Wanted)
    if (Function *F = M.getFunction(Name))
      F->deleteBody();
  for (Function *F : Linked)
    F->deleteBody();
  for (Function *F : Linked)
    if (F->use_empty())
      F->eraseFromParent();
  return Inlined;
}
