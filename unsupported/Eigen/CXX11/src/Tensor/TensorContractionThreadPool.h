// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H

// evaluator for thread pool device
#ifdef EIGEN_USE_THREADS

#include "./InternalHeaderCheck.h"

namespace Eigen {
/*模板函数的声明,Indices 是一个类型参数，用于指定张量收缩操作中的索引序列。在张量收缩中，每个索引对应两个张量中的相同维度，这些维度将进行求和。Indices 指定了这些索引的顺序。
LeftArgType 是一个类型参数，用于指定张量收缩操作中的左侧张量类型。RightArgType 是一个类型参数，用于指定张量收缩操作中的右侧张量类型。
OutputKernelType 是一个类型参数，用于指定输出结果的类型。*/
template<typename Indices, typename LeftArgType, typename RightArgType, typename OutputKernelType>
/*结构体模板的声明,两个模板参数。前一个是类型参数，用于指定要执行的张量收缩操作的模板类型。ThreadPoolDevice用于指定计算设备的类型。指定了使用线程池来执行计算操作。
summary：这个结构体模板的作用是评估张量收缩操作，进行计算并生成输出张量。它包含一些成员函数，如 Eval()，用于执行张量收缩计算，以及一些成员变量，如 m_op 和 m_thread_pool，
分别存储要执行的张量收缩操作和用于执行计算的线程池设备。这个结构体模板的实现需要根据实际数据类型进行特化，以便支持不同的张量类型和计算设备类型。*/
struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType, OutputKernelType>, ThreadPoolDevice> :
/*公共继承语句，它继承了 TensorContractionEvaluatorBase 类的模板实例化版本作为其基类。类型参数是 TensorEvaluator，它表示使用线程池设备来评估张量收缩操作。
这个语句的作用是将 TensorEvaluator 类模板的实例化版本作为基类，以便使用基类中的某些成员函数和变量，从而简化代码实现。*/
    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType, OutputKernelType>, ThreadPoolDevice> > {

  typedef ThreadPoolDevice Device;

  typedef TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType, OutputKernelType>, Device> Self;
  typedef TensorContractionEvaluatorBase<Self> Base;

  typedef TensorContractionOp<Indices, LeftArgType, RightArgType, OutputKernelType> XprType;
  /*定义了一个类型别名 Scalar，它是 XprType 类型中 Scalar 成员的非 const 版本。
  Scalar 是一个定义在 TensorContractionOp 模板类中的类型别名，它表示张量的标量类型。这个类型别名的作用是方便地获取张量的标量类型，并简化代码实现。
  std::remove_const_t 是一个 C++ 标准库类型转换工具，它将类型的 const 限定符移除，以便得到非 const 版本的类型。*/
  typedef std::remove_const_t<typename XprType::Scalar> Scalar;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
//一个静态的、编译期常量的整数类型变量。它的值是通过对 TensorEvaluator<LeftArgType, Device>::Layout 进行求值得到的。
  static constexpr int Layout = TensorEvaluator<LeftArgType, Device>::Layout;

  // Most of the code is assuming that both input tensors are ColMajor. If the
  // inputs are RowMajor, we will "cheat" by swapping the LHS and RHS:
  // If we want to compute A * B = C, where A is LHS and B is RHS, the code
  // will pretend B is LHS and A is RHS.
  //定义一个类型别名 EvalLeftArgType，其类型是根据条件表达式 std::conditional_t 计算得到的。
  //static_cast<int>(Layout) == static_cast<int>(ColMajor) 判断了 Layout 是否等于 ColMajor，如果满足条件则 EvalLeftArgType 的类型为 LeftArgType，否则为 RightArgType。
  //std::conditional_t 的作用是在编译期间选择不同的类型，这在模板编程中非常有用。使用 typedef 关键字将 std::conditional_t 返回的类型定义为 EvalLeftArgType 的别名，方便后续代码的使用。
  typedef std::conditional_t<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType> EvalLeftArgType;
  typedef std::conditional_t<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType> EvalRightArgType;

//静态常量表达式，表示张量求值器的维度
  static constexpr int LDims =
  //通过 TensorEvaluator<EvalRightArgType, Device>::Dimensions 获取到的一个数组类型，然后使用 internal::array_size 获取到该数组的大小（即元素个数），最终将其作为一个编译期常量的整数类型变量的值。
      internal::array_size<typename TensorEvaluator<EvalLeftArgType, Device>::Dimensions>::value;
  static constexpr int RDims =
      internal::array_size<typename TensorEvaluator<EvalRightArgType, Device>::Dimensions>::value;
//静态常量表达式，表示张量求值器的收缩度      
  static constexpr int ContractDims = internal::array_size<Indices>::value;

  typedef array<Index, LDims> left_dim_mapper_t;
  typedef array<Index, RDims> right_dim_mapper_t;

  typedef array<Index, ContractDims> contract_t;
  typedef array<Index, LDims - ContractDims> left_nocontract_t;
  typedef array<Index, RDims - ContractDims> right_nocontract_t;

//表示两个张量进行矩阵乘法后的结果张量的维度数。
  static constexpr int NumDims = LDims + RDims - 2 * ContractDims;

  typedef DSizes<Index, NumDims> Dimensions;

  // typedefs needed in evalTo
  //定义三个类型别名，LhsScalar 是 EvalLeftArgType::Scalar 的去除 const 限定符后的类型，RhsScalar 是 EvalRightArgType::Scalar 的去除 const 限定符后的类型。
  //这两个类型表示左右两个张量的标量类型，用于后续计算时进行类型检查或者类型转换。
  typedef std::remove_const_t<typename EvalLeftArgType::Scalar> LhsScalar;
  typedef std::remove_const_t<typename EvalRightArgType::Scalar> RhsScalar;
  //Traits 是一个类型，表示使用 GEBP（Generalized Expression-Block Partitioning）算法进行矩阵乘法时所需要的类型特征。
  //internal::gebp_traits<LhsScalar, RhsScalar> 是一个模板元函数，用于根据左右两个张量的标量类型推导出对应的 GEBP 特征类型。
  typedef typename internal::gebp_traits<LhsScalar, RhsScalar> Traits;

//它们分别表示左右两个张量的求值器类型。
  typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
  typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;

//模板类 TensorEvaluator 的构造函数，接受两个参数。表示张量表达式的类型 XprType，表示计算设备的类型 Device，并将其传递给 Base 类的构造函数。
//Base 则是它的基类，首先将 op 和 device 分别传递给 Base 类的构造函数，以完成对基类的初始化。
  TensorEvaluator(const XprType& op, const Device& device) :
      Base(op, device) {}

//成员函数evalProduct，接受一个整数类型的模板参数 Alignment 和一个指向标量类型的指针 buffer。
//作用是计算张量表达式的乘积，并将结果存储到 buffer 中。在具体实现中，它调用了 evalProductImpl 函数，传递了 NoCallback 类型的对象作为回调函数。
  template <int Alignment>
  void evalProduct(Scalar* buffer) const {
    evalProductImpl<NoCallback, Alignment>(buffer, NoCallback());
  }

//接受两个参数：一个整数类型的模板参数 Alignment 和一个回调函数类型 EvalToCallback 的对象 done。
//作用是计算张量表达式的乘积，并将结果存储到buffer中。与上一函数不同的是，他在计算完成后调用指定的回调函数。
//在具体实现中，他也是调用的evalProductImpl 函数，传递了 EvalToCallback 类型的对象作为回调函数。
  template <typename EvalToCallback, int Alignment>
  void evalProductAsync(Scalar* buffer, EvalToCallback done) const {
    evalProductImpl<EvalToCallback, Alignment>(buffer, std::move(done));
  }

  template <typename DoneCallback, int Alignment>
  void evalProductImpl(Scalar* buffer, DoneCallback done) const {
    // This function computes a lot of heuristics in multiple steps, and it
    // also has multiple exit points. To keep it sane, readable and all in one
    // place, sync/async execution decision is made at runtime at the very end.
    //
    // (1) In sync mode we allocate Context on the stack, submit computations
    //     to the device thread pool, and block on a barrier until it is
    //     completed.
    //
    // (2) In async mode we allocate Context on the heap, and after all tasks
    //     are finished, we call provided the done callback, and delete a
    //     context from the heap.
    //
    // (*) EvalParallelContext & EvalShardedByInnerDimContext owns all the state
    // and temporary buffers, required for executing the tensor contraction.
    // They are responsible for cleaning it up after contraction is done.
    /*
    这个代码块是一个函数，作用是根据计算的一些启发式标准来决定同步还是异步执行张量表达式的计算。这个函数会根据不同的执行模式，采用不同的策略来完成计算，以保证计算的正确性和性能。
    具体来说，这个函数有多个步骤，并且有多个出口。为了保证代码的可读性和可维护性，它将同步/异步执行的决策推迟到了最后一步。
    如果选择同步模式，函数会在栈上分配一个 `Context` 对象，将计算任务提交到设备线程池，然后在屏障上阻塞等待计算完成。
    如果选择异步模式，函数会在堆上分配一个 `Context` 对象，并在所有任务完成后调用提供的回调函数，然后从堆上删除 `Context` 对象。
    `EvalParallelContext` 和 `EvalShardedByInnerDimContext` 拥有执行张量收缩所需的所有状态和临时缓冲区。它们负责在收缩完成后清理这些缓冲区。*/
    //定义了一个静态常量 IsEvalInSyncMode，它的值取决于模板参数 DoneCallback 是否为 NoCallback 类型。
    //如果是，则 IsEvalInSyncMode 为 true，表示当前执行模式为同步模式；否则为 false，表示当前执行模式为异步模式。
    static const bool IsEvalInSyncMode =
        std::is_same<DoneCallback, NoCallback>::value;

    //码定义了三个变量 m、n、k，分别表示张量表达式的三个维度的大小。
    const Index m = this->m_i_size;
    const Index n = this->m_j_size;
    const Index k = this->m_k_size;
    if (m == 0 || n == 0 || k == 0) return;

    // Compute a set of algorithm parameters:
    // - kernel block sizes (bm, bn, bk)
    // - task grain sizes (number of kernels executed per task: gm, gn)
    // - number of threads
    // - sharding by row/column
    // - parallel packing or first lhs then rhs
    // and some derived parameters:
    // - number of tasks (nm, nn, nk)
    // - number of kernels (nm0, nn0)
    // Unfortunately, all these parameters are tightly interdependent.
    // So in some cases we first compute approximate values, then compute other
    // values based on these approximations and then refine the approximations.

    // There are lots of heuristics here. There is some reasoning behind them,
    // but ultimately they are just tuned on contraction benchmarks for
    // different input configurations, thread counts and instruction sets.
    // So feel free to question any of them.

    // Compute whether we want to shard by row or by column.
    // This is a first approximation, it will be refined later. Since we don't
    // know number of threads yet we use 2, because what's we are most
    // interested in at this point is whether it makes sense to use
    // parallelization at all or not.
    /*
    这段代码的作用是计算一组算法参数，包括：
    内核块大小（bm、bn、bk）
    任务粒度（每个任务执行的内核数量：gm、gn）
    线程数
    行/列分片策略
    并行打包还是先左操作数再右操作数
    一些派生参数，如任务数（nm、nn、nk）和内核数（nm0、nn0）
    这些参数是相互依赖的，因此在某些情况下，我们需要先计算近似值，然后基于这些近似值计算其他参数，最后再对近似值进行细化。
    这个函数中有很多启发式标准和经验参数，它们是根据不同的输入配置、线程数和指令集进行调整的。因此，如果需要，可以对它们进行调整和优化。
    首先，代码计算了一组粗略的参数，包括行/列分片策略。由于此时线程数未知，因此使用了一个默认值 2，主要是为了判断是否应该使用并行化。
    其中，行/列分片策略会在后续的计算中再次进行调整和优化。*/
    //根据张量表达式的维度大小以及默认的线程数 2 来判断是否应该采用列分片策略。
    //具体来说，shardByCol 函数会根据维度大小和线程数来计算一个分片阈值，如果分片阈值小于维度大小，则采用列分片策略；否则采用行分片策略。
    bool shard_by_col = shardByCol(m, n, 2);

    // First approximation of kernel blocking sizes.
    // Again, we don't know number of threads yet, so we use 2.
    //根据之前计算得到的行/列分片策略，计算了内核块的大小 bm、bn、bk。
    //构造 internal::TensorContractionBlocking 对象时，需要指定张量表达式的维度大小和线程数。
    //然后，可以通过 mc()、nc() 和 kc() 方法分别获取内核块在第一、第二和第三维上的大小。
    Index bm, bn, bk;
    if (shard_by_col) {
      //采用列分片策略，则使用 internal::ShardByCol 策略进行计算，得到相应的内核块大小；
      internal::TensorContractionBlocking<Scalar, LhsScalar, RhsScalar, Index,
                                          internal::ShardByCol>
          blocking(k, m, n, 2);
      bm = blocking.mc();
      bn = blocking.nc();
      bk = blocking.kc();
    } else {
      internal::TensorContractionBlocking<Scalar, LhsScalar, RhsScalar, Index,
                                          internal::ShardByRow>
          blocking(k, m, n, 2);
      bm = blocking.mc();
      bn = blocking.nc();
      bk = blocking.kc();
    }

    // Compute optimal number of threads.
    // Note: we use bk instead of k here because we are interested in amount of
    // _parallelizable_ computations, and computations are not parallelizable
    // across k dimension.
    /*作用是计算最优线程数。这里使用 bk 而不是 k 来计算最优线程数，因为我们希望计算的是可并行计算的数量，而在 k 维度上的计算是不可并行的。
    计算最优线程数的方法是根据当前系统的 CPU 核心数量、内存带宽、缓存大小等一系列硬件参数，以及之前计算得到的内核块大小和其他参数，来估计最优线程数。
    根据最优线程数和其他参数来决定是否采用内部维度分片策略
    */
    //调用 contractionCost 函数来计算张量乘积的成本，包括计算时间、内存带宽等因素。
    const TensorOpCost cost =
        contractionCost(m, n, bm, bn, bk, shard_by_col, false);
    //调用 TensorCostModel<ThreadPoolDevice>::numThreads 函数来计算最优线程数，该函数考虑了硬件参数、内核块大小、分片策略等因素。
    int num_threads = TensorCostModel<ThreadPoolDevice>::numThreads(
        static_cast<double>(n) * m, cost, this->m_device.numThreads());
    //用 numThreadsInnerDim 函数来计算在内部维度上的最优线程数。这个函数根据内核块大小、内存带宽、缓存大小等因素来计算最优线程数，并返回一个整数值。
    int num_threads_by_k = numThreadsInnerDim(m, n, k);
    if (shardByInnerDim(m, n, k, num_threads, num_threads_by_k)) {
      // We are in the scenario where it is more effective to shard by the
      // inner dimension.
      //采用内部维度分片策略
      if (IsEvalInSyncMode) {
        //同步执行计算 创建上下文对象 ctx 并且调用run方法启动计算过程
        EvalShardedByInnerDimContext<DoneCallback> ctx(
            this, num_threads_by_k, buffer, m, n, k, std::move(done));
        ctx.template run<Alignment>();
      } else {
        //异步执行计算 创建上下文对象 ctx 并且调用runAsync方法启动计算过程
        //runAsync 方法会在一个新的线程中异步执行计算过程，以避免阻塞主线程。在异步执行完成后，会调用 done 回调函数来通知计算结果。
        auto* ctx = new EvalShardedByInnerDimContext<DoneCallback>(
            this, num_threads_by_k, buffer, m, n, k, std::move(done));
        ctx->template runAsync<Alignment>();
      }

      return;
    }

    // TODO(dvyukov): this is a stop-gap to prevent regressions while the cost
    // model is not tuned. Remove this when the cost model is tuned.
    /*这段代码表示一个临时措施，用于在成本模型不完善时防止出现性能退化。作者建议在成本模型优化之后删除这段代码。
    处理一些特殊情况，例如当张量乘积中的一个维度大小为 1 时，或者最优线程数为 1 时，直接采用顺序计算的方式来完成计算，而不是采用并行计算。
    这种特殊处理可以提高计算的效率和正确性，避免不必要的并行计算开销。*/
    //检查张量乘积的第二个维度大小 n 是否为 1，如果是，则直接将最优线程数设置为 1，因为在这种情况下，并行计算没有任何优势。
    if (n == 1) num_threads = 1;

    //检查最优线程数是否为 1，如果是，则调用 evalProductSequential 函数来采用顺序计算的方式完成计算。
    if (num_threads == 1) {
      TENSOR_CONTRACTION_DISPATCH(this->template evalProductSequential,
                                  Unaligned, (buffer));
      //如果采用异步计算方式，代码会调用 done 回调函数来通知计算结果。
      if (!IsEvalInSyncMode) done();
      return;
    }

    // Now that we know number of threads, recalculate sharding and blocking.
    shard_by_col = shardByCol(m, n, num_threads);
    if (shard_by_col) {
      internal::TensorContractionBlocking<Scalar, LhsScalar, RhsScalar, Index,
                                          internal::ShardByCol>
          blocking(k, m, n, num_threads);
      bm = blocking.mc();
      bn = blocking.nc();
      bk = blocking.kc();
    } else {
      internal::TensorContractionBlocking<Scalar, LhsScalar, RhsScalar, Index,
                                          internal::ShardByRow>
          blocking(k, m, n, num_threads);
      bm = blocking.mc();
      bn = blocking.nc();
      bk = blocking.kc();
    }

    // Number of kernels for each dimension.
    //每个维度的内核数
    Index nm0 = divup(m, bm);
    Index nn0 = divup(n, bn);
    Index nk = divup(k, bk);

    // Calculate task grain size (number of kernels executed per task).
    // This task size coarsening serves two purposes:
    // 1. It reduces per-task overheads including synchronization overheads.
    //它减少了每个任务的开销，包括同步开销。通过每个任务执行更多的内核，可以减少设置和管理每个单独任务的开销，从而更有效地使用计算资源。
    // 2. It allows to use caches better (reuse the same packed rhs in several
    // consecutive kernels).
    //允许更好地使用缓存。通过在几个连续的内核中重复使用相同的打包右手边（packed rhs）数据，可以将数据保留在缓存中，减少昂贵的内存访问的需求。
    //这可以在处理的数据很大且缓存大小有限的情况下带来显着的性能提升。
    Index gm = 1;
    Index gn = 1;
    // If we are sharding by column, then we prefer to reduce rows first.
    //如果按列进行分片，就优先减少行的数量。
    if (shard_by_col) {
      //gm 和 gn 的值会被重新计算，通过调用 coarsenM 和 coarsenN 函数来计算经过缩减后的行和列数。
      gm = coarsenM(m, n, bm, bn, bk, gn, num_threads, shard_by_col);
      gn = coarsenN(m, n, bm, bn, bk, gm, num_threads, shard_by_col);
    } else {
      //按行进行分片，则按照相反的顺序计算 gm 和 gn 的值。
      gn = coarsenN(m, n, bm, bn, bk, gm, num_threads, shard_by_col);
      gm = coarsenM(m, n, bm, bn, bk, gn, num_threads, shard_by_col);
    }
    // Number  of tasks in each dimension.
    Index nm = divup(nm0, gm);
    Index nn = divup(nn0, gn);

    // If there is enough concurrency in the sharding dimension, we choose not
    // to paralellize by the other dimension, and execute all kernels in sync
    // mode. This reduces parallelism from the nm x nn down to nn
    // (shard_by_col==true) or nm (shard_by_col==false).
    /*如果在分片维度上存在足够的并发性，那么我们选择不对另一个维度进行并行化，并以同步模式执行所有内核。
    这将并行性从 nm x nn 减少到 nn（当 shard_by_col==true 时）或 nm（当 shard_by_col==false 时）。
    */
    //用来表示在哪个维度进行分片
    const Index sharding_dim_tasks = shard_by_col ? nn : nm;
    //表示可以使用的线程数，其值为当前设备线程池中的线程数。
    const int num_worker_threads = this->m_device.numThreadsInPool();

    // With small number of threads we want to make sure that we do not reduce
    // parallelism too much. With large number of threads we trade maximum
    // parallelism for better memory locality.
    /*根据可用的线程数来调整并行度。当线程数较少时，我们希望确保不会过度减少并行度。因此，此时我们选择保持较高的并行度，以便更快地完成计算。
    当线程数较多时，我们则希望更好地利用内存局部性来提高性能。因此，此时我们可能会减少一些并行度，以便更好地利用缓存，从而减少内存访问的延迟。
    */
    const float oversharding_factor =
        num_worker_threads <= 4  ? 8.0 :
        num_worker_threads <= 8  ? 4.0 :
        num_worker_threads <= 16 ? 2.0 :
        num_worker_threads <= 32 ? 1.0 :
        num_worker_threads <= 64 ? 0.8 : /* num_worker_threads > 64 */ 0.6;
//代码根据 sharding_dim_tasks 和 oversharding_factor 的值来确定是否仅在分片维度上进行并行化。
//如果 sharding_dim_tasks 大于等于 oversharding_factor 乘以 num_worker_threads，则 parallelize_by_sharding_dim_only 的值为 true，表示仅在分片维度上进行并行化；
//否则其值为 false，表示需要在其他维度上进行并行化。
    const bool parallelize_by_sharding_dim_only =
        sharding_dim_tasks >= oversharding_factor * num_worker_threads;

    // Last by not least, decide whether we want to issue both lhs and rhs
    // packing in parallel; or issue lhs packing first, and then issue rhs
    // packing when lhs packing completes (for !shard_by_col lhs and rhs are
    // swapped). Parallel packing allows more parallelism (for both packing and
    // kernels), while sequential packing provides better locality (once
    // a thread finishes rhs packing it proceed to kernels with that rhs).
    // First, we are interested in parallel packing if there are few tasks.
    /*这段代码的作用是根据任务数量和可用的线程数来决定是否同时对 lhs 和 rhs 进行打包，并行化打包和内核执行。或者先对 lhs 进行打包，等 lhs 打包完成后再进行 rhs 的打包，并行化内核执行。
    其中，如果按行分片，则 lhs 和 rhs 分别表示原始矩阵的行和列，如果按列分片，则 lhs 和 rhs 分别表示原始矩阵的列和行。如果按列分片，则需要先对 lhs 进行打包，然后再对 rhs 进行打包。如果按行分片，则需要先对 rhs 进行打包，然后再对 lhs 进行打包。
    如果任务数量较少，则更倾向于并行化打包。这样可以提高并行性，同时也可以提高内核的并行性。如果任务数量较多，则更倾向于顺序打包。这样可以提高局部性，并且可以减少内存访问的延迟。在实际应用中，应该根据具体情况来选择合适的打包方式。*/
    bool parallel_pack = num_threads >= nm * nn;
    // Also do parallel packing if all data fits into L2$.
    //判断是否要进行并行打包。如果数据大小不超过 L2 缓存的大小，那么就可以使用并行打包。
    if (m * bk * Index(sizeof(LhsScalar)) + n * bk * Index(sizeof(RhsScalar)) <=
        l2CacheSize() * num_threads)
      parallel_pack = true;
    // But don't do it if we will use each rhs only once. Locality seems to be
    // more important in this case.
    //如果 rhs 只会被使用一次，那么局部性可能更为重要，此时不使用并行打包。
    if ((shard_by_col ? nm : nn) == 1) parallel_pack = false;
    // Also don't get in the way of parallelize_by_sharding_dim_only
    // optimization.
    //如果已经选择了仅在分片维度上进行并行化，则也不使用并行打包。
    if (parallelize_by_sharding_dim_only) parallel_pack = false;

    // TODO(ezhulnev): With if contexpr we don't need SyncEvalParallelContext.
    //根据是否处于同步模式来选择并行计算上下文。
    if (IsEvalInSyncMode) {
      //如果处于同步模式，则使用 SyncEvalParallelContext 来进行计算。代码使用 TENSOR_CONTRACTION_DISPATCH 宏来分派计算任务。
#define CONTEXT_ARGS                                                        \
  (this, num_threads, buffer, m, n, k, bm, bn, bk, nm, nn, nk, gm, gn, nm0, \
   nn0, shard_by_col, parallel_pack, parallelize_by_sharding_dim_only,      \
   NoCallback())                                                            \
      .run()
      TENSOR_CONTRACTION_DISPATCH(SyncEvalParallelContext, Alignment,
                                  CONTEXT_ARGS);
#undef CONTEXT_ARGS

    } else {
#define CONTEXT_ARGS                                                        \
  (this, num_threads, buffer, m, n, k, bm, bn, bk, nm, nn, nk, gm, gn, nm0, \
   nn0, shard_by_col, parallel_pack, parallelize_by_sharding_dim_only,      \
   std::move(done))
      TENSOR_CONTRACTION_ASYNC_DISPATCH(EvalParallelContext, DoneCallback,
                                        Alignment, CONTEXT_ARGS, run());
#undef CONTEXT_ARGS
    }
  }

  // ------------------------------------------------------------------------ //

  // Dummy struct to represent an empty DoneCallback.
//定义了一个名为 NoCallback 的结构体，用于表示一个空的回调函数。这个结构体没有任何成员变量，只定义了一个 operator() 函数。
//当这个回调函数被调用时，它会抛出一个异常，因为空回调函数不应该被调用。
//作用是在异步计算中，当计算完成后需要回调函数时，如果没有传递回调函数，就使用这个空回调函数来代替，以避免出现未定义行为。
  struct NoCallback {
    void operator()() {
      eigen_assert(false && "NoCallback should never be called");
    }
  };

  // ------------------------------------------------------------------------ //
//定义了一个名为 EvalParallelNotification 的模板类，表示并行计算完成后的通知器。
//模板参数是 DoneCallback 和 Context，其中 DoneCallback 表示计算完成后的回调函数，Context 表示并行计算的上下文。
  template <typename DoneCallback, typename Context>
  class EvalParallelNotification;

  // Synchronous evaluation notification that blocks caller thread in Wait().
  template <typename Context>
  class EvalParallelNotification<NoCallback, Context> {
   public:
   //第一个特化版本是 EvalParallelNotification<NoCallback, Context>，表示不需要回调函数的情况。
    EvalParallelNotification(Context*, NoCallback) {}
    void Notify() { done_.Notify(); }
    void Wait() { done_.Wait(); }
   private:
   //构造函数中，创建了一个 Eigen::Notification 对象 done_，用于通知计算完成。
    Eigen::Notification done_;
  };

  // Asynchronous evaluation notification that does not block in Wait().
  //第二个特化版本是 EvalParallelNotification<DoneCallback, Context>，表示需要回调函数的情况。
  template <typename DoneCallback, typename Context>
  class EvalParallelNotification {
   public:
    EvalParallelNotification(Context* ctx, DoneCallback done)
        : ctx_(ctx), done_(std::move(done)) {}

    void Notify() {
      // Make a copy of done callback, because it will be destructed when we
      // will delete context in the next line (EvalParallelNotification is a
      // data member of EvalParallelContext class).
      DoneCallback done_copy = std::move(done_);

      // Delete parallel evaluation context.
      delete ctx_;

      // Now safely call the done callback.
      done_copy();
    }

    void Wait() {}

   private:
    Context* ctx_;
    DoneCallback done_;
  };
/*解释了上下文对象 Context 的作用。Context 对象是用来协调同步/异步并行收缩运算的。当以异步模式执行时，它拥有可能被块打包和核心任务访问的所有共享状态。
Context 对象管理并行计算过程中的所有状态和资源，包括块状矩阵的分配和释放、线程池的管理、任务调度和同步等待等。
在同步模式下，Context 对象的作用相对较小，因为所有计算都是在调用上下文对象的 operator() 函数时立即执行的，不需要管理任何状态或资源。
但在异步模式下，Context 对象的作用就非常重要了，因为它需要管理并发执行的多个任务，保证它们能够正确地访问和修改共享状态，以及避免出现竞态条件和死锁等并发问题。*/
  // Context orchestrates sync/async parallel contraction evaluation. When it is
  // executed in asynchronous mode, it owns all the shared state that might be
  // accessible by block packing and kernel tasks.

/*这段代码定义了一个名为 EvalParallelContext 的模板类，表示异步并行计算的上下文。这个模板类有五个模板参数：
DoneCallback：表示计算完成后的回调函数类型，如果不需要回调函数，则传递 NoCallback 类型。
lhs_inner_dim_contiguous：表示左矩阵的内部维度是否连续。
rhs_inner_dim_contiguous：表示右矩阵的内部维度是否连续。
rhs_inner_dim_reordered：表示右矩阵的内部维度是否被重新排序。
Alignment：表示数据在内存中的对齐方式，通常是 16 字节或 32 字节对齐。

这个模板类的作用是在异步模式下管理并行计算的状态和资源，包括：
分配和释放块状矩阵的内存。
管理线程池，调度和执行块打包和核心任务。
跟踪计算进度，等待所有任务完成。
管理共享状态的访问和修改，避免竞态条件和死锁等并发问题。
如果需要，处理计算完成后的回调函数。
这个模板类的实现非常复杂，涉及到许多高级的并发编程技术，例如锁、条件变量、原子操作、任务队列等。它的设计和实现需要考虑许多因素，例如计算规模、硬件特性、内存布局、线程调度、并发竞争等等。
因此，EvalParallelContext 是 Eigen 库中实现高性能矩阵收缩运算的核心组件之一，也是并行编程中的一个典型案例。*/
  template <typename DoneCallback, bool lhs_inner_dim_contiguous,
            bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered,
            int Alignment>
  class EvalParallelContext {
   public:
   //表示左矩阵和右矩阵的映射器类型，这些映射器用于将块状子矩阵打包成连续的内存块，以便于计算。这些映射器的模板参数包括
   //矩阵元素类型、索引类型、矩阵类型、矩阵评估器类型、不参与收缩的维度集合类型、参与收缩的维度集合类型、数据包大小、内部维度是否连续、内部维度是否被重新排序、内存对齐方式等。
    typedef internal::TensorContractionInputMapper<
        LhsScalar, Index, internal::Lhs, LeftEvaluator, left_nocontract_t,
        contract_t, internal::packet_traits<LhsScalar>::size,
        lhs_inner_dim_contiguous, false, Unaligned>
        LhsMapper;
    typedef internal::TensorContractionInputMapper<
        RhsScalar, Index, internal::Rhs, RightEvaluator, right_nocontract_t,
        contract_t, internal::packet_traits<RhsScalar>::size,
        rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Unaligned>
        RhsMapper;
//OutputMapper 表示输出矩阵的映射器类型。这个映射器用于将计算结果存储到输出矩阵中。它的模板参数包括矩阵元素类型、索引类型、矩阵存储顺序等。
    typedef internal::blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;
//TensorContractionKernel 表示张量收缩运算的核心计算单元类型。这个计算单元封装了一系列计算操作，包括块状子矩阵的打包、内积计算、累加等。
//它的模板参数包括矩阵元素类型、索引类型、输出矩阵映射器类型、左矩阵映射器类型、右矩阵映射器类型等。
    typedef internal::TensorContractionKernel<
        Scalar, LhsScalar, RhsScalar, Index, OutputMapper, LhsMapper, RhsMapper>
        TensorContractionKernel;
//表示块状子矩阵、块内存句柄和块内存分配器类型。这些类型用于表示块状子矩阵在内存中的存储方式和管理方式。
    typedef typename TensorContractionKernel::LhsBlock LhsBlock;
    typedef typename TensorContractionKernel::RhsBlock RhsBlock;
    typedef typename TensorContractionKernel::BlockMemHandle BlockMemHandle;

    EvalParallelContext(const Self* self, int num_threads, Scalar* buffer,
                        Index tm, Index tn, Index tk, Index bm, Index bn,
                        Index bk, Index nm, Index nn, Index nk, Index gm,
                        Index gn, Index nm0, Index nn0, bool shard_by_col,
                        bool parallel_pack,
                        bool parallelize_by_sharding_dim_only,
                        DoneCallback done)
        : created_by_thread_id_(std::this_thread::get_id()),//表示创建该对象的线程 ID。
          done_(this, std::move(done)),// 表示异步计算完成后的回调函数，用于通知调用者计算结果已经就绪。
          device_(self->m_device),//表示计算设备的类型，包括 CPU 和各种加速卡。
          lhs_(self->m_leftImpl, self->m_left_nocontract_strides,
               self->m_i_strides, self->m_left_contracting_strides,
               self->m_k_strides),
          rhs_(self->m_rightImpl, self->m_right_nocontract_strides,
               self->m_j_strides, self->m_right_contracting_strides,
               self->m_k_strides),//表示左矩阵和右矩阵的映射器对象。它们用于将矩阵分块、打包成连续的内存块，并在计算时调用 BLAS 库进行内积计算等。
          buffer_(buffer),//存储计算过程中的中间结果的缓冲区。
          output_(buffer, tm),//输出矩阵的映射器对象，用于将计算结果存储到输出矩阵中。
          output_kernel_(self->m_output_kernel),//输出矩阵计算的核心计算单元对象。
          tensor_contraction_params_(self->m_tensor_contraction_params),//张量收缩的参数对象，包括矩阵形状、收缩维度等信息。
          num_threads_(num_threads),//计算时使用的线程数。
          shard_by_col_(shard_by_col),//按列划分子矩阵
          parallel_pack_(parallel_pack),//并行打包子矩阵。
          parallelize_by_sharding_dim_only_(parallelize_by_sharding_dim_only),//是否只在划分维度上并行计算。
          m_(tm),
          n_(tn),
          k_(tk),//输出矩阵的行数、列数和内积的长度。
          bm_(bm),
          bn_(bn),
          bk_(bk),//左矩阵和右矩阵分块后的子矩阵大小。
          nm_(nm),
          nn_(nn),
          nk_(nk),//左矩阵、右矩阵和输出矩阵的内部维度大小。
          gm_(gm),
          gn_(gn),//左矩阵和右矩阵的总大小。
          nm0_(nm0),
          nn0_(nn0),//左矩阵和右矩阵的第一个内部维度大小。
          kernel_(m_, k_, n_, bm_, bk_, bn_),//计算核心对象，用于控制计算过程中的各种参数和状态。
          num_thread_local_allocations_(0),//当前线程的本地内存分配次数。
          // We reserve 2X more capacity for a thread local values, than the
          // number of threads in the pool to efficiently handle task stealing
          // by threads that are not managed by the pool.
          /*解释了为什么在构造函数中为每个线程本地存储分配了足够多的容量。
          为了有效处理由于任务窃取而可能导致的线程间竞争，每个线程本地存储的容量被设置为当前线程池中线程数的两倍。
          这样，即使有一些线程不受线程池管理并且可能窃取任务，也能保证每个线程都有足够的本地存储容量，以避免线程间竞争和同步开销。*/
          thread_local_capacity(2 * (parallelize_by_sharding_dim_only_
                                         ? device_.numThreadsInPool()
                                         : 0)),
          // We will use only one of the Lhs/Rhs thread local storage depending
          // on the shard_by_col value and we parallelize by sharding dim ONLY.
          //根据 shard_by_col 的值仅使用左矩阵或右矩阵的本地存储。shard_by_col 为 true，则使用右矩阵的本地存储，否则使用左矩阵的本地存储。
          //因为在按列划分子矩阵时，右矩阵的子矩阵是连续的内存块，而左矩阵的子矩阵则不是，因此使用右矩阵的本地存储更为高效。
          lhs_thread_local_blocks_(shard_by_col_ ? 0 : thread_local_capacity,
                                   {*this}, {*this}),
          rhs_thread_local_blocks_(shard_by_col_ ? thread_local_capacity : 0,
                                   {*this}, {*this}) {
      // These two options are mutually exclusive.
      // parallel_pack 和 parallelize_by_sharding_dim_only 两个选项是互斥的。
      //如果启用了 parallel_pack，则表示在打包操作中使用并行化，而如果启用了 parallelize_by_sharding_dim_only，则表示只在划分维度上并行化计算。
      eigen_assert(!(parallel_pack && parallelize_by_sharding_dim_only));

      //在进行多线程并行计算的初始化过程中。对于每个 k slice，计算状态需要进行初始化
      for (Index x = 0; x < P; x++) {
        // Normal number of notifications for k slice switch is
        // nm_ + nn_ + nm_ * nn_. However, fir
        st P - 1 slices will receive only
        // nm_ + nn_ notifications, because they will not receive notifications
        // from preceding kernels.
        //每个线程需要发送的通知数量。
        state_switch_[x] =
            x == 0
                ? 1// 第一个 k slice 只需要发送 nm_ + nn_ 条通知
                : (parallel_pack_ ? nn_ + nm_ : (shard_by_col_ ? nn_ : nm_)) +
                      (x == P - 1 ? nm_ * nn_ : 0);// 其他 k slice 需要发送 nm_ + nn_ + nm_ * nn_ 条通知
        //每个线程需要等待的打包操作数量。
        state_packing_ready_[x] =
            parallel_pack_ ? 0 : (shard_by_col_ ? nm_ : nn_);
         // 为每个线程的计算状态 state_kernel_ 分配内存空间
        state_kernel_[x] = new std::atomic<uint8_t>*[nm_];//每个线程需要等待的计算操作数量。
        for (Index m = 0; m < nm_; m++) {
          state_kernel_[x][m] = new std::atomic<uint8_t>[nn_];
          // Kernels generally receive 3 notifications (previous kernel + 2
          // packing), but the first slice won't get notifications from previous
          // kernels.
           // 初始化每个线程需要等待的计算操作数量
          for (Index n = 0; n < nn_; n++)
            state_kernel_[x][m][n].store(
                (x == 0 ? 0 : 1) + (parallel_pack_ ? 2 : 1),
                std::memory_order_relaxed);
                // 如果是第一个 k slice，则初始值为 0，否则为 1
                // 如果启用了 parallel_pack_，则需要额外增加 2 个计算操作，否则只需要增加 1 个计算操作
                // 使用 std::memory_order_relaxed 内存模型进行存储，保证对其它线程的修改可见性
        }
      }

      // Allocate memory for packed rhs/lhs matrices.
      //进行打包操作前，为打包后的左右矩阵分配内存空间。
      packed_mem_ = kernel_.allocateSlices(            //
          device_,                                     //
          /*num_lhs=*/nm0_,                            //左矩阵的维度大小。
          /*num_rhs=*/nn0_,                            //右矩阵的维度大小。
          /*num_slices=*/std::min<Index>(nk_, P - 1),  //需要打包的 k slice 数量，同时限制打包的最大数量为 P - 1
          packed_lhs_, packed_rhs_);//指向打包后的左右矩阵的指针。

      //初始化线程本地内存的相关参数
      if (parallelize_by_sharding_dim_only_) {
        //只对一维进行分片并行化。这时可以使用线程本地内存来缓存打包后的矩阵块，从而避免多次分配内存的开销，提高性能。
        const int num_worker_threads = device_.numThreadsInPool();
        /*如果按列进行分片，则需要为每一列分配一个 can_use_thread_local_packed_ 的原子变量，并将其初始值设为 true。
        然后，计算需要的块数 num_blocks，并调用 kernel_.allocateSlices() 方法为线程本地内存分配内存空间。
        此时，num_lhs 参数为 0，num_rhs 参数为 num_blocks，num_slices 参数为 1，
        lhs_blocks 参数为 nullptr，rhs_thread_local_pre_allocated_ 参数为线程本地内存的指针。*/
        if (shard_by_col) {
          can_use_thread_local_packed_ = new std::atomic<bool>[nn_]; // 为每一列分配一个原子变量，并将其初始值设为 true
          for (int i = 0; i < nn_; ++i)
            can_use_thread_local_packed_[i].store(true,
                                                  std::memory_order_relaxed);

          Index num_blocks = num_worker_threads * gn_;// 计算需要的块数 num_blocks
          thread_local_pre_alocated_mem_ = kernel_.allocateSlices(  //线程本地内存的指针。
              device_,                                              //
              /*num_lhs=*/0,                                        //
              /*num_rhs=*/num_blocks,                               //
              /*num_slices=*/1,                                     //
              /*lhs_blocks=*/nullptr, &rhs_thread_local_pre_allocated_);
              /*若打包操作按列进行，则记录打包后的左矩阵和右矩阵分别在线程本地内存中的偏移量，
              否则分别记录打包后的左矩阵和右矩阵分别在线程本地内存中的偏移量。*/

        } else {
          can_use_thread_local_packed_ = new std::atomic<bool>[nm_];
          for (int i = 0; i < nm_; ++i)
            can_use_thread_local_packed_[i].store(true,
                                                  std::memory_order_relaxed);

          Index num_blocks = num_worker_threads * gm_;
          thread_local_pre_alocated_mem_ = kernel_.allocateSlices(  //
              device_,                                              //
              /*num_lhs=*/num_blocks,                               //
              /*num_rhs=*/0,                                        //
              /*num_slices=*/1, &lhs_thread_local_pre_allocated_,   //
              /*rhs_blocks=*/nullptr);
        }
      }
    }

    ~EvalParallelContext() {
       // 释放 state_kernel_ 数组中的内存空间
      for (Index x = 0; x < P; x++) {
        for (Index m = 0; m < nm_; m++) delete[] state_kernel_[x][m];
        delete[] state_kernel_[x];
      }
      // 使用 kernel_.deallocate() 方法释放 packed_mem_ 数组中的内存空间
      kernel_.deallocate(device_, packed_mem_);
      if (parallelize_by_sharding_dim_only_) {
        //如果只按照某个维度进行分片并行化，则需要释放线程本地内存和 can_use_thread_local_packed_ 数组的内存空间
        kernel_.deallocate(device_, thread_local_pre_alocated_mem_);
        delete[] can_use_thread_local_packed_;
      }
    }

    void run() {
      // Kick off packing of the first slice.
      //开始执行并行计算。
      signal_switch(0, 1);//表示启动对第 0 个切片的打包操作，并将信号切换到下一个状态。

      // Wait for overall completion.等待整体完成。
      //
      // If parallel evaluation is executed in async mode, this is a no-op, and
      // Wait() will return immediately. In synchronous mode it will block the
      // caller thread until it will receive notification from last task.
      //如果是异步模式下的并行计算，该方法是一个空操作，会立即返回。如果是同步模式下的并行计算，则该方法会阻塞当前线程，直到接收到所有任务完成的通知。
      //
      // In async mode, last task when completed will call done callback from
      // the same thread, and will delete this context.
      //在异步模式下，最后一个任务完成后会从同一线程调用 done 回调函数，并删除该上下文对象。
      //
      // TODO(dvyukov): This wait can lead to deadlock if contraction is
      // evaluated in synchronous mode. If nthreads contractions are
      // concurrently submitted from worker threads, this wait will block all
      // worker threads and the system will deadlock.
      //从工作线程并发地提交了 nthreads 个收缩操作，那么该等待操作将会阻塞所有工作线程，从而导致系统死锁。
      //因此，需要注意在同步模式下并发提交任务的数量不要过多，以避免这种情况的发生。
      done_.Wait();
    }

   private:
   //定义了一个 std::thread::id 类型的成员变量 created_by_thread_id_
    std::thread::id created_by_thread_id_;

    // This notification is specialized on the type of DoneCallback and can be
    // blocking or non-blocking.
    EvalParallelNotification<DoneCallback, EvalParallelContext> done_;

    const Device& device_;
    LhsMapper lhs_;
    RhsMapper rhs_;
    Scalar* const buffer_;
    OutputMapper output_;
    OutputKernelType output_kernel_;
    TensorContractionParams tensor_contraction_params_;
    const int num_threads_;
    const bool shard_by_col_;
    const bool parallel_pack_;
    const bool parallelize_by_sharding_dim_only_;
    // Matrix sizes.
    const Index m_;
    const Index n_;
    const Index k_;
    // Block sizes.
    const Index bm_;
    const Index bn_;
    const Index bk_;
    // Number of tasks.
    const Index nm_;
    const Index nn_;
    const Index nk_;
    // Task grain sizes (number of kernels executed per task).
    const Index gm_;
    const Index gn_;
    // Number of blocks (this is different from ni_/nn_ because of task size
    // coarsening).
    const Index nm0_;
    const Index nn0_;
    // Tensor contraction kernel.
    TensorContractionKernel kernel_;

    // Parallelization strategy.
    //
    // Blocks related to the same k block can run in parallel because they write
    // to different output blocks. So we parallelize within k slices, this
    // gives us parallelism level of m x n. Before we can start any kernels
    // related to k-th slice, we need to issue m lhs packing tasks and n rhs
    // packing tasks.
    /*同一 k 块相关的块可以并行运行，因为它们写入不同的输出块。因此，我们在每个 k 切片内并行化，这为我们提供了 m x n 的并行级别。
    在我们开始执行与第 k 切片相关的任何内核之前，我们需要发出 m 个 LHS 打包任务和 n 个 RHS 打包任务。*/
    //
    // However, there is a bottleneck when we are finishing kernels for k-th
    // slice (at the very end there is only 1 runnable kernel). To mitigate this
    // bottleneck we allow kernels from k-th and k+1-th slices to run in
    // parallel. Note that (m, n, k) and (m, n, k+1) kernels write to the same
    // output block, so they must not run in parallel.
    /*然而，当我们完成第 k 切片的内核时，存在瓶颈（最后只有一个可运行的内核）。为了缓解这个瓶颈，我们允许第 k 切片和第 k+1 切片的内核并行运行。
    请注意，(m, n, k) 和 (m, n, k+1) 内核写入相同的输出块，因此它们不能并行运行。*/
    //
    // This gives us the following dependency graph.
    // On each k slice we have m x n kernel tasks, m lhs paking tasks and n rhs
    // packing tasks.
    // Kernel (m, n, k) can start when:
    //  - kernel (m, n, k-1) has finished
    //  - lhs packing (m, k) has finished
    //  - rhs packing (n, k) has finished
    // Lhs/rhs packing can start when:
    //  - all k-1 packing has finished (artificially imposed to limit amount of
    //  parallel packing)
    //
    // On top of that we limit runnable tasks to two consecutive k slices.
    // This is done to limit amount of memory we need for packed lhs/rhs
    // (for each k slice we need m*bk + n*bk memory in packed_lhs_/packed_rhs_).
    /*此外，我们将可运行的任务限制在连续的两个 k 切片中。这样做是为了限制我们需要打包的左手边矩阵和
    右手边矩阵的内存使用量（对于每个 k 切片，我们需要 mbk + nbk 的内存用于 packed_lhs_/packed_rhs_）。*/
    //
    // state_switch_ tracks when we are ready to switch to the next k slice.
    // state_kernel_[m][n] tracks when we are ready to kick off kernel (m, n).
    // These variable are rolling over 3 consecutive k slices: first two we are
    // actively executing + one to track completion of kernels in the second
    // slice.
    /*state_switch_ 变量跟踪我们何时准备切换到下一个 k 切片。state_kernel_[m][n] 变量跟踪我们何时准备启动内核 (m, n)。
    这些变量在连续的三个 k 切片中循环滚动：前两个 k 切片我们正在主动执行，加上一个用于跟踪第二个切片中内核完成的状态。*/
    static constexpr Index P = 3;

    // Handle to the allocated temporary storage for Lhs/Rhs blocks.
    //指向已分配的 Lhs/Rhs 块临时存储的句柄。
    BlockMemHandle packed_mem_;
    //这是一个长度为 P-1 的 vector 数组，用于存储分块后的左矩阵块。每个 vector 存储一个 k 切片中的所有块。
    std::vector<LhsBlock> packed_lhs_[P - 1];
    std::vector<RhsBlock> packed_rhs_[P - 1];

    // If we choose to parallelize only by the sharding dimension, each thread
    // will have it's own "thead local" (not a c++ thread local storage) memory
    // for packed_lhs or packed_rhs (shard_by_col = false of true). This memory
    // can't be passed to a kernel that might execute on a different thread.
    /*如果我们选择仅按分片维度进行并行化，则每个线程将具有自己的“线程本地”（不是 C++ 线程本地存储）内存，
    用于 packed_lhs 或 packed_rhs（当 shard_by_col = false 或 true 时）。这些内存不能传递给可能在不同线程上执行的内核。*/
    //
    // In practice when we are ready to pack memory for the sharding dimension
    // (rhs if shard_by_col==true) of the K-th slice, all kernels for K-1 slice
    // already computed (99% of the time), and we can pack data into the thread
    // local storage, and guarantee that all the kernels will be executed
    // immediately in the same thread. This significantly increases L1 cache hit
    // ratio and reduces pressure on the memory bus.
    /*实际上，当我们准备为第 K 切片的分片维度（如果 shard_by_col==true，则为 rhs）打包内存时，所有第 K-1 切片的内核已经计算完成（99% 的情况下），
    我们可以将数据打包到线程本地存储中，并保证所有内核将在同一线程中立即执行。这显著提高了 L1 缓存命中率，并减少了内存总线的压力。*/
    //
    // It's still possible that kernel for the K-th slice will be ready before
    // completion of the K-1 kernel, so we have to allocate "global" packed_lhs_
    // and packed_rhs_ to allow kernels to be executed later on a thread
    // different from the thread that was used for packing.
    /*仍然可能出现第 K 切片的内核在第 K-1 内核完成之前就已准备好的情况，因此我们必须分配“全局”packed_lhs_ 和 packed_rhs_，
    以允许内核在稍后在不同于用于打包的线程上执行。*/

    // Handle for pre-allocated thread local memory buffers.
    //预分配线程本地存储缓冲区的句柄。
    BlockMemHandle thread_local_pre_alocated_mem_;

    // Only one of these will be initialized depending on shard_by_col value
    // (the size will be `num_worker_threads * num_grains_in_the_sharding_dim`).
    std::vector<LhsBlock> lhs_thread_local_pre_allocated_;
    std::vector<RhsBlock> rhs_thread_local_pre_allocated_;

    // How many thread local blocks were already allocated.
    std::atomic<int> num_thread_local_allocations_;//跟踪已经分配的线程本地存储的数量，std::atomic 类型可以确保并发线程安全地访问该变量，避免出现数据竞争。
    const int thread_local_capacity;//表示线程本地存储的容量。是每个线程独有的存储空间，可以用于存储线程私有的数据，避免线程间的数据竞争。

    // We will use pre-allocated Lhs/Rhs blocks defined above, if the number of
    // unique threads in a system is below or equal to the number of threads in
    // a thread pool. We will fallback on dynamic memory allocation after that.
    /*如果系统中的唯一线程数小于或等于线程池中的线程数，则我们将使用上面定义的预分配的 Lhs/Rhs 块。在此之后，我们将回退到动态内存分配。*/

    // ThreadLocalBlocks is a container for Lhs or Rhs thread local buffers. Its
    // size is equal to the grain size in Lhs/Rhs sharding dimension.
    /*ThreadLocalBlocks 是用于存储 Lhs 或 Rhs 线程本地缓冲区的容器。其大小等于 Lhs/Rhs 分片维度中的粒度大小。*/
    template <typename BlockType>
    class ThreadLocalBlocks {
     public:
      ThreadLocalBlocks() = default;

      ThreadLocalBlocks(BlockType* base, size_t grain_size)//预分配的内存创建 ThreadLocalBlocks 对象。
          : is_pre_allocated_(true),
            thread_local_pre_allocated_base_(base),
            grain_size_(grain_size) {}

      ThreadLocalBlocks(BlockMemHandle mem_handle,
                        std::vector<BlockType> blocks)//使用动态内存分配创建 ThreadLocalBlocks 对象。
          : is_pre_allocated_(false),
            mem_handle_(std::move(mem_handle)),
            blocks_(std::move(blocks)) {}

      BlockType& block(int grain_index) {//返回指定索引处的块。
        eigen_assert(grain_index >= 0);
        eigen_assert(static_cast<size_t>(grain_index) < size());
        return is_pre_allocated_ ? thread_local_pre_allocated_base_[grain_index]
                                 : blocks_[grain_index];
      }

      void Release(EvalParallelContext& ctx) const {//释放内存
        if (!is_pre_allocated_) {
          ctx.kernel_.deallocate(ctx.device_, mem_handle_);
        }
      }

      size_t size() const {//返回 ThreadLocalBlocks 对象中存储块的数量。
        return is_pre_allocated_ ? grain_size_ : blocks_.size();
      }

     private:
      bool is_pre_allocated_;//用于指示是否使用预分配的内存。

      // Reuse pre-allocated thread local buffers.
      BlockType* thread_local_pre_allocated_base_ = nullptr;//指针，指向预分配的内存的起始位置。
      size_t grain_size_ = 0;//预分配的内存中包含的块的数量。

      // These will be initialized only if `is_pre_allocated == false`.
      BlockMemHandle mem_handle_{};//用于动态内存分配的成员变量。
      std::vector<BlockType> blocks_;//存储动态分配的块的数据。
    };

    // ThreadLocalBlocksInitialize callable does custom thread local blocks
    // initialization, and will reuse pre-allocated buffers if possible, or will
    // dynamically allocate new memory.
    /*ThreadLocalBlocksInitialize 可调用对象进行自定义线程本地块的初始化，如果可能会重用预分配的缓冲区，否则将动态分配新的内存。*/
    //
    // Lhs/Rhs blocks might be of the same type, so we have to pass explicitly
    // for what side do we plan to do block allocation.
    /*Lhs/Rhs 块可能是相同类型的，因此我们必须明确地传递要进行块分配的是哪一侧。*/
    //在线程池中对LhsBlock和RhsBlock进行初始化，并根据需要从线程本地块分配器（ThreadLocalBlocksAllocator）中分配内存或重用现有块。
    template <typename BlockType, bool is_rhs>
    class ThreadLocalBlocksInitialize {
      // 判断当前Block是LhsBlock还是RhsBlock，以便后续调用
      static constexpr bool kIsLhs =
          !is_rhs && std::is_same<BlockType, LhsBlock>::value;
      static const bool kIsRhs =
          is_rhs && std::is_same<BlockType, RhsBlock>::value;
      static_assert(kIsLhs || kIsRhs, "Unknown block type");
      // 定义一个别名，方便后续使用
      using Blocks = ThreadLocalBlocks<BlockType>;

     public:
      // 构造函数，传入EvalParallelContext参数，初始化ctx_和num_worker_threads_
      ThreadLocalBlocksInitialize(EvalParallelContext& ctx)
          : ctx_(ctx),
            num_worker_threads_(ctx_.device_.numThreadsInPool()) {}
      // 重载括号运算符，进行Block的初始化
      void operator()(Blocks& blocks) {
        // 获取当前已分配的线程数
        const int n = ctx_.num_thread_local_allocations_.fetch_add(
            1, std::memory_order_relaxed);
        // 如果当前已分配的线程数大于等于总线程数，则需要重新分配空间并初始化
        if (n >= num_worker_threads_) {
          ThreadLocalBlocksAllocator<is_rhs>::allocate(ctx_, blocks);
        } else { // 否则直接复用之前已分配的空间
          ThreadLocalBlocksAllocator<is_rhs>::reuse(ctx_, n, blocks);
        }
      }

     private:
      // NOTE(ezhulenev): Without 'if constexpr' we have to put calls to
      // TensorContractionKernel::allocateSlices into template specializations.
      // Also explicit specializations are not allowed at class scope in C++03,
      // EvalCtx type parameter is just a workaround for that limitation.
      // 定义一个嵌套结构体，用于为Block分配内存并初始化
      // 'if constexpr'语句的作用是根据模板参数是否为true，决定选择哪一个模板参数进行编译
      template <bool pack_rhs, typename EvalCtx = EvalParallelContext>
      struct ThreadLocalBlocksAllocator;
      // 如果是RhsBlock，则分配rhs_blocks并初始化，否则分配lhs_blocks并初始化
      template <typename EvalCtx>
      struct ThreadLocalBlocksAllocator</*pack_rhs=*/true, EvalCtx> {
        static void allocate(EvalCtx& ctx, Blocks& blocks) {
          std::vector<RhsBlock> rhs_blocks;
          BlockMemHandle mem_handle = ctx.kernel_.allocateSlices(
              ctx.device_,
              /*num_lhs=*/0,
              /*num_rhs=*/ctx.gn_,
              /*num_slices=*/1,
              /*lhs_blocks=*/nullptr, /*rhs_blocks=*/&rhs_blocks);

          blocks = ThreadLocalBlocks<RhsBlock>(std::move(mem_handle),
                                               std::move(rhs_blocks));
        }
      //重用之前已分配的 RhsBlock 块时，会调用 reuse 函数，该函数将指向之前已分配的内存的指针赋值给 blocks。
        static void reuse(EvalCtx& ctx, int index, Blocks& blocks) {
          RhsBlock* ptr = &ctx.rhs_thread_local_pre_allocated_[ctx.gn_ * index];
          blocks = ThreadLocalBlocks<RhsBlock>(ptr, ctx.gn_);
        }
      };

      template <typename EvalCtx>
      struct ThreadLocalBlocksAllocator</*pack_rhs=*/false, EvalCtx> {
        static void allocate(EvalCtx& ctx, Blocks& blocks) {
          std::vector<LhsBlock> lhs_blocks;
          BlockMemHandle mem_handle = ctx.kernel_.allocateSlices(
              ctx.device_,
              /*num_lhs=*/ctx.gm_,
              /*num_rhs=*/0,
              /*num_slices=*/1,
              /*lhs_blocks=*/&lhs_blocks, /*rhs_blocks=*/nullptr);

          blocks = ThreadLocalBlocks<LhsBlock>(std::move(mem_handle),
                                               std::move(lhs_blocks));
        }

        static void reuse(EvalCtx& ctx, int index, Blocks& blocks) {
          LhsBlock* ptr = &ctx.lhs_thread_local_pre_allocated_[ctx.gm_ * index];
          blocks = ThreadLocalBlocks<LhsBlock>(ptr, ctx.gm_);
        }
      };

      EvalParallelContext& ctx_;
      const int num_worker_threads_;
    };

    //用于释放线程本地块（ThreadLocalBlocks）。该模板类有一个模板参数BlockType，指定了线程本地块的类型。
    template <typename BlockType>
    class ThreadLocalBlocksRelease {
     public:
     //公共类型别名Blocks，指定了线程本地块的类型为BlockType和ThreadLocalBlocks的结合体。
      using Blocks = ThreadLocalBlocks<BlockType>;
      ThreadLocalBlocksRelease(EvalParallelContext& ctx) : ctx_(ctx) {}
      //释放该线程本地块。释放该线程本地块的内存。
      void operator()(Blocks& blocks) { blocks.Release(ctx_); }

     private:
      EvalParallelContext& ctx_;
    };

    // ThreadLocalBlocks initialization callables.
    using ThreadLocalLhsInit =
        ThreadLocalBlocksInitialize<LhsBlock, /*is_rhs=*/false>;
    using ThreadLocalRhsInit =
        ThreadLocalBlocksInitialize<RhsBlock, /*is_rhs=*/true>;

    // ThreadLocalBlocks release callables.
    using ThreadLocalLhsRelease = ThreadLocalBlocksRelease<LhsBlock>;
    using ThreadLocalRhsRelease = ThreadLocalBlocksRelease<RhsBlock>;

    // Thread local containers for Lhs/Rhs block packs. In practice only one of
    // them will be used, depending on the shard_by_col value.
    //保存左乘矩阵块的线程本地存储；
    Eigen::ThreadLocal<ThreadLocalBlocks<LhsBlock>, ThreadLocalLhsInit,
                       ThreadLocalLhsRelease>
        lhs_thread_local_blocks_;
    //保存右乘矩阵块的线程本地存储；
    Eigen::ThreadLocal<ThreadLocalBlocks<RhsBlock>, ThreadLocalRhsInit,
                       ThreadLocalRhsRelease>
        rhs_thread_local_blocks_;

    // After a particular shard for Kth slice missed thread local execution
    // opportunity (K-1 slice didn't complete kernels execution), we can no
    // longer schedule K+1 and following slices in thread local mode, because
    // there is no more guarantee that previous kernels were executed
    // sequentially in the same thread (size is nn_ or nm_).
    /*在第K个slice错过了线程本地执行的机会（K-1个slice没有完成内核执行）之后，
    我们不能再将K+1和以下的slice安排在线程本地模式下，因为没有了之前内核按照相同线程按顺序执行的保证（大小为nn_或nm_）。
    */
    std::atomic<bool>* can_use_thread_local_packed_;

    std::atomic<uint8_t>** state_kernel_[P];
    // state_switch_ is frequently modified by worker threads, while other
    // fields are read-only after constructor. Let's move it to a separate cache
    // line to reduce cache-coherency traffic.
    /*state_switch_ 经常被工作线程修改，而其他字段在构造函数之后只读。为了减少缓存一致性流量，将其移动到单独的缓存行中。
    */
    char pad_[128];
    std::atomic<Index> state_packing_ready_[P];
    std::atomic<Index> state_switch_[P];
    //从线程局部块和全局块中获取 LhsBlock 的逻辑。
    LhsBlock& packed_lhs(Index m, Index k, Index m1, bool use_thread_local) {
      //使用线程局部块
      if (use_thread_local) {
        eigen_assert(!shard_by_col_);
        ThreadLocalBlocks<LhsBlock>& blocks = lhs_thread_local_blocks_.local();

        Index grain_index = m1 - m * gm_;
        return blocks.block(internal::convert_index<int>(grain_index)); // FIXME better make ThreadLocalBlocks use Eigen::Index?
      } else {
        //使用全局块，此时从 packed_lhs_ 中获取相应的 LhsBlock 对象。
        //由于一个线程可能同时处理多个分片，因此需要使用 m1 和 k % (P - 1) 来确定在 packed_lhs_ 中的索引。
        return packed_lhs_[k % (P - 1)][m1];
      }
    }

    RhsBlock& packed_rhs(Index n, Index k, Index n1, bool use_thread_local) {
      if (use_thread_local) {
        eigen_assert(shard_by_col_);
        ThreadLocalBlocks<RhsBlock>& blocks = rhs_thread_local_blocks_.local();

        Index grain_index = n1 - n * gn_;
        return blocks.block(internal::convert_index<int>(grain_index)); // FIXME better make ThreadLocalBlocks use Eigen::Index?
      } else {
        return packed_rhs_[k % (P - 1)][n1];
      }
    }

    // In following two methods (pack_lhs and pack_rhs), if we know for sure
    // that we'll be able to immediately call a kernel with packed data, and do
    // not submit it to the thread pool, we can use thread local memory for
    // packed data.
    /*在以下两个方法（pack_lhs和pack_rhs）中，如果我们确定能够立即使用打包数据调用内核，
    并且不将其提交到线程池，则可以使用线程本地内存进行打包数据。*/

    // We can only reliably check it if we are running all kernels in sync mode
    // (parallelize only by sharding dim). If kernel for m==0 (n==0) is ready to
    // run, it's guaranteed that all kernels with larger values of m (n) are
    // also ready, because we execute them in the same order for all K slices.
    /*只有在运行所有内核的同步模式下（仅按分片维度并行化），我们才能可靠地检查它。
    如果m == 0（n == 0）的内核已准备好运行，则保证具有更大m（n）值的所有内核也已准备好，因为我们对所有K个切片按相同顺序执行它们。*/

    //将左矩阵LHS打包成块并存储到packed_lhs_中。
    void pack_lhs(Index m, Index k) {
      bool use_thread_local = false;
      // 检查是否可以使用 thread local 内存来存储 packed 数据
      if (parallelize_by_sharding_dim_only_ && !shard_by_col_ &&
          can_use_thread_local_packed_[m].load(std::memory_order_relaxed)) {
        // 如果 m=0 的 kernel 可以执行，则可以确保所有 m 值更大的 kernel 也都已经准备好执行
        if (state_kernel_[k % P][m][0].load(std::memory_order_relaxed) == 1) {
          use_thread_local = true;
        } else {
          // If we can't guarantee that all kernels in `k` slice will be
          // executed sequentially in current thread, it's no longer safe to use
          // thread local memory in following slices along the k dimensions.
          // 如果不能保证当前线程内所有 kernels 执行顺序，那么无法在接下来的 k 维度中继续使用 thread local 内存
          eigen_assert(k > 0);
          can_use_thread_local_packed_[m].store(false,
                                                std::memory_order_relaxed);
        }
      }
      // 对于当前的 m，将 lhs 数据打包到 packed_lhs_ 中
      const Index mend = m * gm_ + gm(m);
      for (Index m1 = m * gm_; m1 < mend; m1++)
        kernel_.packLhs(&packed_lhs(m, k, m1, use_thread_local),
                        lhs_.getSubMapper(m1 * bm_, k * bk_), bk(k), bm(m1));

      // 如果不使用线程池，则在打包完成后向 kernel 发出信号
      if (!parallel_pack_ && shard_by_col_) {
        eigen_assert(!use_thread_local);
        signal_packing(k);
      } else {
        // 如果使用线程池，则向下一步骤发出信号
        signal_switch(k + 1);
        // 逆序循环 nn_ 维度，为每个 (m, n) 发送 kernel 执行信号
        for (Index n = nn_ - 1; n >= 0; n--) {
          // 如果使用 thread local 内存，则同步执行，否则异步执行
          bool sync = parallelize_by_sharding_dim_only_ || n == 0;
          signal_kernel(m, n, k, sync, use_thread_local);
        }
      }
    }

    void pack_rhs(Index n, Index k) {
      bool use_thread_local = false;

      if (parallelize_by_sharding_dim_only_ && shard_by_col_ &&
          can_use_thread_local_packed_[n].load(std::memory_order_relaxed)) {
        if (state_kernel_[k % P][0][n].load(std::memory_order_relaxed) == 1) {
          use_thread_local = true;
        } else {
          // If we can't guarantee that all kernels in `k` slice will be
          // executed sequentially in current thread, it's no longer safe to use
          // thread local memory in following slices along the k dimensions.
          eigen_assert(k > 0);
          can_use_thread_local_packed_[n].store(false,
                                                std::memory_order_relaxed);
        }
      }

      const Index nend = n * gn_ + gn(n);
      for (Index n1 = n * gn_; n1 < nend; n1++) {
        if (!TensorContractionKernel::HasBeta && k == 0) {
          // Zero the output memory in parallel, only if contraction kernel does
          // not support `beta`. Otherwise we will pass beta 0.0 to the first
          // call to the `TensorContractionKernel::invoke()`.
          //
          // On 10000x2x10000 mm zeroing can easily take half of time. Zero (bn
          // x m) row. Safe to do here because all kernels that will write to
          // this memory depend on completion of this task. Note: don't call
          // device_.fill() here. device_.fill() blocks on thread pool
          // worker thread, which can lead to underutilization and deadlocks.
          // 如果 `TensorContractionKernel::HasBeta` 为 false，并且 `k` 为 0，则在并行中将输出内存清零。
          // 只有在缩并核不支持 `beta` 时，才需要进行清零操作。否则，我们将在第一次调用 `TensorContractionKernel::invoke()` 时传递 beta = 0.0。
          // 在 10000x2x10000 的情况下，清零操作很容易占用一半的时间。
          // 安全的在这里进行清零，因为所有将写入此内存的缩并核都依赖于此任务的完成。
          // 注意：不要在此处调用 `device_.fill()`。`device_.fill()` 会阻塞线程池的工作线程，可能导致利用率低下和死锁。

          std::fill_n(buffer_ + n1 * bn_ * m_, bn(n1) * m_, Scalar(0));
        }
        // 将右操作数打包到 `packed_rhs` 中
        kernel_.packRhs(&packed_rhs(n, k, n1, use_thread_local),
                        rhs_.getSubMapper(k * bk_, n1 * bn_), bk(k), bn(n1));
      }

      if (parallel_pack_ || shard_by_col_) {
        // 如果启用了并行打包或按列分片，则发送信号以切换到下一层。然后向后遍历 `m` 并向所有需要同步的缩并核发出信号。
        signal_switch(k + 1);
        for (Index m = nm_ - 1; m >= 0; m--) {
          // 如果仅通过分片维度进行并行处理或当前处于第一个块，则设置 `sync` 为 `true`，否则为 `false`。
          bool sync = parallelize_by_sharding_dim_only_ || m == 0;
          // 向所有需要同步的缩并核发出信号
          signal_kernel(m, n, k, sync, use_thread_local);
        }
      } else {
        // 如果没有启用并行打包且未按列分片，则确保 `use_thread_local` 为 false，并发送信号以表明打包完成。
        eigen_assert(!use_thread_local);
        signal_packing(k);
      }
    }

    //执行张量乘法的内核函数，参数 m、n 和 k 分别表示当前处理的维度编号。
    void kernel(Index m, Index n, Index k, bool use_thread_local) {
      // Note: order of iteration matters here. Iteration over m is innermost
      // because we want to reuse the same packed rhs in consecutive tasks
      // (rhs fits into L2$ while lhs only into L3$).
      /*注意：迭代顺序在这里很重要。迭代m是最内层的，因为我们希望在连续的任务中重复使用相同的打包RHS（RHS适合L2$，而LHS只适合L3$）。*/
      //计算出了处理的行数和列数的范围（nend 和 mend）。
      const Index nend = n * gn_ + gn(n);
      const Index mend = m * gm_ + gm(m);

      // NOTE: output = alpha * LHS * RHS + beta * output.
      //后定义了 alpha 和 beta 的值，它们分别用于执行乘法操作和输出结果的累加操作。
      const Scalar alpha = Scalar(1);
      const Scalar beta =
          (TensorContractionKernel::HasBeta && k == 0) ? Scalar(0) : Scalar(1);

      //迭代处理每个块 [m1, n1]。如果是按列分块，则先迭代行数 n1，再迭代列数 m1，反之亦然。
      if (shard_by_col_) {
        for (Index n1 = n * gn_; n1 < nend; n1++) {
          for (Index m1 = m * gm_; m1 < mend; m1++) {
            const auto output_mapper = output_.getSubMapper(m1 * bm_, n1 * bn_);
            /*调用了张量乘法内核的 invoke 函数，用于执行 LHS 与 RHS 的乘法操作，并将结果写入到 output_mapper 指定的输出内存区域中。*/
            kernel_.invoke(
                output_mapper,
                packed_lhs(m, k, m1, !shard_by_col_ && use_thread_local),
                packed_rhs(n, k, n1, shard_by_col_ && use_thread_local), bm(m1),
                bk(k), bn(n1), alpha, beta);

            // We are done with the last task for the [m1, n1] block.
            //如果 k+1 等于 nk，则说明当前是最后一个乘法操作，需要调用 output_kernel_ 函数输出最终结果。
            if (k + 1 == nk_) {
              output_kernel_(output_mapper, tensor_contraction_params_,
                             m1 * bm_, n1 * bn_, bm(m1), bn(n1));
            }
          }
        }
      } else {
        for (Index m1 = m * gm_; m1 < mend; m1++)
          for (Index n1 = n * gn_; n1 < nend; n1++) {
            const auto output_mapper = output_.getSubMapper(m1 * bm_, n1 * bn_);
            kernel_.invoke(
                output_mapper,
                packed_lhs(m, k, m1, !shard_by_col_ && use_thread_local),
                packed_rhs(n, k, n1, shard_by_col_ && use_thread_local), bm(m1),
                bk(k), bn(n1), alpha, beta);

            // We are done with the last task for the [m1, n1] block.
            if (k + 1 == nk_) {
              output_kernel_(output_mapper, tensor_contraction_params_,
                             m1 * bm_, n1 * bn_, bm(m1), bn(n1));
            }
          }
      }
      //最后调用了 signal_kernel 函数和 signal_switch 函数，用于通知其他线程继续执行。
      signal_kernel(m, n, k + 1, /*sync=*/false, /*use_thread_local=*/false);
      signal_switch(k + 2);
    }

    //触发矩阵打包的操作,参数k指示了矩阵乘积计算的第k个块的位置。
    void signal_packing(Index k) {
      //如果不采用并行打包，则会对所有行或列的矩阵进行打包。
      eigen_assert(!parallel_pack_);
      //函数会将对应的就绪状态减1，如果减1后状态不为0，表示仍有任务未完成，则直接返回。
      Index s = state_packing_ready_[k % P].fetch_sub(1);
      eigen_assert(s > 0);
      if (s != 1) return;
      //如果减1后状态为0，则将状态重置为该行或列的数量，并将打包任务加入队列中。
      state_packing_ready_[k % P] = shard_by_col_ ? nm_ : nn_;
      enqueue_packing(k, shard_by_col_);
    }

    //实现了向量化内核计算的信号发送
    void signal_kernel(Index m, Index n, Index k, bool sync,
                       bool use_thread_local) {
      // 获取指定kernel的状态变量
      std::atomic<uint8_t>* state = &state_kernel_[k % P][m][n];
      // 获取状态变量的值
      Index s = state->load();
      // 确认状态变量大于0
      eigen_assert(s > 0);
      // 如果状态不为1，并且减去1之后的值不为1，则返回
      if (s != 1 && state->fetch_sub(1) != 1) {
        eigen_assert(!use_thread_local);
        return;
      }
      // 将状态变量设为2或3，取决于是否开启了并行packing
      state->store(parallel_pack_ ? 3 : 2, std::memory_order_relaxed);
      // 如果需要同步执行kernel，直接调用kernel函数
      if (sync) {
        kernel(m, n, k, use_thread_local);
      } else {
        // 否则，将kernel函数提交到设备队列中异步执行
        eigen_assert(!use_thread_local);
        device_.enqueueNoNotification(
            [=]() { kernel(m, n, k, use_thread_local); });
      }
    }

    //通知矩阵乘法的切换。  
    //作用是告知从一个k切片转移到下一个。在此过程中，还要进行左右矩阵的打包。
    void signal_switch(Index k, Index v = 1) {
      Index s = state_switch_[k % P].fetch_sub(v);
      eigen_assert(s >= v);
      if (s != v) return;

      // Ready to switch to the next k slice.
      // Reset counter for the next iteration.
      // 准备切换到下一个k片段。
      // 重置下一次迭代的计数器。
      state_switch_[k % P] =
          (parallel_pack_ ? nm_ + nn_ : (shard_by_col_ ? nn_ : nm_)) +
          nm_ * nn_;
      if (k < nk_) {
        // Issue lhs/rhs packing. Their completion will in turn kick off
        // kernels.
        // 发送lhs/rhs打包信号。它们的完成将依次触发内核计算。
        if (parallel_pack_) {
          enqueue_packing(k, !shard_by_col_);
          enqueue_packing(k, shard_by_col_);
        } else if (shard_by_col_) {
          enqueue_packing(k, false);
        } else {
          enqueue_packing(k, true);
        }

        // Termination handling.
        // Because kernel completion signals k + 2 switch, we need to finish nk
        // + 2 slices without issuing any tasks on nk + 1 slice. So here we
        // pretend that all nk + 1 packing tasks just finish instantly; so that
        // nk + 2 switch only waits for completion of nk kernels.
        // 终止处理。
        // 因为内核完成信号触发k+2的切换，我们需要在没有在nk+1片段上发出任何任务的情况下完成nk+2个片段。
        // 所以在这里我们假装nk+1个打包任务立即完成；
        // 以便nk+2的切换只等待nk个内核的完成。
      } else if (k == nk_) {
        signal_switch(k + 1,
                      parallel_pack_ ? nm_ + nn_ : (shard_by_col_ ? nn_ : nm_));
      } else {
        done_.Notify();
      }
    }

    // Enqueue all rhs/lhs packing for k-th slice.
    void enqueue_packing(Index k, bool rhs) {
      enqueue_packing_helper(0, rhs ? nn_ : nm_, k, rhs);
    }

    // 为第 k 个 slice 中的所有 rhs 或 lhs 都加入 packing 任务。
    void enqueue_packing_helper(Index start, Index end, Index k, bool rhs) {
      // 递归函数，为一组数据中的所有 rhs 或 lhs 加入 packing 任务。
      //// start 和 end 指示了这组数据在矩阵中的起始和结束位置，k 表示当前 slice 的编号，rhs 表示当前是在处理 rhs。
      if (end - start == 1) {
        if (rhs)
          pack_rhs(start, k);
        else
          pack_lhs(start, k);
      } else {
        // 否则递归处理左右两组数据。
        while (end - start > 1) {
          Index mid = (start + end) / 2;
          // 为右半部分数据加入一个新的任务。
          device_.enqueueNoNotification(
              [=]() { enqueue_packing_helper(mid, end, k, rhs); });
          end = mid;
        }

        // Decide if we want to run first packing task (start == 0) in
        // async mode if we parallelize only by sharding dim:
        // (1) pack_lhs and pack_rhs call signal_switch before completing
        //     all calls to signal_kernel, which in sync mode might lead
        //     to the execution of the first kernel of the k+1 slice, before
        //     completing a call to the last kernel of the k slice.
        // (2) all pack tasks for sharded dim must be executed in a thread
        //     pool to get pre-allocated thead local buffers.
        /* 在仅通过分片维度并行化的情况下，决定是否以异步模式运行第一个打包任务（start == 0）:
        （1）pack_lhs和pack_rhs在完成对signal_kernel的所有调用之前调用signal_switch，
             在同步模式下可能会导致在完成k切片的最后一个kernel的调用之前执行k + 1切片的第一个kernel。
        （2）所有针对分片维度的打包任务必须在线程池中执行，以获取预分配的线程本地缓冲区。*/
        // 对于 sharded dim，如果只按该维度并行化，则可能需要将第一个 packing 任务设置为异步执行。
        bool pack_async =
          (start == 0) &&
          (parallelize_by_sharding_dim_only_&& shard_by_col_ == rhs) &&
          (k > 0 || std::this_thread::get_id() == created_by_thread_id_);

        // 加入左半部分的 packing 任务。
        if (pack_async) {
          device_.enqueueNoNotification(
              [=]() { enqueue_packing_helper(start, end, k, rhs); });
        } else {
          enqueue_packing_helper(start, end, k, rhs);
        }
      }
    }

    // Block sizes with accounting for potentially incomplete last block.
    // 构造函数，初始化各成员变量
    Index bm(Index m) const { return m + 1 < nm0_ ? bm_ : m_ + bm_ - bm_ * nm0_; }
    Index bn(Index n) const { return n + 1 < nn0_ ? bn_ : n_ + bn_ - bn_ * nn0_; }
    Index bk(Index k) const { return k + 1 < nk_ ? bk_ : k_ + bk_ - bk_ * nk_; }
    // Task grain sizes accounting for potentially incomplete last task.
    Index gm(Index m) const { return m + 1 < nm_ ? gm_ : nm0_ + gm_ - gm_ * nm_; }
    Index gn(Index n) const { return n + 1 < nn_ ? gn_ : nn0_ + gn_ - gn_ * nn_; }

    EvalParallelContext(const EvalParallelContext&) = delete;
    void operator=(const EvalParallelContext&) = delete;
  };

  //lhs_inner_dim_contiguous：一个布尔值，表示左矩阵的内部维度是否为连续存储。
  //rhs_inner_dim_contiguous：一个布尔值，表示右矩阵的内部维度是否为连续存储。
  //rhs_inner_dim_reordered：一个布尔值，表示右矩阵的内部维度是否已重新排序。
  //Alignment：一个整数值，表示矩阵的对齐方式。
  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous,
            bool rhs_inner_dim_reordered, int Alignment>
  using SyncEvalParallelContext =
      EvalParallelContext<NoCallback, lhs_inner_dim_contiguous,
                          rhs_inner_dim_contiguous, rhs_inner_dim_reordered,
                          Alignment>;

  // ------------------------------------------------------------------------ //

  // EvalShardedByInnerDimContext orchestrates sync/async contraction
  // evaluation, when we shard by inner dimension. When it is executed in
  // asynchronous mode, it owns all the shared state that might be accessible by
  // block processing tasks.
  /*EvalShardedByInnerDimContext用于协调按内部维度划分分片的同步/异步收缩运算评估。
  当以异步模式执行时，它拥有所有可能被块处理任务访问的共享状态。*/

  template <typename DoneCallback>
  struct EvalShardedByInnerDimContext {
    EvalShardedByInnerDimContext(const Self* self, int num_threads,
                                 Scalar* result_buffer,
                                 Index m_size, Index n_size, Index k_size,
                                 DoneCallback done_callback)
        : evaluator(self),
          m_lhs_inner_dim_contiguous(evaluator->m_lhs_inner_dim_contiguous),
          m_rhs_inner_dim_contiguous(evaluator->m_rhs_inner_dim_contiguous),
          m_rhs_inner_dim_reordered(evaluator->m_rhs_inner_dim_reordered),
          result(result_buffer),
          m(m_size),
          n(n_size),
          k(k_size),
          done(std::move(done_callback)),
          buffer_size_bytes(m * n * sizeof(Scalar)),
          block_size(blockSize(k, num_threads)),
          num_blocks(divup<Index>(k, block_size)),
          num_pending_blocks(internal::convert_index<int>(num_blocks)),
          l0_ranges(divup<Index>(num_blocks, l0_size)),
          l0_state(l0_ranges),
          block_buffers(num_blocks) {
      // Keep count of pending gemm tasks for each l0 range.
      // 对每个l0范围保持待处理的gemm任务数量的计数。
      for (int i = 0; i < l0_ranges; ++i) {
        const Index num_pending_tasks = actualRangeSize(l0_ranges, l0_size, i);
        l0_state.emplace_back(internal::convert_index<int>(num_pending_tasks));
      }

      // Allocate temporary buffers for each block.
      // 为每个块分配临时缓冲区。
      for (Index block_idx = 0; block_idx < num_blocks; ++block_idx) {
        //为每个块分配一个临时缓冲区。
        //对于第一个块，即乘积矩阵，缓冲区被设置为 result 指向的缓冲区
        //对于其他块，即操作数矩阵的小块，缓冲区通过 evaluator->m_device.allocate(buffer_size_bytes) 分配在 GPU 设备上。
        Scalar* buf = block_idx == 0
                          ? result
                          : static_cast<Scalar*>(evaluator->m_device.allocate(
                                buffer_size_bytes));
        block_buffers.emplace_back(buf);
      }
    }

    ~EvalShardedByInnerDimContext() {
      for (Index i = 1; i < num_blocks; ++i) {
        evaluator->m_device.deallocate(block_buffers[i]);
      }
    }

    // 在指定对齐方式下运行并发评估
    template <int Alignment>
    void run() {
      Barrier barrier(internal::convert_index<int>(num_blocks));
      // 并发评估
      eval<Alignment>(barrier, 0, num_blocks);
      // 等待所有任务完成
      barrier.Wait();

      // Aggregate partial sums from l0 ranges.
      // 对l0范围的部分结果进行聚合
      aggregateL0Blocks<Alignment>();

      // Apply output kernel.
      // 应用输出kernel
      applyOutputKernel();
    }

    //异步运行矩阵乘法的实现，使用模板参数 Alignment 表示对齐方式。
    template <int Alignment>
    void runAsync() {
      // 从第一个块开始异步执行。
      evalAsync<Alignment>(0, num_blocks);
    }

   private:
    // The underlying GEMM kernel assumes that k is a multiple of
    // the packet size and subtle breakage occurs if this is violated.
    //一个数据包的大小，即RhsScalar类型在该平台上的向量长度。
    static const Index packet_size = internal::packet_traits<RhsScalar>::size;

    const Self* evaluator;  // TensorContraction evaluator 对应的张量收缩计算器。

    // These fields required fromTENSOR_CONTRACTION_DISPATCH macro.
    bool m_lhs_inner_dim_contiguous;//表示lhs张量的内部维度是否是紧凑的。
    bool m_rhs_inner_dim_contiguous;//表示rhs张量的内部维度是否是紧凑的。
    bool m_rhs_inner_dim_reordered;//表示rhs张量的内部维度是否被重新排序了。

    Scalar* result;//输出张量的指针。

    Index m;
    Index n;
    Index k;

    DoneCallback done;//异步计算完成后需要执行的回调函数。

    // ----------------------------------------------------------------------//
    // Algorithm parameters.

    // We will compute partial results into the buffers of this size.
    // 计算出的部分结果将储存在这个大小的缓冲区中。
    Index buffer_size_bytes;

    Index block_size;// 每个块的大小
    Index num_blocks;// 每个块的大小

    // Keep track of pending tasks when evaluate in async mode.
    // 在异步模式下，跟踪待处理的任务数。
    std::atomic<int> num_pending_blocks;

    // We compute partial gemm results in parallel, and to get the final result
    // we need to add them all together. For the large number of threads (>= 48)
    // this adds a very expensive sequential step at the end.
    /*我们并行计算了部分的gemm结果，为了得到最终结果，我们需要将它们全部加在一起。
    对于大量的线程（>=48），这会在最后增加一个非常昂贵的顺序步骤。*/
    
    // We split the [0, num_blocks) into small ranges, and when a task for the
    // block finishes its partial gemm computation, it checks if it was the last
    // gemm in the range, and if so, it will add all blocks of the range.
    /*我们将[0，num_blocks）划分为小范围，在块的任务完成部分gemm计算时，
    它会检查它是否是范围内的最后一个gemm，如果是，它将添加该范围内的所有块。*/
    
    // After all tasks done, we need to add only these pre-aggregated blocks.
    /*在所有任务完成后，我们只需要添加这些预聚合的块。*/

    // For now we use just a single level of ranges to compute pre-aggregated
    // partial sums, but in general we can use more layers to compute tree
    // aggregation in parallel and reduce the size of the sequential step.
    /*目前，我们使用单个级别的范围来计算预聚合的部分和，
    但通常我们可以使用更多层来并行计算树聚合并减小顺序步骤的大小。*/
    
    // TODO(ezhulenev): Add multilevel tree aggregation? Probably will make
    // sense only if number of threads >= ~128?
    /*待办事项（ezhulenev）：添加多级树聚合？可能只在线程数> =〜128时有意义？*/
    static const Index l0_size = 4;
    Index l0_ranges;

    // Keep count of pending gemm tasks for each l0 range.
    MaxSizeVector<std::atomic<int>> l0_state;  // [0, l0_ranges)

    // Buffers allocated for each temporary block computation.
    MaxSizeVector<Scalar*> block_buffers;  // [0, num_blocks)

    //使用了一个模板参数 Alignment，它接受一个整数，表示数据对齐方式。
    template <int Alignment>
    void processBlock(Index block_idx, Index begin, Index end) {
      Scalar* buf = block_buffers[block_idx];
      // 并行计算GEMM的局部结果，并将其存储在缓冲区中。
      // 使用TENSOR_CONTRACTION_DISPATCH宏根据指定的内存对齐方式分派到相应的函数中执行。
      TENSOR_CONTRACTION_DISPATCH(
          evaluator->template evalGemmPartialWithoutOutputKernel, Alignment,
          (buf, begin, end,
           /*num_threads=*/internal::convert_index<int>(num_blocks)));

      // Check if it was the last task in l0 range.
      // 检查当前的任务是否为L0范围内的最后一个任务。
      // 若是，则将该范围内所有块的局部结果聚合到该范围内的第一个块中。
      const Index l0_index = block_idx / l0_size;
      const int v = l0_state[l0_index].fetch_sub(1);
      eigen_assert(v >= 1);

      // If we processed the last block of the range, we can aggregate all
      // partial results into the first block of the range.
      if (v == 1) {
        // 获取该L0范围内的块数目。
        const Index rng_size = actualRangeSize(l0_ranges, l0_size, l0_index);
        const Index dst_block_idx = l0_index * l0_size;
        // 如果该L0范围内的块数目等于L0_SIZE，则可以使用addAllToBuffer函数将所有块的局部结果聚合到第一个块中。
        if (rng_size == l0_size) {
          addAllToBuffer<Alignment>(
              m * n,
              /*src_buf0=*/block_buffers[dst_block_idx + 1],
              /*src_buf1=*/block_buffers[dst_block_idx + 2],
              /*src_buf2=*/block_buffers[dst_block_idx + 3],
              /*dst_buf= */ block_buffers[dst_block_idx]);
        } else {
          // 如果该L0范围内的块数目不等于L0_SIZE，则需要对该范围内的所有块进行聚合。
          // Aggregate blocks of potentially incomplete last range.
          for (int i = 1; i < rng_size; ++i) {
            addToBuffer<Alignment>(m * n,
                                   /*src_buf=*/block_buffers[dst_block_idx + i],
                                   /*dst_buf=*/block_buffers[dst_block_idx]);
          }
        }
      }
    }

    // Aggregate partial sums from l0 ranges.
    // 对 l0 范围内的部分积进行聚合
    template <int Alignment>
    void aggregateL0Blocks() const {
      Index l0_index = 1;
      // 每次聚合三个范围的部分积
      for (; l0_index + 2 < l0_ranges; l0_index += 3) {
        addAllToBuffer<Alignment>(
            m * n,
            /*src_buf0=*/block_buffers[(l0_index + 0) * l0_size],
            /*src_buf1=*/block_buffers[(l0_index + 1) * l0_size],
            /*src_buf2=*/block_buffers[(l0_index + 2) * l0_size],
            /*dst_buf= */ block_buffers[0]);
      }
      // 对剩余的范围进行部分积的聚合
      for (; l0_index < l0_ranges; ++l0_index) {
        addToBuffer<Alignment>(m * n, block_buffers[l0_index * l0_size],
                               block_buffers[0]);
      }
    }
    // 应用输出内核计算张量收缩结果。
    void applyOutputKernel() const {
      // 定义输出数据的映射类型。
      typedef internal::blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;
      // 调用输出内核计算结果。
      evaluator->m_output_kernel(
          OutputMapper(result, m), evaluator->m_tensor_contraction_params,
          static_cast<Eigen::Index>(0), static_cast<Eigen::Index>(0), m, n);
    }

    // Compute block size with accounting for potentially incomplete last block.
    // 实际计算块的大小，考虑最后一个块可能不完整。
    Index actualBlockSize(Index block_idx) const {
      return block_idx + 1 < num_blocks
                 ? block_size
                 : k + block_size - block_size * num_blocks;
    };

    // Compute range size with accounting for potentially incomplete last range.
    // 根据最后一个range是否完整来计算range的实际大小
    Index actualRangeSize(Index num_ranges, Index range_size,
                          Index range_idx) const {
      eigen_assert(range_idx < num_ranges); // 确保range_idx在范围内
      return range_idx + 1 < num_ranges// 如果不是最后一个range
                 ? range_size // 直接返回range_size
                 : num_blocks + range_size - range_size * num_ranges;// 否则计算最后一个range的大小
    };

    // 向目标缓冲区添加源缓冲区的值。
    // n：缓冲区的长度。
    // src_buf：源缓冲区的指针。
    // tgt_buf：目标缓冲区的指针。
    template <int Alignment>
    EIGEN_STRONG_INLINE static void addToBuffer(size_t n, const Scalar* src_buf,
                                                Scalar* tgt_buf) {
      // 计算输出数据包的大小。
      const int output_packet_size =
          internal::unpacket_traits<PacketReturnType>::size;
      size_t i = 0;
      // 将缓冲区分成若干个数据包，并计算每个数据包的和。
      const size_t num_packets = n / output_packet_size;
      for (; i < output_packet_size * num_packets; i += output_packet_size) {
        // 加载源缓冲区的数据包。
        const PacketReturnType src_val =
            internal::pload<PacketReturnType>(src_buf + i);
        // 加载目标缓冲区的对应数据包。
        const PacketReturnType tgt_val =
            internal::ploadt<PacketReturnType, Alignment>(tgt_buf + i);
        // 计算两个数据包的和。
        const PacketReturnType sum = internal::padd(src_val, tgt_val);
        // 将和存储到目标缓冲区的对应数据包中。
        internal::pstoret<Scalar, PacketReturnType, Alignment>(tgt_buf + i,
                                                               sum);
      }
      // 处理最后不足一个数据包的部分，直接按元素相加。
      for (; i < n; ++i) {
        tgt_buf[i] += src_buf[i];
      }
    }

    // 将三个源缓冲区中的值加到目标缓冲区中，其中每个缓冲区都是按对齐边界对齐的。
    template <int Alignment>
    EIGEN_STRONG_INLINE static void addAllToBuffer(size_t n,
                                                   const Scalar* src_buf0,
                                                   const Scalar* src_buf1,
                                                   const Scalar* src_buf2,
                                                   Scalar* dst_buf) {
      using ::Eigen::internal::padd;
      using ::Eigen::internal::pload;
      using ::Eigen::internal::ploadt;
      using ::Eigen::internal::pstoret;

      const int output_packet_size =
          internal::unpacket_traits<PacketReturnType>::size;

      size_t i = 0;
      const size_t num_packets = n / output_packet_size;
      // 循环，每次处理一个数据包大小的数据
      for (; i < output_packet_size * num_packets; i += output_packet_size) {
        // 加载数据包
        const auto src_val0 = pload<PacketReturnType>(src_buf0 + i);
        const auto src_val1 = pload<PacketReturnType>(src_buf1 + i);
        const auto src_val2 = pload<PacketReturnType>(src_buf2 + i);

        // 从目标缓冲区中加载数据包
        const auto dst_val = ploadt<PacketReturnType, Alignment>(dst_buf + i);
        // 计算三个数据包的和
        const auto sum =
            padd(padd(dst_val, src_val0), padd(src_val1, src_val2));
        // 将计算结果存储回目标缓冲区
        pstoret<Scalar, PacketReturnType, Alignment>(dst_buf + i, sum);
      }
      // 处理剩余不足一个数据包大小的数据
      for (; i < n; ++i) {
        dst_buf[i] += src_buf0[i] + src_buf1[i] + src_buf2[i];
      }
    }

//使用二分法递归计算矩阵块，知道矩阵块数量为1
    template <int Alignment>
    void eval(Barrier& barrier, Index start_block_idx, Index end_block_idx) {
      while (end_block_idx - start_block_idx > 1) {
        Index mid_block_idx = (start_block_idx + end_block_idx) / 2;
        //将计算设备压入设备队列，等待异步执行
        evaluator->m_device.enqueueNoNotification(
            [this, &barrier, mid_block_idx, end_block_idx]() {
              eval<Alignment>(barrier, mid_block_idx, end_block_idx);
            });
        end_block_idx = mid_block_idx;
      }

      //更新矩阵块数量范围，缩小计算范围
      Index block_idx = start_block_idx;
      Index block_start = block_idx * block_size;
      Index block_end = block_start + actualBlockSize(block_idx);
      // 对当前矩阵块进行计算
      processBlock<Alignment>(block_idx, block_start, block_end);
      //发送计算完成信号
      barrier.Notify();
    }

    template <int Alignment>
    void evalAsync(Index start_block_idx, Index end_block_idx) {
      while (end_block_idx - start_block_idx > 1) {
        //二分法拆分成多个小任务，提交给线程池异步进行
        Index mid_block_idx = (start_block_idx + end_block_idx) / 2;
        evaluator->m_device.enqueueNoNotification(
            [this, mid_block_idx, end_block_idx]() {
              evalAsync<Alignment>(mid_block_idx, end_block_idx);
            });
        end_block_idx = mid_block_idx;
      }

      Index block_idx = start_block_idx;
      //计算当前块的起始位置和终止位置
      Index block_start = block_idx * block_size;
      Index block_end = block_start + actualBlockSize(block_idx);
      //处理当前块
      processBlock<Alignment>(block_idx, block_start, block_end);

      int v = num_pending_blocks.fetch_sub(1);
      eigen_assert(v >= 1);

      if (v == 1) {
        // Aggregate partial sums from l0 ranges.
        //所有块都处理完成，进行结果聚合
        aggregateL0Blocks<Alignment>();

        // Apply output kernel.
        //应用输出内核
        applyOutputKernel();

        // NOTE: If we call `done` callback before deleting this (context),
        // it might deallocate Self* pointer captured by context, and we'll
        // fail in destructor trying to deallocate temporary buffers.

        // Move done call back from context before it will be destructed.
        // 注意：如果在删除当前上下文之前调用 `done` 回调，会尝试释放上下文中捕获的 Self* 指针，导致在析构函数中失败尝试释放临时缓冲区
        // 将 `done` 回调从上下文中移动出来，以避免在删除上下文之前调用它 
        DoneCallback done_copy = std::move(done);

        // We are confident that we are the last one who touches context.
        // 删除当前上下文，释放临时缓冲区
        delete this;

        // Now safely call the done callback.
        // 安全地调用 `done` 回调
        done_copy();
      }
    }

    // Cost model doesn't capture well the cost associated with constructing
    // tensor contraction mappers and computing loop bounds in gemm_pack_lhs
    // and gemm_pack_rhs, so we specify minimum desired block size.
    /*成本模型不能很好地捕捉张量收缩映射器构造以及在 gemm_pack_lhs 和 
    gemm_pack_rhs 中计算循环边界所涉及的成本，因此我们指定最小期望块大小。*/
    // 计算分块大小的函数，考虑了硬件的向量化特性和线程数量等因素
    // 参数 k：要进行分块的维度大小
    // 参数 num_threads：线程数量
    static Index blockSize(Index k, int num_threads) {
      const auto round_up = [=](Index index) -> Index {
        const Index kmultiple = packet_size <= 8 ? 8 : packet_size;// 向量化时要用到的大小
        return divup<Index>(index, kmultiple) * kmultiple;// 取大于等于index的最小的kmultiple的倍数
      };

      const Index target_block_size = round_up(divup<Index>(k, num_threads)); // 目标块大小
      const Index desired_min_block_size = 12 * packet_size;// 期望的最小块大小

      // 返回目标块大小和期望最小块大小中较小的一个，并且不超过k
      return numext::mini<Index>(
          k, numext::maxi<Index>(desired_min_block_size, target_block_size));
    }

    // 禁用拷贝构造函数和赋值函数
    EvalShardedByInnerDimContext(const EvalShardedByInnerDimContext&) = delete;
    void operator=(const EvalShardedByInnerDimContext&) = delete;
  };

  // ------------------------------------------------------------------------ //

  // Below are the function used by evalProductImpl heuristics, trying to select
  // optimcal parameters for parallelization algorithm.
  // 以下是 evalProductImpl 启发式算法中使用的函数，旨在选择并行化算法的最佳参数。

  // Decide whether we want to shard m x n contraction by columns or by rows.
  // 决定我们是要按列还是按行来进行分片m x n的收缩运算。
  static bool shardByCol(Index m, Index n, Index num_threads) {
    // Note: we are comparing both n and m against Traits::nr, it is not
    // a mistake. We are trying to figure out how both n and m will fit into
    // the main sharding dimension.
    /*注意：我们正在将n和m与Traits::nr进行比较，这不是一个错误。我们试图弄清楚n和m将如何适合主分片维度。*/

    // Sharding by column is the default
    // ... unless there is enough data for vectorization over rows
    // 按列进行分片是默认的选择
    // ... 除非有足够的数据可以沿着行进行向量化 
    if (m / num_threads >= Traits::nr &&
        // and not enough data for vectorization over columns
        // 并且数据不足以沿列进行向量化
        (n / num_threads < Traits::nr ||
         // ... or barely enough data for vectorization over columns,
         // but it is not evenly dividable across threads
         // ... 或者数据仅足够沿着列进行向量化，
         // 但无法在线程之间均匀分配
         (n / num_threads < 4 * Traits::nr &&
          (n % (num_threads * Traits::nr)) != 0 &&
          // ... and it is evenly dividable across threads for rows
          // ... 并且它在行上可以被均匀地分配
          ((m % (num_threads * Traits::nr)) == 0 ||
           // .. or it is not evenly dividable for both dimensions but
           // there is much more data over rows so that corner effects are
           // mitigated.
           // .. 或者它在两个维度上都不能被均匀地分配，但是
           // 行上有更多的数据，以便缓解边缘效应。
           (m / n >= 6)))))
      return false;
    // Wait, or if matrices are just substantially prolonged over the other
    // dimension.
    // 或者，如果矩阵在另一个维度上仅长得多。
    if (n / num_threads < 16 * Traits::nr && m > n * 32) return false;
    return true;
  }

  // 确定M方向的粗粒度块大小，以便用于多线程并行
  Index coarsenM(Index m, Index n, Index bm, Index bn, Index bk, Index gn,
                 int num_threads, bool shard_by_col) const {
    Index gm = 1;// 粗粒度块大小的初始值为1
    Index gm1 = 1; // 下一个候选的粗粒度块大小的初始值也为1
    Index nm0 = divup(m, bm);// 沿M方向分成的块的数量，使用输入的M方向块大小bm
    Index nm1 = nm0;
    for (;;) {
      // Find the next candidate for m grain size. It needs to result in
      // different number of blocks. E.g. if we have 10 kernels, we want to try
      // 5 and 10, but not 6, 7, 8 and 9.
      // 找到下一个候选的粗粒度块大小。需要结果产生不同数量的块。
      // 例如，如果有10个内核，则要尝试5和10，但不要尝试6、7、8和9。
      while (gm1 <= nm0 && nm1 == divup(nm0, gm1)) gm1++;
      // 候选的粗粒度块大小已超过M方向块数，退出循环
      if (gm1 > nm0) break;
      // Check the candidate.
      // 检查候选粗粒度块大小，返回-1表示候选无效，返回0表示继续尝试其他候选，返回1表示候选有效
      int res = checkGrain(m, n, bm, bn, bk, gm1, gn, gm, gn, num_threads,
                           shard_by_col);
      if (res < 0) break;// 候选无效，退出循环
      nm1 = divup(nm0, gm1); // 沿N方向的块数更新
      if (res == 0) continue;// 继续尝试其他候选
      // Commit new grain size.
      gm = gm1;// 候选有效，更新粗粒度块大小
    }
    return gm;// 返回确定的M方向粗粒度块大小
  }

  //和上一个对应
  Index coarsenN(Index m, Index n, Index bm, Index bn, Index bk, Index gm,
                 int num_threads, bool shard_by_col) const {
    Index gn = 1;
    Index gn1 = 1;
    Index nn0 = divup(n, bn);
    Index nn1 = nn0;
    for (;;) {
      while (gn1 <= nn0 && nn1 == divup(nn0, gn1)) gn1++;
      if (gn1 > nn0) break;
      int res = checkGrain(m, n, bm, bn, bk, gm, gn1, gm, gn, num_threads,
                           shard_by_col);
      if (res < 0) break;
      nn1 = divup(nn0, gn1);
      if (res == 0) continue;
      gn = gn1;
    }
    return gn;
  }

  // checkGrain checks whether grain (gm, gn) is suitable and is better than
  // (oldgm, oldgn).
  // checkGrain检查(grm, gn)是否合适，以及是否优于(oldgm, oldgn)
  int checkGrain(Index m, Index n, Index bm, Index bn, Index bk, Index gm,
                 Index gn, Index oldgm, Index oldgn, int num_threads,
                 bool shard_by_col) const {
    //计算张量操作成本
    const TensorOpCost cost =
        contractionCost(bm * gm, bn * gn, bm, bn, bk, shard_by_col, true);
    // 计算任务大小，如果任务太小，我们将接受它，否则同步开销会占主导地位
    double taskSize = TensorCostModel<ThreadPoolDevice>::taskSize(
        static_cast<double>(bm) * gm * bn * gn, cost);
    // If the task is too small, then we agree on it regardless of anything
    // else. Otherwise synchronization overheads will dominate.
    if (taskSize < 1) return 1;//任务太小
    // If it is too large, then we reject it and all larger tasks.
    if (taskSize > 2) return -1;//任务太大
    // Now we are in presumably good task size range.
    // The main deciding factor here is parallelism. Consider that we have 12
    // kernels and 4 threads. Grains of 2, 3 and 4 all yield good task sizes.
    // But 2/4 yield 6/3 tasks, which gives us parallelism of 0.75 (at most 3/4
    // of cores will be busy). While grain size 3 gives us 4 tasks, which gives
    // us parallelism of 1 (we can load all cores).
    /*现在我们假定我们处于良好的任务大小范围内。在这里，主要决定因素是并行性。
    假设我们有12个内核和4个线程。2、3和4的粒度都可以产生良好的任务大小。
    但是，2/4会产生6/3个任务，这给我们带来了0.75的并行性（最多只有3/4个核心将忙碌）。
    而3的粒度会产生4个任务，这给我们带来了1的并行性（我们可以利用所有核心）。*/
    Index nm0 = divup(m, bm);
    Index nn0 = divup(n, bn);
    Index new_tasks = divup(nm0, gm) * divup(nn0, gn);
    double new_parallelism = static_cast<double>(new_tasks) /
                             (divup<int>(new_tasks, num_threads) * num_threads);
    Index old_tasks = divup(nm0, oldgm) * divup(nn0, oldgn);
    double old_parallelism = static_cast<double>(old_tasks) /
                             (divup<int>(old_tasks, num_threads) * num_threads);
    if (new_parallelism > old_parallelism || new_parallelism == 1) return 1;
    //如果新的并行度比旧的并行度高或者新的并行度为 1，则返回 1，否则返回 0。
    return 0;
  }

  // 计算矩阵乘法的成本
  TensorOpCost contractionCost(Index m, Index n, Index bm, Index bn, Index bk,
                               bool shard_by_col, bool prepacked) const {
    // 计算矩阵中存储元素的数据类型大小，取左右矩阵数据类型的最小值
    const int packed_size = std::min<int>(PacketType<LhsScalar, Device>::size,
                                          PacketType<RhsScalar, Device>::size);
    // 计算输出的数据类型大小
    const int output_packet_size = internal::unpacket_traits<PacketReturnType>::size;
    // 计算矩阵计算时每个元素的操作数
    const double kd = static_cast<double>(bk);
    // 计算计算带宽
    double compute_bandwidth = computeBandwidth(false, bm, bn, bk);
    // Computations.
    // 计算矩阵计算成本
    TensorOpCost cost = TensorOpCost(0, 0, kd * compute_bandwidth, true, packed_size);
    // Output stores.
    // 计算输出存储成本
    cost += TensorOpCost(0, sizeof(CoeffReturnType), 0, true, output_packet_size);
    if (prepacked) {
      // Packing and kernels are executed in different tasks. When we calculate
      // task grain size we look only at kernel cost assuming that kernel
      // is more expensive than packing.
      // 若已预先打包，则返回仅包含计算成本的结果
      return cost;
    }
    // Lhs/rhs loads + computations.
    // 计算左右矩阵的加载成本和计算成本
    TensorOpCost lhsCost = this->m_leftImpl.costPerCoeff(true) * (kd / n);
    TensorOpCost rhsCost = this->m_rightImpl.costPerCoeff(true) * (kd / m);
    // Lhs packing memory cost does not contribute considerably to overall
    // execution time because lhs is prefetched early and accessed sequentially.
    // 若按列分块，则左矩阵的内存成本不占据执行时间，因为左矩阵已经被预读入内存，顺序访问
    if (shard_by_col)
      lhsCost.dropMemoryCost();
    else
      rhsCost.dropMemoryCost();
    // 返回总成本
    return cost + lhsCost + rhsCost;
  }

  // Decide whether we want to shard m x k x n contraction over the inner
  // (contraction) dimension (k).
  // 决定是否要在内部维度（k）上分片m x k x n的收缩运算。
  static bool shardByInnerDim(Index m, Index n, Index k, int num_threads,
                              int num_threads_by_k) {
    std::ptrdiff_t bufsize = m * n * sizeof(Scalar);
    bool shard_by_k = false;
    if (n == 1 ||                // If mat*vec or...是mat*vec
        num_threads_by_k < 2 ||  // running single threaded or...单线程运行
        num_threads_by_k <
            num_threads ||  // sharding by k gives less parallelism or... 按k分片会导致并行度降低
        bufsize > l3CacheSize() / num_threads_by_k ||  // need more buffer space 需要的缓冲区空间超过L3缓存
        // than L3 cache or...
        k / num_threads_by_k < 2 * Traits::nr) {  // k per thread is tiny.每个线程处理的k非常小
      shard_by_k = false;
    } else if (numext::maxi(m, n) / num_threads <
                   Traits::nr ||  // both other dimensions are tiny or... 两个其他维度非常小
               // k per thread is not small and... 每个线程处理的k不小
               (k / num_threads_by_k > 8 * Traits::nr &&
                // one of the outer dimensions is tiny or sharding by k offers
                // more parallelism.
                // 一个外部维度非常小或按k分片提供了更高的并行度。
                (numext::mini(m, n) < 2 * Traits::nr ||
                 num_threads_by_k > num_threads))) {
      shard_by_k = true;
    }
    return shard_by_k;
  }

  //给定矩阵尺寸m、n和k，计算沿内部维度（k）进行张量收缩的成本。
  TensorOpCost contractionCostPerInnerDim(Index m, Index n, Index k) const {
    // Compute cost.
    const int output_packet_size = internal::unpacket_traits<PacketReturnType>::size;
    TensorOpCost cost(0, 0, (computeBandwidth(true, m, n, k) * m) * n, true, output_packet_size);
    // Output stores.输出存储
    cost += TensorOpCost(0, sizeof(CoeffReturnType), 0, true, output_packet_size);
    TensorOpCost lhsCost = this->m_leftImpl.costPerCoeff(true) * m;
    TensorOpCost rhsCost = this->m_rightImpl.costPerCoeff(true) * n;
    // Since the inner gemm kernel is always sharded by column, the lhs
    // load cost is negligible.
    // 由于内部gemm内核总是按列分片，因此lhs加载成本可以忽略不计。
    lhsCost.dropMemoryCost();
    //返回总成本
    return cost + lhsCost + rhsCost;
  }

//给定矩阵尺寸m、n和k，计算内部维度（k）上的线程数。
  int numThreadsInnerDim(Index m, Index n, Index k) const {
    //计算输出包大小
    const int output_packet_size = internal::unpacket_traits<PacketReturnType>::size;
    //计算收缩成本
    TensorOpCost cost = contractionCostPerInnerDim(m, n, k);
    //计算总的并行成本
    double total_parallel_cost =
        TensorCostModel<ThreadPoolDevice>::totalCost(k, cost);
    // Cost of reduction step accumulating the m*n per-thread buffers into the
    // result.
    //减少步骤的成本，将m*n个线程缓冲区累加到结果中。
    double reduction_cost = TensorCostModel<ThreadPoolDevice>::totalCost(
        m * n, TensorOpCost(2, 1, 1, true, output_packet_size));
    int num_threads = 1;
    double min_cost = total_parallel_cost;
    double kPerThreadOverHead = 3000;//每个线程的开销
    double kFixedOverHead = 100000;//固定的开销
    //通过循环增加线程数，找到成本最小的线程数
    for (int nt = 2; nt <= this->m_device.numThreads(); nt += 2) {
      double sequential_cost =
          kFixedOverHead + nt * (reduction_cost + kPerThreadOverHead);
      double parallel_cost = total_parallel_cost / nt + sequential_cost;
      if (parallel_cost < min_cost) {
        num_threads = nt;
        min_cost = parallel_cost;
      }
    }
    return num_threads;
  }

// 计算矩阵乘法的带宽，即数据传输速度与计算速度的比值
// 参数shard_by_col：是否按列划分，bm：左矩阵的行数，bn：右矩阵的列数，bk：左矩阵的列数或右矩阵的行数（两者相等）
  double computeBandwidth(bool shard_by_col, Index bm, Index bn,
                          Index bk) const {
    // Peak VFMA bandwidth is 0.5. However if we have not enough data for
    // vectorization bandwidth drops. The 4.0 and 2.0 bandwidth is determined
    // experimentally.
    // 峰值VFMA（向量浮点乘累加）带宽为0.5，但是如果数据量不足则带宽会降低。
    // 实验确定4.0和2.0的带宽。
    double computeBandwidth =
        bk == 1 ? 4.0
                : (shard_by_col ? bn : bm) < Traits::nr ||
                          (shard_by_col ? bm : bn) < Traits::mr
                      ? 2.0
                      : 0.5;
#ifndef EIGEN_VECTORIZE_FMA
    // Bandwidth of all of VFMA/MULPS/ADDPS is 0.5 on latest Intel processors.
    // However for MULPS/ADDPS we have dependent sequence of 2 such
    // instructions,
    // so overall bandwidth is 1.0.
    // 在最新的英特尔处理器上，VFMA / MULPS / ADDPS的带宽为0.5。
    // 但是对于MULPS / ADDPS，我们有两个依赖序列的指令，因此总带宽为1.0。
    if (computeBandwidth == 0.5) computeBandwidth = 1.0;
#endif
    return computeBandwidth;
  }

};

} // end namespace Eigen

#endif  // EIGEN_USE_THREADS
#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H
