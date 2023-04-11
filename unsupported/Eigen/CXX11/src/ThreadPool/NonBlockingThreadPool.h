// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
#define EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H

#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Environment>
class ThreadPoolTempl : public Eigen::ThreadPoolInterface {
  //定义了一个名为ThreadPoolTempl的类，该类是从Eigen::ThreadPoolInterface类派生而来的，因此ThreadPoolTempl类将继承Eigen::ThreadPoolInterface类的所有成员和方法。
 public:
  typedef typename Environment::Task Task;
  //定义了一个类型别名，将Environment::Task定义为Task，Environment是一个模板参数，Task是Environment中的一个嵌套类型（也称为成员类型）.
  typedef RunQueue<Task, 1024> Queue;
  //定义了一个类型别名，将RunQueue<Task, 1024>定义为Queue，RunQueue是一个模板类，实现是一个循环队列，用于存储Task类型的任务，并且队列大小为1024。

  ThreadPoolTempl(int num_threads, Environment env = Environment())
      : ThreadPoolTempl(num_threads, true, env) {}
  //前半段是一个构造函数,num_threads线程池的线程数量 第二个参数为一个类型为 Environment 的对象，参数是可选的，如果不传递参数，则使用默认构造函数 Environment() 来创建一个默认对象。
  //：表示在构造函数初始化列表中调用另一个ThreadPoolTempl类型的构造函数,true 表示使用默认的队列类型（即 FIFO 队列）

  ThreadPoolTempl(int num_threads, bool allow_spinning,
                  Environment env = Environment())
      : env_(env),
        num_threads_(num_threads),
        allow_spinning_(allow_spinning),//  allow_spinning_表示是否允许线程自旋等待任务
        thread_data_(num_threads),//是一个std::vector类型的对象，用于存储每个线程的数据结构，它的大小是num_threads，表示线程池中有num_threads个线程。
        all_coprimes_(num_threads),//是一个std::vector类型的对象，用于存储每个线程的互质数，它的大小也是num_threads。
        waiters_(num_threads),//是一个std::vector类型的对象，用于存储每个线程的等待队列，它的大小也是num_threads。
        global_steal_partition_(EncodePartition(0, num_threads_)),//是一个整数，它表示线程池中所有线程的共同负载范围，初始值为0到num_threads
        blocked_(0),//表示当前被阻塞的线程数，初始值为0
        spinning_(0),//表示当前自旋等待的线程数，初始值为0
        done_(false),//线程池是否已经完成所有任务，它的初始值为false。
        cancelled_(false),//线程池是否已经被取消，它的初始值也为false。
        ec_(waiters_) {////创建一个线程池对象并初始化它的成员变量。
    waiters_.resize(num_threads_);//调整waiters_的大小为num_threads_，其中waiters_是一个存储互斥量和条件变量的向量，用于等待线程池中的任务完成。
    // Calculate coprimes of all numbers [1, num_threads].
    // Coprimes are used for random walks over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a random starting thread index t and calculate num_threads - 1 subsequent
    // indices as (t + coprime) % num_threads, we will cover all threads without
    // repetitions (effectively getting a presudo-random permutation of thread
    // indices).
    eigen_plain_assert(num_threads_ < kMaxThreads);//断言，如果条件为假，程序将以错误信息终止。是一种安全措施，以保证不会使用过多线程，从而引发潜在的问题，例如内存溢出或者系统崩溃。
    for (int i = 1; i <= num_threads_; ++i) {//使用容器 all_coprimes_ 和函数 ComputeCoprimes 计算每次迭代的互质值。
      all_coprimes_.emplace_back(i);//将当前值 i 追加到 all_coprimes_ 容器的末尾. emplace_back 方法在不需要复制或移动对象的情况下，在容器的末尾构造一个对象。
      ComputeCoprimes(i, &all_copr imes_.back());//将当前值 i 和指向 all_coprimes_ 容器最后一个元素的指针作为参数传递。
      //ComputeCoprimes函数的目的是计算给定整数的互质值，并将它们附加到由第二个参数指向的容器中。
    }
#ifndef EIGEN_THREAD_LOCAL
    init_barrier_.reset(new Barrier(num_threads_));
    // 如果未定义EIGEN_THREAD_LOCAL，则创建一个Barrier对象并将其初始化。Barrier对象用于等待所有线程都完成初始化之后再开始执行任务。
#endif
    thread_data_.resize(num_threads_);//调整thread_data_的大小
    for (int i = 0; i < num_threads_; i++) {
      SetStealPartition(i, EncodePartition(0, num_threads_));//设置每个线程的任务分区。将所有任务分为num_threads_个连续部分，将第 i 个线程的任务分配给第 i 个分区。
      thread_data_[i].thread.reset(
          env_.CreateThread([this, i]() { WorkerLoop(i); }));
      //thread_data_[i].thread 是一个 std::unique_ptr，指向第 i 个线程的实例。
      //env_.CreateThread 创建一个新线程，并将其包装在一个 std::unique_ptr 中。这个新线程的函数是 WorkerLoop(i)，其中 i 是线程的索引，用于指示该线程的任务分区。
      //每个线程启动后，将开始执行 WorkerLoop 函数，该函数是一个无限循环，它会不断从线程的任务分区中获取任务，并执行这些任务，直到所有任务都已完成。该函数将保持线程处于活动状态，直到整个计算完成。
    }
    //创建一个并行计算框架，并为每个线程分配任务，任务分配给多个线程加速计算。每个线程都可以独立获取和执行任务，不需要与其他线程进行同步或者协调。
#ifndef EIGEN_THREAD_LOCAL
    // Wait for workers to initialize per_thread_map_. Otherwise we might race
    // with them in Schedule or CurrentThreadId.
    init_barrier_->Wait();
    //如果未定义EIGEN_THREAD_LOCAL，则等待所有线程都完成初始化之后再执行下一步。这一步是为了确保线程池中的所有线程都已经完成初始化，以免在调度或获取当前线程ID时发生竞争。
#endif
  }

  ~ThreadPoolTempl() {
    //一个线程池的析构函数。它会在线程池对象被销毁时自动调用。
    done_ = true;
    //设置一个线程池对象的状态，表示线程池不再接受新的任务，即任务队列已经关闭。

    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    //尽管线程池已经关闭，线程仍然可以继续工作，直到它们完成了它们的任务，或者等待新任务的到来。
    if (!cancelled_) {
      //检查线程池是否被取消，如果没有被取消，它会通过 ec_.Notify(true) 来通知所有等待中的线程继续执行。
      ec_.Notify(true);
    } else {
      // Since we were cancelled, there might be entries in the queues.
      // Empty them to prevent their destructor from asserting.
      //如果线程池被取消，那么任务队列中可能还有一些任务，这些任务不应该继续执行，因为它们可能依赖于线程池的状态。
      for (size_t i = 0; i < thread_data_.size(); i++) {
        thread_data_[i].queue.Flush();
        //遍历所有线程数据结构，并调用 Flush() 方法来清空它们的任务队列。
      }
    }
    // Join threads explicitly (by destroying) to avoid destruction order within
    // this class.
    for (size_t i = 0; i < thread_data_.size(); ++i)
      thread_data_[i].thread.reset();
    //使用 reset() 方法将线程数据结构中的线程对象设置为 nullptr，以便它们可以被销毁。
  }

  void SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions) {
    //函数定义，函数名为 SetStealPartitions，参数为std::vector 类型的 std::pair<unsigned, unsigned> 对象，表示一组无序的起始和结束位置。
    eigen_plain_assert(partitions.size() == static_cast<std::size_t>(num_threads_));
    //断言宏，用于确保传递给函数的 partitions 参数的大小等于 num_threads_
    // Pass this information to each thread queue.
    for (int i = 0; i < num_threads_; i++) {
      const auto& pair = partitions[i];
      //定义了一个常量引用 pair，绑定了 partitions 中第 i 个元素。该元素是一个 std::pair<unsigned, unsigned> 对象，表示一个任务子集的起始和结束位置。
      unsigned start = pair.first, end = pair.second;
      //接下来，将 pair 中的起始和结束位置分别存储到变量 start 和 end 中。
      AssertBounds(start, end);
      //该函数用于确保任务子集的起始和结束位置是有效的。
      unsigned val = EncodePartition(start, end);
      //将起始和结束位置编码为一个无符号整数 val，这个值将用于在线程之间传递任务信息。
      SetStealPartition(i, val);
      //将 val 值传递给当前正在处理的线程。
    }
    //该循环的目的是遍历 partitions 容器中的每一个元素，并为每个线程分配一个任务的子集。
  }

  void Schedule(std::function<void()> fn) EIGEN_OVERRIDE {
    ScheduleWithHint(std::move(fn), 0, num_threads_);
    //调用下面的函数
  }

  void ScheduleWithHint(std::function<void()> fn, int start,
                        int limit) override {
    Task t = env_.CreateTask(std::move(fn));
    //使用env_.CreateTask创建一个新的任务t，并且将其绑定到要执行的函数 fn 上
    PerThread* pt = GetPerThread();
    //获取当前线程的特定数据，检查线程是否属于当前线程池。
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue& q = thread_data_[pt->thread_id].queue;
      t = q.PushFront(std::move(t));
      //将任务添加到该线程的任务队列的前端，以实现优先级调度（q.PushFront(std::move(t))）
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      //将任务添加到随机选择的线程队列的末尾，以实现负载均衡（q.PushBack(std::move(t))）
      eigen_plain_assert(start < limit);
      eigen_plain_assert(limit <= num_threads_);
      //数据检验 确保处于有效范围
      int num_queues = limit - start;
      //计算从线程池的第start个线程到第limit-1个线程之家你可以选择的队列数
      int rnd = Rand(&pt->rand) % num_queues;
      //生成一个随机数Rand 用于确定哪个线程队列接受该任务。
      eigen_plain_assert(start + rnd < limit);
      //检查选定的队列索引是否处于有效范围内
      Queue& q = thread_data_[start + rnd].queue;
      t = q.PushBack(std::move(t));
      //将任务添加到该队列的末尾
    }
    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (!t.f) {
      //检查任务是否成功添加到队列中
      ec_.Notify(false);
      //没成功，停止等待该任务的线程条件变量，以便它们继续执行其他任务
    } else {
      env_.ExecuteTask(t);  // Push failed, execute directly.
      //继续执行该任务
    }
  }
//该方法接受一个类型为std::function的参数fn，表示需要在线程池中执行的函数，以及两个整数参数，用于指定将任务添加到的线程队列的子集。

  void Cancel() EIGEN_OVERRIDE {
    cancelled_ = true;//通知所有线程任务已经被取消
    done_ = true;//通知其他线程不再继续等待任务

    // Let each thread know it's been cancelled.
#ifdef EIGEN_THREAD_ENV_SUPPORTS_CANCELLATION
    for (size_t i = 0; i < thread_data_.size(); i++) {
      thread_data_[i].thread->OnCancel();
      //遍历所有的线程，调用thread->OnCancel方法，通知线程任务已经被取消
    }
#endif
//实现了线程池的取消功能 

    // Wake up the threads without work to let them exit on their own.
    ec_.Notify(true);//唤醒所有没有任务的线程，使它们可以退出线程池。
  }

  int NumThreads() const EIGEN_FINAL { return num_threads_; }
  //成员函数,用于获取线程池中的线程数量

  int CurrentThreadId() const EIGEN_FINAL {
    const PerThread* pt = const_cast<ThreadPoolTempl*>(this)->GetPerThread();
    //调用GetPerThread()方法获取当前线程的PerThread对象
    if (pt->pool == this) {
      //如果当前线程的pool指针指向这个ThreadPoolTempl对象
      return pt->thread_id;
    } else {
      return -1;
    }
  }
  //成员函数,用于获取线程池中的线程ID

 private:
  // Create a single atomic<int> that encodes start and limit information for
  // each thread.
  // We expect num_threads_ < 65536, so we can store them in a single
  // std::atomic<unsigned>.
  // Exposed publicly as static functions so that external callers can reuse
  // this encode/decode logic for maintaining their own thread-safe copies of
  // scheduling and steal domain(s).
  //创建一个包含起始和结束信息的std::atomic<int>,可以同时存储每个线程的起始和结束信息。
  //公开为静态函数，以便外部调用者可以重用这些编码/解码逻辑来维护自己的线程安全的调度和抢占域。
  static const int kMaxPartitionBits = 16;
  static const int kMaxThreads = 1 << kMaxPartitionBits;
  //表示最大分区位数和最大线程数的常量。其中，kMaxPartitionBits为16，表示线程池支持的最大线程数为2^16，即65536；
  //kMaxThreads则根据kMaxPartitionBits计算而来，即1左移kMaxPartitionBits

  inline unsigned EncodePartition(unsigned start, unsigned limit) {
    return (start << kMaxPartitionBits) | limit;
    //将一个任务分区的起始位置start和终止位置limit编码成一个无符号整数，以便存储在std::atomic<unsigned>中。
    //具体来说，该函数将start左移kMaxPartitionBits位，然后与limit进行按位或运算，返回结果。
  }

  inline void DecodePartition(unsigned val, unsigned* start, unsigned* limit) {
    *limit = val & (kMaxThreads - 1);
    //将val与kMaxThreads-1进行按位与运算，得到limit
    val >>= kMaxPartitionBits;
    *start = val;
    //将val右移kMaxPartitionBits位，得到start
  }
   //将一个无符号整数val解码成一个任务分区的起始位置start和终止位置limit。

  void AssertBounds(int start, int end) {
    eigen_plain_assert(start >= 0);
    eigen_plain_assert(start < end);  // non-zero sized partition
    eigen_plain_assert(end <= num_threads_);
  }
  // 检查起始和结束位置是否合法

  inline void SetStealPartition(size_t i, unsigned val) {
    thread_data_[i].steal_partition.store(val, std::memory_order_relaxed);
  }
  // 设置线程 i 的偷取分区

  inline unsigned GetStealPartition(int i) {
    return thread_data_[i].steal_partition.load(std::memory_order_relaxed);
  }
  // 获取线程 i 的偷取分区

  void ComputeCoprimes(int N, MaxSizeVector<unsigned>* coprimes) {
    for (int i = 1; i <= N; i++) {
      unsigned a = i;
      unsigned b = N;
      // If GCD(a, b) == 1, then a and b are coprimes.
      while (b != 0) {
        unsigned tmp = a;
        a = b;
        b = tmp % b;
      }
      if (a == 1) {
        coprimes->push_back(i);
      }
    }
  }
  // 计算 N 的所有互质数

  typedef typename Environment::EnvThread Thread;//Thread类型是Environment::EnvThread的别名，它定义了一个平台特定的线程类型。

  struct PerThread {
    constexpr PerThread() : pool(NULL), rand(0), thread_id(-1) {}
    ThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    uint64_t rand;          // Random generator state.
    int thread_id;          // Worker thread index in pool. 该线程在池中的索引号。
#ifndef EIGEN_THREAD_LOCAL
    // Prevent false sharing.
    char pad_[128];
    //防止 false sharing，这是一种多线程编程时的优化技术。
    //False sharing 指的是多个线程同时访问同一缓存行，由于缓存一致性协议的限制，每个线程在修改缓存行时需要将其它线程的缓存行都失效，导致性能下降。
    //为了避免 false sharing，可以在结构体的末尾添加一些填充字段，使得每个结构体占用的空间大小是缓存行的整数倍。
    //在这里，作者添加了 128 字节的填充字段 pad_，使得 PerThread 结构体占用的空间大小为 128 字节的整数倍。
#endif
  };

  struct ThreadData {
    constexpr ThreadData() : thread(), steal_partition(0), queue() {}
    std::unique_ptr<Thread> thread;//Thread对象的unique_ptr
    std::atomic<unsigned> steal_partition;//unsigned类型的原子变量steal_partition
    Queue queue;//Queue是一个模板类型，是一个线程安全的队列，用于在工作线程之间传递任务。
  };

  Environment env_;
  const int num_threads_;
  const bool allow_spinning_;
  MaxSizeVector<ThreadData> thread_data_;
  MaxSizeVector<MaxSizeVector<unsigned>> all_coprimes_;
  MaxSizeVector<EventCount::Waiter> waiters_;
  unsigned global_steal_partition_;
  std::atomic<unsigned> blocked_;
  std::atomic<bool> spinning_;
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;
  EventCount ec_;
#ifndef EIGEN_THREAD_LOCAL
//在不支持线程本地存储的情况下使用。
  std::unique_ptr<Barrier> init_barrier_;
  std::mutex per_thread_map_mutex_;  // Protects per_thread_map_.
  std::unordered_map<uint64_t, std::unique_ptr<PerThread>> per_thread_map_;
#endif0
//并行线程池的类定义

  // Main worker thread loop.
  void WorkerLoop(int thread_id) {
#ifndef EIGEN_THREAD_LOCAL
// 创建新的PerThread实例，并插入到per_thread_map_中
    std::unique_ptr<PerThread> new_pt(new PerThread());
    per_thread_map_mutex_.lock();
    bool insertOK = per_thread_map_.emplace(GlobalThreadIdHash(), std::move(new_pt)).second;
    eigen_plain_assert(insertOK);
    EIGEN_UNUSED_VARIABLE(insertOK);
    per_thread_map_mutex_.unlock();
    init_barrier_->Notify();
    init_barrier_->Wait();
#endif
// 获取当前线程的 PerThread 结构体指针
    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->rand = GlobalThreadIdHash();
    pt->thread_id = thread_id;
    // 获取当前线程对应的 Queue
    Queue& q = thread_data_[thread_id].queue;
    // 获取当前线程对应的 Waiter
    EventCount::Waiter* waiter = &waiters_[thread_id];
    // TODO(dvyukov,rmlarsen): The time spent in NonEmptyQueueIndex() is
    // proportional to num_threads_ and we assume that new work is scheduled at
    // a constant rate, so we set spin_count to 5000 / num_threads_. The
    // constant was picked based on a fair dice roll, tune it.
    /*
    这段代码是一个待办事项，意思是：由于 NonEmptyQueueIndex() 函数的运行时间与线程数成正比，我们假设新的任务以一个恒定的速率被调度，
    因此我们将 spin_count 设置为 5000 / num_threads_。这个常数是根据一个公正的骰子掷骰子得出的，需要进行调整。
    */
    const int spin_count =
        allow_spinning_ && num_threads_ > 0 ? 5000 / num_threads_ : 0;
    if (num_threads_ == 1) {
      // For num_threads_ == 1 there is no point in going through the expensive
      // steal loop. Moreover, since NonEmptyQueueIndex() calls PopBack() on the
      // victim queues it might reverse the order in which ops are executed
      // compared to the order in which they are scheduled, which tends to be
      // counter-productive for the types of I/O workloads the single thread
      // pools tend to be used for.
      /*
      当num_threads_ == 1时，经过昂贵的偷窃循环是没有意义的。此外，由于NonEmptyQueueIndex()在受害者队列上调用PopBack()，
      可能会反转操作执行的顺序，与安排操作的顺序相比，这往往是有害的，因为单线程池通常用于I/O工作负载的类型。
      */
      // 如果 num_threads_ 等于 1，就不需要偷取其他线程的任务了，直接从自己的队列中取任务
      while (!cancelled_) {
        Task t = q.PopFront();
        for (int i = 0; i < spin_count && !t.f; i++) {
          // 如果当前队列中没有任务，就自旋等待 spin_count 次，直到队列非空或者 cancelled_ 标志被设置
          if (!cancelled_.load(std::memory_order_relaxed)) {
            t = q.PopFront();
          }
        }
        if (!t.f) {
          // 如果队列中还是没有任务，则阻塞等待，直到有任务到来或者 cancelled_ 标志被设置
          if (!WaitForWork(waiter, &t)) {
            return;
          }
        }
        if (t.f) {
        // 如果有任务，就执行任务
          env_.ExecuteTask(t);
        }
      }
    } else {
      // 如果 num_threads_ 大于 1，就需要偷取其他线程的任务了
      while (!cancelled_) {
        // 先从自己的队列中取任务
        Task t = q.PopFront();
        if (!t.f) {
          // 如果自己的队列中没有任务，就从其他线程的队列中偷取任务
          t = LocalSteal();
          if (!t.f) {
            t = GlobalSteal();
            if (!t.f) {
              // Leave one thread spinning. This reduces latency.
              // 如果所有线程的队列中都没有任务，则有一个线程自旋等待
              if (allow_spinning_ && !spinning_ && !spinning_.exchange(true)) {
                for (int i = 0; i < spin_count && !t.f; i++) {
                  if (!cancelled_.load(std::memory_order_relaxed)) {
                    t = GlobalSteal();
                  } else {
                    return;
                  }
                }
                spinning_ = false;
              }
              if (!t.f) {
                 // 如果仍然没有任务，就阻塞等待，直到有任务到来或者 cancelled_ 标志
                if (!WaitForWork(waiter, &t)) {
                  return;
                }
              }
            }
          }
        }
        if (t.f) {
          //执行任务
          env_.ExecuteTask(t);
        }
      }
    }
  }

  // Steal tries to steal work from other worker threads in the range [start,
  // limit) in best-effort manner.
  // Steal函数尝试从范围[start, limit)内的其他工作线程中窃取工作。
  Task Steal(unsigned start, unsigned limit) {
    PerThread* pt = GetPerThread();
    const size_t size = limit - start;
    // 生成一个随机数r，用于选择被窃取的队列
    unsigned r = Rand(&pt->rand);
    // Reduce r into [0, size) range, this utilizes trick from
    // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    //判断 all_coprimes_[size - 1] 的大小是否小于 2^30，如果超过了这个值就会导致 unsigned 类型溢出
    //all_coprimes_ 是一个保存着 size - 1 的所有互质数的数组，它用于将随机数 r 转换成一个在 [0, size) 范围内的值。
    //这个数组中的元素个数是固定的，因此可以预先计算并保存在数组中，而不需要每次都重新计算。
    eigen_plain_assert(all_coprimes_[size - 1].size() < (1<<30));
    //随后，将随机数 r 乘以 size 后再右移 32 位，得到在 [0, size) 范围内的一个窃取目标。这里使用了一个技巧，将乘法运算转换成位运算，这样可以减少计算开销。
    unsigned victim = ((uint64_t)r * (uint64_t)size) >> 32;
    //接着，将随机数 r 乘以 all_coprimes_[size - 1].size() 后再右移 32 位，得到一个索引值 index。
    unsigned index = ((uint64_t) all_coprimes_[size - 1].size() * (uint64_t)r) >> 32;
    //从 all_coprimes_[size - 1] 数组中取出 index 对应的互质数作为步长 inc。这个步长用于在窃取时依次访问其他线程的工作队列，以提高窃取的效率。
    unsigned inc = all_coprimes_[size - 1][index];
    // 在[0, size)范围内尝试窃取工作，直到所有队列都尝试过一遍
    for (unsigned i = 0; i < size; i++) {
      // 确定窃取哪个队列
      eigen_plain_assert(start + victim < limit);
      // 从队列中取出任务
      Task t = thread_data_[start + victim].queue.PopBack();
      if (t.f) {
        // 如果窃取成功，返回任务
        return t;
      }
      // 尝试窃取下一个队列
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    // 如果没有窃取到任何任务，则返回一个空任务
    return Task();
  }

  // Steals work within threads belonging to the partition.
  //从所属分区的线程中窃取任务。
  Task LocalSteal() {
    //获取当前线程的 ID
    PerThread* pt = GetPerThread();
    //通过 GetStealPartition 函数获取线程所属的分区号。
    unsigned partition = GetStealPartition(pt->thread_id);
    // If thread steal partition is the same as global partition, there is no
    // need to go through the steal loop twice.
    //如果全局窃取分区与该线程所属的分区相同，就直接返回一个空的 Task，因为不需要做窃取操作。
    if (global_steal_partition_ == partition) return Task();
    unsigned start, limit;
    //通过 DecodePartition 函数解码该分区的起始和结束下标，并调用 Steal 函数进行窃取操作。
    DecodePartition(partition, &start, &limit);
    AssertBounds(start, limit);

    return Steal(start, limit);
  }

  // Steals work from any other thread in the pool.
  Task GlobalSteal() {
    return Steal(0, num_threads_);
  }


  // WaitForWork blocks until new work is available (returns true), or if it is
  // time to exit (returns false). Can optionally return a task to execute in t
  // (in such case t.f != nullptr on return).
  /*WaitForWork()方法会阻塞线程直到有新的任务可用。如果线程池已经关闭或者超时（需要执行参数timeout）则返回false。
  如果等待到新任务可用，则返回true，并且可以在参数t中返回一个任务。*/
  bool WaitForWork(EventCount::Waiter* waiter, Task* t) {
    eigen_plain_assert(!t->f);// t.f 应该为 nullpt
    // We already did best-effort emptiness check in Steal, so prepare for
    // blocking.
    ec_.Prewait();
    // Now do a reliable emptiness check.
    int victim = NonEmptyQueueIndex();
    if (victim != -1) {// 如果存在非空队列
      ec_.CancelWait();// 取消等待
      if (cancelled_) {// 如果取消标志为 true
        return false;// 返回 false，表示需要退出
      } else {
        *t = thread_data_[victim].queue.PopBack();// 取出最后一个任务
        return true;// 返回 true，表示存在新的任务
      }
    }
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    /*blocked_ 记录了当前阻塞的线程数，用于判断当前是否需要终止线程池。
    如果当前正在关闭线程池，并且所有线程都被阻塞而没有任务可执行，那么我们就可以认为线程池已经完成了所有任务的执行，可以进行终止操作。*/
    blocked_++;// 当前线程阻塞，增加阻塞线程数
    // TODO is blocked_ required to be unsigned?
    if (done_ && blocked_ == static_cast<unsigned>(num_threads_)) { // 如果已经完成，且所有线程都已阻塞
      ec_.CancelWait();// 取消等待
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (NonEmptyQueueIndex() != -1) { // 如果存在非空队列
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        /*解释为什么在重新检查队列之前不能从队列中弹出元素。
        假设在当前线程检查队列之前，队列中只有一个元素，如果当前线程首先从队列中弹出了这个元素，那么其他线程可能会退出，这可能会导致未执行的工作项。
        为了避免这种情况，我们必须首先检查队列是否为空，只有在确定队列为空之后才能弹出元素。这样可以确保所有工作线程在同一时间退出。
        */
        blocked_--;// 减少阻塞线程数
        return true;// 返回 true，表示存在新的任务
      }
      // Reached stable termination state.
      ec_.Notify(true);// 唤醒等待中的线程
      return false;// 返回 false，表示需要退出
    }
    ec_.CommitWait(waiter);/ 等待
    blocked_--;// 当前线程解除阻塞
    return true;// 返回 true，表示存在新的任务
  }

  /*函数 NonEmptyQueueIndex() 会尝试从所有线程的队列中随机选取一个非空队列，并返回其对应的线程编号，
  如果没有找到任何非空队列，则返回 -1。该函数的设计目的是避免线程在等待队列中的任务时陷入死循环。*/
  int NonEmptyQueueIndex() {
    PerThread* pt = GetPerThread();//获得当前线程的数据结构指针pt
    // We intentionally design NonEmptyQueueIndex to steal work from
    // anywhere in the queue so threads don't block in WaitForWork() forever
    // when all threads in their partition go to sleep. Steal is still local.
    /*我们有意地设计了 NonEmptyQueueIndex 函数从队列中的任何位置窃取工作项，
    这样线程在其分区中的所有线程都进入睡眠状态时不会永远阻塞在 WaitForWork() 函数中。这个窃取操作仍然是本地的。*/
    const size_t size = thread_data_.size();//获取线程数量size
    unsigned r = Rand(&pt->rand);//随机生成一个种子数r
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];//从all_coprimes_中选取与线程数量相关的质数inc
    unsigned victim = r % size;
    // 遍历所有队列，从随机的 victim 开始，每次增加一个步长 inc
    // 直到遍历完所有队列或者找到非空的队列为止
    for (unsigned i = 0; i < size; i++) {
      if (!thread_data_[victim].queue.Empty()) {
        //如果该线程的队列不为空
        return victim;//返回该线程的索引
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return -1;//最终没有找到非空队列，则返回-1，表示当前没有可用的工作项。
  }

/* GlobalThreadIdHash() 返回当前线程的 ID 的哈希值。该函数可能在选择要从哪个线程的队列中偷取任务时使用到。*/
  static EIGEN_STRONG_INLINE uint64_t GlobalThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }

  EIGEN_STRONG_INLINE PerThread* GetPerThread() {
#ifndef EIGEN_THREAD_LOCAL
   // 没有定义EIGEN_THREAD_LOCAL时，使用全局映射per_thread_map_查找当前线程的PerThread指针
    static PerThread dummy;// dummy变量用于返回默认PerThread指针
    auto it = per_thread_map_.find(GlobalThreadIdHash());
    if (it == per_thread_map_.end()) {
      return &dummy;
    } else {
      return it->second.get();
    }
#else
    // 定义了EIGEN_THREAD_LOCAL时，返回当前线程的局部变量指针
    EIGEN_THREAD_LOCAL PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
#endif
  }

  // 使用 PCG 生成伪随机数
  // 该函数使用了 PCG-XSH-RS 方案生成随机数
  // 生成的随机数是 unsigned 类型
  // 输入参数 state 是一个指向 uint64_t 的指针，用于存储和更新内部状态
  // 函数返回生成的随机数
  static EIGEN_STRONG_INLINE unsigned Rand(uint64_t* state) {
    uint64_t current = *state;
    // Update the internal state
    *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22)) >>
                                 (22 + (current >> 61)));
  }
};

//定义了一个ThreadPool类型，其模板参数为StlThreadEnvironment，即使用STL库实现的线程环境。使用typedef将其定义为ThreadPool。
typedef ThreadPoolTempl<StlThreadEnvironment> ThreadPool;

}  // namespace Eigen

#endif  // EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
