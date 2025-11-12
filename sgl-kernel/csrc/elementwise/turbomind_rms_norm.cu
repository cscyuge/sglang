/* Adapted from https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/norm/rms_norm.cu */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "cub/block/block_reduce.cuh"
#include "utils.h"

// Array type for vectorized operations
template <typename T, int N>
struct Array {
  T data[N];

  __device__ __forceinline__ T& operator[](int i) {
    return data[i];
  }
  __device__ __forceinline__ const T& operator[](int i) const {
    return data[i];
  }
};

template<typename T>
struct plus {
    __device__ T operator()(T a, T b)
    {
        return a + b;
    }
};

template<typename T>
struct multiplies {
    __device__ T operator()(T a, T b)
    {
        return a * b;
    }
};

template<typename T, int N, typename Op>
inline __device__ Array<T, N> binary_op_vv(const Array<T, N>& a, const Array<T, N>& b, Op op)
{
    Array<T, N> c;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        c[i] = op(a[i], b[i]);
    }
    return c;
}

template<typename T, int N>
inline __device__ Array<T, N> operator+(const Array<T, N>& a, const Array<T, N>& b)
{
    return binary_op_vv(a, b, plus<T>{});
}

template<typename T, int N>
inline __device__ Array<T, N> operator*(const Array<T, N>& a, const Array<T, N>& b)
{
    return binary_op_vv(a, b, multiplies<T>{});
}

template<typename To, typename From, int N>
inline __device__ Array<To, N> cast(const Array<From, N>& src)
{
    Array<To, N> dst;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = (To)src[i];
    }
    return dst;
}

// Vectorized load
template <typename T, int N>
inline __device__ void Load(Array<T, N>& dst, const T* src) {
  if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
    (uint4&)dst = *(const uint4*)src;
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
    (uint2&)dst = *(const uint2*)src;
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
    (uint&)dst = *(const uint*)src;
  } else if constexpr (sizeof(Array<T, N>) % sizeof(uint4) == 0) {  //  uncoalesced
    constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
#pragma unroll
    for (int i = 0; i < M; ++i) {
      *((uint4*)&dst + i) = *((uint4*)src + i);
    }
  } else {
    static_assert(!std::is_same_v<T, T>);
  }
}

// Vectorized load with __ldg
template <typename T, int N>
inline __device__ void Ldg(Array<T, N>& dst, const T* src) {
  static_assert(sizeof(Array<T, N>) <= sizeof(uint4));

  if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
    (uint4&)dst = __ldg((const uint4*)src);
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
    (uint2&)dst = __ldg((const uint2*)src);
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
    (uint&)dst = __ldg((const uint*)src);
  } else {
    static_assert(!std::is_same_v<T, T>);
  }
}

// Vectorized store
template <typename T, int N>
inline __device__ void Store(T* __restrict__ dst, const Array<T, N>& src) {
  if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
    *(uint4*)dst = (const uint4&)src;
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
    *(uint2*)dst = (const uint2&)src;
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
    *(uint*)dst = (const uint&)src;
  } else if constexpr (sizeof(Array<T, N>) % sizeof(uint4) == 0) {  //  uncoalesced
    constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
#pragma unroll
    for (int i = 0; i < M; ++i) {
      *((uint4*)dst + i) = *((uint4*)&src + i);
    }
  } else {
    static_assert(!std::is_same_v<T, T>);
  }
}

template <class T, int vec_size>
__global__ void RMSNorm(
    T* data,  //
    int ld,
    const T* weight,
    int dim,
    int n,
    int token_num,
    float eps,
    float inv_dim) {
  // vec_size = 16/2 = 8 for float16, 4 for float32
  constexpr int thr_per_qk = 128 / vec_size;

  const int bi = (threadIdx.x + blockIdx.x * blockDim.x) / thr_per_qk;
  const int di = threadIdx.x % thr_per_qk * vec_size;
  const int ti = bi / n;  // token
  const int hi = bi % n;  // head

  if (bi >= token_num * n) {
    return;
  }

  data += ti * ld + hi * dim;

  Array<T, vec_size> vec{};
  if (di < dim) {
    Load(vec, &data[di]);
  }

  float acc[vec_size];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    acc[i] = static_cast<float>(vec[i]);
    sum += acc[i] * acc[i];
  }

  // Reduce sum within warp
  for (int mask = thr_per_qk / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync((uint32_t)-1, sum, mask);
  }

  float rms = rsqrtf(sum * inv_dim + eps);

  if (di < dim) {
    Array<T, vec_size> w{};
    Ldg(w, &weight[di]);

#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      vec[i] = static_cast<T>(acc[i] * rms * static_cast<float>(w[i]));
    }

    // Store back efficiently
    Store(&data[di], vec);
  }
}

template <class T, class Accum, int block_dim, int vec_size>
__global__ void RMSNorm_v0(
    T*       dst,
    int      dst_ld,
    const T* src,
    int      src_ld,
    const T* __restrict__ weights,
    int   dims,
    int   num,
    float eps,
    float inv_dims) 
{
    const int ti = blockIdx.x;
    const int di = threadIdx.x * vec_size;

    if (ti >= num) {
        return;
    }

    src += src_ld * ti;

    Array<Accum, vec_size> accum{};
    Array<T, vec_size>     vec;

    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(vec, &src[i]);
        Array<Accum, vec_size> tmp = cast<Accum>(vec);
        accum = accum + tmp * tmp;
    }

    float sum{};
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
        sum += accum[i];
    }

    using BlockReduce = cub::BlockReduce<Accum, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    sum = BlockReduce{temp_storage}.Sum(sum);

    __shared__ float shared_sum;

    if (threadIdx.x == 0) {
        shared_sum = rsqrtf(sum * inv_dims + eps);
    }

    __syncthreads();

    sum = shared_sum;

    dst += dst_ld * ti;

    Array<T, vec_size> sv;
    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(vec, &src[i]);
        Ldg(sv, &weights[i]);
#pragma unroll
        for (int c = 0; c < vec_size; ++c) {
            vec[c] = (T)((float)vec[c] * sum) * sv[c];
        }
        Store(&dst[i], vec);
    }
}

void turbomind_rms_norm(
    torch::Tensor& data,
    const torch::Tensor& weight,
    double eps,
    int64_t token_num,
    int64_t head_num,
    int64_t head_dim,
    int64_t stride) {
  TORCH_CHECK(head_dim <= 128, "head_dim must be <= 128");

  constexpr int vec_size = sizeof(uint4) / sizeof(float);  // 16/4 = 4 for float32
  constexpr int thr_per_qk = 128 / vec_size;

  TORCH_CHECK(head_dim % vec_size == 0, "head_dim must be divisible by vec_size");

  const int threads = token_num * head_num * thr_per_qk;
  const int block_dim = 512;
  const int grid_dim = (threads + block_dim - 1) / block_dim;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(data.scalar_type(), c_type, [&] {
    // Determine vec_size based on data type
    constexpr int actual_vec_size = sizeof(uint4) / sizeof(c_type);
    if (actual_vec_size == 4) {  // float32
      RMSNorm<c_type, 4><<<grid_dim, block_dim, 0, stream>>>(
          static_cast<c_type*>(data.data_ptr()),
          stride,
          static_cast<const c_type*>(weight.data_ptr()),
          head_dim,
          head_num,
          token_num,
          eps,
          1.f / head_dim);
    } else if (actual_vec_size == 8) {  // float16, bfloat16
      RMSNorm<c_type, 8><<<grid_dim, block_dim, 0, stream>>>(
          static_cast<c_type*>(data.data_ptr()),
          stride,
          static_cast<const c_type*>(weight.data_ptr()),
          head_dim,
          head_num,
          token_num,
          eps,
          1.f / head_dim);
    }
    return true;
  });
}

void turbomind_rms_norm_v0(
    torch::Tensor& out,
    const torch::Tensor& x,
    const torch::Tensor& w,
    double eps) {
    if (x.numel() == 0) {
        return;
    }
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == x.size(1));
    TORCH_CHECK(w.size(-1) == x.size(-1));

    const auto num = x.size(0);
    const auto dim = x.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(x.scalar_type(), c_type, [&] {
        constexpr int vec_size = 16 / sizeof(c_type);
        constexpr int threads = 512;
        const int blocks = num;

        RMSNorm_v0<c_type, float, threads, vec_size><<<blocks, threads, 0, stream>>>(
            static_cast<c_type*>(out.data_ptr()),
            out.stride(0),
            static_cast<const c_type*>(x.data_ptr()),
            x.stride(0),
            static_cast<const c_type*>(w.data_ptr()),
            dim,
            num,
            eps,
            1.f / dim);
        return true;
    });
}
