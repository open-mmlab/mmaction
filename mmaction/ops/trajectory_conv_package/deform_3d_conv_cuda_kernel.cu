#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>


using namespace at;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 512; // 1024;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__device__ scalar_t deformable_im2col_bilinear(const scalar_t* bottom_data, const int data_height, const int data_width,
    const int time, const int height, const int width, scalar_t t, scalar_t h, scalar_t w) {
    int t_low = floor(t);
    int h_low = floor(h);
    int w_low = floor(w);
    int t_high = t_low + 1;
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    scalar_t lt = t - t_low;
    scalar_t lh = h - h_low;
    scalar_t lw = w - w_low;
    scalar_t ht = 1 - lt, hh = 1 - lh, hw = 1 - lw;

    scalar_t v1 = 0;
    if (t_low >= 0 && h_low >= 0 && w_low >= 0)
      v1 = bottom_data[(t_low * data_height + h_low) * data_width + w_low];
    scalar_t v2 = 0;
    if (t_low >= 0 && h_low >= 0 && w_high <= width - 1)
      v2 = bottom_data[(t_low * data_height + h_low) * data_width + w_high];
    scalar_t v3 = 0;
    if (t_low >= 0 && h_high <= height - 1 && w_low >= 0)
      v3 = bottom_data[(t_low * data_height + h_high) * data_width + w_low];
    scalar_t v4 = 0;
    if (t_low >= 0 && h_high <= height - 1 && w_high <= width - 1)
      v4 = bottom_data[(t_low * data_height + h_high) * data_width + w_high];
    scalar_t v5 = 0;
    if (t_high <= time - 1 && h_low >= 0 && w_low >= 0)
      v5 = bottom_data[(t_high * data_height + h_low) * data_width + w_low];
    scalar_t v6 = 0;
    if (t_high <= time - 1 && h_low >= 0 && w_high <= width - 1)
      v6 = bottom_data[(t_high * data_height + h_low) * data_width + w_high];
    scalar_t v7 = 0;
    if (t_high <= time - 1 && h_high <= height - 1 && w_low >= 0)
      v7 = bottom_data[(t_high * data_height + h_high) * data_width + w_low];
    scalar_t v8 = 0;
    if (t_high <= time - 1 && h_high <= height - 1 && w_high <= width - 1)
      v8 = bottom_data[(t_high * data_height + h_high) * data_width + w_high];

    scalar_t w1 = ht * hh * hw, w2 = ht * hh * lw, w3 = ht * lh * hw, w4 = ht * lh * lw;
    scalar_t w5 = lt * hh * hw, w6 = lt * hh * lw, w7 = lt * lh * hw, w8 = lt * lh * lw;

    scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
    return val;
}


template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t argmax_t, scalar_t argmax_h, scalar_t argmax_w,
    const int t, const int h, const int w, const int time, const int height, const int width) {
    if (argmax_t <= -1 || argmax_t >= time || argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
      return 0;
    }

    int argmax_t_low = floor(argmax_t);
    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_t_high = argmax_t_low + 1;
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;
    
    scalar_t weight = 0;
    if (t == argmax_t_low) {
      if (h == argmax_h_low) {
        if (w == argmax_w_low) {
          weight = (t + 1 - argmax_t) * (h + 1 - argmax_h) * (w + 1 - argmax_w);
        } else if (w == argmax_w_high) {
          weight = (t + 1 - argmax_t) * (h + 1 - argmax_h) * (argmax_w + 1 - w);
        }
      }
      else if (h == argmax_h_high) {
        if (w == argmax_w_low) {
          weight = (t + 1 - argmax_t) * (argmax_h + 1 - h) * (w + 1 - argmax_w);
        } else if (w == argmax_w_high) {
          weight = (t + 1 - argmax_t) * (argmax_h + 1 - h) * (argmax_w + 1 - w);
        }
      }
    } else if (t == argmax_t_high) {
      if (h == argmax_h_low) {
        if (w == argmax_w_low) {
          weight = (argmax_t + 1 - t) * (h + 1 - argmax_h) * (w + 1 - argmax_w);
        } else if (w == argmax_w_high) {
          weight = (argmax_t + 1 - t) * (h + 1 - argmax_h) * (argmax_w + 1 - w);
        }
      }
      else if (h == argmax_h_high) {
        if (w == argmax_w_low) {
          weight = (argmax_t + 1 - t) * (argmax_h + 1 - h) * (w + 1 - argmax_w);
        } else if (w == argmax_w_high) {
          weight = (argmax_t + 1 - t) * (argmax_h + 1 - h) * (argmax_w + 1 - w);
        }
      }
    }
    return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(scalar_t argmax_t, scalar_t argmax_h, scalar_t argmax_w,
    const int time, const int height, const int width, const scalar_t* im_data,
    const int data_height, const int data_width, const int bp_dir) {
    if (argmax_t <= -1 || argmax_t >= time || argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
      return 0;
    }


    int argmax_t_low = floor(argmax_t);
    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_t_high = argmax_t_low + 1;
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;
    
    scalar_t weight = 0;

    if (bp_dir == 0) {
      if (argmax_t_low >= 0 && argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_t_high - argmax_t) * (argmax_w_high - argmax_w) * im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_low];
      if (argmax_t_low >= 0 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += -1 * (argmax_t_high - argmax_t) * (argmax_w - argmax_w_low) * im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_high];
      if (argmax_t_low >= 0 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += (argmax_t_high - argmax_t) * (argmax_w_high - argmax_w) * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_low];
      if (argmax_t_low >= 0 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_t_high - argmax_t) * (argmax_w - argmax_w_low) * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_high];
      if (argmax_t_high <= time - 1 && argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_t - argmax_t_low) * (argmax_w_high - argmax_w) * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_low];
      if (argmax_t_high <= time - 1 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += -1 * (argmax_t - argmax_t_low) * (argmax_w - argmax_w_low) * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_high];
      if (argmax_t_high <= time - 1 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += (argmax_t - argmax_t_low) * (argmax_w_high - argmax_w) * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_low];
      if (argmax_t_high <= time - 1 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_t - argmax_t_low) * (argmax_w - argmax_w_low) * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_high];
    } else if (bp_dir == 1) {
      if (argmax_t_low >= 0 && argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_t_high - argmax_t) * (argmax_h_high - argmax_h) * im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_low];
      if (argmax_t_low >= 0 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += (argmax_t_high - argmax_t) * (argmax_h_high - argmax_h) * im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_high];
      if (argmax_t_low >= 0 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += -1 * (argmax_t_high - argmax_t) * (argmax_h - argmax_h_low) * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_low];
      if (argmax_t_low >= 0 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_t_high - argmax_t) * (argmax_h - argmax_h_low) * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_high];
      if (argmax_t_high <= time - 1 && argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_t - argmax_t_low) * (argmax_h_high - argmax_h) * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_low];
      if (argmax_t_high <= time - 1 && argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += (argmax_t - argmax_t_low) * (argmax_h_high - argmax_h) * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_high];
      if (argmax_t_high <= time - 1 && argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += -1 * (argmax_t - argmax_t_low) * (argmax_h - argmax_h_low) * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_low];
      if (argmax_t_high <= time - 1 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_t - argmax_t_low) * (argmax_h - argmax_h_low) * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_high];
    }

    return weight;
}


template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(const int n, const scalar_t* data_im,
    const scalar_t* data_offset, const int time, const int height, const int width,
    const int kernel_t, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int time_col, const int height_col, const int width_col,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int t_col = (index / width_col / height_col) % time_col;
    const int b_col = (index / width_col / height_col / time_col) % batch_size;
    const int c_im = (index / width_col / height_col / time_col) / batch_size;
    const int c_col = c_im * kernel_t * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int t_in = t_col * stride_t - pad_t;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    scalar_t* data_col_ptr = data_col;
    data_col_ptr += ( ( (c_col * batch_size + b_col) * time_col + t_col) * height_col + h_col) * width_col + w_col;
    const scalar_t* data_im_ptr = data_im;
    data_im_ptr += (b_col * num_channels + c_im) * time * height * width;
    const scalar_t* data_offset_ptr = data_offset;
    data_offset_ptr += (b_col * deformable_group + deformable_group_index) * 2 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;
    for (int k = 0; k < kernel_t; ++k) {
      for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
          const int data_offset_h_ptr = ( ( (2 * ((k * kernel_h + i) * kernel_w + j) + 0) * time_col + t_col) *
                                         height_col + h_col) * width_col + w_col;
          const int data_offset_w_ptr = ( ( (2 * ((k * kernel_h + i) * kernel_w + j) + 1) * time_col + t_col) *
                                         height_col + h_col) * width_col + w_col;
          const scalar_t offset_t = 0; 
          const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
          const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
          scalar_t val = static_cast<scalar_t>(0);
          const scalar_t t_im = t_in + k * dilation_t + offset_t;
          const scalar_t h_im = h_in + i * dilation_h + offset_h;
          const scalar_t w_im = w_in + j * dilation_w + offset_w;
          if (h_im > -1 && w_im > -1 && h_im < height && w_im < width && t_im > -1 && t_im < time) {
            val = deformable_im2col_bilinear(data_im_ptr, height, width, time, height, width, t_im, h_im, w_im);
          }
          *data_col_ptr = val;
          data_col_ptr += batch_size * time_col * height_col * width_col;
        }
      }
    }
  }
}

void deformable_im2col(const at::Tensor data_im, const at::Tensor data_offset,
    const int channels, const int time, const int height, const int width,
    const int kernel_t, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group, at::Tensor data_col) {
  int time_col = (time + 2 * pad_t -
      (dilation_t * (kernel_t - 1) + 1)) / stride_t + 1;
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * time_col * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    data_im.type(), "deformable_im2col_kernel", ([&] {
      const scalar_t *_data_im = data_im.data<scalar_t>();
      const scalar_t *_data_offset = data_offset.data<scalar_t>();
      scalar_t * _data_col = data_col.data<scalar_t>();
      deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels),
                                 CUDA_NUM_THREADS>>>(
          num_kernels, _data_im, _data_offset, time, height, width, kernel_t, kernel_h, kernel_w,
          pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w,
          channel_per_deformable_group, parallel_imgs, channels, deformable_group,
          time_col, height_col, width_col, _data_col);
    }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
  }
}


template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(const int n, const scalar_t* data_col,
    const scalar_t* data_offset, const int channels,
    const int time, const int height, const int width,
    const int kernel_t, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int deformable_group,
    const int time_col, const int height_col, const int width_col,
    scalar_t* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / time_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / time_col / batch_size / kernel_w) % kernel_h;
    const int k = (index / width_col / height_col / time_col / batch_size / kernel_w / kernel_h) % kernel_t;
    const int c = index / width_col / height_col / time_col / batch_size / kernel_w / kernel_h / kernel_t;

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int t_out = (index / width_col / height_col) % time_col;
    int b = (index / width_col / height_col / time_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;
    int t_in = t_out * stride_t - pad_t;

    const scalar_t* data_offset_ptr = data_offset +
         (b * deformable_group + deformable_group_index) * 2 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;
    const int data_offset_h_ptr = ( ( (2 * ((k * kernel_h + i) * kernel_w + j)) * time_col + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ( ( (2 * ((k * kernel_h + i) * kernel_w + j) + 1) * time_col + t_out) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_t = 0;
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_t_data = t_in + k * dilation_t + offset_t;
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_t = (int)cur_inv_t_data;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dt = -2; dt <= 2; dt++) {
      for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
          if (cur_t + dt >= 0 && cur_t + dt < time &&
             cur_h + dy >= 0 && cur_h + dy < height &&
             cur_w + dx >= 0 && cur_w + dx < width &&
             abs(cur_inv_t_data - (cur_t + dt)) < 1 &&
             abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
             abs(cur_inv_w_data - (cur_w + dx)) < 1
             ) {
             int cur_bottom_grad_pos = ( ( (b * channels + c) * time + cur_t + dt ) * height + cur_h + dy ) * width + cur_w + dx;
             scalar_t weight = get_gradient_weight(cur_inv_t_data, cur_inv_h_data, cur_inv_w_data,
                                                cur_t + dt, cur_h + dy, cur_w + dx, time, height, width);
             atomicAdd(data_im + cur_bottom_grad_pos, weight * cur_top_grad);
           }
        }
      }
    }
  }
}

void deformable_col2im(
    const at::Tensor data_col, const at::Tensor data_offset,
    const int channels, const int time, const int height, const int width,
    const int kernel_t, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group, at::Tensor data_im) {
  int time_col = (time + 2 * pad_t - (dilation_t * (kernel_t - 1) + 1)) /
      stride_t + 1;
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.type(), "deformable_col2im_kernel", ([&] {
    const scalar_t *_data_col = data_col.data<scalar_t>();
    const scalar_t *_data_offset = data_offset.data<scalar_t>();
    scalar_t *_data_im = data_im.data<scalar_t>();

    deformable_col2im_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels),
                               CUDA_NUM_THREADS>>>(
        num_kernels, _data_col, _data_offset, channels, time, height, width, kernel_t, kernel_h, kernel_w,
        pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w,
        channel_per_deformable_group, parallel_imgs, deformable_group, time_col, height_col, width_col, _data_im);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
  }
}


template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(const int n, const scalar_t* data_col,
    const scalar_t* data_im, const scalar_t* data_offset,
    const int channels, const int time, const int height, const int width,
    const int kernel_t, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int time_col, const int height_col, const int width_col,
    scalar_t* grad_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int t = (index / width_col / height_col) % time_col;
    int c = (index / width_col / height_col / time_col) % offset_channels;
    int b = (index / width_col / height_col / time_col) / offset_channels;

    const int deformable_group_index = c / (2 * kernel_t * kernel_h * kernel_w);
    const int col_step = kernel_t * kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t* data_col_ptr = data_col + deformable_group_index *
        channel_per_deformable_group * batch_size * time_col * width_col * height_col;
    const scalar_t* data_im_ptr = data_im + (b * deformable_group + deformable_group_index) *
        channel_per_deformable_group / kernel_t / kernel_h / kernel_w * time * height * width;
    const scalar_t* data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) *
        2 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_t * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
      const int col_pos = ((( (col_c * batch_size + b) * time_col + t) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / time_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / time_col / batch_size / kernel_w) % kernel_h;
      int k = (col_pos / width_col / height_col / time_col / batch_size / kernel_w / kernel_h) % kernel_t;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int t_out = (col_pos / width_col / height_col) % time_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      int t_in = t_out * stride_t - pad_t;
      const int data_offset_h_ptr = ( ( ((2 * (( k * kernel_h + i) * kernel_w + j) ) * height_col + h_out) * time_col + t_out) * width_col + w_out);
      const int data_offset_w_ptr = ( ( ((2 * (( k * kernel_h + i) * kernel_w + j) + 1) * height_col + h_out) * time_col + t_out) * width_col + w_out);
      const scalar_t offset_t = 0;
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_t = t_in + k * dilation_t + offset_t;
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_t <= -1 || inv_h <= -1 || inv_w <= -1 || inv_t >= time || inv_h >= height || inv_w >= width) {
        inv_t = inv_h = inv_w = -2;
      }
      const scalar_t weight = get_coordinate_weight(inv_t, inv_h, inv_w, time, height, width,
                                                 data_im_ptr + cnt * time * height * width,
                                                 height, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }
    grad_offset[index] = val;
  }
}

void deformable_col2im_coord(
    const at::Tensor data_col,
    const at::Tensor data_im, const at::Tensor data_offset,
    const int channels, const int time, const int height, const int width,
    const int kernel_t, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group, at::Tensor grad_offset) {
  int time_col = (time + 2 * pad_t -
      (dilation_t * (kernel_t - 1) + 1)) / stride_t + 1;
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = time_col * height_col * width_col * 2 * kernel_t * kernel_h * kernel_w * deformable_group;
  int channel_per_deformable_group = channels * kernel_t * kernel_h * kernel_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.type(), "deformable_col2im_coord_kernel", ([&] {
    const scalar_t *_data_col = data_col.data<scalar_t>();
    const scalar_t *_data_im = data_im.data<scalar_t>();
    const scalar_t *_data_offset = data_offset.data<scalar_t>();
    scalar_t *_grad_offset = grad_offset.data<scalar_t>();

    deformable_col2im_coord_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels),
                               CUDA_NUM_THREADS>>>(
        num_kernels, _data_col, _data_im, _data_offset, channels, time, height, width,
        kernel_t, kernel_h, kernel_w, pad_t, pad_h, pad_w,
        stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w,
        channel_per_deformable_group, parallel_imgs, 2 * kernel_t * kernel_h * kernel_w * deformable_group,
        deformable_group, time_col, height_col, width_col, _grad_offset);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im_coord: %s\n", cudaGetErrorString(err));
  }
}

