void deformable_im2col(const at::Tensor data_im,
                       const at::Tensor data_offset, const int channels,
                       const int time, const int height, const int width,
                       const int ksize_t, const int ksize_h, const int ksize_w,
                       const int pad_t, const int pad_h, const int pad_w,
                       const int stride_t, const int stride_h, const int stride_w,
                       const int dilation_t, const int dilation_h, const int dilation_w,
                       const int parallel_imgs,
                       const int deformable_groups,
                       at::Tensor data_col);

void deformable_col2im(const at::Tensor data_col,
                       const at::Tensor data_offset, const int channels,
                       const int time, const int height, const int width,
                       const int ksize_t, const int ksize_h, const int ksize_w,
                       const int pad_t, const int pad_h, const int pad_w,
                       const int stride_t, const int stride_h, const int stride_w,
                       const int dilation_t, const int dilation_h, const int dilation_w,
                       const int parallel_imgs,
                       const int deformable_groups,
                       at::Tensor grad_im);

void deformable_col2im_coord(const at::Tensor data_col,
                       const at::Tensor data_im, const at::Tensor data_offset, const int channels,
                       const int time, const int height, const int width,
                       const int ksize_t, const int ksize_h, const int ksize_w,
                       const int pad_t, const int pad_h, const int pad_w,
                       const int stride_t, const int stride_h, const int stride_w,
                       const int dilation_t, const int dilation_h, const int dilation_w,
                       const int parallel_imgs,
                       const int deformable_groups,
                       at::Tensor grad_offset);

