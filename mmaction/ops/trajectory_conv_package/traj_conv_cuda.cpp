#include <ATen/ATen.h>
#include <torch/torch.h>

#include "deform_3d_conv_cuda_kernel.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void shape_check(at::Tensor input, at::Tensor offset,
                 at::Tensor *gradOutput, at::Tensor weight, int kT, int kH, int kW,
                 int dT, int dH, int dW, int padT, int padH, int padW,
                 int dilationT, int dilationH, int dilationW, int deformable_group)
{

    TORCH_CHECK(weight.ndimension() == 5,
             "5D weight tensor (nOutputPlane,nInputPlane,kT,kH,kW) expected, "
             "but got: %s",
             weight.ndimension());

    TORCH_CHECK(weight.is_contiguous(),
             "weight tensor has to be contiguous");

    TORCH_CHECK(kT >0 && kW > 0 && kH > 0,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d",
             kT, kH, kW);

    TORCH_CHECK((weight.size(2) == kT &&
              weight.size(3) == kH &&
              weight.size(4) == kW),
             "kernel size should be consistent with weight, ",
             "but got kT: %d kH: %d kW: %d weight.size(2): %d weight.size(3): %d, weight.size(4): %d",
             kT, kH, kW, weight.size(2), weight.size(3), weight.size(4));

    TORCH_CHECK(dW > 0 && dH > 0 && dT > 0,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);

    TORCH_CHECK(dilationW > 0 && dilationH > 0 && dilationT > 0,
             "dilation should be greater than 0, but got dilationT: %d dilationH: %d dilationW: %d",
             dilationT, dilationH, dilationW);

    int ndim = input.ndimension();
    int dimf = 0;
    int dimt = 1;
    int dimh = 2;
    int dimw = 3;

    if (ndim == 5)
    {
        dimf++;
        dimt++;
        dimh++;
        dimw++;
    }

    TORCH_CHECK(ndim == 4 || ndim == 5,
             "4D or 5D input tensor expected but got: %s", ndim);

    long nInputPlane = weight.size(1);
    long inputTime = input.size(dimt);
    long inputHeight = input.size(dimh);
    long inputWidth = input.size(dimw);
    long nOutputPlane = weight.size(0);
    long outputTime = (inputTime + 2 * padT - (dilationT * (kT - 1) + 1)) / dT + 1;
    long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

    TORCH_CHECK(nInputPlane % deformable_group == 0,
             "input channels must divide deformable group size");

    if (outputWidth < 1 || outputHeight < 1 || outputTime < 1)
        AT_ERROR(
            "Given input size: (%ld x %ld x %ld x %ld). "
            "Calculated output size: (%ld x %ld x %ld x %ld). Output size is too small",
            nInputPlane, inputTime, inputHeight, inputWidth, nOutputPlane,
            outputTime, outputHeight, outputWidth);

    TORCH_CHECK(input.size(1) == nInputPlane,
             "invalid number of input planes, expected: %d, but got: %d",
             nInputPlane, input.size(1));

    TORCH_CHECK((inputTime >= kT && inputHeight >= kH && inputWidth >= kW),
             "input image is smaller than kernel");

    TORCH_CHECK(
        (offset.size(2) == outputTime && offset.size(3) == outputHeight && offset.size(4) == outputWidth),
        "invalid spatial size of offset, expected time: %d height: %d width: %d, but got time: %d height: %d width: %d",
        outputTime, outputHeight, outputWidth, offset.size(2), offset.size(3), offset.size(4));

    TORCH_CHECK((offset.size(1) == deformable_group * 2 * kT * kH * kW),
             "invalid number of channels of offset, offset.size(1): %d kT: %d kH: %d kW: %d", offset.size(1), kT, kH, kW);

    if (gradOutput != NULL)
    {
        TORCH_CHECK(gradOutput->size(dimf) == nOutputPlane,
                 "invalid number of gradOutput planes, expected: %d, but got: %d",
                 nOutputPlane, gradOutput->size(dimf));

        TORCH_CHECK((gradOutput->size(dimt) == outputTime && 
                  gradOutput->size(dimh) == outputHeight &&
                  gradOutput->size(dimw) == outputWidth),
                 "invalid size of gradOutput, expected time: %d height: %d width: %d , but got time: %d height: %d width: %d",
                 outputTime, outputHeight, outputWidth, gradOutput->size(dimt), gradOutput->size(dimh), gradOutput->size(dimw));
    }
}


int deform_3d_conv_forward_cuda(at::Tensor &input, at::Tensor &weight, at::Tensor &bias,
                                at::Tensor &offset, at::Tensor &output,
                                at::Tensor &columns, at::Tensor &ones,
                                int kT, int kH, int kW,
                                int strideT, int strideH, int strideW,
                                int padT, int padH, int padW,
                                int dilationT, int dilationH, int dilationW,
                                int deformable_group, int im2col_step) {

  shape_check(input, offset, NULL, weight, kT, kH, kW, strideT, strideH, strideW,
              padT, padH, padW, dilationT, dilationH, dilationW, deformable_group);

  CHECK_INPUT(input);
  CHECK_INPUT(offset);
  CHECK_INPUT(weight); 

  int batch = 1;
  if (input.ndimension() == 4) {
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2), input.size(3)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2), offset.size(3)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputTime = input.size(2);
  long inputHeight = input.size(3);
  long inputWidth = input.size(4);

  long nOutputPlane = weight.size(0);

  long outputTime = (inputTime + 2 * padT - (dilationT * (kT - 1) + 1)) / strideT + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / strideH + 1;
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / strideW + 1;

  TORCH_CHECK(offset.size(0) == batchSize, "Invalid batch size of offset");

  output = output.view({batchSize / im2col_step, im2col_step, nOutputPlane, outputTime, outputHeight, outputWidth});
  columns = at::zeros({nInputPlane * kT * kH * kW, im2col_step * outputTime * outputHeight * outputWidth}, input.type());

  ones = at::ones({outputTime, outputHeight, outputWidth}, input.type());

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane, inputTime, inputHeight, inputWidth});
  offset = offset.view({batchSize / im2col_step, im2col_step, deformable_group * 2 * kT * kH * kW, outputTime, outputHeight, outputWidth});
  

  auto output_buffer = at::zeros({batchSize / im2col_step, nOutputPlane, im2col_step, outputTime, outputHeight, outputWidth}, output.type());

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    auto input_n = input.select(0, elt);
    auto offset_n = offset.select(0, elt);
    auto output_buffer_n = output_buffer.select(0, elt);
    

    // if (bias) {
    //     output_buffer_n = output_buffer_n.flatten(1).addmm_(bias.flatten(1), ones.flatten(1).transpose(1, 0), 0.0f, 1.0f).view_as(output_buffer_n);
    // }
 
    deformable_im2col(
      input_n, offset_n,
      nInputPlane, inputTime, inputHeight, inputWidth, kT, kH, kW,
      padT, padH, padW, strideT, strideH, strideW,
      dilationT, dilationH, dilationW,
      im2col_step, deformable_group, columns);

    output_buffer_n = output_buffer_n.flatten(1).addmm_(weight.flatten(1), columns).view_as(output_buffer_n);
  }

  output_buffer = output_buffer.view({batchSize / im2col_step, nOutputPlane, im2col_step, outputTime, outputHeight, outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputTime, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  offset = offset.view({batchSize, deformable_group * 2 * kT * kH * kW, outputTime, outputHeight, outputWidth});
  
  if (batchSize == 0) {
    output = output.view({nOutputPlane, outputTime, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputTime, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
  }

  if (bias.size(0)!=0) {
      output += bias.view({1, bias.size(0), 1, 1, 1});
  }
  return 1;
}

int deform_3d_conv_backward_input_cuda(at::Tensor &input, at::Tensor &offset,
                                       at::Tensor &gradOutput, at::Tensor &gradInput,
                                       at::Tensor &gradOffset, at::Tensor &weight, at::Tensor &bias,
                                       at::Tensor &columns, int kT, int kH, int kW,
                                       int strideT, int strideH, int strideW,
                                       int padT, int padH, int padW,
                                       int dilationT, int dilationH, int dilationW,
                                       int deformable_group, int im2col_step) {

  shape_check(input, offset, &gradOutput, weight, kT, kH, kW, strideT, strideH, strideW,
              padT, padH, padW, dilationT, dilationH, dilationW, deformable_group);

  CHECK_INPUT(input);
  CHECK_INPUT(offset);
  CHECK_INPUT(gradOutput);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);

  int batch = 1;

  if (input.ndimension() == 4) {
    batch = 0;
    input = input.view({1, input.size(0), input.size(1),
                       input.size(2), input.size(3)});
    offset = offset.view({1, offset.size(0), offset.size(1),
                         offset.size(2), offset.size(3)});
    gradOutput = gradOutput.view({1, gradOutput.size(0), gradOutput.size(1),
                                 gradOutput.size(2), gradOutput.size(3)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputTime = input.size(2);
  long inputHeight = input.size(3);
  long inputWidth = input.size(4);

  long nOutputPlane = weight.size(0);

  long outputTime = (inputTime + 2 * padT - (dilationT * (kT - 1) + 1)) / strideT + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / strideH + 1;
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / strideW + 1;

  TORCH_CHECK(offset.size(0) == batchSize, "invalid batch size of offset");


  gradInput = gradInput.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  columns = at::zeros({nInputPlane * kT * kH * kW, im2col_step * outputTime * outputHeight * outputWidth}, input.type());

  // change order of grad output
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step, nOutputPlane, outputTime, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);

  auto gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer = gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step, outputTime, outputHeight, outputWidth});
  gradOutputBuffer.copy_(gradOutput);
  // gradOutputBuffer = gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step, outputTime, outputHeight * outputWidth});

  gradOutput.transpose_(1, 2);
  gradOutput = gradOutput.view({batchSize, nOutputPlane, outputTime, outputHeight, outputWidth});

  gradInput = gradInput.view({batchSize / im2col_step, im2col_step, nInputPlane, inputTime, inputHeight, inputWidth});
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane, inputTime, inputHeight, inputWidth});
  gradOffset = gradOffset.view({batchSize / im2col_step, im2col_step, deformable_group * 2 * kT * kH * kW,
                               outputTime, outputHeight, outputWidth});
  offset = offset.view({batchSize / im2col_step, im2col_step, deformable_group * 2 * kT * kH * kW,
                       outputTime, outputHeight, outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    auto gradInput_n = gradInput.select(0, elt);
    auto gradOffset_n = gradOffset.select(0, elt);
    auto input_n = input.select(0, elt);
    auto offset_n = offset.select(0, elt);
    auto gradOutput_n = gradOutputBuffer.select(0, elt);

    // TODO: CHECK!!!!
    // columns = columns.addmm_(weight.flatten(1).transpose(0, 1), gradOutput_n.flatten(1));
    columns = columns.addmm_(weight.flatten(1).transpose(0, 1), gradOutput_n.flatten(1), 0.0f, 1.0f);

    deformable_col2im_coord(
      columns, input_n, offset_n,
      nInputPlane, inputTime, inputHeight, inputWidth, kT, kH, kW,
      padT, padH, padW, strideT, strideH, strideW,
      dilationT, dilationH, dilationW, im2col_step, deformable_group,
      gradOffset_n);  

    deformable_col2im(
      columns, offset_n,
      nInputPlane, inputTime, inputHeight, inputWidth,
      kT, kH, kW, padT, padH, padW, strideT, strideH, strideW,
      dilationT, dilationH, dilationW, im2col_step, deformable_group,
      gradInput_n);
  }

  gradInput = gradInput.view({{batchSize, nInputPlane, inputTime, inputHeight, inputWidth}});
  input = input.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  gradOffset = gradOffset.view({batchSize, deformable_group * 2 * kT * kH * kW, outputTime, outputHeight, outputWidth});
  offset = offset.view({batchSize, deformable_group * 2 * kT * kH * kW, outputTime, outputHeight, outputWidth});

  if (batch == 0) {
        gradOutput = gradOutput.view({nOutputPlane, outputTime, outputHeight, outputWidth});
        input = input.view({nInputPlane, inputTime, inputHeight, inputWidth});
        gradInput = gradInput.view({nInputPlane, inputTime, inputHeight, inputWidth});
        offset = offset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
        gradOffset = gradOffset.view({offset.size(1), offset.size(2), offset.size(3), offset.size(4)});
  }

  return 1;
}

int deform_3d_conv_backward_parameters_cuda(at::Tensor input, at::Tensor offset,
                                            at::Tensor gradOutput, at::Tensor gradWeight,
                                            at::Tensor gradBias,
                                            at::Tensor columns, at::Tensor ones,
                                            int kT, int kH, int kW,
                                            int strideT, int strideH, int strideW,
                                            int padT, int padH, int padW,
                                            int dilationT, int dilationH, int dilationW,
                                            int deformable_group, float scale, int im2col_step) {

  shape_check(input, offset, &gradOutput, gradWeight, kT, kH, kW,
              strideT, strideH, strideW, padT, padH, padW,
              dilationT, dilationH, dilationW, deformable_group);

  CHECK_INPUT(input);
  CHECK_INPUT(offset);
  CHECK_INPUT(gradOutput);

  int batch = 1;

  // if (input.ndimension() == 4) {
  //   // Force batch
  //   batch = 0;
  //   input.resize_({1, input.size(0), input.size(1),
  //                 input.size(2), input.size(3)});
  //   gradOutput.resize_({1, gradOutput.size(0), gradOutput.size(1),
  //                      gradOutput.size(2), gradOutput.size(3)});
  // }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputTime = input.size(2);
  long inputHeight = input.size(3);
  long inputWidth = input.size(4);

  long nOutputPlane = gradWeight.size(0);

  long outputTime = (inputTime + 2 * padT - (dilationT * (kT - 1) + 1)) / strideT + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / strideH + 1;
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / strideW + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  columns = at::zeros({nInputPlane * kT * kH * kW, im2col_step * outputTime * outputHeight * outputWidth}, input.type());
  ones = at::ones({im2col_step, outputTime, outputHeight, outputWidth}, input.type());
  
  // change order of grad output
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step, nOutputPlane, outputTime, outputHeight * outputWidth});
  gradOutput.transpose_(1, 2);

  auto gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer = gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step, outputTime, outputHeight * outputWidth});
  gradOutputBuffer.copy_(gradOutput);

  gradOutput.transpose_(1, 2);
  gradOutput = gradOutput.view({batchSize, nOutputPlane, outputTime, outputHeight * outputWidth});
 
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane, inputTime ,inputHeight * inputWidth});
  offset = offset.view({batchSize / im2col_step, im2col_step,
      deformable_group * 2 * kT * kH * kW, outputTime, outputHeight * outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    auto input_n = input.select(0, elt);
    auto offset_n = offset.select(0, elt);
    auto gradOutput_n = gradOutputBuffer.select(0, elt);

    deformable_im2col(
        input_n, offset_n, nInputPlane,
        inputTime, inputHeight, inputWidth, kT, kH, kW,
        padT, padH, padW, strideT, strideH, strideW,
        dilationT, dilationH, dilationW,
        im2col_step, deformable_group, columns);

    gradWeight = gradWeight.flatten(1).addmm_(gradOutput_n.flatten(1), columns.transpose(1,0), 1.0, scale).view_as(gradWeight);

    if (gradBias.size(0) != 0) {
        gradBias = gradBias.view({-1, 1}).addmm_(gradOutput_n.flatten(1), ones.view({-1, 1})).view(-1);
    }

  }

  input = input.view({batchSize, nInputPlane, inputTime, inputHeight, inputWidth});
  offset = offset.view({batchSize, deformable_group * 2 * kT * kH * kW, outputTime, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputTime, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputTime, inputHeight, inputWidth});
  }

  return 1;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_3d_conv_forward_cuda", &deform_3d_conv_forward_cuda, "Deform_3d_Conv forward (CUDA)");
  m.def("deform_3d_conv_backward_input_cuda", &deform_3d_conv_backward_input_cuda, "Deform_3d_Conv backward input (CUDA)");
  m.def("deform_3d_conv_backward_parameters_cuda", &deform_3d_conv_backward_parameters_cuda, "Deform_3d_Conv backward parameters (CUDA)");
}
