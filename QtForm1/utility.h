# ifndef _UTILITY_H_
# define _UTILITY_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

// 解决Qt slot与torch slot的冲突问题
#ifdef QT_CORE_LIB
#undef slots
#endif
#include <torch/script.h>
#ifdef QT_CORE_LIB
#define QT_CORE_LIB slots
#endif

#include <torch/torch.h>
#include <memory>

at::Tensor mat_to_tensor(cv::Mat& img);
cv::Mat load_image_RGB(const std::string path);
void transform_tensor_input(at::Tensor& img_tensor);
void transform_mat_u2net(cv::Mat& img, int new_size);
at::Tensor get_pred_tensor(at::Tensor& raw_tensor);
void transform_tensor_norm(at::Tensor& pred);
cv::Mat tensor_to_mat(at::Tensor& pred);
void transform_mat_alpha(cv::Mat& raw_mask);
cv::Mat get_mask(cv::Mat& raw_mask, int img_h, int img_w);
cv::Mat get_masked_img(cv::Mat& raw_img, cv::Mat& mask);
bool save_img_file(const std::string file_path, cv::Mat& img);

#endif // !_UTILITY_H_

