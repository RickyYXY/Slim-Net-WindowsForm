#include "utility.h"

at::Tensor mat_to_tensor(cv::Mat& img)
{
	at::Tensor img_tensor =
		torch::from_blob(img.data, { 1, img.rows, img.cols, img.channels() }, torch::kByte);
	img_tensor = img_tensor.permute({ 0,3,1,2 });
	img_tensor = img_tensor.toType(torch::kFloat);
	return img_tensor;
}

void transform_tensor_input(at::Tensor& img_tensor)
{
	at::Tensor max_val = torch::max(img_tensor);
	img_tensor.div_(max_val);
	img_tensor[0][0].sub_(0.485).div_(0.229);
	img_tensor[0][1].sub_(0.456).div_(0.224);
	img_tensor[0][2].sub_(0.406).div_(0.225);
}

cv::Mat load_image_RGB(const std::string path)
{
	cv::Mat image = cv::imread(path);
	// default is BGR, need RGB
	cv::cvtColor(image, image, CV_BGR2RGB);
	return image;
}

// 未使用
void transform_mat_u2net(cv::Mat& img, int new_size)
{
	img.convertTo(img, CV_32F);
	cv::resize(img, img, cv::Size(new_size, new_size));
	double max_val;
	cv::minMaxLoc(img, NULL, &max_val);
	img = img.mul(1.0 / max_val);
	double stds[3] = { 0.229,0.224,0.225 };
	double means[3] = { 0.485,0.456,0.406 };
	std::vector<cv::Mat>channels;
	cv::split(img, channels);
	for (int i = 0; i < channels.size(); i++)
	{
		channels[i] = channels[i] - means[i];
		channels[i] = channels[i].mul(1.0 / stds[i]);
	}
	cv::merge(channels, img);
}

at::Tensor get_pred_tensor(at::Tensor& raw_tensor)
{
	using torch::indexing::Slice;
	using torch::indexing::None;
	at::Tensor pred =
		raw_tensor.index({ Slice({None,None}),0,Slice({None,None}),Slice({None,None}) });
	return pred;
}

void transform_tensor_norm(at::Tensor& pred)
{
	at::Tensor max_val = torch::max(pred);
	at::Tensor min_val = torch::min(pred);
	pred.sub_(min_val);
	pred.div_(max_val - min_val);
}

cv::Mat tensor_to_mat(at::Tensor& pred)
{
	pred = pred.to(at::kCPU);
	cv::Mat raw_mask(cv::Size(pred.size(2), pred.size(1)), CV_32F, pred.data_ptr());
	return raw_mask;
}

void transform_mat_alpha(cv::Mat& raw_mask)
{
	raw_mask = raw_mask.mul(255.0);
}

cv::Mat get_mask(cv::Mat& raw_mask, int img_h, int img_w)
{
	cv::Mat mask = raw_mask.clone();
	mask.convertTo(mask, CV_8UC1);
	cv::resize(mask, mask, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);
	/*cv::cvtColor(mask, mask, CV_RGB2GRAY);*/
	return mask;
}

cv::Mat get_masked_img(cv::Mat& raw_img, cv::Mat& mask)
{
	cv::Mat img = raw_img.clone();
	std::vector<cv::Mat>channels;
	cv::split(img, channels);
	channels.push_back(mask.clone());
	cv::merge(channels, img);
	cv::cvtColor(img, img, CV_RGBA2BGRA);
	return img;
}

bool save_img_file(const std::string file_path, cv::Mat& img)
{
	std::vector<int> compression_params;
	if (file_path.find_last_of("png") > 0)
	{
		compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别  
		compression_params.push_back(9);  //这里设置保存的图像质量级别
	}
	return cv::imwrite(file_path, img, compression_params);
}
