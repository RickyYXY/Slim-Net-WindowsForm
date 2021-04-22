#include "u2net.h"

U2NETModel::U2NETModel(const std::string model_path)
{
	try
	{
		model = torch::jit::load(model_path);
	}
	catch (const c10::Error& e)
	{
		throw std::exception("The model file can't be loaded,\nplease check your model.");
	}

	if (torch::cuda::is_available())
		model.to(at::kCUDA);
	else
		model.to(at::kCPU);
	model.eval();
}

U2NETModel::U2NETModel()
{
}

U2NETModel::~U2NETModel()
{
}

void U2NETModel::init(const std::string model_path)
{
	try
	{
		model = torch::jit::load(model_path);
	}
	catch (const c10::Error& e)
	{
		throw std::exception(
			"Model loading Error!\nPossible reason:"
			"\n1. Can't find your *.pt file."
			"\n2. Lack of NIVIDA's driver."
			"\nPlease check again.");
	}

	if (torch::cuda::is_available())
		model.to(at::kCUDA);
	else
		model.to(at::kCPU);
	model.eval();
}

at::Tensor U2NETModel::preprocess_img(cv::Mat& raw_img)
{
	cv::Mat img = raw_img.clone();
	int new_size = 320;
	cv::resize(img, img, cv::Size(new_size, new_size));
	at::Tensor img_tensor = mat_to_tensor(img);

	if (torch::cuda::is_available())
		img_tensor = img_tensor.to(at::kCUDA);
	else
		img_tensor = img_tensor.to(at::kCPU);

	transform_tensor_input(img_tensor);
	return img_tensor;
}

std::pair<cv::Mat, cv::Mat> U2NETModel::forward(cv::Mat& raw_img)
{
	std::vector<torch::jit::IValue> inputs;
	at::Tensor input = preprocess_img(raw_img);
	inputs.push_back(input);

	auto  outputs = model.forward(inputs);
	auto tpl_outputs = outputs.toTuple();
	at::Tensor output = tpl_outputs->elements()[0].toTensor();
	output = get_pred_tensor(output);
	transform_tensor_norm(output);

	cv::Mat raw_mask = tensor_to_mat(output);
	transform_mat_alpha(raw_mask);
	cv::Mat mask = get_mask(raw_mask, raw_img.rows, raw_img.cols);
	cv::Mat masked_img = get_masked_img(raw_img, mask);

	return std::pair<cv::Mat, cv::Mat>(mask, masked_img);
}


