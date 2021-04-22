# ifndef _U2NET_H_
# define _U2NET_H_

#include "utility.h"

class U2NETModel
{
public:
	U2NETModel(const std::string model_path);
	U2NETModel();
	~U2NETModel();
	void init(const std::string model_path);
	std::pair<cv::Mat, cv::Mat> forward(cv::Mat& raw_img);
private:
	torch::jit::script::Module model;
	at::Tensor preprocess_img(cv::Mat& raw_img);
};

#endif // !_U2NET_H_

