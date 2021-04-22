#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>

#include <opencv2/highgui/highgui.hpp>
#include <ctime>

using namespace cv;

MainWindow::MainWindow(QWidget* parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
{
	img_loaded_flag = 0;
	processed_flag = 0;
	std::string model_name = "Slim-Net_model.pt";
	try
	{
		model.init(model_name);
	}
	catch (const std::exception&e)
	{
		QMessageBox::warning(NULL, "System Warning", e.what(),
			QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
		exit(1);
	}
	ui->setupUi(this);
	ui->raw_img->resize(600, 600);
	ui->mask_img->resize(400, 400);
	ui->masked_img->resize(400, 400);
	ui->raw_img->setAlignment(Qt::AlignCenter);
	ui->mask_img->setAlignment(Qt::AlignCenter);
	ui->masked_img->setAlignment(Qt::AlignCenter);

	ui->toolBar->setVisible(false);
	setWindowFlags(windowFlags() & ~Qt::WindowMaximizeButtonHint);
	setFixedSize(this->width(), this->height());
}

MainWindow::~MainWindow()
{
	delete ui;
}


void MainWindow::on_openButton_clicked()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Image File"), "", "Image Files(*.jpg *.png *.jpeg)");
	if (fileName != NULL)
	{
		std::string imgDir = fileName.toLocal8Bit().toStdString();
		int first_index = imgDir.find_last_of('/') + 1;
		int last_index = imgDir.find_last_of('.') - 1;
		imgName = imgDir.substr(first_index, last_index - first_index + 1);
		raw_img = load_image_RGB(imgDir);
		img_loaded_flag = 1;
		processed_flag = 0;
		QPixmap unscaleld_img =
			QPixmap::fromImage(QImage(raw_img.data, raw_img.cols, raw_img.rows, raw_img.step, QImage::Format_RGB888));
		ui->raw_img->setPixmap(unscaleld_img.scaled(ui->raw_img->size(), Qt::KeepAspectRatio));
	}
}


void MainWindow::on_goButton_clicked()
{	
	if (!img_loaded_flag)
		return;

	clock_t start = clock();
	// process
	std::pair<cv::Mat, cv::Mat> results = model.forward(raw_img);
	clock_t end = clock();

	double cost_time = ((double)end - (double)start) / CLOCKS_PER_SEC;
	ui->timeLabel->setText(QString::fromStdString("Time Cost: " + std::to_string(cost_time) + "s"));

	mask = results.first, masked_img = results.second;
	processed_flag = 1;
	cv::Mat tmp_masked_img;
	cv::cvtColor(masked_img, tmp_masked_img, CV_BGRA2RGBA);

	QPixmap unscaleld_img =
		QPixmap::fromImage(
			QImage(mask.data, mask.cols, mask.rows, mask.step,
				QImage::Format_Grayscale8));
	ui->mask_img->setPixmap(unscaleld_img.scaled(ui->mask_img->size(), Qt::KeepAspectRatio));

	QPixmap unscaleld_img2 =
		QPixmap::fromImage(
			QImage(tmp_masked_img.data, tmp_masked_img.cols, tmp_masked_img.rows, tmp_masked_img.step,
				QImage::Format_RGBA8888));
	ui->masked_img->setPixmap(unscaleld_img2.scaled(ui->masked_img->size(), Qt::KeepAspectRatio));
	img_loaded_flag = 1;
}

void MainWindow::on_saveButton_clicked()
{
	if (!processed_flag)
		return;
	QString fileName = QFileDialog::getExistingDirectory(this,
		tr("Save Image File"), "");
	if (fileName != NULL)
	{
		std::string saveDir = fileName.toLocal8Bit().toStdString();
		std::string maskName = saveDir + "/mask_" + imgName + ".png";
		std::string maskedImgName = saveDir + "/masked_" + imgName + ".png";
		save_img_file(maskName, mask);
		save_img_file(maskedImgName, masked_img);
	}
}
