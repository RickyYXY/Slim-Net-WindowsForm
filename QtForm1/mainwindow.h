#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "u2net.h"
#include <iostream>
#include <QMainWindow>


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_openButton_clicked();
    void on_goButton_clicked();
    void on_saveButton_clicked();

private:
    Ui::MainWindow *ui;
    cv::Mat raw_img;
    cv::Mat mask;
    cv::Mat masked_img;
    U2NETModel model;
    std::string imgName;
    int img_loaded_flag;
    int processed_flag;
};
#endif // MAINWINDOW_H
