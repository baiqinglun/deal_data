import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter
import json
from tqdm import tqdm
from tool import get_paths, voltage_to_pressure_factor
from draw_color import Color
import csv


# 获取最大值和最小值
def get_extreme_value(smoothed_pressures):
    max_index = np.argmax(smoothed_pressures)
    max_value = smoothed_pressures[max_index]
    return max_index, max_value


# 获取最大斜率
def get_max_diff(time, smoothed_pressures):
    pressure_diffs = np.diff(smoothed_pressures)
    max_index = np.argmax(smoothed_pressures)
    factor = 0.5  # 想要获取的数据点数量
    left_index = int(max_index - factor * 20000)
    right_index = int(max_index + factor * 20000)
    pressure_diff_points = pressure_diffs[left_index:right_index]
    time_points = time[left_index:right_index]

    return pressure_diff_points, time_points


# 合成输出文件全名
def get_output_filename(csv_file_name):
    return os.path.basename(csv_file_name).split(".")[0] + ".jpg"


class PressureImpulseManager:
    def __init__(self, file_list=None):
        self.time_start_index = []
        self.time_start = []
        self.current = -1
        self.points_count = None
        self.x = []
        self.y = []
        self.all_max_index = []
        self.all_max_value = []
        self.all_min_index = []
        self.all_min_value = []
        with open('settings/get_impulse_settings.json', 'r', encoding='UTF-8') as settings_file:
            settings_data = json.load(settings_file)
            self.output_folder = settings_data["image_output_folder"]
            self.data_folder = settings_data["data_folder"]
            self.gaussian_filter_sigma = settings_data["gaussian_filter_sigma"]
            self.file_list = settings_data["files"] if settings_data["files"] else os.listdir(
                os.path.join(os.getcwd(), self.data_folder))
            self.is_show_image = settings_data["is_show_image"]
            self.points_one_second = settings_data["points_one_second"]
            self.figure_size = settings_data["figure_size"]
            self.x_label = settings_data["x_label"]
            self.y_label = settings_data["y_label"]
            self.figure_title = settings_data["figure_title"]
            self.is_figure_grid = settings_data["is_figure_grid"]
            self.scope_range_factor = settings_data["scope_range_factor"]
            self.mass = settings_data["mass"]
            self.figure_dpi = settings_data["figure_dpi"]
            self.color_count = settings_data["color_count"]
            self.color_id = settings_data["color_id"]
            self.unit = settings_data["unit"]
            self.species = settings_data["species"]
            self.color = Color[self.color_count][self.color_id]
            self.font_size = settings_data["font_size"]
            self.factor = [0.8,0.5,0.3,0.2,0.1]
            if not self.mass:
                for file in self.file_list:
                    if file.endswith(".CSV"):
                        split_string = file.split("-")
                        self.mass.append(split_string[1])
                        # 创建输出文件夹
        if self.output_folder and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

            # 循环所有csv数据文件，开始处理

    def process_all_csv_files(self):
        for csv_file_name in tqdm(get_paths(self.file_list, self.data_folder), desc="Processing CSV Files",
                                  unit="file"):
            self.current += 1
            times, pressures = self.get_pressure_time(csv_file_name)
            smoothed_pressures = gaussian_filter(pressures,
                                                 sigma=self.gaussian_filter_sigma)

            start_index = 10

            for idx, p in enumerate(smoothed_pressures):
                if p > 0.001:
                    start_index = idx
                    break

            self.time_start.append(times[start_index]) # 开始时间
            end_index = start_index + 2 * self.points_one_second
            pressure_subset = smoothed_pressures[start_index:end_index]
            self.y.append(pressure_subset)
            if csv_file_name.split("\\")[4].split("-")[1] == "50":
                self.x2 = np.arange(0,len(pressure_subset)) / self.points_one_second
            elif csv_file_name.split("\\")[4].split("-")[1] == "75":
                self.x3 = np.arange(0,len(pressure_subset)) / self.points_one_second
            elif csv_file_name.split("\\")[4].split("-")[1] == "25":
                self.x1 = np.arange(0,len(pressure_subset)) / self.points_one_second
            elif csv_file_name.split("\\")[4].split("-")[1] == "100":
                self.x4 = np.arange(0,len(pressure_subset)) / self.points_one_second

            max_index,max_value = get_extreme_value(smoothed_pressures)
            print(f"最大值时间：{times[max_index]}")
            pressure_impulse_range = smoothed_pressures[start_index:max_index]
            time_impulse_range = times[start_index:max_index]
            impulse_integral = np.trapz(pressure_impulse_range, x=time_impulse_range)
            impulse_time = times[max_index] - times[start_index]
            impulse_pressure = impulse_integral / impulse_time
            print(f"脉冲压力{impulse_pressure}")
            print(f"时间差{impulse_time}")

            with open('time_differences.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([impulse_integral, impulse_time, impulse_pressure])

        self.x = np.arange(0, 0.00005 * len(self.y[0]), 0.00005)
        self.process_and_plot_csv()

        # 处理一个csv文件

    def process_and_plot_csv(self):

        fig, ax = plt.subplots(figsize=self.figure_size)
        # self.draw_main_figure(ax,x_limit,y_limit)
        self.draw_main_figure(ax)

        # 读取文件中的数据，并转化为压力同时生成x轴时间数据

    def get_pressure_time(self, csv_file_name):
        file_data = None
        file_data = pd.read_csv(csv_file_name, skiprows=range(1, 16))
        voltages = file_data.iloc[:, 0]
        pressures = voltages / 0.725 * voltage_to_pressure_factor(self.unit)
        if csv_file_name.split("\\")[4].split("-")[1] == "50":
            self.points_one_second = 5000
        elif csv_file_name.split("\\")[4].split("-")[1] == "75":
            self.points_one_second = 5000
        elif csv_file_name.split("\\")[4].split("-")[1] == "100":
            self.points_one_second = 20000
        times = np.arange(0,len(pressures)) / self.points_one_second
        self.points_count = len(pressures)
        return times, pressures

    def draw_main_figure(self, ax):
        plt.plot(self.x1, self.y[0], label=self.mass[0], color=self.color[0])
        plt.plot(self.x2, self.y[1], label=self.mass[1], color=self.color[1])
        plt.plot(self.x3, self.y[2], label=self.mass[2], color=self.color[2])
        plt.plot(self.x4, self.y[3], label=self.mass[3], color=self.color[3])
        plt.xlabel(self.x_label)
        plt.ylabel(f'{self.y_label}({self.unit})')
        ax.tick_params(axis='y', direction='in', labelsize=self.font_size)
        ax.tick_params(axis='x', direction='in', labelsize=self.font_size)
        if len(self.y) == 1:
            plt.title(f'{self.figure_title}{self.species}({self.mass[self.current]}g)', pad=15)
            plt.legend(loc="upper right", labels=["Pressure"])
        else:
            plt.title(f'{self.figure_title}{self.species}', pad=15)
            mass_g = [item + 'g' for item in self.mass]
            plt.legend(loc="upper right", labels=mass_g)
        plt.grid(self.is_figure_grid)

        self.output_image(file_path=os.path.join(self.output_folder, get_output_filename(self.output_folder)))
        if self.is_show_image:
            plt.show()

            # 保存图片

    def output_image(self, file_path):
        plt.savefig(file_path, dpi=self.figure_dpi)

    def drawBar(self):
        # 从CSV文件中读取数据
        file_data = pd.read_csv("time_differences.csv")
        mass, impulse_integral, impulse_time, impulse_pressure = file_data.iloc[:, 0:4].T.values
        print(impulse_integral)
        print(impulse_time)
        fig, ax1 = plt.subplots(figsize=self.figure_size)
        bars1 = ax1.bar(mass, impulse_integral, 1.5, label="impulse_integral", color=self.color[0],
                        align='center')
        ax1.set_xlabel("mass", fontsize=self.font_size)
        ax1.set_ylabel("impulse_integral")
        # 设置第一个y轴的范围
        # 设置第一个y轴的刻度线颜色与柱状图的颜色一致，设置字体大小
        ax1.tick_params(axis='y', labelcolor=self.color[0], color=self.color[0], direction='in',
                        labelsize=self.font_size)
        ax1.tick_params(axis='x', direction='in', labelsize=self.font_size)
        ax1.spines['left'].set_color(self.color[0])
        ax1.spines['right'].set_color(self.color[1])
        # 调整左侧y轴标题与刻度线的位置
        ax1.yaxis.set_label_coords(-0.1, 0.5)
        plt.show()