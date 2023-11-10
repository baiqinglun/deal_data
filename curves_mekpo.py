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


# 获取最大值和最小值
def get_extreme_value(smoothed_pressures):
    max_index = np.argmax(smoothed_pressures)
    max_value = smoothed_pressures[max_index]
    min_index = np.argmin(smoothed_pressures)
    min_value = smoothed_pressures[min_index]
    return min_index, min_value, max_index, max_value


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


class PressureLimitCurves:
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
        with open('settings/curves_settings_mekpo.json', 'r', encoding='UTF-8') as settings_file:
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
            self.major_width = settings_data["major_width"]
            self.major_size = settings_data["major_size"]
            self.all_time = settings_data["all_time"]
            self.title_padding = settings_data["title_padding"]
            # 统一绘图属性
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = self.font_size  # 默认字体大小
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.major.width'] = self.major_width  # X轴刻度线的粗细
            plt.rcParams['ytick.major.width'] = self.major_width  # Y轴刻度线的粗
            plt.rcParams['axes.linewidth'] = self.major_width  # 设置图表的边框线粗细
            plt.rcParams['xtick.major.size'] = self.major_size  # X轴刻度线的长度
            plt.rcParams['ytick.major.size'] = self.major_size  # Y轴刻度线的长度
            plt.rcParams['lines.linewidth'] = self.major_width  # 设置绘图线的宽度
            plt.rcParams['lines.markersize'] = self.major_size  # 设置点图的大小
            plt.rcParams['lines.markeredgewidth'] = self.major_size  # 设置点图中点的粗细
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.titleweight'] = 'bold'
            plt.rcParams['axes.labelweight'] = 'bold'
            plt.rcParams.update({'axes.titlesize': self.font_size})

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

            found_index = 0
            for idx, p in enumerate(smoothed_pressures):
                if p > 0.001:
                    found_index = idx
                    break
            self.time_start.append(times[found_index])
            end_index = found_index + self.all_time * self.points_one_second
            pressure_subset = smoothed_pressures[found_index:end_index]
            if csv_file_name.split("\\")[5].split("-")[1] == "50":
                x2 = np.arange(0, len(pressure_subset)) / self.points_one_second
                self.x.append(x2)
            elif csv_file_name.split("\\")[5].split("-")[1] == "75":
                x3 = np.arange(0, len(pressure_subset)) / self.points_one_second
                self.x.append(x3)
            elif csv_file_name.split("\\")[5].split("-")[1] == "25":
                x1 = np.arange(0, len(pressure_subset)) / self.points_one_second
                self.x.append(x1)
            elif csv_file_name.split("\\")[5].split("-")[1] == "100":
                x4 = np.arange(0, len(pressure_subset)) / self.points_one_second
                self.x.append(x4)

            self.y.append(pressure_subset)
        self.process_and_plot_csv()

        # 处理一个csv文件

    def process_and_plot_csv(self):
        fig, ax = plt.subplots(figsize=self.figure_size)
        self.draw_main_figure(ax)

    def get_pressure_time(self, csv_file_name):
        file_data = pd.read_csv(csv_file_name, skiprows=range(1, 16))
        voltages = file_data.iloc[:, 0]
        pressures = voltages / 0.725 * voltage_to_pressure_factor(self.unit)
        print(csv_file_name)
        if csv_file_name.split("\\")[5].split("-")[1] == "50":
            self.points_one_second = 5000
        elif csv_file_name.split("\\")[5].split("-")[1] == "75":
            self.points_one_second = 5000
        elif csv_file_name.split("\\")[5].split("-")[1] == "100":
            self.points_one_second = 20000
        times = np.arange(0, len(pressures)) / self.points_one_second
        self.points_count = len(pressures)
        return times, pressures

    def draw_main_figure(self, ax):
        for index,y in enumerate(self.y):
            plt.plot(self.x[index],y,label=self.mass[index], color=self.color[index])
        plt.xlabel(self.x_label)
        plt.ylabel(f'{self.y_label}({self.unit})')
        ax.tick_params(axis='y', direction='in', labelsize=self.font_size)
        ax.tick_params(axis='x', direction='in', labelsize=self.font_size)
        if len(self.y) == 1:
            plt.title(f'{self.figure_title}{self.species}({self.mass[self.current]}g)', pad=self.title_padding,fontsize=self.font_size)
            plt.legend(loc="upper right", labels=["Pressure"])
        else:
            plt.title(f'{self.figure_title}{self.species}', pad=self.title_padding)
            mass_g = [item + 'g' for item in self.mass]
            plt.legend(loc="upper right", labels=mass_g)
        plt.grid(self.is_figure_grid)

        self.output_image(file_path=os.path.join(self.output_folder, get_output_filename(self.output_folder)))
        if self.is_show_image:
            plt.show()

            # 保存图片

    def output_image(self, file_path):
        plt.savefig(file_path, dpi=self.figure_dpi)