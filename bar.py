import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
from tool import get_paths, get_output_filename
from draw_color import Color


def get_y_limit(value, id):
    max_index = np.argmax(value)
    max_value = value[max_index]
    pressure_ax_y_limit = [0, max_value + 0.2] if id == 0 else [0, max_value + 2]
    return pressure_ax_y_limit


def read_file_data(csv_file_name):
    file_data = pd.read_csv(csv_file_name)
    mass, max_pressure, max_pressure_rise = file_data.iloc[:, 0:3].T.values
    max_pressure = max_pressure
    max_pressure_rise = max_pressure_rise
    return mass, max_pressure, max_pressure_rise


class BarDrawer:
    def __init__(self):
        self.current = -1
        with open('settings/bar_settings.json', 'r', encoding='UTF-8') as settings_file:
            settings_data = json.load(settings_file)
            self.file_list = settings_data["file_list"]
            self.font_size = settings_data["font_size"]
            self.mass_label_unit = settings_data["mass_label_unit"]
            self.y_label = settings_data["y_label"]
            self.image_output_folder = settings_data["image_output_folder"]
            self.figure_dpi = settings_data["figure_dpi"]
            self.bar_width = settings_data["bar_width"]
            self.figure_size = settings_data["figure_size"]
            self.is_show_image = settings_data["is_show_image"]
            self.title_padding = settings_data["title_padding"]
            self.sub_right_white = settings_data["sub_right_white"]
            self.output_folder = settings_data["output_folder"]
            self.color_count = settings_data["color_count"]
            self.color_id = settings_data["color_id"]
            self.color = Color[self.color_count][self.color_id]
            self.data_folder = settings_data["data_folder"]
            self.is_show_data = settings_data["is_show_data"]
            self.pressure_unit = settings_data["pressure_unit"]
            self.time_unit = settings_data["time_unit"]
            self.major_width = settings_data["major_width"]
            self.major_size = settings_data["major_size"]
            self.species = settings_data["species"]

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
            # 创建输出文件夹
        if self.output_folder and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

            # 循环所有csv数据文件，开始处理

    def process_all_csv_files(self):
        for csv_file_name in tqdm(get_paths(self.file_list, self.data_folder), desc="Processing CSV Files",
                                  unit="file"):
            self.current += 1
            self.process_and_plot(csv_file_name)

    def process_and_plot(self, csv_file_name):
        mass, max_pressure, max_pressure_rise = read_file_data(csv_file_name)
        x = np.arange(len(mass))
        self.draw_figure(csv_file_name, x, mass, max_pressure, max_pressure_rise)

    def draw_figure(self, csv_file_name, x, mass, max_pressure, max_pressure_rise):
        pressure_ax1_y_limit = get_y_limit(max_pressure, 0)
        pressure_ax2_y_limit = get_y_limit(max_pressure_rise, 1)

        fig, ax1 = plt.subplots(figsize=self.figure_size)
        self.draw_first_bar(ax1, x, max_pressure, pressure_ax1_y_limit)
        ax2 = ax1.twinx()
        self.draw_second_bar(ax2, x, max_pressure_rise, pressure_ax2_y_limit)

        # 合并图例，设置字体大小
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        legend = plt.legend(lines, labels, loc='upper left', fontsize=self.font_size)
        legend.get_frame().set_facecolor('lightgray')  # 设置背景颜色
        legend.get_frame().set_alpha(0.4)  # 设置透明度
        plt.xticks(x + self.bar_width / 2, mass)

        # 调整右侧空白
        plt.subplots_adjust(right=self.sub_right_white)
        plt.title(f"{self.y_label[0]},{self.y_label[1]} of {self.species}", fontsize=self.font_size,
                  pad=self.title_padding)
        self.output_image(file_path=os.path.join(self.output_folder, get_output_filename(csv_file_name)))
        if self.is_show_image:
            plt.show()

    def draw_first_bar(self, ax1, x, max_pressure, pressure_ax1_y_limit):
        # 绘制第一个y轴的柱状图
        bars1 = ax1.bar(x, max_pressure, self.bar_width, label=self.y_label[0], color=self.color[0],
                        align='center')
        ax1.set_xlabel(self.mass_label_unit, fontsize=self.font_size)
        ax1.set_ylabel(f'{self.y_label[0]}({self.pressure_unit})', color=self.color[0], fontsize=self.font_size,
                       labelpad=self.title_padding)
        # 设置第一个y轴的范围
        ax1.set_ylim(pressure_ax1_y_limit)
        # 设置第一个y轴的刻度线颜色与柱状图的颜色一致，设置字体大小
        ax1.tick_params(axis='y', labelcolor=self.color[0], color=self.color[0], direction='in',
                        labelsize=self.font_size)
        ax1.tick_params(axis='x', direction='in', labelsize=self.font_size)
        ax1.spines['left'].set_color(self.color[0])
        ax1.spines['right'].set_color(self.color[1])
        # 调整左侧y轴标题与刻度线的位置
        ax1.yaxis.set_label_coords(-0.05, 0.5)
        if self.is_show_data:
            for x_pos, height in zip(x, max_pressure):
                ax1.annotate(f'{height:.2f}', (x_pos, height), ha='center', va='bottom', fontsize=self.font_size)

    def draw_second_bar(self, ax2, x, max_pressure_rise, pressure_ax2_y_limit):
        bars2 = ax2.bar(x + self.bar_width, max_pressure_rise, self.bar_width, label=self.y_label[1],
                        color=self.color[1],
                        alpha=0.5,
                        align='center')
        # 添加标签给第二个y轴，设置字体大小
        ax2.set_ylabel(f'{self.y_label[0]}({self.pressure_unit}/{self.time_unit})', color=self.color[1],
                       fontsize=self.font_size)
        ax2.tick_params(axis='y', labelcolor=self.color[1], color=self.color[1], direction='in',
                        labelsize=self.font_size)
        ax2.set_ylim(pressure_ax2_y_limit)
        ax2.spines['left'].set_color(self.color[0])
        ax2.spines['right'].set_color(self.color[1])
        # 调整右侧y轴标题与刻度线的位置
        ax2.yaxis.set_label_coords(1.05, 0.5)

        if self.is_show_data:
            for x_pos, height in zip(x, max_pressure_rise):
                ax2.annotate(f'{height:.2f}', (x_pos + self.bar_width, height), ha='center', va='bottom',
                             fontsize=self.font_size)

    def output_image(self, file_path):
        plt.savefig(file_path, dpi=self.figure_dpi)