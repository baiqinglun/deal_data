import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
from tool import get_paths, get_output_filename
from draw_color import Color
from scipy.interpolate import make_interp_spline
from tool import get_paths

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


# 读取csv文件数据
def read_file_data(csv_file_name):
    file_data = pd.read_csv(csv_file_name)
    mass, impulse_integral, impulse_time, impulse_pressure = file_data.iloc[:, 0:4].T.values
    return mass, impulse_integral, impulse_time, impulse_pressure


def draw_one_dot(ax, index, mass, value, color, label):
    mass_new = np.linspace(mass.min(), mass.max(), 300)
    spl = make_interp_spline(mass, value, k=3)  # 使用三次样条曲线插值
    y_smooth = spl(mass_new)

    ax.scatter(mass, value, marker='o', facecolors='none', edgecolors=color, s=100, label=label)
    ax.plot(mass_new, y_smooth, c=color, label='')


class DotDiagramManager:
    def __init__(self):
        self.xtick_major_width = None
        self.time_start_index = None
        self.time_start = None

        with open('settings/impulse_settings.json', 'r', encoding='UTF-8') as settings_file:
            settings_data = json.load(settings_file)
            self.file_list = settings_data["file_list"]
            self.font_size = settings_data["font_size"]
            self.mass_label_unit = settings_data["mass_label_unit"]
            self.impulse = settings_data["impulse"]
            self.image_output_folder = settings_data["image_output_folder"]
            self.figure_dpi = settings_data["figure_dpi"]
            self.bar_width = settings_data["bar_width"]
            self.figure_size = settings_data["figure_size"]
            self.is_show_image = settings_data["is_show_image"]
            self.title_padding = settings_data["title_padding"]
            self.output_folder = settings_data["output_folder"]
            self.color_count = settings_data["color_count"]
            self.color_id = settings_data["color_id"]
            self.color = Color[self.color_count][self.color_id]
            self.data_folder = settings_data["data_folder"]
            self.is_show_data = settings_data["is_show_data"]
            self.pressure_unit = settings_data["pressure_unit"]
            self.time_unit = settings_data["time_unit"]
            self.species = settings_data["species"]
            self.figure_type = settings_data["figure_type"]
            self.figure_type_show = (
            self.figure_type in ["curve", "curve_bar"], self.figure_type in ["bar", "curve_bar"])
            self.major_width = settings_data["major_width"]
            self.major_size = settings_data["major_size"]
            self.dot_size = settings_data["dot_size"]
            # 统一绘图属性
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = self.font_size  # 默认字体大小
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.major.width'] = self.major_width  # X轴刻度线的粗细
            plt.rcParams['ytick.major.width'] = self.major_width  # Y轴刻度线的粗
            plt.rcParams['xtick.major.pad'] = self.title_padding
            plt.rcParams['ytick.major.pad'] = self.title_padding
            plt.rcParams['axes.linewidth'] = self.major_width  # 设置图表的边框线粗细
            plt.rcParams['xtick.major.size'] = self.major_size  # X轴刻度线的长度
            plt.rcParams['ytick.major.size'] = self.major_size  # Y轴刻度线的长度
            plt.rcParams['lines.linewidth'] = self.major_width  # 设置绘图线的宽度
            plt.rcParams['lines.markersize'] = self.dot_size  # 设置点图的大小
            plt.rcParams['lines.markeredgewidth'] = self.major_size  # 设置点图中点的粗细
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.titleweight'] = 'bold'
            plt.rcParams['axes.labelweight'] = 'bold'

        if self.output_folder and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process_all_csv_files(self):
        for csv_file_name in tqdm(get_paths(self.file_list, self.data_folder), desc="Processing CSV Files",
                                  unit="file"):
            self.process_and_plot(csv_file_name)

    def process_and_plot(self, csv_file_name):
        # 获取数据
        mass, impulse_integral, impulse_time, impulse_pressure = read_file_data(csv_file_name)
        data = [
            [impulse_integral, f"{self.impulse[0]}({self.pressure_unit}▪{self.time_unit})", 0],
            [impulse_time, f"{self.impulse[1]}({self.time_unit})", 1],
            [impulse_pressure, f"{self.impulse[2]}({self.pressure_unit})", 2]
        ]
        # 绘图
        self.draw_figure(csv_file_name, mass, data)

    def draw_figure(self, csv_file_name, mass, data):
        # 创建子图
        fig, ax1 = plt.subplots(figsize=self.figure_size)
        fig.subplots_adjust(left=0.085, right=0.85, top=0.9, bottom=0.1)  # 调整子图的位置
        # plt.subplots_adjust(left=0.08)  # 调整左侧留白的大小，可以根据需要进行调整
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax = [ax1, ax2, ax3]
        plt.title(f"{self.impulse[0]},{self.impulse[1]},{self.impulse[2]} of {self.species} ", fontsize=self.font_size,
                  pad=self.title_padding)  # 设置标题文本、字体大小和位置
        for index, value in enumerate(data):
            if self.is_show_data:
                for x_pos, height in zip(mass, value[0]):
                    print(f'{height:.4f}')
                    ax[index].annotate(f'{height:.4f}', (x_pos + self.bar_width * value[2], height), ha='center', va='bottom', fontsize=self.font_size, zorder=100000)
            # 设置坐标轴并绘制线图
            ax[index].set_xlabel('Mass(g)', fontsize=self.font_size, labelpad=self.title_padding)
            ax[index].set_ylabel(value[1], color=self.color[index], labelpad=self.title_padding)
            if self.figure_type_show[1]:
                ax[index].bar(mass + self.bar_width * value[2], value[0], self.bar_width, color=self.color[index],label=f"{value[1]}",zorder=2)
                # 平滑数据绘制曲线图
            mass_new = np.linspace((mass + self.bar_width * value[2]).min(), (mass + self.bar_width * value[2]).max(),
                                   300)
            spl = make_interp_spline((mass + self.bar_width * value[2]), value[0], k=3)  # 使用三次样条曲线插值
            y_smooth = spl(mass_new)
            if self.figure_type_show[0]:
                ax[index].plot(mass_new, y_smooth, color=self.color[index], label=value[1])
                # 找出极值设置坐标轴范围
            max_value = np.max(value[0])
            min_value = np.min(value[0])
            ax[index].set_ylim(
                min_value * 0.8 if (self.figure_type_show[0] and self.figure_type_show[1] != True) else 0,
                max_value * 1.3)
            # 刻度线与轴线的颜色
            ax[index].tick_params(axis='y', labelcolor=self.color[index], color=self.color[index])
            ax[index].spines['left'].set_color(self.color[0])
            if index > 0:
                ax[index].spines['right'].set_color(self.color[index])
                # 绘制点图
            marker_styles = ['*', 'o', 'D']  # 定义不同的标记样式
            marker_style = marker_styles[index]
            if self.figure_type_show[0]:
                ax[index].scatter(mass + self.bar_width * value[2], value[0], marker=marker_style, facecolors='none',
                                  edgecolor=self.color[index], color=self.color[index])
                # 移动y轴坐标和x轴坐标标题
        ax3.spines['right'].set_position(('outward',85))
        plt.xticks(mass + self.bar_width, mass)

        sub_sequence_y = np.linspace(0, 0.032, 9)
        sub_sequence_y_rounded = np.round(sub_sequence_y, 4)
        sub_sequence_y2 = np.linspace(0, 0.32, 9)
        sub_sequence_y_rounded2 = np.round(sub_sequence_y2,2)
        sub_sequence_y3 = np.linspace(0, 0.24, 9)
        sub_sequence_y_rounded3 = np.round(sub_sequence_y3, 2)
        print(sub_sequence_y_rounded2)
        ax1.set_yticks(sub_sequence_y_rounded)
        ax2.set_yticks(sub_sequence_y_rounded2)
        ax3.set_yticks(sub_sequence_y_rounded3)

        # 组合图例
        if self.figure_type == "curve_bar":
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines = lines1 + lines2 + lines3
            labels = labels1 + labels2 + labels3
            legend = plt.legend([lines[1] , lines[3],lines[5]],[labels[1] , labels[3] , labels[5]], loc='upper left', fontsize=self.font_size)
            legend.get_frame().set_facecolor('lightgray')  # 设置背景颜色
            legend.get_frame().set_alpha(0.4)  # 设置透明度
        else:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines = lines1 + lines2 + lines3
            labels = labels1 + labels2 + labels3
            legend = plt.legend(lines, labels, loc='upper left', fontsize=self.font_size)
            legend.get_frame().set_facecolor('lightgray')  # 设置背景颜色
            legend.get_frame().set_alpha(0.4)  # 设置透明度

        # plt.subplots_adjust(left=0.07, right=0.87, bottom=0.09, top=0.93) # MEKPO
        plt.subplots_adjust(left=0.07, right=0.87, bottom=0.09, top=0.93) # AIBN、BPO
            # 图片保存与现实
        self.output_image(file_path=os.path.join(self.output_folder, get_output_filename(csv_file_name)))
        if self.is_show_image:
            plt.show()

    def output_image(self, file_path):
        plt.savefig(file_path, dpi=self.figure_dpi)