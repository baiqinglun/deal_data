import os


# 获取所有csv文件的绝对路径
def get_paths(file_list,data_folder):
    csv_file_list = []
    for filename in file_list:
        if filename.endswith(".CSV"):
            file_path = os.path.join(os.getcwd(),data_folder, filename)
            csv_file_list.append(file_path)
    return csv_file_list


# 合成输出文件全名
def get_output_filename(csv_file_name):
    return os.path.basename(csv_file_name).split(".")[0] + ".jpg"


def voltage_to_pressure_factor( unit='kPa'):
    conversion_factor = 1000 if unit == 'kPa' else 1  # 1000 for kPa, 1 for MPa
    return conversion_factor