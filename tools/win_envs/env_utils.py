import os, sys 
import shutil, re
import json
import importlib.util
import winreg
import ctypes

#判断路径是否含有中文字符
def has_chinese_in_path(path: str) -> bool:
	for char in path:
		if '\u4e00' <= char <= '\u9fff':
			return True
	return False

def find_tcim_path():
	spec = importlib.util.find_spec("tcim_lite")
	if spec is None or spec.origin is None:
		print("❌ 'tcim_lite' package not found. Please install houmo_tcim_runtime_xh2 package first.")
		return None

	tcim_package_path = os.path.dirname(spec.origin)
	if os.path.exists(tcim_package_path):
		print(f"✅ Found tcim_package_path at path:\n{tcim_package_path}")
		return tcim_package_path
	else:
		print(f"❌ tcim_runtime_lite.dll not found, try to reinstall it!")
	return None
      
def generate_environments(env_vars):
	tcim_dll_path = os.path.join(env_vars["TCIM_RUNTIME_PATH"], "bin")
	xh2a_dll_path = os.path.join(env_vars["HOUMO_DRV_PATH"], "hal/lib")
	env_vars["PATH"] = f'{xh2a_dll_path};{tcim_dll_path};%PATH%'
	return env_vars


def generate_bat_scripts(env_vars):
	bat_file = os.path.join(env_vars["HOUMO_EXAMPLES_PATH"], "env.bat")
	bat_content = [
		"@echo off",
		"chcp 65001 >nul",
		"cls",
		"==============================================",
		"echo --- Setting environment variables ---",
    ]

	for key, value in env_vars.items():
		bat_content.append(f'set "{key}={value}"')

	bat_content.extend([
		"echo Setting environment variables completed.",
		"==============================================",
	])

	with open(bat_file, "w", encoding="utf-8") as f:
		f.write("\n".join(bat_content))

	print(f"✅ Generated Success, Path : {bat_file}")


def show_win_envs(env_vars):
	print("=" * 60)
	print("✅ Win11 environments list ")
	for key, value in env_vars.items():
		print(f'set "{key}={value}"')
	print("=" * 60)  
  
def generate_json_scripts(env_vars):
    """新增：生成env.json文件"""
    json_file = os.path.join(env_vars["HOUMO_EXAMPLES_PATH"], "env.json")
    
    # 将环境变量写入JSON文件（保持键值对结构）
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(env_vars, f, ensure_ascii=False, indent=4)  # indent=4 格式化输出
    
    print(f"✅ Generated Success, Path : {json_file}")


def set_permanent_env_var(name, value, is_user=True):
    """
    永久设置Windows环境变量
    :param name: 环境变量名
    :param value: 环境变量值
    :param is_user: True=用户变量（无需管理员），False=系统变量（需管理员）
    """
    # 选择注册表路径
    if is_user:
        key_path = r"Environment"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
    else:
        key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE)
    
    # 设置环境变量（REG_EXPAND_SZ 支持变量扩展，如 %PATH%）
    winreg.SetValueEx(key, name, 0, winreg.REG_EXPAND_SZ, value)
    winreg.CloseKey(key)
    
    # 通知系统环境变量已更新（否则需重启才生效）
    ctypes.windll.user32.SendMessageTimeoutW(
        0xFFFF,  # HWND_BROADCAST
        0x001A,  # WM_SETTINGCHANGE
        0,
        "Environment",
        0x0002,  # SMTO_ABORTIFHUNG
        5000,    # 超时时间（毫秒）
        None
    )
    
def delete_permanent_env_var(name, is_user=True):
    """
    永久删除Windows环境变量
    :param name: 要删除的环境变量名
    :param is_user: True=删除用户变量，False=删除系统变量（需管理员权限）
    :return: 是否删除成功（bool）
    """
    try:
        if is_user:
            key_path = r"Environment"
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                key_path,
                0,
                winreg.KEY_SET_VALUE  # 需写入权限
            )
        else:
            key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                key_path,
                0,
                winreg.KEY_SET_VALUE  # 系统变量需管理员权限
            )

        # 尝试删除环境变量
        winreg.DeleteValue(key, name)
        winreg.CloseKey(key)

        # 通知系统更新环境变量
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF,  # HWND_BROADCAST
            0x001A,  # WM_SETTINGCHANGE
            0,
            "Environment",
            0x0002,  # SMTO_ABORTIFHUNG
            5000,
            None
        )
        print(f"env {name} delete!")
    except FileNotFoundError:
        print(f"env {name} not exist!")

def print_support_lists(list_name: str, support_list: dict):
    print("-" * 50)
    print(list_name.center(50))
    print("-" * 50)
    for demo, value in support_list.items():
        if type(value) is bool:
            print(f"{demo:<15} {'√' if value else '×'}")
        else:
            print(f"√ {demo:<15} : {', '.join(value)}")
    print("-" * 50)