import os, sys
import shutil, re
import json
import importlib.util
import winreg
import ctypes
from typing import Dict, Any


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


class EnvManager:
    def __init__(self, initial_backup_path: str = "initial_env_backup.json"):
        """
        Manager to back up and restore ALL environment variables (not just PATH)
        :param initial_backup_path: Path for initial full environment backup
        """
        self.initial_backup_path = initial_backup_path
        # Registry paths for environment variables
        self.user_reg_path = r"Environment"
        # Create initial full backup if it doesn't exist
        if not os.path.exists(self.initial_backup_path):
            self._create_initial_backup()

    def refresh_envs(self):
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF,  # HWND_BROADCAST
            0x001A,  # WM_SETTINGCHANGE
            0,
            "Environment",
            0x0002,  # SMTO_ABORTIFHUNG
            5000,    # 超时时间（毫秒）
            None
        )

    def _enum_reg_values(self, reg_path: str) -> Dict[str, Any]:
        """Enumerate all values in a registry key (returns {name: value} dict)"""
        values = {}
        try:
            root = winreg.HKEY_CURRENT_USER
            with winreg.OpenKey(root, reg_path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                i = 0
                while True:
                    try:
                        # Get value name and data (skip default value with None name)
                        name, value, _ = winreg.EnumValue(key, i)
                        if name:  # Only save named values (not default)
                            values[name] = value
                        i += 1
                    except OSError:  # No more values
                        break
            return values
        except Exception as e:
            raise RuntimeError(f"Failed to enumerate registry values: {e}")

    def _set_reg_values(self, reg_path: str, values: Dict[str, Any]) -> None:
        """Set multiple values in a registry key (overwrites existing values)"""
        try:
            current_reg_vars = self._enum_reg_values(reg_path)
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                reg_path,
                0,
                winreg.KEY_SET_VALUE  # Simplified access rights
            ) as key:
                existing_names = list(current_reg_vars.keys())

                for name in existing_names:
                    if name not in values:
                        try:
                            winreg.DeleteValue(key, name)
                            print(f"Deleted variable: {name}")
                        except Exception as e:
                            print(f"Warning: Could not delete {name}: {e}")

                # Restore variables from backup
                for name, value in values.items():
                    # Handle empty values correctly
                    reg_type = winreg.REG_EXPAND_SZ if "%" in str(value) else winreg.REG_SZ
                    try:
                        winreg.SetValueEx(key, name, 0, reg_type, value)
                        print(f"Restored variable: {name}")
                    except Exception as e:
                        print(f"Warning: Could not restore {name}: {e}")
        except PermissionError:
            raise PermissionError("Admin rights required for system-level variables. Run as administrator.")
        except Exception as e:
            raise RuntimeError(f"Failed to set registry values: {e}")

    def _create_initial_backup(self) -> None:
        """Create backup of ALL environment variables (user and system) on first run"""
        try:
            # Backup all user-level environment variables
            user_vars = self._enum_reg_values(self.user_reg_path)

            # Save to backup file
            initial_data = {
                "user_variables": user_vars
            }
            with open(self.initial_backup_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
            print(f"Created initial full environment backup: {os.path.abspath(self.initial_backup_path)}")
        except Exception as e:
            print(f"Failed to create initial backup: {e}")

    def set_env(self, var_name: str, value: str) -> None:
        """
        Set a specific environment variable permanently
        """
        reg_path = self.user_reg_path
        try:
            root = winreg.HKEY_CURRENT_USER
            access = winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY
            with winreg.OpenKey(root, reg_path, 0, access) as key:
                reg_type = winreg.REG_EXPAND_SZ if "%" in value else winreg.REG_SZ
                winreg.SetValueEx(key, var_name, 0, reg_type, value)
            print(f"Set {var_name} to {value} ('user-level')")
        except PermissionError:
            raise PermissionError("Admin rights required for system-level variables. Run as administrator.")
        except Exception as e:
            print(f"Failed to set {var_name}: {e}")

    def restore_to_initial(self) -> None:
        """Restore ALL environment variables to state before first script run"""
        if not os.path.exists(self.initial_backup_path):
            print(f"Initial backup not found: {self.initial_backup_path}")
            return

        try:
            with open(self.initial_backup_path, "r", encoding="utf-8") as f:
                initial_data = json.load(f)

            # Restore user-level variables
            self._set_reg_values(self.user_reg_path, initial_data["user_variables"])

            print("Restored ALL environment variables to initial state (before first run)")
            print(f"Backup source: {os.path.abspath(self.initial_backup_path)}")
        except Exception as e:
            print(f"Restore failed: {e}")

    def add_to_path(self, new_path: str) -> None:
        """
        Add a new path to the PATH environment variable (permanent)
        """
        # Validate path format
        new_path = new_path.rstrip('\\')  # Remove trailing backslash if present

        # Get current PATH value
        reg_path = self.user_reg_path
        current_vars = self._enum_reg_values(reg_path)
        current_path = current_vars.get("Path", "")

        # Check if path already exists
        path_list = [p.strip() for p in current_path.split(";") if p.strip()]
        if new_path in path_list:
            print(f"Path already in PATH: {new_path}")
            return

        # Add new path to PATH
        updated_path = f"{current_path};{new_path}" if current_path else new_path

        # Update PATH in registry
        try:
            root = winreg.HKEY_CURRENT_USER
            access = winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY
            with winreg.OpenKey(root, reg_path, 0, access) as key:
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, updated_path)
            print(f"Added to 'user-level' PATH: {new_path}")
            print("Note: New PATH takes effect in new processes. Restart window to apply.")
        except PermissionError:
            raise PermissionError("Admin rights required for system-level PATH. Run as administrator.")
        except Exception as e:
            print(f"Failed to update PATH: {e}")