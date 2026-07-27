# Static Source Review Rules

本 reference 定义纯 AI reviewer 对 Python、C/C++、Bash、CMake、MSVC、Android NDK 和 first-party 文件头的静态检查规则。Reviewer 不执行 parser、compiler、CMake 或交叉编译，只报告能够从 diff 和仓库上下文确定的问题。

## 目录

- [适用范围与证据门槛](#适用范围与证据门槛)
- [语言级静态语法检查](#语言级静态语法检查)
- [MSVC 可移植性](#msvc-可移植性)
- [Android NDK 交叉编译可移植性](#android-ndk-交叉编译可移植性)
- [First-party 源文件头](#first-party-源文件头)
- [Finding 与严重程度](#finding-与严重程度)

## 适用范围与证据门槛

- 变更包含 `.py`、`.pyi`、`.c`、`.cc`、`.cpp`、`.cxx`、`.h`、`.hh`、`.hpp`、`.hxx`、`.sh`、`CMakeLists.txt` 或 `.cmake` 时，应用相应语言规则。
- 只有组件的 CMake、脚本、测试配置、README 或支持矩阵声明支持 Windows/MSVC 或 Android/NDK 时，才应用对应平台清单；不要要求未声明的平台支持。
- 检查 changed hunk 所在的完整语法结构，并打开必要的宏、声明、CMake target、包装脚本和直接调用方。不要只看单行 token。
- 明确区分“静态可确定失败”和“可能因编译器/环境差异失败”。前者可以形成 finding；后者没有具体 contract 时不报告。
- 不声称执行了 `py_compile`、`bash -n`、CMake configure、MSVC、Clang、GCC 或 Android NDK 编译。

## 语言级静态语法检查

### Python

检查：

- 括号、方括号、花括号、字符串、三引号和 f-string 表达式是否闭合。
- `if/elif/else`、`for/while`、`def/class`、`try/except/else/finally`、`with`、`match/case` 是否包含必需冒号并保持合法层级。
- 缩进是否使语句落入预期 block，是否出现明显的 unexpected indent/dedent。
- 函数参数顺序、默认参数、`*`/`**`、重复 keyword、decorator 和 async/await 位置是否语法合法。
- import、赋值、comprehension、lambda、yield/return 和类型注解是否存在明显不完整结构。

不要把未定义名称、错误 import path 或错误类型称为“语法错误”；按实际 semantic/contract 问题描述。

### C/C++

检查：

- 圆括号、方括号、花括号、字符串、字符字面量、注释和语句分号是否闭合或完整。
- namespace、class/struct/enum、函数、lambda、initializer list 和 template 结构是否完整。
- 声明与定义的参数、限定符、namespace、返回类型和 template 参数是否明显不匹配。
- `#if/#ifdef/#ifndef/#elif/#else/#endif` 是否配对，条件分支是否导致某平台缺少声明、include 或 closing token。
- 宏定义和续行反斜杠是否会吞掉下一行、截断表达式或产生明显非法 token。
- include、类型和 symbol 的使用是否在当前 translation unit 中有直接可见的声明来源；不能确定 transitive include 时不要武断报告。

复杂 template 实例化、overload resolution、宏展开或标准库实现差异没有确定证据时，不要声称一定无法编译。

### Bash

先根据 shebang 和现有脚本约定确认方言，再检查：

- `if/then/elif/else/fi`、`case/esac`、`for/while/until/do/done`、函数和 subshell 是否配对。
- 单双引号、`${...}`、`$(...)`、算术扩展、数组和 `[[ ... ]]` 是否闭合。
- `case` pattern 是否有正确 terminator，pipeline、重定向和续行反斜杠是否完整。
- 声明为 `/bin/sh` 的脚本是否无意使用 Bash-only 的 `[[ ]]`、数组、`source` 或 `BASH_SOURCE`。
- heredoc delimiter 是否一致，quoted delimiter 与变量展开语义是否符合意图。

Shell quoting、word splitting、glob 和退出码属于语义问题，不要只标成语法问题。

### CMake

检查：

- command 括号、字符串、变量引用和 generator expression 是否闭合。
- `if/elseif/else/endif`、`foreach/endforeach`、`while/endwhile`、`function/endfunction`、`macro/endmacro` 是否配对。
- target 是否在 `target_*` 命令前创建，target 名、source 变量、install/export 名是否一致。
- list/string/path 参数是否因未引用变量、分号展开或空变量产生明显错误调用。
- platform branch 是否完整，是否在一个分支中引用只在另一个分支创建的 target、变量或 imported library。
- CMake minimum version 与使用的 command、policy 或 generator expression 是否存在直接可见冲突。

## MSVC 可移植性

当组件声明支持 Windows/MSVC 时，沿 C/C++、CMake、`run.bat`、测试配置和 README 检查：

- POSIX-only header/API，如 `unistd.h`、`dlfcn.h`、`pthread_*`、`fork`、`readlink`、`strcasecmp`，是否有 `_WIN32`/`MSVC` guard、兼容实现或明确不进入 Windows target。
- GCC/Clang-only extension，如 `__attribute__`、`__builtin_*`、`typeof`、statement expression、variable-length array，以及 `-fPIC`、`-Wl,...`、`-pthread` 等 flag，是否被限制在非 MSVC 分支。
- Windows 所需的类型、API 和宏差异，如 `ssize_t`、socket 类型、`NOMINMAX`、`min/max`、`__declspec(dllexport/dllimport)` 和 DLL symbol visibility，是否由公共兼容层处理。
- C++ standard 与 `/std:c++...` 是否匹配源码特性；不要把 C99/C++ extension 无条件用于 MSVC C/C++ target。
- CMake 是否用 `MSVC`/`WIN32` 正确区分 compile option、link library、runtime DLL、`.lib`、`.dll`、install 和 copy 路径。
- 多配置 generator 是否错误依赖单配置的 `CMAKE_BUILD_TYPE`，Debug/Release 产物名和目录是否与脚本、测试和 README 一致。
- Windows 路径、quoting、驱动器号、反斜杠、空格目录和 `run.bat` 的错误传播是否被正确处理。

若代码无条件包含明确不存在于 MSVC 的 header、传入 MSVC 不接受的 compile flag、链接仅 Unix 存在的 library，或 Windows 分支引用未定义 target/symbol，可形成“Windows/MSVC build 必然失败”的 finding。

## Android NDK 交叉编译可移植性

当组件声明支持 Android/NDK 时，沿 C/C++、CMake、`build_ndk.sh`、测试配置和 README 检查：

- 构建入口是否传入 `${NDK}/build/cmake/android.toolchain.cmake`，并保持 `ANDROID_ABI`、`ANDROID_PLATFORM`、NDK 路径和 install prefix 一致。
- include/library 是否来自目标 Android ABI/SDK，而不是宿主机 `/usr/lib`、x86_64 库或桌面 Linux runtime。
- `ANDROID`、`__ANDROID__`、`CMAKE_SYSTEM_NAME`、architecture guard 是否正确区分 Android、桌面 Linux、Windows 和其他交叉编译目标。
- glibc-only API、host tool、desktop-only library、GNU linker option 或 Linux-only system facility 是否被无条件用于 Android target。
- x86/SSE/AVX intrinsic 是否被无条件编入 `arm64-v8a`，NEON/architecture-specific 代码是否有匹配 guard 和 fallback。
- exception、RTTI、STL/runtime、atomic、thread、`dl` 等 link requirement 是否与项目既有 Android CMake 约定一致；没有仓库 contract 时不要猜测具体 NDK 行为。
- CMake configure/build/install 输出目录是否与 adb/push/run 脚本和 README 一致，不能把 host executable 或 host `.so` 当作 Android 产物。
- ABI、API level、runtime/HAL library、package 名和目标设备目录在脚本、CMake、测试配置和 README 中保持一致。

若脚本缺少 toolchain file、目标 `arm64-v8a` 却链接明确的 x86_64/host library、Android branch 使用只在 desktop Linux 分支定义的 target，或 install/push 路径确定性指向错误产物，可形成“Android NDK 交叉编译必然失败”的 finding。

## First-party 源文件头

对新增、复制到 first-party 路径或本次变更删除/破坏文件头的 Python 与 C/C++ 源文件检查文件头。适用扩展名：`.py`、`.pyi`、`.c`、`.cc`、`.cpp`、`.cxx`、`.h`、`.hh`、`.hpp`、`.hxx`。

新增 first-party 文件必须在首个代码/import/include/header guard 之前包含：

- `Copyright (c) <创建年份> HOUMO AI`；新文件使用创建年份，不因普通修改重写历史年份。
- `File: <实际 basename>`，必须与文件名一致。
- 非空且准确的 `Description:`，说明文件职责，不能保留复制来源的模型或组件名称。
- Apache License 2.0 notice。
- `SPDX-License-Identifier: Apache-2.0`。

Python 使用 `#` 注释；允许 shebang 和 encoding declaration 位于版权头之前。C/C++ 使用仓库既有 `/* ... */` 风格，并放在 include、pragma once 或 header guard 之前。

不要把 HOUMO AI 模板覆盖到 third-party/vendored code；third-party 文件保留原作者、原许可和 SPDX。生成文件、构建输出和全局 exclusions 仍按排除规则处理。

对既有文件中早于本次变更的缺失文件头，不单独制造 finding；只有本次新增/复制文件、删除或损坏已有文件头、修改后 `File:` 不再匹配、或 `Description:` 因复制残留而错误时报告。

## Finding 与严重程度

- 语法或平台问题使标准 `test.sh`、主 `run.sh`、README Quick Start 或其必经入口确定性失败时，按通用规则定为 P0。
- 语法或平台问题使声明支持的非主源码/平台构建确定性失败，但不满足标准入口 P0 条件时，通常为 P1。
- 新增 first-party Python/C/C++ 文件缺少规定文件头、文件名或描述不匹配，通常为 P2；若涉及删除或替换 third-party license、错误版权声明或更广泛合规风险，按具体影响提高严重程度。
- finding 必须定位到引入错误的最窄 changed line，并说明语言结构、目标平台或文件头字段为何与直接上下文冲突。不要仅写“可能编译失败”或“缺少注释”。
