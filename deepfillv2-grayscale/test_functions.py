#!/usr/bin/env python3
"""
测试trainer.py中定义的函数是否存在
"""

import ast
import sys

def check_functions_exist(filename):
    """检查文件中是否定义了所需的函数"""
    with open(filename, 'r') as f:
        tree = ast.parse(f.read())
    
    defined_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined_functions.add(node.name)
    
    required_functions = {
        '_compute_batch_psnr',
        '_compute_batch_mse', 
        '_compute_batch_ssim',
        '_compute_batch_psnr_gpu',
        '_compute_batch_mse_gpu',
        '_compute_batch_ssim_gpu'
    }
    
    missing_functions = required_functions - defined_functions
    if missing_functions:
        print(f"❌ 缺失函数: {missing_functions}")
        return False
    else:
        print("✅ 所有必需的函数都已定义")
        print(f"✅ 定义的函数: {sorted(defined_functions.intersection(required_functions))}")
        return True

if __name__ == "__main__":
    if check_functions_exist('trainer.py'):
        print("函数检查通过！")
        sys.exit(0)
    else:
        print("函数检查失败！")
        sys.exit(1)
