"""
Calculator Tools - 计算器工具集合
包含基本的数学计算功能
版本: 1.2.0
"""

def add(a: float, b: float) -> float:
    """
    加法运算
    
    Args:
        a: 第一个数字
        b: 第二个数字
    
    Returns:
        两数之和
    """
    return a + b

def subtract(a: float, b: float) -> float:
    """
    减法运算
    
    Args:
        a: 被减数
        b: 减数
    
    Returns:
        两数之差
    """
    return a - b

def multiply(a: float, b: float) -> float:
    """
    乘法运算
    
    Args:
        a: 第一个数字
        b: 第二个数字
    
    Returns:
        两数之积
    """
    return a * b

def divide(a: float, b: float) -> float:
    """
    除法运算
    
    Args:
        a: 被除数
        b: 除数
    
    Returns:
        两数之商
    
    Raises:
        ZeroDivisionError: 当除数为0时
    """
    if b == 0:
        raise ZeroDivisionError("除数不能为0")
    return a / b

def power(base: float, exponent: float) -> float:
    """
    幂运算
    
    Args:
        base: 底数
        exponent: 指数
    
    Returns:
        base的exponent次幂
    """
    return base ** exponent

def calculate_area_circle(radius: float) -> float:
    """
    计算圆的面积
    
    Args:
        radius: 圆的半径
    
    Returns:
        圆的面积
    """
    import math
    return math.pi * radius ** 2