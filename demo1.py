from typing import Tuple

import gradio as gr

"""
入力と出力を表示するプログラム
"""


def greet(name: str, is_morning: bool, temperature: int) -> Tuple[str, float]:
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)


demo = gr.Interface(fn=greet, inputs=["text", "checkbox", gr.Slider(0, 100)], outputs=["text", "number"])
demo.launch()
