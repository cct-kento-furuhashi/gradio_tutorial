import gradio as gr
import numpy as np
import numpy.typing as npt

"""
画像を表示するプログラム
"""


def sepia(input_img: npt.NDArray[np.uint8]) -> npt.ArrayLike:
    sepia_filter = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])

    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img  # type: ignore


demo = gr.Interface(fn=sepia, inputs=gr.Image(shape=(200, 200)), outputs="image")
demo.launch()
