import os

import gradio as gr
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def create_model() -> str:
    iris_data = load_iris()
    X, Y = iris_data.data, iris_data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=0, shuffle=True)
    model = MLPClassifier(solver="adam", random_state=0, max_iter=5000)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    joblib.dump(model, os.path.join(MODEL_DIR, "nn.pkl"))
    out.visible = True
    return f"Score: {model.score(X_test, Y_test):.2f}\n{classification_report(Y_test, Y_pred)}"


def predict(x1: float, x2: float, x3: float, x4: float) -> str:
    x = np.array([x1, x2, x3, x4]).reshape(1, -1)
    model = joblib.load(os.path.join(MODEL_DIR, "nn.pkl"))
    pred = int(model.predict(x)[0])
    labels = ["Setosa", "Versicolor", "Virginica"]
    return labels[pred]


# 表示部分
with gr.Blocks() as demo:
    # 訓練部分
    with gr.Tab("Train Model"):
        gr.Markdown("### Create Model")
        inp = gr.Button("Create")
        out = gr.Textbox()
        inp.click(fn=create_model, outputs=out, show_progress=True)
    # 予測部分
    with gr.Tab("Predict"):
        gr.Markdown("### Predict iris species")
        visible_flg = os.path.exists(os.path.join(MODEL_DIR, "nn.pkl"))
        with gr.Row():
            with gr.Column():
                inp1 = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                    label="Sepal Length",
                    interactive=True,
                    visible=visible_flg,
                )
                inp2 = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                    label="Sepal Width",
                    interactive=True,
                    visible=visible_flg,
                )
                inp3 = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                    label="Petal Length",
                    interactive=True,
                    visible=visible_flg,
                )
                inp4 = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                    label="Petal Width",
                    interactive=True,
                    visible=visible_flg,
                )
            with gr.Column():
                out = gr.Textbox(label="Predict Iris Species", visible=visible_flg)
        inp1.release(fn=predict, inputs=[inp1, inp2, inp3, inp4], outputs=out, queue=True)
        inp2.release(fn=predict, inputs=[inp1, inp2, inp3, inp4], outputs=out, queue=True)
        inp3.release(fn=predict, inputs=[inp1, inp2, inp3, inp4], outputs=out, queue=True)
        inp4.release(fn=predict, inputs=[inp1, inp2, inp3, inp4], outputs=out, queue=True)

demo.queue()
demo.launch()
