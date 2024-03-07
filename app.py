from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import InternLM2_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import gradio as gr

from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download("zxdlalala/Internlm2_LaborLaw", cache_dir="./models")


def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    # embeddings = HuggingFaceEmbeddings(model_name="/root/models/sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = "./data_base/vector_db/law"

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # 加载自定义 LLM
    llm = InternLM2_LLM(model_path="./models/zxdlalala/Internlm2_LaborLaw")

    # 定义一个 Prompt Template
    template = """回答规范：
    使用为你提供的相关法律条文来回答最后的问题。
    如果你不知道答案，就说你不知道，不要试图编造答案。
    尽量分条列点回答，并使答案简明扼要。
    总是在回答的最后说“谢谢你的提问！”。
    你的回答应该包含具体法律条文。
    如果用户的提问并非劳动法相关问题，你应该拒绝作答，并感谢用户的提问

相关法律条文：
{context}

问题: {question}

有用的回答：
"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    return qa_chain


class Model_center:
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """调用问答链进行回答"""
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append((question, self.chain({"query": question})["result"]))
            return "", chat_history
        except Exception as e:
            return e, chat_history


# 实例化核心功能对象
model_center = Model_center()
# 创建界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown(
                """<h1><center>InternLM</center></h1>
                <center>劳动法知识库检索小助手</center>"""
            )

    with gr.Row():
        with gr.Column(scale=4):
            # 创建聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建文本框组件，用于输入prompt
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                clear = gr.ClearButton(components=[chatbot], value="Clear console")

        db_wo_his_btn.click(
            model_center.qa_chain_self_answer,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

    gr.Markdown(
        """提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """
    )

gr.close_all()
# 直接启动
demo.launch()
