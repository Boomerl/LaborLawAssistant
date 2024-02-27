import json
import os
import re

import pandas as pd
from langchain.document_loaders import (UnstructuredFileLoader,
                                        UnstructuredMarkdownLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


def get_files(dir_path, need_type="md"):
    file_list = []
    # os.walk函数，递归遍历dir_path
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(need_type):
                file_list.append(os.path.join(filepath, filename))

    return file_list


def get_text(dir_path, need_type="md"):
    file_lst = get_files(dir_path=dir_path, need_type=need_type)

    docs = []
    for one_file in tqdm(file_lst):
        file_type = one_file.split(".")[-1]
        if file_type == "md":
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == "txt":
            loader = UnstructuredFileLoader(one_file)
        docs.extend(loader.load())

    return docs


def clean_documents(documents):
    # 设置匹配模式，将匹配结果进行分块
    pattern1 = r"(案例\d.|案例['一'-'十']{1,3})|0\d\n"
    replacement1 = r"<chunk1>"
    pattern2 = r"([\u4e00-\u9fa5]{2}结果|专家点评)"
    replacement2 = r"<chunk2>处理结果"

    for doc in documents:
        doc.page_content = (
            re.sub(r"\n+", "\n", doc.page_content).strip().replace("\u3000", "")
        )
        doc.page_content = re.sub(pattern1, replacement1, doc.page_content)
        doc.page_content = re.sub(pattern2, replacement2, doc.page_content)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separator="<chunk>\1"
    )
    split_docs = splitter.split_documents(documents)

    return split_docs


def generate_qa(documents):
    qa_lst = []
    for doc in documents:
        chunk1s = doc.page_content.split("<chunk1>")
        for chunk in chunk1s[1:]:
            q_a = chunk.split("<chunk2>")
            qa_lst.append({"question": q_a[0], "answer": q_a[1]})

    return qa_lst


if __name__ == "__main__":
    data_path = "/root/data/law"
    docs = get_text(dir_path=data_path, need_type="txt")
    # clean documents
    docs = clean_documents(docs)
    # generate question & answer pair
    qa_list = generate_qa(docs)
    # save in xlsx
    df = pd.DataFrame(qa_list)
    df.to_excel("/root/data/law/ft_data.xlsx", index=False)
