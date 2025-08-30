import os
import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # 改善版のSplitterを使用
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 初期設定（アプリの初回起動時に一度だけ実行される） ---
@st.cache_resource
def setup_rag_chain():
    # 1. APIキーの設定 (Streamlitのシークレット機能から読み込む)
    try:
        os.environ["OPENAI_API_VERSION"] = st.secrets["AZURE_OPENAI"]["API_VERSION"]
        os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI"]["ENDPOINT"]
        os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI"]["API_KEY"]
        os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = st.secrets["AZURE_OPENAI"]["CHAT_DEPLOYMENT_NAME"]
        os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] = st.secrets["AZURE_OPENAI"]["EMBEDDING_DEPLOYMENT_NAME"]
    except KeyError:
        st.error("StreamlitのシークレットにAzure OpenAIの認証情報を設定してください。")
        st.stop()

    # 2. ドキュメントの読み込みと分割
    loader = TextLoader("./0830LLM.txt", encoding='utf8')
    documents = loader.load()
    
    # Colabの警告を解消するため、より高性能な分割方法に変更
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(documents)

    # 3. ベクトル化とVector Storeの準備（永続化対応）
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    
    # 永続化ディレクトリ
    persist_directory = "./chroma_db"
    
    # 既にDBが存在すれば読み込み、なければ作成して保存
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        
    retriever = vectorstore.as_retriever()
    llm = AzureChatOpenAI(azure_deployment="gpt-4.1")

    # 4. RAGチェーンの構築
    template = """
    あなたは親切な片付けアドバイザーです。
    以下の「部屋の状態」を参考にして、質問に答えてください。

    # 部屋の状態
    {context}

    # 質問
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever.invoke, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Streamlit UI部分 ---

# RAGチェーンを準備
chain = setup_rag_chain()

st.title("お片付け相談AIアシスタント 🤖")
st.caption("AIがあなたのお家の情報に基づいて、片付けのアドバイスをします。")

# 会話履歴をセッション状態で管理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け取るチャット入力欄
if user_input := st.chat_input("今日はどこを片付けたいですか？"):
    # ユーザーの入力を履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AIの応答を生成
    with st.chat_message("assistant"):
        with st.spinner("AIが考えています..."):
            response = chain.invoke(user_input)
            st.markdown(response)
    
    # AIの応答を履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response})